import json
import logging
import os
import sys

import pandas as pd
import streamlit as st

# Add project root to path to allow importing constants
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants.shipping_zones import load_shipping_zones
from constants.schemas import SchemaManager

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data loading, processing, and transformation for the fulfillment application.
    """

    def __init__(self, use_airtable: bool = True):
        """
        Initialize the DataProcessor with required tracking variables.
        
        Args:
            use_airtable (bool): Whether to use Airtable for schema data (default: True)
        """
        # Set to track which inventory issues have been logged to prevent duplicate log messages
        self._logged_inventory_issues = set()
        # Track recalculation operations for logging
        self._recalculation_log = []
        # Track which SKUs were affected by recalculation
        self._recalculated_skus = set()
        # Track warehouse transformations for debugging
        self._warehouse_transformations = []
        # Schema manager for Airtable integration
        self.use_airtable = use_airtable
        self.schema_manager = SchemaManager() if use_airtable else None

    def normalize_warehouse_name(self, warehouse_name, log_transformations=True):
        """
        Centralized warehouse name normalization function that handles all variations
        and provides backward compatibility with existing data.
        
        This function:
        - Accepts any warehouse name variation (moorpark, Moorpark, CA-Moorpark-93021, oxnard, Oxnard, etc.)
        - Returns standardized names ("Oxnard" or "Wheeling")
        - Handles ZIP code mapping (93021 -> Oxnard operations)
        - Logs transformations for debugging
        - Is backward compatible with existing data
        
        Args:
            warehouse_name (str): The original warehouse name to normalize
            log_transformations (bool): Whether to log transformations for debugging
            
        Returns:
            str: Standardized warehouse name ("Oxnard" or "Wheeling")
        """
        if not warehouse_name or pd.isna(warehouse_name):
            standardized_name = "Oxnard"  # Default to Oxnard for missing values
            if log_transformations:
                transformation = {
                    "original": warehouse_name,
                    "standardized": standardized_name,
                    "reason": "default_for_missing_value"
                }
                self._warehouse_transformations.append(transformation)
                logger.debug(f"Warehouse normalization: {warehouse_name} -> {standardized_name} (default for missing value)")
            return standardized_name
        
        # Convert to string and normalize case for comparison
        original_name = str(warehouse_name)
        normalized_input = original_name.lower().strip()
        
        # Comprehensive mapping for Oxnard (includes legacy Moorpark references)
        oxnard_variations = [
            # Current Oxnard variations
            "oxnard", "ca-oxnard", "oxnard-ca", "california-oxnard",
            # Legacy Moorpark variations (93021 ZIP code area now operates as Oxnard)
            "moorpark", "ca-moorpark", "moorpark-ca", "california-moorpark",
            # ZIP code based variations
            "93021", "93030", "ca-93021", "ca-93030",
            # Full format variations from CSV files
            "ca-moorpark-93021", "ca-oxnard-93030", "ca-oxnard-93021",
            # State-based variations
            "california", "ca", "west", "west-coast"
        ]
        
        # Comprehensive mapping for Wheeling
        wheeling_variations = [
            # Current Wheeling variations
            "wheeling", "il-wheeling", "wheeling-il", "illinois-wheeling",
            # ZIP code based variations
            "60090", "il-60090",
            # Full format variations from CSV files
            "il-wheeling-60090",
            # State-based variations
            "illinois", "il", "midwest", "central"
        ]
        
        # Check for partial matches in the normalized input
        # This handles cases like "CA-Moorpark-93021" where we need to find substrings
        def contains_variation(input_str, variations):
            return any(variation in input_str for variation in variations)
        
        # Determine standardized warehouse name
        if contains_variation(normalized_input, oxnard_variations):
            standardized_name = "Oxnard"
            if "moorpark" in normalized_input or "93021" in normalized_input:
                reason = "legacy_moorpark_to_oxnard"
            elif "93030" in normalized_input:
                reason = "zip_code_93030_to_oxnard"
            else:
                reason = "oxnard_variation"
        elif contains_variation(normalized_input, wheeling_variations):
            standardized_name = "Wheeling"
            if "60090" in normalized_input:
                reason = "zip_code_60090_to_wheeling"
            else:
                reason = "wheeling_variation"
        else:
            # If no match found, try to infer from first character or default to Oxnard
            if normalized_input.startswith(('ca', 'c')) or any(char in normalized_input for char in ['9']):
                standardized_name = "Oxnard"
                reason = "inferred_california_to_oxnard"
            elif normalized_input.startswith(('il', 'i')) or any(char in normalized_input for char in ['6']):
                standardized_name = "Wheeling"
                reason = "inferred_illinois_to_wheeling"
            else:
                standardized_name = "Oxnard"  # Default fallback
                reason = "default_fallback_to_oxnard"
        
        # Log transformation if requested and there was a change
        if log_transformations and original_name != standardized_name:
            transformation = {
                "original": original_name,
                "standardized": standardized_name,
                "reason": reason
            }
            self._warehouse_transformations.append(transformation)
            logger.debug(f"Warehouse normalization: {original_name} -> {standardized_name} ({reason})")
        
        return standardized_name
    
    def get_warehouse_transformations_summary(self):
        """
        Get a summary of all warehouse transformations performed during processing.
        
        Returns:
            pandas.DataFrame: DataFrame containing transformation details
        """
        if not self._warehouse_transformations:
            return pd.DataFrame(columns=["original", "standardized", "reason", "count"])
        
        # Convert to DataFrame and group by transformation
        df = pd.DataFrame(self._warehouse_transformations)
        summary = df.groupby(["original", "standardized", "reason"]).size().reset_index(name="count")
        summary = summary.sort_values(["count", "original"], ascending=[False, True])
        
        return summary

    def _normalize_warehouse_key(self, sku, warehouse):
        """
        Helper function to normalize warehouse keys for inventory comparison.
        
        Args:
            sku (str): The SKU part of the key
            warehouse (str): The warehouse part of the key
            
        Returns:
            str: Normalized key in format "SKU|NormalizedWarehouse"
        """
        normalized_warehouse = self.normalize_warehouse_name(warehouse, log_transformations=False)
        return f"{sku}|{normalized_warehouse}"

    def load_orders(self, file):
        """
        Load and preprocess orders CSV file

        Args:
            file: Uploaded CSV file object

        Returns:
            pandas.DataFrame: Processed orders dataframe
        """
        try:
            # Read CSV file
            df = pd.read_csv(file)

            # Basic preprocessing
            # Ensure required columns exist
            required_columns = [
                "Date",
                "Name",
                "order id",
                "Customer: First Name",
                "Customer: Last Name",
                "Email",
                "Shipping: Name",
                "Shipping: Address 1",
                "Shipping: Address 2",
                "Shipping: City",
                "Shipping: Province Code",
                "Shipping: Zip",
                "Note",
                "SKU Helper",
                "Line: Fulfillable Quantity",
                "Line: ID",
                "NEW Tags",
                "MAX PKG NUM",
                "Fulfillment Center",
                "Saturday Shipping",
            ]

            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]

            # If columns are missing, try to find them with case-insensitive matching
            if missing_columns:
                # Create a mapping of lowercase column names to actual column names
                col_mapping = {col.lower(): col for col in df.columns}

                # Try to map missing columns
                for missing_col in missing_columns.copy():
                    if missing_col.lower() in col_mapping:
                        # Rename column to expected name
                        df.rename(
                            columns={col_mapping[missing_col.lower()]: missing_col}, inplace=True
                        )
                        missing_columns.remove(missing_col)

            # If columns are still missing, check if we have sample data in docs folder
            if missing_columns:
                st.warning(
                    f"Some columns are missing in the uploaded file. Using sample data from docs folder."
                )
                try:
                    # Try to use the sample file from docs folder
                    sample_path = "docs/orders_placed.csv"
                    if os.path.exists(sample_path):
                        df = pd.read_csv(sample_path)
                        st.success("Successfully loaded sample orders data.")
                    else:
                        raise ValueError(
                            f"Missing required columns and sample file not found: {', '.join(missing_columns)}"
                        )
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")
                    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

            # Convert date format with error handling
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            else:
                df["Date"] = pd.to_datetime("today")

            # Fill NA values for required fields
            for col in ["Shipping: Address 2", "Note", "NEW Tags"]:
                if col in df.columns:
                    df[col] = df[col].fillna("")
                else:
                    df[col] = ""

            if "MAX PKG NUM" in df.columns:
                df["MAX PKG NUM"] = pd.to_numeric(df["MAX PKG NUM"], errors="coerce").fillna(1)
            else:
                df["MAX PKG NUM"] = 1

            if "Fulfillment Center" in df.columns:
                df["Fulfillment Center"] = df["Fulfillment Center"].fillna("")
            else:
                df["Fulfillment Center"] = ""

            if "Saturday Shipping" in df.columns:
                df["Saturday Shipping"] = df["Saturday Shipping"].fillna(False)
                # Convert Saturday Shipping to boolean
                df["Saturday Shipping"] = df["Saturday Shipping"].map(
                    lambda x: str(x).upper() == "TRUE" if pd.notnull(x) else False
                )
            else:
                df["Saturday Shipping"] = False

            # Convert Line: Fulfillable Quantity to numeric if it exists
            if "Line: Fulfillable Quantity" in df.columns:
                df["Line: Fulfillable Quantity"] = pd.to_numeric(
                    df["Line: Fulfillable Quantity"], errors="coerce"
                ).fillna(1)
            else:
                df["Line: Fulfillable Quantity"] = 1

            # Clean up and standardize SKU format
            if "SKU Helper" in df.columns:
                df["SKU"] = df["SKU Helper"].str.replace("f.", "", regex=False)
            elif "SKU" not in df.columns:
                df["SKU"] = "unknown"

            # Extract priority tags from NEW Tags if it exists
            if "NEW Tags" in df.columns:
                df["Priority"] = df["NEW Tags"].str.extract(r"(P\d+|P\*)", expand=False).fillna("")
            else:
                df["Priority"] = ""

            # Parse zip codes to ensure consistent format if it exists
            if "Shipping: Zip" in df.columns:
                df["Shipping: Zip"] = (
                    df["Shipping: Zip"].astype(str).str.strip().str.split("-").str[0]
                )
            else:
                df["Shipping: Zip"] = ""

            return df

        except Exception as e:
            st.error(f"Error processing orders file: {str(e)}")
            # Create a minimal dataframe with required columns
            return pd.DataFrame(columns=required_columns)

    def load_sku_mappings(self):
        """
        Load SKU mappings from Airtable or JSON file for Oxnard, Moorpark, and Wheeling fulfillment centers
        with fallback to CSV files if neither is available.

        Handles both individual SKUs (using picklist_sku) and bundles (using component_sku).
        For bundles, all components and their quantities are stored in a structured format.

        Returns:
            dict: Dictionary containing:
                - SKU mappings by fulfillment center
                - Bundle information by fulfillment center
        """
        try:
            # Initialize mappings and bundle information dictionaries
            mappings = {"Oxnard": {}, "Wheeling": {}}
            bundle_info = {"Oxnard": {}, "Wheeling": {}}
            
            # Try Airtable first if enabled
            if self.use_airtable and self.schema_manager:
                try:
                    airtable_result = self._load_sku_mappings_from_airtable()
                    if airtable_result and any(len(airtable_result["mappings"][center]) > 0 for center in airtable_result["mappings"]):
                        logger.info("Successfully loaded SKU mappings from Airtable")
                        return airtable_result
                except Exception as e:
                    logger.warning(f"Failed to load SKU mappings from Airtable: {e}")
                    logger.info("Falling back to JSON/CSV files...")

            # First try to load from JSON file
            json_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "constants", "data", "sku_mappings.json"
            )

            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        sku_json = json.load(f)

                    # Process each fulfillment center
                    for center in ["Wheeling", "Oxnard"]:
                        if center not in sku_json:
                            logger.warning(f"{center} not found in SKU mappings JSON")
                            continue

                        center_map = {}

                        # Process individual SKUs
                        if "all_skus" in sku_json[center]:
                            for sku, data in sku_json[center]["all_skus"].items():
                                if "picklist_sku" in data:
                                    # Map the Shopify SKU to the picklist SKU
                                    center_map[sku] = data["picklist_sku"]

                        # Process bundles
                        if "bundles" in sku_json[center]:
                            for bundle_sku, components in sku_json[center]["bundles"].items():
                                # Store all component information for the bundle
                                bundle_components = []
                                for component in components:
                                    if "component_sku" in component:
                                        component_info = {
                                            "sku": component["component_sku"],
                                            "qty": float(component.get("actualqty", 1.0)),
                                            "weight": float(component.get("weight", 0.0)),
                                            "type": component.get("type", ""),
                                        }
                                        bundle_components.append(component_info)

                                # Store the bundle components information
                                if bundle_components:
                                    bundle_info[center][bundle_sku] = bundle_components

                                    # Also map the bundle to the first component for backward compatibility
                                    # This allows the bundle to be mapped to a warehouse SKU for picking
                                    center_map[bundle_sku] = bundle_components[0]["sku"]
                                    logger.debug(
                                        f"Mapped bundle {bundle_sku} to component {bundle_components[0]['sku']}"
                                    )

                            logger.info(
                                f"Processed {len(sku_json[center]['bundles'])} bundles for {center}"
                            )

                        mappings[center] = center_map
                        logger.info(f"Loaded {len(center_map)} SKU mappings for {center} from JSON")

                    # If we successfully loaded mappings from JSON, return them along with bundle info
                    if any(len(mappings[center]) > 0 for center in mappings):
                        return {"mappings": mappings, "bundle_info": bundle_info}

                except Exception as e:
                    logger.error(f"Error loading SKU mappings from JSON: {e}")
                    logger.info("Falling back to CSV files for SKU mappings")

            # Fallback to CSV files if JSON loading failed or no mappings were found
            # Define paths to mapping files
            oxnard_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "docs", "sku_shopify_to_oxnard.csv"
            )
            wheeling_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "docs", "sku_shopify_to_wheeling.csv"
            )

            # Load Oxnard mappings from CSV
            if os.path.exists(oxnard_path):
                oxnard_df = pd.read_csv(oxnard_path, skiprows=1)  # Skip the duplicate header row
                # Clean up columns by stripping whitespace
                oxnard_df["shopifysku2"] = oxnard_df["shopifysku2"].astype(str).str.strip()
                oxnard_df["picklist sku"] = oxnard_df["picklist sku"].astype(str).str.strip()
                # Filter out empty or NaN picklist SKUs
                oxnard_df = oxnard_df[
                    oxnard_df["picklist sku"].notna() & (oxnard_df["picklist sku"] != "nan")
                ]
                # Create mapping from shopify SKU to picklist SKU
                oxnard_map = dict(zip(oxnard_df["shopifysku2"], oxnard_df["picklist sku"]))
                mappings["Oxnard"] = oxnard_map
                logger.info(f"Loaded {len(oxnard_map)} SKU mappings for Oxnard from CSV")
            else:
                logger.warning(f"Warning: Oxnard SKU mapping file not found at {oxnard_path}")

            # Load Wheeling mappings from CSV
            if os.path.exists(wheeling_path):
                try:
                    # First attempt: skip first row which might be a duplicate header
                    wheeling_df = pd.read_csv(wheeling_path, skiprows=1)
                    logger.info(
                        f"Successfully loaded Wheeling SKU file with skiprows=1, found {len(wheeling_df)} rows"
                    )
                except Exception as e1:
                    logger.error(f"Error loading Wheeling SKU file with skiprows=1: {e1}")
                    try:
                        # Second attempt: load without skipping rows
                        wheeling_df = pd.read_csv(wheeling_path)
                        logger.info(
                            f"Successfully loaded Wheeling SKU file without skiprows, found {len(wheeling_df)} rows"
                        )
                    except Exception as e2:
                        logger.error(f"Error loading Wheeling SKU file without skiprows: {e2}")
                        # If both attempts fail, create an empty DataFrame
                        wheeling_df = pd.DataFrame(columns=["shopifysku2", "picklist sku"])
                        logger.info(
                            "Created empty DataFrame for Wheeling SKUs due to loading errors"
                        )

                # Show sample of the data to verify structure
                if not wheeling_df.empty:
                    logger.info(f"Wheeling SKU file columns: {wheeling_df.columns.tolist()}")
                    logger.info("Sample of first 3 Wheeling SKU mappings:")
                    for i, row in wheeling_df.head(3).iterrows():
                        logger.info(f"  {row['shopifysku2']} -> {row['picklist sku']}")

                # Clean up columns by stripping whitespace
                wheeling_df["shopifysku2"] = wheeling_df["shopifysku2"].astype(str).str.strip()
                wheeling_df["picklist sku"] = wheeling_df["picklist sku"].astype(str).str.strip()

                # Filter out empty or NaN picklist SKUs
                wheeling_df_filtered = wheeling_df[
                    wheeling_df["picklist sku"].notna()
                    & (wheeling_df["picklist sku"] != "nan")
                    & (wheeling_df["picklist sku"] != "")
                ]

                logger.info(
                    f"After filtering, Wheeling SKU mappings reduced from {len(wheeling_df)} to {len(wheeling_df_filtered)} rows"
                )

                # Create mapping from shopify SKU to picklist SKU
                wheeling_map = dict(
                    zip(wheeling_df_filtered["shopifysku2"], wheeling_df_filtered["picklist sku"])
                )
                mappings["Wheeling"] = wheeling_map

                logger.info(f"Loaded {len(wheeling_map)} SKU mappings for Wheeling from CSV")
                logger.info("Sample of Wheeling mappings:")
                sample_keys = list(wheeling_map.keys())[:5]
                for key in sample_keys:
                    logger.info(f"  {key} -> {wheeling_map[key]}")

                # Check for specific SKUs that should be mapped
                test_skus = ["f.loquat-5lb", "f.loquat-2lb", "f.avocado_hass-2lb"]
                for test_sku in test_skus:
                    if test_sku in wheeling_map:
                        logger.info(
                            f"Test SKU '{test_sku}' is mapped to '{wheeling_map[test_sku]}'"
                        )
                    else:
                        logger.info(f"Test SKU '{test_sku}' is NOT found in Wheeling mappings")
            else:
                logger.warning(f"Warning: Wheeling SKU mapping file not found at {wheeling_path}")

            return mappings

        except Exception as e:
            logger.error(f"Error loading SKU mappings: {str(e)}")
            return {"Oxnard": {}, "Wheeling": {}}
    
    def _load_sku_mappings_from_airtable(self):
        """
        Load SKU mappings from Airtable.
        
        Returns:
            dict: Dictionary containing mappings and bundle_info
        """
        logger.info("Loading SKU mappings from Airtable...")
        
        mappings = {"Oxnard": {}, "Wheeling": {}}
        bundle_info = {"Oxnard": {}, "Wheeling": {}}
        
        for center in ["Oxnard", "Wheeling"]:
            try:
                # Get SKU mappings from Airtable for this warehouse
                sku_mappings = self.schema_manager.get_sku_mappings(center)
                
                center_map = {}
                
                for mapping in sku_mappings:
                    # Handle individual SKUs
                    if mapping.picklist_sku:
                        center_map[mapping.order_sku] = mapping.picklist_sku
                    
                    # Handle bundles if they have components
                    bundle_components = mapping.get_bundle_components()
                    if bundle_components:
                        # Store bundle components
                        bundle_info[center][mapping.order_sku] = [
                            {
                                "sku": comp.sku,
                                "qty": comp.qty,
                                "weight": 0.0,  # Weight not stored in current schema
                                "type": "",     # Type not stored in current schema
                            }
                            for comp in bundle_components
                        ]
                        
                        # Map bundle to first component for backward compatibility
                        if bundle_components:
                            center_map[mapping.order_sku] = bundle_components[0].sku
                            logger.debug(f"Mapped bundle {mapping.order_sku} to component {bundle_components[0].sku}")
                
                mappings[center] = center_map
                logger.info(f"Loaded {len(center_map)} SKU mappings for {center} from Airtable")
                
            except Exception as e:
                logger.warning(f"Error loading SKU mappings for {center} from Airtable: {e}")
        
        return {"mappings": mappings, "bundle_info": bundle_info}

    def load_inventory(self, file):
        """
        Load and preprocess inventory CSV file

        Args:
            file: Uploaded CSV file object or file path

        Returns:
            pandas.DataFrame: Processed inventory dataframe with normalized columns
        """
        try:
            # Handle file object or path string
            file_path = None
            if hasattr(file, "read"):
                # It's a file object (like from Streamlit uploader)
                # Save to a temporary file to handle complex CSV structure
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                    file_path = temp_file.name
                    file.seek(0)
                    temp_file.write(file.read())
            else:
                # It's a file path string
                file_path = file
                if not os.path.exists(file_path):
                    logger.error(
                        f"Error processing inventory file: [Errno 2] No such file or directory: '{file_path}'"
                    )
                    return pd.DataFrame()

            # Read the first few lines to understand the structure
            with open(file_path, "r") as f:
                first_lines = [next(f) for _ in range(3) if f]

            # Check if the first line indicates a complex header structure
            if ",,,,,,,,,,,,,,,,Oxnard,,,Wheeling," in first_lines[0]:
                # This is the specific format we observed in the inventory file
                # Read the main inventory data (skipping the complex header)
                df = pd.read_csv(file_path, skiprows=1)
                logger.info("Detected complex inventory file format with multiple headers")

                # For this specific format, we know the first column is WarehouseName
                # and contains values like CA-Moorpark-93021 and IL-Wheeling-60090
                if df.columns[0] == "WarehouseName":
                    logger.info("Found WarehouseName column in first position")

                    # Create a FulfillmentCenter column based on the warehouse name using centralized normalization
                    df["FulfillmentCenter"] = df["WarehouseName"].apply(
                        lambda x: self.normalize_warehouse_name(x, log_transformations=True)
                    )
                    logger.info("Created FulfillmentCenter column from WarehouseName")
            else:
                # Try standard CSV reading
                df = pd.read_csv(file_path)
                logger.info("Detected standard inventory file format")

            # Basic preprocessing
            # Check if required columns exist
            required_columns = ["WarehouseName", "Sku", "AvailableQty"]

            # Check for required columns with case-insensitive matching
            for req_col in required_columns:
                found = False
                for col in df.columns:
                    if col.lower() == req_col.lower():
                        # Rename to standard case if needed
                        if col != req_col:
                            df = df.rename(columns={col: req_col})
                        found = True
                        break

                if not found:
                    logger.warning(
                        f"Warning: Required column '{req_col}' not found in inventory file"
                    )
                    # Create empty column if missing
                    df[req_col] = None

            # Ensure SKU column is properly formatted
            # The inventory file uses 'Sku' but our code often looks for 'SKU'
            if "Sku" in df.columns:
                # Convert SKU to string type to ensure consistent matching
                df["Sku"] = df["Sku"].astype(str).str.strip()
                # Create a standardized SKU column if it doesn't exist
                if "SKU" not in df.columns:
                    df["SKU"] = df["Sku"]
                    logger.info("Created standardized SKU column from Sku column")

            # Extract warehouse name from the first column if it exists
            # This handles the CA-Moorpark-93021 and CA-Oxnard-93030 format in the inventory file
            if df.columns[0].startswith("CA-") or df.columns[0].startswith("IL-"):
                # Create WarehouseName column from the first column
                df["WarehouseName"] = df.iloc[:, 0]
                logger.info(f"Created WarehouseName column from first column: {df.columns[0]}")

            # Ensure WarehouseName is properly formatted
            if "WarehouseName" in df.columns:
                # Fill any missing warehouse names with 'Oxnard' as default
                df["WarehouseName"] = df["WarehouseName"].fillna("Oxnard")

                # Create a FulfillmentCenter column for standardized names using centralized normalization
                df["FulfillmentCenter"] = df["WarehouseName"].apply(
                    lambda x: self.normalize_warehouse_name(x, log_transformations=True)
                )

                # Keep the original WarehouseName for reference
                # This preserves CA-Moorpark-93021 and CA-Oxnard-93030 distinctions
                # Log unique warehouse names after standardization for debugging
                unique_warehouses = df["WarehouseName"].unique()
                logger.info(f"Unique warehouse names after standardization: {unique_warehouses}")

            # Convert numeric columns with error handling
            if "AvailableQty" in df.columns:
                # Handle commas in numeric values
                if df["AvailableQty"].dtype == object:
                    df["AvailableQty"] = df["AvailableQty"].astype(str).str.replace(",", "")
                df["AvailableQty"] = pd.to_numeric(df["AvailableQty"], errors="coerce").fillna(0)
                # Log sample of AvailableQty values for debugging
                logger.info(
                    f"Sample AvailableQty values after conversion: {df['AvailableQty'].head(5).tolist()}"
                )

            # Handle Balance column which might have commas in numbers
            if "Balance" in df.columns:
                # Make sure we're only using the Balance column from the main section
                if df["Balance"].dtype == object:
                    df["Balance"] = df["Balance"].astype(str).str.replace(",", "")
                df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce").fillna(0)
                # Log sample of Balance values for debugging
                logger.info(
                    f"Sample Balance values after conversion: {df['Balance'].head(5).tolist()}"
                )

                # Fix: Use AvailableQty for Balance when Balance is 0 but AvailableQty is not
                if "AvailableQty" in df.columns:
                    mask = (df["Balance"] == 0) & (df["AvailableQty"] > 0)
                    if mask.any():
                        logger.info(
                            f"Fixing {mask.sum()} rows where Balance is 0 but AvailableQty > 0"
                        )
                        df.loc[mask, "Balance"] = df.loc[mask, "AvailableQty"]

                # Log the updated Balance values
                logger.info(f"Balance values after fixing zeros: {df['Balance'].head(10).tolist()}")

                # Log the Balance values for debugging
                logger.info(f"Balance values after fixing zeros: {df['Balance'].head(10).tolist()}")

            # Convert DaysOnHand to numeric
            if "DaysOnHand" in df.columns:
                df["DaysOnHand"] = pd.to_numeric(df["DaysOnHand"], errors="coerce").fillna(0)

            # Clean up warehouse names and standardize SKU format
            # Filter for only SellableIndividual items if Type column exists
            if "Type" in df.columns:
                # Log unique types for debugging
                unique_types = df["Type"].unique()
                logger.info(f"Unique item types in inventory: {unique_types}")

                # Filter for fruit items (SellableIndividual) to exclude packaging materials
                sellable_df = df[df["Type"] == "SellableIndividual"].copy()
                if not sellable_df.empty:
                    logger.info(
                        f"Filtered inventory to {len(sellable_df)} sellable items out of {len(df)} total items"
                    )
                    df = sellable_df
                else:
                    logger.warning("Warning: No SellableIndividual items found in inventory")

            # Log inventory summary by warehouse
            if "WarehouseName" in df.columns and "AvailableQty" in df.columns:
                inventory_summary = df.groupby("WarehouseName")["AvailableQty"].agg(
                    ["sum", "count"]
                )
                logger.info("Inventory summary by warehouse:")
                logger.info(inventory_summary)

            # Log sample of fruit SKUs for debugging
            if "Sku" in df.columns:
                fruit_skus = df[df["Sku"].str.contains("_") | df["Sku"].str.contains("-")][
                    "Sku"
                ].unique()[:10]
                logger.info(f"Sample fruit SKUs: {fruit_skus}")
            if "WarehouseName" in df.columns:
                df["WarehouseName"] = df["WarehouseName"].astype(str).str.strip()

            if "Sku" in df.columns:
                df["Sku"] = df["Sku"].astype(str).str.strip()

            # Filter out rows with empty or invalid SKUs
            if "Sku" in df.columns:
                df = df[df["Sku"].notna() & (df["Sku"] != "")]

            # Group by warehouse and SKU to get total available quantity
            group_columns = [col for col in ["WarehouseName", "Sku"] if col in df.columns]
            if len(group_columns) == 2:  # Need both warehouse and SKU for proper grouping
                agg_dict = {}
                if "AvailableQty" in df.columns:
                    agg_dict["AvailableQty"] = "sum"
                if "Balance" in df.columns:
                    # Use max instead of sum for Balance to avoid double-counting
                    # This ensures we get the correct balance for each SKU
                    agg_dict["Balance"] = "max"  # Use max instead of sum to avoid double-counting
                if "Name" in df.columns:
                    agg_dict["Name"] = "first"
                if "Type" in df.columns:
                    agg_dict["Type"] = "first"
                if "DaysOnHand" in df.columns:
                    agg_dict["DaysOnHand"] = "mean"

                if agg_dict:  # Only aggregate if we have columns to aggregate
                    # Log before aggregation
                    logger.info(
                        f"Pre-aggregation Balance values for first 5 rows: {df[['Sku', 'Balance']].head(5).to_dict('records')}"
                    )

                    inventory_summary = df.groupby(group_columns).agg(agg_dict).reset_index()

                    # Log after aggregation
                    logger.info(
                        f"Post-aggregation Balance values for first 5 rows: {inventory_summary[['Sku', 'Balance']].head(5).to_dict('records')}"
                    )
                else:
                    inventory_summary = df.copy()

                # Add inventory status column
                if "AvailableQty" in inventory_summary.columns:
                    inventory_summary["Status"] = inventory_summary.apply(
                        lambda row: "Low"
                        if row["AvailableQty"] < 10
                        else ("Critical" if row["AvailableQty"] <= 0 else "Good"),
                        axis=1,
                    )
            else:
                # If we can't group, just return the original dataframe
                inventory_summary = df.copy()
                inventory_summary["Status"] = "Unknown"

            # Log summary statistics
            logger.info(
                f"Loaded inventory with {len(inventory_summary)} unique SKU-warehouse combinations"
            )
            if "WarehouseName" in inventory_summary.columns:
                warehouse_counts = inventory_summary["WarehouseName"].value_counts()
                logger.info(f"Warehouses: {dict(warehouse_counts)}")

            return inventory_summary

        except Exception as e:
            logger.error(f"Error processing inventory file: {str(e)}")
            # Return an empty DataFrame with the required columns as a fallback
            return pd.DataFrame(columns=["WarehouseName", "Sku", "AvailableQty", "Status"])

    def check_inventory_in_other_fc(self, sku, inventory_df, fc_key):
        """
        Check if an item is available in the specified fulfillment center.

        Args:
            sku: The SKU to check
            inventory_df: Inventory dataframe
            fc_key: Fulfillment center key to check (moorpark or wheeling)

        Returns:
            float: Available quantity in the other fulfillment center, 0 if not available
        """
        if inventory_df is None or inventory_df.empty:
            return 0

        # Normalize fulfillment center key using centralized function
        fc_key = self.normalize_warehouse_name(fc_key, log_transformations=False)

        # Find the SKU in the inventory for the specified fulfillment center
        # First, determine which columns to use for SKU and quantity
        sku_col = next((col for col in ["SKU", "Sku", "sku"] if col in inventory_df.columns), None)
        qty_col = next(
            (
                col
                for col in ["AvailableQty", "Available Qty", "Quantity"]
                if col in inventory_df.columns
            ),
            None,
        )

        if sku_col and qty_col:
            # Check if we need to look for both Moorpark and Oxnard
            if fc_key.lower() == "oxnard":
                # Look for both Oxnard and Moorpark in the inventory
                # First check if we have a first column that contains warehouse info
                first_col = inventory_df.columns[0]

                if "FulfillmentCenter" in inventory_df.columns:
                    # Use the standardized column if available
                    matching_rows = inventory_df[
                        (inventory_df[sku_col] == sku)
                        & (inventory_df["FulfillmentCenter"] == "Oxnard")
                    ]
                elif (
                    first_col.startswith("CA-")
                    or inventory_df[first_col].astype(str).str.contains("CA-").any()
                ):
                    # Use the raw warehouse column that contains CA-Moorpark or CA-Oxnard
                    matching_rows = inventory_df[
                        (inventory_df[sku_col] == sku)
                        & (
                            inventory_df[first_col]
                            .astype(str)
                            .str.contains("CA-Moorpark|CA-Oxnard", case=False, na=False, regex=True)
                        )
                    ]
                elif "WarehouseName" in inventory_df.columns:
                    # Use WarehouseName if available
                    matching_rows = inventory_df[
                        (inventory_df[sku_col] == sku)
                        & (
                            inventory_df["WarehouseName"]
                            .astype(str)
                            .str.contains("Moorpark|Oxnard|CA-", case=False, na=False, regex=True)
                        )
                    ]
                else:
                    # Fallback - look for any columns that might contain CA, Moorpark or Oxnard
                    ca_columns = [
                        col
                        for col in inventory_df.columns
                        if any(x in col for x in ["CA", "Moorpark", "Oxnard"])
                    ]
                    if ca_columns:
                        matching_rows = inventory_df[inventory_df[sku_col] == sku]
                    else:
                        matching_rows = pd.DataFrame()  # Empty dataframe as fallback
            else:  # Wheeling
                # Look for Wheeling inventory
                first_col = inventory_df.columns[0]

                if "FulfillmentCenter" in inventory_df.columns:
                    # Use the standardized column if available
                    matching_rows = inventory_df[
                        (inventory_df[sku_col] == sku)
                        & (inventory_df["FulfillmentCenter"] == "Wheeling")
                    ]
                elif (
                    first_col.startswith("IL-")
                    or inventory_df[first_col].astype(str).str.contains("IL-").any()
                ):
                    # Use the raw warehouse column that contains IL-Wheeling
                    matching_rows = inventory_df[
                        (inventory_df[sku_col] == sku)
                        & (
                            inventory_df[first_col]
                            .astype(str)
                            .str.contains("IL-|Wheeling", case=False, na=False, regex=True)
                        )
                    ]
                elif "WarehouseName" in inventory_df.columns:
                    # Use WarehouseName if available
                    matching_rows = inventory_df[
                        (inventory_df[sku_col] == sku)
                        & (
                            inventory_df["WarehouseName"]
                            .astype(str)
                            .str.contains("Wheeling|IL-", case=False, na=False, regex=True)
                        )
                    ]
                else:
                    # Fallback - look for any columns that might contain IL or Wheeling
                    il_columns = [
                        col
                        for col in inventory_df.columns
                        if any(x in col for x in ["IL", "Wheeling"])
                    ]
                    if il_columns:
                        matching_rows = inventory_df[inventory_df[sku_col] == sku]
                    else:
                        matching_rows = pd.DataFrame()  # Empty dataframe as fallback

            if not matching_rows.empty:
                # Sum up all available quantities if multiple rows match
                if len(matching_rows) > 1:
                    return float(matching_rows[qty_col].sum())
                else:
                    # Return the available quantity
                    return float(matching_rows.iloc[0][qty_col])

        return 0

    def find_substitution_options(
        self, missing_sku, needed_qty, inventory_df, sku_mappings=None, fc_key=None
    ):
        """
        Find potential substitutions for a missing bundle component based on similar weight/type.

        Args:
            missing_sku (str): The SKU that is missing or insufficient
            needed_qty (float): The quantity needed
            inventory_df (DataFrame): Inventory dataframe
            sku_mappings (dict, optional): SKU mappings dictionary. Defaults to None.
            fc_key (str, optional): Fulfillment center key (Wheeling or Oxnard). Defaults to None.

        Returns:
            list: List of potential substitution options with similarity scores
        """
        substitution_options = []

        # If no SKU mappings, return empty list
        if not sku_mappings or not fc_key or fc_key not in sku_mappings:
            return substitution_options

        # Find the SKU column in inventory_df
        sku_column = None
        for col in inventory_df.columns:
            if col.lower() == "sku":
                sku_column = col
                break

        if not sku_column:
            return substitution_options

        # Get balance column if it exists
        balance_column = None
        for col in inventory_df.columns:
            if col.lower() in ["balance", "availableqty"]:
                balance_column = col
                break

        if not balance_column:
            return substitution_options

        # Find the missing SKU's details from sku_mappings
        missing_sku_details = None
        missing_sku_type = ""
        missing_sku_weight = 0.0

        # Check if the missing SKU is in all_skus
        if "all_skus" in sku_mappings[fc_key] and missing_sku in sku_mappings[fc_key]["all_skus"]:
            missing_sku_details = sku_mappings[fc_key]["all_skus"][missing_sku]
            missing_sku_weight = float(missing_sku_details.get("Total_Pick_Weight", 0.0))
            missing_sku_type = missing_sku_details.get("Pick_Type", "")

        # If we couldn't find the SKU details, try to find it in inventory
        if not missing_sku_details and sku_column in inventory_df.columns:
            missing_rows = inventory_df[inventory_df[sku_column] == missing_sku]
            if not missing_rows.empty:
                # Try to extract type from SKU name or description
                for col in missing_rows.columns:
                    if col.lower() in ["description", "productname", "name"]:
                        missing_sku_type = missing_rows.iloc[0][col]
                        break

        # Find potential substitutions
        for sku, details in sku_mappings[fc_key].get("all_skus", {}).items():
            # Skip the missing SKU itself
            if sku == missing_sku:
                continue

            # Check if this SKU has sufficient inventory
            current_balance = 0
            if sku_column in inventory_df.columns and balance_column in inventory_df.columns:
                sku_rows = inventory_df[inventory_df[sku_column] == sku]
                if not sku_rows.empty:
                    try:
                        current_balance = (
                            float(sku_rows.iloc[0][balance_column])
                            if pd.notna(sku_rows.iloc[0][balance_column])
                            else 0
                        )
                    except (ValueError, TypeError):
                        current_balance = 0

            # Skip if not enough inventory
            if current_balance < needed_qty:
                continue

            # Calculate similarity score based on weight and type
            similarity_score = 0
            sku_weight = float(details.get("Total_Pick_Weight", 0.0))
            sku_type = details.get("Pick_Type", "")

            # Weight similarity (higher score for closer weights)
            if missing_sku_weight > 0 and sku_weight > 0:
                weight_ratio = min(sku_weight / missing_sku_weight, missing_sku_weight / sku_weight)
                weight_similarity = weight_ratio * 50  # 50% of score based on weight
            else:
                weight_similarity = 0

            # Type similarity (50% of score based on type match)
            type_similarity = 0
            if missing_sku_type and sku_type:
                # Simple string matching for now
                if missing_sku_type.lower() == sku_type.lower():
                    type_similarity = 50  # Exact match
                elif any(word in sku_type.lower() for word in missing_sku_type.lower().split()):
                    type_similarity = 30  # Partial match

            # Total similarity score
            similarity_score = weight_similarity + type_similarity

            # Add to options if score is above threshold
            if similarity_score > 20:  # Arbitrary threshold
                substitution_options.append(
                    {
                        "sku": sku,
                        "picklist_sku": details.get("picklist_sku", ""),
                        "weight": sku_weight,
                        "type": sku_type,
                        "available_qty": current_balance,
                        "similarity_score": similarity_score,
                    }
                )

        # Sort by similarity score (highest first)
        substitution_options.sort(key=lambda x: x["similarity_score"], reverse=True)

        # Return top 3 options
        return substitution_options[:3]

    def generate_shortage_summary(self, shortages):
        """
        Generate a summary table of all inventory shortages.

        Args:
            shortages (list): List of shortage dictionaries with component_sku, shopify_sku, order_id, current_balance, needed_qty, shortage_qty

        Returns:
            tuple: (DataFrame: Detailed shortage summary dataframe, DataFrame: Grouped shortage summary with totals)
        """
        if not shortages:
            return pd.DataFrame(), pd.DataFrame()

        # Convert to dataframe and sort
        shortage_df = pd.DataFrame(shortages)
        grouped_shortage_df = pd.DataFrame()

        if not shortage_df.empty:
            # Create a copy of the dataframe with numeric values for grouping
            numeric_shortage_df = shortage_df.copy()

            # Convert numeric columns to float for calculations
            numeric_cols = ["current_balance", "needed_qty", "shortage_qty"]
            for col in numeric_cols:
                if col in numeric_shortage_df.columns:
                    numeric_shortage_df[col] = pd.to_numeric(
                        numeric_shortage_df[col], errors="coerce"
                    )

            # Create grouped summary with totals
            if "component_sku" in numeric_shortage_df.columns:
                # Check if fulfillment_center column exists
                fc_column_exists = "fulfillment_center" in numeric_shortage_df.columns

                # Group by component_sku, shopify_sku, and fulfillment_center if it exists
                group_columns = ["component_sku", "shopify_sku"]
                if fc_column_exists:
                    group_columns.append("fulfillment_center")

                # Collect order IDs for each group
                order_id_list = numeric_shortage_df.groupby(group_columns)["order_id"].apply(list)

                grouped_shortage_df = (
                    numeric_shortage_df.groupby(group_columns)
                    .agg(
                        {
                            "needed_qty": "sum",
                            "shortage_qty": "sum",
                            "order_id": "count",  # Count of affected orders
                        }
                    )
                    .reset_index()
                )

                # Add the list of order IDs to the grouped dataframe
                grouped_shortage_df = grouped_shortage_df.merge(
                    order_id_list.reset_index().rename(columns={"order_id": "order_ids"}),
                    on=group_columns,
                )

                # Rename columns for clarity
                grouped_shortage_df = grouped_shortage_df.rename(
                    columns={
                        "needed_qty": "total_needed_qty",
                        "shortage_qty": "total_shortage_qty",
                        "order_id": "affected_orders_count",
                    }
                )

                # Sort by component_sku and fulfillment_center if it exists
                sort_columns = ["component_sku"]
                if fc_column_exists:
                    sort_columns.append("fulfillment_center")
                grouped_shortage_df = grouped_shortage_df.sort_values(sort_columns)

            # Sort detailed view by SKU and order ID
            if "component_sku" in shortage_df.columns and "order_id" in shortage_df.columns:
                shortage_df = shortage_df.sort_values(["component_sku", "order_id"])

            # Format numeric columns in the detailed view
            for col in numeric_cols:
                if col in shortage_df.columns:
                    shortage_df[col] = shortage_df[col].apply(
                        lambda x: str(int(float(x)))
                        if pd.notnull(x) and float(x) == int(float(x))
                        else ("{:.3f}".format(float(x)) if pd.notnull(x) else x)
                    )

            # Format numeric columns in the grouped view
            numeric_cols = ["total_needed_qty", "total_shortage_qty", "affected_orders_count"]
            for col in numeric_cols:
                if col in grouped_shortage_df.columns:
                    grouped_shortage_df[col] = grouped_shortage_df[col].apply(
                        lambda x: str(int(float(x)))
                        if pd.notnull(x) and float(x) == int(float(x))
                        else ("{:.3f}".format(float(x)) if pd.notnull(x) else x)
                    )

        return shortage_df, grouped_shortage_df

    def generate_inventory_summary(self, running_balances, inventory_df, sku_mappings=None):
        """
        Generate a summary table of current inventory levels after order processing.

        Args:
            running_balances (dict): Dictionary of SKU to current balance after order processing
            inventory_df (DataFrame): Original inventory dataframe
            sku_mappings (dict, optional): SKU mappings dictionary. Defaults to None.

        Returns:
            DataFrame: Inventory summary dataframe with current levels
        """
        # Create a dataframe from the running balances
        inventory_summary = []

        # Find the SKU column in inventory_df
        sku_column = None
        for col in inventory_df.columns:
            if col.lower() == "sku":
                sku_column = col
                break

        if not sku_column:
            logger.warning("No 'SKU' column found in inventory dataframe")
            return pd.DataFrame()

        # Get warehouse column if it exists
        warehouse_column = None
        for col in inventory_df.columns:
            if col.lower() in ["warehousename", "warehouse"]:
                warehouse_column = col
                break

        # Create a mapping from inventory SKU to Shopify SKU if available
        inv_to_shopify = {}
        if sku_mappings:
            for fc_key in sku_mappings:
                if "all_skus" in sku_mappings[fc_key]:
                    for shopify_sku, data in sku_mappings[fc_key]["all_skus"].items():
                        if "picklist_sku" in data:
                            inv_to_shopify[data["picklist_sku"]] = shopify_sku

        # Process all inventory items directly from inventory_df to ensure we get all warehouses
        processed_items = set()  # Track processed (sku, warehouse) combinations
        
        # First process items from running_balances (items that were affected by orders)
        for composite_key, balance in running_balances.items():
            if "|" in composite_key:
                # New composite key format: sku|warehouse
                sku, warehouse = composite_key.split("|", 1)
            else:
                # Legacy format: just sku - need to find warehouse from inventory
                sku = composite_key
                warehouse = "Unknown"
                if warehouse_column and sku_column:
                    sku_rows = inventory_df[inventory_df[sku_column] == sku]
                    if not sku_rows.empty and warehouse_column in sku_rows.columns:
                        raw_warehouse = sku_rows.iloc[0][warehouse_column]
                        warehouse = self.normalize_warehouse_name(raw_warehouse, log_transformations=False)

            # Try to find the Shopify SKU for this inventory SKU
            shopify_sku = inv_to_shopify.get(sku, "")

            # Add to summary
            inventory_summary.append(
                {
                    "Warehouse": warehouse,
                    "Inventory SKU": sku,
                    "Shopify SKU": shopify_sku,
                    "Current Balance": balance,
                    "Is Bundle Component": shopify_sku.startswith("m.") if shopify_sku else False,
                }
            )
            
            # Mark this combination as processed
            processed_items.add((sku, warehouse))

        # Now add any SKUs from inventory that weren't processed above
        if sku_column and warehouse_column:
            for _, row in inventory_df.iterrows():
                sku = row[sku_column]
                raw_warehouse = row[warehouse_column] if warehouse_column in row else "Unknown"
                warehouse = self.normalize_warehouse_name(raw_warehouse, log_transformations=False) if raw_warehouse != "Unknown" else "Unknown"
                
                # Skip if this (sku, warehouse) combination was already processed
                if (sku, warehouse) not in processed_items:
                    shopify_sku = inv_to_shopify.get(sku, "")

                    # Get the balance from inventory - use only the main Balance column (columns 1-11)
                    balance = 0

                    # Use Balance column from the main section
                    if "Balance" in row.index:
                        try:
                            # Get value from the main Balance column
                            bal_value = row["Balance"]
                            if isinstance(bal_value, str):
                                bal_value = bal_value.replace(",", "")
                            balance = float(bal_value) if pd.notna(bal_value) else 0
                            logger.info(f"Using main Balance column for {sku}: {balance}")
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Could not convert Balance value for {sku}: {row['Balance']} - Error: {e}"
                            )

                    # Only use AvailableQty if Balance is not available
                    if balance == 0 and "AvailableQty" in row.index:
                        try:
                            avail_value = row["AvailableQty"]
                            if isinstance(avail_value, str):
                                avail_value = avail_value.replace(",", "")
                            avail_qty = float(avail_value) if pd.notna(avail_value) else 0
                            balance = avail_qty
                            logger.info(
                                f"Using AvailableQty column for {sku} because Balance is not available: {balance}"
                            )
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Could not convert AvailableQty value for {sku}: {row['AvailableQty']} - Error: {e}"
                            )

                    # Log the final balance value
                    logger.info(f"Final balance for {sku}: {balance}")

                    inventory_summary.append(
                        {
                            "Warehouse": warehouse,
                            "Inventory SKU": sku,
                            "Shopify SKU": shopify_sku,
                            "Current Balance": balance,
                            "Is Bundle Component": shopify_sku.startswith("m.")
                            if shopify_sku
                            else False,
                        }
                    )

        # Convert to dataframe and sort
        summary_df = pd.DataFrame(inventory_summary)
        if not summary_df.empty:
            # Sort by warehouse, then by whether it's a bundle component, then by SKU
            summary_df = summary_df.sort_values(
                ["Warehouse", "Is Bundle Component", "Inventory SKU"]
            )

            # Ensure Current Balance is numeric
            summary_df["Current Balance"] = pd.to_numeric(
                summary_df["Current Balance"], errors="coerce"
            ).fillna(0)

            # Format Current Balance with commas for thousands
            summary_df["Current Balance"] = summary_df["Current Balance"].apply(
                lambda x: f"{int(x):,}" if x == int(x) else f"{x:,.2f}"
            )

            # Log sample of Current Balance values for debugging
            logger.info(
                f"Final Current Balance values in inventory summary: {summary_df['Current Balance'].head(5).tolist()}"
            )

        return summary_df

    def process_orders(
        self,
        orders_df,
        inventory_df,
        shipping_zones_df=None,
        sku_mappings=None,
        sku_weight_data=None,
        affected_skus=None,
    ):
        """Process orders and apply SKU mappings.

        Args:
            orders_df (DataFrame): Orders dataframe
            inventory_df (DataFrame): Inventory dataframe
            shipping_zones_df (DataFrame, optional): Shipping zones dataframe. Defaults to None.
            sku_mappings (dict, optional): SKU mappings dictionary. Defaults to None.
            sku_weight_data (dict, optional): SKU weight data dictionary. Defaults to None.
            affected_skus (set, optional): Set of SKUs that were affected by changes. If provided, only these SKUs
                                           will be recalculated for inventory updates. Defaults to None.

        Returns:
            tuple: (DataFrame: Processed orders dataframe, list: Inventory issues)
        """
        # Extract the mappings and bundle info from the new structure if needed
        mappings_dict = None
        bundle_info_dict = None

        if isinstance(sku_mappings, dict) and "mappings" in sku_mappings:
            # New format with separate mappings and bundle_info
            mappings_dict = sku_mappings["mappings"]
            if "bundle_info" in sku_mappings:
                bundle_info_dict = sku_mappings["bundle_info"]
            logger.info(
                f"Using new format SKU mappings with {sum(len(mappings_dict[k]) for k in mappings_dict)} total mappings"
            )
        else:
            # Old format (direct dictionary)
            mappings_dict = sku_mappings
            logger.info("Using old format SKU mappings")

        # Initialize a fresh dictionary to track running balances for each SKU
        # Always create a new copy to avoid state persistence across calls
        running_balances = {}
        
        # Store initial inventory state for before/after comparison
        initial_inventory_state = {}
        if inventory_df is not None and not inventory_df.empty:
            # Find the SKU column with case-insensitive matching
            sku_column = None
            for col in inventory_df.columns:
                if col.lower() == 'sku':
                    sku_column = col
                    break
            
            if sku_column:
                # Store initial balance for each SKU
                for _, row in inventory_df.iterrows():
                    sku = str(row[sku_column]).strip()
                    warehouse = None
                    if 'WarehouseName' in row:
                        warehouse = str(row['WarehouseName']).strip()
                    
                    # Try to get balance from different possible column names
                    balance = 0
                    if 'Balance' in row and pd.notna(row['Balance']):
                        try:
                            balance = float(row['Balance'])
                        except (ValueError, TypeError):
                            pass
                    elif 'AvailableQty' in row and pd.notna(row['AvailableQty']):
                        try:
                            balance = float(row['AvailableQty'])
                        except (ValueError, TypeError):
                            pass
                    
                    # Use composite key format: SKU|warehouse to track inventory per warehouse
                    # This ensures we don't overwrite inventory from different warehouses
                    normalized_warehouse = self.normalize_warehouse_name(warehouse, log_transformations=False) if warehouse else "Unknown"
                    composite_key = f"{sku}|{normalized_warehouse}"
                    
                    initial_inventory_state[composite_key] = {
                        'balance': balance,
                        'warehouse': normalized_warehouse,
                        'sku': sku
                    }
                    
                    # Initialize running balances with composite key
                    running_balances[composite_key] = balance
                
                logger.info(f"Initialized initial inventory state with {len(initial_inventory_state)} entries")
                # Log a few sample entries
                sample_items = list(initial_inventory_state.items())[:5]
                logger.info(f"Sample initial inventory: {sample_items}")

        # If we have affected_skus, initialize their balances from inventory_df
        # This allows us to only recalculate the affected SKUs
        if affected_skus:
            logger.info(f"Selective recalculation for {len(affected_skus)} affected SKUs")
            # Initialize running balances for affected SKUs from inventory
            for sku in affected_skus:
                # Find this SKU in inventory
                # Use 'Sku' column (not 'SKU') based on the inventory data structure
                sku_inventory = inventory_df[inventory_df["Sku"] == sku]
                if not sku_inventory.empty:
                    # Process each warehouse separately to maintain composite keys
                    for _, row in sku_inventory.iterrows():
                        warehouse = row.get("WarehouseName", "Unknown")
                        normalized_warehouse = self.normalize_warehouse_name(warehouse, log_transformations=False) if warehouse else "Unknown"
                        composite_key = f"{sku}|{normalized_warehouse}"
                        
                        # Use the Balance column if available, otherwise AvailableQty
                        balance = 0.0
                        if "Balance" in row and pd.notna(row["Balance"]):
                            balance = float(row["Balance"])
                        elif "AvailableQty" in row and pd.notna(row["AvailableQty"]):
                            balance = float(row["AvailableQty"])
                        
                        running_balances[composite_key] = balance
                        
                        # Also update initial_inventory_state for consistency
                        initial_inventory_state[composite_key] = {
                            'balance': balance,
                            'warehouse': normalized_warehouse,
                            'sku': sku
                        }
                        
                        # Log the initial balance for this SKU
                        self._recalculation_log.append({
                            "timestamp": pd.Timestamp.now(),
                            "sku": sku,
                            "operation": "initial_balance",
                            "value": balance,
                            "warehouse": normalized_warehouse
                        })

        # Initialize a list to track inventory shortages
        all_shortages = []

        # Keep track of unique shortages to prevent duplications
        # Using a set of tuples (component_sku, order_id) to track what's been added
        shortage_tracker = set()
        
        # Clear previous recalculation log if starting a new recalculation
        self._recalculation_log = []
        # Clear recalculated SKUs tracking
        self._recalculated_skus = set()

        # Load SKU weight data from mapping files if not provided
        if sku_weight_data is None:
            sku_weight_data = {}
            # Try to load from constants directory
            try:
                import os

                # Load SKU mappings from JSON file
                sku_mappings_path = os.path.join("constants", "data", "sku_mappings.json")
                if os.path.exists(sku_mappings_path):
                    try:
                        with open(sku_mappings_path, "r") as f:
                            sku_mappings = json.load(f)

                        # Extract Wheeling data if available
                        if "Wheeling" in sku_mappings and "all_skus" in sku_mappings["Wheeling"]:
                            wheeling_data = {}
                            for sku, data in sku_mappings["Wheeling"]["all_skus"].items():
                                # Store the picklist_sku as well
                                wheeling_data[sku] = {
                                    "actualqty": data.get("actualqty", ""),
                                    "Total Pick Weight": data.get("Total_Pick_Weight", ""),
                                    "picklist_sku": data.get("picklist_sku", ""),
                                }
                                # Also add the SKU without 'f.' prefix if it has one
                                if sku.startswith("f."):
                                    wheeling_data[sku[2:]] = wheeling_data[sku]
                                # Also add with 'f.' prefix if it doesn't have one
                                elif not sku.startswith("f."):
                                    wheeling_data[f"f.{sku}"] = wheeling_data[sku]

                            sku_weight_data["Wheeling"] = wheeling_data
                            logger.info(
                                f"Loaded weight data for {len(wheeling_data)} SKUs from Wheeling mapping file"
                            )
                    except Exception as e:
                        logger.error(f"Error loading SKU mappings from JSON file: {e}")

                # Oxnard data should already be in the sku_mappings.json file
                # Extract Oxnard data if available
                if "Oxnard" in sku_mappings and "all_skus" in sku_mappings["Oxnard"]:
                    oxnard_data = {}
                    for sku, data in sku_mappings["Oxnard"]["all_skus"].items():
                        oxnard_data[sku] = {
                            "actualqty": data.get("actualqty", ""),
                            "Total Pick Weight": data.get("Total_Pick_Weight", ""),
                        }
                        # Also add the SKU without 'f.' prefix if it has one
                        if sku.startswith("f."):
                            oxnard_data[sku[2:]] = oxnard_data[sku]

                    sku_weight_data["Oxnard"] = oxnard_data
                    logger.info(
                        f"Loaded weight data for {len(oxnard_data)} SKUs from Oxnard mapping file"
                    )
            except Exception as e:
                logger.error(f"Error loading SKU weight data: {str(e)}")
        # Define the streamlined output columns based on the fulfillment-focused structure
        # Removed delivery service-related columns (carrier_name, service_name, delivery_days)
        output_columns = [
            "externalorderid",
            "ordernumber",
            "CustomerFirstName",
            "customerLastname",
            "customeremail",
            "customerphone",
            "shiptoname",
            "shiptostreet1",
            "shiptostreet2",
            "shiptocity",
            "shiptostate",
            "shiptopostalcode",
            "note",
            "placeddate",
            "preferredcarrierserviceid",
            "totalorderamount",
            "shopsku",
            "shopquantity",
            "externalid",
            "Tags",
            "MAX PKG NUM",
            "Fulfillment Center",
            "shopifysku2",
            "sku",
            "actualqty",
            "Total Pick Weight",
            "quantity",
            "Starting Balance",
            "Transaction Quantity",
            "Ending Balance",
            "Issues",
        ]

        output_df = pd.DataFrame(columns=output_columns)

        # Rename columns to match expected format if needed
        column_mapping = {
            "Name": "externalorderid",  # Name contains order reference like #71184
            "order id": "ordernumber",  # order id contains the numeric ID
            "Customer: First Name": "CustomerFirstName",
            "Customer: Last Name": "customerLastname",
            "Email": "customeremail",
            "Shipping: Name": "shiptoname",
            "Shipping: Address 1": "shiptostreet1",
            "Shipping: Address 2": "shiptostreet2",
            "Shipping: City": "shiptocity",
            "Shipping: Province Code": "shiptostate",
            "Shipping: Zip": "shiptopostalcode",
            "Date": "placeddate",
            "Note": "note",
            "SKU Helper": "shopsku",
            "Line: Fulfillable Quantity": "shopquantity",
            "Line: ID": "externalid",
            "NEW Tags": "Tags",
            "MAX PKG NUM": "MAX PKG NUM",
            "Fulfillment Center": "Fulfillment Center",
            "Saturday Shipping": "Saturday Shipping",
        }

        # Apply column mapping where columns exist
        for old_col, new_col in column_mapping.items():
            if old_col in orders_df.columns and new_col not in orders_df.columns:
                orders_df[new_col] = orders_df[old_col]

        # Check for SKU columns in the dataframe
        # Some SKUs might be in columns with fruit names like 'f.loquat-5lb'
        sku_columns = [
            col for col in orders_df.columns if col.startswith("f.") or col.startswith("m.")
        ]

        # Track which orders we've processed for selective recalculation
        processed_order_ids = set()

        # Process each order
        for _, row in orders_df.iterrows():
            # Get order ID for tracking
            order_id = row.get("externalorderid", row.get("ordernumber", "unknown"))

            # First try to get SKU from shopsku column
            shopify_sku = None
            if "shopsku" in row and pd.notna(row["shopsku"]):
                shopify_sku = row["shopsku"]
            elif "SKU Helper" in row and pd.notna(row["SKU Helper"]):
                shopify_sku = row["SKU Helper"]
            else:
                # Check other possible SKU columns
                standard_sku_columns = ["sku", "lineitemsku", "shopifysku", "shopifysku2"]
                for col in standard_sku_columns:
                    if col in row and pd.notna(row[col]):
                        shopify_sku = row[col]
                        break

                # If still no SKU, check fruit-specific columns
                if shopify_sku is None:
                    for col in sku_columns:
                        if col in row and pd.notna(row[col]) and float(row[col]) > 0:
                            shopify_sku = col
                            break

            # Skip this order if no SKU found
            if shopify_sku is None:
                logger.warning(
                    f"Warning: No SKU found for order {row.get('externalorderid', 'unknown')}"
                )
                continue

            # Determine fulfillment center for this order
            fulfillment_center = row.get("Fulfillment Center", "Moorpark")

            # Map Shopify SKU to warehouse SKU using the mappings
            warehouse_sku = shopify_sku

            # Convert fulfillment center name to match the keys in sku_mappings
            # Use centralized warehouse normalization function
            if fulfillment_center:
                fc_key = self.normalize_warehouse_name(fulfillment_center, log_transformations=True)
            else:
                fc_key = "Oxnard"  # Default for missing fulfillment center

            # Skip this order if selective recalculation is enabled and this order doesn't contain affected SKUs
            if affected_skus is not None and shopify_sku is not None:
                # Check if this SKU is in the affected_skus set
                sku_affected = shopify_sku in affected_skus

                # If it's a bundle, check if any of its components are in affected_skus
                if (
                    not sku_affected
                    and bundle_info_dict
                    and fc_key in bundle_info_dict
                    and shopify_sku in bundle_info_dict[fc_key]
                ):
                    # Check if any component SKU is in affected_skus
                    for component in bundle_info_dict[fc_key][shopify_sku]:
                        component_sku = component.get("component_sku")
                        if component_sku in affected_skus:
                            sku_affected = True
                            # Log that this bundle is affected because of a component
                            self._recalculation_log.append({
                                "timestamp": pd.Timestamp.now(),
                                "sku": shopify_sku,
                                "operation": "bundle_affected",
                                "value": component_sku,
                                "order_id": order_id,
                                "reason": f"Bundle affected by component {component_sku}"
                            })
                            break

                # Skip this order if neither the SKU nor its components are affected
                if not sku_affected:
                    continue

                # Track that we're processing this order
                processed_order_ids.add(order_id)
                
                # Log that we're processing this order due to affected SKU
                self._recalculation_log.append({
                    "timestamp": pd.Timestamp.now(),
                    "sku": shopify_sku,
                    "operation": "process_order",
                    "value": "",
                    "order_id": order_id,
                    "reason": "SKU directly affected by changes"
                })
            else:
                # Do not default to any fulfillment center if none is specified
                # This is a critical logistics task and we need to preserve the original value
                if not fulfillment_center:
                    # Just log the warning once per order
                    warning_key = f"no_fc_specified_{order_id}"
                    if warning_key not in self._logged_inventory_issues:
                        logger.warning(f"Warning: No fulfillment center specified for order {order_id}")
                        self._logged_inventory_issues.add(warning_key)
            # Apply SKU mapping if available
            # Use the mappings_dict we extracted earlier
            fc_mappings = {}
            if mappings_dict and fc_key in mappings_dict:
                # Get the mapping for this fulfillment center
                fc_mappings = mappings_dict[fc_key]

                # If this SKU has a mapping, use it
                if (
                    shopify_sku in fc_mappings
                    and fc_mappings[shopify_sku]
                    and fc_mappings[shopify_sku] != "nan"
                ):
                    warehouse_sku = fc_mappings[shopify_sku]
                    logger.info(
                        f"Mapped Shopify SKU '{shopify_sku}' to warehouse SKU '{warehouse_sku}' for {fc_key}"
                    )
                else:
                    logger.info(
                        f"No SKU mapping found for '{shopify_sku}' in {fc_key}, using original SKU"
                    )
            else:
                logger.info(f"No SKU mappings available for {fc_key}, using original SKU")

            # Use the warehouse SKU for inventory matching
            sku = warehouse_sku

            # Map SKU to inventory SKU
            # Pass the appropriate fulfillment center mappings
            inventory_sku, inventory_balance, issue = self._map_sku_to_inventory(
                warehouse_sku,
                inventory_df,
                sku_mappings.get(fc_key),
                bundle_info_dict,
                fc_key,
                running_balances,  # Pass running balances to the mapping function
            )

            # Extract priority tag
            if "Tags" in row and pd.notna(row["Tags"]):
                tags = str(row["Tags"]).upper()
                if "P1" in tags or "P*" in tags:
                    pass

            # Extract ZIP code for shipping zone lookup
            if "shiptopostalcode" in row and pd.notna(row["shiptopostalcode"]):
                str(row["shiptopostalcode"])

            # Create order data dictionary with basic fields
            order_data = {col: "" for col in output_columns}  # Initialize with empty strings

            # Copy existing fields from the order row
            for field in output_columns:
                if field in row and pd.notna(row[field]):
                    if field == "placeddate":
                        # Format date to match expected format (M/D/YYYY)
                        try:
                            date_value = pd.to_datetime(row[field])
                            order_data[field] = date_value.strftime("%-m/%-d/%Y")  # Format as M/D/YYYY
                        except:
                            order_data[field] = row[field]  # Keep original if conversion fails
                    else:
                        order_data[field] = row[field]

            # Add SKU information
            order_data["shopsku"] = shopify_sku
            order_data["shopifysku2"] = shopify_sku  # Also add as shopifysku2 for compatibility

            # Check if this is a bundle
            is_bundle = False
            bundle_components = []

            if (
                bundle_info_dict
                and fc_key in bundle_info_dict
                and shopify_sku in bundle_info_dict[fc_key]
            ):
                is_bundle = True
                bundle_components = bundle_info_dict[fc_key][shopify_sku]
                logger.info(
                    f"Found bundle {shopify_sku} with {len(bundle_components)} components: {bundle_components}"
                )

            # Find the SKU and balance columns in the inventory DataFrame
            sku_column = None
            balance_column = None
            for col in inventory_df.columns:
                if col.lower() == "sku":
                    sku_column = col
                elif col.lower() == "balance":
                    balance_column = col

            if sku_column is None:
                logger.warning("Warning: 'Sku' column not found in inventory DataFrame")
                continue

            # Add fulfillment center
            order_data["Fulfillment Center"] = fulfillment_center

            # Add weight and actual quantity information from the JSON file
            # First try to get data from the mappings_dict structure
            found_match = False

            # Check if we have the new format with mappings in the JSON file
            if mappings_dict and fc_key in mappings_dict:
                # Try to find the SKU in the JSON file structure
                if shopify_sku in mappings_dict[fc_key]:
                    # For the new format, we need to look up the data in the original JSON file
                    try:
                        json_path = os.path.join(
                            os.path.dirname(os.path.dirname(__file__)),
                            "constants",
                            "data",
                            "sku_mappings.json",
                        )

                        if os.path.exists(json_path):
                            with open(json_path, "r") as f:
                                sku_json = json.load(f)

                                # Check if the SKU exists in the all_skus section
                                if (
                                    "all_skus" in sku_json[fc_key]
                                    and shopify_sku in sku_json[fc_key]["all_skus"]
                                ):
                                    sku_data = sku_json[fc_key]["all_skus"][shopify_sku]
                                    order_data["actualqty"] = sku_data.get("actualqty", "")
                                    order_data["Total Pick Weight"] = sku_data.get(
                                        "Total_Pick_Weight", ""
                                    )
                                    logger.info(
                                        f"Found weight/quantity data for {shopify_sku} in JSON: actualqty={order_data['actualqty']}, weight={order_data['Total Pick Weight']}"
                                    )
                                    found_match = True
                    except Exception as e:
                        logger.error(f"Error loading SKU data from JSON: {e}")

            # If we didn't find a match in the JSON file, try the old sku_weight_data
            if not found_match and sku_weight_data:
                if fc_key in sku_weight_data and shopify_sku in sku_weight_data[fc_key]:
                    sku_data = sku_weight_data[fc_key][shopify_sku]
                    order_data["actualqty"] = sku_data.get("actualqty", "")
                    order_data["Total Pick Weight"] = sku_data.get("Total Pick Weight", "")
                    logger.info(
                        f"Found weight/quantity data for {shopify_sku}: actualqty={order_data['actualqty']}, weight={order_data['Total Pick Weight']}"
                    )
                    found_match = True
                else:
                    # Try alternative SKU formats
                    alt_skus = []
                    if shopify_sku.startswith("f."):
                        alt_skus.append(shopify_sku[2:])  # Try without 'f.' prefix
                    else:
                        alt_skus.append(f"f.{shopify_sku}")  # Try with 'f.' prefix

                    # Try case variations
                    alt_skus.append(shopify_sku.lower())
                    alt_skus.append(shopify_sku.upper())

                    # Check all alternative formats
                    for alt_sku in alt_skus:
                        if fc_key in sku_weight_data and alt_sku in sku_weight_data[fc_key]:
                            sku_data = sku_weight_data[fc_key][alt_sku]
                            order_data["actualqty"] = sku_data.get("actualqty", "")
                            order_data["Total Pick Weight"] = sku_data.get("Total Pick Weight", "")
                            logger.info(
                                f"Found weight/quantity data for alternative SKU {alt_sku}: actualqty={order_data['actualqty']}, weight={order_data['Total Pick Weight']}"
                            )
                            found_match = True
                            break

                    # Try the other fulfillment center's mapping as a fallback
                    if not found_match:
                        other_fc = "Oxnard" if fc_key == "Wheeling" else "Wheeling"
                        if other_fc in sku_weight_data and shopify_sku in sku_weight_data[other_fc]:
                            sku_data = sku_weight_data[other_fc][shopify_sku]
                            order_data["actualqty"] = sku_data.get("actualqty", "")
                            order_data["Total Pick Weight"] = sku_data.get("Total Pick Weight", "")
                            logger.info(
                                f"Found weight/quantity data in {other_fc} for {shopify_sku}: actualqty={order_data['actualqty']}, weight={order_data['Total Pick Weight']}"
                            )
                            found_match = True

            # If still no match, set empty values
            if not found_match:
                other_fc = "Oxnard" if fc_key == "Wheeling" else "Wheeling"
                if other_fc in sku_weight_data and shopify_sku in sku_weight_data[other_fc]:
                    sku_data = sku_weight_data[other_fc][shopify_sku]
                    order_data["actualqty"] = sku_data.get("actualqty", "")
                    order_data["Total Pick Weight"] = sku_data.get("Total Pick Weight", "")
                    logger.info(
                        f"Found weight/quantity data in {other_fc} for {shopify_sku}: actualqty={order_data['actualqty']}, weight={order_data['Total Pick Weight']}"
                    )
                    found_match = True

            # If still no match, set empty values
            if not found_match:
                logger.info(f"No weight/quantity data found for {shopify_sku} in any mapping")
                order_data["actualqty"] = ""
                order_data["Total Pick Weight"] = ""

            # Add balance information
            shop_quantity = (
                int(row.get("shopquantity", 1)) if pd.notna(row.get("shopquantity")) else 1
            )
            order_data["shopquantity"] = shop_quantity

            # Set the quantity field based on actualqty if available, otherwise use shopquantity
            if order_data.get("actualqty") and str(order_data["actualqty"]).strip():
                try:
                    order_data["quantity"] = float(order_data["actualqty"])
                except (ValueError, TypeError):
                    order_data["quantity"] = shop_quantity
            else:
                order_data["quantity"] = shop_quantity

            # Process differently based on whether this is a bundle or not
            if is_bundle and bundle_components:
                # For bundles, create a separate row for each component
                for component in bundle_components:
                    # Create a copy of the order data for this component
                    component_order_data = order_data.copy()

                    # Get the component SKU and quantity
                    component_sku = component["sku"]
                    component_qty = float(component.get("qty", 1.0))
                    component_weight = float(component.get("weight", 0.0))

                    # Set component data
                    component_order_data["sku"] = component_sku
                    component_order_data["actualqty"] = component_qty
                    component_order_data["Total Pick Weight"] = component_weight
                    component_order_data["quantity"] = component_qty

                    # Get the current running balance for this SKU using composite key
                    component_composite_key = f"{component_sku}|{fc_key.capitalize()}"
                    if component_composite_key in running_balances:
                        current_balance = running_balances[component_composite_key]
                    elif component_sku in running_balances:  # Fallback to simple key
                        current_balance = running_balances[component_sku]
                    else:
                        # Look up the balance in inventory
                        component_inventory = inventory_df[
                            inventory_df[sku_column] == component_sku
                        ]
                        if component_inventory.empty:
                            logger.warning(
                                f"Component {component_sku} not found in inventory for bundle {shopify_sku}"
                            )
                            component_order_data[
                                "Issues"
                            ] = f"Bundle component {component_sku} not found in inventory"
                            component_order_data["Starting Balance"] = 0
                            component_order_data["Transaction Quantity"] = component_qty
                            component_order_data["Ending Balance"] = 0
                            # Add the component order to the output DataFrame
                            output_df = pd.concat(
                                [output_df, pd.DataFrame([component_order_data])], ignore_index=True
                            )
                            continue
                        else:
                            # Get the balance from inventory
                            if balance_column in component_inventory.columns:
                                current_balance = (
                                    float(component_inventory.iloc[0][balance_column])
                                    if pd.notna(component_inventory.iloc[0][balance_column])
                                    else 0
                                )
                            else:
                                current_balance = 0

                    # Check if we have enough inventory
                    if current_balance < component_qty:
                        # Not enough inventory to satisfy order
                        shortage = component_qty - current_balance

                        # Track this shortage for the summary
                        order_id = row.get("externalorderid", "") or row.get("id", "")

                        # Create a unique key for this shortage to prevent duplications
                        shortage_key = (component_sku, str(order_id))

                        # Only add if we haven't seen this exact shortage before
                        if shortage_key not in shortage_tracker:
                            all_shortages.append(
                                {
                                    "component_sku": component_sku,
                                    "shopify_sku": shopify_sku,
                                    "order_id": order_id,
                                    "current_balance": current_balance,
                                    "needed_qty": component_qty,
                                    "shortage_qty": shortage,
                                    "fulfillment_center": fc_key,
                                }
                            )
                            # Add to tracker to prevent future duplicates
                            shortage_tracker.add(shortage_key)

                        # Find potential substitutions
                        substitution_options = self.find_substitution_options(
                            component_sku, shortage, inventory_df, sku_mappings, fc_key
                        )

                        # Create the issue message
                        issue_message = f"Issue: Insufficient inventory | Item: {component_sku} | Shopify SKU: {shopify_sku} | Current: {current_balance} | Needed: {component_qty} | Short by: {shortage} units"

                        # Add substitution suggestions if available
                        if substitution_options:
                            issue_message += "\nSuggested substitutions:\n"
                            for i, option in enumerate(substitution_options, 1):
                                issue_message += f"{i}. {option['sku']} (weight: {option['weight']}, available: {option['available_qty']}, similarity: {option['similarity_score']}%)\n"

                        component_order_data["Issues"] = issue_message
                        
                        # Create a unique key for this inventory issue to prevent duplicate warnings
                        issue_key = f"bundle_shortage_{component_sku}_{shopify_sku}_{shortage}"
                        
                        # Only log if we haven't seen this exact issue before
                        if issue_key not in self._logged_inventory_issues:
                            logger.warning(
                                f"Insufficient inventory for bundle component {component_sku} (Shopify SKU: {shopify_sku}): current balance {current_balance}, needed {component_qty}, short by {shortage} units"
                            )
                            # Add to tracked issues to prevent duplicate warnings
                            self._logged_inventory_issues.add(issue_key)

                        # Use what we have
                        transaction_quantity = current_balance
                    else:
                        transaction_quantity = component_qty

                    # Set the starting balance for this component
                    component_order_data["Starting Balance"] = current_balance
                    component_order_data["Transaction Quantity"] = transaction_quantity

                    # Calculate ending balance and update running balance
                    ending_balance = max(0, current_balance - transaction_quantity)
                    component_order_data["Ending Balance"] = ending_balance

                    # Update the running balance for this component using composite key
                    running_balances[component_composite_key] = ending_balance

                    # Add the component order to the output DataFrame
                    output_df = pd.concat(
                        [output_df, pd.DataFrame([component_order_data])], ignore_index=True
                    )
            else:
                # For regular SKUs, process normally
                order_data["sku"] = inventory_sku  # Mapped inventory SKU

                # Add issue if one was returned from _map_sku_to_inventory
                if issue:
                    order_data["Issues"] = issue

                # Get the current running balance for this SKU or use the inventory balance if not yet tracked
                inventory_composite_key = f"{inventory_sku}|{fc_key.capitalize()}"
                if inventory_composite_key in running_balances:
                    current_balance = running_balances[inventory_composite_key]
                elif inventory_sku in running_balances:  # Fallback to simple key
                    current_balance = running_balances[inventory_sku]
                else:
                    current_balance = inventory_balance

                # Set the starting balance for this order
                order_data["Starting Balance"] = current_balance
                order_data["Transaction Quantity"] = shop_quantity

                # Calculate ending balance and update running balance
                transaction_quantity = (
                    order_data["quantity"]
                    if isinstance(order_data["quantity"], (int, float))
                    else shop_quantity
                )

                # Check if we have enough inventory
                if current_balance < transaction_quantity:
                    # Not enough inventory to satisfy order
                    shortage = transaction_quantity - current_balance

                    # Track this shortage for the summary
                    order_id = row.get("externalorderid", "") or row.get("id", "")

                    # Create a unique key for this shortage to prevent duplications
                    shortage_key = (inventory_sku, str(order_id))

                    # Only add if we haven't seen this exact shortage before
                    if shortage_key not in shortage_tracker:
                        all_shortages.append(
                            {
                                "component_sku": inventory_sku,
                                "shopify_sku": shopify_sku,
                                "order_id": order_id,
                                "current_balance": current_balance,
                                "needed_qty": transaction_quantity,
                                "shortage_qty": shortage,
                                "fulfillment_center": fc_key.capitalize() if fc_key else "Unknown",
                            }
                        )
                        # Add to tracker to prevent future duplicates
                        shortage_tracker.add(shortage_key)

                    # Determine if this is a zero inventory item or just a partial shortage
                    if current_balance == 0:
                        # Zero inventory - check if it's available in the other fulfillment center
                        other_fc = "wheeling" if fc_key.lower() == "oxnard" else "oxnard"
                        other_fc_inventory = self.check_inventory_in_other_fc(
                            inventory_sku, inventory_df, other_fc
                        )

                        if other_fc_inventory > 0:
                            # Item is available in the other fulfillment center - log only once per SKU
                            if inventory_sku not in self._logged_inventory_issues:
                                logger.info(
                                    f"Item {inventory_sku} is out of stock in {fc_key} but available in {other_fc} ({other_fc_inventory} units)"
                                )
                                self._logged_inventory_issues.add(inventory_sku)
                            issue_message = f"Issue: Inventory in wrong location | Item: {inventory_sku} | Shopify SKU: {shopify_sku} | Location: {fc_key.capitalize()} | Available in {other_fc.capitalize()}: {other_fc_inventory} units"
                        else:
                            # Item is not available in either fulfillment center - log only once per SKU
                            if inventory_sku not in self._logged_inventory_issues:
                                logger.warning(
                                    f"Item {inventory_sku} is completely out of stock in both fulfillment centers"
                                )
                                self._logged_inventory_issues.add(inventory_sku)
                            issue_message = f"Issue: Out of stock | Item: {inventory_sku} | Shopify SKU: {shopify_sku} | Status: No inventory in any fulfillment center"
                    else:
                        # Partial shortage - some inventory is available but not enough
                        issue_message = f"Issue: Insufficient inventory | Item: {inventory_sku} | Shopify SKU: {shopify_sku} | Current: {current_balance} | Needed: {transaction_quantity} | Short by: {shortage} units"

                    # Find potential substitutions
                    substitution_options = self.find_substitution_options(
                        inventory_sku, shortage, inventory_df, sku_mappings, fc_key
                    )

                    # Create the issue message
                    issue_message = f"Issue: Insufficient inventory | Item: {inventory_sku} | Shopify SKU: {shopify_sku} | Current: {current_balance} | Needed: {transaction_quantity} | Short by: {shortage} units"

                    # Add substitution suggestions if available
                    if substitution_options:
                        issue_message += "\nSuggested substitutions:\n"
                        for i, option in enumerate(substitution_options, 1):
                            issue_message += f"{i}. {option['sku']} (weight: {option['weight']}, available: {option['available_qty']}, similarity: {option['similarity_score']}%)\n"

                    order_data["Issues"] = issue_message
                    
                    # Create a unique key for this inventory issue to prevent duplicate warnings
                    issue_key = f"inventory_shortage_{inventory_sku}_{shopify_sku}_{shortage}"
                    
                    # Only log if we haven't seen this exact issue before
                    if issue_key not in self._logged_inventory_issues:
                        logger.warning(
                            f"Insufficient inventory for {inventory_sku} (Shopify SKU: {shopify_sku}): current balance {current_balance}, needed {transaction_quantity}, short by {shortage} units"
                        )
                        # Add to tracked issues to prevent duplicate warnings
                        self._logged_inventory_issues.add(issue_key)

                    # Use what we have
                    transaction_quantity = current_balance

                ending_balance = max(0, current_balance - transaction_quantity)
                order_data["Ending Balance"] = ending_balance

                # Update the running balance for this SKU using composite key
                running_balances[inventory_composite_key] = ending_balance
                
                # Track this SKU as recalculated
                self._recalculated_skus.add(inventory_sku)
                
                # Log the balance update
                self._recalculation_log.append({
                    "timestamp": pd.Timestamp.now(),
                    "sku": inventory_sku,
                    "operation": "update_balance",
                    "value": ending_balance,
                    "starting_balance": current_balance,
                    "transaction_quantity": transaction_quantity,
                    "order_id": order_id,
                    "warehouse": fc_key
                })

                # Add the order to the output DataFrame
                output_df = pd.concat([output_df, pd.DataFrame([order_data])], ignore_index=True)

        # Ensure all columns are in the right order
        for col in output_columns:
            if col not in output_df.columns:
                output_df[col] = ""

        # Generate inventory summary
        inventory_summary = self.generate_inventory_summary(
            running_balances, inventory_df, sku_mappings
        )

        # Generate shortage summary - now returns both detailed and grouped views
        shortage_summary, grouped_shortage_summary = self.generate_shortage_summary(all_shortages)

        # Log information about selective recalculation if applicable
        if affected_skus:
            logger.info(f"Selective recalculation completed for {len(affected_skus)} SKUs")
            logger.info(
                f"Processed {len(processed_order_ids)} orders out of {len(orders_df)} total orders"
            )

            # Calculate percentage of orders processed
            percentage = (
                (len(processed_order_ids) / len(orders_df)) * 100 if len(orders_df) > 0 else 0
            )
            logger.info(f"Selective recalculation processed {percentage:.1f}% of orders")

        # Convert recalculation log to DataFrame for easier display
        recalculation_log_df = pd.DataFrame(self._recalculation_log) if self._recalculation_log else pd.DataFrame()
        
        # Return the processed orders, inventory summary, both shortage summaries, processed order IDs, and recalculation log
        # Generate inventory comparison if initial state was captured
        inventory_comparison = None
        if initial_inventory_state:
            inventory_comparison = self.generate_inventory_comparison(
                initial_inventory_state, running_balances, inventory_df, sku_mappings
            )
            
        return {
            "orders": output_df[
                output_columns
            ],  # Return only the specified columns in the correct order
            "inventory_summary": inventory_summary,
            "shortage_summary": shortage_summary,
            "grouped_shortage_summary": grouped_shortage_summary,
            "processed_order_ids": list(processed_order_ids) if affected_skus else None,
            "recalculation_log": recalculation_log_df,
            "recalculated_skus": list(self._recalculated_skus),
            "inventory_comparison": inventory_comparison
        }

    def generate_inventory_comparison(self, initial_inventory_state, current_balances, inventory_df, sku_mappings=None):
        """
        Generate a comparison of inventory levels before and after order processing.
        
        Args:
            initial_inventory_state (dict): Dictionary of SKU to initial balance before processing
            current_balances (dict): Dictionary of SKU to current balance after processing
            inventory_df (DataFrame): Original inventory dataframe
            sku_mappings (dict, optional): SKU mappings dictionary. Defaults to None.
            
        Returns:
            DataFrame: Comparison of inventory levels before and after processing
        """
        import pandas as pd
        
        # Find the SKU column with case-insensitive matching
        sku_column = None
        for col in inventory_df.columns:
            if col.lower() == 'sku':
                sku_column = col
                break
        
        if not sku_column:
            logger.warning("Could not find SKU column in inventory dataframe")
            return pd.DataFrame()
        
        # Create a list to hold comparison data
        comparison_data = []
        
        # Log the initial inventory state and current balances
        logger.info(f"Initial inventory state has {len(initial_inventory_state)} entries")
        logger.info(f"Current balances has {len(current_balances)} entries")
        
        # Handle new dictionary format with composite keys (sku|warehouse)
        normalized_initial_state = {}
        for key, data in initial_inventory_state.items():
            if isinstance(data, dict) and 'balance' in data:
                # New format: data contains balance and warehouse
                normalized_initial_state[key] = data['balance']
            else:
                # Old format: data is just the balance
                normalized_initial_state[key] = data
            
        # Current balances should already be in the correct format (composite_key -> balance)
        normalized_current_balances = current_balances.copy()
        
        # Process all inventory items from the inventory dataframe to ensure we capture all warehouses
        all_inventory_items = set()
        if sku_column:
            for _, row in inventory_df.iterrows():
                sku = row[sku_column]
                if "WarehouseName" in row:
                    warehouse_name = str(row["WarehouseName"])
                    normalized_warehouse = self.normalize_warehouse_name(warehouse_name, log_transformations=False)
                    composite_key = f"{sku}|{normalized_warehouse}"
                    all_inventory_items.add(composite_key)
                    
                    # If not in initial state, add it with inventory balance
                    if composite_key not in normalized_initial_state:
                        balance = 0
                        if "Balance" in row and pd.notna(row["Balance"]):
                            try:
                                balance = float(row["Balance"])
                            except (ValueError, TypeError):
                                pass
                        elif "AvailableQty" in row and pd.notna(row["AvailableQty"]):
                            try:
                                balance = float(row["AvailableQty"])
                            except (ValueError, TypeError):
                                pass
                        normalized_initial_state[composite_key] = balance
        
        # Combine all keys from initial state, current balances, and inventory items
        initial_skus = set(normalized_initial_state.keys())
        current_skus = set(normalized_current_balances.keys())
        all_keys = initial_skus.union(current_skus).union(all_inventory_items)
        
        logger.info(f"Processing comparison for {len(all_keys)} unique SKU-warehouse combinations")
        
        # Create reverse mapping from inventory SKU to Shopify SKU
        shopify_sku_map = {}
        if sku_mappings and isinstance(sku_mappings, dict):
            # Handle both old and new format of sku_mappings
            mappings_dict = sku_mappings.get("mappings", sku_mappings)
            
            # Flatten all mappings from different fulfillment centers
            for fc, fc_data in mappings_dict.items():
                if isinstance(fc_data, dict) and "all_skus" in fc_data:
                    # New format: fc -> {all_skus: {shopify_sku: {picklist_sku: inventory_sku}}}
                    all_skus = fc_data["all_skus"]
                    for shopify_sku, sku_data in all_skus.items():
                        if isinstance(sku_data, dict) and "picklist_sku" in sku_data:
                            inventory_sku = sku_data["picklist_sku"]
                            shopify_sku_map[inventory_sku] = shopify_sku
                elif isinstance(fc_data, dict):
                    # Legacy format: fc -> {shopify_sku: inventory_sku}
                    for shopify_sku, inventory_sku in fc_data.items():
                        # Only use inventory_sku as a key if it's hashable (string, int, etc.)
                        if isinstance(inventory_sku, (str, int, float, bool, tuple)):
                            shopify_sku_map[inventory_sku] = shopify_sku
                        elif isinstance(inventory_sku, dict) and "picklist_sku" in inventory_sku:
                            # Handle case where inventory_sku is a dict with picklist_sku
                            picklist_sku = inventory_sku["picklist_sku"]
                            shopify_sku_map[picklist_sku] = shopify_sku
                        else:
                            logger.warning(f"Skipping unhashable inventory SKU mapping: {type(inventory_sku)} for Shopify SKU: {shopify_sku}")
            
        
        # Get all bundle component SKUs if available
        bundle_components = set()
        if sku_mappings and isinstance(sku_mappings, dict) and "bundle_info" in sku_mappings:
            bundle_info = sku_mappings["bundle_info"]
            for fc, bundles in bundle_info.items():
                for _, components in bundles.items():
                    for component in components:
                        if "sku" in component:
                            bundle_components.add(component["sku"])
                        elif "component_sku" in component:
                            bundle_components.add(component["component_sku"])
        
        # Process each normalized key
        for composite_key in all_keys:
            # Extract SKU and warehouse from composite key
            if "|" in composite_key:
                sku, warehouse_display = composite_key.split("|", 1)
            else:
                # Legacy format: just SKU
                sku = composite_key
                warehouse_display = "Unknown"
                # Try to find warehouse from inventory data
                sku_rows = inventory_df[inventory_df[sku_column] == sku]
                if not sku_rows.empty and "WarehouseName" in sku_rows.columns:
                    warehouse_name = str(sku_rows.iloc[0]["WarehouseName"])
                    warehouse_display = self.normalize_warehouse_name(warehouse_name, log_transformations=False)
            
            # Get Shopify SKU if available
            shopify_sku = shopify_sku_map.get(sku, "")
            
            # Check if this SKU is used in bundles
            is_used_in_bundle = sku in bundle_components
            
            # Get initial balance from normalized dictionary using composite key
            initial_balance = normalized_initial_state.get(composite_key, 0)
            
            # If still not found, look for this SKU+warehouse combination in the inventory dataframe
            if initial_balance == 0:
                sku_rows = inventory_df[inventory_df[sku_column] == sku]
                if not sku_rows.empty:
                    # Filter by warehouse if we have warehouse info
                    if warehouse_display != "Unknown" and "WarehouseName" in sku_rows.columns:
                        warehouse_rows = sku_rows[sku_rows["WarehouseName"].apply(
                            lambda x: self.normalize_warehouse_name(str(x), log_transformations=False) == warehouse_display
                        )]
                        if not warehouse_rows.empty:
                            row = warehouse_rows.iloc[0]
                        else:
                            row = sku_rows.iloc[0]  # Fallback to first match
                    else:
                        row = sku_rows.iloc[0]  # Take first match
                    
                    # Use Balance column if available, otherwise use AvailableQty
                    if "Balance" in row and pd.notna(row["Balance"]):
                        try:
                            initial_balance = float(row["Balance"])
                        except (ValueError, TypeError):
                            pass
                    elif "AvailableQty" in row and pd.notna(row["AvailableQty"]):
                        try:
                            initial_balance = float(row["AvailableQty"])
                        except (ValueError, TypeError):
                            pass
            
            # Get current balance from normalized dictionary using composite key
            current_balance = normalized_current_balances.get(composite_key, initial_balance)
            
            # Calculate difference
            difference = current_balance - initial_balance
            
            # Only include items with actual changes or significant balances
            # This makes the comparison report more useful by filtering out unchanged items
            if abs(difference) > 0.001 or initial_balance > 0 or current_balance > 0:
                comparison_data.append({
                    "SKU": sku,
                    "Shopify SKU": shopify_sku,
                    "Warehouse": warehouse_display,
                    "Initial Balance": initial_balance,
                    "Current Balance": current_balance,
                    "Difference": difference,
                    "Is Used In Bundle": is_used_in_bundle
                })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by absolute difference (descending) to show items with biggest changes first
        if not comparison_df.empty:
            comparison_df['abs_difference'] = comparison_df['Difference'].abs()
            comparison_df = comparison_df.sort_values(
                by=["abs_difference", "Is Used In Bundle"], 
                ascending=[False, False]
            ).drop('abs_difference', axis=1)
        
        return comparison_df

    def calculate_processing_stats(self, processed_orders, inventory_summary, shortage_summary):
        """
        Calculate comprehensive processing statistics for decision making.
        
        Args:
            processed_orders: DataFrame of processed orders
            inventory_summary: DataFrame of inventory summary
            shortage_summary: DataFrame of shortage summary
            
        Returns:
            dict: Dictionary containing various processing statistics
        """
        stats = {}
        
        if processed_orders is not None and not processed_orders.empty:
            # Order statistics
            stats['total_orders'] = len(processed_orders['ordernumber'].unique()) if 'ordernumber' in processed_orders.columns else 0
            stats['total_line_items'] = len(processed_orders)
            stats['unique_skus'] = len(processed_orders['sku'].unique()) if 'sku' in processed_orders.columns else 0
            
            # Fulfillment center distribution
            if 'Fulfillment Center' in processed_orders.columns:
                fc_dist = processed_orders['Fulfillment Center'].value_counts().to_dict()
                stats['fulfillment_center_distribution'] = fc_dist
                stats['primary_fulfillment_center'] = max(fc_dist.items(), key=lambda x: x[1])[0] if fc_dist else "Unknown"
            
            # Issues and shortages
            if 'Issues' in processed_orders.columns:
                stats['items_with_issues'] = len(processed_orders[processed_orders['Issues'] != ""])
                stats['issue_rate'] = round((stats['items_with_issues'] / len(processed_orders)) * 100, 2)
            
            # Quantity analysis
            if 'Transaction Quantity' in processed_orders.columns:
                quantities = pd.to_numeric(processed_orders['Transaction Quantity'], errors='coerce')
                stats['total_quantity_processed'] = quantities.sum()
                stats['avg_quantity_per_item'] = round(quantities.mean(), 2)
                stats['max_quantity_item'] = quantities.max()
        
        if shortage_summary is not None and not shortage_summary.empty:
            stats['total_shortages'] = len(shortage_summary)
            stats['unique_shortage_skus'] = len(shortage_summary['component_sku'].unique()) if 'component_sku' in shortage_summary.columns else 0
            
            if 'shortage_qty' in shortage_summary.columns:
                shortage_qty = pd.to_numeric(shortage_summary['shortage_qty'], errors='coerce')
                stats['total_shortage_quantity'] = shortage_qty.sum()
            
            if 'fulfillment_center' in shortage_summary.columns:
                shortage_by_fc = shortage_summary['fulfillment_center'].value_counts().to_dict()
                stats['shortages_by_fulfillment_center'] = shortage_by_fc
        
        if inventory_summary is not None and not inventory_summary.empty:
            stats['total_inventory_items'] = len(inventory_summary)
            
            if 'Current Balance' in inventory_summary.columns:
                balances = pd.to_numeric(inventory_summary['Current Balance'], errors='coerce')
                stats['total_inventory_balance'] = balances.sum()
                stats['zero_balance_items'] = len(inventory_summary[balances == 0])
                stats['low_balance_items'] = len(inventory_summary[balances <= 10])
        
        # Calculate processing timestamp
        stats['processing_timestamp'] = pd.Timestamp.now().isoformat()
        
        return stats

    def calculate_warehouse_performance(self, processed_orders, inventory_summary):
        """
        Calculate warehouse performance metrics for decision making.
        
        Args:
            processed_orders: DataFrame of processed orders
            inventory_summary: DataFrame of inventory summary
            
        Returns:
            dict: Dictionary containing warehouse performance metrics
        """
        performance = {}
        
        if processed_orders is not None and not processed_orders.empty and 'Fulfillment Center' in processed_orders.columns:
            warehouses = processed_orders['Fulfillment Center'].unique()
            
            for warehouse in warehouses:
                warehouse_orders = processed_orders[processed_orders['Fulfillment Center'] == warehouse]
                warehouse_perf = {}
                
                # Order volume metrics
                warehouse_perf['total_orders'] = len(warehouse_orders['ordernumber'].unique()) if 'ordernumber' in warehouse_orders.columns else 0
                warehouse_perf['total_line_items'] = len(warehouse_orders)
                warehouse_perf['unique_skus'] = len(warehouse_orders['sku'].unique()) if 'sku' in warehouse_orders.columns else 0
                
                # Issue metrics
                if 'Issues' in warehouse_orders.columns:
                    items_with_issues = len(warehouse_orders[warehouse_orders['Issues'] != ""])
                    warehouse_perf['items_with_issues'] = items_with_issues
                    warehouse_perf['issue_rate'] = round((items_with_issues / len(warehouse_orders)) * 100, 2) if len(warehouse_orders) > 0 else 0
                
                # Quantity metrics
                if 'Transaction Quantity' in warehouse_orders.columns:
                    quantities = pd.to_numeric(warehouse_orders['Transaction Quantity'], errors='coerce')
                    warehouse_perf['total_quantity'] = quantities.sum()
                    warehouse_perf['avg_quantity_per_item'] = round(quantities.mean(), 2)
                
                # Bundle performance
                if 'Bundle' in warehouse_orders.columns:
                    bundle_items = len(warehouse_orders[warehouse_orders['Bundle'] != ""])
                    warehouse_perf['bundle_items'] = bundle_items
                    warehouse_perf['bundle_rate'] = round((bundle_items / len(warehouse_orders)) * 100, 2) if len(warehouse_orders) > 0 else 0
                
                performance[warehouse] = warehouse_perf
        
        # Add inventory distribution by warehouse
        if inventory_summary is not None and not inventory_summary.empty and 'Warehouse' in inventory_summary.columns:
            warehouse_inventory = inventory_summary.groupby('Warehouse').agg({
                'Current Balance': 'sum',
                'Inventory SKU': 'count'
            }).to_dict('index')
            
            for warehouse, inv_data in warehouse_inventory.items():
                if warehouse in performance:
                    performance[warehouse]['inventory_balance'] = inv_data.get('Current Balance', 0)
                    performance[warehouse]['inventory_sku_count'] = inv_data.get('Inventory SKU', 0)
                else:
                    performance[warehouse] = {
                        'inventory_balance': inv_data.get('Current Balance', 0),
                        'inventory_sku_count': inv_data.get('Inventory SKU', 0)
                    }
        
        return performance

    def _map_sku_to_inventory(
        self,
        sku,
        inventory_df,
        sku_mapping=None,
        bundle_info=None,
        fc_key=None,
        running_balances=None,
    ):
        """
        Map order SKU to inventory SKU using various matching strategies.
        Prioritizes exact SKU mappings from the JSON file before attempting any fuzzy matching.

        Args:
            sku: SKU from order (already mapped from Shopify to warehouse SKU if mapping exists)
            inventory_df: Inventory dataframe
            sku_mapping: Optional dictionary mapping order SKUs to inventory SKUs
            bundle_info: Optional dictionary containing bundle component information
            fc_key: Fulfillment center key (Wheeling or Oxnard)
        """
        # Initialize issue tracking
        issue = None
        # Check if inventory_df is valid
        if inventory_df is None or inventory_df.empty:
            logger.warning("Warning: Empty inventory DataFrame")
            return "", 0, issue

        # Ensure sku is a string and strip whitespace
        sku = str(sku).strip() if sku is not None else ""

        # Check if required columns exist with case-insensitive matching
        sku_column = None
        for col in inventory_df.columns:
            if col.lower() == "sku":
                sku_column = col
            elif col.lower() == "balance":
                pass

        if sku_column is None:
            logger.warning("Warning: 'Sku' column not found in inventory DataFrame")
            return "", 0, None

        # Debug output
        logger.info(f"Looking for SKU '{sku}' in inventory")
        logger.info(
            f"Inventory has {len(inventory_df)} rows with {len(inventory_df[sku_column].unique())} unique SKUs"
        )

        # Ensure all inventory SKUs are strings and stripped
        inventory_df[sku_column] = inventory_df[sku_column].astype(str).str.strip()

        # Function to get balance for a given SKU
        def get_balance_for_sku(sku_value):
            # First check if we have a running balance for this SKU
            if running_balances is not None and sku_value in running_balances:
                return running_balances[sku_value]

            # If no running balance, check inventory
            # Look for the SKU in Wheeling warehouse first
            wheeling_rows = inventory_df[
                (inventory_df[sku_column] == sku_value)
                & (
                    inventory_df["WarehouseName"]
                    .str.lower()
                    .str.contains("wheeling|il-wheeling", na=False)
                )
            ]

            # If found in Wheeling, check Balance
            if not wheeling_rows.empty:
                if "Balance" in wheeling_rows.columns:
                    try:
                        balance_value = wheeling_rows.iloc[0]["Balance"]
                        if isinstance(balance_value, str):
                            balance_value = balance_value.replace(",", "")
                        balance = float(balance_value) if pd.notna(balance_value) else 0
                        logger.info(f"Using Balance column for {sku_value} in Wheeling: {balance}")
                        return balance
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Could not convert Balance value for {sku_value}: {balance_value} - Error: {e}"
                        )

            # If not found in Wheeling or no Balance, try Oxnard (Moorpark) warehouse
            oxnard_rows = inventory_df[
                (inventory_df[sku_column] == sku_value)
                & (
                    inventory_df["WarehouseName"]
                    .str.lower()
                    .str.contains("oxnard|moorpark", na=False)
                )
            ]
            if not oxnard_rows.empty:
                if "Balance" in oxnard_rows.columns:
                    try:
                        balance_value = oxnard_rows.iloc[0]["Balance"]
                        if isinstance(balance_value, str):
                            balance_value = balance_value.replace(",", "")
                        balance = float(balance_value) if pd.notna(balance_value) else 0
                        logger.info(
                            f"Using Balance column for {sku_value} in other warehouse: {balance}"
                        )
                        return balance
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Could not convert Balance value for {sku_value}: {balance_value} - Error: {e}"
                        )

            # If we get here, no valid Balance found
            logger.info(f"No valid Balance found for {sku_value}, returning 0")
            return 0

        # STEP 1: If we have a mapping from the JSON file, use it as highest priority
        if sku_mapping and sku in sku_mapping:
            mapped_sku = sku_mapping[sku]
            logger.info(f"Using mapped SKU from JSON: {sku} -> {mapped_sku}")

            # Check if the mapped SKU exists in inventory
            matching_rows = inventory_df[inventory_df[sku_column] == mapped_sku]
            if not matching_rows.empty:
                matched_balance = get_balance_for_sku(mapped_sku)
                logger.info(
                    f"Found mapped SKU in inventory: {mapped_sku} with balance {matched_balance}"
                )
                # Check if there's enough inventory
                if matched_balance <= 0:
                    issue = f"No inventory available for {mapped_sku}"
                    logger.warning(issue)
                return mapped_sku, matched_balance, issue
            else:
                # Try case-insensitive match with mapped SKU
                mapped_sku_lower = mapped_sku.lower()
                for inv_sku in inventory_df[sku_column].unique():
                    if inv_sku.lower() == mapped_sku_lower:
                        matched_balance = get_balance_for_sku(inv_sku)
                        logger.info(
                            f"Found case-insensitive match for mapped SKU: '{mapped_sku}' -> '{inv_sku}' with balance {matched_balance}"
                        )
                        return inv_sku, matched_balance, issue

                logger.warning(
                    f"Warning: Mapped SKU {mapped_sku} not found in inventory (original SKU: {sku})"
                )
                issue = f"Mapped SKU {mapped_sku} not found in inventory (original SKU: {sku})"
                return mapped_sku, 0, issue  # Return it anyway, as it's the best match we have

        # STEP 2: If no mapping exists, try direct matches
        # Try exact match
        matching_rows = inventory_df[inventory_df[sku_column] == sku]

        if not matching_rows.empty:
            # Return the first matching SKU and balance
            inv_sku = matching_rows.iloc[0][sku_column]
            matched_balance = get_balance_for_sku(inv_sku)
            logger.info(f"Found exact match: '{sku}' with balance {matched_balance}")
            # Check if there's enough inventory
            if matched_balance <= 0:
                issue = f"No inventory available for {inv_sku}"
                logger.warning(issue)
            return inv_sku, matched_balance, issue

        # If no exact match, try removing 'f.' prefix if present
        if sku.startswith("f."):
            sku_no_prefix = sku[2:]
            matching_rows = inventory_df[inventory_df[sku_column] == sku_no_prefix]
            if not matching_rows.empty:
                inv_sku = matching_rows.iloc[0][sku_column]
                matched_balance = get_balance_for_sku(inv_sku)
                logger.info(
                    f"Found match after removing 'f.' prefix: '{sku}' -> '{inv_sku}' with balance {matched_balance}"
                )
                # Check if there's enough inventory
                if matched_balance <= 0:
                    issue = f"No inventory available for {inv_sku}"
                    logger.warning(issue)
                return inv_sku, matched_balance, issue

        # Try case-insensitive match
        sku_lower = sku.lower()
        for inv_sku in inventory_df[sku_column].unique():
            if inv_sku.lower() == sku_lower:
                matched_balance = get_balance_for_sku(inv_sku)
                logger.info(
                    f"Found case-insensitive match: '{sku}' -> '{inv_sku}' with balance {matched_balance}"
                )
                # Check if there's enough inventory
                if matched_balance <= 0:
                    issue = f"No inventory available for {inv_sku}"
                    logger.warning(issue)
                return inv_sku, matched_balance, issue

        # If we get here, we couldn't find a match with any of the exact matching methods
        logger.warning(f"No exact match found for SKU: '{sku}'. Fuzzy matching is disabled.")
        issue = f"No exact match found for SKU: '{sku}'. Please add this SKU to the mapping file."
        return "", 0, issue

    def get_optimal_fulfillment_center(
        self, zip_code, shipping_zones_df=None, priority=None, inventory_df=None, sku=None
    ):
        """
        Determine the optimal fulfillment center based on shipping zones, inventory, and priority.

        Args:
            zip_code: Customer ZIP code
            shipping_zones_df: DataFrame containing shipping zone data or path to CSV file
            priority: Order priority (P1, P*, etc.)
            inventory_df: DataFrame containing inventory data
            sku: SKU to check inventory for

        Returns:
            dict: Dictionary containing fulfillment center information
        """
        # Import shipping zones utilities
        from constants.shipping_zones import (
            DEFAULT_SHIPPING_ZONES_PATH,
            estimate_zone_by_geography,
            get_zone_by_zip,
            load_shipping_zones,
        )

        # Initialize fulfillment center data
        fulfillment_centers = {
            "moorpark": {"zone": None, "estimated_delivery_days": None},
            "wheeling": {"zone": None, "estimated_delivery_days": None},
        }

        # Load shipping zones if needed
        if shipping_zones_df is None:
            try:
                shipping_zones_df = load_shipping_zones(DEFAULT_SHIPPING_ZONES_PATH)
            except Exception as e:
                logger.error(f"Error loading shipping zones from default location: {e}")
                shipping_zones_df = pd.DataFrame()
        elif isinstance(shipping_zones_df, str):
            try:
                shipping_zones_df = load_shipping_zones(shipping_zones_df)
            except Exception as e:
                logger.error(f"Error loading shipping zones from file: {e}")
                shipping_zones_df = pd.DataFrame()
        elif isinstance(shipping_zones_df, list):
            # If shipping_zones_df is a list (likely rules), we don't use it for shipping zones
            shipping_zones_df = pd.DataFrame()

        # Get shipping zones for each fulfillment center
        if not shipping_zones_df.empty and zip_code:
            # Get zone for Moorpark
            moorpark_zone = get_zone_by_zip(shipping_zones_df, zip_code, "moorpark")
            if moorpark_zone is not None:
                fulfillment_centers["moorpark"]["zone"] = moorpark_zone
                # Estimate delivery days based on zone
                fulfillment_centers["moorpark"]["estimated_delivery_days"] = min(moorpark_zone, 8)

            # Get zone for Wheeling
            wheeling_zone = get_zone_by_zip(shipping_zones_df, zip_code, "wheeling")
            if wheeling_zone is not None:
                fulfillment_centers["wheeling"]["zone"] = wheeling_zone
                # Estimate delivery days based on zone
                fulfillment_centers["wheeling"]["estimated_delivery_days"] = min(wheeling_zone, 8)

        # If zones are not available, estimate them based on geography
        if zip_code:
            if fulfillment_centers["moorpark"]["zone"] is None:
                moorpark_zone = estimate_zone_by_geography(zip_code, "moorpark")
                fulfillment_centers["moorpark"]["zone"] = moorpark_zone
                fulfillment_centers["moorpark"]["estimated_delivery_days"] = min(moorpark_zone, 8)

            if fulfillment_centers["wheeling"]["zone"] is None:
                wheeling_zone = estimate_zone_by_geography(zip_code, "wheeling")
                fulfillment_centers["wheeling"]["zone"] = wheeling_zone
                fulfillment_centers["wheeling"]["estimated_delivery_days"] = min(wheeling_zone, 8)

        # Check inventory availability for each fulfillment center with detailed information
        inventory_availability = {
            "moorpark": {
                "available": False,
                "qty": 0,
                "starting_balance": 0,
                "transaction_qty": 0,
                "ending_balance": 0,
            },
            "wheeling": {
                "available": False,
                "qty": 0,
                "starting_balance": 0,
                "transaction_qty": 0,
                "ending_balance": 0,
            },
        }

        if inventory_df is not None and sku:
            # Map warehouse names to fulfillment centers (case-insensitive matching)
            warehouse_mapping = {
                "moorpark": ["moorpark", "oxnard", "ca-moorpark", "ca-oxnard", "93021", "93030"],
                "wheeling": ["wheeling", "illinois", "il-wheeling", "60090"],
            }

            # Check inventory for each fulfillment center
            for fc, warehouses in warehouse_mapping.items():
                # Create a regex pattern to match any of the warehouses for this fulfillment center
                "|".join(warehouses)

                # Find the Sku column with case-insensitive matching if not already done
                if "Sku" not in inventory_df.columns:
                    sku_column = None
                    for col in inventory_df.columns:
                        if col.lower() == "sku":
                            sku_column = col
                            inventory_df["Sku"] = inventory_df[sku_column]
                            break

                # Find the WarehouseName column with case-insensitive matching
                warehouse_column = "WarehouseName"
                if warehouse_column not in inventory_df.columns:
                    for col in inventory_df.columns:
                        if col.lower() == "warehousename":
                            warehouse_column = col
                            break

                # Filter inventory for this SKU and fulfillment center
                fc_inventory = pd.DataFrame()
                if "Sku" in inventory_df.columns and warehouse_column in inventory_df.columns:
                    # Convert warehouse names to lowercase for case-insensitive matching
                    fc_inventory = inventory_df[
                        (inventory_df["Sku"] == sku)
                        & (
                            inventory_df[warehouse_column]
                            .str.lower()
                            .str.contains("|".join(warehouses), regex=True)
                        )
                    ]

                # Check if there's available inventory
                if not fc_inventory.empty and "AvailableQty" in fc_inventory.columns:
                    total_available = fc_inventory["AvailableQty"].sum()

                    # Update inventory availability information
                    inventory_availability[fc]["available"] = total_available > 0
                    inventory_availability[fc]["qty"] = total_available
                    inventory_availability[fc]["starting_balance"] = total_available
                    inventory_availability[fc]["transaction_qty"] = -1  # Default to taking 1 unit
                    inventory_availability[fc]["ending_balance"] = total_available - 1

        # Determine optimal fulfillment center
        optimal_center = "moorpark"  # Default to Moorpark

        # Handle priority orders
        if priority == "P1" or priority == "P*":
            # For priority orders, use the closest fulfillment center but respect inventory availability
            moorpark_zone = fulfillment_centers["moorpark"]["zone"]
            wheeling_zone = fulfillment_centers["wheeling"]["zone"]

            # If we have zone information for both centers
            if moorpark_zone is not None and wheeling_zone is not None:
                if moorpark_zone <= wheeling_zone:
                    # Moorpark is closer or equal distance
                    if (
                        inventory_availability["moorpark"]["available"]
                        or not inventory_availability["wheeling"]["available"]
                    ):
                        optimal_center = "moorpark"
                    else:
                        # Only Wheeling has inventory
                        optimal_center = "wheeling"
                else:
                    # Wheeling is closer
                    if (
                        inventory_availability["wheeling"]["available"]
                        or not inventory_availability["moorpark"]["available"]
                    ):
                        optimal_center = "wheeling"
                    else:
                        # Only Moorpark has inventory
                        optimal_center = "moorpark"
            # If we only have zone information for one center
            elif moorpark_zone is not None:
                optimal_center = "moorpark"
            elif wheeling_zone is not None:
                optimal_center = "wheeling"
            # If no zone info, prefer Moorpark by default
            else:
                # Default to Moorpark unless we have a specific reason to use Wheeling
                optimal_center = "moorpark"

        # Determine inventory status based on availability
        inventory_status = "Out of Stock"
        if inventory_availability[optimal_center]["available"]:
            inventory_status = "Good"
            if inventory_availability[optimal_center]["qty"] < 10:
                inventory_status = "Low Stock"
            if inventory_availability[optimal_center]["qty"] < 5:
                inventory_status = "Critical Stock"

        # Format the fulfillment center name with location code
        formatted_fc = f"{'CA' if optimal_center == 'moorpark' else 'IL'}-{optimal_center.title()}-{93021 if optimal_center == 'moorpark' else 60090}"

        # Set default values for zone and estimated delivery days
        zone = fulfillment_centers[optimal_center]["zone"]
        estimated_days = fulfillment_centers[optimal_center]["estimated_delivery_days"]

        # If no zone information is available, use geographic heuristic to estimate
        if zone is None:
            if zip_code and len(zip_code) >= 1:
                first_digit = int(zip_code[0])
                if optimal_center == "moorpark":
                    # For Moorpark: West Coast (closer) = lower zone, East Coast (further) = higher zone
                    zone = 1 if first_digit >= 9 else (3 if first_digit >= 7 else 5)
                else:  # wheeling
                    # For Wheeling: East Coast (closer) = lower zone, West Coast (further) = higher zone
                    zone = 1 if first_digit <= 3 else (3 if first_digit <= 5 else 5)
            else:
                # Default to middle zone if no ZIP code available
                zone = 4

        # Estimate delivery days based on zone if not already set
        if estimated_days is None and zone is not None:
            estimated_days = min(zone, 7)  # Estimate based on zone number

        # Get delivery service information based on ZIP code and fulfillment center
        delivery_service = self.get_delivery_service(zip_code, optimal_center)

        result = {
            "fulfillment_center": formatted_fc,
            "zone": zone,
            "estimated_delivery_days": estimated_days,
            "inventory_available": inventory_availability[optimal_center]["available"],
            "inventory_status": inventory_status,
            "available_qty": inventory_availability[optimal_center]["qty"],
            "moorpark_inventory": inventory_availability["moorpark"],
            "wheeling_inventory": inventory_availability["wheeling"],
            "carrier_name": delivery_service.get("carriername", "UPS"),
            "service_name": delivery_service.get("servicename", "Ground"),
            "delivery_days": delivery_service.get("days", estimated_days),
        }

        return result

    def get_delivery_service(self, zip_code, fulfillment_center):
        """
        Determine the appropriate delivery service based on customer ZIP code and fulfillment center.

        Args:
            zip_code: Customer ZIP code
            fulfillment_center: Selected fulfillment center ('moorpark' or 'wheeling')

        Returns:
            dict: Dictionary containing carrier name, service name, and estimated delivery days
        """
        # Default delivery service if no match is found - only used as fallback for errors
        default_service = {"carriername": None, "servicename": None, "days": None}

        if not zip_code or len(zip_code) < 3:
            logger.warning(f"Invalid ZIP code: {zip_code}")
            return default_service

        # Get the first 3 digits of the ZIP code for matching
        # Extract first 3 digits without leading zeros to match the format in the CSV
        zip_short = str(int(zip_code[:3]))

        # Determine origin ZIP code based on fulfillment center
        # Convert to integer to match the format in the CSV
        origin_zip = 93021 if fulfillment_center == "moorpark" else 60090

        logger.info(
            f"Looking for delivery service for ZIP: {zip_code} (prefix: {zip_short}), origin: {origin_zip}"
        )

        try:
            # Load delivery services data
            # First try to load from JSON file
            delivery_services_json_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "constants",
                "data",
                "delivery_services.json",
            )

            if os.path.exists(delivery_services_json_path):
                try:
                    # Load from JSON
                    with open(delivery_services_json_path, "r") as f:
                        delivery_services_data = json.load(f)

                    # Convert to DataFrame
                    if "delivery_services" in delivery_services_data:
                        delivery_df = pd.DataFrame(delivery_services_data["delivery_services"])
                        # Rename columns to match expected format
                        delivery_df = delivery_df.rename(
                            columns={
                                "destination_zip_short": "destination zip short",
                                "carrier_name": "carriername",
                                "service_name": "servicename",
                                "days": "days",
                            }
                        )
                        logger.info(f"Loaded {len(delivery_df)} delivery service entries from JSON")
                    else:
                        logger.warning(f"Invalid format in delivery services JSON file")
                        return default_service
                except Exception as e:
                    logger.warning(f"Error loading delivery services from JSON: {e}")
                    # Fall back to CSV
                    delivery_services_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "constants",
                        "delivery_services.csv",
                    )
                    if not os.path.exists(delivery_services_path):
                        logger.warning(
                            f"Warning: Delivery services file not found at {delivery_services_path}"
                        )
                        return default_service
                    delivery_df = pd.read_csv(delivery_services_path, sep=";", encoding="utf-8")
                    logger.info(f"Loaded {len(delivery_df)} delivery service entries from CSV")
            else:
                # Fall back to CSV
                delivery_services_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), "constants", "delivery_services.csv"
                )
                if not os.path.exists(delivery_services_path):
                    logger.warning(
                        f"Warning: Delivery services file not found at {delivery_services_path}"
                    )
                    return default_service
                delivery_df = pd.read_csv(delivery_services_path, sep=";", encoding="utf-8")
                logger.info(f"Loaded {len(delivery_df)} delivery service entries from CSV")

            # Ensure destination zip short column is string type and remove leading zeros
            delivery_df["destination zip short"] = delivery_df["destination zip short"].astype(str)

            # Ensure origin column is integer type
            delivery_df["origin"] = delivery_df["origin"].astype(int)

            # Log sample of delivery services for debugging
            sample_services = delivery_df.head(3).to_dict("records")
            logger.info(f"Sample delivery services: {sample_services}")

            # Log unique ZIP prefixes in the delivery services data
            unique_zips = delivery_df["destination zip short"].unique()[:10]
            logger.info(f"Sample ZIP prefixes in delivery services: {unique_zips}")

            # Log unique origins in the delivery services data
            unique_origins = delivery_df["origin"].unique()
            logger.info(f"Origins in delivery services: {unique_origins}")

            # Check if the ZIP prefix exists in the data
            zip_exists = zip_short in delivery_df["destination zip short"].values
            logger.info(f"ZIP prefix {zip_short} exists in delivery services: {zip_exists}")

            # Check if the origin exists in the data
            origin_exists = origin_zip in delivery_df["origin"].values
            logger.info(f"Origin {origin_zip} exists in delivery services: {origin_exists}")

            # Filter by ZIP code and origin
            matching_services = delivery_df[
                (delivery_df["destination zip short"] == zip_short)
                & (delivery_df["origin"] == origin_zip)
            ]

            logger.info(
                f"Found {len(matching_services)} matching services for ZIP {zip_short} and origin {origin_zip}"
            )

            if matching_services.empty:
                # If no exact match, try to find a service for a similar ZIP code range
                first_digit = zip_short[0]
                similar_services = delivery_df[
                    (delivery_df["destination zip short"].str.startswith(first_digit))
                    & (delivery_df["origin"] == origin_zip)
                ]

                logger.info(
                    f"Found {len(similar_services)} similar services starting with {first_digit} for origin {origin_zip}"
                )

                if not similar_services.empty:
                    # Use the first matching service
                    service = similar_services.iloc[0].to_dict()
                    return {
                        "carriername": service.get("carriername"),
                        "servicename": service.get("servicename"),
                        "days": service.get("days"),
                    }
                else:
                    # No matching service found
                    logger.warning(
                        f"No matching or similar delivery service found for ZIP {zip_code} (prefix {zip_short}) and origin {origin_zip}"
                    )
                    return default_service
            else:
                # Use the first matching service
                service = matching_services.iloc[0].to_dict()
                return {
                    "carriername": service.get("carriername"),
                    "servicename": service.get("servicename"),
                    "days": service.get("days"),
                }

        except Exception as e:
            logger.error(f"Error loading delivery services: {str(e)}")
            return default_service


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_processor = DataProcessor()
    
    # Import datetime for timestamp generation
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_to_file = f"inventory_comparison_{timestamp}.csv"
    shipping_zones_df = load_shipping_zones("docs/shipping_zones.csv")
    orders_df = data_processor.load_orders("docs/orders.csv")
    inventory_df = data_processor.load_inventory("docs/inventory.csv")
    sku_mappings = data_processor.load_sku_mappings()

    if not orders_df.empty:
        # Process orders and get orders, inventory summary, and shortage summary
        result = data_processor.process_orders(
            orders_df, inventory_df, shipping_zones_df, sku_mappings
        )
        processed_orders = result["orders"]
        inventory_summary = result["inventory_summary"]
        shortage_summary = result["shortage_summary"]

        # Format numeric columns to 3 decimal places, but only if there are non-zero digits after the decimal point
        numeric_columns = [
            "actualqty",
            "Total Pick Weight",
            "quantity",
            "Starting Balance",
            "Transaction Quantity",
            "Ending Balance",
        ]
        for col in numeric_columns:
            if col in processed_orders.columns:
                processed_orders[col] = processed_orders[col].apply(
                    lambda x: str(int(float(x)))
                    if pd.notnull(x) and str(x).strip() != "" and float(x) == int(float(x))
                    else (
                        "{:.3f}".format(float(x)).rstrip("0").rstrip(".")
                        if pd.notnull(x) and str(x).strip() != ""
                        else x
                    )
                )

        # Also format numeric columns in inventory summary
        if "Current Balance" in inventory_summary.columns:
            # First ensure all values are numeric
            inventory_summary["Current Balance"] = pd.to_numeric(
                inventory_summary["Current Balance"], errors="coerce"
            ).fillna(0)

            # Format the values for display - preserve commas for thousands
            inventory_summary["Current Balance"] = inventory_summary["Current Balance"].apply(
                lambda x: f"{int(x):,}"
                if pd.notnull(x) and x == int(x)
                else (f"{x:,.3f}".rstrip("0").rstrip(".") if pd.notnull(x) else "0")
            )

            # Log sample of formatted Current Balance values
            logger.info(
                f"Sample formatted Current Balance values: {inventory_summary['Current Balance'].head(5).tolist()}"
            )

        # Save processed orders to the output file
        orders_output_path = "output.csv"
        processed_orders.to_csv(orders_output_path, index=False)
        logger.info(f"Processed {len(processed_orders)} orders and saved to {orders_output_path}")

        # Save inventory summary to a separate file
        inventory_output_path = "inventory_summary.csv"
        inventory_summary.to_csv(inventory_output_path, index=False)
        logger.info(
            f"Generated inventory summary with {len(inventory_summary)} SKUs and saved to {inventory_output_path}"
        )

        # Save shortage summary to a separate file if there are any shortages
        if not shortage_summary.empty:
            shortage_output_path = "shortage_summary.csv"
            shortage_summary.to_csv(shortage_output_path, index=False)
            logger.info(
                f"Generated shortage summary with {len(shortage_summary)} items and saved to {shortage_output_path}"
            )
            print(
                f"\nWARNING: {len(shortage_summary)} inventory shortages detected! See {shortage_output_path} for details."
            )
            
        # Save inventory comparison to a file if it exists
        inventory_comparison = result.get("inventory_comparison")
        if inventory_comparison is not None and not inventory_comparison.empty:
            inventory_comparison.to_csv(save_to_file, index=False)
            logger.info(f"Generated inventory comparison with {len(inventory_comparison)} items and saved to {save_to_file}")
