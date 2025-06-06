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
from utils.data_parser import DataParser
from utils.airtable_handler import AirtableHandler

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data loading, processing, and transformation for the fulfillment application.
    """

    def __init__(self, use_airtable: bool = True) -> None:
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
        self._data_parser = DataParser()
        self._airtable_handler = AirtableHandler()
        # SKU mappings cache
        self.sku_mappings = None

    def normalize_warehouse_name(self, warehouse_name, log_transformations=True) -> str:
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

    def load_orders(self, file) -> pd.DataFrame:
        """
        Load and preprocess orders CSV file

        Args:
            file: Uploaded CSV file object

        Returns:
            pandas.DataFrame: Processed orders dataframe
        """
        df_orders = self._data_parser.parse_orders(file)
        return df_orders

    def load_sku_mappings(self):
        """
        Load SKU mappings from Airtable for Oxnard and Wheeling fulfillment centers.

        Handles both individual SKUs (using picklist_sku) and bundles (using component_sku).
        For bundles, all components and their quantities are stored in a structured format.

        Returns:
            dict: Dictionary containing:
                - SKU mappings by fulfillment center
                - Bundle information by fulfillment center
        """
        try:
            # Initialize empty mappings and bundle information dictionaries as fallback
            mappings = {"Oxnard": {}, "Wheeling": {}}
            bundle_info = {"Oxnard": {}, "Wheeling": {}}
            
            # Check if Airtable is enabled and schema manager exists
            if not self.use_airtable or not self.schema_manager:
                logger.warning("Airtable integration is disabled or schema manager not initialized.")
                result = {"mappings": mappings, "bundle_info": bundle_info}
                self.sku_mappings = result  # Store the mappings in the instance
                return result
        
            # Load SKU mappings from Airtable
            # Initialize merged results
            result = {"mappings": {"Oxnard": {}, "Wheeling": {}}, "bundle_info": {"Oxnard": {}, "Wheeling": {}}}
            
            # Load mappings for each center
            for center in ["Oxnard", "Wheeling"]:
                try:
                    center_result = self._airtable_handler.load_sku_mappings_from_airtable(center)
                    if center_result:
                        # If the center has mappings, update our result
                        if "mappings" in center_result and center in center_result["mappings"]:
                            result["mappings"][center] = center_result["mappings"][center]
                        if "bundle_info" in center_result and center in center_result["bundle_info"]:
                            result["bundle_info"][center] = center_result["bundle_info"][center]
                    logger.info(f"Loaded {len(result['mappings'][center])} mappings for {center}")
                except Exception as e:
                    logger.warning(f"Error loading mappings for {center}: {e}")
        
            # Use the merged results
            airtable_result = result
            
            if airtable_result and any(len(airtable_result["mappings"][center]) > 0 for center in airtable_result["mappings"]):
                logger.info("Successfully loaded SKU mappings from Airtable")
                self.sku_mappings = airtable_result  # Store the mappings in the instance
                return airtable_result
            else:
                logger.warning("No SKU mappings found in Airtable")
                result = {"mappings": mappings, "bundle_info": bundle_info}
                self.sku_mappings = result  # Store the mappings in the instance
                return result
            
        except Exception as e:
            logger.error(f"Error loading SKU mappings from Airtable: {str(e)}")
            result = {"mappings": mappings, "bundle_info": bundle_info}
            self.sku_mappings = result  # Store the mappings in the instance
            return result
    
    def load_inventory(self, file):
        """
        Load and preprocess inventory CSV file

        Args:
            file: Uploaded CSV file object or file path

        Returns:
            pandas.DataFrame: Processed inventory dataframe with normalized columns
        """
        try:
            # Use DataParser to parse the inventory file
            inventory_df = self._data_parser.parse_inventory(file)
            
            # Check if the DataFrame is empty
            if inventory_df.empty:
                logger.warning("Empty inventory DataFrame after initial parsing")
                return pd.DataFrame(columns=['WarehouseName', 'Sku', 'Name', 'Type', 'Balance', 'Status'])
            
            # Ensure the 'Sku' column exists
            if 'Sku' not in inventory_df.columns:
                # Try to find an alternative sku column
                sku_col = next((col for col in ['SKU', 'sku', 'Item #', 'Item#', 'Item Number', 'ItemNumber']
                               if col in inventory_df.columns), None)
                
                if sku_col:
                    # Rename the column to 'Sku'
                    inventory_df = inventory_df.rename(columns={sku_col: 'Sku'})
                    logger.info(f"Renamed column '{sku_col}' to 'Sku'")
                else:
                    logger.warning("'Sku' column not found in inventory DataFrame and no alternative column found")
                    # Create an empty 'Sku' column
                    inventory_df['Sku'] = ''
                    
            # Ensure other required columns exist
            required_columns = ['WarehouseName', 'Name', 'Balance', 'Status', 'Type']
            for col in required_columns:
                if col not in inventory_df.columns:
                    logger.warning(f"Required column '{col}' not found in inventory DataFrame")
                    inventory_df[col] = '' if col != 'Balance' else 0
                    
            # Convert Balance column to numeric
            if 'Balance' in inventory_df.columns:
                inventory_df['Balance'] = pd.to_numeric(inventory_df['Balance'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # Add tracking key for SKU and warehouse if not already present
            if 'tracking_key' not in inventory_df.columns:
                inventory_df['tracking_key'] = inventory_df['WarehouseName'] + '|' + inventory_df['Sku']
                logger.info(f"Inventory has {len(inventory_df)} rows with {inventory_df['tracking_key'].nunique()} unique SKU-warehouse combinations")
            
            # We do NOT group or sum balances as per requirement - keep all rows as is
            # This preserves the individual Balance values for each SKU and warehouse combination
            
            return inventory_df
            
        except Exception as e:
            logger.error(f"Error processing inventory file: {str(e)}")
            return pd.DataFrame(columns=['WarehouseName', 'Sku', 'Name', 'Type', 'Balance', 'Status'])

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
            if col.lower() == "balance":
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
        # Also track which items are bundle components
        inv_to_shopify = {}
        bundle_component_skus = set()
        
        if sku_mappings:
            for fc_key in sku_mappings:
                # First, collect all bundle component SKUs
                if "bundles" in sku_mappings[fc_key]:
                    for bundle_sku, components in sku_mappings[fc_key]["bundles"].items():
                        for component in components:
                            if "component_sku" in component:
                                bundle_component_skus.add(component["component_sku"])
                
                # Then process all_skus, but skip bundle SKUs to avoid incorrect mappings
                if "all_skus" in sku_mappings[fc_key]:
                    for shopify_sku, data in sku_mappings[fc_key]["all_skus"].items():
                        if "picklist_sku" in data and shopify_sku not in sku_mappings[fc_key].get("bundles", {}):
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

            # Determine if this is a bundle component and find appropriate Shopify SKU
            is_bundle_component = sku in bundle_component_skus
            
            if is_bundle_component:
                # For bundle components, find the first bundle that uses this component
                shopify_sku = ""
                if sku_mappings:
                    for fc_key in sku_mappings:
                        if "bundles" in sku_mappings[fc_key]:
                            for bundle_sku, components in sku_mappings[fc_key]["bundles"].items():
                                for component in components:
                                    if component.get("component_sku") == sku:
                                        shopify_sku = bundle_sku
                                        break
                                if shopify_sku:
                                    break
                        if shopify_sku:
                            break
            else:
                # For regular SKUs, use the mapping
                shopify_sku = inv_to_shopify.get(sku, "")

            # Add to summary
            inventory_summary.append(
                {
                    "Warehouse": warehouse,
                    "Inventory SKU": sku,
                    "Shopify SKU": shopify_sku,
                    "Current Balance": balance,
                    "Is Bundle Component": is_bundle_component,
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
                    # Determine if this is a bundle component and find appropriate Shopify SKU
                    is_bundle_component = sku in bundle_component_skus
                    
                    if is_bundle_component:
                        # For bundle components, find the first bundle that uses this component
                        shopify_sku = ""
                        if sku_mappings:
                            for fc_key in sku_mappings:
                                if "bundles" in sku_mappings[fc_key]:
                                    for bundle_sku, components in sku_mappings[fc_key]["bundles"].items():
                                        for component in components:
                                            if component.get("component_sku") == sku:
                                                shopify_sku = bundle_sku
                                                break
                                        if shopify_sku:
                                            break
                                if shopify_sku:
                                    break
                    else:
                        # For regular SKUs, use the mapping
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

                    # We only use the Balance column as requested, no fallback to AvailableQty

                    # Log the final balance value
                    logger.info(f"Final balance for {sku}: {balance}")

                    inventory_summary.append(
                        {
                            "Warehouse": warehouse,
                            "Inventory SKU": sku,
                            "Shopify SKU": shopify_sku,
                            "Current Balance": balance,
                            "Is Bundle Component": is_bundle_component,
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
        """Process orders and apply SKU mappings with proper inventory allocation.

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

        # Initialize tracking variables
        running_balances = {}
        initial_inventory_state = {}
        all_shortages = []
        shortage_tracker = set()
        processed_orders = []
        
        # Clear previous recalculation log if starting a new recalculation
        self._recalculation_log = []
        self._recalculated_skus = set()

        # Initialize inventory state
        if inventory_df is not None and not inventory_df.empty:
            self._initialize_inventory_state(inventory_df, running_balances, initial_inventory_state, affected_skus)

        # Load SKU weight data if not provided
        if sku_weight_data is None:
            sku_weight_data = self._load_sku_weight_data(sku_mappings)

        # Define output columns
        output_columns = [
            "externalorderid", "ordernumber", "CustomerFirstName", "customerLastname",
            "customeremail", "customerphone", "shiptoname", "shiptostreet1", "shiptostreet2",
            "shiptocity", "shiptostate", "shiptopostalcode", "note", "placeddate",
            "preferredcarrierserviceid", "totalorderamount", "shopsku", "shopquantity",
            "externalid", "Tags", "MAX PKG NUM", "Fulfillment Center", "shopifysku2", "sku",
            "actualqty", "Total Pick Weight", "quantity", "Starting Balance",
            "Transaction Quantity", "Ending Balance", "Issues",
        ]
        output_df = pd.DataFrame(columns=output_columns)

        # Apply column mapping
        self._apply_column_mapping(orders_df)

        # Process each order
        for _, row in orders_df.iterrows():
            order_id = row.get("externalorderid", row.get("ordernumber", "unknown"))
            
            # Extract SKU information
            shopify_sku = self._extract_shopify_sku(row, orders_df.columns)
            if shopify_sku is None:
                logger.warning(f"Warning: No SKU found for order {order_id}")
                continue

            # Skip if selective recalculation and SKU not affected
            if affected_skus is not None and not self._is_sku_affected(shopify_sku, affected_skus, bundle_info_dict, row):
                continue

            # Get fulfillment center and normalize
            fulfillment_center = row.get("Fulfillment Center", "Moorpark")
            fc_key = self.normalize_warehouse_name(fulfillment_center, log_transformations=True) if fulfillment_center else "Oxnard"

            # Create base order data
            order_data = self._create_base_order_data(row, output_columns, shopify_sku, fulfillment_center)

            # Find weight data
            found_match = self._find_weight_data_in_mappings(shopify_sku, fc_key, sku_weight_data, order_data)
            if not found_match:
                order_data["actualqty"] = ""
                order_data["Total Pick Weight"] = ""

            # Get shop quantity and set order quantity
            shop_quantity = int(row.get("shopquantity", 1)) if pd.notna(row.get("shopquantity")) else 1
            order_data["shopquantity"] = shop_quantity
            self._set_order_quantity(order_data, shop_quantity)

            # Check if this is a bundle and process accordingly
            is_bundle = (bundle_info_dict and fc_key in bundle_info_dict and shopify_sku in bundle_info_dict[fc_key])
            
            if is_bundle:
                bundle_components = bundle_info_dict[fc_key][shopify_sku]
                output_df = self._process_bundle_order(
                    order_data, bundle_components, shop_quantity, inventory_df,
                    running_balances, all_shortages, shortage_tracker, output_df,
                    sku_mappings, fc_key, row
                )
            else:
                output_df = self._process_single_sku_order(
                    order_data, shopify_sku, shop_quantity, inventory_df,
                    running_balances, all_shortages, shortage_tracker, output_df,
                    sku_mappings, fc_key, row
                )

        # Log completion
        logger.info(f"Processed {len(orders_df)} orders, generated {len(output_df)} output rows")
        logger.info(f"Found {len(all_shortages)} inventory shortages")

        return (output_df, all_shortages)

    def _find_sku_columns(self, df):
        """
        Find SKU and balance columns in a DataFrame with case-insensitive matching.
        
        Returns:
            tuple: (sku_column, balance_column) where each is the column name or None if not found
        """
        sku_column = None
        balance_column = None
        
        # Find SKU column
        sku_candidates = ['Sku', 'SKU', 'sku', 'Item #', 'Item#', 'Item Number', 'ItemNumber', 'inventory_sku', 'Inventory SKU']
        for candidate in sku_candidates:
            if candidate in df.columns:
                sku_column = candidate
                break
        
        # Try case-insensitive match for SKU if exact match not found
        if not sku_column:
            for col in df.columns:
                if col.lower() == 'sku':
                    sku_column = col
                    break
        
        # Find balance column
        balance_candidates = ['Balance', 'AvailableQty', 'Available Qty', 'Quantity', 'balance', 'availableqty']
        for candidate in balance_candidates:
            if candidate in df.columns:
                balance_column = candidate
                break
        
        return sku_column, balance_column

    def _find_weight_data_in_mappings(self, shopify_sku, fc_key, sku_weight_data, order_data):
        """
        Find weight data for a SKU in the mappings and set actualqty and Total Pick Weight.
        Enhanced to work with Airtable data structure.
        
        Args:
            shopify_sku (str): The Shopify SKU to look up
            fc_key (str): Fulfillment center key
            sku_weight_data (dict): Weight data dictionary (supports both old and new format)
            order_data (dict): Order data dictionary to update
            
        Returns:
            bool: True if weight data was found and set, False otherwise
        """
        try:
            # Normalize fc_key
            normalized_fc = self.normalize_warehouse_name(fc_key, log_transformations=False)
            
            # Handle both old and new data structures
            search_data = sku_weight_data
            
            # If using Airtable structure, get the mappings
            if isinstance(sku_weight_data, dict) and "mappings" in sku_weight_data:
                search_data = sku_weight_data["mappings"]
                logger.debug(f"Using Airtable format mappings for {shopify_sku}")
            
            # Check if we have weight data for this fulfillment center
            if not search_data or normalized_fc not in search_data:
                logger.debug(f"No weight data found for warehouse {normalized_fc}")
                return False
            
            fc_weight_data = search_data[normalized_fc]
            
            # Try multiple SKU variations
            sku_variations = [
                shopify_sku,  # Direct match
                shopify_sku[2:] if shopify_sku.startswith("f.") else shopify_sku,  # Without f. prefix
                f"f.{shopify_sku}" if not shopify_sku.startswith("f.") else shopify_sku  # With f. prefix
            ]
            
            for sku_variant in sku_variations:
                if sku_variant in fc_weight_data:
                    sku_data = fc_weight_data[sku_variant]
                    
                    if isinstance(sku_data, dict):
                        # Handle multiple field name variations
                        actualqty = (sku_data.get("actualqty") or 
                                   sku_data.get("Actualqty") or 
                                   sku_data.get("actual_qty") or "")
                        
                        weight = (sku_data.get("Total Pick Weight") or 
                                sku_data.get("Total_Pick_Weight") or 
                                sku_data.get("total_pick_weight") or "")
                        
                        mapped_sku = (sku_data.get("sku") or 
                                    sku_data.get("inventory_sku") or 
                                    sku_data.get("Inventory SKU") or "")
                        
                        # Set the data
                        order_data["actualqty"] = actualqty
                        order_data["Total Pick Weight"] = weight
                        
                        # Set the mapped inventory SKU if available
                        if mapped_sku:
                            order_data["sku"] = mapped_sku
                        
                        logger.debug(f"Found mapping: {shopify_sku} -> {mapped_sku} (qty: {actualqty}, weight: {weight})")
                        return True
            
            # Try alternate warehouse if current one doesn't have the mapping
            alternate_fc = "Wheeling" if normalized_fc == "Oxnard" else "Oxnard"
            if alternate_fc in search_data:
                alt_fc_data = search_data[alternate_fc]
                for sku_variant in sku_variations:
                    if sku_variant in alt_fc_data:
                        sku_data = alt_fc_data[sku_variant]
                        if isinstance(sku_data, dict):
                            actualqty = (sku_data.get("actualqty") or 
                                       sku_data.get("Actualqty") or 
                                       sku_data.get("actual_qty") or "")
                            
                            weight = (sku_data.get("Total Pick Weight") or 
                                    sku_data.get("Total_Pick_Weight") or 
                                    sku_data.get("total_pick_weight") or "")
                            
                            mapped_sku = (sku_data.get("sku") or 
                                        sku_data.get("inventory_sku") or 
                                        sku_data.get("Inventory SKU") or "")
                            
                            order_data["actualqty"] = actualqty
                            order_data["Total Pick Weight"] = weight
                            if mapped_sku:
                                order_data["sku"] = mapped_sku
                            
                            logger.debug(f"Found mapping in alternate warehouse {alternate_fc}: {shopify_sku} -> {mapped_sku}")
                            return True
            
            logger.debug(f"No mapping found for SKU {shopify_sku} in any warehouse")
            return False
            
        except Exception as e:
            logger.warning(f"Error finding weight data for SKU {shopify_sku}: {str(e)}")
            return False

    def _set_order_quantity(self, order_data, shop_quantity):
        """
        Set the order quantity in order_data.
        
        Args:
            order_data (dict): Order data dictionary to update
            shop_quantity (int): The shop quantity to set
        """
        order_data["quantity"] = shop_quantity

    def _initialize_inventory_state(self, inventory_df, running_balances, initial_inventory_state, affected_skus=None):
        """Initialize inventory state from inventory DataFrame with proper warehouse separation."""
        sku_column, balance_column = self._find_sku_columns(inventory_df)
        if not sku_column:
            logger.warning("Warning: 'Sku' column not found in inventory DataFrame")
            return

        # Initialize counters for debugging
        total_entries = 0
        warehouse_counts = {}
        
        # Track processed keys to avoid duplicate overwriting
        processed_keys = set()
        
        for _, row in inventory_df.iterrows():
            sku = str(row[sku_column]).strip()
            
            # Get warehouse - prefer 'Warehouse' over 'WarehouseName' if available
            warehouse = row.get('Warehouse', row.get('WarehouseName', 'Unknown'))
            normalized_warehouse = self.normalize_warehouse_name(warehouse, log_transformations=False) if warehouse else "Unknown"
            
            # Track warehouse distribution for debugging
            warehouse_counts[normalized_warehouse] = warehouse_counts.get(normalized_warehouse, 0) + 1
            
            # Create composite key for unique inventory tracking per warehouse
            composite_key = f"{sku}|{normalized_warehouse}"
            
            # Only process each SKU+warehouse combination once (use the first/highest value)
            if composite_key in processed_keys:
                continue
            
            # Get balance from appropriate column
            balance = 0.0
            if balance_column and balance_column in row and pd.notna(row[balance_column]):
                balance = float(row[balance_column])
            elif "Balance" in row and pd.notna(row["Balance"]):
                balance = float(row["Balance"])
            elif "Current Balance" in row and pd.notna(row["Current Balance"]):
                balance = float(row["Current Balance"])
            
            # Store in running balances for order processing
            running_balances[composite_key] = balance
            
            # Store in initial state for tracking changes
            initial_inventory_state[composite_key] = {
                'balance': balance,
                'warehouse': normalized_warehouse,
                'sku': sku,
                'original_warehouse': warehouse  # Keep original for reference
            }
            
            # Mark this SKU+warehouse combination as processed
            processed_keys.add(composite_key)
            
            total_entries += 1

        logger.info(f"Initialized inventory state with {total_entries} entries across warehouses: {warehouse_counts}")
        logger.info(f"Used first entry for each unique SKU+warehouse combination, ignoring duplicates")

    def _load_sku_weight_data(self, sku_mappings):
        """Load SKU weight data from mapping files"""
        sku_weight_data = {}
        try:
            import os
            sku_mappings_path = os.path.join("constants", "data", "sku_mappings.json")
            if os.path.exists(sku_mappings_path):
                with open(sku_mappings_path, "r") as f:
                    loaded_mappings = json.load(f)
                
                for warehouse in ["Wheeling", "Oxnard"]:
                    if warehouse in loaded_mappings and "all_skus" in loaded_mappings[warehouse]:
                        warehouse_data = {}
                        for sku, data in loaded_mappings[warehouse]["all_skus"].items():
                            warehouse_data[sku] = {
                                "actualqty": data.get("actualqty", ""),
                                "Total Pick Weight": data.get("Total_Pick_Weight", ""),
                            }
                            # Handle SKU variations
                            if sku.startswith("f."):
                                warehouse_data[sku[2:]] = warehouse_data[sku]
                            elif not sku.startswith("f."):
                                warehouse_data[f"f.{sku}"] = warehouse_data[sku]
                        
                        sku_weight_data[warehouse] = warehouse_data
                        logger.info(f"Loaded weight data for {len(warehouse_data)} SKUs from {warehouse} mapping file")
        except Exception as e:
            logger.error(f"Error loading SKU weight data: {str(e)}")
        
        return sku_weight_data

    def _apply_column_mapping(self, orders_df):
        """Apply column mapping to standardize order DataFrame columns"""
        column_mapping = {
            "Name": "externalorderid",
            "order id": "ordernumber",
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
        }

        for old_col, new_col in column_mapping.items():
            if old_col in orders_df.columns and new_col not in orders_df.columns:
                orders_df[new_col] = orders_df[old_col]

    def _extract_shopify_sku(self, row, columns):
        """Extract Shopify SKU from order row"""
        # Try standard SKU columns first
        for col in ["shopsku", "SKU Helper", "sku", "lineitemsku", "shopifysku", "shopifysku2"]:
            if col in row and pd.notna(row[col]):
                return row[col]
        
        # Check fruit-specific columns
        sku_columns = [col for col in columns if col.startswith("f.") or col.startswith("m.")]
        for col in sku_columns:
            if col in row and pd.notna(row[col]) and float(row[col]) > 0:
                return col
        
        return None

    def _is_sku_affected(self, shopify_sku, affected_skus, bundle_info_dict, row):
        """Check if SKU is affected for selective recalculation"""
        if shopify_sku in affected_skus:
            return True
        
        # Check if any bundle components are affected
        fc_key = self.normalize_warehouse_name(row.get("Fulfillment Center", "Moorpark"))
        if bundle_info_dict and fc_key in bundle_info_dict and shopify_sku in bundle_info_dict[fc_key]:
            for component in bundle_info_dict[fc_key][shopify_sku]:
                if component.get("component_sku") in affected_skus:
                    return True
        
        return False

    def _create_base_order_data(self, row, output_columns, shopify_sku, fulfillment_center):
        """Create base order data dictionary"""
        order_data = {col: "" for col in output_columns}
        
        # Copy existing fields from the order row
        for field in output_columns:
            if field in row and pd.notna(row[field]):
                if field == "placeddate":
                    try:
                        date_value = pd.to_datetime(row[field])
                        order_data[field] = date_value.strftime("%-m/%-d/%Y")
                    except:
                        order_data[field] = row[field]
                else:
                    order_data[field] = row[field]

        # Add SKU information
        order_data["shopsku"] = shopify_sku
        order_data["shopifysku2"] = shopify_sku
        order_data["Fulfillment Center"] = fulfillment_center
        
        return order_data

    def _process_bundle_order(self, order_data, bundle_components, shop_quantity, inventory_df,
                             running_balances, all_shortages, shortage_tracker, output_df,
                             sku_mappings, fc_key, row):
        """Process a bundle order by handling each component"""
        sku_column, balance_column = self._find_sku_columns(inventory_df)
        if not sku_column:
            return output_df

        for component in bundle_components:
            component_order_data = order_data.copy()
            component_sku = component["sku"]
            base_component_qty = float(component.get("qty", 1.0))
            component_weight = float(component.get("weight", 0.0))
            
            component_qty = base_component_qty * shop_quantity
            total_component_weight = component_weight * shop_quantity

            # Set component data
            component_order_data["sku"] = component_sku
            component_order_data["actualqty"] = component_qty
            component_order_data["Total Pick Weight"] = total_component_weight
            component_order_data["quantity"] = component_qty

            # Map component SKU to inventory SKU
            component_inventory_sku = self.map_shopify_to_inventory_sku(
                shopify_sku=component_sku,
                fulfillment_center=fc_key,
                sku_mappings=sku_mappings
            )
            
            # Use the inventory balance directly - no more actualqty overrides from mapping

            # Get current balance and process allocation
            current_balance = self._get_current_balance(
                component_inventory_sku, fc_key, running_balances, inventory_df, sku_column, balance_column
            )

            # Check for shortages and allocate
            self._allocate_inventory(
                component_order_data, component_inventory_sku, component_qty, current_balance,
                running_balances, all_shortages, shortage_tracker, fc_key, row, order_data["shopsku"]
            )

            # Add to output
            output_df = pd.concat([output_df, pd.DataFrame([component_order_data])], ignore_index=True)
        
        return output_df

    def _process_single_sku_order(self, order_data, shopify_sku, shop_quantity, inventory_df,
                                 running_balances, all_shortages, shortage_tracker, output_df,
                                 sku_mappings, fc_key, row) -> pd.DataFrame:
        """Process a single SKU order"""
        sku_column, balance_column = self._find_sku_columns(inventory_df)
        if not sku_column:
            return output_df

        # Map SKU to inventory SKU
        inventory_sku = self.map_shopify_to_inventory_sku(
            shopify_sku=shopify_sku,
            fulfillment_center=fc_key,
            sku_mappings=sku_mappings
        )

        order_data["sku"] = inventory_sku

        # Use shop_quantity directly as requested quantity - no actualqty override
        requested_quantity = shop_quantity

        # Get current balance and process allocation
        current_balance = self._get_current_balance(
            inventory_sku, fc_key, running_balances, inventory_df, sku_column, balance_column
        )

        # Check for shortages and allocate
        self._allocate_inventory(
            order_data, inventory_sku, requested_quantity, current_balance,
            running_balances, all_shortages, shortage_tracker, fc_key, row, shopify_sku
        )

        # Add to output
        output_df = pd.concat([output_df, pd.DataFrame([order_data])], ignore_index=True)
        
        return output_df

    def _get_current_balance(self, inventory_sku, fc_key, running_balances, inventory_df, sku_column, balance_column):
        """Get current balance for a SKU with fallback to inventory lookup"""
        normalized_warehouse = self.normalize_warehouse_name(fc_key, log_transformations=False)
        composite_key = f"{inventory_sku}|{normalized_warehouse}"
        
        if composite_key in running_balances:
            return running_balances[composite_key]
        elif inventory_sku in running_balances:
            return running_balances[inventory_sku]
        else:
            # Look up in inventory
            sku_inventory = inventory_df[inventory_df[sku_column] == inventory_sku]
            if sku_inventory.empty:
                return 0
            
            if balance_column and balance_column in sku_inventory.columns:
                balance = float(sku_inventory.iloc[0][balance_column]) if pd.notna(sku_inventory.iloc[0][balance_column]) else 0
                running_balances[composite_key] = balance
                return balance
            
            return 0

    def _allocate_inventory(self, order_data, inventory_sku, requested_qty, current_balance,
                           running_balances, all_shortages, shortage_tracker, fc_key, row, shopify_sku) -> None:
        """Allocate inventory and handle shortages"""
        normalized_warehouse = self.normalize_warehouse_name(fc_key, log_transformations=False)
        composite_key = f"{inventory_sku}|{normalized_warehouse}"
        
        order_data["Starting Balance"] = current_balance
        
        if current_balance < requested_qty:
            # Handle shortage
            shortage = requested_qty - current_balance
            order_id = row.get("externalorderid", "") or row.get("id", "")
            shortage_key = (inventory_sku, str(order_id))

            if shortage_key not in shortage_tracker:
                all_shortages.append({
                    "component_sku": inventory_sku,
                    "shopify_sku": shopify_sku,
                    "order_id": order_id,
                    "current_balance": current_balance,
                    "needed_qty": requested_qty,
                    "shortage_qty": shortage,
                    "fulfillment_center": fc_key,
                })
                shortage_tracker.add(shortage_key)

            order_data["Issues"] = f"Issue: Insufficient inventory | Item: {inventory_sku} | Shopify SKU: {shopify_sku} | Current: {current_balance} | Needed: {requested_qty} | Short by: {shortage} units"
            transaction_quantity = min(current_balance, requested_qty)
        else:
            transaction_quantity = requested_qty

        order_data["Transaction Quantity"] = transaction_quantity
        ending_balance = max(0, current_balance - transaction_quantity)
        order_data["Ending Balance"] = ending_balance
        
        # Update running balance
        running_balances[composite_key] = ending_balance

    def generate_inventory_comparison(self, initial_inventory_state, current_balances, inventory_df, sku_mappings=None) -> pd.DataFrame:
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
                        # Only using the Balance column as requested, not falling back to AvailableQty
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
                    # Only using the Balance column as requested, not falling back to AvailableQty
            
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

    def calculate_processing_stats(self, processed_orders, inventory_summary, shortage_summary) -> dict:
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

    def _apply_inventory_threshold_rules(self, orders_df, inventory_rules) -> pd.DataFrame:
        """
        Apply inventory threshold rules to orders.
        
        Args:
            orders_df (pd.DataFrame): DataFrame containing order data
            inventory_rules (list): List of inventory threshold rules to apply
            
        Returns:
            pd.DataFrame: Updated orders DataFrame with inventory rules applied
        """
        if orders_df is None or orders_df.empty or not inventory_rules:
            return orders_df
            
        result_df = orders_df.copy()
        
        # Get current inventory levels
        # This would typically come from Airtable or another source
        # For now, we'll use a simple approach to demonstrate the concept
        try:
            inventory_data = {}
            
            # If we have a schema manager, try to get inventory data
            if self.schema_manager:
                # Get inventory data from Airtable
                inventory_records = self.schema_manager.get_inventory_records()
                
                # Process inventory records
                for record in inventory_records:
                    sku = record.get('sku', '')
                    warehouse = record.get('warehouse', '')
                    balance = record.get('balance', 0)
                    
                    if sku and warehouse:
                        # Normalize warehouse name
                        norm_warehouse = self.normalize_warehouse_name(warehouse)
                        
                        # Create warehouse entry if it doesn't exist
                        if norm_warehouse not in inventory_data:
                            inventory_data[norm_warehouse] = {}
                            
                        # Store balance
                        inventory_data[norm_warehouse][sku] = balance
        except Exception as e:
            logger.error(f"Error getting inventory data: {str(e)}")
            # Continue with empty inventory data
            inventory_data = {'Oxnard': {}, 'Wheeling': {}}
        
        # Process each rule
        for rule in inventory_rules:
            # Skip inactive rules
            if not rule.is_active:
                continue
                
            # Get rule action and condition
            action = rule.get_action()
            condition = rule.get_condition()
            
            # Check if this is a valid inventory threshold rule
            if not condition or 'sku' not in condition or 'threshold' not in condition or 'warehouse' not in condition:
                continue
                
            # Get rule parameters
            target_sku = condition.get('sku', '')
            threshold = condition.get('threshold', 0)
            warehouse = condition.get('warehouse', '')
            target_warehouse = action.get('target_warehouse', '')
            
            # Normalize warehouse names
            source_warehouse = self.normalize_warehouse_name(warehouse)
            target_warehouse = self.normalize_warehouse_name(target_warehouse)
            
            # Check if we have inventory data for this warehouse and SKU
            if source_warehouse in inventory_data and target_sku in inventory_data[source_warehouse]:
                current_balance = inventory_data[source_warehouse][target_sku]
                
                # Check if balance is below threshold
                if current_balance <= threshold:
                    # Find orders with the target SKU in the source warehouse
                    mask = (result_df['sku'].str.lower() == target_sku.lower()) & \
                           (result_df['Fulfillment Center'] == source_warehouse)
                    
                    # Update fulfillment center for matching orders
                    if mask.any() and target_warehouse:
                        result_df.loc[mask, 'Fulfillment Center'] = target_warehouse
                        
                        # Log the change
                        affected_count = mask.sum()
                        logger.info(f"Inventory threshold rule applied: Moved {affected_count} orders with SKU {target_sku} from {source_warehouse} to {target_warehouse}")
        
        return result_df
        
    def _apply_zone_override_rules(self, orders_df, zone_rules) -> pd.DataFrame:
        """
        Apply zone override rules to orders.
        
        Args:
            orders_df (pd.DataFrame): DataFrame containing order data
            zone_rules (list): List of zone override rules to apply
            
        Returns:
            pd.DataFrame: Updated orders DataFrame with zone overrides applied
        """
        if orders_df is None or orders_df.empty or not zone_rules:
            return orders_df
            
        result_df = orders_df.copy()
        
        # Get shipping zones data
        try:
            # Load shipping zones from the schema manager if available
            shipping_zones_df = None
            if self.schema_manager:
                # Get fulfillment zones from Airtable
                fulfillment_zones = self.schema_manager.get_fulfillment_zones()
                
                if fulfillment_zones:
                    # Convert to DataFrame
                    zones_data = []
                    for zone in fulfillment_zones:
                        # Remember that zone is a string in Airtable, not an integer
                        # And fulfillment_center is a list of IDs in Airtable format
                        zones_data.append({
                            'zip_prefix': zone.zip_prefix,
                            'zone': zone.zone,  # This is a string
                            'fulfillment_center': zone.fulfillment_center[0] if zone.fulfillment_center else None
                        })
                    
                    if zones_data:
                        shipping_zones_df = pd.DataFrame(zones_data)
            
            # If we couldn't get zones from schema manager, try loading from file
            if shipping_zones_df is None or shipping_zones_df.empty:
                from constants.shipping_zones import load_shipping_zones
                shipping_zones_df = load_shipping_zones()
        except Exception as e:
            logger.error(f"Error loading shipping zones: {str(e)}")
            # Continue with empty shipping zones
            shipping_zones_df = pd.DataFrame(columns=['zip_prefix', 'zone', 'fulfillment_center'])
        
        # Process each rule
        for rule in zone_rules:
            # Skip inactive rules
            if not rule.is_active:
                continue
                
            # Get rule action and condition
            action = rule.get_action()
            condition = rule.get_condition()
            
            # Check if this is a valid zone override rule
            if not condition or 'zip_code' not in condition or not action or 'target_fc' not in action:
                continue
                
            # Get rule parameters
            zip_code_pattern = condition.get('zip_code', '')
            target_fc = action.get('target_fc', '')
            
            # Normalize target fulfillment center
            target_fc = self.normalize_warehouse_name(target_fc)
            
            # Find orders with matching zip code pattern
            if 'ShipToZip' in result_df.columns:
                # Handle different pattern types
                if zip_code_pattern.endswith('*'):  # Prefix match
                    prefix = zip_code_pattern.rstrip('*')
                    mask = result_df['ShipToZip'].astype(str).str.startswith(prefix)
                elif '*' in zip_code_pattern:  # Pattern match
                    pattern = zip_code_pattern.replace('*', '.*')
                    mask = result_df['ShipToZip'].astype(str).str.match(pattern)
                else:  # Exact match
                    mask = result_df['ShipToZip'].astype(str) == zip_code_pattern
                
                # Update fulfillment center for matching orders
                if mask.any() and target_fc:
                    result_df.loc[mask, 'Fulfillment Center'] = target_fc
                    
                    # Log the change
                    affected_count = mask.sum()
                    logger.info(f"Zone override rule applied: Moved {affected_count} orders with zip pattern {zip_code_pattern} to {target_fc}")
        
        return result_df

    def calculate_warehouse_performance(self, processed_orders, inventory_summary) -> dict:
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

    def map_shopify_to_inventory_sku(self, shopify_sku, fulfillment_center=None, sku_mappings=None) -> str:
        """
        Maps a Shopify SKU to the corresponding inventory SKU using the Airtable mappings.
        This function focuses solely on the SKU mapping logic, following the single responsibility principle.
        
        Args:
            shopify_sku (str): The Shopify SKU to map
            fulfillment_center (str, optional): The fulfillment center ("Oxnard" or "Wheeling")
            sku_mappings (dict, optional): SKU mapping dictionary (if already loaded)
            
        Returns:
            str: The mapped inventory SKU or the original SKU if no mapping exists
        """
        if not shopify_sku:
            return ""
            
        # Normalize input
        shopify_sku = str(shopify_sku).strip()
        
        # If fulfillment center isn't specified, default to both centers
        fc_keys = []
        if fulfillment_center:
            normalized_fc = self.normalize_warehouse_name(fulfillment_center)
            if normalized_fc.lower() == "oxnard":
                fc_keys = ["oxnard", "moorpark"]
            elif normalized_fc.lower() == "wheeling":
                fc_keys = ["wheeling", "il-wheeling"]
        else:
            fc_keys = ["wheeling", "il-wheeling", "oxnard", "moorpark"]
        
        # Use provided mappings or load from instance
        mappings = sku_mappings or self.sku_mappings
        if not mappings:
            logger.warning("No SKU mappings available for mapping Shopify SKU to inventory SKU")
            return shopify_sku
            
        # Check each fulfillment center for mappings
        # First, check if mappings has a "mappings" structure (new Airtable format)
        if "mappings" in mappings:
            # New Airtable format: mappings["mappings"]["Oxnard"][shopify_sku] = inventory_sku
            for fc_key in fc_keys:
                center_name = "Oxnard" if fc_key.lower() in ["oxnard", "moorpark"] else "Wheeling"
                if center_name in mappings["mappings"]:
                    center_mappings = mappings["mappings"][center_name]
                    if shopify_sku in center_mappings:
                        inventory_sku = center_mappings[shopify_sku]
                        if inventory_sku:
                            logger.info(f"Mapped Shopify SKU '{shopify_sku}' to inventory SKU '{inventory_sku}' for {center_name}")
                            return inventory_sku
        else:
            # Legacy format: try the old structure
            for fc_key in fc_keys:
                if fc_key in mappings and "mappings" in mappings[fc_key]:
                    # Look for direct mapping
                    for mapping in mappings[fc_key]["mappings"]:
                        if mapping.get("shopify_sku", "").lower() == shopify_sku.lower():
                            inventory_sku = mapping.get("inventory_sku", "")
                            if inventory_sku:
                                logger.info(f"Mapped Shopify SKU '{shopify_sku}' to inventory SKU '{inventory_sku}' for {fc_key}")
                                return inventory_sku
        
        # If no mapping found, return original SKU
        logger.info(f"No mapping found for Shopify SKU '{shopify_sku}', using the original SKU")
        return shopify_sku

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
            if col.lower() in ["sku", "inventory sku", "inventory_sku", "sku"]:
                sku_column = col
                break

        if sku_column is None:
            logger.warning(f"Warning: 'Sku' or 'Inventory SKU' column not found in inventory DataFrame. Available columns: {inventory_df.columns.tolist()}")
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
            # Handle both old format (WarehouseName/Balance) and new format (Warehouse/Current Balance)
            warehouse_col = None
            balance_col = None
            
            # Determine which column names to use
            for col in inventory_df.columns:
                if col.lower() in ["warehousename", "warehouse", "fulfillmentcenter"]:
                    warehouse_col = col
                elif col.lower() in ["balance", "current balance", "availableqty", "current_balance"]:
                    balance_col = col
            
            if warehouse_col is None or balance_col is None:
                logger.warning(f"Could not find warehouse or balance columns in inventory. Available columns: {inventory_df.columns.tolist()}")
                return 0

            # Look for the SKU in Wheeling warehouse first
            wheeling_rows = inventory_df[
                (inventory_df[sku_column] == sku_value)
                & (
                    inventory_df[warehouse_col]
                    .str.lower()
                    .str.contains("wheeling|il-wheeling|il-", na=False)
                )
            ]

            # If found in Wheeling, check Balance
            if not wheeling_rows.empty:
                try:
                    balance_value = wheeling_rows.iloc[0][balance_col]
                    if isinstance(balance_value, str):
                        balance_value = balance_value.replace(",", "")
                    balance = float(balance_value) if pd.notna(balance_value) else 0
                    logger.info(f"Using {balance_col} column for {sku_value} in Wheeling: {balance}")
                    return balance
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Could not convert {balance_col} value for {sku_value}: {balance_value} - Error: {e}"
                    )

            # If not found in Wheeling or no Balance, try Oxnard (Moorpark) warehouse
            oxnard_rows = inventory_df[
                (inventory_df[sku_column] == sku_value)
                & (
                    inventory_df[warehouse_col]
                    .str.lower()
                    .str.contains("oxnard|moorpark|ca-", na=False)
                )
            ]
            if not oxnard_rows.empty:
                try:
                    balance_value = oxnard_rows.iloc[0][balance_col]
                    if isinstance(balance_value, str):
                        balance_value = balance_value.replace(",", "")
                    balance = float(balance_value) if pd.notna(balance_value) else 0
                    logger.info(
                        f"Using {balance_col} column for {sku_value} in other warehouse: {balance}"
                    )
                    return balance
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Could not convert {balance_col} value for {sku_value}: {balance_value} - Error: {e}"
                    )

            # If we get here, no valid Balance found
            logger.info(f"No valid {balance_col} found for {sku_value}, returning 0")
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

        # STEP 2: If no mapping is provided (None), try direct inventory lookup
        # This happens when the SKU was already mapped earlier in the process
        if sku_mapping is None:
            # Try exact match
            matching_rows = inventory_df[inventory_df[sku_column] == sku]
            
            if not matching_rows.empty:
                # Return the first matching SKU and balance
                inv_sku = matching_rows.iloc[0][sku_column]
                matched_balance = get_balance_for_sku(inv_sku)
                logger.info(f"Found exact match in inventory: '{sku}' with balance {matched_balance}")
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
                        f"Found case-insensitive match in inventory: '{sku}' -> '{inv_sku}' with balance {matched_balance}"
                    )
                    # Check if there's enough inventory
                    if matched_balance <= 0:
                        issue = f"No inventory available for {inv_sku}"
                        logger.warning(issue)
                    return inv_sku, matched_balance, issue
            
            # If no exact match found in inventory
            logger.warning(f"SKU '{sku}' not found in inventory.")
            issue = f"SKU '{sku}' not found in inventory."
            return sku, 0, issue  # Return the SKU as-is with 0 balance

        # STEP 3: If mapping exists but SKU not found in mappings, try direct inventory lookup as fallback
        # Try exact match in inventory
        matching_rows = inventory_df[inventory_df[sku_column] == sku]
        
        if not matching_rows.empty:
            # Return the first matching SKU and balance
            inv_sku = matching_rows.iloc[0][sku_column]
            matched_balance = get_balance_for_sku(inv_sku)
            logger.info(f"Found exact match in inventory as fallback: '{sku}' with balance {matched_balance}")
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
                    f"Found case-insensitive match in inventory as fallback: '{sku}' -> '{inv_sku}' with balance {matched_balance}"
                )
                # Check if there's enough inventory
                if matched_balance <= 0:
                    issue = f"No inventory available for {inv_sku}"
                    logger.warning(issue)
                return inv_sku, matched_balance, issue
        
        # If still no match found
        logger.warning(f"SKU '{sku}' not found in SKU mappings JSON file or inventory.")
        issue = f"SKU '{sku}' not found in inventory or mappings."
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
    import sys
    logging.basicConfig(level=logging.INFO)
    data_processor = DataProcessor()
    
    # Import datetime for timestamp generation
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if user wants to process raw inventory
    if len(sys.argv) > 1 and sys.argv[1] == "process_inventory":
        print("🔬 Processing raw inventory data...")
        
        # Look for raw inventory file
        inventory_file = "docs/raw_inventory.csv"
        if len(sys.argv) > 2:
            inventory_file = sys.argv[2]
        
        if not os.path.exists(inventory_file):
            print(f"❌ Error: Inventory file '{inventory_file}' not found")
            print("Usage: python data_processor.py process_inventory [path_to_inventory.csv]")
            sys.exit(1)
        
        # Process the raw inventory
        processed_inventory = data_processor.process_raw_inventory(inventory_file)
        
        if not processed_inventory.empty:
            # Save processed inventory
            output_file = f"processed_inventory_{timestamp}.csv"
            processed_inventory.to_csv(output_file, index=False)
            
            print(f"✅ Successfully processed {len(processed_inventory)} inventory items!")
            print(f"📁 Saved to: {output_file}")
            
            # Show summary statistics
            total_balance = processed_inventory['Current Balance'].sum()
            warehouses = processed_inventory['Warehouse'].unique()
            unique_skus = processed_inventory['Inventory SKU'].nunique()
            
            print(f"\n📊 Summary:")
            print(f"  • Total SKUs: {unique_skus}")
            print(f"  • Warehouses: {', '.join(warehouses)}")
            print(f"  • Total Inventory: {total_balance:,.0f}")
            
            # Show warehouse breakdown
            print(f"\n🏭 Warehouse Breakdown:")
            warehouse_summary = processed_inventory.groupby('Warehouse').agg({
                'Inventory SKU': 'count',
                'Current Balance': 'sum'
            })
            for warehouse, data in warehouse_summary.iterrows():
                print(f"  • {warehouse}: {data['Inventory SKU']} SKUs, {data['Current Balance']:,.0f} total balance")
            
            # Test SKU mappings if requested
            if len(sys.argv) > 3 and sys.argv[3] == "test_mappings":
                print(f"\n🔍 Testing SKU mappings...")
                sku_mappings = data_processor.load_sku_mappings()
                
                if sku_mappings and "mappings" in sku_mappings:
                    mappings_dict = sku_mappings["mappings"]
                    missing_skus = []
                    
                    for warehouse in ["Oxnard", "Wheeling"]:
                        if warehouse in mappings_dict:
                            for shopify_sku, mapped_sku in mappings_dict[warehouse].items():
                                # Check if mapped SKU exists in processed inventory
                                inventory_matches = processed_inventory[
                                    (processed_inventory['Inventory SKU'] == mapped_sku) &
                                    (processed_inventory['Warehouse'] == warehouse)
                                ]
                                
                                if inventory_matches.empty:
                                    missing_skus.append(f"{warehouse}: {shopify_sku} → {mapped_sku}")
                    
                    if missing_skus:
                        print(f"⚠️  Found {len(missing_skus)} mapped SKUs not in inventory:")
                        for missing in missing_skus[:10]:  # Show first 10
                            print(f"   {missing}")
                        if len(missing_skus) > 10:
                            print(f"   ... and {len(missing_skus) - 10} more")
                    else:
                        print("✅ All mapped SKUs found in inventory!")
                else:
                    print("❌ Could not load SKU mappings")
        else:
            print("❌ Failed to process inventory. Check file format and required columns.")
        
        sys.exit(0)
    
    # Default behavior: process orders
    save_to_file = f"inventory_comparison_{timestamp}.csv"
    shipping_zones_df = load_shipping_zones("docs/shipping_zones.csv")
    orders_df = data_processor.load_orders("docs/orders.csv")
    inventory_df = data_processor.load_inventory("docs/inventory.csv")
    sku_mappings = data_processor.load_sku_mappings()

    if not orders_df.empty:
        # Save initial inventory state BEFORE processing orders
        initial_inventory_file = f"inventory_before_processing_{timestamp}.csv"
        
        # Create initial inventory summary from the loaded inventory_df
        inventory_summary_initial = []
        
        # Check if we have the expected columns from inventory loading
        if 'WarehouseName' in inventory_df.columns and 'Sku' in inventory_df.columns:
            for _, row in inventory_df.iterrows():
                warehouse = data_processor.normalize_warehouse_name(row['WarehouseName'])
                sku = str(row['Sku']).strip()
                balance = row.get('Balance', 0) or row.get('AvailableQty', 0)
                
                if sku and sku != 'nan':
                    inventory_summary_initial.append({
                        'Warehouse': warehouse,
                        'Inventory SKU': sku,
                        'Current Balance': balance
                    })
        
        inventory_summary_initial = pd.DataFrame(inventory_summary_initial)
        
        # Group inventory by SKU and warehouse separately (do not sum across warehouses)
        if not inventory_summary_initial.empty:
            # Get the maximum balance per SKU-warehouse combination (keep warehouses separate)
            inventory_grouped = inventory_summary_initial.groupby(['Inventory SKU', 'Warehouse'])['Current Balance'].max().reset_index()
            # Reorder columns: SKU, Warehouse, Balance
            inventory_grouped = inventory_grouped[['Inventory SKU', 'Warehouse', 'Current Balance']]
            inventory_summary_initial = inventory_grouped
        inventory_summary_initial.to_csv(initial_inventory_file, index=False)
        print(f"📦 Saved initial inventory state to: {initial_inventory_file}")
        print(f"   Initial inventory items: {len(inventory_summary_initial)}")
        
        # Show initial inventory summary by warehouse
        if not inventory_summary_initial.empty and 'Warehouse' in inventory_summary_initial.columns:
            initial_warehouse_summary = inventory_summary_initial.groupby('Warehouse').agg({
                'Inventory SKU': 'count',
                'Current Balance': 'sum'
            })
            print(f"   Initial inventory by warehouse:")
            for warehouse, data in initial_warehouse_summary.iterrows():
                total_balance = data['Current Balance']
                sku_count = data['Inventory SKU']
                print(f"     • {warehouse}: {sku_count} SKUs, {total_balance:,.0f} total balance")
        
        print(f"\n🔄 Processing {len(orders_df)} orders...")
        
        # Process orders and get orders, inventory summary, and shortage summary
        result = data_processor.process_orders(
            orders_df, inventory_df, shipping_zones_df, sku_mappings
        )
        processed_orders, shortage_summary = result
        
        # Generate inventory summary from current balances if needed
        # Note: For now we'll create an empty inventory summary since the main focus 
        # is on processed orders and shortages
        inventory_summary = pd.DataFrame()

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

        # Save inventory summary to a separate file (AFTER processing)
        inventory_output_path = "inventory_summary.csv"
        inventory_summary.to_csv(inventory_output_path, index=False)
        logger.info(
            f"Generated inventory summary with {len(inventory_summary)} SKUs and saved to {inventory_output_path}"
        )
        
        # Also save the final inventory state with timestamp for comparison
        final_inventory_file = f"inventory_after_processing_{timestamp}.csv"
        
        # Before saving the final inventory, check if we have a regular inventory comparison file
        # which would have the correct values for SKUs that might have multiple mappings
        regular_comparison_file = f"inventory_comparison_{timestamp}.csv"
        reference_values = {}
        
        if os.path.exists(regular_comparison_file):
            try:
                # Load the regular comparison file which has the correct values
                reference_df = pd.read_csv(regular_comparison_file)
                
                # Create a lookup dictionary with (SKU, Warehouse) as key
                for _, row in reference_df.iterrows():
                    key = (row['SKU'], row['Warehouse'])
                    reference_values[key] = {
                        'Current Balance': row['Current Balance'],
                        'Shopify SKU': row['Shopify SKU'],
                        'Is Used In Bundle': row['Is Used In Bundle']
                    }
                logger.info(f"Loaded reference values from comparison file with {len(reference_values)} entries")
            except Exception as e:
                logger.error(f"Error loading reference values from comparison file: {e}")
        
        # Create a copy without the formatting for numerical analysis
        inventory_final_numeric = inventory_summary.copy()
        
        # Update inventory values based on reference values if available
        if reference_values:
            # Create a new column to track if a row has been updated from reference
            inventory_final_numeric['Updated_From_Reference'] = False
            
            # For each row in inventory_final_numeric, check if we have a reference value
            for idx, row in inventory_final_numeric.iterrows():
                key = (row['Inventory SKU'], row['Warehouse'])
                if key in reference_values:
                    # Update the current balance and other fields from the reference
                    inventory_final_numeric.at[idx, 'Current Balance'] = reference_values[key]['Current Balance']
                    inventory_final_numeric.at[idx, 'Shopify SKU'] = reference_values[key]['Shopify SKU']
                    inventory_final_numeric.at[idx, 'Is Bundle Component'] = reference_values[key]['Is Used In Bundle']
                    inventory_final_numeric.at[idx, 'Updated_From_Reference'] = True
                    
                    logger.info(f"Updated inventory value for {key} from reference: {reference_values[key]['Current Balance']}")
            
            # Remove the tracking column
            inventory_final_numeric.drop('Updated_From_Reference', axis=1, inplace=True)
        
        # Convert to numeric for comparison
        if "Current Balance" in inventory_final_numeric.columns:
            # Convert back to numeric for comparison
            inventory_final_numeric["Current Balance"] = pd.to_numeric(
                inventory_final_numeric["Current Balance"].astype(str).str.replace(",", ""), 
                errors="coerce"
            ).fillna(0)
        
        # Make sure we're saving the right columns in the right order for inventory_after_processing
        if 'Warehouse' in inventory_final_numeric.columns and 'Inventory SKU' in inventory_final_numeric.columns:
            # Create a new DataFrame with just the columns we need in the right order
            # We don't aggregate balances - maintain individual rows for the same SKU-warehouse combinations
            inventory_after_processing = pd.DataFrame({
                'Warehouse': inventory_final_numeric['Warehouse'],
                'Inventory SKU': inventory_final_numeric['Inventory SKU'],
                'Shopify SKU': inventory_final_numeric['Shopify SKU'] if 'Shopify SKU' in inventory_final_numeric.columns else '',
                'Current Balance': inventory_final_numeric['Current Balance'],
                'Is Bundle Component': inventory_final_numeric['Is Bundle Component'] if 'Is Bundle Component' in inventory_final_numeric.columns else False
            })
            
            # Add tracking key for verification and debugging
            inventory_after_processing['tracking_key'] = inventory_after_processing['Warehouse'] + '|' + inventory_after_processing['Inventory SKU']
            unique_combinations = inventory_after_processing['tracking_key'].nunique()
            logger.info(f"Final inventory has {len(inventory_after_processing)} rows with {unique_combinations} unique SKU-warehouse combinations")
            
            # Save this formatted DataFrame
            inventory_after_processing.to_csv(final_inventory_file, index=False)
            logger.info(f"Saved final inventory state to {final_inventory_file} with {len(inventory_after_processing)} entries")
        else:
            # Fallback to saving the original DataFrame if columns are missing
            inventory_final_numeric.to_csv(final_inventory_file, index=False)
            logger.info(f"Saved final inventory state to {final_inventory_file} using original format")
        
        # Calculate and show inventory changes
        if not inventory_summary_initial.empty and not inventory_final_numeric.empty:
            print(f"\n📊 Inventory Changes Summary:")
            
            # Compare total balances by warehouse
            if 'Warehouse' in inventory_final_numeric.columns:
                final_warehouse_summary = inventory_final_numeric.groupby('Warehouse').agg({
                    'Inventory SKU': 'count',
                    'Current Balance': 'sum'
                })
                
                print(f"   Final inventory by warehouse:")
                for warehouse, data in final_warehouse_summary.iterrows():
                    total_balance = data['Current Balance']
                    sku_count = data['Inventory SKU']
                    print(f"     • {warehouse}: {sku_count} SKUs, {total_balance:,.0f} total balance")
                
                # Calculate changes
                print(f"\n   Changes (Final - Initial):")
                for warehouse in initial_warehouse_summary.index:
                    if warehouse in final_warehouse_summary.index:
                        initial_balance = initial_warehouse_summary.loc[warehouse, 'Current Balance']
                        final_balance = final_warehouse_summary.loc[warehouse, 'Current Balance']
                        change = final_balance - initial_balance
                        print(f"     • {warehouse}: {change:+,.0f} balance change")
                    else:
                        print(f"     • {warehouse}: Not found in final inventory")
            
            # Save detailed comparison
            comparison_file = f"inventory_comparison_detailed_{timestamp}.csv"
            
            # Instead of using the inventory_summary_initial and inventory_final_numeric,
            # use the regular inventory comparison file as the source of truth
            regular_comparison_file = f"inventory_comparison_{timestamp}.csv"
            skip_detailed_comparison = False
            
            if os.path.exists(regular_comparison_file):
                try:
                    # Load the regular comparison file which has the correct values
                    regular_comparison = pd.read_csv(regular_comparison_file)
                    
                    # Create a detailed comparison from the regular comparison
                    detailed_comparison = pd.DataFrame({
                        'Warehouse': regular_comparison['Warehouse'],
                        'Inventory SKU': regular_comparison['SKU'],
                        'Initial_Balance': pd.to_numeric(regular_comparison['Initial Balance'], errors='coerce'),
                        'Final_Balance': pd.to_numeric(regular_comparison['Current Balance'], errors='coerce')
                    })
                    
                    # Calculate Balance_Change directly from the regular comparison file's Difference column
                    # This ensures consistency between the two files
                    detailed_comparison['Balance_Change'] = pd.to_numeric(regular_comparison['Difference'], errors='coerce')
                    
                    logger.info(f"Created detailed comparison from regular comparison file with {len(detailed_comparison)} entries")
                    
                    # Skip the rest of the detailed comparison generation since we've already created it
                    # from the regular comparison file
                    logger.info("Using values directly from regular comparison file for detailed comparison")
                    
                    # Save the detailed comparison to a file
                    detailed_comparison.to_csv(comparison_file, index=False)
                    logger.info(f"Saved detailed inventory comparison to {comparison_file}")
                    
                    # Set a flag to skip the rest of the detailed comparison generation
                    skip_detailed_comparison = True
                    
                except Exception as e:
                    logger.error(f"Error loading regular comparison file: {e}")
                    skip_detailed_comparison = False
                    
                    # Fall back to the old method if there's an error
                    # Merge initial and final inventories for detailed comparison
                    initial_merge = inventory_summary_initial.rename(columns={'Current Balance': 'Initial_Balance'})
                    final_merge = inventory_final_numeric.rename(columns={'Current Balance': 'Final_Balance'})
                    
                    # Convert string balances to numeric before merging
                    for df in [initial_merge, final_merge]:
                        if 'Initial_Balance' in df.columns:
                            df['Initial_Balance'] = pd.to_numeric(df['Initial_Balance'].astype(str).str.replace(',', ''), errors='coerce')
                        if 'Final_Balance' in df.columns:
                            df['Final_Balance'] = pd.to_numeric(df['Final_Balance'].astype(str).str.replace(',', ''), errors='coerce')
                    
                    # Merge on both Warehouse and Inventory SKU to ensure we're comparing the same items in the same warehouse
                    detailed_comparison = pd.merge(
                        initial_merge[['Warehouse', 'Inventory SKU', 'Initial_Balance']], 
                        final_merge[['Warehouse', 'Inventory SKU', 'Final_Balance']], 
                        on=['Warehouse', 'Inventory SKU'], 
                        how='outer'
                    )
                    
                    # For items that exist in initial but not final inventory, preserve the initial balance
                    # This prevents items that weren't consumed from showing as 0 in final balance
                    detailed_comparison['Final_Balance'] = detailed_comparison.apply(
                        lambda row: row['Initial_Balance'] if pd.isna(row['Final_Balance']) else row['Final_Balance'],
                        axis=1
                    )
                    
                    # For items that exist in final but not initial inventory, set initial balance to 0
                    detailed_comparison['Initial_Balance'] = detailed_comparison['Initial_Balance'].fillna(0)
            else:
                # If the regular comparison file doesn't exist, use the old method
                skip_detailed_comparison = False
                # Merge initial and final inventories for detailed comparison
                initial_merge = inventory_summary_initial.rename(columns={'Current Balance': 'Initial_Balance'})
                final_merge = inventory_final_numeric.rename(columns={'Current Balance': 'Final_Balance'})
                
                # Convert string balances to numeric before merging
                for df in [initial_merge, final_merge]:
                    if 'Initial_Balance' in df.columns:
                        df['Initial_Balance'] = pd.to_numeric(df['Initial_Balance'].astype(str).str.replace(',', ''), errors='coerce')
                    if 'Final_Balance' in df.columns:
                        df['Final_Balance'] = pd.to_numeric(df['Final_Balance'].astype(str).str.replace(',', ''), errors='coerce')
                
                # Merge on both Warehouse and Inventory SKU to ensure we're comparing the same items in the same warehouse
                detailed_comparison = pd.merge(
                    initial_merge[['Warehouse', 'Inventory SKU', 'Initial_Balance']], 
                    final_merge[['Warehouse', 'Inventory SKU', 'Final_Balance']], 
                    on=['Warehouse', 'Inventory SKU'], 
                    how='outer'
                )
                
                # For items that exist in initial but not final inventory, preserve the initial balance
                # This prevents items that weren't consumed from showing as 0 in final balance
                detailed_comparison['Final_Balance'] = detailed_comparison.apply(
                    lambda row: row['Initial_Balance'] if pd.isna(row['Final_Balance']) else row['Final_Balance'],
                    axis=1
                )
                
                # For items that exist in final but not initial inventory, set initial balance to 0
                detailed_comparison['Initial_Balance'] = detailed_comparison['Initial_Balance'].fillna(0)
            
            detailed_comparison['Balance_Change'] = detailed_comparison['Final_Balance'] - detailed_comparison['Initial_Balance']
            
            # Only show items with changes
            changed_items = detailed_comparison[detailed_comparison['Balance_Change'] != 0]
            if not changed_items.empty:
                detailed_comparison.to_csv(comparison_file, index=False)
                print(f"📋 Saved detailed comparison to: {comparison_file}")
                print(f"   Items with balance changes: {len(changed_items)}")
            else:
                print(f"   No balance changes detected in any inventory items")

        # Save shortage summary to a separate file if there are any shortages
        if shortage_summary and len(shortage_summary) > 0:
            shortage_output_path = "shortage_summary.csv"
            # Convert shortage list to DataFrame if needed
            if isinstance(shortage_summary, list):
                shortage_df = pd.DataFrame(shortage_summary)
            else:
                shortage_df = shortage_summary
            shortage_df.to_csv(shortage_output_path, index=False)
            logger.info(
                f"Generated shortage summary with {len(shortage_df)} items and saved to {shortage_output_path}"
            )
            print(
                f"\nWARNING: {len(shortage_df)} inventory shortages detected! See {shortage_output_path} for details."
            )
