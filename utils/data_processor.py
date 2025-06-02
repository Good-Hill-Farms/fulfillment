import os
import sys

import pandas as pd
import streamlit as st

# Add project root to path to allow importing constants
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from constants.shipping_zones import load_shipping_zones


class DataProcessor:
    """
    Handles data loading, processing, and transformation for the fulfillment application.
    """

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
        """Load SKU mappings from CSV files for Oxnard and Wheeling fulfillment centers

        Returns:
            dict: Dictionary of SKU mappings by fulfillment center
        """
        try:
            mappings = {}

            # Define paths to mapping files
            oxnard_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "docs", "sku_shopify_to_oxnard.csv"
            )
            wheeling_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "docs", "sku_shopify_to_wheeling.csv"
            )

            # Load Oxnard mappings
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
                print(f"Loaded {len(oxnard_map)} SKU mappings for Oxnard")
            else:
                print(f"Warning: Oxnard SKU mapping file not found at {oxnard_path}")
                mappings["Oxnard"] = {}

            # Load Wheeling mappings
            if os.path.exists(wheeling_path):
                # Print file existence confirmation
                print(f"Found Wheeling SKU mapping file at {wheeling_path}")

                # Try loading with different approaches to handle potential CSV format issues
                try:
                    # First attempt: skip first row which might be a duplicate header
                    wheeling_df = pd.read_csv(wheeling_path, skiprows=1)
                    print(
                        f"Successfully loaded Wheeling SKU file with skiprows=1, found {len(wheeling_df)} rows"
                    )
                except Exception as e1:
                    print(f"Error loading Wheeling SKU file with skiprows=1: {e1}")
                    try:
                        # Second attempt: load without skipping rows
                        wheeling_df = pd.read_csv(wheeling_path)
                        print(
                            f"Successfully loaded Wheeling SKU file without skiprows, found {len(wheeling_df)} rows"
                        )
                    except Exception as e2:
                        print(f"Error loading Wheeling SKU file without skiprows: {e2}")
                        # If both attempts fail, create an empty DataFrame
                        wheeling_df = pd.DataFrame(columns=["shopifysku2", "picklist sku"])
                        print("Created empty DataFrame for Wheeling SKUs due to loading errors")

                # Show sample of the data to verify structure
                if not wheeling_df.empty:
                    print(f"Wheeling SKU file columns: {wheeling_df.columns.tolist()}")
                    print("Sample of first 3 Wheeling SKU mappings:")
                    for i, row in wheeling_df.head(3).iterrows():
                        print(f"  {row['shopifysku2']} -> {row['picklist sku']}")

                # Clean up columns by stripping whitespace
                wheeling_df["shopifysku2"] = wheeling_df["shopifysku2"].astype(str).str.strip()
                wheeling_df["picklist sku"] = wheeling_df["picklist sku"].astype(str).str.strip()

                # Filter out empty or NaN picklist SKUs
                wheeling_df_filtered = wheeling_df[
                    wheeling_df["picklist sku"].notna()
                    & (wheeling_df["picklist sku"] != "nan")
                    & (wheeling_df["picklist sku"] != "")
                ]

                print(
                    f"After filtering, Wheeling SKU mappings reduced from {len(wheeling_df)} to {len(wheeling_df_filtered)} rows"
                )

                # Create mapping from shopify SKU to picklist SKU
                wheeling_map = dict(
                    zip(wheeling_df_filtered["shopifysku2"], wheeling_df_filtered["picklist sku"])
                )
                mappings["Wheeling"] = wheeling_map

                # Print sample of mappings to verify
                print(f"Loaded {len(wheeling_map)} SKU mappings for Wheeling")
                print("Sample of Wheeling mappings:")
                sample_keys = list(wheeling_map.keys())[:5]
                for key in sample_keys:
                    print(f"  {key} -> {wheeling_map[key]}")

                # Check for specific SKUs that should be mapped
                test_skus = ["f.loquat-5lb", "f.loquat-2lb", "f.avocado_hass-2lb"]
                for test_sku in test_skus:
                    if test_sku in wheeling_map:
                        print(f"Test SKU '{test_sku}' is mapped to '{wheeling_map[test_sku]}'")
                    else:
                        print(f"Test SKU '{test_sku}' is NOT found in Wheeling mappings")
            else:
                print(f"Warning: Wheeling SKU mapping file not found at {wheeling_path}")
                mappings["Wheeling"] = {}

            return mappings

        except Exception as e:
            print(f"Error loading SKU mappings: {str(e)}")
            return {"Oxnard": {}, "Wheeling": {}}

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
                    print(
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
                print("Detected complex inventory file format with multiple headers")
            else:
                # Try standard CSV reading
                df = pd.read_csv(file_path)

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
                    print(f"Warning: Required column '{req_col}' not found in inventory file")
                    # Create empty column if missing
                    df[req_col] = None

            # Ensure SKU column is properly formatted
            if "Sku" in df.columns:
                # Convert SKU to string type to ensure consistent matching
                df["Sku"] = df["Sku"].astype(str).str.strip()

            # Ensure WarehouseName is properly formatted
            if "WarehouseName" in df.columns:
                # Fill any missing warehouse names with 'Moorpark' as default
                df["WarehouseName"] = df["WarehouseName"].fillna("Moorpark")
                # Standardize warehouse names
                df["WarehouseName"] = df["WarehouseName"].apply(
                    lambda x: "Oxnard"
                    if "moorpark" in str(x).lower()
                    or "oxnard" in str(x).lower()
                    or "ca" in str(x).lower()
                    else (
                        "Wheeling" if "wheeling" in str(x).lower() or "il" in str(x).lower() else x
                    )
                )
                # Print unique warehouse names after standardization for debugging
                unique_warehouses = df["WarehouseName"].unique()
                print(f"Unique warehouse names after standardization: {unique_warehouses}")

            # Convert numeric columns with error handling
            if "AvailableQty" in df.columns:
                # Handle commas in numeric values
                if df["AvailableQty"].dtype == object:
                    df["AvailableQty"] = df["AvailableQty"].astype(str).str.replace(",", "")
                df["AvailableQty"] = pd.to_numeric(df["AvailableQty"], errors="coerce").fillna(0)

            # Handle Balance column which might have commas in numbers
            if "Balance" in df.columns:
                if df["Balance"].dtype == object:
                    df["Balance"] = df["Balance"].astype(str).str.replace(",", "")
                df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce").fillna(0)

            # Convert DaysOnHand to numeric
            if "DaysOnHand" in df.columns:
                df["DaysOnHand"] = pd.to_numeric(df["DaysOnHand"], errors="coerce").fillna(0)

            # Clean up warehouse names and standardize SKU format
            # Filter for only SellableIndividual items if Type column exists
            if "Type" in df.columns:
                # Print unique types for debugging
                unique_types = df["Type"].unique()
                print(f"Unique item types in inventory: {unique_types}")

                # Filter for fruit items (SellableIndividual) to exclude packaging materials
                sellable_df = df[df["Type"] == "SellableIndividual"].copy()
                if not sellable_df.empty:
                    print(
                        f"Filtered inventory to {len(sellable_df)} sellable items out of {len(df)} total items"
                    )
                    df = sellable_df
                else:
                    print("Warning: No SellableIndividual items found in inventory")

            # Print inventory summary by warehouse
            if "WarehouseName" in df.columns and "AvailableQty" in df.columns:
                inventory_summary = df.groupby("WarehouseName")["AvailableQty"].agg(
                    ["sum", "count"]
                )
                print("Inventory summary by warehouse:")
                print(inventory_summary)

            # Print sample of fruit SKUs for debugging
            if "Sku" in df.columns:
                fruit_skus = df[df["Sku"].str.contains("_") | df["Sku"].str.contains("-")][
                    "Sku"
                ].unique()[:10]
                print(f"Sample fruit SKUs: {fruit_skus}")
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
                    agg_dict["Balance"] = "first"
                if "Name" in df.columns:
                    agg_dict["Name"] = "first"
                if "Type" in df.columns:
                    agg_dict["Type"] = "first"
                if "DaysOnHand" in df.columns:
                    agg_dict["DaysOnHand"] = "mean"

                if agg_dict:  # Only aggregate if we have columns to aggregate
                    inventory_summary = df.groupby(group_columns).agg(agg_dict).reset_index()
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

            # Print summary statistics
            print(
                f"Loaded inventory with {len(inventory_summary)} unique SKU-warehouse combinations"
            )
            if "WarehouseName" in inventory_summary.columns:
                warehouse_counts = inventory_summary["WarehouseName"].value_counts()
                print(f"Warehouses: {dict(warehouse_counts)}")

            return inventory_summary

        except Exception as e:
            print(f"Error processing inventory file: {str(e)}")
            # Return an empty DataFrame with the required columns as a fallback
            return pd.DataFrame(columns=["WarehouseName", "Sku", "AvailableQty", "Status"])

    def process_orders(self, orders_df, inventory_df, shipping_zones_df=None, sku_mappings=None):
        """
        Process orders, map SKUs to inventory, and determine optimal fulfillment center

        Args:
            orders_df: DataFrame containing order data
            inventory_df: DataFrame containing inventory data
            shipping_zones_df: DataFrame containing shipping zone data
            sku_mappings: Dictionary of SKU mappings

        Returns:
            DataFrame: Processed orders with fulfillment center and inventory information
        """
        # Define the streamlined output columns based on the fulfillment-focused structure
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
            "order id": "externalorderid",
            "Name": "ordernumber",
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

        # Process each order
        for _, row in orders_df.iterrows():
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

            if shopify_sku is None:
                print(f"Warning: No SKU found for order {row.get('externalorderid', 'unknown')}")
                continue

            # Determine fulfillment center for this order
            fulfillment_center = row.get("Fulfillment Center", "Moorpark")

            # Map Shopify SKU to warehouse SKU using the mappings
            warehouse_sku = shopify_sku

            # Convert fulfillment center name to match the keys in sku_mappings
            # Standardize fulfillment center names to match mapping keys
            if fulfillment_center:
                if (
                    "moorpark" in fulfillment_center.lower()
                    or "oxnard" in fulfillment_center.lower()
                    or "ca" in fulfillment_center.lower()
                ):
                    fc_key = "Oxnard"
                elif "wheeling" in fulfillment_center.lower() or "il" in fulfillment_center.lower():
                    fc_key = "Wheeling"
                else:
                    # Default to Oxnard if unknown
                    fc_key = "Oxnard"
                    print(
                        f"Warning: Unknown fulfillment center '{fulfillment_center}', defaulting to Oxnard"
                    )
            else:
                # Default to Oxnard if None
                fc_key = "Oxnard"
                print("Warning: No fulfillment center specified, defaulting to Oxnard")

            # Apply SKU mapping if available
            if sku_mappings and fc_key in sku_mappings:
                # Get the mapping for this fulfillment center
                fc_mappings = sku_mappings[fc_key]

                # If this SKU has a mapping, use it
                if (
                    shopify_sku in fc_mappings
                    and fc_mappings[shopify_sku]
                    and fc_mappings[shopify_sku] != "nan"
                ):
                    warehouse_sku = fc_mappings[shopify_sku]
                    print(
                        f"Mapped Shopify SKU '{shopify_sku}' to warehouse SKU '{warehouse_sku}' for {fc_key}"
                    )
                else:
                    print(
                        f"No SKU mapping found for '{shopify_sku}' in {fc_key}, using original SKU"
                    )
            else:
                print(f"No SKU mappings available for {fc_key}, using original SKU")

            # Use the warehouse SKU for inventory matching
            sku = warehouse_sku

            # Map SKU to inventory SKU
            inventory_sku = self._map_sku_to_inventory(sku, inventory_df)

            # Extract priority tag
            priority = None
            if "Tags" in row and pd.notna(row["Tags"]):
                tags = str(row["Tags"]).upper()
                if "P1" in tags or "P*" in tags:
                    priority = "P1"

            # Extract ZIP code for shipping zone lookup
            zip_code = None
            if "shiptopostalcode" in row and pd.notna(row["shiptopostalcode"]):
                zip_code = str(row["shiptopostalcode"])
                # Extract first 3 digits for zone lookup
                if len(zip_code) >= 3:
                    zip_code = zip_code[:3]

            # Get optimal fulfillment center
            fulfillment_info = self.get_optimal_fulfillment_center(
                zip_code, shipping_zones_df, priority, inventory_df, inventory_sku
            )

            # Create streamlined output row with only the necessary fields
            output_row = {col: "" for col in output_columns}  # Initialize with empty strings

            # Map basic order information
            for field in [
                "externalorderid",
                "ordernumber",
                "CustomerFirstName",
                "customerLastname",
                "customeremail",
                "shiptoname",
                "shiptostreet1",
                "shiptostreet2",
                "shiptocity",
                "shiptostate",
                "shiptopostalcode",
                "note",
                "Tags",
                "MAX PKG NUM",
            ]:
                if field in row:
                    output_row[field] = row[field]

            # Format date fields
            if "placeddate" in row and pd.notna(row["placeddate"]):
                try:
                    # Try to parse the date and format it as MM/DD/YYYY
                    placed_date = pd.to_datetime(row["placeddate"])
                    output_row["placeddate"] = placed_date.strftime("%m/%d/%Y")
                except:
                    output_row["placeddate"] = row[
                        "placeddate"
                    ]  # Keep original format if parsing fails

            # Add SKU information
            output_row["shopsku"] = sku  # Original SKU
            output_row["shopifysku2"] = sku  # Also add as shopifysku2 for compatibility
            output_row["sku"] = inventory_sku  # Mapped inventory SKU

            # Add quantity information
            if "shopquantity" in row and pd.notna(row["shopquantity"]):
                output_row["shopquantity"] = row["shopquantity"]
                output_row["quantity"] = row[
                    "shopquantity"
                ]  # Also add as quantity for compatibility

            if "externalid" in row and pd.notna(row["externalid"]):
                output_row["externalid"] = row["externalid"]

            # Add fulfillment center
            output_row["Fulfillment Center"] = fulfillment_info["fulfillment_center"]

            # Add inventory quantities
            selected_fc = (
                "moorpark"
                if "moorpark" in fulfillment_info["fulfillment_center"].lower()
                else "wheeling"
            )

            # Set actual quantity
            if "available_qty" in fulfillment_info:
                output_row["actualqty"] = fulfillment_info["available_qty"]
            else:
                output_row["actualqty"] = 0

            # Calculate pick weight (simplified estimation based on SKU type)
            # This is a placeholder - you may want to implement a more accurate weight calculation
            weight_per_unit = 0.5625  # Default weight per unit in lbs
            if "shopquantity" in row and pd.notna(row["shopquantity"]):
                try:
                    quantity = float(row["shopquantity"])
                    output_row["Total Pick Weight"] = quantity * weight_per_unit
                except (ValueError, TypeError):
                    output_row["Total Pick Weight"] = weight_per_unit

            # Get detailed inventory information
            if (
                "moorpark_inventory" in fulfillment_info
                and "wheeling_inventory" in fulfillment_info
            ):
                fc_inventory = fulfillment_info[f"{selected_fc}_inventory"]
                if isinstance(fc_inventory, dict):
                    # Get quantity from shopquantity if available
                    qty = 1  # Default quantity
                    if "shopquantity" in row and pd.notna(row["shopquantity"]):
                        try:
                            qty = int(float(row["shopquantity"]))
                        except (ValueError, TypeError):
                            pass

                    # Set inventory balances
                    output_row["Starting Balance"] = fc_inventory.get("starting_balance", 0)
                    output_row["Transaction Quantity"] = qty  # Use actual order quantity
                    output_row["Ending Balance"] = fc_inventory.get("starting_balance", 0) - qty
                else:
                    # Fallback if inventory structure is not as expected
                    output_row["Starting Balance"] = 0
                    output_row["Transaction Quantity"] = 0
                    output_row["Ending Balance"] = 0

            # Add warnings for SKU not found or inventory issues
            warnings = []
            if inventory_sku == "":
                warnings.append("SKU not found in inventory")
            elif "inventory_status" in fulfillment_info:
                if fulfillment_info["inventory_status"] == "Out of Stock":
                    warnings.append("Item out of stock")
                elif fulfillment_info["inventory_status"] == "Low Stock":
                    warnings.append("Item low in stock")
                elif fulfillment_info["inventory_status"] == "Critical Stock":
                    warnings.append("Item critically low in stock")
            output_row["Issues"] = "; ".join(warnings) if warnings else ""

            # Add row to output dataframe
            output_df = pd.concat([output_df, pd.DataFrame([output_row])], ignore_index=True)

        # Ensure all columns are in the right order
        for col in output_columns:
            if col not in output_df.columns:
                output_df[col] = ""

        return output_df[output_columns]  # Return only the specified columns in the correct order

    def _map_sku_to_inventory(self, sku, inventory_df, sku_mapping=None):
        """Map order SKU to inventory SKU using various matching strategies.

        Args:
            sku: SKU from order (already mapped from Shopify to warehouse SKU if mapping exists)
            inventory_df: Inventory dataframe
            sku_mapping: Optional dictionary mapping order SKUs to inventory SKUs

        Returns:
            str: Matching inventory SKU or empty string if not found
        """
        # Check if inventory_df is valid
        if inventory_df is None or inventory_df.empty:
            print("Warning: Empty inventory DataFrame")
            return ""

        # Ensure sku is a string and strip whitespace
        sku = str(sku).strip() if sku is not None else ""

        # Check if required columns exist with case-insensitive matching
        sku_column = None
        for col in inventory_df.columns:
            if col.lower() == "sku":
                sku_column = col
                break

        if sku_column is None:
            print("Warning: 'Sku' column not found in inventory DataFrame")
            return ""

        # Debug output
        print(f"Looking for SKU '{sku}' in inventory")
        print(
            f"Inventory has {len(inventory_df)} rows with {len(inventory_df[sku_column].unique())} unique SKUs"
        )

        # Ensure all inventory SKUs are strings and stripped
        inventory_df[sku_column] = inventory_df[sku_column].astype(str).str.strip()

        # Print some sample SKUs from inventory for debugging
        sample_skus = inventory_df[sku_column].unique()[:10]
        print(f"Sample SKUs in inventory: {', '.join(sample_skus)}")

        # Try exact match first
        matching_rows = inventory_df[inventory_df[sku_column] == sku]

        if not matching_rows.empty:
            # Return the first matching SKU
            matched_sku = matching_rows.iloc[0][sku_column]
            print(f"Found exact match for '{sku}': '{matched_sku}'")
            return matched_sku

        # If no exact match, try removing 'f.' prefix if present
        if sku.startswith("f."):
            sku_no_prefix = sku[2:]
            matching_rows = inventory_df[inventory_df[sku_column] == sku_no_prefix]
            if not matching_rows.empty:
                matched_sku = matching_rows.iloc[0][sku_column]
                print(f"Found match after removing 'f.' prefix: '{sku}' -> '{matched_sku}'")
                return matched_sku

        # Try case-insensitive match
        sku_lower = sku.lower()
        for inv_sku in inventory_df[sku_column].unique():
            if inv_sku.lower() == sku_lower:
                print(f"Found case-insensitive match: '{sku}' -> '{inv_sku}'")
                return inv_sku

        # If we have a mapping, try using it
        if sku_mapping and sku in sku_mapping:
            mapped_sku = sku_mapping[sku]
            print(f"Using mapped SKU: {sku} -> {mapped_sku}")

            # Check if the mapped SKU exists in inventory
            matching_rows = inventory_df[inventory_df[sku_column] == mapped_sku]
            if not matching_rows.empty:
                print(f"Found mapped SKU in inventory: {mapped_sku}")
                return mapped_sku
            else:
                # Try case-insensitive match with mapped SKU
                mapped_sku_lower = mapped_sku.lower()
                for inv_sku in inventory_df[sku_column].unique():
                    if inv_sku.lower() == mapped_sku_lower:
                        print(
                            f"Found case-insensitive match for mapped SKU: '{mapped_sku}' -> '{inv_sku}'"
                        )
                        return inv_sku

                print(f"Warning: Mapped SKU {mapped_sku} not found in inventory")
                return mapped_sku  # Return it anyway, as it's the best match we have

        # Try more advanced matching strategies
        try:
            # Try to find a match by removing special characters
            clean_sku = "".join(c for c in sku if c.isalnum() or c == "_" or c == "-").lower()

            for inv_sku in inventory_df[sku_column].unique():
                clean_inv_sku = "".join(
                    c for c in str(inv_sku) if c.isalnum() or c == "_" or c == "-"
                ).lower()
                if clean_sku == clean_inv_sku:
                    print(f"Found match after cleaning special characters: '{sku}' -> '{inv_sku}'")
                    return inv_sku

            # Try matching by fruit base name
            # Extract the base fruit name for matching (e.g., 'mango' from 'mango_honey-09x16')
            if "_" in sku or "-" in sku:
                base_name = sku.split("-")[0].split("_")[0].lower()
                print(f"Trying to match by base fruit name: '{base_name}'")

                matching_skus = []
                for inv_sku in inventory_df[sku_column].unique():
                    inv_sku_str = str(inv_sku)
                    if "_" in inv_sku_str or "-" in inv_sku_str:
                        inv_base = inv_sku_str.split("-")[0].split("_")[0].lower()
                        if base_name == inv_base:
                            matching_skus.append(inv_sku)

                if matching_skus:
                    best_match = matching_skus[0]  # Take the first match
                    print(f"Found match by base fruit name: '{sku}' -> '{best_match}'")
                    return best_match

            # Try fuzzy matching on product name if available
            if "Name" in inventory_df.columns:
                product_name = sku.replace("_", " ").replace("-", " ").lower()
                print(f"Trying to match by product name: '{product_name}'")

                for idx, row in inventory_df.iterrows():
                    row_name = str(row["Name"]).lower()
                    if product_name in row_name or any(
                        word in row_name for word in product_name.split() if len(word) > 3
                    ):
                        print(f"Found match by product name: '{sku}' -> '{row[sku_column]}'")
                        return row[sku_column]
        except Exception as e:
            print(f"Error in advanced SKU matching for {sku}: {str(e)}")

        print(f"No match found for SKU: '{sku}'")
        return ""

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
                print(f"Error loading shipping zones from default location: {e}")
                shipping_zones_df = pd.DataFrame()
        elif isinstance(shipping_zones_df, str):
            try:
                shipping_zones_df = load_shipping_zones(shipping_zones_df)
            except Exception as e:
                print(f"Error loading shipping zones from file: {e}")
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

        result = {
            "fulfillment_center": formatted_fc,
            "zone": zone,
            "estimated_delivery_days": estimated_days,
            "inventory_available": inventory_availability[optimal_center]["available"],
            "inventory_status": inventory_status,
            "available_qty": inventory_availability[optimal_center]["qty"],
            "moorpark_inventory": inventory_availability["moorpark"],
            "wheeling_inventory": inventory_availability["wheeling"],
        }

        return result


if __name__ == "__main__":
    data_processor = DataProcessor()
    shipping_zones_df = load_shipping_zones("docs/shipping_zones.csv")
    orders_df = data_processor.load_orders("docs/orders.csv")
    inventory_df = data_processor.load_inventory("docs/inventory.csv")
    sku_mappings = data_processor.load_sku_mappings()

    if not orders_df.empty:
        processed_orders = data_processor.process_orders(
            orders_df, inventory_df, shipping_zones_df, sku_mappings
        )
        processed_orders.to_csv("output.csv", index=False)
        print(f"Processed {len(processed_orders)} orders and saved to output.csv")
