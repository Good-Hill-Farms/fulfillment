import json
import logging
import os
from typing import Any, Dict, List

import pandas as pd
from google.auth import default
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.pickle.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# GHF Inventory Table
GHF_INVENTORY_ID = "19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY"
PIECES_VS_LB_CONVERSION_SHEET_NAME = "INPUT_picklist_sku"

# GHF Aggregation Dashboard Table
GHF_AGGREGATION_DASHBOARD_ID = "1CdTTV8pMqq_wS9vu0qa8HMykNkqtOverrIsP0WLSUeM"
AGG_ORDERS_SHEET_NAME = "Agg_Orders"
ALL_PICKLIST_V2_SHEET_NAME = "ALL_Picklist_V2"  # Fixed capitalization to match actual sheet name
ALL_PICKLIST_V2_NEEDED_COLUMNS = [
    "A",  # Product Type
    "N", "Q", "T",  # Weight related metrics
    "W", "Z",  # Inventory and Confirmed Aggregation
    # Oxnard 1 (OX1)
    "AL", "AM", "AN", "AO",  # Projection, Weight, Inventory + Confirmed Agg, Needs
    # Wheeling 1 (WH1)
    "AT", "AU", "AV", "AW",  # Projection, Weight, Inventory + Confirmed Agg, Needs
    # Oxnard 2 (OX2)
    "BC", "BE", "BF", "BH",  # Weight, Inventory + Confirmed Agg, Inventory, Needs
    # Wheeling 2 (WH2)
    "BM", "BO", "BP", "BR"   # Weight, Inventory + Confirmed Agg, Inventory, Needs
]

"""
A: Product Type - Type of product being tracked
N, Q, T: Weight related metrics - Product weight measurements
W, Z: Inventory and Confirmed Aggregation - Current inventory levels and confirmed aggregated amounts

Oxnard 1 (OX1) - First Projection Period:
AL: OX1: Projection - Projected values for Oxnard 1
AM: OX1: Weight - Weight measurements for Oxnard 1
AN: OX1: Inventory + Confirmed Agg - Combined inventory and confirmed aggregation for Oxnard 1
AO: OX1: Needs - Required inventory needs for Oxnard 1

Wheeling 1 (WH1) - First Projection Period:
AT: WH1: Projection 1 - Projected values for Wheeling 1
AU: WH1: Weight - Weight measurements for Wheeling 1
AV: WH1: Inventory + Confirmed Agg - Combined inventory and confirmed aggregation for Wheeling 1
AW: WH1: Needs - Required inventory needs for Wheeling 1

Oxnard 2 (OX2) - Second Projection Period:
BC: OX2: Weight - Weight measurements for Oxnard 2
BE: OX2: Inventory + Confirmed Agg - Combined inventory and confirmed aggregation for Oxnard 2
BF: OX2: Inventory - Current inventory levels for Oxnard 2
BH: OX2: Needs - Required inventory needs for Oxnard 2

Wheeling 2 (WH2) - Second Projection Period:
BM: WH2: Weight - Weight measurements for Wheeling 2
BO: WH2: Inventory + Confirmed Agg - Combined inventory and confirmed aggregation for Wheeling 2
BP: WH2: Inventory - Current inventory levels for Wheeling 2
BR: WH2: Needs - Required inventory needs for Wheeling 2
"""

# GHF AGG/FRUIT Table
GHF_AGG_FRUIT = "1-lTQJWHutgBM-oN_hYFpgc12WwxxyeZtidvylvSAAWI"
INVENTORY_OXNARD_SHEET = "Inventory_Oxnard"
INVENTORY_WHEELING_SHEET = "Inventory_Wheeling"

# GHF: Fruit Tracking 
GHF_FRUIT_TRACKING = "1B_uRcYEqCdR5O3h5BiyvL92Q1v4BlNPxZTsZ-nihNbI"
ORDERS_NEW_SHEET_NAME = "Orders_new"
ORDERS_NEW_NEDDED_COLUMNS = ["B", "D", "E", "P", "Q"] # invoice date, Aggregator / Vendor, Product Type, Price per lb, Actual Total Cost

def get_credentials():
    """Gets service account credentials using the simplest approach that works."""
    try:
        # First, check if we have a local service account file
        json_file = "nca-toolkit-project-446011-67d246fdbccf.json"
        if os.path.exists(json_file):
            logger.info("Using local service account JSON file")
            creds = service_account.Credentials.from_service_account_file(json_file, scopes=SCOPES)
            return creds
    except Exception as e:
        logger.warning(f"Could not use local JSON file: {e}")

    try:
        # Use Application Default Credentials (works with Cloud Run, gcloud auth, etc.)
        creds, project = default(scopes=SCOPES)
        logger.info(f"Using Application Default Credentials for project: {project}")
        return creds
    except Exception as e:
        logger.error(f"Error getting Application Default Credentials: {e}")
        raise Exception(
            f"Could not authenticate. Make sure you have either the JSON file locally or have run 'gcloud auth application-default login'"
        )


def get_sheet_data(sheet_name: str) -> List[List[Any]]:
    """Gets data from a sheet in the GHF Inventory spreadsheet."""
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=GHF_INVENTORY_ID, range=sheet_name)
            .execute()
        )

        values = result.get("values", [])
        if not values:
            logger.warning(f"No data found in sheet {sheet_name}")
            return []

        return values

    except Exception as e:
        logger.error(f"Error fetching data from sheet {sheet_name}: {e}")
        return []


def get_agg_order_data(sheet_name: str) -> List[List[Any]]:
    """Gets data from a sheet in the GHF Aggregation Dashboard spreadsheet."""
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=GHF_AGGREGATION_DASHBOARD_ID, range=sheet_name)
            .execute()
        )

        values = result.get("values", [])
        if not values:
            logger.warning(f"No data found in sheet {sheet_name}")
            return []

        return values

    except Exception as e:
        logger.error(f"Error fetching data from sheet {sheet_name}: {e}")
        return []


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames duplicate columns by appending a suffix."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df


def load_agg_orders() -> pd.DataFrame | None:
    """Fetches order data from the 'Agg_Orders' Google Sheet."""
    logger.info("Fetching data from Agg_Orders sheet...")
    try:
        values = get_agg_order_data(AGG_ORDERS_SHEET_NAME)
        if not values:
            return None

        headers = values[0]
        data = values[1:]
        df = pd.DataFrame(data, columns=headers)
        df = deduplicate_columns(df)
        return df

    except Exception as e:
        logger.error(f"Failed to load Agg_Orders sheet: {e}")
        return None


def _safe_float_convert(value, default=0.0):
    """
    Safely convert a value to float, handling '#VALUE!' and other invalid cases.

    Args:
        value: The value to convert
        default: Default value to return if conversion fails

    Returns:
        float: The converted value or default if conversion fails
    """
    if not value or not str(value).strip():
        return default
    try:
        # Handle '#VALUE!' and similar Excel error values
        if isinstance(value, str) and value.startswith("#"):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def process_sku_data(data: List[List[Any]], center: str) -> Dict[str, Dict[str, Any]]:
    """Process raw sheet data into the required format compatible with data_processor.py."""
    if not data:
        logger.warning(f"No data to process for {center}")
        return {center: {"singles": {}, "bundles": {}}}

    # Get headers and rows
    headers = data[0]
    rows = data[1:]
    
    
    # Print first 5 rows to see the actual data
    for i, row in enumerate(rows[:5]):
        print(f"Row {i+2}: {row}")
    
    # Initialize result structure
    result = {center: {"singles": {}, "bundles": {}}}

    # Group rows by shopifysku2 to handle bundles
    sku_groups = {}
    for row_idx, row in enumerate(rows, start=2):
        try:
            mapping = dict(zip(headers, row))
            if "shopifysku2" not in mapping or not mapping["shopifysku2"]:
                continue

            order_sku = mapping["shopifysku2"]
            
            if order_sku not in sku_groups:
                sku_groups[order_sku] = []
            sku_groups[order_sku].append((row_idx, mapping))
        except Exception as e:
            logger.error(f"Error grouping row {row_idx}: {e}")
            continue

    # Process each SKU group
    for order_sku, group_rows in sku_groups.items():
        try:
            if len(group_rows) > 1:
                # This is a bundle - combine components
                components_list = []
                for row_idx, mapping in group_rows:
                    try:
                        mix_qty = _safe_float_convert(mapping.get("Mix Quantity", 1), default=1)
                        pick_weight = _safe_float_convert(mapping.get("Pick Weight LB", 0))

                        component_data = {
                            "component_sku": mapping.get("picklist sku", ""),
                            "actualqty": mix_qty,
                            "weight": pick_weight,
                            "pick_type": mapping.get("Pick Type", ""),
                            "pick_type_inventory": mapping.get("Product Type", ""),
                        }
                        components_list.append(component_data)
                    except Exception as e:
                        logger.warning(
                            f"Error parsing bundle component for {order_sku} at row {row_idx}: {e}"
                        )

                if components_list:
                    result[center]["bundles"][order_sku] = components_list
                    logger.debug(
                        f"Processed bundle {order_sku} with {len(components_list)} components"
                    )
            else:
                # This is a single SKU
                row_idx, mapping = group_rows[0]
                try:
                    actualqty = _safe_float_convert(mapping.get("actualqty", 1), default=1)
                    total_pick_weight = _safe_float_convert(mapping.get("Total Pick Weight", 0))

                    # Ensure we have a picklist_sku (inventory SKU)
                    picklist_sku = mapping.get("picklist sku", "")
                    if not picklist_sku:
                        logger.warning(
                            f"No picklist sku found for single SKU {order_sku}, using order SKU as fallback"
                        )
                        picklist_sku = order_sku

                    result[center]["singles"][order_sku] = {
                        "picklist_sku": picklist_sku,
                        "actualqty": actualqty,
                        "total_pick_weight": total_pick_weight,
                        "pick_type": mapping.get("Pick Type", ""),
                        "pick_type_inventory": mapping.get("Product Type", ""),
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing single SKU {order_sku} at row {row_idx}: {e}")

        except Exception as e:
            logger.error(f"Error processing SKU group {order_sku}: {e}")
            continue

    logger.info(
        f"Processed {len(result[center]['singles'])} singles and {len(result[center]['bundles'])} bundles for {center}"
    )
    return result


def load_sku_mappings_from_sheets(center=None):
    """
    Load all SKU mappings from Google Sheets with a format compatible with data_processor.py.

    Args:
        center (str, optional): If provided, filter by fulfillment center ("Oxnard" or "Wheeling")
                               If None, load data for all centers

    Returns:
        dict: Dictionary with warehouse names as top-level keys, each containing 'singles' and 'bundles'
              Format: {"Oxnard": {"singles": {...}, "bundles": {...}}, "Wheeling": {"singles": {...}, "bundles": {...}}}
    """
    logger.info("Loading all SKU mappings from Google Sheets...")

    # Initialize output structure
    result = {}

    try:
        # Get all centers or use the specified one
        centers = [center] if center else ["Oxnard", "Wheeling"]

        # Process each center
        for current_center in centers:
            logger.info(f"Processing {current_center} SKU mappings...")

            # Get data from appropriate sheet
            sheet_name = f"INPUT_bundles_cvr_{current_center.lower()}"
            data = get_sheet_data(sheet_name)

            if not data:
                logger.warning(f"No data found for {current_center}")
                result[current_center] = {"singles": {}, "bundles": {}}
                continue

            # Process the data
            center_data = process_sku_data(data, current_center)
            result.update(center_data)

            logger.info(
                f"Loaded {len(result[current_center]['singles'])} simple SKUs and {len(result[current_center]['bundles'])} bundles for {current_center}"
            )

    except Exception as e:
        logger.error(f"Error loading SKU mappings: {e}")
        # Return empty structure if there's an error
        if not result:
            result = {
                "Wheeling": {"singles": {}, "bundles": {}},
                "Oxnard": {"singles": {}, "bundles": {}},
            }

    # Validate the result structure
    for warehouse_name in ["Oxnard", "Wheeling"]:
        if warehouse_name not in result:
            result[warehouse_name] = {"singles": {}, "bundles": {}}
        if "singles" not in result[warehouse_name]:
            result[warehouse_name]["singles"] = {}
        if "bundles" not in result[warehouse_name]:
            result[warehouse_name]["bundles"] = {}

    # Log summary
    total_singles = sum(len(result[warehouse]["singles"]) for warehouse in result)
    total_bundles = sum(len(result[warehouse]["bundles"]) for warehouse in result)
    logger.info(
        f"Total loaded: {total_singles} singles and {total_bundles} bundles across all warehouses"
    )

    return result


def get_fruit_inventory_data(sheet_name: str) -> List[List[Any]]:
    """Fetches data from the fruit inventory Google Sheet."""
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=GHF_AGG_FRUIT, range=sheet_name)
            .execute()
        )

        values = result.get("values", [])
        if not values:
            logger.warning(f"No data found in sheet {sheet_name}")
            return []

        return values

    except Exception as e:
        logger.error(f"Error fetching data from sheet {sheet_name}: {e}")
        return []

def load_inventory_data(sheet_name: str) -> pd.DataFrame | None:
    """Loads inventory data from specified sheet."""
    logger.info(f"Fetching data from {sheet_name} sheet...")
    try:
        values = get_fruit_inventory_data(sheet_name)
        if not values:
            return None

        headers = values[0]
        data = values[1:]
        df = pd.DataFrame(data, columns=headers)
        df = deduplicate_columns(df)
        
        # Convert numeric columns
        if 'Total Weight' in df.columns:
            df['Total Weight'] = pd.to_numeric(df['Total Weight'], errors='coerce').fillna(0)
        
        # Convert date columns
        date_cols = ['INVENTORY DATE', 'FRUIT DATE']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df

    except Exception as e:
        logger.error(f"Failed to load {sheet_name} sheet: {e}")
        return None

def load_oxnard_inventory() -> pd.DataFrame | None:
    """Fetches inventory data from the Inventory_Oxnard sheet."""
    return load_inventory_data(INVENTORY_OXNARD_SHEET)

def load_wheeling_inventory() -> pd.DataFrame | None:
    """Fetches inventory data from the Inventory_Wheeling sheet."""
    return load_inventory_data(INVENTORY_WHEELING_SHEET)

def load_pieces_vs_lb_conversion() -> pd.DataFrame | None:
    """Loads the pieces vs lb conversion data."""
    logger.info("Loading pieces vs lb conversion data...")
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)
        
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=GHF_INVENTORY_ID, range=PIECES_VS_LB_CONVERSION_SHEET_NAME)
            .execute()
        )
        
        values = result.get("values", [])
        if not values:
            logger.warning("No pieces vs lb conversion data found")
            return None
            
        headers = values[0]
        data = values[1:]
        df = pd.DataFrame(data, columns=headers)
        df = deduplicate_columns(df)
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load pieces vs lb conversion data: {e}")
        return None

def load_all_picklist_v2() -> pd.DataFrame | None:
    """Loads the All Picklist V2 data from GHF Aggregation Dashboard."""
    logger.info(f"Loading All Picklist V2 data from GHF Aggregation Dashboard...")
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)
        
        sheet = service.spreadsheets()
        
        # Create range string like "A:BR" to get all needed columns
        last_column = max(ALL_PICKLIST_V2_NEEDED_COLUMNS, key=lambda x: ord(x[0]) if len(x) == 1 else ord(x[0]) * 26 + ord(x[1]))
        range_string = f"{ALL_PICKLIST_V2_SHEET_NAME}!A:{last_column}"
        
        result = (
            sheet.values()
            .get(
                spreadsheetId=GHF_AGGREGATION_DASHBOARD_ID,
                range=range_string
            )
            .execute()
        )
        
        values = result.get("values", [])
        if not values:
            logger.warning(f"No data found in {ALL_PICKLIST_V2_SHEET_NAME}")
            return None
            
        # Find the first row that has the most columns - that's likely our real header row
        max_cols = 0
        header_idx = 0
        for idx, row in enumerate(values[:10]):  # Check first 10 rows
            if len(row) > max_cols:
                max_cols = len(row)
                header_idx = idx
        
        headers = values[header_idx]
        data = values[header_idx + 2:]  # Skip one more row after the header row
        
        # Create a mapping of column letter to index
        col_to_idx = {}
        for i in range(len(headers)):
            letter = chr(65 + i) if i < 26 else chr(64 + i//26) + chr(65 + i%26)
            col_to_idx[letter] = i
            
        # Get indices for the columns we want
        selected_indices = []
        for col in ALL_PICKLIST_V2_NEEDED_COLUMNS:
            if col in col_to_idx:
                selected_indices.append(col_to_idx[col])
            else:
                logger.warning(f"Column {col} not found in the sheet")
        
        # Select only the columns we want
        selected_headers = [headers[i] for i in selected_indices if i < len(headers)]
        selected_data = []
        for row in data:
            # Pad short rows with None
            if len(row) < len(headers):
                row = row + [None] * (len(headers) - len(row))
            # Select only the columns we want
            selected_row = []
            for i in selected_indices:
                value = row[i] if i < len(row) else None
                # Clean up special values
                if value in ['#DIV/0!', '#N/A', '#VALUE!', '']:
                    value = '0'
                elif isinstance(value, str) and '%' in value:
                    # Convert percentage to decimal
                    try:
                        value = str(float(value.replace('%', '')) / 100)
                    except ValueError:
                        value = '0'
                selected_row.append(value)
            selected_data.append(selected_row)
        
        logger.info(f"Found {len(selected_data)} rows with {len(selected_headers)} columns")
        
        df = pd.DataFrame(selected_data, columns=selected_headers)
        
        # Convert numeric columns to float
        numeric_columns = df.columns.difference(['Product Type'])  # All columns except Product Type should be numeric
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Remove empty rows and columns
        df = df.dropna(how='all')  # Drop rows where all values are NA
        df = df.dropna(axis=1, how='all')  # Drop columns where all values are NA
        
        # Filter out rows where Product Type is empty, null, or just whitespace
        df = df[df['Product Type'].notna() & (df['Product Type'].str.strip() != '')]
        
        # Filter out rows where all numeric columns are 0
        numeric_mask = df[numeric_columns].ne(0).any(axis=1)
        df = df[numeric_mask]
        
        logger.info(f"Successfully loaded {len(df)} rows from {ALL_PICKLIST_V2_SHEET_NAME}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load All Picklist V2 data: {str(e)}")
        logger.exception("Full traceback:")
        return None

def get_sku_info(df: pd.DataFrame, product_type: str) -> Dict[str, Any]:
    """
    Get detailed information for a specific product type from the All Picklist V2 data.
    
    Args:
        df (pd.DataFrame): The All Picklist V2 dataframe
        product_type (str): The product type to look up
        
    Returns:
        Dict[str, Any]: Dictionary containing product information with the following structure:
        {
            'product_type': str,
            'weight_metrics': {
                'weight_1': float,  # Column N
                'weight_2': float,  # Column Q
                'weight_3': float   # Column T
            },
            'inventory': {
                'current_level': float,    # Column W
                'confirmed_agg': float     # Column Z
            },
            'oxnard_1': {
                'projection': float,       # Column AL
                'weight': float,           # Column AM
                'inventory_agg': float,    # Column AN
                'needs': float             # Column AO
            },
            'wheeling_1': {
                'projection': float,       # Column AT
                'weight': float,           # Column AU
                'inventory_agg': float,    # Column AV
                'needs': float             # Column AW
            }
        }
    """
    # Find the row for the product type
    product_row = df[df.iloc[:, 0] == product_type]  # Column A is Product Type
    if product_row.empty:
        logger.warning(f"Product Type {product_type} not found in All Picklist V2 data")
        return None
        
    # Helper function to safely convert values to float
    def safe_float(value) -> float:
        try:
            if pd.isna(value) or value == '':
                return 0.0
            if isinstance(value, str):
                # Remove any currency symbols and commas
                value = value.replace('$', '').replace(',', '').strip()
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    # Extract data from the row
    row = product_row.iloc[0]
    
    # Column mapping for reference:
    # A: Product Type
    # N, Q, T: Weight metrics
    # W: Inventory, Z: Confirmed Agg
    # AL-AO: Oxnard 1 data
    # AT-AW: Wheeling 1 data
    
    # Get column indices (0-based)
    col_indices = {
        'N': 13,  # N is the 14th column (0-based)
        'Q': 16,  # Q is the 17th column
        'T': 19,  # T is the 20th column
        'W': 22,  # W is the 23rd column
        'Z': 25,  # Z is the 26th column
        'AL': 37, # AL is the 38th column
        'AM': 38, # AM is the 39th column
        'AN': 39, # AN is the 40th column
        'AO': 40, # AO is the 41st column
        'AT': 45, # AT is the 46th column
        'AU': 46, # AU is the 47th column
        'AV': 47, # AV is the 48th column
        'AW': 48  # AW is the 49th column
    }
    
    return {
        'product_type': str(row.iloc[0]),  # Column A
        'weight_metrics': {
            'weight_1': safe_float(row.iloc[col_indices['N']]),
            'weight_2': safe_float(row.iloc[col_indices['Q']]),
            'weight_3': safe_float(row.iloc[col_indices['T']])
        },
        'inventory': {
            'current_level': safe_float(row.iloc[col_indices['W']]),
            'confirmed_agg': safe_float(row.iloc[col_indices['Z']])
        },
        'oxnard_1': {
            'projection': safe_float(row.iloc[col_indices['AL']]),
            'weight': safe_float(row.iloc[col_indices['AM']]),
            'inventory_agg': safe_float(row.iloc[col_indices['AN']]),
            'needs': safe_float(row.iloc[col_indices['AO']])
        },
        'wheeling_1': {
            'projection': safe_float(row.iloc[col_indices['AT']]),
            'weight': safe_float(row.iloc[col_indices['AU']]),
            'inventory_agg': safe_float(row.iloc[col_indices['AV']]),
            'needs': safe_float(row.iloc[col_indices['AW']])
        }
    }

def get_fruit_tracking_data(sheet_name: str) -> List[List[Any]]:
    """Fetches data from the fruit tracking Google Sheet."""
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=GHF_FRUIT_TRACKING, range=sheet_name)
            .execute()
        )

        values = result.get("values", [])
        if not values:
            logger.warning(f"No data found in sheet {sheet_name}")
            return []

        return values

    except Exception as e:
        logger.error(f"Error fetching data from sheet {sheet_name}: {e}")
        return []

def load_orders_new() -> pd.DataFrame | None:
    """Loads the Orders_new data from GHF Fruit Tracking sheet."""
    logger.info(f"Loading Orders_new data from GHF Fruit Tracking...")
    try:
        values = get_fruit_tracking_data(ORDERS_NEW_SHEET_NAME)
        if not values:
            return None

        # Get all data first
        headers = values[0]
        data = values[1:]
        
        # Extract only needed columns using their letter indices
        needed_indices = [ord(col.upper()) - ord('A') for col in ORDERS_NEW_NEDDED_COLUMNS]
        
        # Extract only the needed columns from both headers and data
        selected_headers = [headers[i] for i in needed_indices if i < len(headers)]
        selected_data = []
        for row in data:
            # Pad row if it's shorter than headers
            if len(row) < len(headers):
                row = row + [''] * (len(headers) - len(row))
            # Extract only needed columns
            selected_row = [row[i] if i < len(row) else '' for i in needed_indices]
            
            # Clean price and cost values
            if len(selected_row) > 3:  # Price column
                price_str = str(selected_row[3]).strip()
                # Remove currency symbols and commas
                price_str = price_str.replace('$', '').replace(',', '')
                # Convert to float, defaulting to 0 if invalid
                try:
                    selected_row[3] = float(price_str) if price_str else 0
                except ValueError:
                    selected_row[3] = 0
                    
            if len(selected_row) > 4:  # Total cost column
                cost_str = str(selected_row[4]).strip()
                # Remove currency symbols and commas
                cost_str = cost_str.replace('$', '').replace(',', '')
                # Convert to float, defaulting to 0 if invalid
                try:
                    selected_row[4] = float(cost_str) if cost_str else 0
                except ValueError:
                    selected_row[4] = 0
                    
            selected_data.append(selected_row)
        
        # Create DataFrame with only the selected columns
        df = pd.DataFrame(selected_data, columns=selected_headers)
        
        # Convert date column (first column) to datetime
        if len(df.columns) > 0:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        
        logger.info(f"Successfully loaded {len(df)} rows from {ORDERS_NEW_SHEET_NAME}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load Orders_new data: {str(e)}")
        logger.exception("Full traceback:")
        return None


if __name__ == "__main__":
    try:
        # Create a directory for the output files if it doesn't exist
        output_dir = "sheet_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # # 1. Load and save SKU mappings
        # sku_mappings = load_sku_mappings_from_sheets()
        # with open(f"{output_dir}/sku_mappings.json", "w") as f:
        #     json.dump(sku_mappings, f, indent=2)
        # print(f"✓ Saved SKU mappings to {output_dir}/sku_mappings.json")

        # # 2. Load and save Agg Orders
        # agg_orders_df = load_agg_orders()
        # if agg_orders_df is not None:
        #     agg_orders_df.to_csv(f"{output_dir}/agg_orders.csv", index=False)
        #     print(f"✓ Saved {len(agg_orders_df)} rows to {output_dir}/agg_orders.csv")

        # 3. Load and save All Picklist V2
        picklist_df = load_all_picklist_v2()
        if picklist_df is not None:
            picklist_df.to_csv(f"{output_dir}/all_picklist_v2.csv", index=False)
            print(f"✓ Saved {len(picklist_df)} rows to {output_dir}/all_picklist_v2.csv")

        # # 4. Load and save Pieces vs LB Conversion
        # conversion_df = load_pieces_vs_lb_conversion()
        # if conversion_df is not None:
        #     conversion_df.to_csv(f"{output_dir}/pieces_vs_lb_conversion.csv", index=False)
        #     print(f"✓ Saved {len(conversion_df)} rows to {output_dir}/pieces_vs_lb_conversion.csv")

        # # 5. Load and save Oxnard Inventory
        # oxnard_inv_df = load_oxnard_inventory()
        # if oxnard_inv_df is not None:
        #     oxnard_inv_df.to_csv(f"{output_dir}/oxnard_inventory.csv", index=False)
        #     print(f"✓ Saved {len(oxnard_inv_df)} rows to {output_dir}/oxnard_inventory.csv")

        # # 6. Load and save Wheeling Inventory
        # wheeling_inv_df = load_wheeling_inventory()
        # if wheeling_inv_df is not None:
        #     wheeling_inv_df.to_csv(f"{output_dir}/wheeling_inventory.csv", index=False)
        #     print(f"✓ Saved {len(wheeling_inv_df)} rows to {output_dir}/wheeling_inventory.csv")

        # # 7. Load and save Orders New from Fruit Tracking
        # orders_new_df = load_orders_new()
        # if orders_new_df is not None:
        #     orders_new_df.to_csv(f"{output_dir}/orders_new.csv", index=False)
        #     print(f"✓ Saved {len(orders_new_df)} rows to {output_dir}/orders_new.csv")

        # print("\nAll data has been saved to the 'sheet_data' directory!")
        # print("Summary of files saved:")
        # print("------------------------")
        # for file in os.listdir(output_dir):
        #     size = os.path.getsize(os.path.join(output_dir, file)) / 1024  # Convert to KB
        #     print(f"{file:<30} {size:.1f} KB")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.exception("Full traceback:")
