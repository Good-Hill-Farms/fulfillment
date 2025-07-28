import json
import logging
import os
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
from google.auth import default
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Set up logging
logging.basicConfig(level=logging.INFO)  # Changed from DEBUG to INFO
logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.pickle.
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",  # Full access to Google Sheets
    "https://www.googleapis.com/auth/drive.file",    # Access to files created by the app
    "https://www.googleapis.com/auth/drive"          # Full access to Google Drive
]

# GHF Inventory Table
GHF_INVENTORY_ID = "19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY"
PIECES_VS_LB_CONVERSION_SHEET_NAME = "INPUT_picklist_sku"
INPUT_SKU_TYPE_SHEET_NAME = "INPUT_SKU_TYPE"
INPUT_SKU_TYPE_NEEDED_COLUMNS = ["A", "B", "I"]
# SKU
# PRODUCT_TYPE
# SKU Helper


# GHF Aggregation Dashboard Table
GHF_AGGREGATION_DASHBOARD_ID = "1CdTTV8pMqq_wS9vu0qa8HMykNkqtOverrIsP0WLSUeM"
AGG_ORDERS_SHEET_NAME = "Agg_Orders"
ALL_PICKLIST_V2_SHEET_NAME = "ALL_Picklist_V2"  # Fixed capitalization to match actual sheet name

# Projection Snapshots Folder
PROJECTION_SNAPSHOTS_FOLDER_ID = "1k5S2YQJ5T_z6QtkB_u7jf-dFYbjhkjhg"  # Projection snapshots folder from user


"""
Full column structure for ALL_Picklist_V2:

A: Product Type
B: Fulfillable Weight
C: Moorpark Fulfillable
D: Wheeling Fulfillable
E: Trailing X Days Order Volume
F: Moorpark Volume
G: Wheeling Volume
H: Total Projection
I: Moorpark Projection
J: Wheeling Projection
K: Padded Weight
L: Moorpark Padded
M: Wheeling Padded
N: Total Weight
O: Moorpark Weight
P: Wheeling Weight
Q: Inventory
R: Moorpark Inventory
S: Wheeling Inventory
T: Confirmed Agg
U: Moorpark Confirmed Agg
V: Wheeling Confirmed Agg
W: Total
X: Moorpark Total
Y: Wheeling Total
Z: Total Needs (LBS)
AA: Moorpark Needs (LBS)
AB: Wheeling Needs (LBS)
AC: Trailing Days
AD: Volume Start Date
AE: Volume End Date
AF: Trailing orders per day
AG: Projected orders per day
AH: Projection. Days
AI: Projection End Date
AJ: Projection Factor
AK: Inventory Adjustment
AL: OX 1: Projection
AM: OX 1: Weight
AN: OX 1: Inventory + Confirmed Agg
AO: OX 1: Needs
AP: Projection. Days
AQ: Projection End Date
AR: Projection Factor
AS: Inventory Adjustment | MP
AT: WH: Projection 1
AU: WH 1: Weight
AV: WH 1: Inventory + Confirmed Agg
AW: WH 1: Needs
AX: OX: Projected orders per day 2
AY: OX: Projection. Days 2
AZ: OX: Projection End Date
BA: OX: Projection Factor
BB: OX: Projection 2
BC: OX 2: Weight
BD: OX 2: Inventory Carry
BE: OX 2: Inventory Adjustment
BF: OX 2: Inventory
BG: OX 2: Inventory + Confirmed Agg
BH: OX 2: Needs
BI: WH: Projected orders per day 2
BJ: WH: Projection. Days 2
BK: WH: Projection End Date
BL: WH: Projection Factor
BM: WH: Projection 2
BN: WH 2: Weight
BO: WH 2: Inventory Carry
BP: WH 2: Inventory Adjustment
BQ: WH 2: Inventory
BR: WH 2: Inventory + Confirmed Agg
BS: WH 2: Needs
"""

# GHF AGG/FRUIT Table
GHF_AGG_FRUIT = "1-lTQJWHutgBM-oN_hYFpgc12WwxxyeZtidvylvSAAWI"
INVENTORY_OXNARD_SHEET = "Inventory_Oxnard"
INVENTORY_WHEELING_SHEET = "Inventory_Wheeling"
WOW_SHEET = "WoW"
"""
So we have A column, that starts from A2
All the next columns contain date range 5/4-5/10, 5/11-5/17
We should get all columns that have date range, parse date and display data.
"""

# GHF: Fruit Tracking 
GHF_FRUIT_TRACKING = "1B_uRcYEqCdR5O3h5BiyvL92Q1v4BlNPxZTsZ-nihNbI"
ORDERS_NEW_SHEET_NAME = "Orders_new"
ORDERS_NEW_NEDDED_COLUMNS = ["B", "D", "E", "P", "Q"] # invoice date, Aggregator / Vendor, Product Type, Price per lb, Actual Total Cost

def parse_date_range(date_range_str: str) -> Tuple[datetime, datetime]:
    """Parse date range string into start and end dates.
    Example: '5/4-5/10' -> (datetime(2024,5,4), datetime(2024,5,10))
    """
    try:
        # Split the range
        if not date_range_str or '-' not in date_range_str:
            return None, None
            
        start_str, end_str = date_range_str.split('-')
        
        # Parse start date
        start_month, start_day = map(int, start_str.strip().split('/'))
        
        # Parse end date
        end_month, end_day = map(int, end_str.strip().split('/'))
        
        # Determine year (assuming we're in current year, adjust if month indicates previous/next year)
        current_year = datetime.now().year
        
        # Handle year transition (e.g., Dec-Jan)
        start_year = current_year
        end_year = current_year
        
        # If end month is less than start month, it's a year transition
        if end_month < start_month:
            end_year = current_year + 1
        
        # Create datetime objects
        start_date = datetime(start_year, start_month, start_day)
        end_date = datetime(end_year, end_month, end_day)
        
        return start_date, end_date
    except Exception as e:
        logger.warning(f"Could not parse date range '{date_range_str}': {str(e)}")
        return None, None

def get_credentials():
    """Gets service account credentials using the simplest approach that works."""
    try:
        # First, check if we have a local service account file
        json_file = "nca-toolkit-project-446011-67d246fdbccf.json"
        if os.path.exists(json_file):
            logger.debug("Using local service account JSON file")
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


def load_agg_orders(filter_by_projection_period: bool = False, group_by_projection: bool = False) -> pd.DataFrame | None:
    """
    Fetches order data from the 'Agg_Orders' Google Sheet.
    
    Args:
        filter_by_projection_period: If True, ensures data includes projection period information
        group_by_projection: If True, groups data by projection period
        
    Returns:
        DataFrame with order data, optionally filtered and grouped by projection period
    """
    logger.info("Fetching data from Agg_Orders sheet...")
    try:
        values = get_agg_order_data(AGG_ORDERS_SHEET_NAME)
        if not values:
            return None

        headers = values[0]
        data = values[1:]
        df = pd.DataFrame(data, columns=headers)
        df = deduplicate_columns(df)
        
        if filter_by_projection_period:
            # Convert date columns to datetime with explicit format
            date_columns = ['Date', 'Date_1', 'Date_2']
            for col in date_columns:
                if col in df.columns:
                    # Try common date formats
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y']:
                        try:
                            df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                            break
                        except:
                            continue
            
            # Check if Projection Period exists and handle accordingly
            if 'Projection Period' not in df.columns:
                logger.warning("No Projection Period column found in data")
                return df
            
            # Remove rows where Projection Period is empty or NaN
            df = df[df['Projection Period'].notna()]
            
            # Sort by Date and Projection Period
            sort_cols = ['Projection Period']
            if 'Date' in df.columns:
                sort_cols.insert(0, 'Date')
            df = df.sort_values(by=sort_cols)
            
            # Group by Projection Period if requested
            if group_by_projection:
                df['Projection Period'] = df['Projection Period'].astype(str)
                
                # Define columns to aggregate
                sum_columns = [
                    'Moorpark Needs (LBS)', 'Wheeling Needs (LBS)', 
                    'Oxnard Weight Needed', 'Wheeling Weight Needed',
                    'Oxnard Order', 'Wheeling Order',
                    'Oxnard Actual Order', 'Wheeling Actual Order'
                ]
                
                # Create aggregation dictionary
                agg_dict = {}
                for col in df.columns:
                    if col in sum_columns:
                        agg_dict[col] = 'sum'
                    elif col == 'Date':
                        agg_dict[col] = 'first'
                    else:
                        agg_dict[col] = lambda x: ' | '.join(str(i) for i in x.unique() if pd.notna(i) and str(i).strip())
                
                # Group by Projection Period
                df = df.groupby('Projection Period', as_index=False).agg(agg_dict)
        
        # Add P1 and P2 flags
        if 'Projection Period' in df.columns:
            df['is_p1'] = df['Projection Period'].astype(str).str.contains('1', na=False)
            df['is_p2'] = df['Projection Period'].astype(str).str.contains('2', na=False)
        
        return df

    except Exception as e:
        logger.error(f"Failed to load Agg_Orders sheet: {e}")
        return None


def _safe_float_convert(value, default=0.0):
    """
    Safely convert a value to float, handling Excel/Sheets formula errors and other invalid cases.

    Args:
        value: The value to convert
        default: Default value to return if conversion fails

    Returns:
        float: The converted value or default if conversion fails
    """
    if not value or not str(value).strip():
        return default
    try:
        # Handle all common Excel/Google Sheets formula error values
        str_value = str(value).strip().upper()
        if str_value.startswith("#") or str_value in ['#REF!', '#VALUE!', '#DIV/0!', '#N/A', '#NAME?', '#NULL!', '#NUM!']:
            return default
        
        # Remove common formatting characters
        clean_value = str(value).replace('$', '').replace(',', '').replace('%', '').strip()
        return float(clean_value)
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
    
    # Debug: Print first 5 rows to see the actual data (only in debug mode)
    if logger.isEnabledFor(logging.DEBUG):
        for i, row in enumerate(rows[:5]):
            logger.debug(f"Row {i+2}: {row}")
    
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
                        logger.debug(
                            f"No picklist sku found for single SKU {order_sku}, skipping SKU"
                        )
                        continue  # Skip this SKU instead of falling back

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
    logger.debug("Loading all SKU mappings from Google Sheets...")

    # Initialize output structure
    result = {}

    try:
        # Get all centers or use the specified one
        centers = [center] if center else ["Oxnard", "Wheeling"]

        # Process each center
        for current_center in centers:
            logger.debug(f"Processing {current_center} SKU mappings...")

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

            logger.debug(
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

    logger.debug(
        f"Total loaded: {sum(len(center_data['singles']) for center_data in result.values())} singles and {sum(len(center_data['bundles']) for center_data in result.values())} bundles across all warehouses"
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
    logger.debug(f"Fetching data from {sheet_name} sheet...")
    try:
        values = get_fruit_inventory_data(sheet_name)
        if not values:
            return None

        headers = values[0]
        data = values[1:]
        
        # Log the column count information for debugging
        logger.info(f"Headers contain {len(headers)} columns")
        if data:
            row_lengths = [len(row) for row in data[:5]]  # Check first 5 rows
            logger.info(f"First 5 data rows have lengths: {row_lengths}")
        
        # Pad short rows with None values to match header length
        padded_data = []
        for i, row in enumerate(data):
            if len(row) < len(headers):
                # Pad the row with None values
                padded_row = row + [None] * (len(headers) - len(row))
                if i < 3:  # Log first few padding operations for debugging
                    logger.info(f"Row {i+2} padded from {len(row)} to {len(padded_row)} columns")
                padded_data.append(padded_row)
            else:
                padded_data.append(row)
        
        df = pd.DataFrame(padded_data, columns=headers)
        df = deduplicate_columns(df)
        
        # Convert numeric columns
        if 'Total Weight' in df.columns:
            df['Total Weight'] = pd.to_numeric(df['Total Weight'], errors='coerce').fillna(0)
        
        # Convert date columns
        date_cols = ['INVENTORY DATE', 'FRUIT DATE']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        logger.info(f"Successfully created DataFrame with {len(df)} rows and {len(df.columns)} columns")
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
    logger.debug("Loading pieces vs lb conversion data...")
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
    logger.debug(f"Loading All Picklist V2 data from GHF Aggregation Dashboard...")
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)
        
        sheet = service.spreadsheets()
        
        # Get ALL data from the sheet without range restriction
        result = (
            sheet.values()
            .get(
                spreadsheetId=GHF_AGGREGATION_DASHBOARD_ID,
                range=ALL_PICKLIST_V2_SHEET_NAME
            )
            .execute()
        )
        
        values = result.get("values", [])
        if not values:
            logger.warning(f"No data found in {ALL_PICKLIST_V2_SHEET_NAME}")
            return None
            
        # Debug raw data structure
        logger.debug("\n=== DETAILED DATA INSPECTION ===")
        if values and len(values) > 0:
            logger.debug("\nAll Headers:")
            for idx, header in enumerate(values[0]):
                col_letter = chr(65 + idx) if idx < 26 else chr(64 + idx//26) + chr(65 + idx%26)
                logger.debug(f"Column {col_letter} ({idx}): {header}")
            
            logger.debug("\nFirst 10 Rows of Data:")
            for row_idx, row in enumerate(values[1:11]):  # Skip header, show next 10 rows
                logger.debug(f"\nRow {row_idx + 1}:")
                for col_idx, value in enumerate(row):
                    if col_idx < len(values[0]):  # Only show columns that have headers
                        col_letter = chr(65 + col_idx) if col_idx < 26 else chr(64 + col_idx//26) + chr(65 + col_idx%26)
                        header = values[0][col_idx]
                        logger.debug(f"  {col_letter} ({header}): {value}")
            
            logger.debug("\n=== END DATA INSPECTION ===\n")
        
        # Create range string to get all data from the sheet
        range_string = f"{ALL_PICKLIST_V2_SHEET_NAME}!A:BZ"  # Using BZ to ensure we get all columns
            
        # Debug raw data
        logger.debug("First few rows of raw data:")
        for i, row in enumerate(values[:5]):
            logger.debug(f"Row {i}: {row}")
            
        # Find the first row that has the most columns - that's likely our real header row
        max_cols = 0
        header_idx = 0
        for idx, row in enumerate(values[:10]):  # Check first 10 rows
            logger.debug(f"Row {idx} length: {len(row)}")
            if len(row) > max_cols:
                max_cols = len(row)
                header_idx = idx
        
        logger.debug(f"Selected header row {header_idx} with {max_cols} columns")
        
        headers = values[header_idx]
        logger.debug(f"Headers: {headers}")
        
        # Create a mapping of column letter to index
        col_to_idx = {}
        for i in range(len(headers)):
            letter = chr(65 + i) if i < 26 else chr(64 + i//26) + chr(65 + i%26)
            col_to_idx[letter] = i
            
        # Debug column mapping
        logger.debug("Column letter to index mapping:")
        logger.debug(col_to_idx)
        
        # Use all available columns instead of selecting specific ones
        logger.debug(f"Loading all {len(headers)} columns from the sheet")
        
        # Process data rows
        data = values[header_idx + 2:]  # Skip one more row after the header row
        
        processed_data = []
        for row in data:  # Process all rows
            # Pad short rows with None
            if len(row) < len(headers):
                row = row + [None] * (len(headers) - len(row))
            
            # Process all columns
            processed_row = []
            for i, value in enumerate(row[:len(headers)]):
                # Clean up special values
                if value in ['#DIV/0!', '#N/A', '#VALUE!', '#REF!', '#NAME?', '#NULL!', '#NUM!', '', None]:
                    value = '0'
                elif isinstance(value, str):
                    # Remove any currency symbols, commas and extra spaces
                    value = value.replace('$', '').replace(',', '').strip()
                    # Handle percentage values
                    if '%' in value:
                        try:
                            value = str(float(value.replace('%', '')) / 100)
                        except ValueError:
                            value = '0'
                processed_row.append(value)
            processed_data.append(processed_row)
        
        df = pd.DataFrame(processed_data, columns=headers)
        
        # Debug initial data
        logger.debug("Initial column names:")
        logger.debug(df.columns.tolist())
        
        # Define column groups
        product_type_col = 'Product Type'
        numeric_cols = [
            'Total Weight', 'Inventory', 'Confirmed Agg', 'Total', 'Total Needs (LBS)',
            'OX 1: Projection', 'OX 1: Weight', 'OX 1: Inventory + Confirmed Agg',
            'WH: Projection 1', 'WH 1: Weight',
            'OX: Projection 2', 'OX 2: Weight', 'OX 2: Inventory', 'OX 2: Inventory + Confirmed Agg', 'OX 2: Needs'
        ]
        adjustment_cols = [col for col in df.columns if 'Inventory Adjustment' in col]
        projection_factor_cols = [col for col in df.columns if 'Projection Factor' in col]
        
        # Define date columns that should NOT be converted to numeric
        date_cols = [
            'Volume Start Date', 'Volume End Date', 'Projection End Date',
            'OX: Projection End Date', 'WH: Projection End Date'
        ]
        
        # Clean and convert numeric columns
        columns_to_process = [col for col in df.columns if col != product_type_col]
        
        for col in columns_to_process:
            try:
                # Check if this column exists and is accessible
                if col not in df.columns:
                    continue
                    
                # Get the column as a Series - handle duplicate column names
                column_data = df[col]
                if isinstance(column_data, pd.DataFrame):
                    # If we get a DataFrame (due to duplicate column names), take the first column
                    column_data = column_data.iloc[:, 0]
                
                # Convert to string first to clean up
                column_data = column_data.astype(str)
                
                # Clean up the values
                column_data = column_data.apply(lambda x: '0' if str(x) in ['#DIV/0!', '#N/A', '#VALUE!', '#REF!', '#NAME?', '#NULL!', '#NUM!', '', 'None'] else x)
                
                # Remove currency symbols and commas
                column_data = column_data.astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                
                # Handle different column types
                if col in adjustment_cols:
                    column_data = column_data.apply(lambda x: float(1) if str(x).strip() == '1' else float(0.2))
                elif col in projection_factor_cols:
                    column_data = column_data.apply(lambda x: float(1) if str(x).strip() in ['1', '1.0'] else float(x))
                elif col in date_cols:
                    # Keep date columns as strings, just clean them up
                    column_data = column_data.apply(lambda x: '' if str(x).strip() in ['0', '0.0', 'nan', 'None', 'NaN'] else str(x).strip())
                else:
                    column_data = pd.to_numeric(column_data, errors='coerce').fillna(0).round(2)
                
                # Assign back to DataFrame
                df[col] = column_data
                
            except Exception as e:
                logger.error(f"Error processing column {col}: {e}")
                # Set column to default values if processing fails
                df[col] = 0
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        # Filter out rows where Product Type is empty or whitespace
        if product_type_col in df.columns:
            df = df[df[product_type_col].notna() & (df[product_type_col].astype(str).str.strip() != '')]
        
        # Filter out rows where all numeric columns are 0
        # Only use numeric columns that actually exist in the DataFrame
        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
        if existing_numeric_cols:
            numeric_mask = df[existing_numeric_cols].ne(0).any(axis=1)
            df = df[numeric_mask]
        else:
            logger.warning("No numeric columns found for filtering, keeping all rows")
        
        logger.debug(f"Successfully loaded {len(df)} rows from {ALL_PICKLIST_V2_SHEET_NAME}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load All Picklist V2 data: {str(e)}")
        logger.exception("Full traceback:")
        return None


def load_projection_snapshot(spreadsheet_id: str = None, latest: bool = True) -> pd.DataFrame | None:
    """
    Loads projection snapshot data from the projection snapshots folder.
    
    Args:
        spreadsheet_id (str, optional): Specific spreadsheet ID to load from.
                                       If None, will find the latest snapshot.
        latest (bool): If True and spreadsheet_id is None, loads the most recent snapshot.
                      If False, returns a list of available snapshots.
    
    Returns:
        pd.DataFrame | None: DataFrame with projection data, or None if failed
    """
    logger.debug(f"Loading projection snapshot data...")
    try:
        creds = get_credentials()
        
        # If no specific spreadsheet ID provided, find the latest one
        if spreadsheet_id is None:
            drive_service = build("drive", "v3", credentials=creds)
            
            # Search for spreadsheets in the projection snapshots folder
            query = f"'{PROJECTION_SNAPSHOTS_FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
            
            # Get all files in the folder and subfolders
            all_files = []
            page_token = None
            
            while True:
                results = drive_service.files().list(
                    q=query,
                    fields="nextPageToken, files(id, name, createdTime, parents)",
                    orderBy="createdTime desc",
                    pageToken=page_token
                ).execute()
                
                files = results.get('files', [])
                all_files.extend(files)
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            # Also search in year/month subfolders
            # First get all folders in the projection snapshots folder
            folder_query = f"'{PROJECTION_SNAPSHOTS_FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            folder_results = drive_service.files().list(
                q=folder_query,
                fields="files(id, name)"
            ).execute()
            
            year_folders = folder_results.get('files', [])
            
            # Search in each year folder for month folders
            for year_folder in year_folders:
                month_folder_query = f"'{year_folder['id']}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                month_results = drive_service.files().list(
                    q=month_folder_query,
                    fields="files(id, name)"
                ).execute()
                
                month_folders = month_results.get('files', [])
                
                # Search for spreadsheets in each month folder
                for month_folder in month_folders:
                    spreadsheet_query = f"'{month_folder['id']}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
                    spreadsheet_results = drive_service.files().list(
                        q=spreadsheet_query,
                        fields="files(id, name, createdTime, parents)",
                        orderBy="createdTime desc"
                    ).execute()
                    
                    spreadsheets = spreadsheet_results.get('files', [])
                    all_files.extend(spreadsheets)
            
            if not all_files:
                logger.warning("No projection snapshots found in the folder")
                return None
            
            # Sort by creation time (most recent first)
            all_files.sort(key=lambda x: x['createdTime'], reverse=True)
            
            if not latest:
                logger.info("Available projection snapshots:")
                for file in all_files[:10]:  # Show first 10
                    logger.info(f"  - {file['name']} (ID: {file['id']}, Created: {file['createdTime']})")
                return None
            
            # Use the most recent file
            latest_file = all_files[0]
            spreadsheet_id = latest_file['id']
            logger.info(f"Loading latest projection snapshot: {latest_file['name']} (Created: {latest_file['createdTime']})")
        
        # Now load the data from the specific spreadsheet
        sheets_service = build("sheets", "v4", credentials=creds)
        sheet = sheets_service.spreadsheets()
        
        # Get ALL data from the ALL_Picklist_V2 sheet
        result = (
            sheet.values()
            .get(
                spreadsheetId=spreadsheet_id,
                range=ALL_PICKLIST_V2_SHEET_NAME
            )
            .execute()
        )
        
        values = result.get("values", [])
        if not values:
            logger.warning(f"No data found in {ALL_PICKLIST_V2_SHEET_NAME} of projection snapshot")
            return None
        
        # Use the same processing logic as load_all_picklist_v2
        # Find the first row that has the most columns - that's likely our real header row
        max_cols = 0
        header_idx = 0
        for idx, row in enumerate(values[:10]):  # Check first 10 rows
            if len(row) > max_cols:
                max_cols = len(row)
                header_idx = idx
        
        logger.debug(f"Selected header row {header_idx} with {max_cols} columns")
        
        headers = values[header_idx]
        logger.debug(f"Headers: {headers[:10]}...")  # Show first 10 headers
        
        # Create a mapping of column letter to index
        col_to_idx = {}
        for i in range(len(headers)):
            letter = chr(65 + i) if i < 26 else chr(64 + i//26) + chr(65 + i%26)
            col_to_idx[letter] = i
        
        # Use all available columns instead of selecting specific ones
        logger.debug(f"Loading all {len(headers)} columns from the projection snapshot")
        
        # Process data rows
        data = values[header_idx + 2:]  # Skip one more row after the header row
        
        processed_data = []
        for row in data:  # Process all rows
            # Pad short rows with None
            if len(row) < len(headers):
                row = row + [None] * (len(headers) - len(row))
            
            # Process all columns
            processed_row = []
            for i, value in enumerate(row[:len(headers)]):
                # Clean up special values
                if value in ['#DIV/0!', '#N/A', '#VALUE!', '#REF!', '#NAME?', '#NULL!', '#NUM!', '', None]:
                    value = '0'
                elif isinstance(value, str):
                    # Remove any currency symbols, commas and extra spaces
                    value = value.replace('$', '').replace(',', '').strip()
                    # Handle percentage values
                    if '%' in value:
                        try:
                            value = str(float(value.replace('%', '')) / 100)
                        except ValueError:
                            value = '0'
                processed_row.append(value)
            processed_data.append(processed_row)
        
        df = pd.DataFrame(processed_data, columns=headers)
        
        # Define column groups (same as load_all_picklist_v2)
        product_type_col = 'Product Type'
        numeric_cols = [
            'Total Weight', 'Inventory', 'Confirmed Agg', 'Total', 'Total Needs (LBS)',
            'OX 1: Projection', 'OX 1: Weight', 'OX 1: Inventory + Confirmed Agg',
            'WH: Projection 1', 'WH 1: Weight',
            'OX: Projection 2', 'OX 2: Weight', 'OX 2: Inventory', 'OX 2: Inventory + Confirmed Agg', 'OX 2: Needs'
        ]
        adjustment_cols = [col for col in df.columns if 'Inventory Adjustment' in col]
        projection_factor_cols = [col for col in df.columns if 'Projection Factor' in col]
        
        # Define date columns that should NOT be converted to numeric
        date_cols = [
            'Volume Start Date', 'Volume End Date', 'Projection End Date',
            'OX: Projection End Date', 'WH: Projection End Date'
        ]
        
        # Clean and convert numeric columns (same logic as load_all_picklist_v2)
        columns_to_process = [col for col in df.columns if col != product_type_col]
        
        for col in columns_to_process:
            try:
                # Check if this column exists and is accessible
                if col not in df.columns:
                    continue
                    
                # Get the column as a Series - handle duplicate column names
                column_data = df[col]
                if isinstance(column_data, pd.DataFrame):
                    # If we get a DataFrame (due to duplicate column names), take the first column
                    column_data = column_data.iloc[:, 0]
                
                # Convert to string first to clean up
                column_data = column_data.astype(str)
                
                # Clean up the values
                column_data = column_data.apply(lambda x: '0' if str(x) in ['#DIV/0!', '#N/A', '#VALUE!', '#REF!', '#NAME?', '#NULL!', '#NUM!', '', 'None'] else x)
                
                # Remove currency symbols and commas
                column_data = column_data.astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                
                # Handle different column types
                if col in adjustment_cols:
                    column_data = column_data.apply(lambda x: float(1) if str(x).strip() == '1' else float(0.2))
                elif col in projection_factor_cols:
                    column_data = column_data.apply(lambda x: float(1) if str(x).strip() in ['1', '1.0'] else float(x))
                elif col in date_cols:
                    # Keep date columns as strings, just clean them up
                    column_data = column_data.apply(lambda x: '' if str(x).strip() in ['0', '0.0', 'nan', 'None', 'NaN'] else str(x).strip())
                else:
                    column_data = pd.to_numeric(column_data, errors='coerce').fillna(0).round(2)
                
                # Assign back to DataFrame
                df[col] = column_data
                
            except Exception as e:
                logger.error(f"Error processing column {col}: {e}")
                # Set column to default values if processing fails
                df[col] = 0
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        # Filter out rows where Product Type is empty or whitespace
        if product_type_col in df.columns:
            df = df[df[product_type_col].notna() & (df[product_type_col].astype(str).str.strip() != '')]
        
        # Filter out rows where all numeric columns are 0
        # Only use numeric columns that actually exist in the DataFrame
        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
        if existing_numeric_cols:
            numeric_mask = df[existing_numeric_cols].ne(0).any(axis=1)
            df = df[numeric_mask]
        else:
            logger.warning("No numeric columns found for filtering, keeping all rows")
        
        logger.debug(f"Successfully loaded {len(df)} rows from projection snapshot")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load projection snapshot data: {str(e)}")
        logger.exception("Full traceback:")
        return None


def list_available_projection_snapshots() -> List[Dict[str, str]]:
    """
    Lists all available projection snapshots in the projection snapshots folder.
    
    Returns:
        List[Dict[str, str]]: List of dictionaries with snapshot information
                             Each dict contains: id, name, createdTime, folder_path
    """
    logger.debug("Listing available projection snapshots...")
    try:
        creds = get_credentials()
        drive_service = build("drive", "v3", credentials=creds)
        
        all_snapshots = []
        
        # Search for spreadsheets directly in the projection snapshots folder
        query = f"'{PROJECTION_SNAPSHOTS_FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
        results = drive_service.files().list(
            q=query,
            fields="files(id, name, createdTime, parents)",
            orderBy="createdTime desc"
        ).execute()
        
        files = results.get('files', [])
        for file in files:
            all_snapshots.append({
                'id': file['id'],
                'name': file['name'],
                'createdTime': file['createdTime'],
                'folder_path': 'Root'
            })
        
        # Also search in year/month subfolders
        folder_query = f"'{PROJECTION_SNAPSHOTS_FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        folder_results = drive_service.files().list(
            q=folder_query,
            fields="files(id, name)"
        ).execute()
        
        year_folders = folder_results.get('files', [])
        
        for year_folder in year_folders:
            month_folder_query = f"'{year_folder['id']}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            month_results = drive_service.files().list(
                q=month_folder_query,
                fields="files(id, name)"
            ).execute()
            
            month_folders = month_results.get('files', [])
            
            for month_folder in month_folders:
                spreadsheet_query = f"'{month_folder['id']}' in parents and mimeType='application/vnd.google-apps.spreadsheet' and trashed=false"
                spreadsheet_results = drive_service.files().list(
                    q=spreadsheet_query,
                    fields="files(id, name, createdTime, parents)",
                    orderBy="createdTime desc"
                ).execute()
                
                spreadsheets = spreadsheet_results.get('files', [])
                for spreadsheet in spreadsheets:
                    all_snapshots.append({
                        'id': spreadsheet['id'],
                        'name': spreadsheet['name'],
                        'createdTime': spreadsheet['createdTime'],
                        'folder_path': f"{year_folder['name']}/{month_folder['name']}"
                    })
        
        # Sort by creation time (most recent first)
        all_snapshots.sort(key=lambda x: x['createdTime'], reverse=True)
        
        logger.info(f"Found {len(all_snapshots)} projection snapshots")
        return all_snapshots
        
    except Exception as e:
        logger.error(f"Failed to list projection snapshots: {str(e)}")
        return []

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
            'projections': {
                'projection_1': float,     # AL (OX 1: Projection) + AT (WH: Projection 1)
                'projection_2': float,     # BB (OX: Projection 2) + BM (WH: Projection 2)
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
    # AL: OX 1: Projection
    # AT: WH: Projection 1
    # BB: OX: Projection 2
    # BM: WH: Projection 2
    
    # Get column indices (0-based)
    col_indices = {
        'N': 13,  # N is the 14th column (0-based)
        'Q': 16,  # Q is the 17th column
        'T': 19,  # T is the 20th column
        'W': 22,  # W is the 23rd column
        'Z': 25,  # Z is the 26th column
        'AL': 37, # AL is the 38th column (OX 1: Projection)
        'AM': 38, # AM is the 39th column
        'AN': 39, # AN is the 40th column
        'AO': 40, # AO is the 41st column
        'AT': 45, # AT is the 46th column (WH: Projection 1)
        'AU': 46, # AU is the 47th column
        'AV': 47, # AV is the 48th column
        'AW': 48, # AW is the 49th column
        'BB': 53, # BB is the 54th column (OX: Projection 2)
        'BM': 64  # BM is the 65th column (WH: Projection 2)
    }
    
    # Calculate combined projections
    projection_1 = safe_float(row.iloc[col_indices['AL']]) + safe_float(row.iloc[col_indices['AT']])
    projection_2 = safe_float(row.iloc[col_indices['BB']]) + safe_float(row.iloc[col_indices['BM']])
    
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
        'projections': {
            'projection_1': projection_1,  # AL + AT
            'projection_2': projection_2   # BB + BM
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
    """Load Orders_new data from GHF Fruit Tracking spreadsheet."""
    logger.debug(f"Loading Orders_new data from GHF Fruit Tracking...")
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)
        
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=GHF_FRUIT_TRACKING, range=ORDERS_NEW_SHEET_NAME)
            .execute()
        )
        
        values = result.get("values", [])
        if not values:
            logger.warning(f"No data found in {ORDERS_NEW_SHEET_NAME}")
            return None
            
        headers = values[0]
        data = values[1:]
        
        # Select only the columns we need (columns A, B, C, D, E)
        needed_columns = ["A", "B", "C", "D", "E"]
        max_col_index = min(len(headers), 5)  # Only take first 5 columns
        
        selected_headers = headers[:max_col_index]
        selected_data = []
        
        for row in data:
            # Ensure row has enough columns
            if len(row) < max_col_index:
                row = row + [''] * (max_col_index - len(row))
            
            selected_row = row[:max_col_index]
            
            # Clean up the data
            for i in range(len(selected_row)):
                if selected_row[i] is None:
                    selected_row[i] = ''
                else:
                    selected_row[i] = str(selected_row[i]).strip()
                    
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
        
        logger.debug(f"Successfully loaded {len(df)} rows from {ORDERS_NEW_SHEET_NAME}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load Orders_new data: {str(e)}")
        logger.exception("Full traceback:")
        return None

def load_wow_data():
    """Load Week over Week data from the GHF AGG/FRUIT Google Sheet."""
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()
        
        # Get all data from the sheet by just specifying the sheet name
        result = sheet.values().get(
            spreadsheetId=GHF_AGG_FRUIT,
            range=WOW_SHEET,
            valueRenderOption='FORMATTED_VALUE'
        ).execute()
        
        values = result.get('values', [])
        if not values:
            logger.error("No data found in the sheet")
            return pd.DataFrame()

        # Get headers (date ranges) from first row
        headers = values[0]
        logger.debug(f"Found {len(headers)} columns in the sheet")
        
        # Initialize lists to store data
        data = []
        metric_order = []
        
        # Process each row
        for row_idx, row in enumerate(values[1:], start=1):
            metric = row[0] if row else ""
            if not metric or metric.strip().startswith('#'):  # Skip empty rows or comments
                continue
                
            # Track original metric order
            if metric not in metric_order:
                metric_order.append(metric)
                
            # Extend row with empty values if needed
            row_extended = row + [''] * (len(headers) - len(row))
            
            # Process each column for this metric
            for col_idx, (header, value) in enumerate(zip(headers[1:], row_extended[1:]), start=1):
                if not header or not isinstance(header, str):
                    continue
                    
                # Clean and convert the value
                clean_value = str(value).strip() if value else ""
                if clean_value.lower() in ['#n/a', '#div/0!', '#div/0', 'n/a', '-', '', '#ref!', '#value!', '#name?', '#null!', '#num!', '#div/0! (function divide parameter 2 cannot be zero.)', '#div/0! (function divide parameter 2 cannot be zero.)']:
                    clean_value = '0'
                
                try:
                    # Detect formatting from the original value
                    original_value_str = str(value) if value else ""
                    has_dollar_sign = '$' in original_value_str
                    has_percent_sign = '%' in original_value_str
                    
                    # Convert to float if possible (remove formatting symbols)
                    numeric_value = float(clean_value.replace('$', '').replace(',', '').replace('%', ''))
                    
                    # Determine data type based on original formatting from Google Sheets
                    is_percentage = has_percent_sign  # Only if cell has % symbol
                    is_currency = has_dollar_sign     # Only if cell has $ symbol
                    is_weight = not is_percentage and not is_currency  # Everything else is weight (lb)
                    
                    # Parse the date range to get start and end dates
                    try:
                        start_date, end_date = parse_date_range(header)
                    except (ValueError, AttributeError) as date_e:
                        logger.debug(f"Could not parse date range '{header}': {date_e}")
                        continue
                    
                    # Add to data list
                    data.append({
                        'Metric': metric,
                        'Date Range': header,
                        'Start Date': start_date,
                        'End Date': end_date,
                        'Value': numeric_value,
                        'Original Value': str(value) if value else "",
                        'Is Percentage': is_percentage,
                        'Is Currency': is_currency,
                        'Is Weight': is_weight,
                        'Metric_Order': metric_order.index(metric)
                    })
                    
                except (ValueError, TypeError) as e:
                    # Only log debug for common Excel/Sheets formula errors
                    error_str = str(e).lower()
                    if any(excel_error in error_str for excel_error in ['#ref!', '#value!', '#div/0!', '#n/a']):
                        logger.debug(f"Skipping formula error for {metric} at {header}: {e}")
                    else:
                        logger.warning(f"Error processing value for {metric} at {header}: {e}")
                    continue

        # Convert to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            logger.warning("No data was processed successfully")
            return df
            
        # Sort by date (newest first) and original metric order (preserve sheet order)
        df = df.sort_values(['Start Date', 'Metric_Order'], ascending=[False, True])
        # Log summary of data issues encountered
        total_data_points = len(data)
        successful_data_points = len(df)
        skipped_points = total_data_points - successful_data_points
        
        if skipped_points > 0:
            logger.debug(f"Processed {successful_data_points} data points, skipped {skipped_points} due to formula errors or invalid data")
        else:
            logger.debug(f"Successfully loaded {successful_data_points} data points across {len(df['Date Range'].unique())} date ranges")
        
        return df

    except Exception as e:
        logger.error(f"Error loading WoW data: {str(e)}")
        raise

def load_sku_type_data() -> pd.DataFrame | None:
    """Load SKU Type data from INPUT_SKU_TYPE sheet."""
    logger.debug(f"Loading SKU Type data from {INPUT_SKU_TYPE_SHEET_NAME}...")
    try:
        # Get raw data
        data = get_sheet_data(INPUT_SKU_TYPE_SHEET_NAME)
        if not data:
            logger.warning(f"No data found in {INPUT_SKU_TYPE_SHEET_NAME}")
            return None
        
        # Get headers
        headers = data[0] if data else []
        if not headers:
            logger.warning("No headers found in SKU Type data")
            return None
        
        # Convert column letters to indices
        column_map = {}
        for i, col_letter in enumerate(INPUT_SKU_TYPE_NEEDED_COLUMNS):
            if col_letter == "A":
                column_map[col_letter] = 0
            elif col_letter == "B":
                column_map[col_letter] = 1
            elif col_letter == "I":
                column_map[col_letter] = 8
            else:
                # Calculate column index for multi-letter columns
                if len(col_letter) == 1:
                    column_map[col_letter] = ord(col_letter) - ord('A')
                else:
                    # For columns like AA, AB, etc.
                    column_map[col_letter] = (ord(col_letter[0]) - ord('A') + 1) * 26 + (ord(col_letter[1]) - ord('A'))
        
        # Check if needed columns exist
        needed_indices = []
        for col in INPUT_SKU_TYPE_NEEDED_COLUMNS:
            if col in column_map and column_map[col] < len(headers):
                needed_indices.append(column_map[col])
            else:
                logger.warning(f"Column index {column_map.get(col, 'unknown')} not found in headers")
        
        if not needed_indices:
            logger.warning("No valid columns found in SKU Type data")
            return None
            
        # Process data rows
        selected_data = []
        for row in data[1:]:  # Skip header row
            # Pad row if it's shorter than needed
            if len(row) < max(needed_indices) + 1:
                row = row + [''] * (max(needed_indices) + 1 - len(row))
            
            # Extract only needed columns
            selected_row = []
            for idx in needed_indices:
                value = row[idx] if idx < len(row) else ''
                # Clean up the value
                if value is None:
                    value = ''
                selected_row.append(str(value).strip())
            
            # Only add rows that have at least SKU (first column)
            if selected_row[0]:
                selected_data.append(selected_row)
        
        # Create DataFrame with proper column names
        column_names = ['SKU', 'PRODUCT_TYPE', 'SKU_Helper']
        df = pd.DataFrame(selected_data, columns=column_names)
        
        # Remove empty rows
        df = df[df['SKU'].str.strip() != '']
        
        logger.debug(f"Successfully loaded {len(df)} rows from {INPUT_SKU_TYPE_SHEET_NAME}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load SKU Type data: {str(e)}")
        logger.exception("Full traceback:")
        return None


# Google Drive Upload Functions
def file_exists_in_drive(filename: str, folder_id: str = None) -> bool:
    """
    Check if a file already exists in Google Drive
    
    Args:
        filename (str): Name of the file to check
        folder_id (str): ID of the folder to search in (optional)
        
    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        creds = get_credentials()
        service = build("drive", "v3", credentials=creds)
        
        # Build query to search for the file
        query = f"name='{filename}' and trashed=false"
        if folder_id:
            query += f" and '{folder_id}' in parents"
        
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        existing_files = results.get('files', [])
        return len(existing_files) > 0
        
    except Exception as e:
        logger.error(f"Error checking if file '{filename}' exists: {str(e)}")
        return False


def create_folder_if_not_exists(folder_name: str, parent_folder_id: str = None) -> str:
    """
    Create a folder in Google Drive if it doesn't already exist
    
    Args:
        folder_name (str): Name of the folder to create
        parent_folder_id (str): ID of the parent folder (optional)
        
    Returns:
        str: Folder ID of the created or existing folder, or None if failed
    """
    try:
        creds = get_credentials()
        service = build("drive", "v3", credentials=creds)
        
        # Check if folder already exists
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_folder_id:
            query += f" and '{parent_folder_id}' in parents"
        
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        existing_folders = results.get('files', [])
        
        if existing_folders:
            # Folder already exists, return the first one
            folder_id = existing_folders[0]['id']
            logger.info(f"Folder '{folder_name}' already exists with ID: {folder_id}")
            return folder_id
        
        # Create new folder
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_folder_id:
            folder_metadata['parents'] = [parent_folder_id]
        
        folder = service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()
        
        folder_id = folder.get('id')
        logger.info(f"Created folder '{folder_name}' with ID: {folder_id}")
        return folder_id
        
    except Exception as e:
        logger.error(f"Error creating folder '{folder_name}': {str(e)}")
        return None


def upload_file_to_drive(file_path: str, folder_id: str = None, filename: str = None) -> str:
    """
    Upload a file to Google Drive
    
    Args:
        file_path (str): Path to the local file to upload
        folder_id (str): Optional Google Drive folder ID to upload to
        filename (str): Optional custom filename for the uploaded file
        
    Returns:
        str: Google Drive file ID of the uploaded file, or None if upload failed
    """
    try:
        creds = get_credentials()
        service = build("drive", "v3", credentials=creds)
        
        # Prepare file metadata
        file_metadata = {}
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        if filename:
            file_metadata['name'] = filename
        else:
            # Use the original filename
            file_metadata['name'] = os.path.basename(file_path)
        
        # Upload the file
        logger.info(f"Uploading {file_path} to Google Drive...")
        
        with open(file_path, 'rb') as file:
            media = service.files().create(
                body=file_metadata,
                media_body=file,
                fields='id,name,webViewLink'
            ).execute()
        
        file_id = media.get('id')
        file_name = media.get('name')
        web_link = media.get('webViewLink')
        
        logger.info(f"Successfully uploaded {file_name} to Google Drive")
        logger.info(f"File ID: {file_id}")
        logger.info(f"Web Link: {web_link}")
        
        return file_id
        
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to Google Drive: {str(e)}")
        return None


def upload_csv_to_drive(csv_content: str, filename: str, folder_id: str = None) -> str:
    """
    Upload CSV content directly to Google Drive
    
    Args:
        csv_content (str): CSV content as string
        filename (str): Name for the file
        folder_id (str): Google Drive folder ID to upload to (optional)
        
    Returns:
        str: Google Drive file ID of the uploaded file or None if failed
    """
    try:
        # Check if file already exists
        if file_exists_in_drive(filename, folder_id):
            logger.info(f"File '{filename}' already exists in Drive, skipping upload")
            return None
        
        creds = get_credentials()
        service = build("drive", "v3", credentials=creds)
        
        logger.info(f"Uploading CSV content as {filename} to Google Drive...")
        
        # Create file metadata
        file_metadata = {
            'name': filename,
            'mimeType': 'text/csv'
        }
        
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        # Create media upload object
        from io import BytesIO
        media = MediaIoBaseUpload(
            BytesIO(csv_content.encode('utf-8')),
            mimetype='text/csv',
            resumable=True
        )
        
        # Upload the file
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink'
        ).execute()
        
        file_id = file.get('id')
        web_link = file.get('webViewLink')
        
        logger.info(f"Successfully uploaded {filename} to Google Drive")
        logger.info(f"File ID: {file_id}")
        logger.info(f"Web Link: {web_link}")
        
        return file_id
        
    except Exception as e:
        logger.error(f"Error uploading CSV to Drive: {str(e)}")
        return None


def list_folder_contents(folder_id: str) -> list:
    """
    List contents of a Google Drive folder
    
    Args:
        folder_id (str): Google Drive folder ID
        
    Returns:
        list: List of file information dictionaries
    """
    try:
        creds = get_credentials()
        service = build("drive", "v3", credentials=creds)
        
        results = service.files().list(
            q=f"'{folder_id}' in parents",
            fields="files(id,name,mimeType,createdTime,size)"
        ).execute()
        
        files = results.get('files', [])
        logger.info(f"Found {len(files)} files in folder {folder_id}")
        
        return files
        
    except Exception as e:
        logger.error(f"Failed to list folder contents: {str(e)}")
        return []

if __name__ == "__main__":
    try:
        # Create a directory for the output files if it doesn't exist
        output_dir = "sheet_data"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 1. Load and save SKU mappings
        sku_mappings = load_sku_mappings_from_sheets()
        with open(f"{output_dir}/sku_mappings.json", "w") as f:
            json.dump(sku_mappings, f, indent=2)
        print(f"✓ Saved SKU mappings to {output_dir}/sku_mappings.json")

        # 2. Load and save Agg Orders
        agg_orders_df = load_agg_orders(filter_by_projection_period=True, group_by_projection=True)
        if agg_orders_df is not None:
            agg_orders_df.to_csv(f"{output_dir}/agg_orders.csv", index=False)
            print(f"✓ Saved {len(agg_orders_df)} rows to {output_dir}/agg_orders.csv")

        # # 3. Load and save All Picklist V2
        picklist_df = load_all_picklist_v2()
        if picklist_df is not None:
            picklist_df.to_csv(f"{output_dir}/all_picklist_v2.csv", index=False)
            print(f"✓ Saved {len(picklist_df)} rows to {output_dir}/all_picklist_v2.csv")

        # 4. Load and save Pieces vs LB Conversion
        conversion_df = load_pieces_vs_lb_conversion()
        if conversion_df is not None:
            conversion_df.to_csv(f"{output_dir}/pieces_vs_lb_conversion.csv", index=False)
            print(f"✓ Saved {len(conversion_df)} rows to {output_dir}/pieces_vs_lb_conversion.csv")

        # 5. Load and save Oxnard Inventory
        oxnard_inv_df = load_oxnard_inventory()
        if oxnard_inv_df is not None:
            oxnard_inv_df.to_csv(f"{output_dir}/oxnard_inventory.csv", index=False)
            print(f"✓ Saved {len(oxnard_inv_df)} rows to {output_dir}/oxnard_inventory.csv")

        # 6. Load and save Wheeling Inventory
        wheeling_inv_df = load_wheeling_inventory()
        if wheeling_inv_df is not None:
            wheeling_inv_df.to_csv(f"{output_dir}/wheeling_inventory.csv", index=False)
            print(f"✓ Saved {len(wheeling_inv_df)} rows to {output_dir}/wheeling_inventory.csv")

        # 7. Load and save Orders New from Fruit Tracking
        orders_new_df = load_orders_new()
        if orders_new_df is not None:
            orders_new_df.to_csv(f"{output_dir}/orders_new.csv", index=False)
            print(f"✓ Saved {len(orders_new_df)} rows to {output_dir}/orders_new.csv")

        # 8. Load and save Week over Week data
        wow_df = load_wow_data()
        if wow_df is not None:
            wow_df.to_csv(f"{output_dir}/wow_data.csv", index=False)
            print(f"✓ Saved {len(wow_df)} rows to {output_dir}/wow_data.csv")

        # 9. Load and save SKU Type data
        sku_type_df = load_sku_type_data()
        if sku_type_df is not None:
            sku_type_df.to_csv(f"{output_dir}/sku_type.csv", index=False)
            print(f"✓ Saved {len(sku_type_df)} rows to {output_dir}/sku_type.csv")

        # List available projection snapshots
        snapshots = list_available_projection_snapshots()
        if snapshots:
            print("\nAvailable Projection Snapshots:")
            for snapshot in snapshots:
                print(f"ID: {snapshot['id']}")
                print(f"Name: {snapshot['name']}")
                print(f"Created: {snapshot['createdTime']}")
                print(f"Folder Path: {snapshot['folder_path']}")
                print("------------------------")
        else:
            print("No projection snapshots found.")

        projection_snapshot = load_projection_snapshot(spreadsheet_id="1vswH1dqlWR-aQcv6Bs9eZHV0jVZG8ONdbJr2YhVbyV4")
        if projection_snapshot is not None:
            projection_snapshot.to_csv(f"{output_dir}/projection_snapshot.csv", index=False)
            print(f"✓ Saved {len(projection_snapshot)} rows to {output_dir}/projection_snapshot.csv")

        print("\nAll data has been saved to the 'sheet_data' directory!")
        print("Summary of files saved:")
        print("------------------------")
        for file in os.listdir(output_dir):
            size = os.path.getsize(os.path.join(output_dir, file)) / 1024  # Convert to KB
            print(f"{file:<30} {size:.1f} KB")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logger.exception("Full traceback:")
