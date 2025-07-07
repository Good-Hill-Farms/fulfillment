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

# Sheet IDs
GHF_INVENTORY_ID = "19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY"
AGG_ORDERS_SHEET_NAME = "Agg_Orders"
GHF_AGGREGATION_DASHBOARD_ID = "1CdTTV8pMqq_wS9vu0qa8HMykNkqtOverrIsP0WLSUeM"


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
    """Fetches data from a Google Sheet."""
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=GHF_INVENTORY_ID, range=f"{sheet_name}")
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
    """Fetches data from the Aggregation Google Sheet."""
    try:
        creds = get_credentials()
        service = build("sheets", "v4", credentials=creds)
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(
                spreadsheetId=GHF_AGGREGATION_DASHBOARD_ID,
                range=f"{sheet_name}",
            )
            .execute()
        )
        values = result.get("values", [])
        if not values:
            logger.warning(f"No data found in sheet {sheet_name}")
            return []
        logger.info(f"Successfully fetched {len(values)} rows from {sheet_name}")
        return values
    except Exception as e:
        logger.error(f"Error fetching agg data from sheet {sheet_name}: {e}")
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
    
    # DEBUG: Print headers and first few rows to see what we're getting from Google Sheets
    print(f"\n=== DEBUG: Processing {center} ===")
    print(f"Headers: {headers}")
    print(f"Total rows: {len(rows)}")
    
    # Print first 5 rows to see the actual data
    for i, row in enumerate(rows[:5]):
        print(f"Row {i+2}: {row}")
    
    # Look for the specific SKUs you mentioned
    target_skus = ["f.avocado_hass-2lb", "f.avocado_hass-2lb_3-bab"]
    print(f"\nLooking for target SKUs: {target_skus}")

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
            
            # DEBUG: Print info for target SKUs
            if order_sku in target_skus:
                print(f"\nFOUND TARGET SKU '{order_sku}' at row {row_idx}:")
                print(f"  Raw row: {row}")
                print(f"  Mapping: {mapping}")
                print(f"  picklist sku: {mapping.get('picklist sku', 'NOT FOUND')}")
                print(f"  actualqty: {mapping.get('actualqty', 'NOT FOUND')}")
                print(f"  Total Pick Weight: {mapping.get('Total Pick Weight', 'NOT FOUND')}")
            
            if order_sku not in sku_groups:
                sku_groups[order_sku] = []
            sku_groups[order_sku].append((row_idx, mapping))
        except Exception as e:
            logger.error(f"Error grouping row {row_idx}: {e}")
            continue

    # DEBUG: Show what SKU groups we found
    print(f"\nFound {len(sku_groups)} unique SKUs")
    for sku in target_skus:
        if sku in sku_groups:
            print(f"Target SKU '{sku}' has {len(sku_groups[sku])} rows")
        else:
            print(f"Target SKU '{sku}' NOT FOUND in data!")

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
                    
                    # DEBUG: Print result for target SKUs
                    if order_sku in target_skus:
                        print(f"\nPROCESSED TARGET SKU '{order_sku}':")
                        print(f"  Result: {result[center]['singles'][order_sku]}")
                    
                    logger.debug(f"Processed single SKU {order_sku}")
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


if __name__ == "__main__":
    try:
        # Load all SKU mappings
        result = load_sku_mappings_from_sheets()

        # Save to JSON for inspection
        with open("google_sheets_sku_mappings.json", "w") as f:
            json.dump(result, f, indent=2)

        print("Data saved to google_sheets_sku_mappings.json")
        print(
            f"Loaded {sum(len(result[warehouse]['singles']) for warehouse in result)} singles and {sum(len(result[warehouse]['bundles']) for warehouse in result)} bundles"
        )

        result = load_agg_orders()
        if result is not None:
            with open("agg_orders.json", "w") as f:
                # Convert DataFrame to a list of dictionaries before dumping to JSON
                f.write(result.to_json(orient="records", indent=4))

        print("Data saved to agg_orders.json")
        print(f"Loaded {len(result)} rows from Agg_Orders")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
