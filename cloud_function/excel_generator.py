import pandas as pd
import os
from datetime import datetime
import pytz
from inventory_api import get_inventory_data
from google_sheets import get_sheets_service, get_drive_service, create_folder_if_not_exists, load_pieces_vs_lb_conversion
import re

# Google Drive folder IDs
OXNARD_FOLDER_ID = "1BDW2dd41h6_gvdUWVsUmZHIdd8gkIU_c"
WHEELING_FOLDER_ID = "1xogLAldd3dUGKaEXAk_0UUILWxEpIkSP"

def get_sheet_id_by_name(spreadsheet_id, sheet_name):
    """Get sheet ID by name"""
    service = get_sheets_service()
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sheet in spreadsheet['sheets']:
        if sheet['properties']['title'] == sheet_name:
            return sheet['properties']['sheetId']
    return None

def get_weight_from_pieces_conversion(sku, quantity, pieces_df):
    """Convert quantity to weight in lbs using the pieces vs lb conversion table"""
    if pieces_df is None or pd.isna(sku) or not sku:
        return 0
    
    conversion_row = pieces_df[pieces_df['picklist sku'] == sku]
    if not conversion_row.empty:
        weight_per_unit = pd.to_numeric(conversion_row.iloc[0].get('Weight (lbs)', 0), errors='coerce')
        calculated_weight = float(quantity) * float(weight_per_unit) if not pd.isna(weight_per_unit) else 0
        return round(calculated_weight, 2)
    return 0

def extract_date_from_batch(batch_code):
    """Extract date from batch code and format it as 'Month DD, YYYY'."""
    if pd.isna(batch_code):
        return ''
    
    # Convert to string and trim
    batch_str = str(batch_code).strip()
    
    # Format 1: standard 'delivered_MMDDYY'
    new_pattern = r'delivered_(\d{6})'
    match = re.search(new_pattern, batch_str)
    if match:
        date_str = match.group(1)
        try:
            date = datetime.strptime(date_str, '%m%d%y')
            return date.strftime('%B %d, %Y')
        except ValueError:
            pass
    
    # Format 2: fallback 5 or 6 digit formats (like #110424 or 83024)
    fallback_str = re.sub(r'^#', '', batch_str).strip()
    fallback_pattern = r'^(\d{5,6})$'
    fallback_match = re.search(fallback_pattern, fallback_str)
    if fallback_match:
        raw = fallback_match.group(1)
        padded = f"0{raw}" if len(raw) == 5 else raw
        try:
            date = datetime.strptime(padded, '%m%d%y')
            return date.strftime('%B %d, %Y')
        except ValueError:
            pass
    
    # Format 3: MM/DD/YY
    date_pattern = r'delivered__(\d{2}/\d{2}/\d{2})'
    match = re.search(date_pattern, batch_str)
    if match:
        date_str = match.group(1)
        try:
            date = datetime.strptime(date_str, '%m/%d/%y')
            return date.strftime('%B %d, %Y')
        except ValueError:
            pass
    
    # Format 4: date after #
    hash_pattern = r'#.*?(\d{2}/\d{2}/\d{2})'
    match = re.search(hash_pattern, batch_str)
    if match:
        date_str = match.group(1)
        try:
            date = datetime.strptime(date_str, '%m/%d/%y')
            return date.strftime('%B %d, %Y')
        except ValueError:
            pass
    
    return ''

def format_google_sheet(spreadsheet_id, df, sheet_name="Inventory", include_qty=False):
    """Format a Google Sheet with standard styling"""
    service = get_sheets_service()
    
    # Get sheet ID by name
    sheet_id = get_sheet_id_by_name(spreadsheet_id, sheet_name)
    if sheet_id is None:
        print(f"Warning: Could not find sheet ID for {sheet_name}")
        return
    
    # Add date header
    current_time_la = datetime.now(pytz.timezone('America/Los_Angeles'))
    date_str = f"Data as of {current_time_la.strftime('%B %d, %Y %H:%M:%S')} (Los Angeles)"
    
    requests = [
        # Insert row for date header
        {
            "insertDimension": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "ROWS",
                    "startIndex": 0,
                    "endIndex": 1
                },
                "inheritFromBefore": False
            }
        },
        # Add date header
        {
            "updateCells": {
                "rows": [{
                    "values": [{
                        "userEnteredValue": {"stringValue": date_str},
                        "userEnteredFormat": {
                            "textFormat": {"bold": True},
                            "horizontalAlignment": "LEFT",
                            "verticalAlignment": "TOP"
                        }
                    }]
                }],
                "fields": "userEnteredValue,userEnteredFormat",
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": 1
                }
            }
        },
        # Format header row
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": 2,
                    "startColumnIndex": 0,
                    "endColumnIndex": len(df.columns)
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 0, "green": 0.4, "blue": 0.8},
                        "textFormat": {
                            "foregroundColor": {"red": 1, "green": 1, "blue": 1},
                            "bold": True
                        },
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE"
                    }
                },
                "fields": "userEnteredFormat"
            }
        },
        # Freeze header row
        {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": sheet_id,
                    "gridProperties": {
                        "frozenRowCount": 2
                    }
                },
                "fields": "gridProperties.frozenRowCount"
            }
        }
    ]
    
    # Add column formatting
    for i, col in enumerate(df.columns):
        # Set column widths
        width = 200 if col in ['Name', 'Sku'] else 100
        requests.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": i,
                    "endIndex": i + 1
                },
                "properties": {
                    "pixelSize": width
                },
                "fields": "pixelSize"
            }
        })
        
        # Format input columns with light green background
        if col in ['Counted QTY', 'Unit (lb, ea, cs)', 'Notes', 'QTY']:
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 2,  # Start after headers
                        "startColumnIndex": i,
                        "endColumnIndex": i + 1
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.9, "green": 1, "blue": 0.9}
                        }
                    },
                    "fields": "userEnteredFormat.backgroundColor"
                }
            })
    
    # Execute all formatting requests
    try:
        service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": requests}
        ).execute()
    except Exception as e:
        print(f"Warning: Error formatting sheet {sheet_name}: {e}")

def generate_inventory_sheet(warehouse_name, parent_folder_id):
    """Generate inventory Google Sheet for a specific warehouse"""
    print(f"Generating inventory Google Sheet for {warehouse_name}...")
    
    # Get inventory data
    df = get_inventory_data()
    if df is None or df.empty:
        print("❌ No data received from the API")
        print("Please check if your API token is correctly set in the environment variables.")
        return
    
    print("Debug: DataFrame columns:", df.columns.tolist())
    
    # Filter data based on warehouse
    if warehouse_name == "Wheeling":
        df = df[df['WarehouseName'] == "IL-Wheeling-60090"]
    else:  # Oxnard
        df = df[df['WarehouseName'].str.contains('CA-Oxnard-93030|Moorpark', case=False, na=False)]
    
    # Add empty columns for QTY, Unit, and Notes
    df['Counted QTY'] = ''
    df['Unit (lb, ea, cs)'] = ''
    df['Notes'] = ''
    
    # Extract date from BatchCode to LOT
    df['LOT'] = df['BatchCode'].apply(extract_date_from_batch)
    
    # Convert ItemId to string
    df['ItemId'] = df['ItemId'].astype(str)
    
    # Ensure Expected AvailableQty is numeric
    df['Expected AvailableQty (ea)'] = pd.to_numeric(df['AvailableQty'], errors='coerce')
    
    # Load pieces vs lb conversion data
    pieces_df = load_pieces_vs_lb_conversion()
    if pieces_df is None:
        print("⚠️ Warning: Could not load pieces vs lb conversion data. Weights will be set to 0.")
    
    # Calculate weight in lbs
    df['Weight (lbs)'] = df.apply(lambda row: get_weight_from_pieces_conversion(row['Sku'], row['Expected AvailableQty (ea)'], pieces_df), axis=1)
    
    # Reorder columns to match the inventory pages with Weight (lbs) after Expected AvailableQty (ea)
    df = df[['ItemId', 'Name', 'Sku', 'LOT', 'Expected AvailableQty (ea)', 'Weight (lbs)', 'Counted QTY', 'Unit (lb, ea, cs)', 'Notes']]
    
    # Filter out items with 0 or negative quantity but keep empty rows
    mask = (df['Expected AvailableQty (ea)'].fillna(-1) <= 0) & (df['Expected AvailableQty (ea)'].notna())
    df = df[~mask]
    
    # Add 100 empty rows for manual input
    empty_rows = pd.DataFrame({
        'ItemId': [''] * 100,
        'Name': [''] * 100,
        'Sku': [''] * 100,
        'LOT': [''] * 100,
        'Expected AvailableQty (ea)': [''] * 100,
        'Weight (lbs)': [''] * 100,
        'Counted QTY': [''] * 100,
        'Unit (lb, ea, cs)': [''] * 100,
        'Notes': [''] * 100
    })
    df = pd.concat([df, empty_rows], ignore_index=True)
    
    # Replace NaN values with empty strings
    df = df.fillna('')
    
    # Get Google API services
    service = get_sheets_service()
    drive_service = get_drive_service()
    
    try:
        # Create month folder (e.g., "July_2024")
        current_time_la = datetime.now(pytz.timezone('America/Los_Angeles'))
        month_folder_name = current_time_la.strftime("%B_%Y")
        month_folder_id = create_folder_if_not_exists(month_folder_name, parent_folder_id)
        
        # Create new spreadsheet
        timestamp = current_time_la.strftime("%Y%m%d_%H%M%S")
        filename = f"{warehouse_name.lower()}_inventory_{timestamp}"
        
        # Create the initial spreadsheet with all three sheets
        spreadsheet = service.spreadsheets().create(
            body={
                'properties': {'title': filename},
                'sheets': [
                    {'properties': {'title': 'Inventory'}},
                    {'properties': {'title': 'Disposable Inventory'}},
                    {'properties': {'title': 'Dry Inventory'}}
                ]
            }
        ).execute()
        spreadsheet_id = spreadsheet['spreadsheetId']
        
        # Move to correct folder
        file = drive_service.files().get(fileId=spreadsheet_id, fields='parents').execute()
        previous_parents = ",".join(file.get('parents', []))
        drive_service.files().update(
            fileId=spreadsheet_id,
            addParents=month_folder_id,
            removeParents=previous_parents,
            fields='id, parents'
        ).execute()
        
        # Write and format Inventory sheet
        values = [df.columns.tolist()] + df.values.tolist()
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="Inventory!A1",
            valueInputOption='RAW',
            body={'values': values}
        ).execute()
        format_google_sheet(spreadsheet_id, df, sheet_name="Inventory", include_qty=True)
        
        # Prepare and write Disposable Inventory sheet
        disposable_df = df.copy()
        disposable_df['Expected AvailableQty (ea)'] = ''  # Clear Expected AvailableQty
        if 'LOT dates' in disposable_df.columns:
            disposable_df = disposable_df.drop('LOT dates', axis=1)
        disposable_df = disposable_df.rename(columns={'Expected AvailableQty (ea)': 'QTY ea'})
        disposable_df = disposable_df.drop(['Weight (lbs)', 'QTY ea'], axis=1)
        values = [disposable_df.columns.tolist()] + disposable_df.values.tolist()
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="Disposable Inventory!A1",
            valueInputOption='RAW',
            body={'values': values}
        ).execute()
        format_google_sheet(spreadsheet_id, disposable_df, sheet_name="Disposable Inventory", include_qty=True)
        
        # Prepare and write Dry Inventory sheet
        dry_items = [
            "Box: 8x8x8",
            "Box: 10x10x10",
            "Vented Box: 10x10x10",
            "Box: 12x12x12",
            "Vented Box: 12x12x12",
            "Honeycomb paper",
            "Wood Shavings",
            "Ethelyne bags, 20g",
            "Cardboard Pads",
            "Clamshell",
            "Regular Inserts (Get to Know)",
            "Branded tape - Normal",
            "Absorbent Pads",
            "Fruit netting",
            "1lb Nut bags (large)",
            "Gold Flower Spoons",
            "Wood Baskets"
        ]
        
        dry_df = pd.DataFrame({
            'Packaging | Units': dry_items,
            'QTY': [''] * len(dry_items)
        })
        
        values = [dry_df.columns.tolist()] + dry_df.values.tolist()
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="Dry Inventory!A1",
            valueInputOption='RAW',
            body={'values': values}
        ).execute()
        format_google_sheet(spreadsheet_id, dry_df, sheet_name="Dry Inventory", include_qty=True)
        
        # Get the spreadsheet URL
        sheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        print(f"✅ Successfully generated {filename}")
        print(f"Google Sheet URL: {sheet_url}")
        return sheet_url
    except Exception as e:
        print(f"❌ Error generating inventory sheet: {e}")
        print("Please check if your API token is correctly set in the environment variables.")
        return None

if __name__ == "__main__":
    print("Generating inventory Google Sheets...")
    
    # Generate Wheeling inventory
    wheeling_url = generate_inventory_sheet("Wheeling", WHEELING_FOLDER_ID)
    
    # Generate Oxnard inventory
    oxnard_url = generate_inventory_sheet("Oxnard", OXNARD_FOLDER_ID) 
