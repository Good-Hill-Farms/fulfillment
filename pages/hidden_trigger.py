import streamlit as st
import pandas as pd
import os
import sys
import json
import time
import uuid
from datetime import datetime
import streamlit as st

# Add the parent directory to the Python path to fix import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import DataProcessor
from utils.airtable_handler import AirtableHandler

# Import Google API libraries
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Import constants from google_sheets.py
from utils.google_sheets import GHF_AGGREGATION_DASHBOARD_ID, ALL_PICKLIST_V2_SHEET_NAME

# This page will not appear in the navigation sidebar
st.set_page_config(
    page_title="System Maintenance",
    page_icon="🔧",
    layout="wide"
)

# CSS to hide this page from navigation
st.markdown("""
<style>
[data-testid="stSidebarNavItems"] div:has(div:has(p:contains("hidden_trigger"))) {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# Google Drive folder ID where the spreadsheet will be created
# Note: This should be just the folder ID without any additional characters
# Make sure there's no trailing period in the folder ID
DRIVE_FOLDER_ID = "1-uUvyCTEx_TLKOF46jHD3Kpsp8aO9W9b"  # The folder ID from the provided URL

# Set this to True to enable folder moving (only if you've shared the folder with the service account)
ENABLE_FOLDER_MOVING = True

# Google API scopes
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Function to check query parameters
def get_google_credentials():
    """Get Google API credentials for accessing Drive and Sheets"""
    try:
        # First, check if we have a local service account file
        json_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "nca-toolkit-project-446011-67d246fdbccf.json")
        
        if os.path.exists(json_file):
            return service_account.Credentials.from_service_account_file(
                json_file, scopes=SCOPES)
        else:
            st.error("Service account credentials file not found")
            return None
    except Exception as e:
        st.error(f"Error loading credentials: {str(e)}")
        return None

def create_spreadsheet(title, data=None, data_type="inventory"):
    """Create a new spreadsheet in the specified Google Drive folder"""
    try:
        # Get credentials
        creds = get_google_credentials()
        if not creds:
            return None, "Failed to get Google credentials"
            
        # Build the Drive and Sheets services
        drive_service = build('drive', 'v3', credentials=creds)
        sheets_service = build('sheets', 'v4', credentials=creds)
        
        # Create proper naming format with date
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M")
        spreadsheet_title = f"ALL_Picklist_{date_str}_{time_str}"
        
        # Get the service account email for debugging
        credentials_info = service_account.Credentials.from_service_account_file(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "nca-toolkit-project-446011-67d246fdbccf.json")
        ).service_account_email
        
        # Log the service account being used
        st.info(f"Using service account: {credentials_info}")
        
        # Create year/month directory structure
        year_str = now.strftime("%Y")
        month_str = now.strftime("%m-%B")  # e.g., "07-July"
        
        try:
            # First, check if the base folder exists
            base_folder = drive_service.files().get(fileId=DRIVE_FOLDER_ID).execute()
            st.success(f"Base folder found: {base_folder.get('name')}")
            
            # Create or find year folder
            st.info(f"Creating/finding year folder: {year_str}")
            year_folder_id = create_or_find_folder(drive_service, year_str, DRIVE_FOLDER_ID)
            st.success(f"Year folder ready: {year_str}")
            
            # Create or find month folder within year folder
            st.info(f"Creating/finding month folder: {month_str}")
            month_folder_id = create_or_find_folder(drive_service, month_str, year_folder_id)
            st.success(f"Month folder ready: {month_str}")
            
            # Create a new spreadsheet with only one sheet (ALL_Picklist_V2)
            spreadsheet = {
                'properties': {
                    'title': spreadsheet_title
                },
                'sheets': [
                    {
                        'properties': {
                            'title': 'ALL_Picklist_V2',
                            'gridProperties': {
                                'rowCount': 1000,
                                'columnCount': 26
                            }
                        }
                    }
                ]
            }
            
            # Create the spreadsheet
            spreadsheet = sheets_service.spreadsheets().create(body=spreadsheet).execute()
            spreadsheet_id = spreadsheet.get('spreadsheetId')
            
            # Move the file to the month folder
            file = drive_service.files().update(
                fileId=spreadsheet_id,
                addParents=month_folder_id,
                fields='id, parents'
            ).execute()
            
            st.success(f"Spreadsheet '{spreadsheet_title}' created in {year_str}/{month_str} folder successfully")
            
            # Create shareable link
            drive_service.permissions().create(
                fileId=spreadsheet_id,
                body={
                    'type': 'anyone',
                    'role': 'reader'
                }
            ).execute()
            
            # Get the web view link
            file = drive_service.files().get(
                fileId=spreadsheet_id,
                fields='webViewLink'
            ).execute()
            
            web_link = file.get('webViewLink')
            
            # Special case for picklist data - copy from source spreadsheet
            if data_type == "picklist":
                st.info(f"Starting picklist data copy process...")
                st.info(f"Source spreadsheet ID: {GHF_AGGREGATION_DASHBOARD_ID}")
                st.info(f"Target spreadsheet ID: {spreadsheet_id}")
                st.info(f"Main sheet to copy: {ALL_PICKLIST_V2_SHEET_NAME}")
                
                try:
                    # Update the scopes to include full access
                    credentials = get_google_credentials()
                    st.info("✓ Google credentials loaded successfully")
                    
                    # Create a new sheets service with full access scopes
                    full_sheets_service = build('sheets', 'v4', credentials=credentials)
                    st.info("✓ Google Sheets service created successfully")
                    
                    copy_success = copy_sheet_with_related_data(
                        full_sheets_service, 
                        GHF_AGGREGATION_DASHBOARD_ID, 
                        ALL_PICKLIST_V2_SHEET_NAME, 
                        spreadsheet_id
                    )
                    if copy_success:
                        st.success(f"✅ Successfully copied {ALL_PICKLIST_V2_SHEET_NAME} and related data to the new spreadsheet")
                    else:
                        st.error(f"❌ Failed to copy {ALL_PICKLIST_V2_SHEET_NAME} and related data")
                        
                except Exception as e:
                    st.error(f"❌ Error in picklist copy process: {str(e)}")
                    import traceback
                    st.error(f"Full traceback: {traceback.format_exc()}")
            # If other data is provided, add it to the spreadsheet
            elif data is not None and isinstance(data, pd.DataFrame):
                # Convert DataFrame to values list for the API
                values = [data.columns.tolist()] + data.values.tolist()
                
                body = {
                    'values': values
                }
                
                # Update the spreadsheet with data
                sheets_service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range='Sheet1!A1',
                    valueInputOption='RAW',
                    body=body
                ).execute()
                
                st.success(f"Data added to spreadsheet successfully")
            
            return spreadsheet_id, web_link
            
        except Exception as folder_error:
            st.error(f"Target folder not found: {str(folder_error)}")
            st.info(f"Please make sure the folder exists and is shared with the service account: {credentials_info}")
            return None, f"Cannot create spreadsheet: target folder not accessible"
        
    except HttpError as error:
        return None, f"An error occurred: {error}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def create_or_find_folder(drive_service, folder_name, parent_folder_id):
    """Create a folder if it doesn't exist, or return existing folder ID"""
    try:
        # Search for existing folder
        query = f"name='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = drive_service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get('files', [])
        
        if folders:
            # Folder exists, return its ID
            folder_id = folders[0]['id']
            st.info(f"Found existing folder: {folder_name}")
            return folder_id
        else:
            # Create new folder
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_folder_id]
            }
            folder = drive_service.files().create(body=folder_metadata, fields='id').execute()
            folder_id = folder.get('id')
            st.info(f"Created new folder: {folder_name}")
            return folder_id
            
    except Exception as e:
        st.error(f"Error creating/finding folder {folder_name}: {str(e)}")
        return parent_folder_id  # Fallback to parent folder

def test_source_access(sheets_service, source_spreadsheet_id, source_sheet_name):
    """Test if we can access the source spreadsheet and sheet"""
    try:
        st.info(f"Testing access to source spreadsheet: {source_spreadsheet_id}")
        
        # Try to get basic spreadsheet info
        source_spreadsheet = sheets_service.spreadsheets().get(
            spreadsheetId=source_spreadsheet_id
        ).execute()
        
        st.info(f"✓ Successfully accessed source spreadsheet: {source_spreadsheet.get('properties', {}).get('title', 'Unknown')}")
        
        # Get all sheet names
        sheet_names = [sheet['properties']['title'] for sheet in source_spreadsheet['sheets']]
        st.info(f"Available sheets: {', '.join(sheet_names)}")
        
        # Check if our target sheet exists
        if source_sheet_name in sheet_names:
            st.info(f"✓ Found target sheet: {source_sheet_name}")
            
            # Try to read some data from the sheet
            result = sheets_service.spreadsheets().values().get(
                spreadsheetId=source_spreadsheet_id,
                range=f"{source_sheet_name}!A1:E10",  # Just first 10 rows, 5 columns
                valueRenderOption='FORMATTED_VALUE'
            ).execute()
            
            values = result.get('values', [])
            st.info(f"✓ Successfully read {len(values)} rows from {source_sheet_name}")
            
            if values:
                st.info(f"Sample data from row 1: {values[0][:3] if len(values[0]) > 3 else values[0]}")
                return True
            else:
                st.warning(f"Sheet {source_sheet_name} appears to be empty")
                return False
        else:
            st.error(f"❌ Sheet {source_sheet_name} not found in source spreadsheet")
            return False
            
    except Exception as e:
        st.error(f"❌ Error accessing source spreadsheet: {str(e)}")
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return False

def copy_sheet_with_related_data(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id):
    """Copy the main sheet and any related sheets used in calculations"""
    try:
        # First test if we can access the source
        if not test_source_access(sheets_service, source_spreadsheet_id, source_sheet_name):
            st.error("Cannot access source spreadsheet - aborting copy")
            return False
        
        # Get source spreadsheet metadata (without grid data to avoid timeout)
        source_spreadsheet = sheets_service.spreadsheets().get(
            spreadsheetId=source_spreadsheet_id
        ).execute()
        
        # Copy only the main ALL_Picklist_V2 sheet with data and styling (no formulas)
        st.info(f"Copying {source_sheet_name} with data and styling (no formulas)...")
        
        success = copy_sheet_data_with_styling(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id)
        if success:
            st.success(f"✓ Successfully copied {source_sheet_name} with data and styling")
            
            # Remove the default "Sheet1" that was created automatically
            try:
                st.info("Removing default Sheet1...")
                target_spreadsheet = sheets_service.spreadsheets().get(
                    spreadsheetId=target_spreadsheet_id
                ).execute()
                
                # Find Sheet1 (the default sheet)
                sheet1_id = None
                for sheet in target_spreadsheet['sheets']:
                    if sheet['properties']['title'] == 'Sheet1':
                        sheet1_id = sheet['properties']['sheetId']
                        break
                
                if sheet1_id:
                    sheets_service.spreadsheets().batchUpdate(
                        spreadsheetId=target_spreadsheet_id,
                        body={
                            "requests": [
                                {
                                    "deleteSheet": {
                                        "sheetId": sheet1_id
                                    }
                                }
                            ]
                        }
                    ).execute()
                    st.success("✓ Removed default Sheet1")
                else:
                    st.info("No default Sheet1 found to remove")
                    
            except Exception as e:
                st.warning(f"Could not remove default Sheet1: {str(e)}")
                # This is not critical, continue anyway
            
            return True
        else:
            st.error(f"✗ Failed to copy {source_sheet_name}")
            return False
        
    except Exception as e:
        st.error(f"Error copying related sheets: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def copy_single_sheet_with_data(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id):
    """Copy a single sheet with all data, formulas, and formatting preserved using Google Sheets copyTo API"""
    try:
        st.info(f"Starting copy of {source_sheet_name} using Google Sheets copyTo API...")

        # 1. First, get the source sheet ID
        st.info("Getting source sheet information...")
        source_spreadsheet = sheets_service.spreadsheets().get(
            spreadsheetId=source_spreadsheet_id
        ).execute()
        
        source_sheet_id = None
        for sheet in source_spreadsheet['sheets']:
            if sheet['properties']['title'] == source_sheet_name:
                source_sheet_id = sheet['properties']['sheetId']
                break
        
        if source_sheet_id is None:
            st.error(f"Source sheet '{source_sheet_name}' not found")
            return False
        
        st.info(f"Found source sheet ID: {source_sheet_id}")
        
        # 2. Use Google Sheets copyTo API to copy the entire sheet with all formatting
        st.info(f"Copying {source_sheet_name} to target spreadsheet...")
        try:
            copy_response = sheets_service.spreadsheets().sheets().copyTo(
                spreadsheetId=source_spreadsheet_id,
                sheetId=source_sheet_id,
                body={
                    'destinationSpreadsheetId': target_spreadsheet_id
                }
            ).execute()
        except Exception as copy_error:
            st.warning(f"copyTo API failed: {str(copy_error)}")
            st.info("Trying fallback method: copying data values only...")
            return copy_sheet_data_fallback(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id)
        
        # Get the new sheet ID and title
        new_sheet_id = copy_response['sheetId']
        new_sheet_title = copy_response['title']
        
        st.success(f"✅ Successfully copied {source_sheet_name} as '{new_sheet_title}'")
        st.info(f"New sheet ID: {new_sheet_id}")
        
        # 3. Rename the copied sheet to remove "Copy of" prefix
        target_sheet_name = source_sheet_name  # Use original name without _Copy suffix
        if new_sheet_title != target_sheet_name:
            st.info(f"Renaming sheet to: {target_sheet_name}")
            try:
                sheets_service.spreadsheets().batchUpdate(
                    spreadsheetId=target_spreadsheet_id,
                    body={
                        "requests": [
                            {
                                "updateSheetProperties": {
                                    "properties": {
                                        "sheetId": new_sheet_id,
                                        "title": target_sheet_name
                                    },
                                    "fields": "title"
                                }
                            }
                        ]
                    }
                ).execute()
                st.success(f"✓ Renamed sheet to: {target_sheet_name}")
            except Exception as e:
                st.warning(f"Could not rename sheet: {str(e)}")
                # This is not critical, continue anyway
        
        st.success(f"✅ Complete! Sheet copied with all data, formulas, and formatting preserved")
        return True
        
    except Exception as e:
        st.error(f"Error copying sheet {source_sheet_name}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def copy_sheet_data_with_styling(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id):
    """Copy sheet data with styling but no formulas - gets actual calculated values"""
    try:
        st.info(f"Copying {source_sheet_name} with data and styling...")
        
        # 1. Get the source sheet information
        st.info("Getting source sheet information...")
        source_spreadsheet = sheets_service.spreadsheets().get(
            spreadsheetId=source_spreadsheet_id
        ).execute()
        
        source_sheet_props = None
        source_sheet_id = None
        for sheet in source_spreadsheet['sheets']:
            if sheet['properties']['title'] == source_sheet_name:
                source_sheet_props = sheet['properties']
                source_sheet_id = sheet['properties']['sheetId']
                break
        
        if not source_sheet_props:
            st.error(f"Source sheet {source_sheet_name} not found")
            return False
        
        st.info(f"Found source sheet ID: {source_sheet_id}")
        
        # 2. First, detect the actual used range to avoid empty rows/columns
        st.info(f"Detecting actual data range in {source_sheet_name}...")
        
        # Get a large range to detect the actual boundaries
        data_result = sheets_service.spreadsheets().values().get(
            spreadsheetId=source_spreadsheet_id,
            range=f"{source_sheet_name}!A1:ZZZ50000",
            valueRenderOption='FORMATTED_VALUE'
        ).execute()
        all_values = data_result.get('values', [])
        
        # Find the actual last row with data (trim empty rows from the end)
        last_row_with_data = 0
        for i, row in enumerate(all_values):
            if any(cell.strip() for cell in row if cell):  # Check if row has any non-empty cells
                last_row_with_data = i + 1
        
        # Find the actual last column with data
        last_col_with_data = 0
        for row in all_values[:last_row_with_data]:
            for j, cell in enumerate(row):
                if cell and cell.strip():
                    last_col_with_data = max(last_col_with_data, j + 1)
        
        # Trim to actual data range
        values = all_values[:last_row_with_data] if last_row_with_data > 0 else []
        
        st.info(f"Actual data range: {last_row_with_data} rows x {last_col_with_data} columns")
        st.info(f"Found {len(values)} rows of actual data (trimmed empty rows)")
        
        if not values:
            st.warning(f"No data found in {source_sheet_name}")
            return False
        
        # 3. Create new sheet with exact dimensions (no unnecessary rows/columns)
        sheet_properties = {
            "title": source_sheet_name,
            "gridProperties": {
                "rowCount": max(1000, last_row_with_data + 50),  # Small buffer but not excessive
                "columnCount": max(26, last_col_with_data + 5)   # Small buffer but not excessive
            }
        }
        
        # 3. Find the existing sheet (we created it when we created the spreadsheet)
        st.info(f"Finding existing sheet: {source_sheet_name}")
        try:
            target_spreadsheet = sheets_service.spreadsheets().get(
                spreadsheetId=target_spreadsheet_id
            ).execute()
            
            new_sheet_id = None
            for sheet in target_spreadsheet['sheets']:
                if sheet['properties']['title'] == source_sheet_name:
                    new_sheet_id = sheet['properties']['sheetId']
                    break
            
            if new_sheet_id is None:
                st.error(f"Target sheet {source_sheet_name} not found")
                return False
                
            st.success(f"✓ Found existing sheet with ID: {new_sheet_id}")
            
            # Update the sheet properties to match source (frozen rows/columns, dimensions)
            update_requests = []
            
            # Update grid properties
            grid_properties = {
                "rowCount": max(1000, last_row_with_data + 50),
                "columnCount": max(26, last_col_with_data + 5)
            }
            
            # Copy frozen rows/columns from source
            if 'gridProperties' in source_sheet_props:
                source_grid = source_sheet_props['gridProperties']
                if 'frozenRowCount' in source_grid:
                    grid_properties['frozenRowCount'] = source_grid['frozenRowCount']
                    st.info(f"Setting frozen rows: {source_grid['frozenRowCount']}")
                if 'frozenColumnCount' in source_grid:
                    grid_properties['frozenColumnCount'] = source_grid['frozenColumnCount']
                    st.info(f"Setting frozen columns: {source_grid['frozenColumnCount']}")
            
            update_requests.append({
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": new_sheet_id,
                        "gridProperties": grid_properties
                    },
                    "fields": "gridProperties"
                }
            })
            
            # Apply the updates
            if update_requests:
                sheets_service.spreadsheets().batchUpdate(
                    spreadsheetId=target_spreadsheet_id,
                    body={"requests": update_requests}
                ).execute()
                st.success("✓ Updated sheet properties")
                
        except Exception as e:
            st.error(f"Failed to configure sheet: {str(e)}")
            return False
        
        # 4. Write the data
        st.info(f"Writing {len(values)} rows of data...")
        try:
            sheets_service.spreadsheets().values().update(
                spreadsheetId=target_spreadsheet_id,
                range=f"{source_sheet_name}!A1",
                valueInputOption="RAW",
                body={
                    "values": values
                }
            ).execute()
            st.success(f"✓ Data written successfully")
        except Exception as e:
            st.error(f"Failed to write data: {str(e)}")
            return False
        
        # 5. Copy comprehensive formatting using the actual data range
        st.info("Copying formatting and styling...")
        try:
            # Use the actual data range we detected for formatting
            end_col_letter = chr(ord('A') + min(last_col_with_data - 1, 25)) if last_col_with_data <= 26 else 'Z'
            format_range = f"{source_sheet_name}!A1:{end_col_letter}{last_row_with_data}"
            
            st.info(f"Getting formatting for range: {format_range}")
            
            # Get source sheet with formatting data for the actual range
            source_sheet_data = sheets_service.spreadsheets().get(
                spreadsheetId=source_spreadsheet_id,
                ranges=[format_range],
                includeGridData=True
            ).execute()
            
            # Apply comprehensive formatting if available
            if 'sheets' in source_sheet_data and source_sheet_data['sheets']:
                source_data = source_sheet_data['sheets'][0]
                if 'data' in source_data and source_data['data']:
                    grid_data = source_data['data'][0]
                    
                    # Create formatting requests
                    format_requests = []
                    
                    # Copy row data with comprehensive formatting
                    if 'rowData' in grid_data:
                        for row_idx, row_data in enumerate(grid_data['rowData'][:last_row_with_data]):
                            if 'values' in row_data:
                                for col_idx, cell_data in enumerate(row_data['values'][:last_col_with_data]):
                                    if 'userEnteredFormat' in cell_data:
                                        format_requests.append({
                                            "repeatCell": {
                                                "range": {
                                                    "sheetId": new_sheet_id,
                                                    "startRowIndex": row_idx,
                                                    "endRowIndex": row_idx + 1,
                                                    "startColumnIndex": col_idx,
                                                    "endColumnIndex": col_idx + 1
                                                },
                                                "cell": {
                                                    "userEnteredFormat": cell_data['userEnteredFormat']
                                                },
                                                "fields": "userEnteredFormat"
                                            }
                                        })
                    
                    # Also copy column widths and row heights
                    if 'columnMetadata' in grid_data:
                        for col_idx, col_meta in enumerate(grid_data['columnMetadata'][:last_col_with_data]):
                            if 'pixelSize' in col_meta:
                                format_requests.append({
                                    "updateDimensionProperties": {
                                        "range": {
                                            "sheetId": new_sheet_id,
                                            "dimension": "COLUMNS",
                                            "startIndex": col_idx,
                                            "endIndex": col_idx + 1
                                        },
                                        "properties": {
                                            "pixelSize": col_meta['pixelSize']
                                        },
                                        "fields": "pixelSize"
                                    }
                                })
                    
                    if 'rowMetadata' in grid_data:
                        for row_idx, row_meta in enumerate(grid_data['rowMetadata'][:last_row_with_data]):
                            if 'pixelSize' in row_meta:
                                format_requests.append({
                                    "updateDimensionProperties": {
                                        "range": {
                                            "sheetId": new_sheet_id,
                                            "dimension": "ROWS",
                                            "startIndex": row_idx,
                                            "endIndex": row_idx + 1
                                        },
                                        "properties": {
                                            "pixelSize": row_meta['pixelSize']
                                        },
                                        "fields": "pixelSize"
                                    }
                                })
                    
                    # Apply formatting in batches
                    if format_requests:
                        batch_size = 50  # Smaller batches for stability
                        for i in range(0, len(format_requests), batch_size):
                            batch = format_requests[i:i+batch_size]
                            sheets_service.spreadsheets().batchUpdate(
                                spreadsheetId=target_spreadsheet_id,
                                body={"requests": batch}
                            ).execute()
                        st.success(f"✓ Applied {len(format_requests)} formatting and styling rules")
        except Exception as e:
            st.warning(f"Could not copy all formatting: {str(e)}")
            # This is not critical, continue anyway
        
        # 6. Copy filters if they exist
        st.info("Checking for filters...")
        try:
            # Check if the source sheet has filters
            source_sheet_full = None
            for sheet in source_spreadsheet['sheets']:
                if sheet['properties']['sheetId'] == source_sheet_id:
                    source_sheet_full = sheet
                    break
            
            if source_sheet_full and 'basicFilter' in source_sheet_full:
                basic_filter = source_sheet_full['basicFilter']
                st.info(f"Found basic filter: {basic_filter}")
                
                # Apply the same filter to the new sheet
                filter_range = {
                    "sheetId": new_sheet_id,
                    "startRowIndex": basic_filter['range'].get('startRowIndex', 0),
                    "endRowIndex": min(basic_filter['range'].get('endRowIndex', last_row_with_data), last_row_with_data),
                    "startColumnIndex": basic_filter['range'].get('startColumnIndex', 0),
                    "endColumnIndex": min(basic_filter['range'].get('endColumnIndex', last_col_with_data), last_col_with_data)
                }
                
                filter_request = {
                    "setBasicFilter": {
                        "filter": {
                            "range": filter_range
                        }
                    }
                }
                
                sheets_service.spreadsheets().batchUpdate(
                    spreadsheetId=target_spreadsheet_id,
                    body={"requests": [filter_request]}
                ).execute()
                
                st.success("✓ Successfully copied filters!")
            else:
                # If no existing filter, create a basic filter for the data range
                st.info("No existing filter found, creating basic filter for data range...")
                
                filter_range = {
                    "sheetId": new_sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": last_row_with_data,
                    "startColumnIndex": 0,
                    "endColumnIndex": last_col_with_data
                }
                
                filter_request = {
                    "setBasicFilter": {
                        "filter": {
                            "range": filter_range
                        }
                    }
                }
                
                sheets_service.spreadsheets().batchUpdate(
                    spreadsheetId=target_spreadsheet_id,
                    body={"requests": [filter_request]}
                ).execute()
                
                st.success("✓ Successfully enabled filters for the data range!")
                
        except Exception as e:
            st.warning(f"Could not copy/enable filters: {str(e)}")
            # This is not critical, continue anyway
        
        st.success(f"✅ Successfully copied {source_sheet_name} with data, styling, and filters!")
        return True
        
    except Exception as e:
        st.error(f"Failed to copy {source_sheet_name}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def copy_sheet_data_fallback(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id):
    """Fallback method to copy sheet data when copyTo API fails"""
    try:
        st.info(f"Using fallback method for {source_sheet_name}...")
        
        # Get the actual data values from the source sheet with a larger range
        st.info(f"Reading data values from {source_sheet_name}...")
        data_result = sheets_service.spreadsheets().values().get(
            spreadsheetId=source_spreadsheet_id,
            range=f"{source_sheet_name}!A1:ZZZ50000",  # Much larger range to capture all data
            valueRenderOption='FORMATTED_VALUE'
        ).execute()
        values = data_result.get('values', [])
        st.info(f"Found {len(values)} rows of data")
        
        # Also get the source sheet properties for formatting
        st.info("Getting source sheet formatting...")
        source_spreadsheet = sheets_service.spreadsheets().get(
            spreadsheetId=source_spreadsheet_id
        ).execute()
        
        source_sheet_props = None
        for sheet in source_spreadsheet['sheets']:
            if sheet['properties']['title'] == source_sheet_name:
                source_sheet_props = sheet['properties']
                break
        
        if not values:
            st.warning(f"No data found in {source_sheet_name}")
            return False
        
        # Create a new sheet in the target spreadsheet
        target_sheet_name = source_sheet_name  # Use original name without _Copy suffix
        st.info(f"Creating new sheet: {target_sheet_name}")
        
        max_cols = max(len(row) for row in values) if values else 26
        
        # Use source sheet properties if available
        sheet_properties = {
            "title": target_sheet_name,
            "gridProperties": {
                "rowCount": max(2000, len(values) + 100),  # Larger buffer
                "columnCount": max(50, max_cols + 10)     # Larger buffer
            }
        }
        
        # Copy some formatting properties from source if available
        if source_sheet_props and 'gridProperties' in source_sheet_props:
            source_grid = source_sheet_props['gridProperties']
            sheet_properties['gridProperties'].update({
                "rowCount": max(source_grid.get('rowCount', 2000), len(values) + 100),
                "columnCount": max(source_grid.get('columnCount', 50), max_cols + 10)
            })
            # Copy frozen rows/columns if they exist
            if 'frozenRowCount' in source_grid:
                sheet_properties['gridProperties']['frozenRowCount'] = source_grid['frozenRowCount']
            if 'frozenColumnCount' in source_grid:
                sheet_properties['gridProperties']['frozenColumnCount'] = source_grid['frozenColumnCount']
        
        try:
            add_sheet_response = sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=target_spreadsheet_id,
                body={
                    "requests": [
                        {
                            "addSheet": {
                                "properties": sheet_properties
                            }
                        }
                    ]
                }
            ).execute()
            st.success(f"✓ Successfully created sheet: {target_sheet_name}")
        except Exception as e:
            st.error(f"Failed to create sheet: {str(e)}")
            return False
        
        # Write the data to the new sheet
        st.info(f"Writing {len(values)} rows of data to {target_sheet_name}...")
        try:
            sheets_service.spreadsheets().values().update(
                spreadsheetId=target_spreadsheet_id,
                range=f"{target_sheet_name}!A1",
                valueInputOption="RAW",
                body={
                    "values": values
                }
            ).execute()
            st.success(f"✓ Data written successfully to {target_sheet_name}")
        except Exception as e:
            st.error(f"Failed to write data: {str(e)}")
            return False
        
        st.success(f"✅ Successfully copied {source_sheet_name} using fallback method")
        return True
        
    except Exception as e:
        st.error(f"Fallback method failed for {source_sheet_name}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def check_trigger_params():
    """Check for trigger parameters in URL and execute corresponding actions"""
    query_params = st.query_params
    
    # Secret key to prevent unauthorized access
    secret_key = os.getenv("TRIGGER_SECRET_KEY", "default_secret_key")
    if "key" in query_params and query_params["key"] == secret_key:
        # Get the action parameter
        action = query_params.get("action", "")
        
        if action == "refresh_data":
            # Trigger data refresh
            with st.spinner("Refreshing data from sources..."):
                refresh_data()
            st.success("Data refresh completed")
            return True
            
        elif action == "sync_airtable":
            # Trigger Airtable sync
            with st.spinner("Syncing with Airtable..."):
                sync_airtable()
            st.success("Airtable sync completed")
            return True
            
        elif action == "clear_cache":
            # Clear Streamlit cache
            with st.spinner("Clearing application cache..."):
                clear_app_cache()
            st.success("Cache cleared")
            return True
            
        elif action == "reset_session":
            # Reset session state
            with st.spinner("Resetting session state..."):
                reset_session_state()
            st.success("Session state reset")
            return True
            
        elif action == "create_spreadsheet":
            # Create a Google Sheets spreadsheet
            title = query_params.get("title", f"Fulfillment_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            include_data = query_params.get("include_data", "false").lower() == "true"
            data_type = query_params.get("data_type", "inventory")
            
            # Create spreadsheet with optional data
            with st.spinner("Creating spreadsheet in Google Drive..."):
                # Check if we should include data
                data = None
                if include_data:
                    # Generate sample data based on data_type
                    data = generate_report_data(data_type)
                    
                # Continue with the original create_spreadsheet logic
                if data is None:
                    # Default sample data if no specific data type is available
                    data = pd.DataFrame({
                        'Date': [datetime.now().strftime("%Y-%m-%d")],
                        'Report Type': [title],
                        'Generated By': ['Fulfillment System'],
                        'Status': ['Complete']
                    })
            
                # Create the spreadsheet
                spreadsheet_id, result = create_spreadsheet(title, data, data_type)
                
                if spreadsheet_id:
                    st.success(f"Spreadsheet created successfully!")
                    st.markdown(f"[Open Spreadsheet]({result})")
                    
                    # Log the operation
                    log_operation("create_spreadsheet", {
                        "spreadsheet_id": spreadsheet_id,
                        "title": title,
                        "data_type": data_type,
                        "include_data": include_data
                    })
                else:
                    st.error(f"Failed to create spreadsheet: {result}")
            return True
                    
    # Check for the specific fulfillment projection snapshot trigger
    if "key" in query_params and query_params["key"] == "fulfillment_projection_snapshot_trigger":
        st.title("📊 Fulfillment Projection Snapshot")
        st.info("Creating snapshot of ALL_Picklist_V2 with complete data and formatting...")
        
        # Execute the copy operation
        with st.spinner("Creating projection snapshot..."):
            try:
                # Get credentials
                creds = get_google_credentials()
                if not creds:
                    st.error("❌ Failed to get Google credentials")
                    return True
                
                # Build the Sheets service
                sheets_service = build('sheets', 'v4', credentials=creds)
                
                # Create the new spreadsheet
                spreadsheet_id, spreadsheet_url = create_spreadsheet("Fulfillment_Projection_Snapshot")
                
                if spreadsheet_id:
                    st.success(f"✅ Created spreadsheet: {spreadsheet_url}")
                    
                    # Copy the ALL_Picklist_V2 sheet
                    success = copy_sheet_with_related_data(
                        sheets_service, 
                        GHF_AGGREGATION_DASHBOARD_ID, 
                        ALL_PICKLIST_V2_SHEET_NAME, 
                        spreadsheet_id
                    )
                    
                    if success:
                        st.success("🎉 Fulfillment projection snapshot created successfully!")
                        st.markdown(f"**📋 Access your snapshot:** [Open Spreadsheet]({spreadsheet_url})")
                        
                        # Show summary
                        st.markdown("### ✅ What was copied:")
                        st.markdown("- 📊 All data from ALL_Picklist_V2 (calculated values, no formulas)")
                        st.markdown("- 🎨 Complete formatting (colors, fonts, borders, column widths)")
                        st.markdown("- 🔒 Frozen headers for easy navigation")
                        st.markdown("- 🔍 Filters enabled for sorting and filtering")
                        st.markdown("- 📁 Organized in year/month folders")
                        st.markdown("- 📅 Timestamped filename for easy identification")
                    else:
                        st.error("❌ Failed to copy sheet data")
                else:
                    st.error("❌ Failed to create spreadsheet")
                    
            except Exception as e:
                st.error(f"❌ Error creating snapshot: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        return True
    
    return False

def refresh_data():
    """Refresh data from external sources"""
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Reload SKU mappings
    if "sku_mappings" in st.session_state:
        st.session_state.sku_mappings = data_processor.load_sku_mappings()
    
    # Log the refresh operation
    log_operation("refresh_data")

def sync_airtable():
    """Sync data with Airtable"""
    airtable_handler = AirtableHandler()
    
    # Perform sync operations
    try:
        # Example: Sync SKU mappings
        if "sku_mappings" in st.session_state and st.session_state.sku_mappings:
            # Your sync logic here
            pass
        
        # Log the sync operation
        log_operation("sync_airtable")
    except Exception as e:
        st.error(f"Error during Airtable sync: {str(e)}")

def clear_app_cache():
    """Clear application cache"""
    # List of cache keys to clear
    cache_keys = [
        "orders_df", 
        "inventory_df", 
        "processed_orders",
        "inventory_summary",
        "shortage_summary"
    ]
    
    for key in cache_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Log the cache clear operation
    log_operation("clear_cache")

def reset_session_state():
    """Reset the entire session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Log the session reset operation
    log_operation("reset_session")

def generate_report_data(data_type):
    """Generate sample data for reports based on data type"""
    try:
        # Initialize data processor
        data_processor = DataProcessor()
        
        if data_type == "inventory":
            # Try to get inventory data from session state or generate sample
            if "inventory_df" in st.session_state and st.session_state.inventory_df is not None:
                return st.session_state.inventory_df
            else:
                # Create sample inventory data
                return pd.DataFrame({
                    'SKU': [f'SKU-{i}' for i in range(1, 6)],
                    'Product': [f'Product {i}' for i in range(1, 6)],
                    'Warehouse': ['Oxnard', 'Wheeling', 'Oxnard', 'Wheeling', 'Oxnard'],
                    'Quantity': [100, 75, 50, 125, 30],
                    'Last Updated': [datetime.now().strftime("%Y-%m-%d")] * 5
                })
                
        elif data_type == "orders":
            # Try to get orders data from session state or generate sample
            if "orders_df" in st.session_state and st.session_state.orders_df is not None:
                return st.session_state.orders_df
            else:
                # Create sample orders data
                return pd.DataFrame({
                    'Order ID': [f'ORD-{uuid.uuid4().hex[:6].upper()}' for _ in range(5)],
                    'Customer': [f'Customer {i}' for i in range(1, 6)],
                    'Product': [f'Product {i}' for i in range(1, 6)],
                    'Quantity': [1, 2, 3, 1, 2],
                    'Status': ['Fulfilled', 'Pending', 'Processing', 'Fulfilled', 'Pending'],
                    'Order Date': [datetime.now().strftime("%Y-%m-%d")] * 5
                })
                
        elif data_type == "fulfillment":
            # Try to get processed orders or generate sample
            if "processed_orders" in st.session_state and st.session_state.processed_orders is not None:
                return st.session_state.processed_orders
            else:
                # Create sample fulfillment data
                return pd.DataFrame({
                    'Order ID': [f'ORD-{uuid.uuid4().hex[:6].upper()}' for _ in range(5)],
                    'Fulfillment Center': ['Oxnard', 'Wheeling', 'Oxnard', 'Wheeling', 'Oxnard'],
                    'Shipping Zone': [1, 3, 2, 4, 1],
                    'Delivery Service': ['UPS Ground', 'USPS Priority', 'UPS Ground', 'FedEx', 'UPS Ground'],
                    'Estimated Delivery': ['1-2 days', '3-4 days', '2-3 days', '4-5 days', '1-2 days']
                })
        else:
            # Default to a simple status report
            return pd.DataFrame({
                'Report Type': ['System Status'],
                'Status': ['Operational'],
                'Last Check': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                'Notes': ['Generated via hidden trigger']
            })
    except Exception as e:
        # Return error information if data generation fails
        return pd.DataFrame({
            'Error': ['Failed to generate report data'],
            'Details': [str(e)],
            'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })

def log_operation(operation_type, details=None):
    """Log the trigger operation"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "trigger_operations.log")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "operation": operation_type,
        "success": True
    }
    
    # Add additional details if provided
    if details:
        log_entry["details"] = details
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# Main function
def main():
    # Check for trigger parameters
    if check_trigger_params():
        # If trigger was executed, don't show the main content
        return
    
    # If no valid trigger parameters, show a minimal interface
    # This page will appear empty to regular users
    st.markdown("<div style='height: 100vh'></div>", unsafe_allow_html=True)
    
    # Hidden admin interface that only appears if admin=true parameter is present
    query_params = st.query_params
    if "admin" in query_params and query_params["admin"] == "true":
        # Verify the secret key is also present
        secret_key = os.getenv("TRIGGER_SECRET_KEY", "default_secret_key")
        provided_key = query_params.get("key", "")
        
        if provided_key == secret_key:
            st.header("Hidden Trigger Admin Panel", divider="rainbow")
            
            # Create a form for creating spreadsheets
            with st.form("create_spreadsheet_form"):
                st.subheader("Create Google Spreadsheet")
                title = st.text_input("Spreadsheet Title", "Fulfillment_Report")
                include_data = st.checkbox("Include Data", value=True)
                data_type = st.selectbox("Data Type", ["inventory", "orders", "fulfillment", "status", "picklist"])
                
                submitted = st.form_submit_button("Create Spreadsheet")
                
                if submitted:
                    with st.spinner("Creating spreadsheet..."):
                        # Generate data if requested
                        data = None
                        if include_data:
                            data = generate_report_data(data_type)
                            
                        # Create the spreadsheet
                        spreadsheet_id, result = create_spreadsheet(title, data, data_type)
                        
                        if spreadsheet_id:
                            st.success("Spreadsheet created successfully!")
                            st.markdown(f"[Open Spreadsheet]({result})")
                        else:
                            st.error(f"Failed to create spreadsheet: {result}")

if __name__ == "__main__":
    main()
