"""
Snapshot creation utilities for fulfillment projection and inventory data.
This module contains the core logic for creating snapshots without UI dependencies.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Add the parent directory to the Python path to fix import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.google_sheets import (
    GHF_AGGREGATION_DASHBOARD_ID, 
    ALL_PICKLIST_V2_SHEET_NAME,
    get_credentials
)
from utils.inventory_api import get_inventory_data, get_formatted_inventory

logger = logging.getLogger(__name__)

async def create_fulfillment_snapshot() -> Dict[str, Any]:
    """
    Create a UNIFIED snapshot with both projection and inventory data in one spreadsheet.
    
    Returns:
        Dict containing success status, spreadsheet URL, and other details
    """
    try:
        logger.info("🚀 Starting unified fulfillment snapshot creation...")
        
        # Get credentials
        creds = get_google_credentials()
        if not creds:
            logger.error("❌ Failed to get Google credentials")
            return {
                "success": False,
                "error": "Failed to get Google credentials"
            }
        
        # Build the Sheets service
        sheets_service = build('sheets', 'v4', credentials=creds)
        drive_service = build('drive', 'v3', credentials=creds)
        
        # Create a single spreadsheet for both projection and inventory
        logger.info("📊 Creating unified snapshot spreadsheet...")
        spreadsheet_id, spreadsheet_url = create_spreadsheet(
            "Fulfillment_Unified_Snapshot", 
            data_type="picklist"
        )
        
        projection_success = False
        inventory_success = False
        inventory_api_working = True  # Flag to track if inventory API is working
        
        if spreadsheet_id:
            logger.info(f"✅ Created unified spreadsheet: {spreadsheet_url}")
            
            # Copy the ALL_Picklist_V2 sheet
            logger.info("📋 Copying projection data...")
            projection_success = copy_sheet_with_related_data(
                sheets_service, 
                GHF_AGGREGATION_DASHBOARD_ID, 
                ALL_PICKLIST_V2_SHEET_NAME, 
                spreadsheet_id
            )
            
            if projection_success:
                logger.info("✅ Projection data copied successfully!")
            else:
                logger.error("❌ Failed to copy projection sheet data")
            
            # Add inventory data as a separate tab
            logger.info("📦 Adding inventory data as a separate tab...")
            
            # Temporarily skip inventory API testing and tab creation
            logger.info("Skipping inventory tab creation as requested")
            inventory_api_working = False
            inventory_success = False
        else:
            logger.error("❌ Failed to create spreadsheet")
        
        # Determine overall success - consider it successful if at least projection worked
        overall_success = projection_success
        
        if overall_success:
            logger.info("🎉 Unified snapshot creation completed successfully!")
            return {
                "success": True,
                "spreadsheet_url": spreadsheet_url,
                "spreadsheet_id": spreadsheet_id,
                "projection_success": projection_success,
                "inventory_success": inventory_success,
                "inventory_api_working": inventory_api_working,
                "message": "Unified snapshot created successfully"
            }
        else:
            logger.error("❌ Unified snapshot creation failed")
            return {
                "success": False,
                "error": "Failed to create unified snapshot"
            }
            
    except Exception as e:
        logger.exception(f"❌ Error in create_fulfillment_snapshot: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def create_fulfillment_snapshot_sync() -> Dict[str, Any]:
    """
    Synchronous wrapper for the async snapshot creation function.
    Creates a UNIFIED snapshot with both projection and inventory data in one spreadsheet.
    
    Returns:
        Dict containing success status, spreadsheet URL, and other details
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(create_fulfillment_snapshot())
    return result

def create_separate_snapshots() -> Dict[str, Any]:
    """
    Create separate snapshots for projection and inventory (legacy approach).
    
    Returns:
        Dict containing success status and details for both snapshots
    """
    try:
        logger.info("🚀 Starting separate snapshots creation...")
        
        # Get credentials
        creds = get_google_credentials()
        if not creds:
            logger.error("❌ Failed to get Google credentials")
            return {
                "success": False,
                "error": "Failed to get Google credentials"
            }
        
        # Build the Sheets service
        sheets_service = build('sheets', 'v4', credentials=creds)
        
        # Create the projection spreadsheet
        logger.info("📊 Creating projection snapshot spreadsheet...")
        projection_spreadsheet_id, projection_spreadsheet_url = create_spreadsheet(
            "Fulfillment_Projection_Snapshot", 
            data_type="picklist"
        )
        
        # Create the inventory spreadsheet
        logger.info("📦 Creating inventory snapshot spreadsheet...")
        inventory_spreadsheet_id, inventory_spreadsheet_url = create_spreadsheet(
            "ColdCart_Inventory_Snapshot", 
            data_type="coldcart_inventory"
        )
        
        projection_success = False
        inventory_success = False
        
        # Handle projection spreadsheet
        if projection_spreadsheet_id:
            logger.info(f"✅ Created projection spreadsheet: {projection_spreadsheet_url}")
            
            # Copy the ALL_Picklist_V2 sheet
            success = copy_sheet_with_related_data(
                sheets_service, 
                GHF_AGGREGATION_DASHBOARD_ID, 
                ALL_PICKLIST_V2_SHEET_NAME, 
                projection_spreadsheet_id
            )
            
            if success:
                logger.info("🎉 Fulfillment projection snapshot created successfully!")
                projection_success = True
            else:
                logger.error("❌ Failed to copy projection sheet data")
        else:
            logger.error("❌ Failed to create projection spreadsheet")
        
        # Handle inventory spreadsheet
        if inventory_spreadsheet_id:
            logger.info(f"✅ Created inventory spreadsheet: {inventory_spreadsheet_url}")
            inventory_success = True
        else:
            logger.error("❌ Failed to create inventory spreadsheet")
        
        # Return results
        overall_success = projection_success or inventory_success
        
        result = {
            "success": overall_success,
            "projection": {
                "success": projection_success,
                "spreadsheet_id": projection_spreadsheet_id,
                "spreadsheet_url": projection_spreadsheet_url
            },
            "inventory": {
                "success": inventory_success,
                "spreadsheet_id": inventory_spreadsheet_id,
                "spreadsheet_url": inventory_spreadsheet_url
            }
        }
        
        if overall_success:
            logger.info("🎉 Separate snapshots creation completed!")
        else:
            logger.error("❌ Both snapshots failed")
            result["error"] = "Both snapshots failed"
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in separate snapshots creation: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
        }

# ============================================================================
# ALL FUNCTIONS FROM HIDDEN_TRIGGER.PY
# ============================================================================

# Google Drive folder ID where the spreadsheet will be created
# This is the ID of the "Projection Snapshots" folder from the link
DRIVE_FOLDER_ID = "1-uUvyCTEx_TLKOF46jHD3Kpsp8aO9W9b"  # The folder ID from the provided URL
ENABLE_FOLDER_MOVING = True

# Google API scopes
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

def get_google_credentials():
    """Get Google API credentials for accessing Drive and Sheets using the exact same approach as google_sheets.py"""
    try:
        # Use the proven get_credentials function from google_sheets.py but with Drive scopes
        # First get the base credentials
        base_creds = get_credentials()
        
        # Create new credentials with the required scopes for Drive and Sheets
        if hasattr(base_creds, 'with_scopes'):
            creds = base_creds.with_scopes(SCOPES)
        else:
            creds = base_creds
            
        logger.info("✅ Successfully obtained Google credentials")
        return creds
    except Exception as e:
        logger.error(f"❌ Error getting Google credentials: {str(e)}")
        logger.error("Could not authenticate. Make sure you have either the JSON file locally or have run 'gcloud auth application-default login'")
        return None

def create_spreadsheet(title, data=None, data_type="inventory"):
    """Create a new spreadsheet in the specified Google Drive folder"""
    try:
        logger.info(f"Creating spreadsheet: {title}")
        
        # Get credentials
        creds = get_google_credentials()
        if not creds:
            logger.error("Failed to get Google credentials")
            return None, None
        
        # Build services
        sheets_service = build('sheets', 'v4', credentials=creds)
        drive_service = build('drive', 'v3', credentials=creds)
        
        # Create proper naming format with date
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H%M")
        
        # Use proper naming convention based on data type
        if data_type == "picklist":
            spreadsheet_title = f"Fulfillment_Projection_{date_str}_{time_str}"
        elif data_type == "coldcart_inventory":
            spreadsheet_title = f"ColdCart_Inventory_{date_str}_{time_str}"
        else:
            spreadsheet_title = f"{title}_{date_str}_{time_str}"
        
        logger.info(f"Creating spreadsheet with title: {spreadsheet_title}")
        
        # Create year/month directory structure
        year_str = now.strftime("%Y")
        month_str = now.strftime("%m-%B")  # e.g., "07-July"
        
        # First, check if the base folder exists
        try:
            # Try to verify base folder exists
            try:
                base_folder = drive_service.files().get(fileId=DRIVE_FOLDER_ID).execute()
                logger.info(f"Base folder found: {base_folder.get('name')}")
            except Exception as e:
                logger.warning(f"Could not verify base folder: {str(e)}")
            
            # Create or find year folder
            logger.info(f"Creating/finding year folder: {year_str} in parent folder: {DRIVE_FOLDER_ID}")
            year_folder_id = create_or_find_folder(drive_service, year_str, DRIVE_FOLDER_ID)
            logger.info(f"Year folder ready: {year_str} (ID: {year_folder_id})")
            
            # Create or find month folder within year folder
            logger.info(f"Creating/finding month folder: {month_str} in year folder: {year_folder_id}")
            month_folder_id = create_or_find_folder(drive_service, month_str, year_folder_id)
            logger.info(f"Month folder ready: {month_str} (ID: {month_folder_id})")
            
            # Set the target folder to the month folder
            target_folder_id = month_folder_id
            
            # Create a new spreadsheet with proper sheet name
            sheet_name = "ALL_Picklist_V2" if data_type == "picklist" else "ColdCart_Inventory"
            
            spreadsheet = {
                'properties': {
                    'title': spreadsheet_title
                },
                'sheets': [
                    {
                        'properties': {
                            'title': sheet_name,
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
            
            # Move the file to the target folder
            file = drive_service.files().update(
                fileId=spreadsheet_id,
                addParents=target_folder_id,
                removeParents='root',
                fields='id, parents'
            ).execute()
            
            logger.info(f"Spreadsheet '{spreadsheet_title}' created in {year_str}/{month_str} folder successfully")
            
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
            logger.info(f"Spreadsheet shared with link: {web_link}")
            
            return spreadsheet_id, web_link
            
        except Exception as folder_error:
            logger.error(f"Target folder structure issue: {str(folder_error)}")
            # Try to get service account info for error message
            try:
                if hasattr(creds, 'service_account_email'):
                    logger.info(f"Please make sure the folder exists and is shared with the service account: {creds.service_account_email}")
            except Exception:
                pass
                
            # Fallback to simple creation without folders - this matches the original code
            logger.warning("⚠️ Falling back to simple spreadsheet creation without folder structure")
            spreadsheet_body = {
                'properties': {
                    'title': spreadsheet_title
                }
            }
            
            spreadsheet = sheets_service.spreadsheets().create(body=spreadsheet_body).execute()
            spreadsheet_id = spreadsheet['spreadsheetId']
            spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
            
            logger.info(f"✅ Spreadsheet created (without folder structure): {spreadsheet_url}")
            return spreadsheet_id, spreadsheet_url
        
    except Exception as e:
        logger.error(f"Error creating spreadsheet: {str(e)}")
        return None, None

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
            logger.info(f"Found existing folder: {folder_name} (ID: {folder_id})")
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
            logger.info(f"Created new folder: {folder_name} (ID: {folder_id})")
            return folder_id
            
    except Exception as e:
        logger.error(f"Error creating/finding folder {folder_name}: {str(e)}")
        # Fallback to parent folder - this is what the original code did
        logger.warning(f"⚠️ Falling back to parent folder {parent_folder_id} for {folder_name}")
        return parent_folder_id

def test_source_access(sheets_service, source_spreadsheet_id, source_sheet_name):
    """Test if we can access the source spreadsheet and sheet"""
    try:
        # Get spreadsheet metadata
        spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=source_spreadsheet_id).execute()
        
        # Check if the specific sheet exists
        sheet_found = False
        for sheet in spreadsheet.get('sheets', []):
            if sheet['properties']['title'] == source_sheet_name:
                sheet_found = True
                break
        
        if not sheet_found:
            logger.error(f"Sheet '{source_sheet_name}' not found in source spreadsheet")
            return False
        
        # Try to read a small range to test access
        range_name = f"{source_sheet_name}!A1:B2"
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=source_spreadsheet_id,
            range=range_name
        ).execute()
        
        logger.info(f"✅ Successfully accessed source sheet: {source_sheet_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cannot access source sheet: {str(e)}")
        return False

def copy_sheet_with_related_data(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id):
    """Copy the main sheet and any related sheets used in calculations"""
    try:
        # First test if we can access the source
        if not test_source_access(sheets_service, source_spreadsheet_id, source_sheet_name):
            logger.error("Cannot access source spreadsheet - aborting copy")
            return False
        
        # Get source spreadsheet metadata (without grid data to avoid timeout)
        source_spreadsheet = sheets_service.spreadsheets().get(
            spreadsheetId=source_spreadsheet_id
        ).execute()
        
        # Copy only the main ALL_Picklist_V2 sheet with data and styling (no formulas)
        logger.info(f"Copying {source_sheet_name} with data and styling (no formulas)...")
        
        success = copy_sheet_data_with_styling(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id)
        if success:
            logger.info(f"✓ Successfully copied {source_sheet_name} with data and styling")
            
            # Remove the default "Sheet1" that was created automatically
            try:
                logger.info("Removing default Sheet1...")
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
                    logger.info("✓ Removed default Sheet1")
                else:
                    logger.info("No default Sheet1 found to remove")
                    
            except Exception as e:
                logger.warning(f"Could not remove default Sheet1: {str(e)}")
                # This is not critical, continue anyway
            
            return True
        else:
            logger.error(f"✗ Failed to copy {source_sheet_name}")
            return False
        
    except Exception as e:
        logger.error(f"Error copying related sheets: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def copy_single_sheet_with_data(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id):
    """Copy a single sheet with all data, formulas, and formatting preserved using Google Sheets copyTo API"""
    try:
        # Get source spreadsheet metadata to find the sheet ID
        source_spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=source_spreadsheet_id).execute()
        
        source_sheet_id = None
        for sheet in source_spreadsheet.get('sheets', []):
            if sheet['properties']['title'] == source_sheet_name:
                source_sheet_id = sheet['properties']['sheetId']
                break
        
        if source_sheet_id is None:
            logger.error(f"Sheet '{source_sheet_name}' not found in source spreadsheet")
            return False
        
        # Copy the sheet using copyTo API
        copy_request = {
            'destinationSpreadsheetId': target_spreadsheet_id
        }
        
        response = sheets_service.spreadsheets().sheets().copyTo(
            spreadsheetId=source_spreadsheet_id,
            sheetId=source_sheet_id,
            body=copy_request
        ).execute()
        
        logger.info(f"✅ Sheet copied successfully using copyTo API")
        return True
        
    except Exception as e:
        logger.error(f"Error in copy_single_sheet_with_data: {str(e)}")
        return False

def copy_sheet_data_with_styling(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id):
    """Copy sheet data with styling but no formulas - gets actual calculated values"""
    try:
        logger.info(f"Copying {source_sheet_name} with data and styling...")
        
        # 1. Get the source sheet information
        logger.info("Getting source sheet information...")
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
            logger.error(f"Source sheet {source_sheet_name} not found")
            return False
        
        logger.info(f"Found source sheet ID: {source_sheet_id}")
        
        # 2. First, detect the actual used range to avoid empty rows/columns
        logger.info(f"Detecting actual data range in {source_sheet_name}...")
        
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
            if any(cell.strip() for cell in row if isinstance(cell, str)):  # Check if row has any non-empty cells
                last_row_with_data = i + 1
        
        # Find the actual last column with data
        last_col_with_data = 0
        for row in all_values[:last_row_with_data]:
            for j, cell in enumerate(row):
                if cell and isinstance(cell, str) and cell.strip():
                    last_col_with_data = max(last_col_with_data, j + 1)
        
        # Trim to actual data range
        values = all_values[:last_row_with_data] if last_row_with_data > 0 else []
        
        logger.info(f"Actual data range: {last_row_with_data} rows x {last_col_with_data} columns")
        logger.info(f"Found {len(values)} rows of actual data (trimmed empty rows)")
        
        if not values:
            logger.warning(f"No data found in {source_sheet_name}")
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
        logger.info(f"Finding existing sheet: {source_sheet_name}")
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
                # Create the sheet if it doesn't exist
                add_sheet_request = {
                    'addSheet': {
                        'properties': sheet_properties
                    }
                }
                
                response = sheets_service.spreadsheets().batchUpdate(
                    spreadsheetId=target_spreadsheet_id,
                    body={'requests': [add_sheet_request]}
                ).execute()
                
                new_sheet_id = response['replies'][0]['addSheet']['properties']['sheetId']
                logger.info(f"Created new sheet with ID: {new_sheet_id}")
            else:
                logger.info(f"Found existing sheet with ID: {new_sheet_id}")
                
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
                        logger.info(f"Setting frozen rows: {source_grid['frozenRowCount']}")
                    if 'frozenColumnCount' in source_grid:
                        grid_properties['frozenColumnCount'] = source_grid['frozenColumnCount']
                        logger.info(f"Setting frozen columns: {source_grid['frozenColumnCount']}")
                
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
                    logger.info("Updated sheet properties")
                    
        except Exception as e:
            logger.error(f"Failed to configure sheet: {str(e)}")
            return False
        
        # 4. Write the data
        logger.info(f"Writing {len(values)} rows of data...")
        try:
            sheets_service.spreadsheets().values().update(
                spreadsheetId=target_spreadsheet_id,
                range=f"{source_sheet_name}!A1",
                valueInputOption="RAW",
                body={
                    "values": values
                }
            ).execute()
            logger.info(f"Data written successfully")
        except Exception as e:
            logger.error(f"Failed to write data: {str(e)}")
            return False
        
        # 5. Copy comprehensive formatting using the actual data range
        logger.info("Copying formatting and styling...")
        try:
            # Use the actual data range we detected for formatting
            end_col_letter = chr(ord('A') + min(last_col_with_data - 1, 25)) if last_col_with_data <= 26 else 'Z'
            format_range = f"{source_sheet_name}!A1:{end_col_letter}{last_row_with_data}"
            
            logger.info(f"Getting formatting for range: {format_range}")
            
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
                        logger.info(f"Applied {len(format_requests)} formatting and styling rules")
        except Exception as e:
            logger.warning(f"Could not copy all formatting: {str(e)}")
            # This is not critical, continue anyway
        
        # 6. Copy filters if they exist
        logger.info("Checking for filters...")
        try:
            # Check if the source sheet has filters
            source_sheet_full = None
            for sheet in source_spreadsheet['sheets']:
                if sheet['properties']['sheetId'] == source_sheet_id:
                    source_sheet_full = sheet
                    break
            
            if source_sheet_full and 'basicFilter' in source_sheet_full:
                basic_filter = source_sheet_full['basicFilter']
                logger.info(f"Found basic filter: {basic_filter}")
                
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
                
                logger.info("Successfully copied filters!")
            else:
                # If no existing filter, create a basic filter for the data range
                logger.info("No existing filter found, creating basic filter for data range...")
                
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
                
                logger.info("Created basic filter for data range")
        except Exception as e:
            logger.warning(f"Could not copy or create filters: {str(e)}")
            # This is not critical, continue anyway
        
        logger.info(f"✅ Successfully copied {source_sheet_name} with data and styling")
        return True
        
    except Exception as e:
        logger.error(f"Error in copy_sheet_data_with_styling: {str(e)}")
        return False

def copy_sheet_data_fallback(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id):
    """Fallback method to copy sheet data when copyTo API fails"""
    try:
        logger.info(f"Using fallback method to copy: {source_sheet_name}")
        
        # Use the styling method as fallback
        return copy_sheet_data_with_styling(sheets_service, source_spreadsheet_id, source_sheet_name, target_spreadsheet_id)
        
    except Exception as e:
        logger.error(f"Error in copy_sheet_data_fallback: {str(e)}")
        return False

def add_inventory_sheet(sheets_service, spreadsheet_id: str, df, sheet_name: str) -> bool:
    """Helper function to add a DataFrame as a new sheet to the spreadsheet with formatting."""
    try:
        if df is None or df.empty:
            logger.warning(f"⚠️ No data available for sheet {sheet_name}")
            return False
            
        # Add new sheet and capture the response to get the sheet ID
        add_sheet_request = {
            'addSheet': {
                'properties': {
                    'title': sheet_name,
                    'gridProperties': {
                        'rowCount': len(df) + 10,
                        'columnCount': len(df.columns) + 5
                    }
                }
            }
        }
        
        response = sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={'requests': [add_sheet_request]}
        ).execute()
        
        # Extract the sheet ID from the response
        sheet_id = None
        if 'replies' in response and len(response['replies']) > 0:
            if 'addSheet' in response['replies'][0]:
                sheet_id = response['replies'][0]['addSheet']['properties']['sheetId']
        
        if sheet_id is None:
            logger.warning("⚠️ Could not determine sheet ID, using default formatting")
            # Get spreadsheet info to find the sheet ID
            spreadsheet_info = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            for sheet in spreadsheet_info.get('sheets', []):
                if sheet.get('properties', {}).get('title') == sheet_name:
                    sheet_id = sheet.get('properties', {}).get('sheetId')
                    break
        
        logger.info(f"📝 Using sheet ID: {sheet_id}")
        
        # Convert DataFrame to values list for the API
        values = [df.columns.tolist()] + df.values.tolist()
        
        # Update the spreadsheet with inventory data
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=f'{sheet_name}!A1',
            valueInputOption='RAW',
            body={'values': values}
        ).execute()
        
        return sheet_id
    except Exception as e:
        logger.error(f"Error adding sheet {sheet_name}: {str(e)}")
        return None

def add_inventory_tab_to_spreadsheet(sheets_service, spreadsheet_id: str) -> bool:
    """Add inventory data as separate tabs to the existing spreadsheet."""
    try:
        logger.info("📦 Fetching current ColdCart inventory...")
        summary_df, detailed_df = get_formatted_inventory()
        
        if (summary_df is None or summary_df.empty) and (detailed_df is None or detailed_df.empty):
            logger.warning("⚠️ No inventory data available")
            return False
            
        # Add summary inventory tab
        summary_sheet_name = "ColdCart_Inventory_Summary"
        summary_sheet_id = add_inventory_sheet(sheets_service, spreadsheet_id, summary_df, summary_sheet_name)
        
        # Add detailed inventory tab if available
        detailed_sheet_id = None
        if detailed_df is not None and not detailed_df.empty:
            detailed_sheet_name = "ColdCart_Inventory_Detailed"
            detailed_sheet_id = add_inventory_sheet(sheets_service, spreadsheet_id, detailed_df, detailed_sheet_name)
            
        # Apply formatting to both sheets
        success = False
        
        # Format summary sheet if it was created
        if summary_sheet_id:
            success = format_inventory_sheet(sheets_service, spreadsheet_id, summary_sheet_id, summary_sheet_name)
            if success:
                logger.info(f"✅ Added and formatted inventory summary data in sheet: {summary_sheet_name}")
            
        # Format detailed sheet if it was created
        if detailed_sheet_id:
            success = format_inventory_sheet(sheets_service, spreadsheet_id, detailed_sheet_id, detailed_sheet_name)
            if success:
                logger.info(f"✅ Added and formatted detailed inventory data in sheet: {detailed_sheet_name}")
                
        return success
    except Exception as e:
        logger.error(f"Error adding inventory tabs: {str(e)}")
        return False

def format_inventory_sheet(sheets_service, spreadsheet_id, sheet_id, sheet_name):
    """Apply formatting to an inventory sheet"""
    try:
        # Add formatting - blue headers, bold text, frozen rows
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={
                'requests': [
                    # Blue header with white text
                    {
                        'repeatCell': {
                            'range': {
                                'sheetId': sheet_id,  # Use the actual sheet ID
                                'startRowIndex': 0,
                                'endRowIndex': 1
                                },
                                'cell': {
                                    'userEnteredFormat': {
                                        'backgroundColor': {
                                            'red': 0.2,
                                            'green': 0.6,
                                            'blue': 0.9
                                        },
                                        'textFormat': {
                                            'bold': True,
                                            'foregroundColor': {
                                                'red': 1.0,
                                                'green': 1.0,
                                                'blue': 1.0
                                            }
                                        }
                                    }
                                },
                                'fields': 'userEnteredFormat(backgroundColor,textFormat)'
                            }
                        },
                        # Freeze header row
                        {
                            'updateSheetProperties': {
                                'properties': {
                                    'sheetId': sheet_id,  # Use the actual sheet ID
                                    'gridProperties': {
                                        'frozenRowCount': 1
                                    }
                                },
                                'fields': 'gridProperties.frozenRowCount'
                            }
                        }
                    ]
                }
            ).execute()
            
        logger.info("✅ Inventory data formatted with headers and frozen rows")
        logger.info(f"✅ Successfully formatted inventory data in '{sheet_name}' tab")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in format_inventory_sheet: {str(e)}")
        return False


if __name__ == "__main__":
    """Create real snapshots with real data and formatting."""
    logging.basicConfig(level=logging.INFO)
    
    print("\n🚀 CREATING REAL PRODUCTION SNAPSHOTS...\n")
    
    # Run the snapshot creation
    result = create_fulfillment_snapshot_sync()
    
    # Show results
    print("\n=== SNAPSHOT CREATION RESULTS ===")
    print(f"Success: {result.get('success')}")
    
    if result.get('success'):
        if result.get('projection_success'):
            print(f"\n📊 PROJECTION SNAPSHOT CREATED!")
            print(f"URL: {result.get('projection_spreadsheet_url')}")
            print(f"ID: {result.get('projection_spreadsheet_id')}")
        
        if result.get('inventory_success'):
            print(f"\n📦 INVENTORY SNAPSHOT CREATED!")
            print(f"URL: {result.get('inventory_spreadsheet_url')}")
            print(f"ID: {result.get('inventory_spreadsheet_id')}")
    else:
        print(f"\n❌ SNAPSHOT CREATION FAILED: {result.get('error')}")
    
    print("\n=== FULL RESULT OBJECT ===")
    print(result)
    

    