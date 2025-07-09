import os
import requests
import pandas as pd
from io import StringIO
import tempfile
import pytz
from datetime import datetime, timedelta
import re

def get_inventory_data():
    """
    Fetch inventory data from the ColdCart API
    Returns a pandas DataFrame with the inventory data or None if the API token is missing
    """
    api_token = os.getenv('COLDCART_API_TOKEN')
    if not api_token:
        return None
        
    api_url = "https://api-direct.coldcartfulfill.com/inventory/242/items/export"
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Origin": "https://portal.coldcartfulfill.com",
        "Referer": "https://portal.coldcartfulfill.com/",
        "Accept": "*/*",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        if not response.text.strip():
            return None
        
        # Convert response to DataFrame
        df = pd.read_csv(StringIO(response.text))
        
        # Check if DataFrame is empty
        if df.empty:
            return None
            
        return df
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch inventory data: {str(e)}")
    except pd.errors.EmptyDataError:
        return None
    except Exception as e:
        raise Exception(f"Error processing inventory data: {str(e)}")

def save_as_excel(df, filename, colorful=False):
    """Save DataFrame as Excel file with optional colorful formatting"""
    filepath = os.path.join(tempfile.gettempdir(), filename)
    
    # Create a new Excel writer object
    writer = pd.ExcelWriter(
        filepath,
        engine='xlsxwriter',
        engine_kwargs={'options': {'nan_inf_to_errors': True}}
    )
    
    # Replace NaN values with empty strings before writing
    df = df.fillna('')
    
    # Write the DataFrame to Excel, starting from row 1 to leave space for the date header
    df.to_excel(writer, index=False, sheet_name='Inventory', startrow=1)
    
    if colorful:
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Inventory']
        
        # Add date header format
        date_header_format = workbook.add_format({
            'bold': True,
            'font_size': 12,
            'align': 'left',
            'valign': 'top'
        })
        
        # Add current date in Los Angeles timezone
        la_tz = pytz.timezone('America/Los_Angeles')
        current_time_la = datetime.now(la_tz)
        date_str = f"Data as of {current_time_la.strftime('%B %d, %Y %H:%M:%S')} (Los Angeles)"
        worksheet.write(0, 0, date_str, date_header_format)
        
        # Add some cell formats
        header_format = workbook.add_format({
            'bold': True,
            'font_color': 'white',
            'bg_color': '#0066cc',
            'align': 'center',
            'valign': 'vcenter',  # Changed from 'top' to 'vcenter'
            'border': 1
        })
        
        input_format = workbook.add_format({
            'bg_color': '#e6ffe6',  # Light green background
            'align': 'center',
            'valign': 'vcenter',  # Changed from 'top' to 'vcenter'
            'border': 1
        })
        
        lot_format = workbook.add_format({
            'bg_color': '#cce6ff',
            'align': 'center',
            'valign': 'vcenter',  # Changed from 'top' to 'vcenter'
            'border': 1
        })
        
        notes_format = workbook.add_format({
            'bg_color': '#e6ffe6',  # Light green background
            'align': 'left',
            'valign': 'vcenter',  # Changed from 'top' to 'vcenter'
            'text_wrap': True,
            'border': 1
        })
        
        batch_format = workbook.add_format({
            'text_wrap': True,
            'align': 'left',
            'valign': 'vcenter',  # Changed from 'top' to 'vcenter'
            'border': 1
        })
        
        # Default format for other columns
        default_format = workbook.add_format({
            'align': 'center',  # Changed from 'left' to 'center'
            'valign': 'vcenter',  # Changed from 'top' to 'vcenter'
            'border': 1
        })
        
        # Format the header row (now in row 1 instead of 0)
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(1, col_num, value, header_format)
        
        # Format the data columns (now starting from row 2 instead of 1)
        for row in range(2, len(df) + 2):
            # Write each column only if it exists in the DataFrame
            for col in df.columns:
                col_idx = df.columns.get_loc(col)
                value = df.iloc[row-2][col]
                
                # Choose format based on column name
                if col in ['Counted QTY', 'Unit (lb, ea, cs)']:  # Updated Unit column name
                    fmt = input_format
                elif col == 'Notes':
                    fmt = notes_format
                elif col == 'LOT':
                    fmt = lot_format
                elif col in ['BatchCode', 'BatchDetails']:
                    fmt = batch_format
                else:
                    fmt = default_format
                
                worksheet.write(row, col_idx, value, fmt)
        
        # Set column widths
        col_widths = {
            'ItemId': 15,
            'Name': 50,  # Increased from 30
            'Sku': 25,   # Increased from 20
            'LOT': 15,   # Increased from 10
            'Expected AvailableQty (ea)': 20,  # Increased from 15
            'Counted QTY': 25,  # Increased from 15
            'Unit (lb, ea, cs)': 20,  # Increased from 15
            'Notes': 60,  # Increased from 40
            'BatchCode': 20,    # Increased from 15
            'BatchDetails': 60  # Increased from 50
        }
        
        # Apply column widths for existing columns
        for col, width in col_widths.items():
            if col in df.columns:
                col_letter = chr(ord('A') + df.columns.get_loc(col))
                worksheet.set_column(f'{col_letter}:{col_letter}', width)
        
        # Set row height for data rows (make them taller)
        for row in range(2, len(df) + 2):
            worksheet.set_row(row, 35)  # Increased from 25 to 35
            
        # Set header row slightly taller
        worksheet.set_row(1, 40)  # Increased from 30 to 40
        
        # Freeze the header row (now row 1 instead of 0)
        worksheet.freeze_panes(2, 0)
    
    # Save the workbook
    writer.close()
    
    return filepath

def save_as_csv(df, filename):
    """Save DataFrame as CSV file"""
    filepath = os.path.join(tempfile.gettempdir(), filename)
    
    # Create a string buffer
    output = StringIO()
    
    # Add current date in Los Angeles timezone
    la_tz = pytz.timezone('America/Los_Angeles')
    current_time_la = datetime.now(la_tz)
    date_str = f"Data as of {current_time_la.strftime('%B %d, %Y %H:%M:%S')} (Los Angeles)"
    output.write(date_str + '\n\n')  # Add an empty line after the date
    
    # Write DataFrame to the buffer
    df.to_csv(output, index=False)
    
    # Get the complete string and write to file
    with open(filepath, 'w', newline='') as f:
        f.write(output.getvalue())
    
    return filepath 

def get_formatted_inventory(include_batch_details=True, highlight_old=True):
    """
    Get formatted inventory data with optional batch code details and age highlighting
    
    Args:
        include_batch_details (bool): Whether to include items with batch codes
        highlight_old (bool): Whether to highlight items older than 2 weeks
        
    Returns:
        tuple: (summary_df, detailed_df) or None if no data
    """
    df = get_inventory_data()
    if df is None:
        return None, None
        
    # Consolidate Moorpark and Oxnard warehouses
    df['WarehouseName'] = df['WarehouseName'].replace({
        'CA-Moorpark-93021': 'Oxnard',
        'CA-Oxnard-93030': 'Oxnard'
    })
        
    # Extract fruit name from SKU if possible
    df['Fruit'] = df['Sku'].str.extract(r'^([^_-]+)').fillna(df['Name'].str.split().str[0])
    
    # Filter out zero quantity items
    df = df[df['AvailableQty'] > 0]
    
    # Create summary by SKU and Warehouse
    summary_df = df.groupby(['Fruit', 'Sku', 'Name', 'WarehouseName'])['AvailableQty'].sum().reset_index()
    summary_df = summary_df.sort_values(['Fruit', 'AvailableQty'], ascending=[True, False])
    
    # For detailed view with batch codes
    if include_batch_details:
        # Filter for items with batch codes
        detailed_df = df[df['BatchCode'].notna() & (df['BatchCode'] != '')].copy()
        
        if not detailed_df.empty:
            # Extract delivery date from batch code
            detailed_df['DeliveryDate'] = detailed_df['BatchCode'].apply(extract_delivery_date)
            
            # Convert to PST for comparison
            pst = pytz.timezone('America/Los_Angeles')
            now_pst = datetime.now(pst)
            two_weeks_ago = now_pst - timedelta(days=14)
            
            # Add age indicator
            detailed_df['IsOld'] = detailed_df['DeliveryDate'].apply(
                lambda x: x < two_weeks_ago if pd.notna(x) else False
            )
            
            # Sort by delivery date (newest first) and quantity
            detailed_df = detailed_df.sort_values(
                ['DeliveryDate', 'AvailableQty'],
                ascending=[False, False]
            )
            
            # Select and rename columns for display
            detailed_df = detailed_df[[
                'Fruit', 'Sku', 'Name', 'AvailableQty', 'BatchCode',
                'DeliveryDate', 'IsOld', 'WarehouseName'
            ]]
            
            return summary_df, detailed_df
    
    return summary_df, None

def extract_delivery_date(batch_code):
    """Extract delivery date from batch code string"""
    if not isinstance(batch_code, str):
        return None
        
    # Look for date pattern MMDDYY
    match = re.search(r'delivered_(\d{6})', batch_code)
    if match:
        date_str = match.group(1)
        try:
            # Convert MMDDYY to datetime
            date = datetime.strptime(date_str, '%m%d%y')
            # Make timezone-aware in PST
            pst = pytz.timezone('America/Los_Angeles')
            return pst.localize(date)
        except ValueError:
            return None
    return None

if __name__ == "__main__":
    summary_df, detailed_df = get_formatted_inventory()
    
    if summary_df is not None:
        print("\n=== Total Inventory by SKU ===")
        print(summary_df)
        
        if detailed_df is not None:
            print("\n=== Detailed Inventory with Batch Codes ===")
            print("\nOlder than 2 weeks (PST):")
            old_items = detailed_df[detailed_df['IsOld']]
            if not old_items.empty:
                print(old_items[['Fruit', 'Sku', 'AvailableQty', 'DeliveryDate', 'WarehouseName']])
            else:
                print("No items older than 2 weeks")
                
            print("\nNewer items:")
            new_items = detailed_df[~detailed_df['IsOld']]
            if not new_items.empty:
                print(new_items[['Fruit', 'Sku', 'AvailableQty', 'DeliveryDate', 'WarehouseName']])
            else:
                print("No items newer than 2 weeks")
    else:
        print("No inventory data available") 