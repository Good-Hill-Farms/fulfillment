import os
import requests
import pandas as pd
from io import StringIO
import tempfile
import pytz
from datetime import datetime

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
        date_str = f"Data as of {current_time_la.strftime('%Y-%m-%d %H:%M:%S')} (Los Angeles)"
        worksheet.write(0, 0, date_str, date_header_format)
        
        # Add some cell formats
        header_format = workbook.add_format({
            'bold': True,
            'font_color': 'white',
            'bg_color': '#0066cc',
            'align': 'center',
            'valign': 'top',
            'border': 1
        })
        
        input_format = workbook.add_format({
            'bg_color': '#e6ffe6',  # Light green background
            'align': 'center',
            'valign': 'top',
            'border': 1
        })
        
        lot_format = workbook.add_format({
            'bg_color': '#cce6ff',
            'align': 'center',
            'valign': 'top',
            'border': 1
        })
        
        notes_format = workbook.add_format({
            'bg_color': '#e6ffe6',  # Light green background
            'align': 'left',
            'valign': 'top',
            'text_wrap': True,
            'border': 1
        })
        
        batch_format = workbook.add_format({
            'text_wrap': True,
            'align': 'left',
            'valign': 'top',
            'border': 1
        })
        
        # Default format for other columns
        default_format = workbook.add_format({
            'align': 'left',
            'valign': 'top',
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
                if col in ['Counted QTY', 'Unit']:
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
            'ItemId': 10,
            'Sku': 20,
            'Name': 30,
            'BatchCode': 15,
            'Expected AvailableQty (ea)': 15,
            'Counted QTY': 10,
            'LOT': 10,
            'Notes': 30,
            'BatchDetails': 50
        }
        
        # Apply column widths for existing columns
        for col, width in col_widths.items():
            if col in df.columns:
                col_letter = chr(ord('A') + df.columns.get_loc(col))
                worksheet.set_column(f'{col_letter}:{col_letter}', width)
        
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
    date_str = f"Data as of {current_time_la.strftime('%Y-%m-%d %H:%M:%S')} (Los Angeles)"
    output.write(date_str + '\n\n')  # Add an empty line after the date
    
    # Write DataFrame to the buffer
    df.to_csv(output, index=False)
    
    # Get the complete string and write to file
    with open(filepath, 'w', newline='') as f:
        f.write(output.getvalue())
    
    return filepath 