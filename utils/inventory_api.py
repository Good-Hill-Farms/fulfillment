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
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
    
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
        
        qty_format = workbook.add_format({
            'bg_color': '#b3d9ff',
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
            'bg_color': '#e6f2ff',
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
            # Default format for standard columns
            worksheet.write(row, df.columns.get_loc('ItemId'), df.iloc[row-2]['ItemId'], default_format)
            worksheet.write(row, df.columns.get_loc('Sku'), df.iloc[row-2]['Sku'], default_format)
            worksheet.write(row, df.columns.get_loc('Name'), df.iloc[row-2]['Name'], default_format)
            worksheet.write(row, df.columns.get_loc('Type'), df.iloc[row-2]['Type'], default_format)
            
            # Handle either AvailableQty or Expected AvailableQty
            qty_col = 'Expected AvailableQty' if 'Expected AvailableQty' in df.columns else 'AvailableQty'
            worksheet.write(row, df.columns.get_loc(qty_col), df.iloc[row-2][qty_col], default_format)
            
            # Only write QTY/LOT columns if they exist
            for col in ['QTY_1', 'QTY_2', 'QTY_3']:
                if col in df.columns:
                    worksheet.write(row, df.columns.get_loc(col), df.iloc[row-2][col], qty_format)
            
            for col in ['LOT_1', 'LOT_2', 'LOT_3']:
                if col in df.columns:
                    worksheet.write(row, df.columns.get_loc(col), df.iloc[row-2][col], lot_format)
            
            # Only write Notes if it exists
            if 'Notes' in df.columns:
                worksheet.write(row, df.columns.get_loc('Notes'), df.iloc[row-2]['Notes'], notes_format)
            
            # Only write BatchDetails if it exists
            if 'BatchDetails' in df.columns:
                worksheet.write(row, df.columns.get_loc('BatchDetails'), df.iloc[row-2]['BatchDetails'], batch_format)
        
        # Set column widths
        worksheet.set_column('A:A', 10)  # ItemId
        worksheet.set_column('B:B', 20)  # Sku
        worksheet.set_column('C:C', 30)  # Name
        worksheet.set_column('D:D', 15)  # Type
        worksheet.set_column('E:E', 15)  # AvailableQty/Expected AvailableQty
        
        # Only set QTY/LOT column widths if they exist
        if any(col in df.columns for col in ['QTY_1', 'LOT_1', 'QTY_2', 'LOT_2', 'QTY_3', 'LOT_3']):
            worksheet.set_column('F:K', 10)  # QTY/LOT columns
        
        # Only set Notes/BatchDetails widths if they exist
        if 'Notes' in df.columns:
            notes_col = chr(ord('A') + df.columns.get_loc('Notes'))
            worksheet.set_column(f'{notes_col}:{notes_col}', 30)  # Notes
        
        if 'BatchDetails' in df.columns:
            batch_col = chr(ord('A') + df.columns.get_loc('BatchDetails'))
            worksheet.set_column(f'{batch_col}:{batch_col}', 50)  # BatchDetails
        
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