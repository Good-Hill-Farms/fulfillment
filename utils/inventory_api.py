import os
import requests
import pandas as pd
from io import StringIO

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

def save_as_excel(df, filename="inventory_export.xlsx"):
    """
    Save the DataFrame as an Excel file
    """
    df.to_excel(filename, index=False)
    return filename

def save_as_csv(df, filename="inventory_export.csv"):
    """
    Save the DataFrame as a CSV file
    """
    df.to_csv(filename, index=False)
    return filename 