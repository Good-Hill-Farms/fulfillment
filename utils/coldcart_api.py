import os
import requests
import pandas as pd
import logging
from io import StringIO
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

def get_api_headers():
    """Get standard headers for ColdCart API requests"""
    api_token = os.getenv('COLDCART_API_TOKEN')
    if not api_token:
        raise ValueError("COLDCART_API_TOKEN not found in environment variables")
    
    return {
        "Authorization": f"Bearer {api_token}",
        "Origin": "https://portal.coldcartfulfill.com",
        "Referer": "https://portal.coldcartfulfill.com/",
        "Accept": "*/*"
    }

def format_date_for_api(date_obj, include_time=True):
    """Format date object for API requests"""
    if isinstance(date_obj, str):
        try:
            if 'T' in date_obj:
                parsed_date = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
            else:
                parsed_date = datetime.strptime(date_obj, '%Y-%m-%d')
            if include_time:
                return parsed_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            return parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            return date_obj
    elif isinstance(date_obj, datetime):
        if include_time:
            return date_obj.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        return date_obj.strftime('%Y-%m-%d')
    elif hasattr(date_obj, 'strftime'):
        if include_time:
            return datetime.combine(date_obj, datetime.min.time()).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        return date_obj.strftime('%Y-%m-%d')
    else:
        raise ValueError(f"Unsupported date type: {type(date_obj)}")

def get_shipment_batch_summary(from_date=None, to_date=None, warehouse_id=0, shipment_id=0, next_id=None):
    """
    Fetch shipment batch summary from the ColdCart API
    
    Args:
        from_date (datetime.date or str): Start date (default: 30 days ago)
        to_date (datetime.date or str): End date (default: today)
        warehouse_id (int): Warehouse ID (default: 0 for all warehouses)
        shipment_id (int): Shipment ID (default: 0 for all shipments)
        next_id (int): Next page ID for pagination (default: None for first page)
    
    Returns:
        dict: JSON response from the API or None if failed
    """
    try:
        # Set default dates
        if not from_date:
            from_date = datetime.now() - timedelta(days=30)
        if not to_date:
            to_date = datetime.now()
        
        from_date_str = format_date_for_api(from_date)
        to_date_str = format_date_for_api(to_date)
        
        api_url = "https://api-direct.coldcartfulfill.com/shipments/242/batchsummary"
        params = {
            'warehouseId': warehouse_id,
            'shipmentId': shipment_id,
            'from': from_date_str,
            'to': to_date_str
        }
        
        if next_id is not None:
            params['nextId'] = next_id
        
        headers = get_api_headers()
        
        logger.info(f"Fetching batch summary from {from_date_str} to {to_date_str}")
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()
        
        if not response.text.strip():
            logger.warning("Empty response received from batch summary API")
            return None
        
        return response.json()
        
    except Exception as e:
        logger.error(f"Failed to fetch batch summary: {str(e)}")
        return None

def get_batch_csv_content(csv_filename: str) -> str:
    """
    Download CSV content for a specific batch
    
    Args:
        csv_filename (str): The batch CSV filename
        
    Returns:
        str: CSV content or None if failed
    """
    try:
        api_url = f"https://api-direct.coldcartfulfill.com/shipments/242/batches/{csv_filename}"
        headers = get_api_headers()
        
        logger.info(f"Downloading batch CSV: {csv_filename}")
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        if not response.text.strip():
            logger.warning(f"Empty response received for batch CSV: {csv_filename}")
            return None
            
        return response.text
        
    except Exception as e:
        logger.error(f"Failed to download batch CSV {csv_filename}: {str(e)}")
        return None

def get_shipment_stats(from_date=None, to_date=None) -> str:
    """
    Fetch shipment stats from the ColdCart API
    
    Args:
        from_date (datetime.date or str): Start date (default: 30 days ago)
        to_date (datetime.date or str): End date (default: today)
        
    Returns:
        str: CSV content or None if failed
    """
    try:
        # Set default dates
        if not from_date:
            from_date = datetime.now() - timedelta(days=30)
        if not to_date:
            to_date = datetime.now()
        
        from_date_str = format_date_for_api(from_date, include_time=False)
        to_date_str = format_date_for_api(to_date, include_time=False)
        
        api_url = "https://api-direct.coldcartfulfill.com/stats/242/shipments/exportcsv"
        headers = get_api_headers()
        params = {
            'from': from_date_str,
            'to': to_date_str
        }
        
        logger.info(f"Fetching shipment stats from {from_date_str} to {to_date_str}")
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()
        
        if not response.text.strip():
            logger.warning("Empty response received from shipment stats API")
            return None
        
        return response.text
        
    except Exception as e:
        logger.error(f"Failed to fetch shipment stats: {str(e)}")
        return None

def get_all_shipment_batch_summaries(from_date=None, to_date=None, warehouse_id=0, shipment_id=0):
    """
    Fetch ALL shipment batch summaries using pagination
    
    Args:
        from_date (datetime.date or str): Start date (default: 30 days ago)
        to_date (datetime.date or str): End date (default: today)
        warehouse_id (int): Warehouse ID (default: 0 for all warehouses)
        shipment_id (int): Shipment ID (default: 0 for all shipments)
    
    Returns:
        list: List of all batch data from all pages
    """
    all_batches = []
    next_id = None
    has_more = True
    
    logger.info("Starting to fetch all shipment batch summaries with pagination")
    
    while has_more:
        batch_summary = get_shipment_batch_summary(from_date, to_date, warehouse_id, shipment_id, next_id)
        
        if not batch_summary or 'data' not in batch_summary:
            logger.warning("No data received from batch summary API")
            break
        
        current_batches = batch_summary.get('data', [])
        all_batches.extend(current_batches)
        
        next_id = batch_summary.get('nextId')
        has_more = batch_summary.get('hasMore', False)
        
        logger.info(f"Fetched {len(current_batches)} batches (total so far: {len(all_batches)})")
        
        if has_more and next_id:
            logger.info(f"More data available, next ID: {next_id}")
        else:
            logger.info("No more data available")
            break
    
    logger.info(f"Successfully fetched {len(all_batches)} total batches across all pages")
    return all_batches

def get_inventory_data():
    """
    Fetch inventory data from the ColdCart API
    Returns a pandas DataFrame with the inventory data or None if the API token is missing
    """
    api_token = os.getenv('COLDCART_API_TOKEN')
    if not api_token:
        logger.warning("COLDCART_API_TOKEN not found in environment variables")
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
        logger.info("Fetching inventory data from ColdCart API")
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        if not response.text.strip():
            logger.warning("Empty response received from inventory API")
            return None
        
        # Convert response to DataFrame
        df = pd.read_csv(StringIO(response.text))
        
        # Check if DataFrame is empty
        if df.empty:
            logger.warning("Empty DataFrame returned from inventory API")
            return None
            
        logger.info(f"Successfully fetched inventory data with {len(df)} records")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch inventory data: {str(e)}")
        raise Exception(f"Failed to fetch inventory data: {str(e)}")
    except pd.errors.EmptyDataError:
        logger.warning("Empty data received from inventory API")
        return None
    except Exception as e:
        logger.error(f"Error processing inventory data: {str(e)}")
        raise Exception(f"Error processing inventory data: {str(e)}")


if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test with Python date objects
    from datetime import date
    
    # Test with date objects - using the same dates as the working curl command
    test_from_date = date(2025, 6, 19)
    test_to_date = date(2025, 7, 19)
    
    logger.info("Testing with date objects")
    df = get_shipment_batch_summary(from_date=test_from_date, to_date=test_to_date)
    print(f"Date objects test result: {len(df.get('data', [])) if df else 0} records")
    
    # Test pagination to get ALL batches
    logger.info("Testing pagination to get ALL batches...")
    all_batches = get_all_shipment_batch_summaries(from_date=test_from_date, to_date=test_to_date)
    print(f"Pagination test result: {len(all_batches)} total batches across all pages")
    
    # Test single page with nextId to verify format
    logger.info("Testing single page with nextId to verify URL format...")
    first_page = get_shipment_batch_summary(from_date=test_from_date, to_date=test_to_date)
    if first_page and 'nextId' in first_page:
        next_id = first_page['nextId']
        logger.info(f"First page nextId: {next_id}")
        second_page = get_shipment_batch_summary(from_date=test_from_date, to_date=test_to_date, next_id=next_id)
        if second_page:
            print(f"Second page test successful: {len(second_page.get('data', []))} records")
        else:
            print("Second page test failed")
    else:
        print("No pagination data found in first page")
    
    # Test download functionality
    if all_batches:
        # Get the first batch for testing
        first_batch = all_batches[0]
        pdf_url = first_batch.get('pdfUrl')
        if pdf_url:
            csv_filename = pdf_url.replace('.pdf', '.csv')
            logger.info(f"Testing download of: {csv_filename}")
            
            # Test 1: Download locally
            downloaded_content = get_batch_csv_content(csv_filename)
            if downloaded_content:
                print(f"Successfully downloaded locally: {len(downloaded_content)} bytes")
            else:
                print("Local download failed")
            
            # Test 2: Fetch shipment stats
            logger.info("Testing shipment stats...")
            stats_content = get_shipment_stats(test_from_date, test_to_date)
            if stats_content:
                print(f"Successfully fetched shipment stats: {len(stats_content)} bytes")
            else:
                print("Shipment stats failed")
            
            # Test 3: Upload multiple batches with dates and times (first 3 for testing)
            logger.info("Testing multiple batch uploads with dates and times...")
            # Take first 3 batches for testing to avoid overwhelming the API
            test_batches = all_batches[:3]
            # The original code had Google Drive specific logic here, which is removed.
            # This test will now just print the number of batches processed.
            print(f"Processed {len(test_batches)} batches for upload test.")
            
            # Test 4: Fetch ALL batches
            logger.info("Testing fetch of ALL batches...")
            all_fetched_batches = get_all_shipment_batch_summaries(test_from_date, test_to_date)
            print(f"Pagination test result: {len(all_fetched_batches)} total batches across all pages")
