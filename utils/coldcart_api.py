import os
import requests
import pandas as pd
import logging
from io import StringIO
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)

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

def get_shipment_batch_summary(from_date=None, to_date=None, warehouse_id=0, shipment_id=0, next_id=None):
    """
    Fetch shipment batch summary from the ColdCart API
    
    Args:
        from_date (datetime.date or str): Start date as Python date object or ISO string (default: 30 days ago)
        to_date (datetime.date or str): End date as Python date object or ISO string (default: today)
        warehouse_id (int): Warehouse ID (default: 0 for all warehouses)
        shipment_id (int): Shipment ID (default: 0 for all shipments)
        next_id (int): Next page ID for pagination (default: None for first page)
    
    Returns:
        dict: JSON response from the API or None if the API token is missing
    """
    api_token = os.getenv('COLDCART_API_TOKEN')
    if not api_token:
        logger.warning("COLDCART_API_TOKEN not found in environment variables")
        return None
    
    # Set default dates if not provided
    if not from_date:
        from_date = datetime.now() - timedelta(days=30)
    if not to_date:
        to_date = datetime.now()
    
    # Convert date objects to ISO format strings
    def format_date_for_api(date_obj):
        if isinstance(date_obj, str):
            # If it's already a string, try to parse and format it
            try:
                if 'T' in date_obj:
                    # Parse ISO string
                    parsed_date = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
                else:
                    # Parse date string
                    parsed_date = datetime.strptime(date_obj, '%Y-%m-%d')
                return parsed_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            except ValueError:
                # If parsing fails, return as is
                return date_obj
        elif isinstance(date_obj, datetime):
            return date_obj.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        elif hasattr(date_obj, 'strftime'):
            # Handle date objects
            return datetime.combine(date_obj, datetime.min.time()).strftime('%Y-%m-%dT%H:%M:%S.000Z')
        else:
            raise ValueError(f"Unsupported date type: {type(date_obj)}")
    
    from_date_str = format_date_for_api(from_date)
    to_date_str = format_date_for_api(to_date)
    
    api_url = f"https://api-direct.coldcartfulfill.com/shipments/242/batchsummary"
    
    params = {
        'warehouseId': warehouse_id,
        'shipmentId': shipment_id,
        'from': from_date_str,
        'to': to_date_str
    }
    
    # Add next_id parameter if provided for pagination
    if next_id is not None:
        params['nextId'] = next_id
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Origin": "https://portal.coldcartfulfill.com",
        "Referer": "https://portal.coldcartfulfill.com/",
        "Accept": "*/*",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
    }
    
    try:
        if next_id:
            logger.info(f"Fetching shipment batch summary page with nextId: {next_id}")
        else:
            logger.info(f"Fetching shipment batch summary from {from_date_str} to {to_date_str}")
        
        # Log the full URL for debugging
        full_url = f"{api_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
        logger.debug(f"Full API URL: {full_url}")
        
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()
        
        if not response.text.strip():
            logger.warning("Empty response received from shipment batch summary API")
            return None
        
        # Return JSON response
        result = response.json()
        logger.info(f"Successfully fetched shipment batch summary with {len(result.get('data', []))} records")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch shipment batch summary: {str(e)}")
        raise Exception(f"Failed to fetch shipment batch summary: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing shipment batch summary: {str(e)}")
        raise Exception(f"Error processing shipment batch summary: {str(e)}")


def download_batch_csv(batch_filename: str, save_path: str = None, upload_to_drive: bool = False, drive_folder_id: str = None) -> str:
    """
    Download a batch CSV file from the ColdCart API
    
    Args:
        batch_filename (str): The batch filename (e.g., 'batch-242-638883511879128844.csv')
        save_path (str): Optional path to save the file. If None, saves to current directory
        upload_to_drive (bool): Whether to upload to Google Drive after downloading
        drive_folder_id (str): Google Drive folder ID to upload to
        
    Returns:
        str: Path to the downloaded file or None if download failed
    """
    api_token = os.getenv('COLDCART_API_TOKEN')
    if not api_token:
        logger.warning("COLDCART_API_TOKEN not found in environment variables")
        return None
    
    # Ensure the filename has .csv extension
    if not batch_filename.endswith('.csv'):
        batch_filename = batch_filename.replace('.pdf', '.csv')
    
    api_url = f"https://api-direct.coldcartfulfill.com/shipments/242/batches/{batch_filename}"
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Origin": "https://portal.coldcartfulfill.com",
        "Referer": "https://portal.coldcartfulfill.com/",
        "Accept": "*/*",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
    }
    
    try:
        logger.info(f"Downloading batch CSV: {batch_filename}")
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        if not response.text.strip():
            logger.warning(f"Empty response received for batch CSV: {batch_filename}")
            return None
        
        # Determine save path
        if save_path is None:
            save_path = batch_filename
        elif os.path.isdir(save_path):
            save_path = os.path.join(save_path, batch_filename)
        
        # Save the file locally
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        logger.info(f"Successfully downloaded batch CSV to: {save_path}")
        
        # Upload to Google Drive if requested
        if upload_to_drive and drive_folder_id:
            try:
                import sys
                import os as os_module
                sys.path.append(os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))
                from utils.google_sheets import upload_csv_to_drive
                drive_file_id = upload_csv_to_drive(response.text, batch_filename, drive_folder_id)
                if drive_file_id:
                    logger.info(f"Successfully uploaded to Google Drive with file ID: {drive_file_id}")
                else:
                    logger.warning("Failed to upload to Google Drive")
            except Exception as e:
                logger.error(f"Error uploading to Google Drive: {str(e)}")
        
        return save_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download batch CSV {batch_filename}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error saving batch CSV {batch_filename}: {str(e)}")
        return None


def download_and_upload_to_drive(batch_filename: str, drive_folder_id: str = "1jmkrNtctQX4sEOtOphNBR6FydQu32Htw", batch_date: str = None, batch_time: str = None) -> str:
    """
    Download a batch CSV file and upload it directly to the specified Google Drive folder
    
    Args:
        batch_filename (str): The batch filename (e.g., 'batch-242-638883511879128844.csv')
        drive_folder_id (str): Google Drive folder ID (defaults to the specified folder)
        batch_date (str): Optional date string to include in the filename (e.g., '2025-07-17')
        batch_time (str): Optional time string to include in the filename (e.g., '12-13-07')
        
    Returns:
        str: Google Drive file ID of the uploaded file or None if failed
    """
    api_token = os.getenv('COLDCART_API_TOKEN')
    if not api_token:
        logger.warning("COLDCART_API_TOKEN not found in environment variables")
        return None
    
    # Ensure the filename has .csv extension
    if not batch_filename.endswith('.csv'):
        batch_filename = batch_filename.replace('.pdf', '.csv')
    
    # Create filename with date and time if provided
    if batch_date and batch_time:
        # Extract the base filename without extension
        base_name = batch_filename.replace('.csv', '')
        # Create new filename with date and time
        drive_filename = f"{base_name}_{batch_date}_{batch_time}.csv"
    elif batch_date:
        # Extract the base filename without extension
        base_name = batch_filename.replace('.csv', '')
        # Create new filename with date only
        drive_filename = f"{base_name}_{batch_date}.csv"
    else:
        drive_filename = batch_filename
    
    api_url = f"https://api-direct.coldcartfulfill.com/shipments/242/batches/{batch_filename}"
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Origin": "https://portal.coldcartfulfill.com",
        "Referer": "https://portal.coldcartfulfill.com/",
        "Accept": "*/*",
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
    }
    
    try:
        logger.info(f"Downloading and uploading batch CSV: {batch_filename} as {drive_filename}")
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        if not response.text.strip():
            logger.warning(f"Empty response received for batch CSV: {batch_filename}")
            return None
        
        # Determine target folder based on date
        target_folder_id = drive_folder_id
        if batch_date:
            try:
                # Parse date to get year and month
                date_obj = datetime.strptime(batch_date, '%Y-%m-%d')
                year = date_obj.strftime('%Y')
                month = date_obj.strftime('%m')
                
                # Create year/month folder structure
                import sys
                import os as os_module
                sys.path.append(os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))
                from utils.google_sheets import create_folder_if_not_exists
                
                # Create year folder first
                year_folder_id = create_folder_if_not_exists(year, drive_folder_id)
                if year_folder_id:
                    # Create month folder inside year folder
                    month_folder_id = create_folder_if_not_exists(month, year_folder_id)
                    if month_folder_id:
                        target_folder_id = month_folder_id
                        logger.info(f"Files will be saved to: {year}/{month}/")
                    else:
                        logger.warning(f"Could not create month folder {month}, using year folder")
                        target_folder_id = year_folder_id
                else:
                    logger.warning(f"Could not create year folder {year}, using root folder")
            except Exception as e:
                logger.warning(f"Error creating date folders: {str(e)}, using root folder")
        
        # Upload directly to Google Drive
        try:
            import sys
            import os as os_module
            sys.path.append(os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))
            from utils.google_sheets import upload_csv_to_drive
            drive_file_id = upload_csv_to_drive(response.text, drive_filename, target_folder_id)
            if drive_file_id:
                logger.info(f"Successfully uploaded {drive_filename} to Google Drive with file ID: {drive_file_id}")
                return drive_file_id
            else:
                logger.error("Failed to upload to Google Drive")
                return None
        except Exception as e:
            logger.error(f"Error uploading to Google Drive: {str(e)}")
            return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download batch CSV {batch_filename}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing batch CSV {batch_filename}: {str(e)}")
        return None


def download_multiple_batch_csvs(batch_filenames: list, save_directory: str = None) -> list:
    """
    Download multiple batch CSV files from the ColdCart API
    
    Args:
        batch_filenames (list): List of batch filenames to download
        save_directory (str): Optional directory to save files. If None, saves to current directory
        
    Returns:
        list: List of successfully downloaded file paths
    """
    downloaded_files = []
    
    for filename in batch_filenames:
        file_path = download_batch_csv(filename, save_directory)
        if file_path:
            downloaded_files.append(file_path)
    
    logger.info(f"Successfully downloaded {len(downloaded_files)} out of {len(batch_filenames)} batch CSV files")
    return downloaded_files


def download_and_upload_batch_with_datetime(batch_data: dict, drive_folder_id: str = "1jmkrNtctQX4sEOtOphNBR6FydQu32Htw") -> str:
    """
    Download a batch CSV file and upload it to Google Drive with the date and time automatically extracted from batch data
    
    Args:
        batch_data (dict): Batch data dictionary containing 'pdfUrl' and 'createdDate'
        drive_folder_id (str): Google Drive folder ID (defaults to the specified folder)
        
    Returns:
        str: Google Drive file ID of the uploaded file or None if failed or already exists
    """
    pdf_url = batch_data.get('pdfUrl')
    created_date = batch_data.get('createdDate')
    
    if not pdf_url:
        logger.error("No PDF URL found in batch data")
        return None
    
    # Extract date and time from createdDate (format: '2025-07-17T12:13:07.920Z')
    batch_date = None
    batch_time = None
    if created_date:
        try:
            # Parse the ISO date string
            date_obj = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            # Format as YYYY-MM-DD
            batch_date = date_obj.strftime('%Y-%m-%d')
            # Format as HH-MM-SS
            batch_time = date_obj.strftime('%H-%M-%S')
        except Exception as e:
            logger.warning(f"Could not parse date from {created_date}: {str(e)}")
    
    # Convert PDF URL to CSV filename
    csv_filename = pdf_url.replace('.pdf', '.csv')
    
    # Upload with date and time
    drive_file_id = download_and_upload_to_drive(csv_filename, drive_folder_id, batch_date, batch_time)
    
    if drive_file_id is None:
        # Check if it's because file already exists
        if batch_date and batch_time:
            drive_filename = f"{csv_filename.replace('.csv', '')}_{batch_date}_{batch_time}.csv"
        elif batch_date:
            drive_filename = f"{csv_filename.replace('.csv', '')}_{batch_date}.csv"
        else:
            drive_filename = csv_filename
        
        logger.info(f"File '{drive_filename}' already exists or upload was skipped")
        return "EXISTS"  # Special return value to indicate file already exists
    
    return drive_file_id


def download_multiple_batches_with_datetime(batch_data_list: list, drive_folder_id: str = "1jmkrNtctQX4sEOtOphNBR6FydQu32Htw") -> list:
    """
    Download multiple batch CSV files and upload them to Google Drive with dates and times in filenames
    
    Args:
        batch_data_list (list): List of batch data dictionaries
        drive_folder_id (str): Google Drive folder ID (defaults to the specified folder)
        
    Returns:
        list: List of successfully uploaded Google Drive file IDs with details
    """
    uploaded_files = []
    skipped_files = []
    failed_files = []
    
    for batch_data in batch_data_list:
        drive_file_id = download_and_upload_batch_with_datetime(batch_data, drive_folder_id)
        if drive_file_id == "EXISTS":
            # File already exists
            skipped_files.append({
                'batch_name': batch_data.get('batchName', 'Unknown'),
                'created_date': batch_data.get('createdDate', 'Unknown'),
                'status': 'already_exists'
            })
        elif drive_file_id:
            # Successfully uploaded
            uploaded_files.append({
                'batch_name': batch_data.get('batchName', 'Unknown'),
                'created_date': batch_data.get('createdDate', 'Unknown'),
                'drive_file_id': drive_file_id,
                'drive_link': f"https://drive.google.com/file/d/{drive_file_id}/view",
                'status': 'uploaded'
            })
        else:
            # Failed to upload
            failed_files.append({
                'batch_name': batch_data.get('batchName', 'Unknown'),
                'created_date': batch_data.get('createdDate', 'Unknown'),
                'status': 'failed'
            })
    
    logger.info(f"Upload Summary: {len(uploaded_files)} uploaded, {len(skipped_files)} already exist, {len(failed_files)} failed out of {len(batch_data_list)} total batches")
    
    # Return all results including skipped and failed files
    all_results = uploaded_files + skipped_files + failed_files
    return all_results


def get_all_shipment_batch_summaries(from_date=None, to_date=None, warehouse_id=0, shipment_id=0):
    """
    Fetch ALL shipment batch summaries from the ColdCart API using pagination
    
    Args:
        from_date (datetime.date or str): Start date as Python date object or ISO string (default: 30 days ago)
        to_date (datetime.date or str): End date as Python date object or ISO string (default: today)
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
        # Get batch summary for current page
        batch_summary = get_shipment_batch_summary(from_date, to_date, warehouse_id, shipment_id, next_id)
        
        if not batch_summary or 'data' not in batch_summary:
            logger.warning("No data received from batch summary API")
            break
        
        # Add current page data to all batches
        current_batches = batch_summary.get('data', [])
        all_batches.extend(current_batches)
        
        # Check pagination info
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


def download_and_upload_all_batches_with_datetime(from_date=None, to_date=None, warehouse_id=0, shipment_id=0, drive_folder_id: str = "1jmkrNtctQX4sEOtOphNBR6FydQu32Htw") -> list:
    """
    Download and upload ALL batch CSV files to Google Drive with dates and times in filenames
    
    Args:
        from_date (datetime.date or str): Start date as Python date object or ISO string (default: 30 days ago)
        to_date (datetime.date or str): End date as Python date object or ISO string (default: today)
        warehouse_id (int): Warehouse ID (default: 0 for all warehouses)
        shipment_id (int): Shipment ID (default: 0 for all shipments)
        drive_folder_id (str): Google Drive folder ID (defaults to the specified folder)
        
    Returns:
        list: List of successfully uploaded Google Drive file IDs with details
    """
    logger.info("Starting to download and upload ALL batches with pagination")
    
    # Get all batches using pagination
    all_batches = get_all_shipment_batch_summaries(from_date, to_date, warehouse_id, shipment_id)
    
    if not all_batches:
        logger.warning("No batches found to process")
        return []
    
    # Upload all batches with datetime
    uploaded_files = download_multiple_batches_with_datetime(all_batches, drive_folder_id)
    
    logger.info(f"Completed processing {len(uploaded_files)} out of {len(all_batches)} total batches")
    return uploaded_files


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
            downloaded_path = download_batch_csv(csv_filename, 'downloads')
            if downloaded_path:
                print(f"Successfully downloaded locally: {downloaded_path}")
            else:
                print("Local download failed")
            
            # Test 2: Download and upload to Google Drive with date and time
            logger.info("Testing Google Drive upload with date and time...")
            drive_file_id = download_and_upload_batch_with_datetime(first_batch)
            if drive_file_id:
                print(f"Successfully uploaded to Google Drive with file ID: {drive_file_id}")
                print(f"Google Drive link: https://drive.google.com/file/d/{drive_file_id}/view")
            else:
                print("Google Drive upload failed")
            
            # Test 3: Upload multiple batches with dates and times (first 3 for testing)
            logger.info("Testing multiple batch uploads with dates and times...")
            # Take first 3 batches for testing to avoid overwhelming the API
            test_batches = all_batches[:3]
            uploaded_files = download_multiple_batches_with_datetime(test_batches)
            if uploaded_files:
                # Group by status
                uploaded = [f for f in uploaded_files if f.get('status') == 'uploaded']
                skipped = [f for f in uploaded_files if f.get('status') == 'already_exists']
                failed = [f for f in uploaded_files if f.get('status') == 'failed']
                
                print(f"\nUpload Results:")
                print(f"  ✅ Uploaded: {len(uploaded)} files")
                print(f"  ⏭️  Skipped (already exist): {len(skipped)} files")
                print(f"  ❌ Failed: {len(failed)} files")
                
                if uploaded:
                    print(f"\nSuccessfully uploaded files:")
                    for file_info in uploaded:
                        print(f"  - {file_info['batch_name']} ({file_info['created_date']})")
                        print(f"    Link: {file_info['drive_link']}")
                
                if skipped:
                    print(f"\nSkipped files (already exist):")
                    for file_info in skipped:
                        print(f"  - {file_info['batch_name']} ({file_info['created_date']})")
            else:
                print("Multiple batch upload failed")
            
            # Test 4: Upload ALL batches
            logger.info("Testing upload of ALL batches...")
            all_uploaded_files = download_and_upload_all_batches_with_datetime(test_from_date, test_to_date)
            if all_uploaded_files:
                # Group by status
                uploaded = [f for f in all_uploaded_files if f.get('status') == 'uploaded']
                skipped = [f for f in all_uploaded_files if f.get('status') == 'already_exists']
                failed = [f for f in all_uploaded_files if f.get('status') == 'failed']
                
                print(f"\n📊 Complete Upload Summary:")
                print(f"  ✅ Uploaded: {len(uploaded)} files")
                print(f"  ⏭️  Skipped (already exist): {len(skipped)} files")
                print(f"  ❌ Failed: {len(failed)} files")
                print(f"  📁 Total processed: {len(all_uploaded_files)} files")
                
                if uploaded:
                    print(f"\nFirst few newly uploaded files:")
                    for i, file_info in enumerate(uploaded[:5]):  # Show first 5
                        print(f"  {i+1}. {file_info['batch_name']} ({file_info['created_date']})")
                        print(f"     Link: {file_info['drive_link']}")
                    if len(uploaded) > 5:
                        print(f"  ... and {len(uploaded) - 5} more newly uploaded files")
            else:
                print("All batch upload failed")
