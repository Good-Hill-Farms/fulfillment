import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)


def parse_shipment_batch_summary(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the response from get_shipment_batch_summary function
    
    Args:
        response (dict): The response from get_shipment_batch_summary
        
    Returns:
        dict: Parsed data with various extracted information
    """
    if not response or not isinstance(response, dict):
        logger.warning("Invalid response format")
        return {}
    
    data = response.get('data', [])
    if not data:
        logger.info("No data found in response")
        return {
            'pdf_urls': [],
            'total_batches': 0,
            'total_shipments': 0,
            'warehouses': [],
            'date_range': {},
            'status_summary': {}
        }
    
    # Extract PDF URLs
    pdf_urls = [item.get('pdfUrl') for item in data if item.get('pdfUrl')]
    
    # Calculate totals
    total_batches = len(data)
    total_shipments = sum(item.get('shipmentCount', 0) for item in data)
    
    # Extract unique warehouses
    warehouse_dict = {}
    for item in data:
        warehouse_id = item.get('warehouseId')
        if warehouse_id is not None and warehouse_id not in warehouse_dict:
            warehouse_dict[warehouse_id] = {
                'id': warehouse_id,
                'name': item.get('warehouseName'),
                'city': item.get('originCity'),
                'state': item.get('originState'),
                'postal_code': item.get('originPostalCode')
            }
    warehouses = list(warehouse_dict.values())
    
    # Extract date range
    created_dates = []
    for item in data:
        if item.get('createdDate'):
            try:
                created_dates.append(datetime.fromisoformat(item['createdDate'].replace('Z', '+00:00')))
            except ValueError:
                logger.warning(f"Invalid date format: {item.get('createdDate')}")
    
    date_range = {}
    if created_dates:
        date_range = {
            'earliest': min(created_dates).isoformat(),
            'latest': max(created_dates).isoformat(),
            'total_days': (max(created_dates) - min(created_dates)).days + 1
        }
    
    # Status summary
    status_counts = {}
    for item in data:
        statuses = item.get('shipmentStatuses', [])
        for status in statuses:
            status_counts[status] = status_counts.get(status, 0) + 1
    
    parsed_data = {
        'pdf_urls': pdf_urls,
        'total_batches': total_batches,
        'total_shipments': total_shipments,
        'warehouses': warehouses,
        'date_range': date_range,
        'status_summary': status_counts,
        'has_more': response.get('hasMore', False),
        'next_id': response.get('nextId')
    }
    
    logger.info(f"Parsed {total_batches} batches with {total_shipments} total shipments")
    logger.info(f"Found {len(pdf_urls)} PDF URLs")
    
    return parsed_data


def get_wave_summary_urls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract PDF URLs with date and batch information from the shipment batch summary response
    
    Args:
        response (dict): The response from get_shipment_batch_summary
        
    Returns:
        list: List of dictionaries containing PDF URL, CSV URL, date, and batch information
    """
    if not response or not isinstance(response, dict):
        return []
    
    data = response.get('data', [])
    pdf_info = []
    
    for item in data:
        pdf_url = item.get('pdfUrl')
        if pdf_url:
            # Convert PDF URL to CSV URL
            csv_url = pdf_url.replace('.pdf', '.csv')
            csv_url = f"https://api-direct.coldcartfulfill.com/shipments/242/batches/{csv_url}"
            
            pdf_info.append({
                'pdf_url': pdf_url,
                'csv_url': csv_url,
                'batch_name': item.get('batchName', ''),
                'created_date': item.get('createdDate', ''),
                'batch_id': item.get('id'),
                'warehouse_name': item.get('warehouseName', ''),
                'shipment_count': item.get('shipmentCount', 0)
            })
    
    return pdf_info


def convert_pdf_to_csv_url(pdf_url: str) -> str:
    """
    Convert a PDF URL to a CSV URL for downloading batch data
    
    Args:
        pdf_url (str): The PDF URL from the batch summary
        
    Returns:
        str: The corresponding CSV URL
    """
    if not pdf_url:
        return ""
    
    # Remove .pdf extension and add .csv
    csv_filename = pdf_url.replace('.pdf', '.csv')
    return f"https://api-direct.coldcartfulfill.com/shipments/242/batches/{csv_filename}"


def get_pdf_urls_simple(response: Dict[str, Any]) -> List[str]:
    """
    Extract only the PDF URLs from the shipment batch summary response (simple version)
    
    Args:
        response (dict): The response from get_shipment_batch_summary
        
    Returns:
        list: List of PDF URLs
    """
    if not response or not isinstance(response, dict):
        return []
    
    data = response.get('data', [])
    return [item.get('pdfUrl') for item in data if item.get('pdfUrl')]


if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Import the API function for testing
    from coldcart_api import get_shipment_batch_summary
    from datetime import date
    
    # Test the parsing functions
    logger.info("Testing parsing functions...")
    
    # Get sample data
    response = get_shipment_batch_summary(
        from_date=date(2025, 6, 19), 
        to_date=date(2025, 7, 19)
    )
    
    if response:
        # Test PDF URL extraction
        pdf_urls = get_wave_summary_urls(response)
        print(f"\nPDF URLs ({len(pdf_urls)} found):")
        for i, url_info in enumerate(pdf_urls, 1):
            print(f"{i}. {url_info['pdf_url']} (Batch: {url_info['batch_name']}, Date: {url_info['created_date']})")
            print(f"   CSV: {url_info['csv_url']}")
    else:
        print("No data received from API") 