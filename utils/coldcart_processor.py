import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import pandas as pd
from datetime import datetime, date, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
from utils.coldcart_api import (
    get_all_shipment_batch_summaries,
    get_batch_csv_content,
    get_shipment_stats
)
from utils.coldcart_parser import get_wave_summary_urls
from utils.google_sheets import get_credentials
import time
from io import StringIO

# Set up logging
logger = logging.getLogger(__name__)

# Define scopes including BigQuery
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/bigquery",
]

def check_file_exists_in_bigquery(client: bigquery.Client, table_ref: str, csv_filename: str) -> bool:
    """Check if data from this CSV file already exists in BigQuery"""
    query = f"""
    SELECT COUNT(*) as count 
    FROM `{table_ref}` 
    WHERE csv_filename = @filename
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("filename", "STRING", csv_filename)
        ]
    )
    query_job = client.query(query, job_config=job_config)
    result = next(query_job.result())
    exists = result.count > 0
    if exists:
        logger.info(f"File {csv_filename} already exists in BigQuery, skipping...")
    return exists

def upload_shipment_stats_to_bigquery(from_date=None, to_date=None, project_id="nca-toolkit-project-446011", dataset_id="fulfillment_cc_shopify", table_id="cc_shipment_stats"):
    """
    Fetch and upload shipment stats to BigQuery.
    
    Args:
        from_date (datetime.date or str): Start date (default: 30 days ago)
        to_date (datetime.date or str): End date (default: today)
        project_id (str): GCP project ID
        dataset_id (str): BigQuery dataset ID
        table_id (str): BigQuery table name
        
    Returns:
        str: BigQuery table reference if successful, else None
    """
    try:
        # Get credentials with BigQuery scope
        creds = get_credentials()
        if hasattr(creds, 'with_scopes'):
            creds = creds.with_scopes(SCOPES)
        client = bigquery.Client(project=project_id, credentials=creds)
        
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        
        # Fetch shipment stats
        logger.info(f"Fetching shipment stats from {from_date} to {to_date}...")
        stats_content = get_shipment_stats(from_date, to_date)
        
        if not stats_content:
            logger.error("Failed to get shipment stats data")
            return None
            
        # Parse CSV content
        stats_df = pd.read_csv(StringIO(stats_content))
        logger.info(f"Uploading {len(stats_df)} shipment stats rows to BigQuery...")
        
        # Upload to BigQuery
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = client.load_table_from_dataframe(
            stats_df,
            table_ref,
            job_config=job_config
        )
        job.result()
        
        logger.info(f"Successfully uploaded shipment stats to {table_ref}")
        return table_ref
        
    except Exception as e:
        logger.error(f"Failed to upload shipment stats to BigQuery: {e}")
        return None

def process_and_upload_all_csvs_to_bigquery(from_date=None, to_date=None, warehouse_id=0, shipment_id=0, project_id="nca-toolkit-project-446011", dataset_id="fulfillment_cc_shopify", table_id="cc_wave_summaries", batch_size=1000):
    """
    Fetch all wave summaries and upload data directly to BigQuery.
    
    Args:
        from_date, to_date, warehouse_id, shipment_id: Filters for the batch summary API
        project_id (str): GCP project ID
        dataset_id (str): BigQuery dataset ID
        table_id (str): BigQuery table name
        batch_size (int): Number of rows to process at a time
        
    Returns:
        str: BigQuery table reference if successful, else None
    """
    try:
        # Get credentials with BigQuery scope
        creds = get_credentials()
        if hasattr(creds, 'with_scopes'):
            creds = creds.with_scopes(SCOPES)
        client = bigquery.Client(project=project_id, credentials=creds)
        
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        
        # Fetch and parse wave summaries
        logger.info("Fetching all shipment batch summaries...")
        all_batches = get_all_shipment_batch_summaries(from_date, to_date, warehouse_id, shipment_id)
        if not all_batches:
            logger.warning("No batch data found for upload.")
            return None
            
        logger.info(f"Parsing {len(all_batches)} batches to wave summary format...")
        parsed = get_wave_summary_urls({'data': all_batches})
        if not parsed:
            logger.warning("No wave summary data to process.")
            return None
            
        failed_files = []
        successful_files = []
        total_rows = 0
        
        # Check if table exists
        table_exists = False
        try:
            client.get_table(table_ref)
            table_exists = True
            logger.info(f"Table {table_ref} exists, will append data")
        except Exception:
            logger.info(f"Table {table_ref} does not exist, will create new")
        
        # Process each batch
        for batch_idx, batch in enumerate(parsed):
            csv_filename = batch.get('csv_url', '').split('/')[-1]
            if not csv_filename:
                logger.warning(f"No csv_filename for batch: {batch}")
                continue
            
            # Skip if already in BigQuery
            if table_exists and check_file_exists_in_bigquery(client, table_ref, csv_filename):
                continue
            
            # Download and process CSV content directly
            try:
                logger.info(f"Downloading {csv_filename} ...")
                csv_content = get_batch_csv_content(csv_filename)
                if not csv_content:
                    logger.error(f"Failed to download {csv_filename}")
                    failed_files.append(csv_filename)
                    continue
                
                # Read CSV content in chunks
                for chunk_idx, chunk in enumerate(pd.read_csv(StringIO(csv_content), chunksize=batch_size)):
                    # Add only csv_filename column
                    chunk['csv_filename'] = csv_filename
                    
                    # Configure job based on whether it's first upload
                    is_first_upload = batch_idx == 0 and chunk_idx == 0 and not table_exists
                    job_config = bigquery.LoadJobConfig(
                        write_disposition="WRITE_TRUNCATE" if is_first_upload else "WRITE_APPEND"
                    )
                    
                    # Upload chunk to BigQuery
                    job = client.load_table_from_dataframe(
                        chunk, 
                        table_ref,
                        job_config=job_config
                    )
                    job.result()  # Wait for job to complete
                    
                    total_rows += len(chunk)
                    logger.info(f"Uploaded {len(chunk)} rows (total: {total_rows})")
                
                successful_files.append(csv_filename)
            except Exception as e:
                logger.error(f"Failed to process {csv_filename}: {e}")
                failed_files.append(csv_filename)
                continue
        
        # Log summary
        logger.info(f"\nProcessing Summary:")
        logger.info(f"  Successfully processed: {len(successful_files)} files")
        logger.info(f"  Failed to process: {len(failed_files)} files")
        logger.info(f"  Total rows uploaded: {total_rows}")
        if failed_files:
            logger.info("  Failed files:")
            for f in failed_files:
                logger.info(f"    - {f}")
        
        return table_ref
    except Exception as e:
        logger.error(f"Failed to process and upload CSVs to BigQuery: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example: run for last 30 days
    to_date = date.today()
    from_date = to_date - timedelta(days=30)
    
    # Upload shipment stats
    stats_ref = upload_shipment_stats_to_bigquery(from_date=from_date, to_date=to_date)
    if stats_ref:
        print(f"Shipment stats uploaded to: {stats_ref}")
    
    # Upload wave summaries (commented out)
    # wave_ref = process_and_upload_all_csvs_to_bigquery(from_date=from_date, to_date=to_date)
    # if wave_ref:
    #     print(f"Wave summaries uploaded to: {wave_ref}") 