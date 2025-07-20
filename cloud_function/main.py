"""
Google Cloud Function for creating fulfillment snapshots.
This implementation calls the snapshot_creator to generate fulfillment snapshots.
"""

import logging
import os
import traceback
import functions_framework
from flask import jsonify
import google.cloud.logging

# Import the snapshot creation functionality
from snapshot_creator import create_fulfillment_snapshot_sync

# Set up logging
try:
    # Use Google Cloud Logging when deployed to Cloud Functions
    client = google.cloud.logging.Client()
    client.setup_logging()
except Exception:
    # Fall back to standard logging when running locally
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

@functions_framework.http
def create_snapshot(request):
    """
    HTTP Cloud Function that creates a fulfillment snapshot.
    
    Args:
        request: The request object.
    Returns:
        The response object.
    """
    # CORS setup for preflight requests
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers for the main request
    headers = {'Access-Control-Allow-Origin': '*'}
    
    try:
        # Get the key from query parameters or request body
        key = None
        if request.method == 'GET':
            key = request.args.get('key')
        else:  # POST
            request_json = request.get_json(silent=True)
            key = request_json.get('key') if request_json else None
        
        # Verify the key
        expected_key = os.environ.get('TRIGGER_SECRET_KEY', 'fulfillment_projection_snapshot_trigger')
        if key != expected_key:
            logger.warning("Invalid authentication key provided")
            return (jsonify({
                'success': False,
                'error': 'Invalid authentication key'
            }), 401, headers)
        
        # Create the fulfillment snapshot
        logger.info("Starting fulfillment snapshot creation...")
        try:
            result = create_fulfillment_snapshot_sync()
            
            if result and result.get('success'):
                logger.info("Snapshot creation completed successfully")
                return (jsonify(result), 200, headers)
            else:
                logger.error("Snapshot creation failed")
                error_detail = result.get('error', 'Unknown error') if result else 'Unknown error'
                return (jsonify({
                    'success': False,
                    'error': error_detail
                }), 500, headers)
        except Exception as e:
            logger.exception(f"Error in snapshot creation: {str(e)}")
            return (jsonify({
                'success': False,
                'error': str(e)
            }), 500, headers)
        
    except Exception as e:
        logger.exception(f"Error in snapshot creation: {str(e)}")
        error_traceback = traceback.format_exc()
        return (jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_traceback
        }), 500, headers)
