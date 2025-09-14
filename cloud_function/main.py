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
from inventory_scheduler import send_inventory_emails_now
from email_service import send_test_email

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

@functions_framework.http  
def test_inventory_emails(request):
    """
    HTTP Cloud Function for testing inventory email automation.
    Creates real Google Sheets but sends test emails to olena@goodhillfarms.com instead of warehouse teams.
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
        
        # Test inventory emails in test mode (sends to olena@goodhillfarms.com)
        logger.info("Starting TEST inventory email automation...")
        try:
            result = send_inventory_emails_now(test_mode=True)
            
            if result and result.get('success'):
                logger.info("Test inventory emails sent successfully")
                return (jsonify(result), 200, headers)
            else:
                logger.error("Test inventory email sending failed")
                error_detail = result.get('error', 'Unknown error') if result else 'Unknown error'
                return (jsonify({
                    'success': False,
                    'error': error_detail
                }), 500, headers)
        except Exception as e:
            logger.exception(f"Error in test inventory email automation: {str(e)}")
            return (jsonify({
                'success': False,
                'error': str(e)
            }), 500, headers)
        
    except Exception as e:
        logger.exception(f"Error in test inventory email automation: {str(e)}")
        error_traceback = traceback.format_exc()
        return (jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_traceback
        }), 500, headers)

@functions_framework.http
def send_inventory_emails(request):
    """
    HTTP Cloud Function for sending REAL inventory emails to warehouse teams.
    Creates real Google Sheets and sends emails to actual warehouse teams (PRODUCTION).
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
        
        # Send REAL inventory emails to warehouse teams (PRODUCTION MODE)
        logger.info("Starting PRODUCTION inventory email automation...")
        try:
            result = send_inventory_emails_now(test_mode=False)  # PRODUCTION MODE
            
            if result and result.get('success'):
                logger.info("Production inventory emails sent successfully to warehouse teams")
                return (jsonify(result), 200, headers)
            else:
                logger.error("Production inventory email sending failed")
                error_detail = result.get('error', 'Unknown error') if result else 'Unknown error'
                return (jsonify({
                    'success': False,
                    'error': error_detail
                }), 500, headers)
        except Exception as e:
            logger.exception(f"Error in production inventory email automation: {str(e)}")
            return (jsonify({
                'success': False,
                'error': str(e)
            }), 500, headers)
        
    except Exception as e:
        logger.exception(f"Error in production inventory email automation: {str(e)}")
        error_traceback = traceback.format_exc()
        return (jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_traceback
        }), 500, headers)

@functions_framework.http
def test_basic_email(request):
    """
    HTTP Cloud Function for testing basic email functionality.
    Sends a simple test email to olena@goodhillfarms.com.
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
        
        # Send basic test email
        logger.info("Sending basic test email...")
        try:
            success = send_test_email()
            
            if success:
                logger.info("Basic test email sent successfully")
                return (jsonify({
                    'success': True,
                    'message': 'Basic test email sent to olena@goodhillfarms.com'
                }), 200, headers)
            else:
                logger.error("Basic test email sending failed")
                return (jsonify({
                    'success': False,
                    'error': 'Failed to send basic test email'
                }), 500, headers)
        except Exception as e:
            logger.exception(f"Error in basic test email: {str(e)}")
            return (jsonify({
                'success': False,
                'error': str(e)
            }), 500, headers)
        
    except Exception as e:
        logger.exception(f"Error in basic test email: {str(e)}")
        error_traceback = traceback.format_exc()
        return (jsonify({
            'success': False,
            'error': str(e),
            'traceback': error_traceback
        }), 500, headers)
