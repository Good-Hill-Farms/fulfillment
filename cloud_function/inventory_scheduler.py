"""
Inventory Email Scheduler
Handles automated sending of inventory forms every Thursday at 4am PST
"""

import logging
from datetime import datetime, timedelta
import pytz
from excel_generator import generate_inventory_sheet, OXNARD_FOLDER_ID, WHEELING_FOLDER_ID

logger = logging.getLogger(__name__)

def is_thursday_4am_pst():
    """
    Check if current time is Thursday 4:00 AM PST
    Returns True if it's within the 4:00-4:59 AM PST window on Thursday
    """
    pst = pytz.timezone('America/Los_Angeles')
    now_pst = datetime.now(pst)
    
    # Check if it's Thursday (weekday 3, where Monday=0)
    is_thursday = now_pst.weekday() == 3
    
    # Check if it's between 4:00 AM and 4:59 AM
    is_4am_hour = now_pst.hour == 4
    
    return is_thursday and is_4am_hour

def should_send_weekly_emails():
    """
    Check if we should send weekly emails
    This can be extended to include additional logic like checking if emails were already sent this week
    """
    return is_thursday_4am_pst()

def send_weekly_inventory_emails(test_mode=False):
    """
    Generate inventory sheets and send emails for both warehouses
    This is the main function to be called by the scheduler
    
    Args:
        test_mode: If True, sends emails to olena@goodhillfarms.com instead of warehouse teams
    
    Returns:
        dict: Result summary with success/failure status for each warehouse
    """
    results = {
        'success': False,
        'test_mode': test_mode,
        'oxnard': {'generated': False, 'emailed': False, 'url': None, 'error': None},
        'wheeling': {'generated': False, 'emailed': False, 'url': None, 'error': None},
        'timestamp': datetime.now(pytz.timezone('America/Los_Angeles')).isoformat()
    }
    
    mode_text = "TEST MODE - " if test_mode else ""
    logger.info(f"Starting {mode_text}weekly inventory email automation...")
    
    try:
        # Generate and email Oxnard inventory
        logger.info("Generating Oxnard inventory sheet...")
        oxnard_url = generate_inventory_sheet("Oxnard", OXNARD_FOLDER_ID)
        
        if oxnard_url:
            results['oxnard']['generated'] = True
            results['oxnard']['url'] = oxnard_url
            
            # Send email after sheet generation
            from email_service import send_warehouse_inventory_email
            email_success = send_warehouse_inventory_email("Oxnard", oxnard_url, test_mode=test_mode)
            results['oxnard']['emailed'] = email_success
            
            if email_success:
                logger.info(f"✅ Oxnard inventory completed: {oxnard_url}")
            else:
                logger.error("❌ Failed to send Oxnard email")
        else:
            results['oxnard']['error'] = "Failed to generate inventory sheet"
            logger.error("❌ Failed to generate Oxnard inventory sheet")
        
        # Generate and email Wheeling inventory
        logger.info("Generating Wheeling inventory sheet...")
        wheeling_url = generate_inventory_sheet("Wheeling", WHEELING_FOLDER_ID)
        
        if wheeling_url:
            results['wheeling']['generated'] = True
            results['wheeling']['url'] = wheeling_url
            
            # Send email after sheet generation
            from email_service import send_warehouse_inventory_email
            email_success = send_warehouse_inventory_email("Wheeling", wheeling_url, test_mode=test_mode)
            results['wheeling']['emailed'] = email_success
            
            if email_success:
                logger.info(f"✅ Wheeling inventory completed: {wheeling_url}")
            else:
                logger.error("❌ Failed to send Wheeling email")
        else:
            results['wheeling']['error'] = "Failed to generate inventory sheet"
            logger.error("❌ Failed to generate Wheeling inventory sheet")
        
        # Mark overall success if at least one warehouse succeeded
        results['success'] = results['oxnard']['generated'] or results['wheeling']['generated']
        
        if results['success']:
            logger.info("✅ Weekly inventory email automation completed successfully")
        else:
            logger.error("❌ Weekly inventory email automation failed for both warehouses")
            
    except Exception as e:
        logger.error(f"❌ Error in weekly inventory email automation: {str(e)}")
        results['error'] = str(e)
    
    return results

def send_inventory_emails_now(test_mode=False):
    """
    Force send inventory emails immediately (for testing or manual triggers)
    
    Args:
        test_mode: If True, sends emails to olena@goodhillfarms.com instead of warehouse teams
    
    Returns:
        dict: Result summary
    """
    mode_text = "TEST MODE - " if test_mode else ""
    logger.info(f"Manually triggering {mode_text}inventory email automation...")
    return send_weekly_inventory_emails(test_mode=test_mode)

def get_next_thursday_4am_pst():
    """
    Get the next Thursday 4:00 AM PST datetime
    Useful for scheduling information
    
    Returns:
        datetime: Next Thursday 4:00 AM PST
    """
    pst = pytz.timezone('America/Los_Angeles')
    now_pst = datetime.now(pst)
    
    # Calculate days until next Thursday
    days_ahead = 3 - now_pst.weekday()  # Thursday is weekday 3
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    
    next_thursday = now_pst + timedelta(days=days_ahead)
    # Set to 4:00 AM
    next_thursday = next_thursday.replace(hour=4, minute=0, second=0, microsecond=0)
    
    return next_thursday

if __name__ == "__main__":
    # For testing purposes
    print("Testing inventory scheduler...")
    print(f"Is Thursday 4am PST: {is_thursday_4am_pst()}")
    print(f"Should send emails: {should_send_weekly_emails()}")
    print(f"Next Thursday 4am PST: {get_next_thursday_4am_pst()}")
    
    # Uncomment to test email sending
    # result = send_inventory_emails_now()
    # print("Result:", result)
