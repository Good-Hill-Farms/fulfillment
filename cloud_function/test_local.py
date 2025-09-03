#!/usr/bin/env python3
"""
Local testing script for warehouse inventory email automation
Run this locally to test email functionality before deploying to cloud
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set required environment variables for testing
os.environ['SMTP_HOST'] = 'smtp.gmail.com'
os.environ['SMTP_PORT'] = '587'
os.environ['SMTP_USERNAME'] = 'hello@goodhillfarms.com'
os.environ['SMTP_PASSWORD'] = 'xufm lmxf ehvx fjxz'
os.environ['SMTP_USE_TLS'] = 'true'

# Set Google credentials using the service account file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'nca-toolkit-project-446011-67d246fdbccf.json'

# Import after setting environment variables
try:
    from email_service import send_test_email, send_warehouse_inventory_email
    from inventory_scheduler import send_inventory_emails_now
    print("✅ Successfully imported email modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required packages are installed")
    sys.exit(1)

def test_basic_email():
    """Test basic email configuration"""
    print("\n🧪 Testing basic email configuration...")
    try:
        success = send_test_email()
        if success:
            print("✅ Basic email test successful! Check olena@goodhillfarms.com")
        else:
            print("❌ Basic email test failed")
        return success
    except Exception as e:
        print(f"❌ Basic email test error: {e}")
        return False

def test_warehouse_email(warehouse_name, test_mode=True):
    """Test warehouse-specific email"""
    print(f"\n🧪 Testing {warehouse_name} warehouse email (test_mode={test_mode})...")
    try:
        # Use a dummy sheet URL for testing
        test_sheet_url = "https://docs.google.com/spreadsheets/d/DUMMY_TEST_ID/edit"
        success = send_warehouse_inventory_email(warehouse_name, test_sheet_url, test_mode=test_mode)
        if success:
            recipient = "olena@goodhillfarms.com" if test_mode else "warehouse teams"
            print(f"✅ {warehouse_name} email test successful! Check {recipient}")
        else:
            print(f"❌ {warehouse_name} email test failed")
        return success
    except Exception as e:
        print(f"❌ {warehouse_name} email test error: {e}")
        return False

def test_full_inventory_automation():
    """Test full inventory automation (creates real Google Sheets)"""
    print("\n🧪 Testing full inventory automation with real Google Sheets...")
    print("⚠️  WARNING: This will create actual Google Sheets and send emails!")
    
    confirm = input("Continue? (y/N): ").lower().strip()
    if confirm != 'y':
        print("Skipped full automation test")
        return True
    
    try:
        # This will create real sheets and send real emails to olena@goodhillfarms.com
        result = send_inventory_emails_now(test_mode=True)
        
        print(f"\n📊 Full automation test results:")
        print(f"Overall success: {result.get('success', False)}")
        print(f"Test mode: {result.get('test_mode', False)}")
        
        for warehouse in ['oxnard', 'wheeling']:
            data = result.get(warehouse, {})
            print(f"\n{warehouse.title()} Warehouse:")
            print(f"  Generated: {data.get('generated', False)}")
            print(f"  Emailed: {data.get('emailed', False)}")
            if data.get('url'):
                print(f"  URL: {data.get('url')}")
            if data.get('error'):
                print(f"  Error: {data.get('error')}")
        
        return result.get('success', False)
    except Exception as e:
        print(f"❌ Full automation test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting local email automation tests...")
    print("📧 All test emails will be sent to: olena@goodhillfarms.com")
    
    # Test 1: Basic email
    basic_success = test_basic_email()
    
    # Test 2: Warehouse emails (email-only, no sheet generation)
    oxnard_success = test_warehouse_email("Oxnard", test_mode=True)
    wheeling_success = test_warehouse_email("Wheeling", test_mode=True)
    
    # Summary of email-only tests
    print(f"\n📋 Email-only test summary:")
    print(f"✅ Basic email: {'PASS' if basic_success else 'FAIL'}")
    print(f"✅ Oxnard email: {'PASS' if oxnard_success else 'FAIL'}")
    print(f"✅ Wheeling email: {'PASS' if wheeling_success else 'FAIL'}")
    
    if basic_success and oxnard_success and wheeling_success:
        print("\n🎉 All email tests passed!")
        print("\n🔍 Optional: Test full automation (creates real Google Sheets)")
        test_full_inventory_automation()
    else:
        print("\n❌ Some email tests failed. Fix email configuration before proceeding.")
        return False
    
    print("\n✅ Local testing complete!")
    print("📧 Check your email: olena@goodhillfarms.com")
    return True

if __name__ == "__main__":
    main()



