#!/usr/bin/env python3
"""
Simple email testing script - tests ONLY email functionality
No Google Sheets or inventory API dependencies required
"""

import os
import sys

# Set email environment variables
os.environ['SMTP_HOST'] = 'smtp.gmail.com'
os.environ['SMTP_PORT'] = '587'
os.environ['SMTP_USERNAME'] = 'hello@goodhillfarms.com'
os.environ['SMTP_PASSWORD'] = 'xufm lmxf ehvx fjxz'
os.environ['SMTP_USE_TLS'] = 'true'

# Import email service
try:
    from email_service import EmailService, send_test_email, send_warehouse_inventory_email
    print("✅ Successfully imported email service")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_email():
    """Test basic email configuration"""
    print("\n🧪 Test 1: Basic Email Configuration")
    print("Sending test email to olena@goodhillfarms.com...")
    
    try:
        success = send_test_email()
        if success:
            print("✅ SUCCESS: Basic email sent!")
            print("📧 Check your email: olena@goodhillfarms.com")
        else:
            print("❌ FAILED: Email not sent")
        return success
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_warehouse_emails():
    """Test warehouse-specific emails with dummy URLs"""
    print("\n🧪 Test 2: Warehouse Email Templates")
    
    # Dummy Google Sheet URL for testing
    test_sheet_url = "https://docs.google.com/spreadsheets/d/TEST_SHEET_ID/edit"
    
    results = []
    
    # Test Oxnard email
    print("\nTesting Oxnard warehouse email...")
    try:
        success = send_warehouse_inventory_email("Oxnard", test_sheet_url, test_mode=True)
        if success:
            print("✅ SUCCESS: Oxnard test email sent!")
        else:
            print("❌ FAILED: Oxnard email not sent")
        results.append(success)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        results.append(False)
    
    # Test Wheeling email
    print("\nTesting Wheeling warehouse email...")
    try:
        success = send_warehouse_inventory_email("Wheeling", test_sheet_url, test_mode=True)
        if success:
            print("✅ SUCCESS: Wheeling test email sent!")
        else:
            print("❌ FAILED: Wheeling email not sent")
        results.append(success)
    except Exception as e:
        print(f"❌ ERROR: {e}")
        results.append(False)
    
    return all(results)

def test_email_service_directly():
    """Test EmailService class directly"""
    print("\n🧪 Test 3: Direct EmailService Test")
    
    try:
        email_service = EmailService()
        print("✅ EmailService initialized successfully")
        
        # Test direct email sending
        success = email_service.send_email(
            to_emails=["olena@goodhillfarms.com"],
            subject="Direct Email Service Test",
            body="This is a direct test of the EmailService class. If you receive this, the email system is working correctly!"
        )
        
        if success:
            print("✅ SUCCESS: Direct email sent!")
        else:
            print("❌ FAILED: Direct email not sent")
        
        return success
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all email tests"""
    print("🚀 Starting Email-Only Tests")
    print("📧 All emails will be sent to: olena@goodhillfarms.com")
    print("=" * 50)
    
    # Run tests
    test1 = test_basic_email()
    test2 = test_warehouse_emails()
    test3 = test_email_service_directly()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print(f"Basic Email Test:     {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Warehouse Email Test: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Direct Service Test:  {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 ALL TESTS PASSED!")
        print("📧 Check your email: olena@goodhillfarms.com")
        print("You should have received 4 emails total:")
        print("  1. Basic test email")
        print("  2. [TEST] Oxnard Inventory Form")
        print("  3. [TEST] Wheeling Inventory Form") 
        print("  4. Direct EmailService test")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Check your SMTP configuration and Gmail App Password")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



