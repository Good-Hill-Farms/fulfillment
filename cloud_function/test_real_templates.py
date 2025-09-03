#!/usr/bin/env python3
"""
Test script that creates REAL Google Sheets inventory templates
and sends them to olena@goodhillfarms.com (instead of warehouse teams)

This simulates the actual production workflow but redirects emails to you.
"""

import os
import sys

# Set up environment variables
os.environ.update({
    # Email configuration
    'SMTP_HOST': 'smtp.gmail.com',
    'SMTP_PORT': '587',
    'SMTP_USERNAME': 'hello@goodhillfarms.com',
    'SMTP_PASSWORD': 'xufm lmxf ehvx fjxz',
    'SMTP_USE_TLS': 'true',
    
    # Google credentials
    'GOOGLE_APPLICATION_CREDENTIALS': 'nca-toolkit-project-446011-67d246fdbccf.json'
})

# Try to import the inventory system
try:
    from inventory_scheduler import send_inventory_emails_now
    print("✅ Successfully imported inventory automation system")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the cloud_function directory")
    sys.exit(1)

def main():
    """
    Run the real inventory template generation and email system
    This creates actual Google Sheets and sends real emails to olena@goodhillfarms.com
    """
    print("🚀 REAL INVENTORY TEMPLATE GENERATION TEST")
    print("=" * 60)
    print("📋 What this will do:")
    print("  ✅ Connect to ColdCart API for real inventory data")
    print("  ✅ Create Google Sheets in warehouse folders")
    print("  ✅ Generate formatted inventory forms with:")
    print("      - SKUs and quantities")
    print("      - LOT dates from batch codes")
    print("      - Weight calculations")
    print("      - Empty rows for manual entry")
    print("      - Proper formatting and styles")
    print("  ✅ Send emails to olena@goodhillfarms.com (TEST MODE)")
    print("  ✅ Both Oxnard and Wheeling warehouses")
    print("=" * 60)
    
    # Confirm before proceeding
    print("⚠️  WARNING: This will create real Google Sheets!")
    confirm = input("\nContinue with real template generation? (y/N): ").lower().strip()
    
    if confirm != 'y':
        print("❌ Test cancelled")
        return False
    
    print("\n🏃‍♂️ Starting real inventory template generation...")
    print("⏱️  This may take 1-2 minutes...")
    
    try:
        # Run the actual inventory automation in test mode
        # This calls the same function that will run in production, but emails go to you
        result = send_inventory_emails_now(test_mode=True)
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        
        print(f"Overall Success: {'✅ YES' if result.get('success') else '❌ NO'}")
        print(f"Test Mode: {'✅ YES' if result.get('test_mode') else '❌ NO'}")
        print(f"Timestamp: {result.get('timestamp', 'Unknown')}")
        
        # Oxnard results
        oxnard = result.get('oxnard', {})
        print(f"\n🏢 OXNARD WAREHOUSE:")
        print(f"  Sheet Generated: {'✅ YES' if oxnard.get('generated') else '❌ NO'}")
        print(f"  Email Sent: {'✅ YES' if oxnard.get('emailed') else '❌ NO'}")
        if oxnard.get('url'):
            print(f"  📄 Google Sheet: {oxnard.get('url')}")
        if oxnard.get('error'):
            print(f"  ❌ Error: {oxnard.get('error')}")
        
        # Wheeling results
        wheeling = result.get('wheeling', {})
        print(f"\n🏢 WHEELING WAREHOUSE:")
        print(f"  Sheet Generated: {'✅ YES' if wheeling.get('generated') else '❌ NO'}")
        print(f"  Email Sent: {'✅ YES' if wheeling.get('emailed') else '❌ NO'}")
        if wheeling.get('url'):
            print(f"  📄 Google Sheet: {wheeling.get('url')}")
        if wheeling.get('error'):
            print(f"  ❌ Error: {wheeling.get('error')}")
        
        # Summary
        if result.get('success'):
            print(f"\n🎉 SUCCESS!")
            print(f"📧 Check your email: olena@goodhillfarms.com")
            print(f"📊 You should receive 2 emails:")
            print(f"  1. [TEST] Oxnard Inventory Form - Weekly Count")
            print(f"  2. [TEST] Wheeling Inventory Form - Weekly Count")
            print(f"\n📁 Google Sheets created in warehouse folders:")
            if oxnard.get('url'):
                print(f"  🔗 Oxnard: {oxnard.get('url')}")
            if wheeling.get('url'):
                print(f"  🔗 Wheeling: {wheeling.get('url')}")
        else:
            print(f"\n❌ Some issues occurred. Check the errors above.")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        print("Full error details:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ Test completed successfully!' if success else '❌ Test had issues.'}")
    sys.exit(0 if success else 1)



