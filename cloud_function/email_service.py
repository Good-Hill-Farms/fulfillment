import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class EmailService:
    """Service for sending emails via SMTP"""
    
    def __init__(self):
        self.smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        self.smtp_username = os.environ.get('SMTP_USERNAME')
        self.smtp_password = os.environ.get('SMTP_PASSWORD')
        self.smtp_use_tls = os.environ.get('SMTP_USE_TLS', 'true').lower() == 'true'
        
        if not self.smtp_username or not self.smtp_password:
            raise ValueError("SMTP_USERNAME and SMTP_PASSWORD environment variables must be set")
    
    def send_email(self, 
                   to_emails: List[str], 
                   subject: str, 
                   body: str, 
                   cc_emails: Optional[List[str]] = None,
                   is_html: bool = False) -> bool:
        """
        Send an email to specified recipients
        
        Args:
            to_emails: List of recipient email addresses
            subject: Email subject
            body: Email body content
            cc_emails: List of CC email addresses (optional)
            is_html: Whether the body is HTML content
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = subject
            
            if cc_emails:
                msg['Cc'] = ', '.join(cc_emails)
            
            # Add body to email
            body_part = MIMEText(body, 'html' if is_html else 'plain')
            msg.attach(body_part)
            
            # Create SMTP session
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            
            if self.smtp_use_tls:
                server.starttls()  # Enable security
            
            server.login(self.smtp_username, self.smtp_password)
            
            # Send email
            all_recipients = to_emails + (cc_emails or [])
            text = msg.as_string()
            server.sendmail(self.smtp_username, all_recipients, text)
            server.quit()
            
            logger.info(f"Email sent successfully to {', '.join(to_emails)}")
            if cc_emails:
                logger.info(f"CC'd to {', '.join(cc_emails)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False

def send_warehouse_inventory_email(warehouse_name: str, sheet_url: str, test_mode: bool = False) -> bool:
    """
    Send warehouse inventory form email with predefined recipients and messages
    
    Args:
        warehouse_name: "Oxnard" or "Wheeling"
        sheet_url: URL of the generated Google Sheet
        test_mode: If True, sends to test email instead of actual recipients
        
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    email_service = EmailService()
    
    # Test mode - send everything to test email
    if test_mode:
        test_email = "olena@goodhillfarms.com"
        
        if warehouse_name.lower() == "oxnard":
            subject = "[TEST] Oxnard Inventory Form - Weekly Count"
            original_recipients = "robert@coldchain3pl.com (CC: mara@goodhillfarms.com, supply@goodhillfarms.com, dara.chapman@coldcart.co, sasha@goodhillfarms.com)"
            
            body = f"""[THIS IS A TEST EMAIL]

Hi Olena,

This is a test of the Oxnard inventory email automation.

Original recipients would be: {original_recipients}

Please fill out the cells in green in Inventory Form: {sheet_url}

This form will show the SKU, lot dates, and expected quantity of each SKU. The goal is to standardize the process across both FCs and improve the accuracy of hard counts each week.

Please let me know if you have any questions.

Best regards,
Good Hill Farms Team

---
Test Mode: Email sent to {test_email} instead of actual recipients"""
            
        elif warehouse_name.lower() == "wheeling":
            subject = "[TEST] Wheeling Inventory Form - Weekly Count"
            original_recipients = "janet@coldchain3pl.com, omar@coldchain3pl.com, armando@coldchain3pl.com (CC: supply@goodhillfarms.com, mara@goodhillfarms.com, dara.chapman@coldcart.co, sasha@goodhillfarms.com)"
            
            body = f"""[THIS IS A TEST EMAIL]

Hi Olena,

This is a test of the Wheeling inventory email automation.

Original recipients would be: {original_recipients}

Please fill out the cells in green in Inventory Form: {sheet_url}

This form will show the SKU, lot dates, and expected quantity of each SKU. We'd like for this form to standardize the process for both FCs and to increase the accuracy of the hard counts each week.

Please let me know if you have any questions.

Best regards,
Good Hill Farms Team

---
Test Mode: Email sent to {test_email} instead of actual recipients"""
        else:
            logger.error(f"Unknown warehouse name: {warehouse_name}")
            return False
        
        return email_service.send_email(
            to_emails=[test_email],
            subject=subject,
            body=body,
            cc_emails=None  # No CC in test mode
        )
    
    # Production mode - original recipients
    email_service = EmailService()
    
    if warehouse_name.lower() == "oxnard":
        # Oxnard configuration
        to_emails = ["robert@coldchain3pl.com"]
        cc_emails = [
            "mara@goodhillfarms.com",
            "supply@goodhillfarms.com", 
            "dara.chapman@coldcart.co",
            "sasha@goodhillfarms.com"
        ]
        
        subject = "Oxnard Inventory Form - Weekly Count"
        
        body = f"""Hi Robert,

Please fill out the cells in green in Inventory Form: {sheet_url}

This form will show the SKU, lot dates, and expected quantity of each SKU. The goal is to standardize the process across both FCs and improve the accuracy of hard counts each week.

Please let me know if you have any questions.

Best regards,
Good Hill Farms Team"""
        
    elif warehouse_name.lower() == "wheeling":
        # Wheeling configuration
        to_emails = [
            "janet@coldchain3pl.com",
            "omar@coldchain3pl.com", 
            "armando@coldchain3pl.com"
        ]
        cc_emails = [
            "supply@goodhillfarms.com",
            "mara@goodhillfarms.com",
            "dara.chapman@coldcart.co",
            "sasha@goodhillfarms.com"
        ]
        
        subject = "Wheeling Inventory Form - Weekly Count"
        
        body = f"""Hi Wheeling Team,

Please fill out the cells in green in Inventory Form: {sheet_url}

This form will show the SKU, lot dates, and expected quantity of each SKU. We'd like for this form to standardize the process for both FCs and to increase the accuracy of the hard counts each week.

Please let me know if you have any questions.

Best regards,
Good Hill Farms Team"""
        
    else:
        logger.error(f"Unknown warehouse name: {warehouse_name}")
        return False
    
    return email_service.send_email(
        to_emails=to_emails,
        subject=subject, 
        body=body,
        cc_emails=cc_emails
    )

def send_test_email() -> bool:
    """Send a test email to verify email configuration"""
    email_service = EmailService()
    
    return email_service.send_email(
        to_emails=["olena@goodhillfarms.com"],  # Send to Olena for testing
        subject="Test Email - Warehouse Inventory Automation",
        body="This is a test email to verify the email configuration is working correctly. If you receive this, the SMTP setup is functioning properly!"
    )
