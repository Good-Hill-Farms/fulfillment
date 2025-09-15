"""
ColdCart Token Extraction Module
Handles token extraction from the new ColdCart token API
"""

import requests
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

def extract_coldcart_token() -> Optional[str]:
    """
    Extract ColdCart API token using the new token extraction service
    
    Returns:
        str: The extracted API token, or None if extraction failed
    """
    token_api_url = "https://coldcart-token-extractor-api-180321025165.us-central1.run.app/extract-token"
    
    headers = {
        "Authorization": "Bearer test-verification-key-456",
        "Content-Type": "application/json"
    }
    
    payload = {
        "force_refresh": False
    }
    
    try:
        logger.info("Extracting ColdCart token...")
        response = requests.post(token_api_url, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('success') and data.get('token'):
            token = data['token']
            logger.info("✅ Successfully extracted ColdCart token")
            return token
        else:
            error_msg = data.get('error', 'Unknown error')
            logger.error(f"❌ Token extraction failed: {error_msg}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to call token extraction API: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"❌ Error extracting token: {str(e)}")
        return None

def get_coldcart_api_token() -> Optional[str]:
    """
    Get ColdCart API token - always uses the automatic token extraction API
    
    Returns:
        str: The API token, or None if not available
    """
    # Always use the automatic token extraction API
    logger.info("Getting fresh ColdCart token from automatic token extractor API...")
    return extract_coldcart_token()

if __name__ == "__main__":
    # Test the token extraction
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ColdCart token extraction...")
    token = get_coldcart_api_token()
    
    if token:
        print(f"✅ Token extracted successfully: {token[:50]}...")
    else:
        print("❌ Failed to extract token")
