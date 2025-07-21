"""
Startup script to run Streamlit in the container.
"""

import os
import subprocess
import logging

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_streamlit(port=8080):
    """Run Streamlit app"""
    logger.info(f"🚀 Starting Streamlit app on port {port}...")
    try:
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Streamlit failed: {e}")
        raise

def main():
    """Start Streamlit application"""
    logger.info("🌟 Starting Fulfillment App...")
    
    # Get the port from Cloud Run environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Cloud Run PORT environment variable: {port}")
    
    # Start Streamlit on the main Cloud Run port
    run_streamlit(port)

if __name__ == "__main__":
    main()
