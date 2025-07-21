"""
Startup script to run both Streamlit and FastAPI in the same container.
"""

import os
import subprocess
import threading
import time
import logging
import socket

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Make sure other loggers are also set to INFO level
logging.getLogger('utils').setLevel(logging.INFO)
logging.getLogger('utils.snapshot_creator').setLevel(logging.INFO)
logging.getLogger('utils.google_sheets').setLevel(logging.INFO)
logging.getLogger('utils.inventory_api').setLevel(logging.INFO)
logging.getLogger('api').setLevel(logging.INFO)

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    logger.warning(f"Could not find available port after {max_attempts} attempts")
    return start_port  # Return original port as last resort

def run_streamlit(port=8501):
    """Run Streamlit app"""
    streamlit_port = port
    if is_port_in_use(streamlit_port):
        streamlit_port = find_available_port(streamlit_port)
        logger.info(f"Port {port} is in use, using port {streamlit_port} for Streamlit instead")
    
    logger.info(f"🚀 Starting Streamlit app on port {streamlit_port}...")
    try:
        subprocess.run([
            "streamlit", "run", "app.py",
            "--server.port", str(streamlit_port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Streamlit failed: {e}")
        raise

def run_fastapi(port=8001):
    """Run FastAPI app"""
    api_port = port
    if is_port_in_use(api_port):
        api_port = find_available_port(api_port)
        logger.info(f"Port {port} is in use, using port {api_port} for FastAPI instead")
    
    logger.info(f"🚀 Starting FastAPI app on port {api_port}...")
    try:
        subprocess.run([
            "uvicorn", "api:app",
            "--host", "0.0.0.0",
            "--port", str(api_port),
            "--reload"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ FastAPI failed: {e}")
        raise

def main():
    """Start both applications"""
    logger.info("🌟 Starting Fulfillment App with API endpoints...")
    
    # Get the port from Cloud Run environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Cloud Run PORT environment variable: {port}")
    
    # Start FastAPI on the Cloud Run port
    api_thread = threading.Thread(target=run_fastapi, args=(port,), daemon=True)
    api_thread.start()
    
    # Give FastAPI a moment to start
    time.sleep(2)
    
    # Start Streamlit on a different port
    streamlit_port = find_available_port(8501)
    run_streamlit(streamlit_port)

if __name__ == "__main__":
    main()
