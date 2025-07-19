"""
FastAPI endpoints for the fulfillment app.
This runs alongside the Streamlit app in the same container.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

# Import the snapshot creation utility
from utils.snapshot_creator import create_fulfillment_snapshot

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Make sure other loggers are also set to INFO level
logging.getLogger('utils.snapshot_creator').setLevel(logging.INFO)
logging.getLogger('utils.google_sheets').setLevel(logging.INFO)
logging.getLogger('utils.inventory_api').setLevel(logging.INFO)

# Create FastAPI app
app = FastAPI(
    title="Fulfillment API",
    description="API endpoints for fulfillment operations",
    version="1.0.0"
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Fulfillment API is running", "status": "healthy"}

@app.get("/api/snapshot")
async def create_snapshot_endpoint(
    key: str = Query(..., description="Secret key for authentication")
):
    """
    Create a fulfillment snapshot with both projection and inventory data.
    
    This endpoint triggers the snapshot creation as a background task and returns immediately.
    """
    try:
        # Verify the secret key
        expected_key = os.environ.get('TRIGGER_SECRET_KEY', 'fulfillment_projection_snapshot_trigger')
        if key != expected_key:
            logger.warning(f"Invalid secret key provided: {key}")
            raise HTTPException(status_code=401, detail="Invalid secret key")
        
        logger.info("🚀 Starting snapshot creation as background task...")
        
        # Start the snapshot creation as a background task
        background_task = asyncio.create_task(run_snapshot_in_background())
        
        # Return immediately with a success message
        return JSONResponse(
            status_code=202,  # 202 Accepted indicates the request was accepted but processing is not complete
            content={
                "success": True,
                "message": "Snapshot creation started in background",
                "status": "processing"
            }
        )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in snapshot endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

async def run_snapshot_in_background():
    """Run the snapshot creation in the background"""
    try:
        logger.info("🚀 Background task: Starting snapshot creation...")
        result = await create_fulfillment_snapshot()
        
        if result and result.get('success'):
            logger.info("✅ Background task: Snapshot creation completed successfully")
            logger.info(f"Spreadsheet URL: {result.get('spreadsheet_url')}")
        else:
            logger.error("❌ Background task: Snapshot creation failed")
            error_detail = result.get('error', 'Unknown error') if result else 'Unknown error'
            logger.error(f"Error details: {error_detail}")
    except Exception as e:
        logger.error(f"❌ Background task: Unexpected error in snapshot creation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

@app.get("/api/health")
async def health_check():
    """Detailed health check with system information"""
    try:
        # Check if we can access environment variables
        has_google_creds = bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') or 
                               os.path.exists('nca-toolkit-project-446011-67d246fdbccf.json'))
        has_coldcart_token = bool(os.environ.get('COLDCART_API_TOKEN'))
        
        return {
            "status": "healthy",
            "environment": {
                "google_credentials": has_google_creds,
                "coldcart_token": has_coldcart_token,
                "trigger_secret_key": bool(os.environ.get('TRIGGER_SECRET_KEY'))
            },
            "endpoints": {
                "snapshot": "/api/snapshot?key=YOUR_SECRET_KEY",
                "health": "/api/health"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

if __name__ == "__main__":
    # This allows running the API standalone for testing
    uvicorn.run(app, host="0.0.0.0", port=8001)
