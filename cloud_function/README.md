# Fulfillment Snapshot Cloud Function

This Cloud Function creates fulfillment snapshots with both projection data and inventory data in a single Google Sheets spreadsheet. It's designed to run independently from the Streamlit app, solving the issue where background tasks would fail when users navigate away from the page.

## Features

- Creates unified snapshots with both projection and inventory data as separate tabs in one spreadsheet
- Organizes snapshots in a year/month folder structure in Google Drive
- Preserves all formatting, styles, and data validation rules using Google Sheets' native copyTo API
- Handles authentication via a secret key
- Provides comprehensive error handling and logging
- Can be triggered via HTTP requests or scheduled via Google Apps Script

## Deployment

The Cloud Function is deployed to Google Cloud Platform using the `deploy.sh` script:

```bash
./deploy.sh
```

## Environment Variables

The Cloud Function requires the following environment variables:

- `TRIGGER_SECRET_KEY`: Secret key for authenticating HTTP requests
- `COLDCART_API_TOKEN`: API token for accessing ColdCart inventory data
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to the Google service account credentials JSON file

## Usage

The Cloud Function can be triggered using either:

- **GET request**: `https://fulfillment-snapshot-creator-ci6tl4rvuq-ue.a.run.app?key=fulfillment_projection_snapshot_trigger`
- **POST request**: 
  ```
  curl -X POST -H "Content-Type: application/json" -d '{"key":"fulfillment_projection_snapshot_trigger"}' https://fulfillment-snapshot-creator-ci6tl4rvuq-ue.a.run.app
  ```

## Google Apps Script Integration

A Google Apps Script is provided (`google_apps_script.js`) that can be used to trigger the Cloud Function on a schedule. It includes functions for:

- Triggering the snapshot creation
- Setting up daily or weekly schedules
- Managing triggers

## Files

- `main.py`: Cloud Function HTTP handler
- `snapshot_creator.py`: Core snapshot creation logic
- `inventory_api.py`: ColdCart API integration
- `google_sheets.py`: Google Sheets API utilities
- `requirements.txt`: Python dependencies
- `deploy.sh`: Deployment script
- `google_apps_script.js`: Google Apps Script for scheduled triggers
- `.env`: Environment variables configuration

## Troubleshooting

If you encounter issues with the Cloud Function, check the logs using:

```bash
gcloud functions logs read fulfillment-snapshot-creator --gen2 --region=us-east1 --limit=20
```
