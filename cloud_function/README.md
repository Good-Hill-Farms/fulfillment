# Fulfillment Snapshot Cloud Function

This Cloud Function creates fulfillment snapshots with both projection data and inventory data in a single Google Sheets spreadsheet. It's designed to run independently from the Streamlit app, solving the issue where background tasks would fail when users navigate away from the page.

## GitHub
**Main Repository**: https://github.com/Good-Hill-Farms/fulfillment  
**Cloud Functions**: https://github.com/Good-Hill-Farms/fulfillment/tree/main/cloud_function

## How It Works

The system pulls data from multiple Google Sheets, combines it with live ColdCart inventory data, and creates organized snapshots and warehouse inventory sheets:

1. **Snapshot Creation**: Copies projection data from GHF Aggregation Dashboard and adds live inventory data
2. **Warehouse Emails**: Generates formatted inventory sheets for Oxnard and Wheeling warehouses
3. **Automated Scheduling**: Sends weekly inventory emails every Thursday at 4:00 AM PST
4. **File Organization**: Creates year/month folder structure in Google Drive

## Google Drive & Sheets Used

### Main Folders
- **Snapshots**: [1-uUvyCTEx_TLKOF46jHD3Kpsp8aO9W9b](https://drive.google.com/drive/folders/1-uUvyCTEx_TLKOF46jHD3Kpsp8aO9W9b)
- **Oxnard Warehouse**: [1BDW2dd41h6_gvdUWVsUmZHIdd8gkIU_c](https://drive.google.com/drive/folders/1BDW2dd41h6_gvdUWVsUmZHIdd8gkIU_c)
- **Wheeling Warehouse**: [1xogLAldd3dUGKaEXAk_0UUILWxEpIkSP](https://drive.google.com/drive/folders/1xogLAldd3dUGKaEXAk_0UUILWxEpIkSP)

### Source Sheets
- **GHF Aggregation Dashboard**: [1CdTTV8pMqq_wS9vu0qa8HMykNkqtOverrIsP0WLSUeM](https://docs.google.com/spreadsheets/d/1CdTTV8pMqq_wS9vu0qa8HMykNkqtOverrIsP0WLSUeM) (ALL_Picklist_V2 sheet)
- **GHF Inventory**: [19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY) (SKU mappings)
- **GHF Fruit Tracking**: [1B_uRcYEqCdR5O3h5BiyvL92Q1v4BlNPxZTsZ-nihNbI](https://docs.google.com/spreadsheets/d/1B_uRcYEqCdR5O3h5BiyvL92Q1v4BlNPxZTsZ-nihNbI) (Orders data)

### Email Recipients
- **Oxnard**: robert@coldchain3pl.com (CC: mara@goodhillfarms.com, supply@goodhillfarms.com, dara.chapman@coldcart.co, sasha@goodhillfarms.com)
- **Wheeling**: janet@coldchain3pl.com, omar@coldchain3pl.com, armando@coldchain3pl.com (CC: supply@goodhillfarms.com, mara@goodhillfarms.com, dara.chapman@coldcart.co, sasha@goodhillfarms.com)

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
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`: Google service account credentials JSON

### Email Configuration
- `SMTP_HOST=smtp.gmail.com`
- `SMTP_PORT=587`
- `SMTP_USERNAME=your_email@goodhillfarms.com`
- `SMTP_PASSWORD=your_gmail_app_password`
- `SMTP_USE_TLS=true`

## API Endpoints

### 1. Create Snapshot (Default)
- **URL**: `/` or `/create_snapshot`
- **Purpose**: Creates unified fulfillment snapshots
- **Example**: `curl -X POST "https://your-function-url/?key=your_secret_key"`

### 2. Test Inventory Emails
- **URL**: `/test_inventory_emails`
- **Purpose**: Sends test emails to olena@goodhillfarms.com
- **Example**: `curl -X POST "https://your-function-url/test_inventory_emails" -H "Content-Type: application/json" -d '{"key":"your_secret_key"}'`

### 3. Production Inventory Emails
- **URL**: `/send_inventory_emails`
- **Purpose**: Sends real emails to warehouse teams
- **Example**: `curl -X POST "https://your-function-url/send_inventory_emails" -H "Content-Type: application/json" -d '{"key":"your_secret_key"}'`

### 4. Basic Email Test
- **URL**: `/test_basic_email`
- **Purpose**: Tests SMTP configuration
- **Example**: `curl -X POST "https://your-function-url/test_basic_email" -H "Content-Type: application/json" -d '{"key":"your_secret_key"}'`

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
