# Cloud Function Deployment Options & Configuration Guide

## Overview

This cloud function provides automated fulfillment and inventory management capabilities for Good Hill Farms, including snapshot creation, inventory tracking, and automated email notifications to warehouse teams. The system supports multiple deployment configurations and operational modes.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Cloud Function Endpoints](#cloud-function-endpoints)
3. [Deployment Options](#deployment-options)
4. [Environment Configuration](#environment-configuration)
5. [Scheduling Options](#scheduling-options)
6. [Testing Configurations](#testing-configurations)
7. [Production Deployment Guide](#production-deployment-guide)
8. [Monitoring & Troubleshooting](#monitoring--troubleshooting)

## Architecture Overview

### Core Components

- **`main.py`**: HTTP Cloud Function entry points with CORS support
- **`snapshot_creator.py`**: Creates unified fulfillment snapshots with projection and inventory data
- **`inventory_scheduler.py`**: Manages automated inventory email scheduling (Thursday 4:00 AM PST)
- **`email_service.py`**: SMTP email service for warehouse notifications
- **`inventory_api.py`**: ColdCart API integration for real-time inventory data
- **`excel_generator.py`**: Generates formatted inventory sheets for warehouses
- **`google_sheets.py`**: Google Sheets API integration and formatting utilities

### External Integrations

- **Google Sheets API**: For creating and formatting inventory spreadsheets
- **Google Drive API**: For organizing files in warehouse-specific folders
- **ColdCart API**: For fetching real-time inventory data
- **Gmail SMTP**: For sending automated emails to warehouse teams
- **Google Cloud Scheduler**: For automated triggering

## Cloud Function Endpoints

### 1. Snapshot Creation
- **Endpoint**: `/` (default) or `/create_snapshot`
- **Function**: `create_snapshot(request)`
- **Purpose**: Creates unified fulfillment snapshots with projection and inventory data
- **Methods**: GET, POST
- **Authentication**: Secret key required

### 2. Test Inventory Emails
- **Endpoint**: `/test_inventory_emails`
- **Function**: `test_inventory_emails(request)`
- **Purpose**: Sends test emails to `olena@goodhillfarms.com` instead of warehouse teams
- **Methods**: GET, POST
- **Authentication**: Secret key required

### 3. Production Inventory Emails
- **Endpoint**: `/send_inventory_emails`
- **Function**: `send_inventory_emails(request)`
- **Purpose**: Sends real inventory emails to warehouse teams (PRODUCTION)
- **Methods**: GET, POST
- **Authentication**: Secret key required

### 4. Basic Email Test
- **Endpoint**: `/test_basic_email`
- **Function**: `test_basic_email(request)`
- **Purpose**: Sends simple test email to verify SMTP configuration
- **Methods**: GET, POST
- **Authentication**: Secret key required

## Deployment Options

### Option 1: Google Cloud Functions (Recommended)

**Deployment Command:**
```bash
gcloud functions deploy fulfillment-automation \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 2GB \
    --timeout 540s \
    --region us-east1 \
    --source . \
    --entry-point create_snapshot
```

**Multi-Function Deployment:**
```bash
# Snapshot creator
gcloud functions deploy fulfillment-snapshot-creator \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 2GB \
    --timeout 540s \
    --region us-east1 \
    --source . \
    --entry-point create_snapshot

# Inventory email automation
gcloud functions deploy inventory-email-automation \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 1GB \
    --timeout 300s \
    --region us-east1 \
    --source . \
    --entry-point send_inventory_emails

# Test email automation
gcloud functions deploy test-inventory-emails \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 1GB \
    --timeout 300s \
    --region us-east1 \
    --source . \
    --entry-point test_inventory_emails
```

### Option 2: Google Cloud Run

**Deployment using cloudbuild.yaml:**
```bash
gcloud builds submit --config=../cloudbuild.yaml
```

**Direct Cloud Run deployment:**
```bash
gcloud run deploy fulfillment-automation \
    --source . \
    --platform managed \
    --region us-east1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 540s \
    --max-instances 10
```

### Option 3: Local Development

**Direct Python execution:**
```bash
cd cloud_function
python test_local.py
```

**Functions Framework (local testing):**
```bash
functions-framework --target=create_snapshot --debug
```

## Environment Configuration

### Required Environment Variables

#### Google Cloud Authentication
```bash
# Service Account JSON (for local development)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# OR use inline JSON (for Cloud Functions)
GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
```

#### Email Configuration
```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=hello@goodhillfarms.com
SMTP_PASSWORD=xufm_lmxf_ehvx_fjxz  # Gmail App Password
SMTP_USE_TLS=true
```

#### API Authentication
```bash
TRIGGER_SECRET_KEY=your_secure_secret_key
COLDCART_API_TOKEN=your_coldcart_api_token
```

### Optional Configuration
```bash
# Logging level
LOG_LEVEL=INFO

# Warehouse folder IDs (Google Drive)
OXNARD_FOLDER_ID=1BDW2dd41h6_gvdUWVsUmZHIdd8gkIU_c
WHEELING_FOLDER_ID=1xogLAldd3dUGKaEXAk_0UUILWxEpIkSP
```

### Setting Environment Variables

#### For Cloud Functions:
```bash
gcloud functions deploy FUNCTION_NAME \
    --set-env-vars TRIGGER_SECRET_KEY=your_key,SMTP_USERNAME=email@domain.com \
    --update-env-vars COLDCART_API_TOKEN=new_token
```

#### For Cloud Run:
```bash
gcloud run deploy SERVICE_NAME \
    --set-env-vars TRIGGER_SECRET_KEY=your_key \
    --update-env-vars SMTP_PASSWORD=new_password
```

## Scheduling Options

### Option 1: Google Cloud Scheduler (Recommended)

#### Weekly Inventory Emails (Thursday 4:00 AM PST)
```bash
gcloud scheduler jobs create http inventory-emails-weekly \
    --schedule="0 4 * * 4" \
    --timezone="America/Los_Angeles" \
    --uri="https://YOUR_FUNCTION_URL/send_inventory_emails" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"key":"your_secret_key"}' \
    --description="Send weekly inventory emails every Thursday at 4am PST"
```

#### Daily Snapshot Creation
```bash
gcloud scheduler jobs create http daily-snapshots \
    --schedule="0 6 * * *" \
    --timezone="America/Los_Angeles" \
    --uri="https://YOUR_FUNCTION_URL/create_snapshot" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"key":"your_secret_key"}' \
    --description="Create daily fulfillment snapshots"
```

### Option 2: Google Apps Script

```javascript
function triggerInventoryEmails() {
  const url = 'https://YOUR_FUNCTION_URL/send_inventory_emails';
  const payload = {
    'key': 'your_secret_key'
  };
  
  const options = {
    'method': 'POST',
    'contentType': 'application/json',
    'payload': JSON.stringify(payload)
  };
  
  UrlFetchApp.fetch(url, options);
}

function createWeeklyTrigger() {
  ScriptApp.newTrigger('triggerInventoryEmails')
    .timeBased()
    .everyWeeks(1)
    .onWeekDay(ScriptApp.WeekDay.THURSDAY)
    .atHour(4)
    .create();
}
```

### Option 3: External Cron Services

#### Cron-job.org Configuration:
- **URL**: `https://YOUR_FUNCTION_URL/send_inventory_emails`
- **Method**: POST
- **Headers**: `Content-Type: application/json`
- **Body**: `{"key":"your_secret_key"}`
- **Schedule**: `0 4 * * 4` (Thursday 4:00 AM)

## Testing Configurations

### Local Testing

#### Test Email Functionality
```bash
cd cloud_function
python test_email_only.py
```

#### Test with Real Templates
```bash
python test_real_templates.py
```

#### Test Complete Flow
```bash
python test_local.py
```

### Cloud Testing

#### Test Basic Email
```bash
curl -X POST "https://YOUR_FUNCTION_URL/test_basic_email" \
  -H "Content-Type: application/json" \
  -d '{"key":"your_secret_key"}'
```

#### Test Inventory Emails (Test Mode)
```bash
curl -X POST "https://YOUR_FUNCTION_URL/test_inventory_emails" \
  -H "Content-Type: application/json" \
  -d '{"key":"your_secret_key"}'
```

#### Test Snapshot Creation
```bash
curl -X GET "https://YOUR_FUNCTION_URL/create_snapshot?key=your_secret_key"
```

## Production Deployment Guide

### Step 1: Prepare Environment
```bash
# Set project
gcloud config set project your-project-id

# Enable required APIs
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable gmail.googleapis.com
```

### Step 2: Deploy Functions
```bash
cd cloud_function

# Deploy main snapshot function
gcloud functions deploy fulfillment-snapshot-creator \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 2GB \
    --timeout 540s \
    --region us-east1 \
    --source . \
    --entry-point create_snapshot \
    --set-env-vars TRIGGER_SECRET_KEY=$SECRET_KEY,SMTP_USERNAME=$SMTP_USER

# Deploy inventory email function
gcloud functions deploy inventory-email-automation \
    --runtime python311 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 1GB \
    --timeout 300s \
    --region us-east1 \
    --source . \
    --entry-point send_inventory_emails \
    --set-env-vars TRIGGER_SECRET_KEY=$SECRET_KEY,SMTP_USERNAME=$SMTP_USER
```

### Step 3: Configure Scheduling
```bash
# Create weekly inventory email job
gcloud scheduler jobs create http inventory-emails-weekly \
    --schedule="0 4 * * 4" \
    --timezone="America/Los_Angeles" \
    --uri="https://inventory-email-automation-HASH-ue.a.run.app" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"key":"'$SECRET_KEY'"}' \
    --description="Weekly inventory emails - Thursday 4am PST"
```

### Step 4: Test Production Deployment
```bash
# Test snapshot creation
curl -X POST "https://fulfillment-snapshot-creator-HASH-ue.a.run.app" \
  -H "Content-Type: application/json" \
  -d '{"key":"'$SECRET_KEY'"}'

# Test email automation (test mode first)
curl -X POST "https://inventory-email-automation-HASH-ue.a.run.app/test_inventory_emails" \
  -H "Content-Type: application/json" \
  -d '{"key":"'$SECRET_KEY'"}'
```

## Monitoring & Troubleshooting

### Logging

#### View Cloud Function Logs
```bash
# Recent logs
gcloud functions logs read fulfillment-snapshot-creator --limit=50

# Follow logs in real-time
gcloud functions logs tail fulfillment-snapshot-creator

# Filter by severity
gcloud functions logs read fulfillment-snapshot-creator --filter="severity>=ERROR"
```

#### View Cloud Scheduler Logs
```bash
gcloud logging read "resource.type=cloud_scheduler_job" --limit=20
```

### Common Issues & Solutions

#### Authentication Errors
```bash
# Check service account permissions
gcloud projects get-iam-policy your-project-id

# Re-authenticate locally
gcloud auth application-default login
```

#### Email Failures
- Verify Gmail App Password is correct
- Check SMTP credentials in environment variables
- Test with `test_basic_email` endpoint first

#### API Timeouts
- Increase function timeout for snapshot creation
- Check ColdCart API connectivity
- Verify Google Sheets API quotas

#### Memory Issues
- Increase memory allocation for functions processing large datasets
- Monitor memory usage in Cloud Console

### Health Checks

#### Function Status
```bash
gcloud functions describe FUNCTION_NAME --region=us-east1
```

#### Scheduler Status
```bash
gcloud scheduler jobs describe JOB_NAME --location=us-east1
```

### Performance Optimization

#### Resource Allocation
- **Snapshot Creation**: 2GB memory, 540s timeout
- **Email Functions**: 1GB memory, 300s timeout
- **Test Functions**: 512MB memory, 60s timeout

#### Concurrency Settings
```bash
gcloud functions deploy FUNCTION_NAME \
    --max-instances 10 \
    --concurrency 1
```

## Security Considerations

### Secret Management
- Use Google Secret Manager for sensitive data
- Rotate API keys regularly
- Implement least-privilege access

### Network Security
- Enable VPC connector if needed
- Use private Google access
- Implement IP allowlisting for sensitive endpoints

### Function Security
```bash
# Deploy with specific service account
gcloud functions deploy FUNCTION_NAME \
    --service-account your-service-account@project.iam.gserviceaccount.com

# Restrict invoker permissions
gcloud functions add-iam-policy-binding FUNCTION_NAME \
    --member="serviceAccount:scheduler@project.iam.gserviceaccount.com" \
    --role="roles/cloudfunctions.invoker"
```

## Cost Optimization

### Pricing Considerations
- **Cloud Functions**: Pay per invocation and compute time
- **Cloud Scheduler**: $0.10 per job per month
- **Google Sheets API**: Free tier: 100 requests/100 seconds/user
- **Gmail API**: Free tier: 1 billion quota units/day

### Optimization Strategies
- Use appropriate memory allocation
- Implement caching for frequently accessed data
- Optimize cold start times with smaller deployment packages
- Monitor and adjust timeout settings

## Backup & Recovery

### Data Backup
- Google Sheets are automatically backed up by Google Drive
- Export critical configuration to version control
- Document environment variable settings

### Disaster Recovery
- Multi-region deployment for critical functions
- Automated health checks and alerting
- Rollback procedures for failed deployments

## Contact & Support

For issues related to this deployment:
- **Technical Issues**: Check logs and common troubleshooting steps
- **Business Logic**: Review function documentation and test locally
- **Infrastructure**: Verify Google Cloud service status and quotas

---

*Last Updated: [Current Date]*
*Version: 1.0*
