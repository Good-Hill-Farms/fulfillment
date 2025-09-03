# Warehouse Inventory Email Automation Setup

## Overview
This setup enables automatic sending of warehouse inventory forms every Thursday at 4:00 AM PST.

## Components Created

1. **email_service.py** - Handles SMTP email sending with Gmail
2. **inventory_scheduler.py** - Manages scheduling logic and batch operations
3. **Updated excel_generator.py** - Integrated email sending with sheet generation
4. **Updated main.py** - Added new endpoints for email automation

## New Endpoints

Your Cloud Function now supports these endpoints:

- `GET/POST /snapshot` - Original snapshot creation (default)
- `GET/POST /inventory-emails` - Send emails only if it's Thursday 4:00 AM PST
- `GET/POST /inventory-emails-now` - Force send emails immediately (for testing)
- `GET/POST /test-email` - Send a test email to verify configuration

All endpoints require the `key` parameter with your trigger secret.

## Environment Variables Required

Make sure these environment variables are set in your Cloud Function:

```bash
# Email Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=hello@goodhillfarms.com
SMTP_PASSWORD=xufm lmxf ehvx fjxz
SMTP_USE_TLS=true

# Existing variables (keep these)
TRIGGER_SECRET_KEY=your_secret_key
GOOGLE_APPLICATION_CREDENTIALS_JSON=your_credentials_json
```

## Google Cloud Scheduler Setup

To automatically trigger the emails every Thursday at 4:00 AM PST, create a Cloud Scheduler job:

### Option 1: Using gcloud CLI

```bash
gcloud scheduler jobs create http inventory-emails-weekly \
    --schedule="0 4 * * 4" \
    --timezone="America/Los_Angeles" \
    --uri="https://YOUR_CLOUD_FUNCTION_URL/inventory-emails" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{"key":"your_secret_key"}' \
    --description="Send weekly inventory emails every Thursday at 4am PST"
```

### Option 2: Using Google Cloud Console

1. Go to Cloud Scheduler in Google Cloud Console
2. Click "Create Job"
3. Fill in:
   - **Name**: `inventory-emails-weekly`
   - **Frequency**: `0 4 * * 4` (cron format for Thursday 4:00 AM)
   - **Timezone**: `America/Los_Angeles`
   - **Target Type**: HTTP
   - **URL**: `https://YOUR_CLOUD_FUNCTION_URL/inventory-emails`
   - **HTTP Method**: POST
   - **Headers**: `Content-Type: application/json`
   - **Body**: `{"key":"your_secret_key"}`

## Testing

### Test Email Configuration
```bash
curl -X POST "https://YOUR_CLOUD_FUNCTION_URL/test-email" \
  -H "Content-Type: application/json" \
  -d '{"key":"your_secret_key"}'
```

### Test Immediate Email Sending
```bash
curl -X POST "https://YOUR_CLOUD_FUNCTION_URL/inventory-emails-now" \
  -H "Content-Type: application/json" \
  -d '{"key":"your_secret_key"}'
```

### Test Scheduled Email Logic
```bash
curl -X POST "https://YOUR_CLOUD_FUNCTION_URL/inventory-emails" \
  -H "Content-Type: application/json" \
  -d '{"key":"your_secret_key"}'
```

## Email Recipients

### Oxnard Warehouse
- **To**: robert@coldchain3pl.com
- **CC**: mara@goodhillfarms.com, supply@goodhillfarms.com, dara.chapman@coldcart.co, sasha@goodhillfarms.com

### Wheeling Warehouse  
- **To**: janet@coldchain3pl.com, omar@coldchain3pl.com, armando@coldchain3pl.com
- **CC**: supply@goodhillfarms.com, mara@goodhillfarms.com, dara.chapman@coldcart.co, sasha@goodhillfarms.com

## Deployment Notes

1. Deploy the updated Cloud Function with the new files
2. Set the required environment variables
3. Create the Cloud Scheduler job
4. Test the setup using the test endpoints
5. Monitor the first scheduled run on Thursday

## Troubleshooting

- Check Cloud Function logs for any errors
- Verify email credentials and Gmail App Password
- Test individual components using the provided endpoints
- Check Cloud Scheduler job execution history



