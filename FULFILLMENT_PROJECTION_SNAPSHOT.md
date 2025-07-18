# Fulfillment Projection Snapshot System

## Overview
The Fulfillment Projection Snapshot system creates perfect copies of the `ALL_Picklist_V2` Google Sheets data with complete formatting, styling, and functionality preserved. This system is designed for creating reliable, timestamped snapshots for reporting and analysis.

## 🚀 Quick Start

### Trigger URL
```
http://localhost:8501/hidden_trigger?key=fulfillment_projection_snapshot_trigger
```

### What It Does
1. **Creates organized directory structure** in Google Drive (`2025/07-July/`)
2. **Generates timestamped spreadsheet** (`ALL_Picklist_2025-07-18_1430`)
3. **Copies complete data and formatting** from source `ALL_Picklist_V2` sheet
4. **Preserves all styling** (colors, fonts, borders, column widths)
5. **Maintains frozen headers** for easy navigation
6. **Enables filters** for sorting and filtering data
7. **Uses calculated values** (no broken formulas)

## 📋 Features

### ✅ Data Integrity
- **Exact data range detection** - No unnecessary empty rows/columns
- **Calculated values only** - No formulas that could break
- **Complete data preservation** - All visible data copied exactly

### ✅ Formatting & Styling
- **Cell formatting** - Colors, fonts, borders, text alignment
- **Column widths** - Exact column sizing preserved
- **Row heights** - Proper row spacing maintained
- **Frozen rows/columns** - Table headers stay visible when scrolling

### ✅ Functionality
- **Filters enabled** - Sort and filter data as needed
- **Proper sheet structure** - Clean, professional layout
- **Single sheet output** - Only `ALL_Picklist_V2` (no extra sheets)

### ✅ Organization
- **Year/Month folders** - Automatic directory structure
- **Timestamped filenames** - Easy identification and sorting
- **Clean naming** - No "_Copy" suffixes or confusing names

## 🗂️ Directory Structure

```
📁 Projection Snapshots (Base Folder)
├── 📁 2025/
│   ├── 📁 07-July/
│   │   ├── 📄 ALL_Picklist_2025-07-18_1430.xlsx
│   │   ├── 📄 ALL_Picklist_2025-07-18_1545.xlsx
│   │   └── 📄 ALL_Picklist_2025-07-19_0900.xlsx
│   ├── 📁 08-August/
│   │   └── 📄 (future snapshots)
│   └── 📁 09-September/
└── 📁 2024/
    └── 📁 (previous year snapshots)
```

## 🔧 Technical Implementation

### Source Configuration
- **Source Spreadsheet**: `GHF_AGGREGATION_DASHBOARD_ID`
- **Source Sheet**: `ALL_Picklist_V2`
- **Target Folder**: Google Drive folder with ID `1-uUvyCTEx_TLKOF46jHD3Kpsp8aO9W9b`

### Authentication
- Uses Google Service Account credentials
- Requires access to both Google Sheets API and Google Drive API
- Service account file: `nca-toolkit-project-446011-67d246fdbccf.json`

### Process Flow
1. **Authenticate** with Google APIs
2. **Create year/month folders** (if they don't exist)
3. **Create new spreadsheet** with `ALL_Picklist_V2` sheet
4. **Detect actual data range** from source sheet
5. **Copy data values** (calculated, not formulas)
6. **Apply comprehensive formatting** (colors, fonts, borders, etc.)
7. **Set column widths and row heights**
8. **Copy frozen rows/columns**
9. **Enable filters** on data range
10. **Remove default Sheet1**
11. **Return success with link**

## 📊 Output Example

### Filename Format
```
ALL_Picklist_2025-07-18_1430
```
- `ALL_Picklist` - Consistent prefix
- `2025-07-18` - Date (YYYY-MM-DD)
- `1430` - Time (HHMM, 24-hour format)

### Sheet Structure
- **Sheet Name**: `ALL_Picklist_V2` (original name preserved)
- **Data Range**: Automatically detected (no empty rows/columns)
- **Headers**: Frozen for easy navigation
- **Filters**: Enabled on entire data range
- **Formatting**: Complete preservation of source styling

## 🛠️ Usage Instructions

### 1. Basic Usage
Simply navigate to the trigger URL:
```
http://localhost:8501/hidden_trigger?key=fulfillment_projection_snapshot_trigger
```

### 2. Monitor Progress
The system provides real-time feedback:
- ✅ Authentication status
- ✅ Folder creation progress
- ✅ Data copying status
- ✅ Formatting application
- ✅ Final success with link

### 3. Access Results
After completion, you'll receive:
- **Direct link** to the new spreadsheet
- **Summary** of what was copied
- **Location** in the folder structure

## 🔍 Troubleshooting

### Common Issues

#### Authentication Errors
- **Symptom**: "Failed to get Google credentials"
- **Solution**: Ensure service account file exists and has proper permissions

#### Permission Errors
- **Symptom**: "Permission denied" or "Access forbidden"
- **Solution**: Verify service account has access to source spreadsheet and target folder

#### Empty Results
- **Symptom**: "No data found in ALL_Picklist_V2"
- **Solution**: Check if source sheet exists and contains data

#### Formatting Issues
- **Symptom**: "Could not copy all formatting"
- **Solution**: This is usually non-critical; data and basic formatting will still be preserved

### Success Indicators
- ✅ "Fulfillment projection snapshot created successfully!"
- ✅ Direct link to new spreadsheet provided
- ✅ Summary of copied features displayed

## 🔐 Security

### Access Control
- **Secret key required**: `fulfillment_projection_snapshot_trigger`
- **Service account authentication**: Secure API access
- **No user data exposure**: Only system-level operations

### Data Protection
- **Read-only source access**: Original data never modified
- **Secure copying**: Data transmitted via encrypted Google APIs
- **Organized storage**: Files stored in designated Google Drive folder

## 📈 Performance

### Typical Processing Time
- **Small datasets** (< 1000 rows): 10-30 seconds
- **Medium datasets** (1000-5000 rows): 30-60 seconds
- **Large datasets** (5000+ rows): 1-3 minutes

### Optimization Features
- **Batch formatting requests**: Efficient API usage
- **Smart data range detection**: Only processes actual data
- **Minimal API calls**: Optimized for speed and quota usage

## 🔄 Maintenance

### Regular Tasks
- **Monitor API quotas**: Ensure sufficient Google API usage limits
- **Check folder permissions**: Verify service account access
- **Review storage usage**: Monitor Google Drive space

### Updates
- **Credential renewal**: Update service account keys as needed
- **Folder structure**: Modify directory organization if required
- **Formatting enhancements**: Add new styling features as needed

## 📞 Support

### Error Reporting
When reporting issues, include:
- **Full error message** from the Streamlit interface
- **Timestamp** of the attempt
- **Expected vs actual behavior**
- **Source data characteristics** (size, complexity)

### Enhancement Requests
For new features or improvements:
- **Describe the use case** clearly
- **Specify desired behavior** in detail
- **Consider impact** on existing functionality

---

## 🎯 Summary

The Fulfillment Projection Snapshot system provides a reliable, automated way to create perfect copies of your picklist data with complete formatting and functionality preserved. Use the simple trigger URL to generate organized, timestamped snapshots whenever needed.

**Trigger URL**: `http://localhost:8501/hidden_trigger?key=fulfillment_projection_snapshot_trigger`
