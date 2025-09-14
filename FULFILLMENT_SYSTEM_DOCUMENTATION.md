# Fulfillment System Documentation

**Production**: https://fulfillment-mixy-matchi-180321025165.us-east1.run.app/

**GitHub**: https://github.com/Good-Hill-Farms/fulfillment

---

## 📖 Table of Contents

1. [🚀 How to Use](#-how-to-use)
2. [📋 Data Processing & File Formats](#-data-processing--file-formats)
3. [🔧 SKU Mapping System](#-sku-mapping-system)
4. [🔄 Smart Workflow](#-smart-workflow)
5. [⚠️ Key Actions](#️-key-actions)
6. [🔧 Troubleshooting](#-troubleshooting)

---

## 🚀 How to Use

### **Sidebar: Upload Data**

1. **📤 Upload Orders CSV** (from Shopify)
   - Required format: Shopify orders export
   - Must contain: Order ID, SKU, Quantity, Customer ZIP
   
2. **📦 Choose Inventory Source**:
   - **Upload File (default)** - Upload Inventory CSV
     - Required columns: SKU, Available Quantity, Warehouse Location
   - **ColdCart API** - Real-time inventory (requires token)
     - Auto-syncs with warehouse management system
     
3. **Auto-processing** - Triggers automatically when both files uploaded
4. **🔄 Reprocess Data** - Manual reprocess button for updates
5. **📄 Output Generation** - Creates fulfillment list for ColdCart import

### 📜 **Orders Tab**
- View all processed orders with warehouse assignments
- Filter orders with/without fulfillment issues
- Select multiple orders and move to staging
- Search orders by any field (Order ID, SKU, Customer, etc.)
- Color-coded status indicators

### 📋 **Staging Tab**
- View staged orders with protected inventory allocation
- Staged orders maintain their inventory reservations during recalculation
- Remove orders from staging when no longer needed
- Bulk staging/unstaging operations

### ⚙️ **SKU Mapping Management**
- **Edit via Google Sheet**: https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY
- **Direct Links by Warehouse**:
  - [📊 Oxnard SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=549145618) (Sheet: INPUT_bundles_cvr_oxnard)
  - [📊 Wheeling SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=0) (Sheet: INPUT_bundles_cvr_wheeling)
  - [📊 Walnut SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=1234567890) (Sheet: INPUT_bundles_cvr_walnut)
  - [📊 Northlake SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=0987654321) (Sheet: INPUT_bundles_cvr_northlake)

---

## 📋 Data Processing & File Formats

### **📤 Input Files**

#### **Orders CSV (Shopify Export)**
**Required Columns:**
```csv
Name,Email,Financial Status,Fulfillment Status,Accepts Marketing,Currency,Tags,Discount Code,Discount Amount,Shipping Method,Created at,Lineitem quantity,Lineitem name,Lineitem price,Lineitem compare at price,Lineitem sku,Lineitem requires shipping,Lineitem taxable,Lineitem fulfillment status,Billing Name,Billing Street,Billing Address1,Billing Address2,Billing Company,Billing City,Billing Zip,Billing Province,Billing Country,Billing Phone,Shipping Name,Shipping Street,Shipping Address1,Shipping Address2,Shipping Company,Shipping City,Shipping Zip,Shipping Province,Shipping Country,Shipping Phone,Notes,Note Attributes
```

**Key Processing Fields:**
- `Name` - Order identifier
- `Lineitem sku` - Product SKU for mapping
- `Lineitem quantity` - Quantity ordered
- `Shipping Zip` - Determines warehouse routing
- `Created at` - Order timestamp
- `Fulfillment Status` - Current fulfillment state

**Processing Logic:**
1. **ZIP Code Analysis**: Customer ZIP → Fulfillment Center assignment
2. **SKU Validation**: Check against SKU mapping database
3. **Bundle Detection**: Identify multi-component SKUs
4. **Inventory Allocation**: Reserve inventory per component

#### **Inventory CSV**
**Required Columns:**
```csv
SKU,Available,Warehouse,Product_Type,Last_Updated
```

**Alternative Column Names Supported:**
- SKU: `sku`, `inventory_sku`, `product_sku`
- Available: `available`, `quantity`, `stock`, `balance`
- Warehouse: `warehouse`, `location`, `fulfillment_center`

**Processing Logic:**
1. **SKU Normalization**: Clean and standardize SKU formats
2. **Warehouse Mapping**: Assign inventory to fulfillment centers
3. **Running Balance**: Track available vs. allocated inventory
4. **Shortage Detection**: Identify insufficient inventory scenarios

#### **ColdCart API Integration**
**Real-time Inventory Sync:**
- Automatic inventory updates every 15 minutes
- Webhook notifications for critical stock changes
- API token required (configured in system settings)
- Fallback to CSV upload if API unavailable

### **📄 Output Files**

#### **Fulfillment Orders CSV**
**Generated Columns:**
```csv
order_id,sku,quantity,actualqty,Total Pick Weight,fulfillment_center,customer_name,shipping_address,shipping_city,shipping_zip,order_date,priority,pick_type,component_sku,bundle_parent,shortage_qty,notes
```

**Column Descriptions:**
- `order_id` - Shopify order identifier
- `sku` - Inventory SKU (mapped from Shopify SKU)
- `quantity` - Original order quantity
- `actualqty` - Actual quantity to pick (after bundle breakdown)
- `Total Pick Weight` - Total weight for shipping calculations
- `fulfillment_center` - Assigned warehouse
- `pick_type` - Product category for warehouse operations
- `component_sku` - Individual component (for bundles)
- `bundle_parent` - Original bundle SKU (if applicable)
- `shortage_qty` - Unfulfillable quantity due to inventory constraints

**File Uses:**
- Direct import to ColdCart warehouse management system
- Pick list generation for warehouse staff
- Shipping label creation and weight calculations
- Inventory tracking and shortage reporting

---

## 🔧 SKU Mapping System

### **📊 Core Architecture**

The SKU mapping system translates Shopify order SKUs into warehouse-specific inventory SKUs and handles complex bundle breakdowns.

#### **Data Sources & Structure**

**Google Sheets Integration:**
- **Master Sheet**: [SKU Mappings Spreadsheet](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY)
- **Warehouse-Specific Sheets**: Each warehouse maintains separate mappings
- **Real-time Updates**: Changes sync immediately when mappings reloaded

**Google Sheets Column Structure:**
The system uses fixed column positions to extract bundle and SKU mapping data. Based on current implementation:

```
Column 0:  shopifysku2 (Shopify SKU identifier)
Column 1:  picklist sku (Inventory SKU for warehouse picking)
Column 2:  Mix Quantity (Component quantity for bundles)
Column 5:  Product Type (Product category classification)  
Column 6:  Pick Type (Warehouse pick category)
Column 8:  Pick Type Inventory (Legacy field, not used)
Column 10: Pick Weight LB (Weight per component in pounds)
Column 12: Total Pick Weight (Total weight calculation)
```

**Important Notes:**
- Columns 3, 4, 7, 9, 11, 13+ contain supporting data and calculations
- Bundle identification: Multiple rows with same Column 0 (shopifysku2) value
- Single SKU identification: One row per unique Column 0 (shopifysku2) value
- Weight data extracted from Column 10 for component weights, Column 12 for totals

### **🔄 SKU Types & Processing Logic**

#### **1. Single SKUs (Direct Mapping)**
**Characteristics:**
- **Format**: Most SKUs starting with `f.` (e.g., `f.apricot-2lb`)
- **Processing**: Direct 1:1 mapping from Shopify SKU → Inventory SKU
- **Google Sheets**: Single row per SKU
- **Distribution**: ~2,088 `f.*` singles across all warehouses

**Example Processing:**
```
Input:  f.apricot-2lb (quantity: 3)
Lookup: f.apricot-2lb → apricot-BG01x01
Output: apricot-BG01x01 (quantity: 3, weight: 6.0 lbs)
```

#### **2. Bundle SKUs - f.* Bundles (Multi-Component)**
**Characteristics:**
- **Format**: SKUs starting with `f.` that have multiple components
- **Identification**: Multiple rows in Google Sheets with same Shopify SKU
- **Processing**: One Shopify SKU breaks down into multiple inventory components
- **Scaling**: Component quantities multiply by order quantity
- **Distribution**: ~104 `f.*` bundles across all warehouses

**Example Processing:**
```
Input: f.stone-fruit_variety-2lb_3 (quantity: 1)

Bundle Breakdown (actual system data):
├── peach-20x50: qty 2.0, weight 1.0 lb
├── nectarine-20x50: qty 2.0, weight 1.0 lb  
└── plum_black-22x64: qty 2.0, weight 1.0 lb

Output:
- Order Line 1: peach-20x50 (qty: 2.0, weight: 1.0)
- Order Line 2: nectarine-20x50 (qty: 2.0, weight: 1.0)
- Order Line 3: plum_black-22x64 (qty: 2.0, weight: 1.0)
```

#### **3. Bundle SKUs - m.* Bundles (Multi-Component)**
**Characteristics:**
- **Format**: SKUs starting with `m.` (e.g., `m.exoticfruit-3lb-bab`)
- **Processing**: Break down into inventory component SKUs (not direct mapping)
- **Key-Based Detection**: System checks bundle mappings by SKU key, not prefix
- **Distribution**: ~164 `m.*` bundles, ~8 `m.*` singles across all warehouses

**Example Processing:**
```
Input: m.exoticfruit-3lb-bab (quantity: 1)

Bundle Breakdown (Wheeling warehouse):
├── lychee-BG0102: qty 1.0, weight 0.5 lb
├── mango_cherry-09x08: qty 1.0, weight 1.13 lb
├── guava_pink-10x33: qty 1.0, weight 0.33 lb
└── avocado_reed-01x01: qty 1.0, weight 1.0 lb

Output:
- Order Line 1: lychee-BG0102 (qty: 1.0, weight: 0.5)
- Order Line 2: mango_cherry-09x08 (qty: 1.0, weight: 1.13)
- Order Line 3: guava_pink-10x33 (qty: 1.0, weight: 0.33)
- Order Line 4: avocado_reed-01x01 (qty: 1.0, weight: 1.0)
```

#### **4. Other SKU Types**
**Additional SKU prefixes found in system:**
- **w.***: ~100 singles (warehouse-specific SKUs)
- **F.***: ~4 singles (uppercase variant)
- **b.***: Bundle SKUs visible in sheets but not processed (legacy/inactive)

**Bundle Detection Logic:**
- System uses **key-based detection**: Checks if SKU exists in bundle mappings
- **Not prefix-based**: A `f.` SKU can be either single or bundle depending on mapping data
- **Warehouse-specific**: Same SKU can have different component configurations per warehouse

### **⚙️ Processing Workflow**

#### **Step 1: Order Ingestion**
```
Shopify Order: 
- Order #DOC_TEST
- SKU: m.exoticfruit-3lb-bab
- Quantity: 1
- Fulfillment Center: IL-Wheeling-60090
```

#### **Step 2: Warehouse Assignment**
```
Fulfillment Center Analysis:
IL-Wheeling-60090 → Wheeling fulfillment center
Load: INPUT_bundles_cvr_wheeling sheet
```

#### **Step 3: Bundle Detection**
```
Key-Based Lookup: m.exoticfruit-3lb-bab in Wheeling bundle mappings
Found: 4 components (Bundle detected)
Components loaded with quantities and weights from Google Sheets
```

#### **Step 4: Bundle Expansion**
```
Original: m.exoticfruit-3lb-bab × 1 bundle
Expansion (actual system output):
├── Component 1: lychee-BG0102
│   ├── Base quantity: 1.0 units
│   ├── Base weight: 0.5 lbs
│   └── Final: 1.0 units, 0.5 lbs
├── Component 2: mango_cherry-09x08
│   ├── Base quantity: 1.0 units
│   ├── Base weight: 1.13 lbs
│   └── Final: 1.0 units, 1.13 lbs
├── Component 3: guava_pink-10x33
│   ├── Base quantity: 1.0 units
│   ├── Base weight: 0.33 lbs
│   └── Final: 1.0 units, 0.33 lbs
└── Component 4: avocado_reed-01x01
    ├── Base quantity: 1.0 units
    ├── Base weight: 1.0 lbs
    └── Final: 1.0 units, 1.0 lbs
```

#### **Step 5: Inventory Allocation**
```
For each component:
1. Check current inventory balance
2. Allocate required quantity
3. Update running balance
4. Flag shortages if insufficient
5. Generate pick list entry
```

### **🏭 Warehouse-Specific Mapping**

#### **Route Determination Process:**
1. **ZIP Code Analysis**: Customer ZIP → Geographic region
2. **Warehouse Selection**: Assign to nearest fulfillment center
3. **SKU Mapping Application**: Load warehouse-specific mappings
4. **Component Resolution**: Use warehouse-specific inventory SKUs

#### **Multi-Warehouse Example (Actual System Data):**
```
Same Bundle, Different Component Configurations:

m.exoticfruit-3lb-bab:

Wheeling (4 components, 2.96 lbs total):
├── lychee-BG0102: 1.0 qty, 0.5 lbs
├── mango_cherry-09x08: 1.0 qty, 1.13 lbs
├── guava_pink-10x33: 1.0 qty, 0.33 lbs
└── avocado_reed-01x01: 1.0 qty, 1.0 lbs

Oxnard (5 components, 2.96 lbs total):
├── lychee-BG0102: 1.0 qty, 0.5 lbs
├── mango_cherry-09x08: 1.0 qty, 1.13 lbs
├── blood_orange-01x04: 1.0 qty, 0.25 lbs
├── avocado_reed-01x01: 1.0 qty, 1.0 lbs
└── pf_purple-01x12: 1.0 qty, 0.08 lbs

Note: Same total weight, different fruit mix per warehouse
```

### **📈 Advanced Features**

#### **Weight Calculations**
- **Per Component**: Individual pick weights tracked for each component
- **Total Weight**: Aggregated for shipping calculations and label generation
- **Scaling Logic**: Weights multiply proportionally with order quantities
- **Precision**: Maintained to 2 decimal places for accuracy

#### **Pick Type Classification**
- **Product Categories**: "Fruit: Apple", "Vegetable: Tomato", "Nuts: Almonds"
- **Inventory Types**: Additional classification for warehouse operations
- **Bundle Inheritance**: Each component maintains its own classification
- **Warehouse Operations**: Used for pick routing and storage optimization

#### **Error Handling & Validation**
- **Missing Mappings**: Flagged as mapping errors with detailed logging
- **Component Shortages**: Tracked separately per individual component
- **Invalid SKUs**: Logged for manual review and correction
- **Data Consistency**: Validation checks ensure mapping integrity

### **🔄 Live Updates & Cache Management**

#### **Google Sheets → System Sync:**
1. **Edit Google Sheet**: Update component quantities or add new bundles
2. **🔄 Reload Mappings**: Click reload button in application interface
3. **Auto-Sync**: Changes applied immediately to new order processing
4. **Staging Protection**: Staged orders maintain their original mappings

#### **Cache Management Strategy:**
- **Memory Caching**: Mappings cached in memory for optimal performance
- **Manual Reload**: Required after Google Sheets edits
- **Background Sync**: Planned for future versions with automatic detection
- **Version Control**: Mapping changes tracked with timestamps

### **🐛 Advanced Troubleshooting**

#### **Bundle Processing Issues**
- ✅ **Bundle Not Breaking Down**: 
  - Verify multiple rows exist in Google Sheet for same Shopify SKU
  - Check all component SKUs are valid inventory items
  - Ensure Pick List SKU column is populated for each component
  - Reload mappings after Google Sheet edits

- ✅ **Component Shortages**:
  - Individual components tracked separately in shortage reports
  - Partial fulfillment possible when some components available
  - Shortage reports show specific component shortfalls with quantities
  - Can stage partial orders to protect available components

- ✅ **Weight Calculation Errors**:
  - Verify Pick Weight LB and Total Pick Weight columns populated
  - Check for numeric formatting issues in Google Sheets
  - Ensure weight scaling matches quantity scaling

#### **Mapping Validation**
- ✅ **SKU Not Found**: 
  - Check SKU exists in correct warehouse sheet
  - Verify exact spelling and formatting
  - Confirm warehouse assignment matches customer ZIP code region

- ✅ **Inventory Mismatch**:
  - Ensure picklist SKU exists in current inventory
  - Check for case sensitivity in SKU matching
  - Verify warehouse codes match between mappings and inventory

---

## 🔄 Smart Workflow

### **Complete Processing Workflow**

1. **📤 Upload Phase**
   - Upload orders CSV + inventory data
   - Automatic validation and format checking
   - Error reporting for malformed data

2. **⚡ Auto-Processing**
   - ZIP code analysis for warehouse routing
   - SKU mapping and bundle breakdown
   - Inventory allocation and shortage detection
   - Initial fulfillment plan generation

3. **📊 Review & Analysis**
   - Review results in Orders tab
   - Identify orders with issues or shortages
   - Analyze warehouse capacity and distribution
   - Check bundle breakdowns for accuracy

4. **🎯 Strategic Staging**
   - Stage high-priority orders to protect their inventory
   - Consider customer importance and order value
   - Protect inventory for time-sensitive shipments
   - Reserve components for complex bundles

5. **🔧 Mapping Optimization**
   - Edit bundle mappings in Google Sheet if needed
   - Adjust component quantities or weights
   - Add new product mappings
   - Update seasonal availability

6. **🔄 Recalculation**
   - Recalculate remaining orders with updated mappings
   - Available inventory = Initial - Staged allocations
   - Apply new mapping rules to pending orders
   - Generate updated shortage reports

7. **♻️ Iterative Refinement**
   - Repeat review and staging process
   - Fine-tune until satisfied with results
   - Export final fulfillment lists
   - Import to warehouse management system

---

## ⚠️ Key Actions

### **🎯 Move to Staging**
**Purpose**: Protects inventory allocation for specific orders
**Process**: 
- Selected orders reserve their required inventory
- Inventory remains allocated during recalculations
- Staged orders appear in dedicated Staging tab
- Can unstage if priorities change

**Best Practices**:
- Stage high-value orders first
- Prioritize time-sensitive shipments
- Consider customer tier and history
- Balance warehouse capacity across regions

### **🔄 Recalculate Orders**
**Purpose**: Reprocess remaining orders with updated inventory
**Process**:
- Uses available inventory (Initial - Staged allocations)
- Applies current SKU mappings from Google Sheets
- Recalculates shortages and allocations
- Updates fulfillment recommendations

**Triggers**:
- After editing SKU mappings
- When staging/unstaging orders
- After refreshing inventory data
- When resolving mapping issues

### **📝 Edit Mappings**
**Purpose**: Modify bundle components and quantities via Google Sheet
**Process**:
- Direct editing in warehouse-specific sheets
- Support for adding new SKUs and bundles
- Component quantity and weight adjustments
- Real-time validation and error checking

**Impact**:
- Only affects new calculations (staged orders protected)
- Requires manual mapping reload in system
- Changes apply immediately to subsequent processing
- Maintains audit trail of modifications

### **🔄 Refresh Inventory**
**Purpose**: Get latest inventory data from ColdCart API
**Process**:
- Real-time sync with warehouse management system
- Updates available quantities for all SKUs
- Maintains inventory accuracy for allocation
- Automatic fallback to manual upload if API fails

**Frequency**:
- Manual refresh on demand
- Automatic updates every 15 minutes (when API enabled)
- Webhook notifications for critical stock changes
- Emergency refresh for urgent orders

---

## 🔧 Troubleshooting

### **📤 Import Issues**

#### **No Processing After Upload**
**Symptoms**: Files uploaded but no order processing occurs
**Diagnosis**:
- ✅ Verify BOTH Orders AND Inventory files uploaded
- ✅ Check file size limits (< 50MB per file)
- ✅ Ensure CSV format with proper encoding (UTF-8)
- ✅ Validate required columns present in both files

**Resolution**:
1. Re-upload missing files
2. Check browser console for JavaScript errors
3. Try smaller file chunks if size exceeds limits
4. Verify network connectivity and session status

#### **CSV Format Errors**
**Symptoms**: Import fails with format validation errors
**Common Issues**:
- ✅ Extra quotes or special characters in data
- ✅ Inconsistent delimiter usage (commas vs semicolons)
- ✅ Missing required columns or headers
- ✅ Encoding issues (non-UTF-8 characters)

**Resolution**:
1. Export fresh CSV from Shopify with standard settings
2. Open in text editor to check for formatting issues
3. Ensure proper comma-separated format throughout
4. Remove any manually added quotes or special characters

#### **Missing Critical Columns**
**Required for Orders**: Order ID, SKU, Quantity, Customer ZIP
**Required for Inventory**: SKU, Available Quantity, Warehouse

**Resolution**:
1. Verify column headers match expected names
2. Check for alternative column naming (case sensitivity)
3. Ensure no missing headers in first row
4. Validate data completeness (no empty required cells)

#### **ColdCart API Connection Failures**
**Symptoms**: API toggle available but connection fails
**Diagnosis**:
- ✅ Verify API token configured and valid
- ✅ Check ColdCart service status
- ✅ Test network connectivity to API endpoints
- ✅ Review API rate limits and usage

**Resolution**:
1. Update API token in system settings
2. Switch to File Upload mode as temporary workaround
3. Contact ColdCart support for service issues
4. Check firewall/proxy settings blocking API calls

### **🎁 SKU Bundle Problems**

#### **Bundles Not Breaking Down**
**Symptoms**: Bundle SKUs appear as single items instead of components
**Diagnosis**:
- ✅ Check Google Sheet has multiple rows for bundle SKU
- ✅ Verify all component rows have same Shopify SKU in Column 0
- ✅ Ensure Picklist SKU column populated for each component
- ✅ Confirm mappings loaded after recent Google Sheet edits

**Resolution**:
1. Edit Google Sheet to add missing component rows
2. Verify exact SKU matching (case sensitive)
3. Click 🔄 Reload Mappings button after sheet edits
4. Check warehouse-specific sheet selection

#### **Missing Bundle Components**
**Symptoms**: Some bundle components missing from breakdown
**Diagnosis**:
- ✅ Verify all bundle components exist in SKU mappings
- ✅ Check component SKUs exist in current inventory
- ✅ Ensure warehouse-specific mappings include all components
- ✅ Validate component quantities and weights specified

**Resolution**:
1. Add missing components to Google Sheet mappings
2. Update inventory data to include component SKUs
3. Verify component quantities are numeric values
4. Check Pick Weight LB and Total Pick Weight columns

#### **Wrong Warehouse Assignments**
**Symptoms**: Orders routed to incorrect fulfillment centers
**Diagnosis**:
- ✅ Verify customer ZIP codes complete and valid (5-digit format)
- ✅ Check ZIP code to warehouse mapping rules
- ✅ Ensure bundle components mapped to correct warehouses
- ✅ Validate warehouse-specific Google Sheet selection

**Resolution**:
1. Update customer ZIP codes to complete 5-digit format
2. Review ZIP code routing logic for edge cases
3. Edit warehouse-specific mappings for consistency
4. Test with known ZIP codes for verification

### **📊 Data Quality Checks**

#### **High Shortage Percentages**
**Symptoms**: Many orders showing inventory shortages
**Investigation**:
- ✅ Verify inventory data current and accurate
- ✅ Check for duplicate inventory entries
- ✅ Ensure proper SKU matching between orders and inventory
- ✅ Review bundle component availability separately

**Resolution**:
1. Refresh inventory data from ColdCart API
2. Audit inventory file for data quality issues
3. Check for case sensitivity in SKU matching
4. Separate shortage analysis by individual components

#### **Incorrect Weight Calculations**
**Symptoms**: Pick weights don't match expected values
**Diagnosis**:
- ✅ Check Pick Weight LB column in Google Sheets
- ✅ Verify Total Pick Weight calculations
- ✅ Ensure weight scaling matches quantity scaling
- ✅ Validate numeric formatting in weight columns

**Resolution**:
1. Update weight values in Google Sheet mappings
2. Verify calculation formulas in spreadsheet
3. Check for text formatting in numeric columns
4. Reload mappings after weight corrections

#### **Bundle Validation Errors**
**Symptoms**: Bundle SKUs flagged with validation errors
**Requirements**:
- ✅ Bundle SKUs should have recognizable prefixes (f.*, m.*)
- ✅ All components must exist in inventory
- ✅ Component quantities should be positive numbers
- ✅ Warehouse assignments must be consistent

**Resolution**:
1. Standardize bundle SKU naming conventions
2. Validate all component inventory availability
3. Audit component quantity and weight data
4. Ensure consistent warehouse mappings across components

### **🚨 Emergency Procedures**

#### **System Performance Issues**
**High Volume Processing**:
- Process orders in smaller batches (< 1000 orders)
- Use staging to prioritize critical orders
- Consider off-peak processing times
- Monitor system resource usage

#### **Critical Shortage Situations**
**Immediate Actions**:
1. Stage highest priority orders first
2. Contact suppliers for emergency inventory
3. Consider partial fulfillment options
4. Communicate delays to affected customers

#### **Data Corruption Recovery**
**Backup Procedures**:
- Export current order processing state
- Save staging configuration
- Backup Google Sheets mapping data
- Document any manual overrides applied

---

## 📞 Support & Resources

### **System Administration**
- **Production Environment**: Monitor system health and performance
- **API Management**: Maintain ColdCart integration and tokens
- **User Access**: Manage permissions and user accounts
- **Data Backup**: Regular backup of critical configuration data

### **Business Operations**
- **Inventory Management**: Coordinate with warehouse teams
- **Customer Service**: Handle fulfillment inquiries and issues
- **Vendor Relations**: Manage supplier relationships and emergency inventory
- **Quality Assurance**: Monitor accuracy and customer satisfaction

### **Technical Support**
- **Google Sheets**: Training on mapping management and validation
- **Troubleshooting**: Systematic approach to common issues
- **Process Documentation**: Maintain updated procedures and workflows
- **System Updates**: Coordinate feature releases and improvements

