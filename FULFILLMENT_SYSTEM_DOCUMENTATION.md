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

**What the System Does:**
1. **SKU Matching**: Matches inventory SKUs with order SKUs
2. **Warehouse Assignment**: Routes orders to appropriate fulfillment centers
3. **Inventory Tracking**: Tracks available vs. allocated inventory
4. **Shortage Detection**: Identifies when inventory is insufficient

#### **ColdCart API Integration**
**Real-time Inventory:**
- Automatic inventory updates
- Use when you need current stock levels
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



### **🔄 Updating Mappings**

**How to Update SKU Mappings:**
1. **Edit Google Sheet**: Update component quantities or add new bundles
2. **🔄 Reload Mappings**: Click reload button in application interface
3. **Changes Apply**: Updates apply immediately to new order processing
4. **Staging Protection**: Staged orders keep their original mappings

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

**When to Use**:
- Manual refresh when needed
- Automatic updates when API enabled
- Use before processing large order batches

---

## 🔧 Troubleshooting

### **📤 File Upload Problems**

#### **Nothing Happens After Upload**
**What to Check**:
- ✅ Upload BOTH Orders AND Inventory files
- ✅ Files are under 50MB each
- ✅ Files are in CSV format

**How to Fix**:
1. Make sure both files are uploaded
2. Try smaller files if too large
3. Refresh page and try again

#### **CSV File Errors**
**Common Problems**:
- Extra quotes or special characters
- Missing required columns
- Wrong file format

**How to Fix**:
1. Use standard Shopify CSV export
2. Don't edit the CSV file manually
3. Make sure all required columns are present:
   - **Orders**: Order ID, SKU, Quantity, Customer ZIP
   - **Inventory**: SKU, Available Quantity, Warehouse

#### **API Not Working**
**How to Fix**:
1. Switch to "Upload File" mode instead
2. Contact support if API is needed

### **🎁 Bundle Problems**

#### **Bundle Not Breaking Into Parts**
**Problem**: Bundle SKU shows as one item instead of separate components

**How to Fix**:
1. Check Google Sheet has multiple rows for the same bundle SKU
2. Make sure all components have inventory SKUs filled in
3. Click **🔄 Reload Mappings** button
4. Check you're using the right warehouse sheet

#### **Missing Parts in Bundle**
**Problem**: Some bundle components are missing

**How to Fix**:
1. Add missing components to Google Sheet
2. Make sure component SKUs exist in your inventory
3. Check all quantities are numbers (not text)
4. Fill in weight columns

#### **Orders Going to Wrong Warehouse**
**Problem**: Orders assigned to incorrect fulfillment center

**How to Fix**:
1. Check customer ZIP codes are complete (5 digits)
2. Verify warehouse assignments in Google Sheets
3. Test with known ZIP codes

### **📊 Common Issues**

#### **Too Many Shortages**
**Problem**: Most orders show inventory shortages

**How to Fix**:
1. Refresh inventory data (use **🔄 Refresh Inventory**)
2. Check if inventory file has the right SKUs
3. Make sure SKU names match exactly (case sensitive)

#### **Wrong Weights**
**Problem**: Pick weights look incorrect

**How to Fix**:
1. Check weight columns in Google Sheets
2. Make sure weights are numbers (not text)
3. Click **🔄 Reload Mappings** after changes

#### **Bundle Errors**
**Problem**: Bundle SKUs flagged with errors

**How to Fix**:
1. Make sure all bundle components exist in inventory
2. Check component quantities are positive numbers
3. Verify all components mapped to same warehouse

### **🚨 When Things Go Wrong**

#### **System Running Slow**
**What to Do**:
- Process fewer orders at a time (under 1000)
- Use staging for urgent orders first
- Try during off-peak hours

#### **Critical Inventory Shortages**
**What to Do**:
1. Stage your most important orders first
2. Contact suppliers for emergency stock
3. Consider partial fulfillment
4. Notify customers of delays

#### **Need to Start Over**
**What to Do**:
- Export your current work before making changes
- Save your staging setup
- Document any manual changes you made

---

## 📞 Support

**For Help With:**
- **SKU Mapping Issues**: Check Google Sheets setup and reload mappings
- **Order Processing**: Verify file formats and required columns
- **Bundle Problems**: Ensure all components exist in mappings and inventory
- **System Issues**: Contact technical support team

