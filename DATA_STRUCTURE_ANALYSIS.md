# 📊 Data Structure Analysis for Fulfillment System

## 📁 Input Files Analysis

### 1. **Inventory CSV Structure** (`/docs/inventory.csv`)

**Header Structure:**
```
Row 1: ,,,,,,,,,,,,,,,,Oxnard,,,Wheeling,
Row 2: WarehouseName,ItemId,Sku,Name,Type,BatchCode,AvailableQty,DaysOnHand,Column 9,Column 10,Balance,Column 12,Column 13,Column 14,Column 15,,Sku,MIN of Balance,,Sku,MIN of Balance
```

**Main Inventory Columns (Use These):**
- `WarehouseName` - CA-Moorpark-93021, CA-Oxnard-93030, IL-Wheeling-60090
- `ItemId` - Internal item identifier
- `Sku` - Inventory SKU (e.g., df_yellow-05x07, loquat-BG01x01)
- `Name` - Product name
- `Type` - SellableIndividual, Packaging, etc.
- `BatchCode` - Batch tracking code
- `AvailableQty` - Available quantity
- `DaysOnHand` - Days of inventory
- `Balance` - **PRIMARY BALANCE COLUMN TO USE**

**Additional Columns (Don't Use):**
- Right side has Oxnard/Wheeling summary columns with "MIN of Balance"
- These are summary/calculated columns, not source data

**Sample Data:**
```
CA-Moorpark-93021,834,"1"" liner - Side A : 12x12x12""","1"" liner - Side A : 12x12x12""",Packaging,,0,,,,0.00
IL-Wheeling-60090,1064,df_yellow-05x07,"Dragonfruit, Yellow",SellableIndividual,#011324,0,,,,0.00
```

### 2. **Orders CSV Structure** (`/docs/orders.csv`)

**Key Columns for Processing:**
- `order id` - Order identifier (e.g., 6944151732594)
- `Name` - Order name (e.g., #71184)
- `Shipping: Zip` - Shipping ZIP code for warehouse assignment
- `SKU Helper` - **SHOPIFY SKU** (e.g., f.loquat-5lb, f.mango-0.5lb-gift)
- `Line: Fulfillable Quantity` - Quantity to fulfill
- `Fulfillment Center` - Assigned warehouse (IL-Wheeling-60090, etc.)

**Sample Order:**
```
Order #71184: f.loquat-5lb (qty: 1) → IL-Wheeling-60090 → ZIP: 60467
```

### 3. **Output CSV Structure** (`/docs/output_list_of_orders_to_fulfill.csv`)

**Critical Output Columns:**
- `shopifysku2` - Original Shopify SKU (f.loquat-5lb)
- `sku` - Mapped inventory SKU (loquat-BG01x01)
- `actualqty` - Quantity per unit conversion
- `Starting Balance` - Inventory before transaction
- `Transaction Quantity` - Amount being fulfilled
- `Ending Balance` - **Starting Balance - Transaction Quantity**
- `Issues` - Any fulfillment problems

**Sample Output:**
```
f.loquat-5lb → loquat-BG01x01: 1075 - 5 = 1070 (no issues)
f.mango-0.5lb-gift → mango_honey-09x16: 1312 - 1 = 1311 (no issues)
```

## 🔄 Correct Data Flow Understanding

### **ACTUAL WORKFLOW:**
1. **Upload orders.csv and inventory.csv** 
2. **System processes orders immediately** (no staging step for inventory allocation)
3. **Orders get allocated to warehouses** with inventory deductions
4. **Output shows processed orders** with allocated fulfillment info
5. **Inventory displays real current state** after order allocation

### **Processing Steps:**
1. **Order Upload** → Raw orders from customers
2. **Inventory Allocation** → System assigns warehouses and calculates inventory usage
3. **Order Output** → Shows allocated orders ready for fulfillment (like output.csv)
4. **Inventory State** → Shows current inventory after allocation

### **SKU Mapping Process:**
1. **Shopify SKU** (from orders) → **Inventory SKU** (from mappings)
2. **f.loquat-5lb** → **loquat-BG01x01** (with actualqty: 5)
3. **f.mango-0.5lb-gift** → **mango_honey-09x16** (with actualqty: 1)

### **Balance Calculation Rules:**
1. **Cannot go negative** - No restocking, only removing inventory
2. **Use main Balance column** - Not "MIN of Balance" summary columns
3. **Track per warehouse** - Each warehouse has separate inventory
4. **Real-time updates** - Starting Balance → Transaction Quantity → Ending Balance

### **Warehouse Assignment:**
- Use ZIP code → Zone mapping to assign fulfillment center
- Multiple warehouses: CA-Moorpark-93021, CA-Oxnard-93030, IL-Wheeling-60090
- Each warehouse has independent inventory levels

## 🎯 Required Inventory Displays (3 Sections)

### **1. ⚠️ INVENTORY SHORTAGES DETECTED**
**Purpose:** Show items that cannot fulfill orders completely
**Data Source:** `st.session_state.shortage_summary` and `st.session_state.grouped_shortage_summary`
**Must Show:**
- SKUs with insufficient inventory
- Requested quantity vs available quantity
- Shortage amount per SKU per warehouse
- Orders affected by shortages

### **2. Inventory Changes**  
**Purpose:** Show items whose inventory was affected by order processing
**Data Source:** Items from processed orders with before/after balances
**Must Show:**
- Only items that had inventory allocated to orders
- Starting balance → Ending balance after order allocation
- Quantity used per item
- Which warehouse the inventory came from

### **3. 📦 Complete Inventory**
**Purpose:** Show ALL inventory items with current real state after order processing
**Data Source:** `st.session_state.inventory_summary` (complete inventory with updated balances)
**Must Show ALL Items:**
- Every line from inventory CSV (all warehouses, all SKUs)
- **Current real balance** after order allocation (not original balance)
- Items used in orders: Original Balance - Allocated Quantity = Current Balance
- Items not used in orders: Original Balance = Current Balance (unchanged)
- SKU mappings where available (Shopify SKU ↔ Inventory SKU)

### **Balance Logic for Complete Inventory:**
```
For each inventory item:
  If item was allocated to orders:
    Current Balance = Starting Balance - Total Allocated Quantity
  Else:
    Current Balance = Starting Balance (no change)

NEVER allow negative balances
Display format: Current Balance (showing real state after processing)
```

### **Display Columns Needed for Complete Inventory:**
- `Warehouse` (normalized warehouse name)
- `Inventory SKU` (from inventory CSV)
- `Shopify SKU` (from mappings, if available)  
- `Current Balance` (real current balance after order allocation)
- `Is Bundle Component` (boolean flag)
- `Status` (available/low/out of stock based on current balance)

## ⚠️ Critical Issues Identified and Fixed

### **Root Cause Analysis - Data Processing Bugs:**

1. **MAJOR: Incorrect Balance Aggregation**
   - **Problem**: Using `"Balance": "max"` instead of `"Balance": "sum"` in groupby
   - **Impact**: Inventory items losing their full balance values
   - **Example**: persimmon_fuyu-01x03 losing 480 units from Oxnard warehouse
   - **Fix**: Changed to `"Balance": "sum"` for proper aggregation

2. **MAJOR: Processing Negative Adjustment Entries**
   - **Problem**: Including negative balance entries (adjustments) in calculations
   - **Impact**: Adjustment entries (-631, -1927) causing inventory corruption
   - **Fix**: Filter out negative balances before processing: `df = df[df["Balance"] >= 0]`

3. **MAJOR: Incorrect Balance Fixing Logic**
   - **Problem**: Auto-replacing Balance=0 with AvailableQty even when Balance=0 is correct
   - **Impact**: False inventory inflation and incorrect balance calculations
   - **Fix**: Only use AvailableQty when Balance is truly missing (NaN), not when it's 0

4. **MAJOR: Duplicate Entry Processing**
   - **Problem**: Creating multiple entries for same SKU-warehouse combinations
   - **Impact**: almnd_chile_limon-BG0102 appearing twice in inventory summary
   - **Fix**: Proper aggregation with sum instead of creating duplicates

### **Specific File Evidence:**
- `inventory_comparison_20250606_103959.csv`: Shows massive incorrect reductions
- `inventory_summary.csv`: Shows zeros and duplicates where shouldn't be
- `shortage_summary.csv`: Shows shortages for items that should have inventory

### **Fixed Issues:**
1. **Complete inventory now shows correct balances** after proper aggregation
2. **Uses main Balance column correctly** without negative adjustments
3. **Prevents inventory loss** during processing  
4. **Shows real current state** with accurate balance calculations
5. **Eliminates duplicate entries** in inventory summary
6. **Proper warehouse handling** with independent inventory tracking
7. **AgGrid enterprise modules ENABLED** for copy/export functionality
8. **Three distinct inventory sections** now show correct data:
   - Shortage Detection (accurate shortage calculations)
   - Inventory Changes (correct before/after balances)  
   - Complete Inventory (ALL items with accurate current balances)

## 🔧 Implementation Notes

### **Key Understanding - Orders Are PROCESSED Not Staged:**
- Orders and inventory are uploaded
- System immediately processes orders and allocates inventory  
- Output shows processed orders ready for fulfillment
- Complete Inventory shows REAL CURRENT STATE after allocation

### **Data Sources:**
- `st.session_state.inventory_summary` - Complete inventory with current balances  
- `st.session_state.shortage_summary` - Items with insufficient inventory
- `st.session_state.grouped_shortage_summary` - Aggregated shortage data
- `st.session_state.processed_orders` - Orders with allocation info

### **BREAKTHROUGH: Using Summary Columns for Clean Data**

#### **1. New Approach - Extract from Summary Columns:**
```python
# USE CLEAN SUMMARY DATA from inventory.csv:
# Oxnard: columns 16-17 (Sku, MIN of Balance)  
# Wheeling: columns 19-20 (Sku, MIN of Balance)

# Example clean data extracted:
inventory_rows.append({
    'WarehouseName': 'CA-Oxnard-93030',
    'Sku': 'blood_orange-01x04',
    'Balance': 1188.0,
    'AvailableQty': 1188.0
})
```

#### **2. Benefits of Summary Column Approach:**
- **No complex aggregation needed** - Data already summarized
- **No negative adjustment entries** - Clean MIN of Balance values  
- **No duplicate processing** - Each SKU appears once per warehouse
- **Accurate inventory balances** - Pre-calculated minimums ready to use
- **Simple, reliable processing** - Avoid complex CSV structure issues

#### **3. Data Sources Now Used:**
- **Oxnard Summary**: Column 16 (Sku) + Column 17 (MIN of Balance)
- **Wheeling Summary**: Column 19 (Sku) + Column 20 (MIN of Balance)
- **Clean mapping**: Direct SKU → Balance extraction per warehouse

### **Implementation Requirements:**
- Start with ALL rows from inventory CSV as base (after filtering negatives)
- Update balances with REAL current amounts after order allocation
- Show current state, not original balances in Complete Inventory
- Use proper warehouse normalization
- Apply SKU mappings for inventory-to-shopify linking
- Validate no negative balances during processing
- Enable AgGrid enterprise modules for copy/export functionality
- **CRITICAL**: Filter adjustment entries before processing to prevent inventory corruption