# 🍍 AI-Powered Fulfillment Assistant – README

This project is a smart inventory + order matching dashboard built with Streamlit and Python. It helps match customer fruit orders to available warehouse inventory using:

* Uploaded CSVs
* Predefined rules (warehouse zones, shipping, SKU mappings)
* Editable dashboards with AG Grid
* Exportable results

> ⚠️ **Important Note:** This system is used for real operational decisions — no default values should be used. Every input must be explicitly defined. Any missing or placeholder values may lead to incorrect fulfillment logic.

---

## 📂 Requirements Overview
Here’s an additional section you can append to your **🍍 AI-Powered Fulfillment Assistant – README** to incorporate the flow from your diagram and explain the phased logic steps.

---

## 🔁 Fulfillment Workflow Logic

This assistant follows a multi-step operational flow to manage and push orders based on conditions, priority, and warehouse zones.

### 🪜 Fulfillment Flow Steps

| Step             | Action                                     | Description                                                          |
| ---------------- | ------------------------------------------ | -------------------------------------------------------------------- |
| Tool Post Upload | ⬆️ Upload `orders.csv` and `inventory.csv` | Data is parsed and stored with a unique batch ID for traceability.   |
| Step 1           | ✅ Push all **priority P1** orders          | Orders tagged as high-priority are processed and pushed immediately. |
| Step 2           | 🔄 Adjust `vbox` for remaining orders      | Orders not in P1 are re-evaluated and routed using fallback logic.   |
| Step 3           | 🚚 Push **California orders** separately   | Specific rule-based routing for California zip codes is applied.     |

---

## 🧠 v1 System Logic

| Module                        | Purpose                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------ |
| `condition states management` | Tracks order statuses and conditions (e.g. zip, bundle type, priority).        |
| `programmatic rules engine`   | Applies fulfillment rules for quantity, zone, fallback, etc.                   |
| `LLM layer`                   | Optional AI layer to assist operators with questions and fulfillment guidance. |

---

## 🧪 Manual + Smart Pushes

* You can selectively **push orders that match certain conditions** (e.g., zip, tags, rules).
* Optionally **show stats and highlight any matching issues** before pushing.

This flow ensures safe batch handling, priority awareness, and clarity in rule execution.

### 🔧 Files Required for the App to Function:

#### 1. `orders.csv`

* Contains customer order line items
* Required fields:

  * `order id`
  * `Shipping: Zip`
  * SKU columns (like `f.mangosteen-2lb`, etc.)
  * May contain **bundles** or **singular items**

#### 2. `inventory.csv`

* Contains all SKUs available in each warehouse
* Columns required:

  * `WarehouseName` (Normalize: treat `Moorpark` and `Oxnard` as same)
  * `Sku`
  * `AvailableQty`
  * `Oxnard` and `Wheeling` columns with picklist SKUs and balances

#### 3. `sku_mappings.json`

* Maps all SKUs in orders to picklist SKUs in inventory
* Includes:

  * `picklist_sku`
  * `actualqty`
  * `Total_Pick_Weight`
  * `Pick_Type`
  * Information about whether a SKU is part of a **bundle** or a **singular item**
  * Must support **manual updates** and **custom entries** via the dashboard

#### 4. `shipping_zones.json`

* Maps `zip_prefix` → `zone` per warehouse
* Used for warehouse prioritization logic
* Must support adding new `zip_prefix` and editing warehouse assignments

#### 5. `delivery_services.json`

* Maps destination zip prefix → shipping method and origin warehouse
* Can be used to calculate delivery days or fallback warehouse
* Should be editable to reflect carrier/service updates

---

## ✅ Features

* Upload and analyze order/inventory files
* Match orders to fulfillment center inventory
* Track shortages and produce warning CSVs
* Use distance logic (zone/zip) to prioritize fulfillment
* Recognize **bundles** vs **singular items**
* Edit and recalculate in dashboard with AgGrid
* Export final matched fulfillment plan
* ⚙️ Fully editable SKU mappings, fulfillment centers, and zone logic from the interface

---

## 🧠 Matching Logic (WIP/Planned)

### 1. Normalize Warehouse Names

```python
if "moorpark" in warehouse.lower() or "oxnard" in warehouse.lower():
    warehouse = 'Oxnard'
elif "wheeling" in warehouse.lower():
    warehouse = 'Wheeling'
```

### 2. For each order line:

* Identify if the line is a **bundle** or **singular** using SKU mapping
* Look up SKU in `sku_mappings.json`
* Determine required quantity from order
* Find available quantity in prioritized warehouses (based on shipping zone logic)
* Assign fulfillment warehouse
* Deduct quantity
* Log shortages if inventory insufficient

### 3. Export Results

* Fulfillment plan (with warehouse assignment)
* Shortage report if needed

---

## ⚙️ Future Rules & Settings

* Warehouse fallback logic
* Priority orders
* Bundle awareness and composition validation
* Partial fill or not
* SLA/delivery days calculator
* Editable interface for SKU mappings, zones, and fulfillment rules

---

## ✅ Inventory Columns To Use

From `inventory.csv`, use:

* `WarehouseName`
* `Sku`
* `AvailableQty`
* `Oxnard`, `Wheeling` SKU & quantity columns (e.g., `blood_orange-01x04`, `peach-22x64`, etc.)

---

## 📌 Next Steps

* [ ] Implement SKU matching and deduction logic
* [ ] Integrate shipping zone-based prioritization
* [ ] Add logic to detect and decompose bundles
* [ ] Improve UI with manual override tools
* [ ] Export fulfillment assignments cleanly
* [ ] Add editable UI for SKU mappings, ZIP logic, fulfillment centers
* [ ] Optional: integrate with Airtable or Google Sheets
