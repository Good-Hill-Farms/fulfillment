# PLAN_V1.md - Fulfillment System Enhancement

## Overview
This plan outlines the development of an enhanced order fulfillment system with programmatic order management, condition-based routing, and AI-powered decision support.

## Current State
- **AI-powered fruit order fulfillment system** with Streamlit UI
- **Multi-warehouse inventory tracking** (CA-Moorpark, CA-Oxnard, IL-Wheeling)
- **Real-time order processing** with SKU mapping and bundle handling
- **Geographic routing** based on ZIP codes and shipping zones
- **Interactive dashboard** with AgGrid tables and Plotly visualizations

⚠️ **Important Note:** This system is used for real operational decisions — no default values should be used. Every input must be explicitly defined. Any missing or placeholder values may lead to incorrect fulfillment logic.

## 🧠 Matching Logic (WIP/Planned)

### 1. Normalize Warehouse Names Since 

```python
if "moorpark" in warehouse.lower() or "oxnard" in warehouse.lower():
    warehouse = 'Oxnard'
elif "wheeling" in warehouse.lower():
    warehouse = 'Wheeling'
```

## Implementation Phases

### **STEP 1: Tool Post Upload** 
*Priority: P1 | Timeline: 1-2 days*

**Objective**: Enhance CSV upload capabilities with validation and preprocessing tools

**Tasks**:
- Add file validation (format, required columns, data types)
- Implement data cleaning and normalization tools
- Create upload history and version tracking
- Add bulk upload support for multiple file types
- Implement rollback functionality for failed uploads

**Deliverables**:
- Enhanced upload UI with drag-and-drop
- Data validation pipeline
- Upload audit trail

### **STEP 2: Condition States Management**
*Priority: P1 | Timeline: 2-3 days*

**Objective**: Implement programmatic order filtering and condition-based routing

**Core Features**:
- **Condition Builder**: Visual interface for creating order filters
- **State Management**: Track order statuses (pending, processing, shipped, etc.)
- **Rule Engine**: Apply business logic based on order attributes

**Implementation**:
```python
# Example condition syntax
conditions = {
    "priority": "P1",
    "destination_state": "CA", 
    "order_value": ">100",
    "sku_category": "exotic_fruits",
    "shipping_zone": [1, 2, 3]
}
```

**Tasks**:
- Build condition parser and validator
- Create order state tracking system
- Implement rule-based routing engine
- Add condition templates for common scenarios

### **STEP 3: Programmatic Order Management**
*Priority: P1 | Timeline: 3-4 days*

**Objective**: Orders that match XYZ conditions → push them automatically

**Key Features**:
- **Automated Order Processing**: Match conditions → trigger actions
- **Batch Operations**: Process multiple orders simultaneously
- **California Orders Pipeline**: Special handling for CA destinations
- **VBox Integration**: Enhanced packaging optimization

**Implementation Flow**:
1. **Condition Matching**: `if order.matches(conditions)`
2. **Action Triggers**: `execute_action(order, action_type)`
3. **Batch Processing**: `process_batch(filtered_orders)`
4. **State Updates**: `update_order_status(order_id, new_status)`

**Tasks**:
- Create condition matching engine
- Implement automated action triggers
- Build batch processing pipeline
- Add California-specific routing logic
- Integrate VBox optimization for non-CA orders

### **STEP 4: Stats and Issues Dashboard**
*Priority: Medium | Timeline: 2-3 days*

**Objective**: Real-time visibility into order processing and system health

**Dashboard Components**:
- **Order Flow Metrics**: Processing rates, completion times
- **Inventory Alerts**: Low stock warnings, shortage predictions
- **System Health**: Error rates, performance metrics
- **Geographic Distribution**: Order volume by region
- **Condition Performance**: Rule effectiveness analytics

**Features**:
- Real-time data updates
- Customizable alert thresholds
- Historical trend analysis
- Export capabilities for reports

### **STEP 5: LLM Layer for Natural Language Interface**
*Priority: Medium | Timeline: 2-3 days*

**Objective**: AI-powered conversational interface for order management

**Capabilities**:
- **Natural Language Queries**: "Show me all P1 orders for California"
- **Intelligent Suggestions**: Recommend optimal fulfillment strategies
- **Dynamic Condition Creation**: Convert text to programmatic conditions
- **Problem Resolution**: Suggest solutions for common issues

**Integration Points**:
- Extend existing chat assistant (`pages/chat_assistant.py`)
- Connect to order processing pipeline
- Integrate with condition management system
- Add voice-to-text capabilities

## Data Schema Analysis

### **Key Data Files and Fields**

#### **1. Orders CSV** (`docs/orders.csv`)
**Core Order Fields:**
- `order id` - Unique identifier (e.g., 6944151732594)
- `Name` - Order reference (e.g., #71184)
- `Customer: First Name`, `Customer: Last Name`, `Email` - Customer info
- `Shipping: Name`, `Shipping: Address 1`, `Shipping: City`, `Shipping: Province Code`, `Shipping: Zip` - Delivery details
- `SKU Helper` - Product SKU (e.g., f.loquat-5lb)
- `Line: Fulfillable Quantity` - Order quantity
- `NEW Tags` - Priority and routing tags (P1, P*, P2, IL-Wheeling-60090, etc.)
- `Fulfillment Center` - Assigned warehouse
- `Saturday Shipping` - Special delivery flag

**Critical Tags for Conditions:**
- `P1` - Priority 1 orders
- `P*` - High priority orders
- `P2` - Priority 2 orders
- `gift order` - Gift processing
- `subscription` - Recurring orders
- `1day_ship` - Express shipping
- `AfterSell Upsell` - Post-sale additions

#### **2. Inventory CSV** (`docs/inventory.csv`)
**Warehouse Inventory Fields:**
- `WarehouseName` - Location (CA-Moorpark-93021, IL-Wheeling-60090)
- `ItemId` - Internal item identifier
- `Sku` - Product SKU
- `Name` - Product description
- `Type` - Category (SellableIndividual, Packaging, etc.)
- `AvailableQty` - Current stock level
- `Balance` - Available balance for fulfillment

#### **3. Output Fulfillment CSV** (`docs/output_list_of_orders_to_fulfill.csv`)
**Fulfillment Processing Fields:**
- `externalorderid` - Order reference (e.g., #71184)
- `ordernumber` - System order number
- `CustomerFirstName`, `customerLastname`, `customeremail` - Customer details
- `shiptoname`, `shiptostreet1`, `shiptocity`, `shiptostate`, `shiptopostalcode` - Shipping address
- `placeddate` - Order date
- `totalorderamount` - Order value
- `shopsku` - Shopify SKU
- `shopquantity` - Ordered quantity
- `Tags` - Processing tags (P1, IL-Wheeling-60090, etc.)
- `Fulfillment Center` - Assigned warehouse
- `sku` - Warehouse SKU mapping
- `actualqty` - Actual quantity to fulfill
- `Starting Balance`, `Transaction Quantity`, `Ending Balance` - Inventory tracking
- `Issues` - Processing problems (e.g., "Insufficient inventory")

### **Condition-Based Field Mapping**

**Geographic Conditions:**
- `shiptostate` → State-based routing (CA orders special handling)
- `shiptopostalcode` → ZIP-based fulfillment center assignment
- `Fulfillment Center` → Warehouse selection logic

**Priority Conditions:**
- `Tags` containing "P1" → High priority processing
- `Tags` containing "P*" → Urgent processing  
- `Tags` containing "1day_ship" → Express shipping
- `Tags` containing "gift order" → Special packaging

**Inventory Conditions:**
- `Starting Balance` vs `Transaction Quantity` → Shortage detection
- `Issues` field → Problem identification
- `AvailableQty` → Stock availability checks

## Technical Architecture

### **Data Flow**
```
CSV Upload → Validation → Condition Matching → Action Triggers → Fulfillment → Tracking
     ↓              ↓              ↓              ↓              ↓
Stats Dashboard ← Order States ← Rule Engine ← Batch Processor ← LLM Interface
```

### **Core Components**

1. **Condition Engine** (`utils/condition_engine.py`)
   - Parse and validate conditions
   - Match orders against criteria
   - Execute programmatic actions

2. **State Manager** (`utils/state_manager.py`)
   - Track order lifecycle
   - Manage status transitions
   - Audit trail maintenance

3. **Batch Processor** (`utils/batch_processor.py`)
   - Queue management
   - Parallel processing
   - Error handling and retry logic

4. **Analytics Engine** (`utils/analytics_engine.py`)
   - Real-time metrics calculation
   - Trend analysis
   - Performance monitoring

### **Database Schema Extensions**
```sql
-- Order States Table
CREATE TABLE order_states (
    order_id VARCHAR,
    status VARCHAR,
    timestamp DATETIME,
    conditions_met JSON,
    action_taken VARCHAR
);

-- Conditions Table  
CREATE TABLE order_conditions (
    condition_id VARCHAR,
    name VARCHAR,
    criteria JSON,
    actions JSON,
    active BOOLEAN
);

-- Processing Queue
CREATE TABLE processing_queue (
    queue_id VARCHAR,
    order_ids JSON,
    batch_type VARCHAR,
    status VARCHAR,
    created_at DATETIME
);
```

## Implementation Timeline

| Week | Focus | Deliverables |
|------|-------|-------------|
| 1 | Tool Post Upload + Condition States | Enhanced upload UI, Basic condition engine |
| 2 | Programmatic Order Management | Automated processing pipeline, CA orders handling |
| 3 | Stats Dashboard + VBox Integration | Real-time analytics, Packaging optimization |
| 4 | LLM Layer + Testing | Natural language interface, System integration |

## Success Metrics

- **Processing Efficiency**: 50% reduction in manual order handling
- **Error Reduction**: 90% decrease in fulfillment errors
- **California Orders**: 100% automated processing for CA destinations
- **User Adoption**: 80% of users utilizing programmatic conditions
- **System Performance**: <2 second response times for condition matching

## Risk Mitigation

- **Data Integrity**: Comprehensive validation and rollback capabilities
- **Performance**: Async processing and caching strategies
- **User Training**: Progressive feature rollout with documentation
- **System Reliability**: Health monitoring and automated alerts

## Next Steps

1. **Immediate**: Begin STEP 1 (Tool Post Upload enhancement)
2. **Week 1**: Complete condition states management framework
3. **Week 2**: Deploy programmatic order processing for P1 orders
4. **Week 3**: Launch California orders automation pipeline
5. **Week 4**: Integrate LLM layer and complete V1 release

---

*This plan provides a roadmap for transforming the current fulfillment system into an intelligent, automated order processing platform with AI-powered decision support.*