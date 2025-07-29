# DataProcessor Tests

This directory contains unit tests for the DataProcessor class that verify all data entry points and outputs using real data from the actual system.

## Directory Structure

```
tests/
├── README.md                 # This file
├── test_data_processor.py    # Main test file
└── data/
    └── sku_mappings.json     # Real SKU mappings from production
```

## Test Overview

The `test_data_processor.py` file contains comprehensive tests that verify:

### 📥 **Data Entry Points:**
1. **Orders CSV Loading** - Tests the `load_orders()` method
2. **Inventory CSV Loading** - Tests the `load_inventory()` method  
3. **SKU Mappings JSON Loading** - Tests SKU mappings from `data/sku_mappings.json`

### 🔄 **Data Processing:**
- SKU mapping (Shopify SKUs → Warehouse SKUs)
- Bundle expansion (Bundle SKUs → Component SKUs)
- Inventory allocation and balance tracking
- Shortage detection and reporting

### 📤 **Data Outputs:**
1. `orders.csv` - Processed orders with inventory allocation
2. `shortage_summary.csv` - Detailed list of inventory shortages
3. `grouped_shortage_summary.csv` - Shortages grouped by warehouse/SKU
4. `initial_inventory.csv` - Snapshot of inventory before processing
5. `inventory_comparison.csv` - Before/after inventory changes

## How to Run Tests

### From the tests directory:
```bash
cd tests
python test_data_processor.py
```

### From the project root:
```bash
python -m tests.test_data_processor
```

## Test Data

The tests use **real data** from the actual system:

- **Real SKU names** from actual order files
- **Real starting balances** from actual inventory files  
- **Real SKU mappings** from production `sku_mappings.json`
- **Real bundle configurations** with actual component relationships

### Example Test SKUs:
- `f.avocado_reed-2lb` → `avocado_reed-01x01`
- `f.guava_pink-2lb` → `guava_pink-10x15` 
- `f.asian_pear-2lb` → `pear_asian-12x18`
- `m.exoticfruit-5lb` → 9 component SKUs (bundle)
- `m.farmbox-3lb` → 6 component SKUs (bundle)

## Expected Results

When running the test, you should see:

```
✅ CORRECTED TEST COMPLETED SUCCESSFULLY!
✅ All data entry points and outputs verified with REAL data
✅ SKU mappings verified
✅ Starting balances verified
✅ Real order processing confirmed
```

The test verifies:
- ✅ 1010 single SKU mappings + 132 bundle mappings loaded
- ✅ 5 test orders → 15 output rows (due to bundle expansion)
- ✅ Proper shortage detection when inventory is insufficient
- ✅ Correct starting balance tracking
- ✅ All output files generated with proper structure

## Troubleshooting

### "sku_mappings.json not found" Error:
- Ensure you're running from the `tests/` directory, OR
- Ensure `tests/data/sku_mappings.json` exists

### Import Errors:
- Ensure the parent directory is in the Python path
- Run from the correct directory as shown above

### Missing Dependencies:
```bash
pip install pandas
```

## Adding New Tests

To add new test cases:

1. Add new test SKUs to `create_real_test_orders()`
2. Add corresponding inventory to `create_real_test_inventory()`  
3. Verify the SKU exists in `data/sku_mappings.json`
4. Run the test to verify it works

The test framework automatically validates that all SKUs exist in the real mappings before processing. 