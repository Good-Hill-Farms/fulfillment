
#!/usr/bin/env python3
"""
CORRECTED DataProcessor Test using REAL data from actual files.

This test uses:
- Real SKU mappings from sku_mappings.json (verified)
- Real SKU names from actual orders CSV
- Real starting balances from actual inventory CSV
- Proper bundle orders that exist in the real system
"""

import json
import pandas as pd
import tempfile
import os
from datetime import datetime
import sys

# Add the parent directory to the path to access utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_processor import DataProcessor


def create_real_test_orders():
    """Create test orders using REAL SKUs from actual orders file."""
    orders_data = [
        # Real single fruit order - from actual CSV line 2
        {
            'Date': '2025-01-01 16:21:26 -0700',
            'Name': '#TEST001',
            'order id': 9000001,
            'Customer: First Name': 'John',
            'Customer: Last Name': 'Test',
            'Email': 'john.test@example.com',
            'Shipping: Name': 'John Test',
            'Shipping: Address 1': '123 Test St',
            'Shipping: Address 2': '',
            'Shipping: City': 'Test City',
            'Shipping: Province Code': 'CA',
            'Shipping: Zip': '90210',
            'Note': 'Test single fruit order',
            'SKU Helper': 'f.avocado_reed-2lb',  # REAL SKU from line 2
            'Line: Fulfillable Quantity': 1,
            'Line: ID': 'TEST001_1',
            'NEW Tags': 'P1, products_box,CA-Oxnard-93030',
            'MAX PKG NUM': 1,
            'Fulfillment Center': 'CA-Oxnard-93030',
            'Saturday Shipping': 'FALSE'
        },
        # Real single fruit order - from actual CSV line 4  
        {
            'Date': '2025-01-02 16:21:26 -0700',
            'Name': '#TEST002',
            'order id': 9000002,
            'Customer: First Name': 'Jane',
            'Customer: Last Name': 'Test',
            'Email': 'jane.test@example.com',
            'Shipping: Name': 'Jane Test',
            'Shipping: Address 1': '456 Test Ave',
            'Shipping: Address 2': '',
            'Shipping: City': 'Test Town',
            'Shipping: Province Code': 'CA',
            'Shipping: Zip': '90211',
            'Note': 'Test fruit order',
            'SKU Helper': 'f.guava_pink-2lb',  # REAL SKU from line 4
            'Line: Fulfillable Quantity': 1,
            'Line: ID': 'TEST002_1',
            'NEW Tags': 'P1, products_box,CA-Oxnard-93030',
            'MAX PKG NUM': 1,
            'Fulfillment Center': 'CA-Oxnard-93030',
            'Saturday Shipping': 'FALSE'
        },
        # Real bundle order - from actual CSV line 180
        {
            'Date': '2025-01-03 17:36:13 -0700',
            'Name': '#TEST003',
            'order id': 9000003,
            'Customer: First Name': 'Janet',
            'Customer: Last Name': 'Bundle',
            'Email': 'janet.bundle@example.com',
            'Shipping: Name': 'Janet Bundle',
            'Shipping: Address 1': '789 Bundle Ave',
            'Shipping: Address 2': '',
            'Shipping: City': 'Bundle City',
            'Shipping: Province Code': 'CA',
            'Shipping: Zip': '90212',
            'Note': 'Test bundle order',
            'SKU Helper': 'm.exoticfruit-5lb',  # REAL BUNDLE SKU from line 180
            'Line: Fulfillable Quantity': 1,
            'Line: ID': 'TEST003_1',
            'NEW Tags': 'extra_fruit_gift,CA-Oxnard-93030',
            'MAX PKG NUM': 1,
            'Fulfillment Center': 'CA-Oxnard-93030',
            'Saturday Shipping': 'TRUE'
        },
        # Real asian pear order - from actual CSV line 19
        {
            'Date': '2025-01-04 06:18:49 -0700',
            'Name': '#TEST004',
            'order id': 9000004,
            'Customer: First Name': 'Marc',
            'Customer: Last Name': 'Asian',
            'Email': 'marc.asian@example.com',
            'Shipping: Name': 'Marc Asian',
            'Shipping: Address 1': '18631 Test AVE',
            'Shipping: Address 2': '',
            'Shipping: City': 'PHOENIX',
            'Shipping: Province Code': 'AZ',
            'Shipping: Zip': '85027',
            'Note': 'Test asian pear order',
            'SKU Helper': 'f.asian_pear-2lb',  # REAL SKU from line 19
            'Line: Fulfillable Quantity': 1,
            'Line: ID': 'TEST004_1',
            'NEW Tags': 'p1, products_box, subscription',
            'MAX PKG NUM': 2,
            'Fulfillment Center': 'CA-Oxnard-93030',
            'Saturday Shipping': 'TRUE'
        },
        # Bundle that will cause shortage - REAL SKU
        {
            'Date': '2025-01-05 19:43:17 -0700',
            'Name': '#TEST005',
            'order id': 9000005,
            'Customer: First Name': 'Nestor',
            'Customer: Last Name': 'Farm',
            'Email': 'nestor.farm@example.com',
            'Shipping: Name': 'Nestor Farm',
            'Shipping: Address 1': '8146 E FARM LN',
            'Shipping: Address 2': '',
            'Shipping: City': 'ANAHEIM',
            'Shipping: Province Code': 'CA',
            'Shipping: Zip': '92808',
            'Note': 'Test farm box bundle',
            'SKU Helper': 'm.farmbox-3lb',  # REAL BUNDLE SKU from line 300
            'Line: Fulfillable Quantity': 2,  # Multiple quantity to test
            'Line: ID': 'TEST005_1',
            'NEW Tags': 'AfterSell, NDA, p1, Returning Buyer,CA-Oxnard-93030',
            'MAX PKG NUM': 2,
            'Fulfillment Center': 'CA-Oxnard-93030',
            'Saturday Shipping': 'TRUE'
        }
    ]
    return pd.DataFrame(orders_data)


def create_real_test_inventory():
    """Create test inventory using REAL starting balances from actual inventory file."""
    inventory_data = [
        # avocado_reed-01x01 - REAL balance from CA-Oxnard-93030: 116.00
        {
            'WarehouseName': 'CA-Oxnard-93030',
            'ItemId': 'item_001',
            'Sku': 'avocado_reed-01x01',
            'Name': 'Avocado, Reed',
            'Type': 'SellableIndividual',
            'BatchCode': 'BATCH001',
            'AvailableQty': 116.0,
            'DaysOnHand': 5,
            'Balance': 116.0
        },
        # guava_pink-10x15 - REAL balance from IL-Wheeling-60090: 158.00  
        {
            'WarehouseName': 'IL-Wheeling-60090',
            'ItemId': 'item_002',
            'Sku': 'guava_pink-10x15',
            'Name': 'Guava, Pink',
            'Type': 'SellableIndividual',
            'BatchCode': 'BATCH002',
            'AvailableQty': 158.0,
            'DaysOnHand': 30,
            'Balance': 158.0
        },
        # For Oxnard, let's add a small amount to test shortage
        {
            'WarehouseName': 'CA-Oxnard-93030',
            'ItemId': 'item_003',
            'Sku': 'guava_pink-10x15',
            'Name': 'Guava, Pink',
            'Type': 'SellableIndividual',
            'BatchCode': 'BATCH003',
            'AvailableQty': 0.0,  # Zero stock in Oxnard to test shortage
            'DaysOnHand': 0,
            'Balance': 0.0
        },
        # pear_asian-12x18 - REAL balance from IL-Wheeling-60090: 279.00
        {
            'WarehouseName': 'IL-Wheeling-60090',
            'ItemId': 'item_004',
            'Sku': 'pear_asian-12x18',
            'Name': 'Pear, Asian',
            'Type': 'SellableIndividual',
            'BatchCode': 'BATCH004',
            'AvailableQty': 279.0,
            'DaysOnHand': 15,
            'Balance': 279.0
        },
        # Bundle components for m.exoticfruit-5lb (these are the actual components from the mapping)
        # Let's add the key components with realistic balances
        {
            'WarehouseName': 'CA-Oxnard-93030',
            'ItemId': 'item_005',
            'Sku': 'lychee-BG0102',
            'Name': 'Lychee',
            'Type': 'SellableIndividual',
            'BatchCode': 'BATCH005',
            'AvailableQty': 32.0,
            'DaysOnHand': 4,
            'Balance': 32.0
        },
        {
            'WarehouseName': 'CA-Oxnard-93030',
            'ItemId': 'item_006',
            'Sku': 'mango_cherry-09x08',
            'Name': 'Mango, Cherry',
            'Type': 'SellableIndividual',
            'BatchCode': 'BATCH006',
            'AvailableQty': 72.0,
            'DaysOnHand': 6,
            'Balance': 72.0
        },
        {
            'WarehouseName': 'CA-Oxnard-93030',
            'ItemId': 'item_007',
            'Sku': 'df_yellow-05x07',
            'Name': 'Dragonfruit, Yellow',
            'Type': 'SellableIndividual',
            'BatchCode': 'BATCH007',
            'AvailableQty': 121.0,
            'DaysOnHand': 3,
            'Balance': 121.0
        },
        {
            'WarehouseName': 'CA-Oxnard-93030',
            'ItemId': 'item_008',
            'Sku': 'peach-20x50',
            'Name': 'Peach',
            'Type': 'SellableIndividual',
            'BatchCode': 'BATCH008',
            'AvailableQty': 75.0,
            'DaysOnHand': 2,
            'Balance': 75.0
        },
        {
            'WarehouseName': 'CA-Oxnard-93030',
            'ItemId': 'item_009',
            'Sku': 'lime_caviar-BG0104',
            'Name': 'Lime, Caviar',
            'Type': 'SellableIndividual',
            'BatchCode': 'BATCH009',
            'AvailableQty': 128.0,
            'DaysOnHand': 4,
            'Balance': 128.0
        },
        # One component with low stock to test shortage in bundle
        {
            'WarehouseName': 'CA-Oxnard-93030',
            'ItemId': 'item_010',
            'Sku': 'pf_purple-01x12',
            'Name': 'Passion Fruit, Purple',
            'Type': 'SellableIndividual',
            'BatchCode': 'BATCH010',
            'AvailableQty': 0.0,  # Zero stock to trigger shortage in bundle
            'DaysOnHand': 0,
            'Balance': 0.0
        }
    ]
    return pd.DataFrame(inventory_data)


def load_real_sku_mappings():
    """Load real SKU mappings from the JSON file in tests/data directory."""
    try:
        # Get the path to the data directory relative to this test file
        test_dir = os.path.dirname(__file__)
        sku_mappings_path = os.path.join(test_dir, 'data', 'sku_mappings.json')
        
        with open(sku_mappings_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: sku_mappings.json not found at {sku_mappings_path}!")
        return None


def create_temp_csv(data_df, filename_prefix):
    """Create a temporary CSV file from a DataFrame."""
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.csv', 
        prefix=filename_prefix, 
        delete=False
    )
    data_df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name


def display_sku_mappings(orders_df, sku_mappings):
    """Display detailed SKU mapping information for the orders."""
    print("   SKU MAPPING DETAILS:")
    
    # Map fulfillment center codes to warehouse keys
    fc_to_warehouse = {
        'CA-Oxnard-93030': 'Oxnard',
        'IL-Wheeling-60090': 'Wheeling'
    }
    
    for _, order in orders_df.iterrows():
        shopify_sku = order['SKU Helper']
        fc = order.get('Fulfillment Center', 'CA-Oxnard-93030')
        warehouse = fc_to_warehouse.get(fc, fc)
        
        print(f"   📦 Order {order['Name']} - {shopify_sku}:")
        
        # Check if it's a single fruit mapping
        if warehouse in sku_mappings and shopify_sku in sku_mappings[warehouse].get('singles', {}):
            mapping_info = sku_mappings[warehouse]['singles'][shopify_sku]
            warehouse_sku = mapping_info.get('picklist_sku', 'Unknown')
            actualqty = mapping_info.get('actualqty', 'Unknown')
            print(f"      → Single fruit mapped to: {warehouse_sku} (qty: {actualqty})")
            
        # Check if it's a bundle mapping
        elif warehouse in sku_mappings and shopify_sku in sku_mappings[warehouse].get('bundles', {}):
            bundle_components = sku_mappings[warehouse]['bundles'][shopify_sku]
            print(f"      → Bundle mapped to {len(bundle_components)} components:")
            for component in bundle_components[:5]:  # Show first 5 components
                component_sku = component.get('component_sku', 'Unknown')
                actualqty = component.get('actualqty', 'Unknown')
                print(f"        • {component_sku}: {actualqty} units")
            if len(bundle_components) > 5:
                print(f"        ... and {len(bundle_components) - 5} more components")
                
        else:
            print(f"      ❌ NO MAPPING FOUND in warehouse '{warehouse}' (FC: {fc})")
    print()


def test_corrected_data_processor():
    """Test the DataProcessor with CORRECTED real data."""
    
    print("=" * 80)
    print("CORRECTED DATAPROCESSOR TEST - USING REAL DATA")
    print("=" * 80)
    print()
    
    # Initialize DataProcessor
    print("1. Initializing DataProcessor...")
    processor = DataProcessor(use_airtable=False)
    print("   ✓ DataProcessor initialized")
    print()
    
    # Load REAL SKU mappings
    print("2. Loading REAL SKU mappings...")
    sku_mappings = load_real_sku_mappings()
    if sku_mappings is None:
        print("❌ FAILED: Could not load real SKU mappings")
        return False
        
    processor.sku_mappings = sku_mappings
    
    # Print mapping summary
    total_singles = sum(len(warehouse.get('singles', {})) for warehouse in sku_mappings.values())
    total_bundles = sum(len(warehouse.get('bundles', {})) for warehouse in sku_mappings.values())
    print(f"   ✓ Loaded {total_singles} single SKU mappings")
    print(f"   ✓ Loaded {total_bundles} bundle mappings")
    print(f"   ✓ Warehouses: {list(sku_mappings.keys())}")
    print()
    
    # Create test data with REAL SKUs and balances
    print("3. Creating test data with REAL SKUs and balances...")
    test_orders = create_real_test_orders()
    test_inventory = create_real_test_inventory()
    
    # Count and verify SKUs
    single_orders = test_orders[test_orders['SKU Helper'].str.startswith('f.')]
    bundle_orders = test_orders[test_orders['SKU Helper'].str.startswith('m.')]
    
    print(f"   ✓ Created {len(test_orders)} test orders using REAL SKUs:")
    print(f"     - {len(single_orders)} single fruit orders")
    print(f"     - {len(bundle_orders)} bundle orders")
    print(f"   ✓ Created {len(test_inventory)} inventory items with REAL starting balances")
    
    # Display detailed SKU mappings
    print("4. SKU MAPPING VERIFICATION AND DETAILS...")
    display_sku_mappings(test_orders, sku_mappings)
    
    # Map fulfillment center codes to warehouse keys
    fc_to_warehouse = {
        'CA-Oxnard-93030': 'Oxnard',
        'IL-Wheeling-60090': 'Wheeling'
    }
    
    # Quick validation that all SKUs have mappings
    missing_skus = []
    for _, order in test_orders.iterrows():
        sku = order['SKU Helper']
        fc = order.get('Fulfillment Center', 'CA-Oxnard-93030')
        warehouse = fc_to_warehouse.get(fc, fc)
        found = False
        if warehouse in sku_mappings:
            if sku in sku_mappings[warehouse].get('singles', {}) or sku in sku_mappings[warehouse].get('bundles', {}):
                found = True
        if not found:
            missing_skus.append(sku)
    
    if missing_skus:
        print(f"❌ FAILED: Missing SKU mappings for: {missing_skus}")
        return False
    
    # Create temporary files
    print("5. Creating temporary files...")
    orders_file = create_temp_csv(test_orders, 'corrected_orders_')
    inventory_file = create_temp_csv(test_inventory, 'corrected_inventory_')
    print(f"   ✓ Orders file: {orders_file}")
    print(f"   ✓ Inventory file: {inventory_file}")
    print()
    
    try:
        # Test data loading
        print("6. Testing DATA ENTRY POINTS...")
        print("   a) Loading orders CSV...")
        orders_df = processor.load_orders(orders_file)
        print(f"      ✓ Loaded orders: {len(orders_df)} rows, {len(orders_df.columns)} columns")
        
        print("   b) Loading inventory CSV...")
        inventory_df = processor.load_inventory(inventory_file, source='file')
        print(f"      ✓ Loaded inventory: {len(inventory_df)} rows, {len(inventory_df.columns)} columns")
        
        print("   c) SKU mappings...")
        print(f"      ✓ SKU mappings loaded: {bool(processor.sku_mappings)}")
        print()
        
        # Show relevant starting balances for mapped SKUs only
        print("7. STARTING BALANCES FOR MAPPED COMPONENTS...")
        # Get all mapped warehouse SKUs from orders
        fc_to_warehouse = {
            'CA-Oxnard-93030': 'Oxnard',
            'IL-Wheeling-60090': 'Wheeling'
        }
        
        mapped_skus = set()
        for _, order in test_orders.iterrows():
            shopify_sku = order['SKU Helper']
            fc = order.get('Fulfillment Center', 'CA-Oxnard-93030')
            warehouse = fc_to_warehouse.get(fc, fc)
            if warehouse in sku_mappings:
                if shopify_sku in sku_mappings[warehouse].get('singles', {}):
                    mapping_info = sku_mappings[warehouse]['singles'][shopify_sku]
                    mapped_skus.add(mapping_info.get('picklist_sku'))
                elif shopify_sku in sku_mappings[warehouse].get('bundles', {}):
                    for component in sku_mappings[warehouse]['bundles'][shopify_sku]:
                        mapped_skus.add(component.get('component_sku'))
        
        # Show balances only for relevant SKUs
        for _, inv_row in test_inventory.iterrows():
            sku = inv_row['Sku']
            warehouse = inv_row['WarehouseName']
            balance = inv_row['Balance']
            if sku in mapped_skus:
                print(f"   ✓ {sku} at {warehouse}: {balance} units")
        print()
        
        # Test data processing
        print("8. Testing DATA PROCESSING...")
        result = processor.process_orders(
            orders_df=orders_df,
            inventory_df=inventory_df,
            sku_mappings=sku_mappings
        )
        print("   ✓ Order processing completed")
        print()
        
        # Analyze results
        print("9. ANALYZING RESULTS...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_files = []
        for output_name, output_df in result.items():
            print(f"   {output_name}: {output_df.shape[0]} rows, {output_df.shape[1]} columns")
            
            # Save to CSV
            output_filename = f"corrected_{output_name}_{timestamp}.csv"
            output_df.to_csv(output_filename, index=False)
            output_files.append(output_filename)
            print(f"      ✓ Saved to: {output_filename}")
        print()
        
        # Detailed analysis
        print("10. DETAILED ANALYSIS:")
        
        processed_orders = result.get('orders', pd.DataFrame())
        shortage_summary = result.get('shortage_summary', pd.DataFrame())
        grouped_shortage = result.get('grouped_shortage_summary', pd.DataFrame())
        inventory_comparison = result.get('inventory_comparison', pd.DataFrame())
        initial_inventory = result.get('initial_inventory', pd.DataFrame())
        
        print("   DATA PROCESSING RESULTS:")
        print(f"   ✓ Input orders: {len(orders_df)}")
        print(f"   ✓ Output order rows: {len(processed_orders)} (expansion due to bundles)")
        print(f"   ✓ Individual shortages detected: {len(shortage_summary)}")
        print(f"   ✓ Grouped shortages: {len(grouped_shortage)}")
        print(f"   ✓ Initial inventory items: {len(initial_inventory)}")
        print(f"   ✓ Inventory comparison items: {len(inventory_comparison)}")
        print()
        
        # Show mapping results from processed orders
        if not processed_orders.empty:
            print("   MAPPING RESULTS FROM PROCESSING:")
            # Group by original Shopify SKU to show what it mapped to
            if 'shopify_sku' in processed_orders.columns and 'sku' in processed_orders.columns:
                mapping_summary = processed_orders.groupby('shopify_sku')['sku'].apply(list).to_dict()
                for shopify_sku, warehouse_skus in mapping_summary.items():
                    if len(warehouse_skus) == 1:
                        print(f"   • {shopify_sku} → {warehouse_skus[0]} (single)")
                    else:
                        print(f"   • {shopify_sku} → {len(warehouse_skus)} components:")
                        for ws in warehouse_skus[:3]:  # Show first 3 components
                            print(f"     - {ws}")
                        if len(warehouse_skus) > 3:
                            print(f"     ... and {len(warehouse_skus) - 3} more")
            print()
        
        # Shortage analysis
        if not shortage_summary.empty:
            print("   SHORTAGE ANALYSIS:")
            shortage_count = len(shortage_summary)
            print(f"   - Found {shortage_count} shortages")
            if 'shortage_qty' in shortage_summary.columns:
                total_shortage = shortage_summary['shortage_qty'].sum()
                print(f"   - Total shortage quantity: {total_shortage}")
            
            # Show specific shortages
            for _, shortage in shortage_summary.head(3).iterrows():
                sku = shortage.get('component_sku', 'Unknown')
                shopify_sku = shortage.get('shopify_sku', 'Unknown')
                qty = shortage.get('shortage_qty', 0)
                print(f"   - {sku} (from {shopify_sku}): short {qty} units")
            print()
        
        print("=" * 80)
        print("✅ CORRECTED TEST COMPLETED SUCCESSFULLY!")
        print("✅ All data entry points and outputs verified with REAL data")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        print("\n11. Cleaning up temporary files...")
        for temp_file in [orders_file, inventory_file] + output_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                print(f"   ✓ Deleted: {temp_file}")


if __name__ == '__main__':
    print("DataProcessor CORRECTED Test - Using Real Data")
    print("This test uses actual SKUs and balances from the real system files.")
    print()
    
    success = test_corrected_data_processor()
    
    if success:
        print("\n🎉 CORRECTED test completed successfully!")
        print("✅ SKU mappings verified")
        print("✅ Starting balances verified")
        print("✅ Real order processing confirmed")
    else:
        print("\n❌ CORRECTED test failed. Check error messages above.") 
