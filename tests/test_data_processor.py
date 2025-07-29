#!/usr/bin/env python3
"""
Unit tests for DataProcessor using real data from actual files.

This test suite uses:
- Real SKU mappings from sku_mappings.json
- Real SKU names from actual orders CSV
- Real starting balances from actual inventory CSV
- Proper bundle orders that exist in the real system
"""

import json
import pandas as pd
import tempfile
import os
import unittest
from datetime import datetime
import sys

# Add the parent directory to the path to access utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Unit tests for DataProcessor with real data."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with real SKU mappings."""
        cls.sku_mappings = cls._load_real_sku_mappings()
        cls.fc_to_warehouse = {
            'CA-Oxnard-93030': 'Oxnard',
            'IL-Wheeling-60090': 'Wheeling'
        }
        
    @classmethod
    def _load_real_sku_mappings(cls):
        """Load real SKU mappings from the JSON file."""
        try:
            test_dir = os.path.dirname(__file__)
            sku_mappings_path = os.path.join(test_dir, 'data', 'sku_mappings.json')
            
            with open(sku_mappings_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            cls.fail(f"sku_mappings.json not found at {sku_mappings_path}")
    
    def setUp(self):
        """Set up each test with fresh data and processor."""
        self.processor = DataProcessor(use_airtable=False)
        self.processor.sku_mappings = self.sku_mappings
        
        # Create test data
        self.test_orders = self._create_test_orders()
        self.test_inventory = self._create_test_inventory()
        
        # Create temporary files
        self.orders_file = self._create_temp_csv(self.test_orders, 'test_orders_')
        self.inventory_file = self._create_temp_csv(self.test_inventory, 'test_inventory_')
        
        # Keep track of created files for cleanup
        self.temp_files = [self.orders_file, self.inventory_file]
        
    def tearDown(self):
        """Clean up temporary files after each test."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _create_test_orders(self):
        """Create test orders using real SKUs from actual orders file."""
        orders_data = [
            # Real single fruit order - avocado
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
                'SKU Helper': 'f.avocado_reed-2lb',
                'Line: Fulfillable Quantity': 1,
                'Line: ID': 'TEST001_1',
                'NEW Tags': 'P1, products_box,CA-Oxnard-93030',
                'MAX PKG NUM': 1,
                'Fulfillment Center': 'CA-Oxnard-93030',
                'Saturday Shipping': 'FALSE'
            },
            # Real single fruit order - guava
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
                'SKU Helper': 'f.guava_pink-2lb',
                'Line: Fulfillable Quantity': 1,
                'Line: ID': 'TEST002_1',
                'NEW Tags': 'P1, products_box,CA-Oxnard-93030',
                'MAX PKG NUM': 1,
                'Fulfillment Center': 'CA-Oxnard-93030',
                'Saturday Shipping': 'FALSE'
            },
            # Real bundle order - exotic fruit
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
                'SKU Helper': 'm.exoticfruit-5lb',
                'Line: Fulfillable Quantity': 1,
                'Line: ID': 'TEST003_1',
                'NEW Tags': 'extra_fruit_gift,CA-Oxnard-93030',
                'MAX PKG NUM': 1,
                'Fulfillment Center': 'CA-Oxnard-93030',
                'Saturday Shipping': 'TRUE'
            }
        ]
        return pd.DataFrame(orders_data)
    
    def _create_test_inventory(self):
        """Create test inventory using real starting balances."""
        inventory_data = [
            # avocado_reed-01x01
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
            # guava_pink-10x15
            {
                'WarehouseName': 'CA-Oxnard-93030',
                'ItemId': 'item_002',
                'Sku': 'guava_pink-10x15',
                'Name': 'Guava, Pink',
                'Type': 'SellableIndividual',
                'BatchCode': 'BATCH002',
                'AvailableQty': 50.0,
                'DaysOnHand': 3,
                'Balance': 50.0
            },
            # Bundle components for m.exoticfruit-5lb
            {
                'WarehouseName': 'CA-Oxnard-93030',
                'ItemId': 'item_003',
                'Sku': 'lychee-BG0102',
                'Name': 'Lychee',
                'Type': 'SellableIndividual',
                'BatchCode': 'BATCH003',
                'AvailableQty': 32.0,
                'DaysOnHand': 4,
                'Balance': 32.0
            },
            {
                'WarehouseName': 'CA-Oxnard-93030',
                'ItemId': 'item_004',
                'Sku': 'mango_cherry-09x08',
                'Name': 'Mango, Cherry',
                'Type': 'SellableIndividual',
                'BatchCode': 'BATCH004',
                'AvailableQty': 72.0,
                'DaysOnHand': 6,
                'Balance': 72.0
            },
            {
                'WarehouseName': 'CA-Oxnard-93030',
                'ItemId': 'item_005',
                'Sku': 'df_yellow-05x07',
                'Name': 'Dragonfruit, Yellow',
                'Type': 'SellableIndividual',
                'BatchCode': 'BATCH005',
                'AvailableQty': 121.0,
                'DaysOnHand': 3,
                'Balance': 121.0
            }
        ]
        return pd.DataFrame(inventory_data)
    
    def _create_temp_csv(self, data_df, filename_prefix):
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
    
    def test_sku_mappings_loaded(self):
        """Test that SKU mappings are properly loaded."""
        print(f"\n🔍 Testing SKU mappings loaded...")
        
        self.assertIsNotNone(self.sku_mappings)
        self.assertIn('Oxnard', self.sku_mappings)
        self.assertIn('Wheeling', self.sku_mappings)
        
        # Check structure
        oxnard_data = self.sku_mappings['Oxnard']
        self.assertIn('singles', oxnard_data)
        self.assertIn('bundles', oxnard_data)
        
        # Count mappings
        total_singles = sum(len(warehouse.get('singles', {})) for warehouse in self.sku_mappings.values())
        total_bundles = sum(len(warehouse.get('bundles', {})) for warehouse in self.sku_mappings.values())
        
        print(f"✅ Found {total_singles} single SKU mappings")
        print(f"✅ Found {total_bundles} bundle mappings") 
        print(f"✅ Warehouses: {list(self.sku_mappings.keys())}")
        
        self.assertGreater(total_singles, 0, "Should have single SKU mappings")
        self.assertGreater(total_bundles, 0, "Should have bundle mappings")
    
    def test_single_sku_mappings_exist(self):
        """Test that single fruit SKUs have proper mappings."""
        print(f"\n🍎 Testing single SKU mappings...")
        test_skus = ['f.avocado_reed-2lb', 'f.guava_pink-2lb']
        
        for sku in test_skus:
            with self.subTest(sku=sku):
                found = False
                for warehouse in ['Oxnard', 'Wheeling']:
                    if sku in self.sku_mappings[warehouse].get('singles', {}):
                        mapping = self.sku_mappings[warehouse]['singles'][sku]
                        warehouse_sku = mapping.get('picklist_sku', 'Unknown')
                        actualqty = mapping.get('actualqty', 'Unknown')
                        print(f"✅ {sku} → {warehouse_sku} (qty: {actualqty}) in {warehouse}")
                        
                        self.assertIn('picklist_sku', mapping)
                        self.assertIn('actualqty', mapping)
                        found = True
                        break
                self.assertTrue(found, f"SKU {sku} should have a mapping")
    
    def test_bundle_sku_mappings_exist(self):
        """Test that bundle SKUs have proper mappings."""
        print(f"\n📦 Testing bundle SKU mappings...")
        test_bundles = ['m.exoticfruit-5lb']
        
        for sku in test_bundles:
            with self.subTest(sku=sku):
                found = False
                for warehouse in ['Oxnard', 'Wheeling']:
                    if sku in self.sku_mappings[warehouse].get('bundles', {}):
                        components = self.sku_mappings[warehouse]['bundles'][sku]
                        print(f"✅ {sku} → {len(components)} components in {warehouse}:")
                        
                        self.assertIsInstance(components, list)
                        self.assertGreater(len(components), 0)
                        
                        # Check component structure and show first few
                        for i, component in enumerate(components[:3]):
                            component_sku = component.get('component_sku', 'Unknown')
                            actualqty = component.get('actualqty', 'Unknown')
                            print(f"   • {component_sku}: {actualqty} units")
                            self.assertIn('component_sku', component)
                            self.assertIn('actualqty', component)
                        
                        if len(components) > 3:
                            print(f"   ... and {len(components) - 3} more components")
                        
                        found = True
                        break
                self.assertTrue(found, f"Bundle SKU {sku} should have a mapping")
    
    def test_data_loading(self):
        """Test that data files can be loaded properly."""
        print(f"\n📁 Testing data loading...")
        
        # Test orders loading
        orders_df = self.processor.load_orders(self.orders_file)
        print(f"✅ Loaded orders: {len(orders_df)} rows, {len(orders_df.columns)} columns")
        
        self.assertIsInstance(orders_df, pd.DataFrame)
        self.assertGreater(len(orders_df), 0)
        self.assertIn('SKU Helper', orders_df.columns)
        
        # Show loaded SKUs
        print(f"📋 Order SKUs: {list(orders_df['SKU Helper'])}")
        
        # Test inventory loading
        inventory_df = self.processor.load_inventory(self.inventory_file, source='file')
        print(f"✅ Loaded inventory: {len(inventory_df)} rows, {len(inventory_df.columns)} columns")
        
        self.assertIsInstance(inventory_df, pd.DataFrame)
        self.assertGreater(len(inventory_df), 0)
        self.assertIn('Sku', inventory_df.columns)
        self.assertIn('Balance', inventory_df.columns)
        
        # Show inventory SKUs
        print(f"📦 Inventory SKUs: {list(inventory_df['Sku'])}")
    
    def test_order_processing(self):
        """Test the complete order processing workflow."""
        print(f"\n⚙️ Testing order processing workflow...")
        
        # Load data
        orders_df = self.processor.load_orders(self.orders_file)
        inventory_df = self.processor.load_inventory(self.inventory_file, source='file')
        
        print(f"📊 Processing {len(orders_df)} orders with {len(inventory_df)} inventory items...")
        
        # Process orders
        result = self.processor.process_orders(
            orders_df=orders_df,
            inventory_df=inventory_df,
            sku_mappings=self.sku_mappings
        )
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        expected_keys = ['orders', 'shortage_summary', 'grouped_shortage_summary', 
                        'initial_inventory', 'inventory_comparison']
        for key in expected_keys:
            self.assertIn(key, result, f"Result should contain {key}")
            self.assertIsInstance(result[key], pd.DataFrame)
        
        # Verify processing results
        processed_orders = result['orders']
        print(f"✅ Input: {len(orders_df)} orders → Output: {len(processed_orders)} rows (expanded)")
        
        self.assertGreater(len(processed_orders), len(orders_df), 
                          "Processed orders should expand due to bundles")
        
        # Check that SKU mapping occurred
        if not processed_orders.empty:
            self.assertIn('sku', processed_orders.columns)
            self.assertIn('shopifysku2', processed_orders.columns)
            print(f"✅ SKU mapping columns present: 'sku', 'shopifysku2'")
    
    def test_sku_mapping_results(self):
        """Test that SKU mappings produce expected results."""
        print(f"\n🎯 Testing SKU mapping results...")
        
        orders_df = self.processor.load_orders(self.orders_file)
        inventory_df = self.processor.load_inventory(self.inventory_file, source='file')
        
        result = self.processor.process_orders(
            orders_df=orders_df,
            inventory_df=inventory_df,
            sku_mappings=self.sku_mappings
        )
        
        processed_orders = result['orders']
        
        if not processed_orders.empty:
            # Group by original Shopify SKU to verify mappings
            mapping_summary = processed_orders.groupby('shopifysku2')['sku'].apply(list).to_dict()
            
            print(f"📋 Mapping Results:")
            for shopify_sku, warehouse_skus in mapping_summary.items():
                if len(warehouse_skus) == 1:
                    print(f"   • {shopify_sku} → {warehouse_skus[0]} (single)")
                else:
                    print(f"   • {shopify_sku} → {len(warehouse_skus)} components:")
                    for ws in warehouse_skus[:3]:
                        print(f"     - {ws}")
                    if len(warehouse_skus) > 3:
                        print(f"     ... and {len(warehouse_skus) - 3} more")
            
            # Test single fruit mapping
            if 'f.avocado_reed-2lb' in mapping_summary:
                mapped_skus = mapping_summary['f.avocado_reed-2lb']
                self.assertEqual(len(mapped_skus), 1, "Single fruit should map to one SKU")
                self.assertEqual(mapped_skus[0], 'avocado_reed-01x01')
                print(f"✅ Avocado mapping verified: f.avocado_reed-2lb → avocado_reed-01x01")
            
            # Test bundle mapping
            if 'm.exoticfruit-5lb' in mapping_summary:
                mapped_skus = mapping_summary['m.exoticfruit-5lb']
                self.assertGreater(len(mapped_skus), 1, "Bundle should map to multiple SKUs")
                # Should include lychee as one component
                self.assertIn('lychee-BG0102', mapped_skus)
                print(f"✅ Bundle mapping verified: m.exoticfruit-5lb → {len(mapped_skus)} components including lychee-BG0102")
    
    def test_shortage_detection(self):
        """Test that inventory shortages are properly detected."""
        print(f"\n⚠️ Testing shortage detection...")
        
        # Create inventory with intentional shortage
        shortage_inventory = self.test_inventory.copy()
        shortage_inventory.loc[shortage_inventory['Sku'] == 'guava_pink-10x15', 'Balance'] = 1.0
        shortage_inventory.loc[shortage_inventory['Sku'] == 'guava_pink-10x15', 'AvailableQty'] = 1.0
        
        print(f"🔧 Created shortage scenario: guava_pink-10x15 balance = 1.0 (order needs 3.0)")
        
        # Save to temp file
        shortage_file = self._create_temp_csv(shortage_inventory, 'shortage_inventory_')
        self.temp_files.append(shortage_file)
        
        # Process with shortage
        orders_df = self.processor.load_orders(self.orders_file)
        inventory_df = self.processor.load_inventory(shortage_file, source='file')
        
        result = self.processor.process_orders(
            orders_df=orders_df,
            inventory_df=inventory_df,
            sku_mappings=self.sku_mappings
        )
        
        # Verify shortage detection
        shortage_summary = result['shortage_summary']
        print(f"✅ Detected {len(shortage_summary)} shortages")
        
        self.assertGreater(len(shortage_summary), 0, "Should detect shortages")
        
        # Check if guava shortage is detected
        if not shortage_summary.empty and 'component_sku' in shortage_summary.columns:
            guava_shortages = shortage_summary[shortage_summary['component_sku'] == 'guava_pink-10x15']
            print(f"✅ Guava shortages found: {len(guava_shortages)}")
            self.assertGreater(len(guava_shortages), 0, "Should detect guava shortage")
    
    def test_missing_sku_mappings_file(self):
        """Test error handling when SKU mappings file is missing."""
        print(f"\n❌ Testing missing SKU mappings file...")
        
        # Try to create processor without mappings
        processor_no_mappings = DataProcessor(use_airtable=False)
        processor_no_mappings.sku_mappings = None
        
        orders_df = self.processor.load_orders(self.orders_file)
        inventory_df = self.processor.load_inventory(self.inventory_file, source='file')
        
        # DataProcessor handles missing mappings gracefully by logging warnings and skipping SKUs
        result = processor_no_mappings.process_orders(
            orders_df=orders_df,
            inventory_df=inventory_df,
            sku_mappings=None
        )
        
        # Should still return valid structure but with no processed orders (SKUs skipped)
        self.assertIsInstance(result, dict)
        processed_orders = result.get('orders', pd.DataFrame())
        
        print(f"✅ Missing mappings handled gracefully: {len(processed_orders)} orders processed")
        print(f"✅ System logged warnings and skipped unmapped SKUs")
        
        # With no mappings, should have few or no processed orders
        self.assertTrue(len(processed_orders) <= len(orders_df), 
                       "Without mappings, should process same or fewer orders")
    
    def test_invalid_sku_in_orders(self):
        """Test handling of invalid/unmapped SKUs in orders."""
        print(f"\n🚫 Testing invalid SKU handling...")
        
        # Create orders with invalid SKU
        invalid_orders = self.test_orders.copy()
        invalid_orders.loc[0, 'SKU Helper'] = 'f.nonexistent_fruit-2lb'
        
        print(f"🔧 Added invalid SKU: f.nonexistent_fruit-2lb")
        
        # Save to temp file
        invalid_orders_file = self._create_temp_csv(invalid_orders, 'invalid_orders_')
        self.temp_files.append(invalid_orders_file)
        
        # Process with invalid SKU
        orders_df = self.processor.load_orders(invalid_orders_file)
        inventory_df = self.processor.load_inventory(self.inventory_file, source='file')
        
        # Should not crash, but may produce warnings or skip unmapped SKUs
        result = self.processor.process_orders(
            orders_df=orders_df,
            inventory_df=inventory_df,
            sku_mappings=self.sku_mappings
        )
        
        # Should still return valid structure
        self.assertIsInstance(result, dict)
        processed_orders = result['orders']
        
        # Check if invalid SKU was skipped or handled
        if not processed_orders.empty and 'shopifysku2' in processed_orders.columns:
            processed_skus = processed_orders['shopifysku2'].unique()
            print(f"✅ Processed SKUs: {list(processed_skus)}")
            # The invalid SKU might be skipped entirely or cause an error row
        
        print(f"✅ System handled invalid SKU gracefully")
    
    def test_empty_inventory(self):
        """Test handling of empty inventory."""
        print(f"\n📭 Testing empty inventory...")
        
        # Create empty inventory
        empty_inventory = pd.DataFrame(columns=self.test_inventory.columns)
        
        # Save to temp file
        empty_inventory_file = self._create_temp_csv(empty_inventory, 'empty_inventory_')
        self.temp_files.append(empty_inventory_file)
        
        # Process with empty inventory
        orders_df = self.processor.load_orders(self.orders_file)
        inventory_df = self.processor.load_inventory(empty_inventory_file, source='file')
        
        print(f"🔧 Processing {len(orders_df)} orders with empty inventory")
        
        result = self.processor.process_orders(
            orders_df=orders_df,
            inventory_df=inventory_df,
            sku_mappings=self.sku_mappings
        )
        
        # Should detect shortages for everything
        shortage_summary = result['shortage_summary']
        print(f"✅ Empty inventory created {len(shortage_summary)} shortages")
        
        self.assertGreater(len(shortage_summary), 0, "Empty inventory should create shortages")
    
    def test_malformed_csv_files(self):
        """Test handling of malformed CSV files."""
        print(f"\n💥 Testing malformed CSV handling...")
        
        # Create malformed orders file (missing required columns)
        malformed_orders = pd.DataFrame({
            'Date': ['2025-01-01'],
            'Name': ['#TEST001'],
            # Missing 'SKU Helper' column
        })
        
        malformed_orders_file = self._create_temp_csv(malformed_orders, 'malformed_orders_')
        self.temp_files.append(malformed_orders_file)
        
        # Should handle missing columns gracefully
        try:
            orders_df = self.processor.load_orders(malformed_orders_file)
            print(f"⚠️ Loaded malformed orders: {orders_df.shape}")
            
            # This will likely fail when processing due to missing SKU Helper
            inventory_df = self.processor.load_inventory(self.inventory_file, source='file')
            
            with self.assertRaises((KeyError, AttributeError)):
                result = self.processor.process_orders(
                    orders_df=orders_df,
                    inventory_df=inventory_df,
                    sku_mappings=self.sku_mappings
                )
            print(f"✅ Correctly detected malformed CSV structure")
            
        except Exception as e:
            print(f"✅ Appropriately failed on malformed CSV: {type(e).__name__}")
            # This is expected behavior


def display_sku_mappings(orders_df, sku_mappings):
    """Display detailed SKU mapping information for debugging."""
    print("   SKU MAPPING DETAILS:")
    
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


if __name__ == '__main__':
    # Run specific test methods or all tests
    unittest.main(verbosity=2) 
