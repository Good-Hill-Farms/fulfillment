#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Dict, List, Set

def parse_csv_file(filepath: str) -> List[Dict]:
    """Parse a CSV file containing SKU mappings."""
    records = []
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Skip header rows or empty rows
            if row.get('shopifysku2') == 'shopifysku2' or not row.get('shopifysku2'):
                continue
            
            # Only process rows with valid picklist SKUs
            if not row.get('picklist sku'):
                continue
                
            # Clean and standardize field names
            record = {
                'order_sku': row.get('shopifysku2', ''),
                'picklist_sku': row.get('picklist sku', ''),
                'actualqty': float(row.get('actualqty', '0') or '0'),
                'total_pick_weight': float(row.get('Total Pick Weight', '0') or '0'),
                'pick_type': row.get('Pick Type', ''),
                'pick_type_inventory': row.get('Pick Type Inventory', '')
            }
            
            # Add the record
            records.append(record)
    
    return records

def create_warehouse_structure(records: List[Dict]) -> Dict:
    """Create the nested warehouse structure with all_skus and bundles."""
    result = {
        "all_skus": {},
        "bundles": {}
    }
    
    # Group records by order_sku to identify bundles vs simple SKUs
    grouped = {}
    for record in records:
        order_sku = record['order_sku']
        if order_sku not in grouped:
            grouped[order_sku] = []
        grouped[order_sku].append(record)
    
    # Process each group
    for order_sku, sku_records in grouped.items():
        if len(sku_records) == 1:
            # Single component = simple SKU
            record = sku_records[0]
            result["all_skus"][order_sku] = {
                "picklist_sku": record["picklist_sku"],
                "actualqty": record["actualqty"],
                "total_pick_weight": record["total_pick_weight"],
                "pick_type": record["pick_type"],
                "pick_type_inventory": record["pick_type_inventory"]
            }
        else:
            # Multiple components = bundle
            components = []
            for record in sku_records:
                components.append({
                    "component_sku": record["picklist_sku"],
                    "actualqty": record["actualqty"],
                    "weight": record["total_pick_weight"],
                    "pick_type": record["pick_type"]
                })
            result["bundles"][order_sku] = components
    
    return result

def main():
    # Define paths to CSV files
    oxnard_csv = 'docs/sku_shopify_to_oxnard.csv'
    wheeling_csv = 'docs/sku_shopify_to_wheeling.csv'
    output_json = 'docs/sku_mappings_updated.json'
    
    # Parse CSV files
    oxnard_records = parse_csv_file(oxnard_csv)
    wheeling_records = parse_csv_file(wheeling_csv)
    
    # Process data for each warehouse
    oxnard_structure = create_warehouse_structure(oxnard_records)
    wheeling_structure = create_warehouse_structure(wheeling_records)
    
    # Build final structure
    final_data = {
        "Oxnard": oxnard_structure,
        "Wheeling": wheeling_structure
    }
    
    # Create parent directories if needed
    output_path = Path(output_json)
    output_path.parent.mkdir(exist_ok=True)
    
    # Write to JSON file
    with open(output_json, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    # Count and display statistics
    oxnard_simple = len(oxnard_structure["all_skus"])
    oxnard_bundles = len(oxnard_structure["bundles"])
    wheeling_simple = len(wheeling_structure["all_skus"])
    wheeling_bundles = len(wheeling_structure["bundles"])
    
    print(f"Successfully created SKU mapping JSON in exact matching format")
    print(f"Output file: {output_json}")
    print(f"Statistics:")
    print(f"  Oxnard: {oxnard_simple} simple SKUs, {oxnard_bundles} bundles")
    print(f"  Wheeling: {wheeling_simple} simple SKUs, {wheeling_bundles} bundles")
    print(f"  Total: {oxnard_simple + wheeling_simple} simple SKUs, {oxnard_bundles + wheeling_bundles} bundles")

if __name__ == "__main__":
    main()
