#!/usr/bin/env python3
"""
Upload SKU Mappings to Airtable

This script takes the updated SKU mappings JSON file and uploads it to Airtable
using the existing AirtableHandler and AirtableDataMigrator classes.
"""

import json
import logging
import os
import sys
from typing import Dict

# Add project root to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.airtable_handler import AirtableHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def upload_sku_mappings(
    json_path: str = "/Users/olenaliubynetska/fulfillment-mixy-matchi/constants/data/sku_mappings.json",
) -> Dict[str, int]:
    """
    Upload SKU mappings from the updated JSON file to Airtable.

    Args:
        json_path (str): Path to the updated SKU mappings JSON file

    Returns:
        Dict[str, int]: Statistics about the upload process
    """
    logger.info(f"Starting upload of SKU mappings from {json_path}")

    try:
        # Load the JSON data
        with open(json_path, "r") as f:
            data = json.load(f)

        # Initialize the Airtable handler
        airtable = AirtableHandler()

        # Get fulfillment centers to map names to IDs
        fc_records = airtable.get_fulfillment_centers()
        fc_map = {fc["name"]: fc["airtable_id"] for fc in fc_records}

        if not fc_map:
            logger.error("No fulfillment centers found in Airtable. Please create them first.")
            return {"error": "No fulfillment centers found"}

        # Statistics dictionary
        stats = {"total": 0, "simple_skus": 0, "bundles": 0, "errors": 0}

        # Process each warehouse's data
        for warehouse_name, warehouse_data in data.items():
            logger.info(f"Processing {warehouse_name} data")

            # Check if this fulfillment center exists in Airtable
            if warehouse_name not in fc_map:
                logger.error(
                    f"Fulfillment center '{warehouse_name}' not found in Airtable. Skipping."
                )
                continue

            fc_id = fc_map[warehouse_name]
            stats[warehouse_name] = 0

            # Process simple SKUs
            all_skus = warehouse_data.get("all_skus", {})
            logger.info(f"Processing {len(all_skus)} simple SKUs for {warehouse_name}")

            for order_sku, mapping in all_skus.items():
                # Check if SKU mapping already exists
                # Get all SKU mappings filtered by warehouse
                sku_mappings = airtable.get_sku_mappings(warehouse=warehouse_name)

                # Find the specific SKU mapping for this order_sku and picklist_sku
                existing = next(
                    (
                        m
                        for m in sku_mappings
                        if m.get("order_sku") == order_sku
                        and m.get("picklist_sku") == mapping["picklist_sku"]
                    ),
                    None,
                )

                # Prepare the SKU mapping data
                sku_data = {
                    "order_sku": order_sku,
                    "picklist_sku": mapping["picklist_sku"],
                    "actual_qty": float(mapping["actualqty"]),
                    "total_pick_weight": float(mapping.get("total_pick_weight", 0))
                    if mapping.get("total_pick_weight")
                    else None,
                    "pick_type": mapping.get("pick_type", ""),
                    # Store fulfillment center as an array of IDs for Airtable link field
                    "fulfillment_center": [fc_id],
                    "is_bundle": False,
                }

                try:
                    if existing:
                        # Update existing record
                        airtable.update_sku_mapping(existing["id"], sku_data)
                        logger.debug(f"Updated SKU mapping for {order_sku}")
                    else:
                        # Create new record
                        airtable.create_sku_mapping(sku_data)
                        logger.debug(f"Created SKU mapping for {order_sku}")

                    stats["simple_skus"] += 1
                    stats[warehouse_name] += 1
                    stats["total"] += 1
                except Exception as e:
                    logger.error(f"Error processing SKU {order_sku}: {e}")
                    stats["errors"] += 1

            # Process bundles
            bundles = warehouse_data.get("bundles", {})
            logger.info(f"Processing {len(bundles)} bundles for {warehouse_name}")

            for order_sku, components in bundles.items():
                # Create JSON string of bundle components
                bundle_components_json = json.dumps(
                    [
                        {
                            "component_sku": c["component_sku"],
                            "actualqty": c["actualqty"],
                            "weight": c["weight"],
                            "pick_type": c.get("pick_type", ""),
                        }
                        for c in components
                    ]
                )

                # Check if SKU mapping already exists
                # Get all SKU mappings filtered by warehouse
                sku_mappings = airtable.get_sku_mappings(warehouse=warehouse_name)

                # Find the specific bundle SKU mapping for this order_sku
                existing = next(
                    (
                        m
                        for m in sku_mappings
                        if m.get("order_sku") == order_sku and m.get("is_bundle") == True
                    ),
                    None,
                )

                # Use the first component's picklist SKU as the primary one for bundles
                first_component = components[0] if components else {}

                # Prepare the SKU mapping data
                sku_data = {
                    "order_sku": order_sku,
                    "picklist_sku": first_component.get("component_sku", ""),
                    "actual_qty": float(first_component.get("actualqty", 1.0)),
                    "total_pick_weight": float(first_component.get("weight", 0.0))
                    if first_component.get("weight")
                    else None,
                    "pick_type": first_component.get("pick_type", ""),
                    # Store fulfillment center as an array of IDs for Airtable link field
                    "fulfillment_center": [fc_id],
                    "is_bundle": True,
                    "bundle_components": bundle_components_json,
                }

                try:
                    if existing:
                        # Update existing record
                        airtable.update_sku_mapping(existing["id"], sku_data)
                        logger.debug(f"Updated bundle mapping for {order_sku}")
                    else:
                        # Create new record
                        airtable.create_sku_mapping(sku_data)
                        logger.debug(f"Created bundle mapping for {order_sku}")

                    stats["bundles"] += 1
                    stats[warehouse_name] += 1
                    stats["total"] += 1
                except Exception as e:
                    logger.error(f"Error processing bundle {order_sku}: {e}")
                    stats["errors"] += 1

        logger.info(f"SKU mappings upload complete. Stats: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Failed to upload SKU mappings: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload SKU mappings to Airtable")
    parser.add_argument(
        "--file",
        "-f",
        help="Path to the SKU mappings JSON file",
        default="/Users/olenaliubynetska/fulfillment-mixy-matchi/constants/data/sku_mappings.json",
    )

    args = parser.parse_args()

    try:
        stats = upload_sku_mappings(args.file)
        print("\nUpload Statistics:")
        print(f"Total SKUs processed: {stats.get('total', 0)}")
        print(f"Simple SKUs: {stats.get('simple_skus', 0)}")
        print(f"Bundles: {stats.get('bundles', 0)}")
        print(f"Errors: {stats.get('errors', 0)}")

        for warehouse, count in stats.items():
            if warehouse not in ["total", "simple_skus", "bundles", "errors"]:
                print(f"{warehouse}: {count} SKUs")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
