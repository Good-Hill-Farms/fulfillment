#!/usr/bin/env python
"""
Script to migrate a single table from JSON files to Airtable.
This allows for more focused error handling and troubleshooting.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

# Add project root to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.airtable_handler import AirtableHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def migrate_fulfillment_centers(handler: AirtableHandler, file_path: str) -> List[Dict[str, Any]]:
    """
    Migrate fulfillment centers from JSON file to Airtable.

    Args:
        handler (AirtableHandler): Airtable handler instance
        file_path (str): Path to the shipping zones JSON file

    Returns:
        List[Dict[str, Any]]: List of created Airtable records
    """
    logger.info(f"Migrating fulfillment centers from {file_path}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        fulfillment_centers = data.get("fulfillment_centers", [])
        created_records = []

        logger.info(f"Found {len(fulfillment_centers)} fulfillment centers to migrate")

        for fc in fulfillment_centers:
            # Format data for Airtable
            fc_data = {"name": fc["name"], "zip_code": fc["zip_code"]}

            try:
                # Create new record in Airtable
                record = handler.create_fulfillment_center(fc_data)
                created_records.append(record)
                logger.info(f"Created fulfillment center {fc['name']}")
            except Exception as e:
                logger.error(f"Failed to create fulfillment center {fc['name']}: {e}")

        logger.info(
            f"Migration complete. Created {len(created_records)} fulfillment center records"
        )
        return created_records

    except Exception as e:
        logger.error(f"Failed to migrate fulfillment centers: {e}")
        raise


def migrate_fulfillment_zones(handler: AirtableHandler, file_path: str) -> List[Dict[str, Any]]:
    """
    Migrate fulfillment zones from JSON file to Airtable.

    Args:
        handler (AirtableHandler): Airtable handler instance
        file_path (str): Path to the shipping zones JSON file

    Returns:
        List[Dict[str, Any]]: List of created Airtable records
    """
    logger.info(f"Migrating fulfillment zones from {file_path}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        fulfillment_centers = data.get("fulfillment_centers", [])
        created_records = []

        # Process all zones from all fulfillment centers
        for fc in fulfillment_centers:
            fc_name = fc["name"]
            logger.info(f"Processing zones for fulfillment center {fc_name}")

            # Process zones in batches
            zones = fc.get("zones", [])

            # First, get the Airtable ID for this fulfillment center
            fc_records = handler.get_fulfillment_centers()
            fc_record = next(
                (record for record in fc_records if record.get("name") == fc_name), None
            )

            if not fc_record:
                logger.error(f"Could not find fulfillment center record for {fc_name}")
                continue

            fc_id = fc_record.get("airtable_id")
            if not fc_id:
                logger.error(f"No Airtable ID found for fulfillment center {fc_name}")
                continue

            logger.info(f"Found Airtable ID for {fc_name}: {fc_id}")

            for zone_data in zones:
                zone_record = {
                    "zip_prefix": zone_data["zip_prefix"],
                    "zone": zone_data["zone"],
                    "FulfillmentCenter": [fc_id],  # Link to fulfillment center using its ID
                }

                try:
                    record = handler.create_fulfillment_zone(zone_record)
                    created_records.append(record)
                    logger.info(f"Created zone for zip prefix {zone_data['zip_prefix']}")
                except Exception as e:
                    logger.error(f"Failed to create fulfillment zone: {e}")

                # Small pause to avoid rate limits
                time.sleep(0.1)

        logger.info(
            f"Fulfillment zones migration complete. Created {len(created_records)} zone records"
        )
        return created_records

    except Exception as e:
        logger.error(f"Failed to migrate fulfillment zones: {e}")
        raise


def migrate_priority_tags(handler: AirtableHandler) -> List[Dict[str, Any]]:
    """
    Migrate priority tags to Airtable.

    Args:
        handler (AirtableHandler): Airtable handler instance

    Returns:
        List[Dict[str, Any]]: List of created Airtable records
    """
    logger.info("Migrating priority tags to Airtable")

    # Default priority tags to create
    priority_tags = [{"name": "High"}, {"name": "Medium"}, {"name": "Low"}]

    created_records = []

    for tag in priority_tags:
        try:
            record = handler.create_priority_tag(tag)
            created_records.append(record)
            logger.info(f"Created priority tag '{tag['name']}'")
        except Exception as e:
            logger.error(f"Failed to create priority tag '{tag['name']}': {e}")

        # Small pause to avoid rate limits
        time.sleep(0.1)

    logger.info(f"Priority tags migration complete. Created {len(created_records)} tags")
    return created_records


def migrate_sku_mappings(handler: AirtableHandler, file_path: str) -> Dict[str, Any]:
    """
    Migrate SKU mappings from JSON file to Airtable.

    Args:
        handler (AirtableHandler): Airtable handler instance
        file_path (str): Path to the SKU mappings JSON file

    Returns:
        Dict[str, Any]: Statistics about the migration
    """
    logger.info(f"Migrating SKU mappings from {file_path}")

    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Stats to track progress
        stats = {"total": 0, "created": 0, "errors": 0}

        # The file structure is different than expected
        # It's organized by fulfillment center, then by SKU
        for fc_name, fc_data in data.items():
            logger.info(f"Processing SKU mappings for fulfillment center: {fc_name}")

            if "all_skus" not in fc_data:
                logger.warning(f"No SKU data found for {fc_name}, skipping")
                continue

            all_skus = fc_data["all_skus"]
            bundles = fc_data.get("bundles", {})
            total_skus = len(all_skus)
            logger.info(f"Found {total_skus} SKUs for {fc_name}")
            logger.info(f"Found {len(bundles)} bundles for {fc_name}")

            # First, get the Airtable ID for this fulfillment center
            fc_records = handler.get_fulfillment_centers()
            fc_record = next(
                (record for record in fc_records if record.get("name") == fc_name), None
            )

            if not fc_record:
                logger.error(f"Could not find fulfillment center record for {fc_name}")
                continue

            fc_id = fc_record.get("airtable_id")
            if not fc_id:
                logger.error(f"No Airtable ID found for fulfillment center {fc_name}")
                continue

            logger.info(f"Found Airtable ID for {fc_name}: {fc_id}")

            # Process SKUs in batches to avoid rate limits
            batch_size = 20
            sku_items = list(all_skus.items())

            for i in range(0, total_skus, batch_size):
                batch = sku_items[i : i + batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 1} of {(total_skus + batch_size - 1)//batch_size}"
                )

                for sku_id, sku_data in batch:
                    try:
                        # Format data for Airtable based on the actual table structure
                        # Use numeric values for quantity fields
                        sku_record = {
                            "order_sku": sku_id,
                            "picklist_sku": sku_data.get("picklist_sku", ""),
                            "actual_qty": float(sku_data.get("actualqty", 1)),  # Use numeric value
                            "total_pick_weight": float(
                                sku_data.get("Total_Pick_Weight", 0)
                            ),  # Use numeric value
                            "pick_type": sku_data.get("Pick_Type", ""),
                            "fulfillment_center": [
                                fc_id
                            ],  # Link to fulfillment center using its ID in array format
                        }

                        # Check if this SKU is a bundle and add bundle components if available
                        if sku_id in bundles:
                            # Format bundle components as a string in the format Airtable expects
                            # Airtable expects a JSON string that it can parse
                            bundle_components_data = []
                            for component in bundles[sku_id]:
                                bundle_components_data.append(
                                    {
                                        "sku": component.get("component_sku", ""),
                                        "qty": float(component.get("actualqty", 0)),
                                    }
                                )

                            # Summary of bundle components handling:
                            # 1. Collect all components for this bundle
                            # 2. Convert each component's quantity to float
                            # 3. Format as JSON string for Airtable compatibility
                            if bundle_components_data:
                                # Convert the bundle components to a JSON string
                                # This will be stored as a string in Airtable and parsed by the application
                                sku_record["bundle_components"] = json.dumps(bundle_components_data)
                                logger.info(
                                    f"Added {len(bundle_components_data)} bundle components for {sku_id} as JSON string"
                                )

                        # Create record in Airtable
                        handler.create_sku_mapping(sku_record)
                        stats["created"] += 1
                        stats["total"] += 1
                        logger.info(f"Created SKU mapping for {sku_id}")

                    except Exception as e:
                        logger.error(f"Failed to create SKU mapping for {sku_id}: {e}")
                        stats["errors"] += 1
                        stats["total"] += 1

                # Pause between batches to avoid rate limits
                if i + batch_size < total_skus:
                    time.sleep(1)

        logger.info(f"Migration complete. Created {stats['created']} SKU mapping records")
        return stats

    except Exception as e:
        logger.error(f"Failed to migrate SKU mappings: {e}")
        raise


def main():
    """Main function to run the data migration"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Migrate a single table to Airtable")
    parser.add_argument(
        "table",
        choices=["fulfillment_centers", "fulfillment_zones", "priority_tags", "sku_mappings"],
        help="Table to migrate",
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Check for required environment variables
    if not os.getenv("AIRTABLE_API_KEY") or not os.getenv("AIRTABLE_BASE_ID"):
        logger.error("Missing required environment variables: AIRTABLE_API_KEY, AIRTABLE_BASE_ID")
        sys.exit(1)

    logger.info(f"Starting migration of {args.table} to Airtable")

    # Create handler
    handler = AirtableHandler()

    try:
        # Migrate the selected table
        if args.table == "fulfillment_centers":
            file_path = os.path.join("constants", "data", "shipping_zones.json")
            records = migrate_fulfillment_centers(handler, file_path)
            logger.info(f"Successfully migrated {len(records)} fulfillment centers")

        elif args.table == "fulfillment_zones":
            file_path = os.path.join("constants", "data", "shipping_zones.json")
            records = migrate_fulfillment_zones(handler, file_path)
            logger.info(f"Successfully migrated {len(records)} fulfillment zones")

        elif args.table == "priority_tags":
            records = migrate_priority_tags(handler)
            logger.info(f"Successfully migrated {len(records)} priority tags")

        elif args.table == "sku_mappings":
            file_path = os.path.join("constants", "data", "sku_mappings.json")
            stats = migrate_sku_mappings(handler, file_path)
            logger.info(f"Successfully migrated {stats['created']} SKU mappings")

        logger.info("Migration completed successfully")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
