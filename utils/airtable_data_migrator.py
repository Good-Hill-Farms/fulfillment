"""
Airtable Data Migration Utility

This script populates Airtable with data from local JSON files for:
- Delivery Services
- Shipping Zones (Fulfillment Centers)
- SKU Mappings

It provides functionality to migrate data from JSON files to Airtable
and includes methods to add, edit, and remove data.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Add project root to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.airtable_handler import AirtableHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AirtableDataMigrator:
    """Handles migration of data from JSON files to Airtable"""

    def __init__(self):
        """Initialize the data migrator with an Airtable handler"""
        self.airtable = AirtableHandler()

        # Default paths to JSON data files
        self.json_paths = {
            "delivery_services": os.path.join("constants", "data", "delivery_services.json"),
            "shipping_zones": os.path.join("constants", "data", "shipping_zones.json"),
            "sku_mappings": os.path.join("constants", "data", "sku_mappings.json"),
        }

    def migrate_delivery_services(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Migrate delivery services data from JSON file to Airtable.

        Args:
            file_path (Optional[str]): Path to the delivery services JSON file.
                                      If None, uses the default path.

        Returns:
            List[Dict[str, Any]]: List of created Airtable records
        """
        if file_path is None:
            file_path = self.json_paths["delivery_services"]

        logger.info(f"Migrating delivery services from {file_path}")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            delivery_services = data.get("delivery_services", [])
            created_records = []

            total = len(delivery_services)
            logger.info(f"Found {total} delivery services to migrate")

            # Process in batches to avoid rate limits
            batch_size = 10
            for i in range(0, total, batch_size):
                batch = delivery_services[i : i + batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} records)"
                )

                for service in batch:
                    # Check if record already exists to avoid duplicates
                    existing = self.airtable.get_delivery_services(
                        zip_prefix=service["destination_zip_short"], origin=service["origin"]
                    )

                    if existing:
                        logger.info(
                            f"Delivery service already exists for zip {service['destination_zip_short']} from {service['origin']}"
                        )
                        continue

                    # Create new record in Airtable
                    try:
                        record = self.airtable.create_delivery_service(service)
                        created_records.append(record)
                        logger.info(
                            f"Created delivery service for zip {service['destination_zip_short']} from {service['origin']}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to create delivery service: {e}")

                # Pause between batches to avoid rate limits
                if i + batch_size < total:
                    time.sleep(1)

            logger.info(
                f"Migration complete. Created {len(created_records)} new delivery service records"
            )
            return created_records

        except Exception as e:
            logger.error(f"Failed to migrate delivery services: {e}")
            raise

    def migrate_fulfillment_centers(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Migrate fulfillment centers from JSON file to Airtable.

        Args:
            file_path (Optional[str]): Path to the shipping zones JSON file.
                                      If None, uses the default path.

        Returns:
            List[Dict[str, Any]]: List of created Airtable records
        """
        if file_path is None:
            file_path = self.json_paths["shipping_zones"]

        logger.info(f"Migrating fulfillment centers from {file_path}")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            fulfillment_centers = data.get("fulfillment_centers", [])
            created_records = []

            logger.info(f"Found {len(fulfillment_centers)} fulfillment centers to migrate")

            for fc in fulfillment_centers:
                # Check if record already exists
                existing = [
                    x
                    for x in self.airtable.get_fulfillment_centers()
                    if x.get("name") == fc["name"]
                ]

                if existing:
                    logger.info(f"Fulfillment center {fc['name']} already exists, updating")
                    # Update existing record
                    record_id = existing[0]["airtable_id"]

                    # Format data for Airtable - no longer storing zones directly in fulfillment center
                    fc_data = {"name": fc["name"], "zip_code": fc["zip_code"]}

                    try:
                        record = self.airtable.update_fulfillment_center(record_id, fc_data)
                        created_records.append(record)
                        logger.info(f"Updated fulfillment center {fc['name']}")
                    except Exception as e:
                        logger.error(f"Failed to update fulfillment center: {e}")
                else:
                    # Format data for Airtable - no longer storing zones directly in fulfillment center
                    fc_data = {"name": fc["name"], "zip_code": fc["zip_code"]}

                    try:
                        record = self.airtable.create_fulfillment_center(fc_data)
                        created_records.append(record)
                        logger.info(f"Created fulfillment center {fc['name']}")
                    except Exception as e:
                        logger.error(f"Failed to create fulfillment center: {e}")

            logger.info(
                f"Migration complete. Created or updated {len(created_records)} fulfillment center records"
            )
            return created_records

        except Exception as e:
            logger.error(f"Failed to migrate fulfillment centers: {e}")
            raise

    def migrate_fulfillment_zones(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Migrate fulfillment zones from JSON file to Airtable.

        Args:
            file_path (Optional[str]): Path to the shipping zones JSON file.
                                      If None, uses the default path.

        Returns:
            List[Dict[str, Any]]: List of created Airtable records
        """
        if file_path is None:
            file_path = self.json_paths["shipping_zones"]

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
                batch_size = 10
                total = len(zones)

                for i in range(0, total, batch_size):
                    batch = zones[i : i + batch_size]
                    logger.info(
                        f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} zones)"
                    )

                    for zone_data in batch:
                        # Check if zone already exists
                        existing_zones = self.airtable.get_fulfillment_zones()
                        existing = [
                            z
                            for z in existing_zones
                            if z.get("zip_prefix") == zone_data["zip_prefix"]
                        ]

                        zone_record = {
                            "zip_prefix": zone_data["zip_prefix"],
                            "zone": zone_data["zone"],
                        }

                        if existing:
                            logger.info(
                                f"Zone for zip prefix {zone_data['zip_prefix']} already exists, updating"
                            )
                            try:
                                record = self.airtable.update_fulfillment_zone(
                                    existing[0]["airtable_id"], zone_record
                                )
                                created_records.append(record)
                            except Exception as e:
                                logger.error(f"Failed to update fulfillment zone: {e}")
                        else:
                            try:
                                record = self.airtable.create_fulfillment_zone(zone_record)
                                created_records.append(record)
                                logger.info(
                                    f"Created zone for zip prefix {zone_data['zip_prefix']}"
                                )
                            except Exception as e:
                                logger.error(f"Failed to create fulfillment zone: {e}")

                    # Pause between batches to avoid rate limits
                    if i + batch_size < total:
                        time.sleep(1)

            logger.info(
                f"Fulfillment zones migration complete. Processed {len(created_records)} zone records"
            )
            return created_records

        except Exception as e:
            logger.error(f"Failed to migrate fulfillment zones: {e}")
            raise

    def migrate_sku_mappings(self, file_path: Optional[str] = None) -> Dict[str, int]:
        """
        Migrate SKU mappings from JSON file to Airtable.

        Args:
            file_path (Optional[str]): Path to the SKU mappings JSON file.
                                      If None, uses the default path.

        Returns:
            Dict[str, int]: Dictionary with counts of created records by warehouse
        """
        if file_path is None:
            file_path = self.json_paths["sku_mappings"]

        logger.info(f"Migrating SKU mappings from {file_path}")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            stats = {"total": 0}

            # Process each warehouse
            for warehouse, warehouse_data in data.items():
                logger.info(f"Processing SKU mappings for warehouse: {warehouse}")
                stats[warehouse] = 0

                # Process regular SKUs
                all_skus = warehouse_data.get("all_skus", {})
                total_skus = len(all_skus)
                logger.info(f"Found {total_skus} SKUs to migrate for {warehouse}")

                # Process in batches to avoid rate limits
                batch_size = 10
                items = list(all_skus.items())

                for i in range(0, total_skus, batch_size):
                    batch = items[i : i + batch_size]
                    logger.info(
                        f"Processing batch {i//batch_size + 1}/{(total_skus-1)//batch_size + 1} ({len(batch)} records)"
                    )

                    for order_sku, mapping in batch:
                        # Create SKU mapping record
                        sku_data = {
                            "order_sku": order_sku,
                            "picklist_sku": mapping["picklist_sku"],
                            "actual_qty": mapping["actualqty"],
                            "total_pick_weight": mapping.get("Total_Pick_Weight"),
                            "pick_type": mapping.get("Pick_Type", ""),
                            "warehouse": warehouse,
                            "is_bundle": False,
                        }

                        try:
                            self.airtable.create_sku_mapping(sku_data)
                            stats[warehouse] += 1
                            stats["total"] += 1
                        except Exception as e:
                            logger.error(f"Failed to create SKU mapping for {order_sku}: {e}")

                    # Pause between batches to avoid rate limits
                    if i + batch_size < total_skus:
                        time.sleep(1)

                # Process bundles
                bundles = warehouse_data.get("bundles", {})
                total_bundles = len(bundles)
                logger.info(f"Found {total_bundles} bundles to migrate for {warehouse}")

                items = list(bundles.items())
                for i in range(0, total_bundles, batch_size):
                    batch = items[i : i + batch_size]
                    logger.info(
                        f"Processing bundle batch {i//batch_size + 1}/{(total_bundles-1)//batch_size + 1} ({len(batch)} records)"
                    )

                    for bundle_sku, components in batch:
                        # Create bundle SKU mapping record
                        bundle_data = {
                            "order_sku": bundle_sku,
                            "picklist_sku": components[0]["component_sku"] if components else "",
                            "actual_qty": 1.0,  # Bundle is counted as 1
                            "warehouse": warehouse,
                            "is_bundle": True,
                            "bundle_components": json.dumps(
                                components
                            ),  # Store components as JSON string
                        }

                        try:
                            self.airtable.create_sku_mapping(bundle_data)
                            stats[warehouse] += 1
                            stats["total"] += 1
                        except Exception as e:
                            logger.error(
                                f"Failed to create bundle SKU mapping for {bundle_sku}: {e}"
                            )

                    # Pause between batches to avoid rate limits
                    if i + batch_size < total_bundles:
                        time.sleep(1)

            logger.info(f"Migration complete. Created {stats['total']} SKU mapping records")
            return stats

        except Exception as e:
            logger.error(f"Failed to migrate SKU mappings: {e}")
            raise

    def migrate_priority_tags(self) -> List[Dict[str, Any]]:
        """
        Migrate priority tags to Airtable.

        Returns:
            List[Dict[str, Any]]: List of created Airtable records
        """
        logger.info("Migrating priority tags to Airtable")

        # Default priority tags to create
        priority_tags = [{"name": "High"}, {"name": "Medium"}, {"name": "Low"}]

        created_records = []

        try:
            # Check existing tags to avoid duplicates
            existing_tags = self.airtable.get_priority_tags()
            existing_tag_names = [tag.get("name") for tag in existing_tags]

            for tag in priority_tags:
                if tag["name"] in existing_tag_names:
                    logger.info(f"Priority tag '{tag['name']}' already exists, skipping")
                    continue

                try:
                    record = self.airtable.create_priority_tag(tag)
                    created_records.append(record)
                    logger.info(f"Created priority tag '{tag['name']}'")
                except Exception as e:
                    logger.error(f"Failed to create priority tag '{tag['name']}': {e}")

            logger.info(f"Priority tags migration complete. Created {len(created_records)} tags")
            return created_records

        except Exception as e:
            logger.error(f"Failed to migrate priority tags: {e}")
            raise

    def migrate_all_data(self) -> Dict[str, Any]:
        """
        Migrate all data from JSON files to Airtable.

        Returns:
            Dict[str, Any]: Statistics about the migration
        """
        logger.info("Starting full data migration to Airtable")

        stats = {}

        try:
            # Migrate fulfillment centers first
            fc_records = self.migrate_fulfillment_centers()
            stats["fulfillment_centers"] = len(fc_records)

            # Then migrate fulfillment zones
            fz_records = self.migrate_fulfillment_zones()
            stats["fulfillment_zones"] = len(fz_records)

            # Then migrate delivery services
            ds_records = self.migrate_delivery_services()
            stats["delivery_services"] = len(ds_records)

            # Then migrate priority tags
            pt_records = self.migrate_priority_tags()
            stats["priority_tags"] = len(pt_records)

            # Finally migrate SKU mappings
            sku_stats = self.migrate_sku_mappings()
            stats["sku_mappings"] = sku_stats

            logger.info(f"Full data migration complete. Results: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to migrate data: {e}")
            raise
            results["delivery_services"] = self.migrate_delivery_services()
        except Exception as e:
            logger.error(f"Failed to migrate delivery services: {e}")
            results["delivery_services"] = {"error": str(e)}

        # Migrate SKU mappings
        try:
            results["sku_mappings"] = self.migrate_sku_mappings()
        except Exception as e:
            logger.error(f"Failed to migrate SKU mappings: {e}")
            results["sku_mappings"] = {"error": str(e)}

        return results

    # Uncomment the operation you want to perform:

    # Migrate all data
    # stats = migrator.migrate_all_data()
    # print(f"Migration complete: {stats}")

    # Or migrate specific data types:
    # migrator.migrate_fulfillment_centers()
    # migrator.migrate_delivery_services()
    # migrator.migrate_sku_mappings()
