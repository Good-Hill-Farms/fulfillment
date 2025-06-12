#!/usr/bin/env python
"""
Script to migrate data from JSON files to Airtable.

This script uses the AirtableDataMigrator to populate Airtable with data from:
- delivery_services.json
- shipping_zones.json
- sku_mappings.json
"""

import logging
import os
import sys

from dotenv import load_dotenv

# Add project root to path to allow importing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.airtable_data_migrator import AirtableDataMigrator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the data migration"""
    # Load environment variables
    load_dotenv()

    # Check for required environment variables
    if not os.getenv("AIRTABLE_API_KEY") or not os.getenv("AIRTABLE_BASE_ID"):
        logger.error("Missing required environment variables: AIRTABLE_API_KEY, AIRTABLE_BASE_ID")
        sys.exit(1)

    logger.info("Starting data migration to Airtable")

    # Create migrator
    migrator = AirtableDataMigrator()

    try:
        # Migrate fulfillment centers
        logger.info("Migrating fulfillment centers...")
        fc_records = migrator.migrate_fulfillment_centers()
        logger.info(f"Successfully migrated {len(fc_records)} fulfillment centers")

        # Migrate fulfillment zones
        logger.info("Migrating fulfillment zones...")
        fz_records = migrator.migrate_fulfillment_zones()
        logger.info(f"Successfully migrated {len(fz_records)} fulfillment zones")

        # Migrate priority tags
        logger.info("Migrating priority tags...")
        pt_records = migrator.migrate_priority_tags()
        logger.info(f"Successfully migrated {len(pt_records)} priority tags")

        # Migrate delivery services
        logger.info("Migrating delivery services...")
        ds_records = migrator.migrate_delivery_services()
        logger.info(f"Successfully migrated {len(ds_records)} delivery services")

        # Migrate SKU mappings
        logger.info("Migrating SKU mappings...")
        sku_stats = migrator.migrate_sku_mappings()
        logger.info(f"Successfully migrated {sku_stats['total']} SKU mappings")

        logger.info("Data migration completed successfully!")
        logger.info(
            f"Summary: {len(fc_records)} fulfillment centers, {len(ds_records)} delivery services, {sku_stats['total']} SKU mappings"
        )

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
