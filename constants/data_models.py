# constants/data_models.py

INVENTORY_COLUMNS = [
    "Fruit SKU",
    "Warehouse",
    "Batch ID",
    "Status",
    "Quantity",
    "Unit Type",
    "Delivery Date",
    "Vendor",
    "Projected Use",
    "Used In Orders",
    "Notes",
]

# Batch code components, can be used for parsing or validation
BATCH_CODE_COMPONENTS = [
    "product_name",
    "size",
    "vendor_initials",
    "vendor_name",
    "status",
    "delivery_datetime",
    "agg_date",
] 