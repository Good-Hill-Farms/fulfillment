import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from utils.airtable_handler import AirtableHandler


def now():
    return datetime.now(ZoneInfo("UTC"))


# --- Data Models Based on Actual Usage in data_processor.py ---


class OrderItem(BaseModel):
    """Represents a single order line item as processed by the system"""

    order_id: str = Field(alias="order id")
    external_order_id: str = Field(alias="externalorderid", default="")
    order_number: str = Field(alias="ordernumber", default="")
    customer_first_name: str = Field(alias="CustomerFirstName", default="")
    customer_last_name: str = Field(alias="customerLastname", default="")
    customer_email: str = Field(alias="customeremail", default="")
    ship_to_name: str = Field(alias="shiptoname", default="")
    ship_to_zip: str = Field(alias="shiptopostalcode", default="")
    shopify_sku: str = Field(alias="shopifysku2", default="")
    shop_sku: str = Field(alias="shopsku", default="")
    shop_quantity: float = Field(alias="shopquantity", default=1)
    warehouse_sku: str = Field(alias="sku", default="")
    actual_qty: float = Field(alias="actualqty", default=1)
    total_pick_weight: float = Field(alias="Total Pick Weight", default=0)
    quantity: float = Field(default=1)
    starting_balance: float = Field(alias="Starting Balance", default=0)
    transaction_quantity: float = Field(alias="Transaction Quantity", default=0)
    ending_balance: float = Field(alias="Ending Balance", default=0)
    fulfillment_center: str = Field(alias="Fulfillment Center", default="")
    issues: str = Field(alias="Issues", default="")
    tags: str = Field(alias="Tags", default="")

    class Config:
        populate_by_name = True


class InventoryItem(BaseModel):
    """Represents an inventory item as used in the system"""

    warehouse_name: str = Field(alias="WarehouseName")
    item_id: Optional[str] = Field(alias="ItemId", default="")
    sku: str = Field(alias="Sku")
    name: str = Field(alias="Name", default="")
    type: str = Field(alias="Type", default="")
    available_qty: float = Field(alias="AvailableQty", default=0)
    balance: float = Field(alias="Balance", default=0)

    class Config:
        populate_by_name = True


class BundleComponent(BaseModel):
    """Represents a component within a bundle as actually used by google_sheets.py and data_processor.py"""

    component_sku: str  # The inventory SKU from 'picklist sku' column
    actualqty: float  # Quantity per bundle from 'Mix Quantity' column
    weight: Optional[float] = 0.0  # From 'Pick Weight LB' column
    pick_type: Optional[str] = ""  # From 'Pick Type' column
    pick_type_inventory: Optional[str] = ""  # From 'Product Type' column

    # Legacy field names that might still be used in some parts of the code
    qty: Optional[float] = None  # Alternative name for actualqty

    def __init__(self, **data):
        # Handle legacy field name mapping
        if "qty" in data and "actualqty" not in data:
            data["actualqty"] = data["qty"]
        elif "actualqty" in data and "qty" not in data:
            data["qty"] = data["actualqty"]
        super().__init__(**data)


class SKUSingle(BaseModel):
    """Represents a single SKU mapping as actually used by google_sheets.py"""

    picklist_sku: str  # Maps to warehouse inventory SKU
    actualqty: float  # Quantity multiplier
    total_pick_weight: float  # From 'Total Pick Weight' column
    pick_type: Optional[str] = ""  # From 'Pick Type' column
    pick_type_inventory: Optional[str] = ""  # From 'Product Type' column


class SKUMappings(BaseModel):
    """Complete SKU mappings structure as used in data_processor.py"""

    oxnard_singles: Dict[str, SKUSingle] = Field(default_factory=dict, alias="Oxnard.singles")
    oxnard_bundles: Dict[str, List[BundleComponent]] = Field(
        default_factory=dict, alias="Oxnard.bundles"
    )
    wheeling_singles: Dict[str, SKUSingle] = Field(default_factory=dict, alias="Wheeling.singles")
    wheeling_bundles: Dict[str, List[BundleComponent]] = Field(
        default_factory=dict, alias="Wheeling.bundles"
    )
    walnut_singles: Dict[str, SKUSingle] = Field(default_factory=dict, alias="Walnut.singles")
    walnut_bundles: Dict[str, List[BundleComponent]] = Field(
        default_factory=dict, alias="Walnut.bundles"
    )
    northlake_singles: Dict[str, SKUSingle] = Field(default_factory=dict, alias="Northlake.singles")
    northlake_bundles: Dict[str, List[BundleComponent]] = Field(
        default_factory=dict, alias="Northlake.bundles"
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SKUMappings":
        """Create from the actual JSON structure used in the system"""
        oxnard = data.get("Oxnard", {})
        wheeling = data.get("Wheeling", {})
        walnut = data.get("Walnut", {})
        northlake = data.get("Northlake", {})

        return cls(
            oxnard_singles={k: SKUSingle(**v) for k, v in oxnard.get("singles", {}).items()},
            oxnard_bundles={
                k: [BundleComponent(**comp) for comp in v]
                for k, v in oxnard.get("bundles", {}).items()
            },
            wheeling_singles={k: SKUSingle(**v) for k, v in wheeling.get("singles", {}).items()},
            wheeling_bundles={
                k: [BundleComponent(**comp) for comp in v]
                for k, v in wheeling.get("bundles", {}).items()
            },
            walnut_singles={k: SKUSingle(**v) for k, v in walnut.get("singles", {}).items()},
            walnut_bundles={
                k: [BundleComponent(**comp) for comp in v]
                for k, v in walnut.get("bundles", {}).items()
            },
            northlake_singles={k: SKUSingle(**v) for k, v in northlake.get("singles", {}).items()},
            northlake_bundles={
                k: [BundleComponent(**comp) for comp in v]
                for k, v in northlake.get("bundles", {}).items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the JSON structure expected by the system"""
        return {
            "Oxnard": {
                "singles": {k: v.dict() for k, v in self.oxnard_singles.items()},
                "bundles": {k: [comp.dict() for comp in v] for k, v in self.oxnard_bundles.items()},
            },
            "Wheeling": {
                "singles": {k: v.dict() for k, v in self.wheeling_singles.items()},
                "bundles": {k: [comp.dict() for comp in v] for k, v in self.wheeling_bundles.items()},
            },
            "Walnut": {
                "singles": {k: v.dict() for k, v in self.walnut_singles.items()},
                "bundles": {k: [comp.dict() for comp in v] for k, v in self.walnut_bundles.items()},
            },
            "Northlake": {
                "singles": {k: v.dict() for k, v in self.northlake_singles.items()},
                "bundles": {k: [comp.dict() for comp in v] for k, v in self.northlake_bundles.items()},
            },
        }


class ShortageItem(BaseModel):
    """Represents a shortage item as generated by the system"""

    component_sku: str
    shopify_sku: str
    order_id: str
    current_balance: float
    needed_qty: float
    shortage_qty: float
    fulfillment_center: str


class ShippingZone(BaseModel):
    """Represents shipping zone mapping as used in the system"""

    zip_prefix: str
    zone: int
    warehouse: Literal["Oxnard", "Wheeling", "Walnut", "Northlake"]


class ShippingZones(BaseModel):
    """Complete shipping zones structure"""

    oxnard_zones: Dict[str, int] = Field(default_factory=dict)
    wheeling_zones: Dict[str, int] = Field(default_factory=dict)
    walnut_zones: Dict[str, int] = Field(default_factory=dict)
    northlake_zones: Dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShippingZones":
        """Create from the actual JSON structure used in the system"""
        return cls(
            oxnard_zones=data.get("Oxnard", {}), 
            wheeling_zones=data.get("Wheeling", {}),
            walnut_zones=data.get("Walnut", {}),
            northlake_zones=data.get("Northlake", {})
        )


class DeliveryService(BaseModel):
    """Represents delivery service mapping"""

    destination_zip_short: str
    origin: str
    carrier_name: str
    service_name: str
    days: int


class ProcessingStats(BaseModel):
    """Statistics generated by the processing system"""

    total_orders: int = 0
    total_line_items: int = 0
    unique_skus: int = 0
    fulfillment_center_distribution: Dict[str, int] = Field(default_factory=dict)
    primary_fulfillment_center: str = ""
    items_with_issues: int = 0
    issue_rate: float = 0.0
    total_quantity_processed: float = 0.0
    avg_quantity_per_item: float = 0.0
    max_quantity_item: float = 0.0
    total_shortages: int = 0
    unique_shortage_skus: int = 0
    total_shortage_quantity: float = 0.0
    shortages_by_fulfillment_center: Dict[str, int] = Field(default_factory=dict)
    total_inventory_items: int = 0
    total_inventory_balance: float = 0.0
    zero_balance_items: int = 0
    low_balance_items: int = 0
    processing_timestamp: str = ""


class WarehousePerformance(BaseModel):
    """Warehouse performance metrics"""

    total_orders: int = 0
    total_line_items: int = 0
    unique_skus: int = 0
    items_with_issues: int = 0
    issue_rate: float = 0.0
    total_quantity: float = 0.0
    avg_quantity_per_item: float = 0.0
    bundle_items: int = 0
    bundle_rate: float = 0.0
    inventory_balance: float = 0.0
    inventory_sku_count: int = 0


class StagingOrder(BaseModel):
    """Represents an order in staging"""

    order_item: OrderItem
    staged_at: datetime = Field(default_factory=now)


class InventoryComparison(BaseModel):
    """Represents inventory comparison data"""

    sku: str = Field(alias="SKU")
    warehouse: str = Field(alias="Warehouse")
    initial_balance: float = Field(alias="Initial Balance")
    current_balance: float = Field(alias="Current Balance")
    difference: float = Field(alias="Difference")
    is_used_in_bundle: bool = Field(alias="Is Used In Bundle")

    class Config:
        populate_by_name = True


# --- Legacy Models for Airtable Integration ---


class DataBatch(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    batch_type: Literal["order", "inventory"]
    source_filename: str
    uploaded_by: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=now)


class OrderLine(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    order_id: str
    zip_code: str
    sku: str
    quantity: float
    is_bundle: bool
    mapped_picklist_sku: Optional[str] = None
    batch_id: uuid.UUID
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None


class SKUMapping(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    order_sku: str
    picklist_sku: str
    actual_qty: float
    total_pick_weight: Optional[float]
    pick_type: str
    bundle_components: Optional[str] = None  # JSON string of bundle components
    fulfillment_center: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None

    def get_bundle_components(self) -> List[BundleComponent]:
        """Parse the bundle_components JSON string into a list of BundleComponent objects"""
        if not self.bundle_components:
            return []

        try:
            components_data = json.loads(self.bundle_components)
            return [BundleComponent(**comp) for comp in components_data]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Error parsing bundle components: {e}")
            return []

    def set_bundle_components(self, components: List[BundleComponent]) -> None:
        """Convert a list of BundleComponent objects to a JSON string"""
        if not components:
            self.bundle_components = None
            return

        components_data = [comp.dict() for comp in components]
        self.bundle_components = json.dumps(components_data)

    @classmethod
    def from_airtable(cls, record: Dict[str, Any]) -> "SKUMapping":
        """Create a SKUMapping instance from an Airtable record"""
        mapping_data = {
            "id": uuid.UUID(record.get("id", str(uuid.uuid4()))),
            "order_sku": record.get("order_sku", ""),
            "picklist_sku": record.get("picklist_sku", ""),
            "actual_qty": float(record.get("actual_qty", 0)),
            "total_pick_weight": float(record.get("total_pick_weight", 0))
            if record.get("total_pick_weight")
            else None,
            "pick_type": record.get("pick_type", ""),
            "bundle_components": record.get("bundle_components"),
            "fulfillment_center": record.get("fulfillment_center"),
            "created_at": record.get("created_at", now()),
            "updated_at": record.get("updated_at"),
        }

        return cls(**mapping_data)

    def to_airtable(self) -> Dict[str, Any]:
        """Convert the SKUMapping instance to an Airtable record format"""
        return {
            "order_sku": self.order_sku,
            "picklist_sku": self.picklist_sku,
            "actual_qty": self.actual_qty,
            "total_pick_weight": self.total_pick_weight,
            "pick_type": self.pick_type,
            "bundle_components": self.bundle_components,
            "fulfillment_center": self.fulfillment_center,
        }


class ZoneMapping(BaseModel):
    zip_prefix: str
    zone: int


class FulfillmentZone(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    zip_prefix: str
    zone: str
    fulfillment_center: List[str]
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None

    @classmethod
    def from_airtable(cls, record: Dict[str, Any]) -> "FulfillmentZone":
        """Create a FulfillmentZone instance from an Airtable record"""
        return cls(
            id=uuid.UUID(record.get("id", str(uuid.uuid4()))),
            zip_prefix=record.get("zip_prefix", ""),
            zone=str(record.get("zone", "")),
            fulfillment_center=record.get("fulfillment_center", []),
            created_at=record.get("created_at", now()),
            updated_at=record.get("updated_at"),
        )

    def to_airtable(self) -> Dict[str, Any]:
        """Convert the FulfillmentZone instance to an Airtable record format"""
        return {
            "zip_prefix": self.zip_prefix,
            "zone": self.zone,
            "fulfillment_center": self.fulfillment_center,
        }


class FulfillmentCenter(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: Literal["Oxnard", "Wheeling", "Walnut", "Northlake"]
    zip_code: str
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None


class FulfillmentPlanLine(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    order_id: str
    sku: str
    picklist_sku: str
    warehouse: str
    assigned_qty: float
    status: Literal["fulfilled", "partial", "shortage"]
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None


class ShortageLog(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    order_id: str
    sku: str
    requested_qty: float
    available_qty: float
    missing_qty: float
    timestamp: datetime
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None


class PriorityTag(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None


class FulfillmentRule(BaseModel):
    """Editable business rules for fulfillment logic"""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    rule_name: str
    rule_type: Literal[
        "inventory_threshold", "shipping_priority", "bundle_substitution", "zone_override", "custom"
    ]
    rule_condition: str  # JSON string describing the condition
    rule_action: str  # JSON string describing the action
    priority: int = 100  # Lower numbers = higher priority
    is_active: bool = True
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None

    @classmethod
    def from_airtable(cls, record: Dict[str, Any]) -> "FulfillmentRule":
        """Create a FulfillmentRule instance from an Airtable record"""
        return cls(
            id=uuid.UUID(record.get("id", str(uuid.uuid4()))),
            rule_name=record.get("rule_name", ""),
            rule_type=record.get("rule_type", "custom"),
            rule_condition=record.get("rule_condition", ""),
            rule_action=record.get("rule_action", ""),
            priority=int(record.get("priority", 100)),
            is_active=bool(record.get("is_active", True)),
            description=record.get("description"),
            created_at=record.get("created_at", now()),
            updated_at=record.get("updated_at"),
        )

    def to_airtable(self) -> Dict[str, Any]:
        """Convert the FulfillmentRule instance to an Airtable record format"""
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type,
            "rule_condition": self.rule_condition,
            "rule_action": self.rule_action,
            "priority": self.priority,
            "is_active": self.is_active,
            "description": self.description,
        }

    def get_condition(self) -> Dict[str, Any]:
        """Parse the rule_condition JSON string"""
        try:
            return json.loads(self.rule_condition) if self.rule_condition else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def get_action(self) -> Dict[str, Any]:
        """Parse the rule_action JSON string"""
        try:
            return json.loads(self.rule_action) if self.rule_action else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def set_condition(self, condition: Dict[str, Any]) -> None:
        """Set the rule condition from a dictionary"""
        self.rule_condition = json.dumps(condition)

    def set_action(self, action: Dict[str, Any]) -> None:
        """Set the rule action from a dictionary"""
        self.rule_action = json.dumps(action)


# --- Schema Manager (Airtable Integration) ---
class SchemaManager:
    """Manager class for handling Airtable-based schemas and rules"""

    def __init__(self):
        self.airtable = AirtableHandler()
        self._cache = {}
        self._cache_expiry = {}

    def get_sku_mappings(
        self, warehouse: Optional[str] = None, use_cache: bool = True
    ) -> List[SKUMapping]:
        """Get SKU mappings from Airtable"""
        cache_key = f"sku_mappings_{warehouse or 'all'}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            records = self.airtable.get_sku_mappings(warehouse)
            mappings = [SKUMapping.from_airtable(record) for record in records]

            if use_cache:
                self._cache[cache_key] = mappings

            return mappings
        except Exception as e:
            print(f"Error fetching SKU mappings: {e}")
            return []

    def get_fulfillment_zones(self, use_cache: bool = True) -> List[FulfillmentZone]:
        """Get fulfillment zones from Airtable"""
        cache_key = "fulfillment_zones"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            records = self.airtable.get_fulfillment_zones()
            zones = []
            for record in records:
                zones.append(
                    FulfillmentZone(
                        zip_prefix=record.get("zip_prefix", ""),
                        zone=str(record.get("zone", "")),
                        fulfillment_center=record.get("fulfillment_center", []),
                    )
                )

            if use_cache:
                self._cache[cache_key] = zones

            return zones
        except Exception as e:
            print(f"Error fetching fulfillment zones: {e}")
            return []

    def get_delivery_services(
        self, zip_prefix: Optional[str] = None, use_cache: bool = True
    ) -> List[DeliveryService]:
        """Get delivery services from Airtable"""
        cache_key = f"delivery_services_{zip_prefix or 'all'}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            records = self.airtable.get_delivery_services(zip_prefix)
            services = [DeliveryService(**record) for record in records]

            if use_cache:
                self._cache[cache_key] = services

            return services
        except Exception as e:
            print(f"Error fetching delivery services: {e}")
            return []

    def get_fulfillment_rules(
        self, rule_type: Optional[str] = None, active_only: bool = True
    ) -> List[FulfillmentRule]:
        """Get fulfillment rules from Airtable"""
        cache_key = f"rules_{rule_type or 'all'}_{active_only}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # For now, return empty list since we need to implement this in AirtableHandler
            rules = []
            self._cache[cache_key] = rules
            return rules
        except Exception as e:
            print(f"Error fetching fulfillment rules: {e}")
            return []

    def clear_cache(self):
        """Clear the schema cache"""
        self._cache.clear()
        self._cache_expiry.clear()

    def get_zip_to_zone_mapping(self) -> Dict[str, Dict[str, int]]:
        """Get zip to zone mapping organized by fulfillment center"""
        zones = self.get_fulfillment_zones()
        mapping = {"Oxnard": {}, "Wheeling": {}, "Walnut": {}, "Northlake": {}}

        for zone in zones:
            for fc in zone.fulfillment_center:
                if fc in mapping:
                    try:
                        mapping[fc][zone.zip_prefix] = int(zone.zone)
                    except (ValueError, TypeError):
                        pass

        return mapping
