import uuid
import json
from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from zoneinfo import ZoneInfo
from utils.airtable_handler import AirtableHandler


def now():
    return datetime.now(ZoneInfo("UTC"))


# --- Batch Upload Tracker ---
class DataBatch(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    batch_type: Literal["order", "inventory"]
    source_filename: str
    uploaded_by: Optional[str] = None
    note: Optional[str] = None
    created_at: datetime = Field(default_factory=now)


# --- Core Entities ---
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


class InventoryItem(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    sku: str
    warehouse_name: Literal["Oxnard", "Wheeling"]
    available_qty: float
    batch_id: uuid.UUID
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None


# --- SKU Mapping and Bundles ---
class BundleComponent(BaseModel):
    sku: str
    qty: float


class SKUMapping(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    order_sku: str
    picklist_sku: str
    actual_qty: float
    total_pick_weight: Optional[float]
    pick_type: str
    # In Airtable, bundle_components is stored as a JSON string
    # When reading from Airtable, we need to parse this string back to a list
    bundle_components: Optional[str] = None  # JSON string of bundle components
    fulfillment_center: Optional[List[str]] = None  # Link to fulfillment center using array format for Airtable
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
        
        components_data = [{
            "sku": comp.sku,
            "qty": comp.qty
        } for comp in components]
        
        self.bundle_components = json.dumps(components_data)
    
    @classmethod
    def from_airtable(cls, record: Dict[str, Any]) -> 'SKUMapping':
        """Create a SKUMapping instance from an Airtable record"""
        # Map Airtable field names to model field names if needed
        mapping_data = {
            "id": uuid.UUID(record.get("id", str(uuid.uuid4()))),
            "order_sku": record.get("order_sku", ""),
            "picklist_sku": record.get("picklist_sku", ""),
            "actual_qty": float(record.get("actual_qty", 0)),
            "total_pick_weight": float(record.get("total_pick_weight", 0)) if record.get("total_pick_weight") else None,
            "pick_type": record.get("pick_type", ""),
            "bundle_components": record.get("bundle_components"),
            "fulfillment_center": record.get("fulfillment_center"),
            "created_at": record.get("created_at", now()),
            "updated_at": record.get("updated_at")
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
            "fulfillment_center": self.fulfillment_center
        }


# --- Shipping Zones ---
class ZoneMapping(BaseModel):
    zip_prefix: str
    zone: int

class FulfillmentZone(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    zip_prefix: str
    zone: str  # String type to match Airtable requirements
    fulfillment_center: List[str]  # Link to FulfillmentCenter using array format for Airtable
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_airtable(cls, record: Dict[str, Any]) -> 'FulfillmentZone':
        """Create a FulfillmentZone instance from an Airtable record"""
        return cls(
            id=uuid.UUID(record.get("id", str(uuid.uuid4()))),
            zip_prefix=record.get("zip_prefix", ""),
            zone=str(record.get("zone", "")),
            fulfillment_center=record.get("fulfillment_center", []),
            created_at=record.get("created_at", now()),
            updated_at=record.get("updated_at")
        )
    
    def to_airtable(self) -> Dict[str, Any]:
        """Convert the FulfillmentZone instance to an Airtable record format"""
        return {
            "zip_prefix": self.zip_prefix,
            "zone": self.zone,
            "fulfillment_center": self.fulfillment_center
        }


class FulfillmentCenter(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: Literal["Oxnard", "Wheeling"]
    zip_code: str
    # Zones are now stored in a separate table (FulfillmentZone)
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None


# --- Delivery Services ---
class DeliveryService(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    destination_zip_short: str
    origin: str
    carrier_name: str
    service_name: str
    days: int
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_airtable(cls, record: Dict[str, Any]) -> 'DeliveryService':
        """Create a DeliveryService instance from an Airtable record"""
        return cls(
            id=uuid.UUID(record.get("id", str(uuid.uuid4()))),
            destination_zip_short=record.get("destination_zip_short", ""),
            origin=record.get("origin", ""),
            carrier_name=record.get("carrier_name", ""),
            service_name=record.get("service_name", ""),
            days=int(record.get("days", 0)),
            created_at=record.get("created_at", now()),
            updated_at=record.get("updated_at")
        )
    
    def to_airtable(self) -> Dict[str, Any]:
        """Convert the DeliveryService instance to an Airtable record format"""
        return {
            "destination_zip_short": self.destination_zip_short,
            "origin": self.origin,
            "carrier_name": self.carrier_name,
            "service_name": self.service_name,
            "days": self.days
        }


# --- Fulfillment Output ---
class FulfillmentPlanLine(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    order_id: str
    sku: str
    picklist_sku: str
    warehouse: str
    assigned_qty: float
    status: Literal['fulfilled', 'partial', 'shortage']
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


# --- Business Rules (Editable) ---
class FulfillmentRule(BaseModel):
    """Editable business rules for fulfillment logic"""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    rule_name: str
    rule_type: Literal["inventory_threshold", "shipping_priority", "bundle_substitution", "zone_override", "custom"]
    rule_condition: str  # JSON string describing the condition
    rule_action: str     # JSON string describing the action
    priority: int = 100  # Lower numbers = higher priority
    is_active: bool = True
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=now)
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_airtable(cls, record: Dict[str, Any]) -> 'FulfillmentRule':
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
            updated_at=record.get("updated_at")
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
            "description": self.description
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
    
    def get_sku_mappings(self, warehouse: Optional[str] = None, use_cache: bool = True) -> List[SKUMapping]:
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
                zones.append(FulfillmentZone(
                    zip_prefix=record.get("zip_prefix", ""),
                    zone=str(record.get("zone", "")),
                    fulfillment_center=record.get("fulfillment_center", [])
                ))
            
            if use_cache:
                self._cache[cache_key] = zones
            
            return zones
        except Exception as e:
            print(f"Error fetching fulfillment zones: {e}")
            return []
    
    def get_delivery_services(self, zip_prefix: Optional[str] = None, use_cache: bool = True) -> List[DeliveryService]:
        """Get delivery services from Airtable"""
        cache_key = f"delivery_services_{zip_prefix or 'all'}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            records = self.airtable.get_delivery_services(zip_prefix)
            services = [DeliveryService.from_airtable(record) for record in records]
            
            if use_cache:
                self._cache[cache_key] = services
            
            return services
        except Exception as e:
            print(f"Error fetching delivery services: {e}")
            return []
    
    def get_fulfillment_rules(self, rule_type: Optional[str] = None, active_only: bool = True) -> List[FulfillmentRule]:
        """Get fulfillment rules from Airtable"""
        cache_key = f"rules_{rule_type or 'all'}_{active_only}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Note: This assumes we have a FulfillmentRule table in Airtable
            # We'll need to add this to the AirtableHandler
            filter_formula = ""
            if rule_type:
                filter_formula += f"{{rule_type}} = '{rule_type}'"
            if active_only:
                active_filter = "{is_active} = TRUE()"
                filter_formula = f"AND({filter_formula}, {active_filter})" if filter_formula else active_filter
            
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
        mapping = {"Oxnard": {}, "Wheeling": {}}
        
        for zone in zones:
            for fc in zone.fulfillment_center:
                if fc in mapping:
                    try:
                        mapping[fc][zone.zip_prefix] = int(zone.zone)
                    except (ValueError, TypeError):
                        pass
        
        return mapping
