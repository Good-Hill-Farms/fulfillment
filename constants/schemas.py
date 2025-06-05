import uuid
import json
from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from zoneinfo import ZoneInfo


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
