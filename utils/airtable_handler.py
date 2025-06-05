"""
Airtable Handler Module

This module provides functionality to interact with Airtable for storing and retrieving 
zip code mapping data for fulfillment operations.
"""

import os
from typing import Dict, List, Optional, Union, Any
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AirtableHandler:
    """Handler for Airtable operations related to zip code mapping and fulfillment centers."""
    
    def __init__(self):
        """Initialize the AirtableHandler with API credentials from environment variables."""
        self.api_key = os.getenv('AIRTABLE_API_KEY')
        self.base_id = os.getenv('AIRTABLE_BASE_ID')
        
        if not self.api_key or not self.base_id:
            raise ValueError("Airtable API key or Base ID not found in environment variables")
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.base_url = f"https://api.airtable.com/v0/{self.base_id}"
        
        # Table names for Airtable
        # Using the table names instead of IDs for the Airtable base
        self.tables = {
            "fulfillment_center": "FulfillmentCenter",
            "fulfillment_zone": "FulfillmentZone",
            "delivery_service": "DeliveryService",
            "sku_mapping": "SKUMapping",
            "inventory_item": "InventoryItem",
            "data_batch": "DataBatch",
            "fulfillment_plan_line": "FulfillmentPlanLine",
            "shortage_log": "ShortageLog",
            "priority_tag": "PriorityTag",
            "fulfillment_rule": "FulfillmentRule"
        }
    
    def get_fulfillment_centers(self) -> List[Dict[str, Any]]:
        """
        Retrieve all fulfillment centers from Airtable.
        
        Returns:
            List[Dict[str, Any]]: List of fulfillment centers with their zones
        """
        table_name = "FulfillmentCenter"  # Use the actual table name instead of the key
        url = f"{self.base_url}/{table_name}"
        
        all_records = []
        offset = None
        
        # Loop to handle pagination
        while True:
            # Add offset parameter if we have one from a previous request
            params = {}
            if offset:
                params['offset'] = offset
                
            # Make the request
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                all_records.extend([self._format_record(record) for record in data.get('records', [])])
                
                # Check if there are more records to fetch
                offset = data.get('offset')
                if not offset:
                    break  # No more pages
            else:
                raise Exception(f"Failed to fetch fulfillment centers: {response.text}")
        
        return all_records
    
    def get_delivery_services(self, zip_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve delivery services from Airtable, optionally filtered by zip prefix.
        
        Args:
            zip_prefix (Optional[str]): Filter by destination zip prefix
            
        Returns:
            List[Dict[str, Any]]: List of delivery service records
        """
        table_name = "DeliveryService"
        url = f"{self.base_url}/{table_name}"
        
        # Prepare parameters
        params = {}
        
        # Add filter if zip_prefix is provided
        if zip_prefix:
            filter_formula = f"FIND('{zip_prefix}', {{destination_zip_short}}) = 1"
            params['filterByFormula'] = filter_formula
        
        all_records = []
        offset = None
        
        # Loop to handle pagination
        while True:
            # Add offset parameter if we have one from a previous request
            if offset:
                params['offset'] = offset
                
            # Make the request
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                all_records.extend([self._format_record(record) for record in data.get('records', [])])
                
                # Check if there are more records to fetch
                offset = data.get('offset')
                if not offset:
                    break  # No more pages
            else:
                raise Exception(f"Failed to fetch delivery services: {response.text}")
        
        return all_records
    
    def get_fulfillment_zones(self) -> List[Dict[str, Any]]:
        """
        Retrieve all fulfillment zones from Airtable.
        
        Returns:
            List[Dict[str, Any]]: List of fulfillment zones
        """
        table_name = "FulfillmentZone"  # Use the actual table name directly
        url = f"{self.base_url}/{table_name}"
        
        all_records = []
        offset = None
        
        # Loop to handle pagination
        while True:
            # Add offset parameter if we have one from a previous request
            params = {}
            if offset:
                params['offset'] = offset
                
            # Make the request
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                all_records.extend([self._format_record(record) for record in data.get('records', [])])
                
                # Check if there are more records to fetch
                offset = data.get('offset')
                if not offset:
                    break  # No more pages
            else:
                raise Exception(f"Failed to fetch fulfillment zones: {response.text}")
        
        return all_records
    
    def create_fulfillment_zone(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new fulfillment zone record in Airtable.
        
        Args:
            data (Dict[str, Any]): Fulfillment zone data
            
        Returns:
            Dict[str, Any]: Created record
        """
        table_name = "FulfillmentZone"  # Use the actual table name directly
        
        # Ensure zone is a string
        if "zone" in data and not isinstance(data["zone"], str):
            data["zone"] = str(data["zone"])
            
        response = requests.post(
            f"{self.base_url}/{table_name}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to create fulfillment zone: {response.text}")
    
    def update_fulfillment_zone(self, record_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a fulfillment zone record in Airtable.
        
        Args:
            record_id (str): Airtable record ID
            data (Dict[str, Any]): Data to update
            
        Returns:
            Dict[str, Any]: Updated record
        """
        table_name = "FulfillmentZone"  # Use the actual table name directly
        response = requests.patch(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to update fulfillment zone: {response.text}")
    
    def delete_fulfillment_zone(self, record_id: str) -> bool:
        """
        Delete a fulfillment zone record from Airtable.
        
        Args:
            record_id (str): Airtable record ID
            
        Returns:
            bool: True if deletion was successful
        """
        table_name = "FulfillmentZone"  # Use the actual table name directly
        response = requests.delete(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return True
        else:
            raise Exception(f"Failed to delete fulfillment zone: {response.text}")
    
    def get_zip_to_zone_mapping(self) -> Dict[str, int]:
        """
        Create a mapping of zip prefixes to zones from fulfillment zones.
        
        Returns:
            Dict[str, int]: Dictionary mapping zip prefixes to zone numbers
        """
        fulfillment_zones = self.get_fulfillment_zones()
        zip_to_zone = {}
        
        for zone in fulfillment_zones:
            if 'zip_prefix' in zone and 'zone' in zone:
                zip_to_zone[zone['zip_prefix']] = zone['zone']
        
        return zip_to_zone
    
    def update_fulfillment_center(self, record_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a fulfillment center record in Airtable.
        
        Args:
            record_id (str): Airtable record ID
            data (Dict[str, Any]): Data to update
            
        Returns:
            Dict[str, Any]: Updated record
        """
        table_name = "FulfillmentCenter"
        response = requests.patch(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to update fulfillment center: {response.text}")
    
    def create_fulfillment_center(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new fulfillment center record in Airtable.
        
        Args:
            data (Dict[str, Any]): Fulfillment center data
            
        Returns:
            Dict[str, Any]: Created record
        """
        table_name = "FulfillmentCenter"
        response = requests.post(
            f"{self.base_url}/{table_name}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to create fulfillment center: {response.text}")
    
    def get_best_fulfillment_center(self, destination_zip: str) -> Dict[str, Any]:
        """
        Determine the best fulfillment center for a given destination zip code.
        
        Args:
            destination_zip (str): Destination zip code
            
        Returns:
            Dict[str, Any]: Best fulfillment center record
        """
        # Get the first 3 digits of the zip code for matching
        zip_prefix = destination_zip[:3]
        
        # Get all delivery services for this zip prefix
        delivery_services = self.get_delivery_services(zip_prefix)
        
        if not delivery_services:
            # If no specific delivery service found, get all fulfillment centers
            fulfillment_centers = self.get_fulfillment_centers()
            # Default to the first one if available
            return fulfillment_centers[0] if fulfillment_centers else None
        
        # Find the delivery service with the shortest delivery time
        best_service = min(delivery_services, key=lambda x: x.get('days', float('inf')))
        
        # Get the origin fulfillment center
        origin = best_service.get('origin')
        
        # Find the fulfillment center matching this origin
        fulfillment_centers = self.get_fulfillment_centers()
        for fc in fulfillment_centers:
            if fc.get('name') == origin:
                return fc
        
        # Default to the first fulfillment center if no match found
        return fulfillment_centers[0] if fulfillment_centers else None
    
    def _format_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format an Airtable record by merging fields with id and created time.
        
        Args:
            record (Dict[str, Any]): Raw Airtable record
            
        Returns:
            Dict[str, Any]: Formatted record
        """
        result = record.get('fields', {})
        result['airtable_id'] = record.get('id')
        result['created_time'] = record.get('createdTime')
        return result


    # SKU Mapping methods
    def get_sku_mappings(self, warehouse: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve SKU mappings from Airtable, optionally filtered by warehouse.
        
        Args:
            warehouse (Optional[str]): Filter by warehouse name (e.g., 'Oxnard', 'Wheeling')
            
        Returns:
            List[Dict[str, Any]]: List of SKU mapping records
        """
        table_name = "SKUMapping"  # Use the actual table name directly
        url = f"{self.base_url}/{table_name}"
        
        # Prepare parameters
        params = {}
        
        # Add filter if warehouse is provided
        if warehouse:
            filter_formula = f"{{warehouse}} = '{warehouse}'"
            params['filterByFormula'] = filter_formula
        
        all_records = []
        offset = None
        
        # Loop to handle pagination
        while True:
            # Add offset parameter if we have one from a previous request
            if offset:
                params['offset'] = offset
                
            # Make the request
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                all_records.extend([self._format_record(record) for record in data.get('records', [])])
                
                # Check if there are more records to fetch
                offset = data.get('offset')
                if not offset:
                    break  # No more pages
            else:
                raise Exception(f"Failed to fetch SKU mappings: {response.text}")
        
        return all_records
    
    def create_sku_mapping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new SKU mapping record in Airtable.
        
        Args:
            data (Dict[str, Any]): SKU mapping data
            
        Returns:
            Dict[str, Any]: Created record
        """
        table_name = "SKUMapping"  # Use the actual table name directly
        response = requests.post(
            f"{self.base_url}/{table_name}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to create SKU mapping: {response.text}")
    
    def update_sku_mapping(self, record_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a SKU mapping record in Airtable.
        
        Args:
            record_id (str): Airtable record ID
            data (Dict[str, Any]): Data to update
            
        Returns:
            Dict[str, Any]: Updated record
        """
        table_name = "SKUMapping"  # Use the actual table name directly
        response = requests.patch(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to update SKU mapping: {response.text}")
    
    def delete_sku_mapping(self, record_id: str) -> bool:
        """
        Delete a SKU mapping record from Airtable.
        
        Args:
            record_id (str): Airtable record ID
            
        Returns:
            bool: True if deletion was successful
        """
        table_name = "SKUMapping"  # Use the actual table name directly
        response = requests.delete(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return True
        else:
            raise Exception(f"Failed to delete SKU mapping: {response.text}")

    # Delivery Services methods
    def get_delivery_services(self, zip_prefix: Optional[str] = None, origin: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve delivery services from Airtable, optionally filtered by zip prefix or origin.
        
        Args:
            zip_prefix (Optional[str]): Filter by destination zip prefix
            origin (Optional[str]): Filter by origin location
            
        Returns:
            List[Dict[str, Any]]: List of delivery service records
        """
        table_name = "DeliveryService"  # Use the actual table name directly
        url = f"{self.base_url}/{table_name}"
        
        # Add filter if parameters are provided
        filters = []
        if zip_prefix:
            filters.append(f"FIND('{zip_prefix}', {{destination_zip_short}}) = 1")
        if origin:
            filters.append(f"{{origin}} = '{origin}'")
        
        if filters:
            filter_formula = "AND(" + ",".join(filters) + ")"
            url += f"?filterByFormula={filter_formula}"
        
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            data = response.json()
            return [self._format_record(record) for record in data.get('records', [])]
        else:
            raise Exception(f"Failed to fetch delivery services: {response.text}")
    
    def create_delivery_service(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new delivery service record in Airtable.
        
        Args:
            data (Dict[str, Any]): Delivery service data
            
        Returns:
            Dict[str, Any]: Created record
        """
        table_name = "DeliveryService"  # Use the actual table name directly
        response = requests.post(
            f"{self.base_url}/{table_name}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to create delivery service: {response.text}")
    
    def update_delivery_service(self, record_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a delivery service record in Airtable.
        
        Args:
            record_id (str): Airtable record ID
            data (Dict[str, Any]): Data to update
            
        Returns:
            Dict[str, Any]: Updated record
        """
        table_name = "DeliveryService"  # Use the actual table name directly
        response = requests.patch(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to update delivery service: {response.text}")
    
    def delete_delivery_service(self, record_id: str) -> bool:
        """
        Delete a delivery service record from Airtable.
        
        Args:
            record_id (str): Airtable record ID
            
        Returns:
            bool: True if deletion was successful
        """
        table_name = "DeliveryService"  # Use the actual table name directly
        response = requests.delete(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return True
        else:
            raise Exception(f"Failed to delete delivery service: {response.text}")

    def explore_table_structure(self, table_name: str) -> Dict[str, Any]:
        """
        Explore the structure of a table in Airtable to understand its fields.
        
        Args:
            table_name (str): Name of the table in Airtable
            
        Returns:
            Dict[str, Any]: Information about the table structure
        """
        url = f"{self.base_url}/{table_name}"
        
        # Get a single record to examine its structure
        params = {"maxRecords": 1}
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get('records', [])
            
            if not records:
                return {"table_name": table_name, "fields": [], "message": "No records found"}
            
            # Extract field names from the first record
            sample_record = records[0]
            fields = sample_record.get('fields', {})
            
            return {
                "table_name": table_name,
                "record_id_example": sample_record.get('id'),
                "fields": list(fields.keys()),
                "sample_values": fields
            }
        else:
            raise Exception(f"Failed to explore table structure: {response.text}")
    
    # PriorityTag methods
    def get_priority_tags(self) -> List[Dict[str, Any]]:
        """
        Retrieve all priority tags from Airtable.
        
        Returns:
            List[Dict[str, Any]]: List of priority tags
        """
        table_name = "PriorityTag"  # Use the actual table name directly
        response = requests.get(f"{self.base_url}/{table_name}", headers=self.headers)
        
        if response.status_code == 200:
            data = response.json()
            return [self._format_record(record) for record in data.get('records', [])]
        else:
            raise Exception(f"Failed to fetch priority tags: {response.text}")
    
    def create_priority_tag(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new priority tag record in Airtable.
        
        Args:
            data (Dict[str, Any]): Priority tag data
            
        Returns:
            Dict[str, Any]: Created record
        """
        table_name = "PriorityTag"  # Use the actual table name directly
        response = requests.post(
            f"{self.base_url}/{table_name}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to create priority tag: {response.text}")
            
    def update_priority_tag(self, record_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a priority tag record in Airtable.
        
        Args:
            record_id (str): Airtable record ID
            data (Dict[str, Any]): Data to update
            
        Returns:
            Dict[str, Any]: Updated record
        """
        table_name = "PriorityTag"  # Use the actual table name directly
        response = requests.patch(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to update priority tag: {response.text}")
    
    def delete_priority_tag(self, record_id: str) -> bool:
        """
        Delete a priority tag record from Airtable.
        
        Args:
            record_id (str): Airtable record ID
            
        Returns:
            bool: True if deletion was successful
        """
        table_name = "PriorityTag"  # Use the actual table name directly
        response = requests.delete(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return True
        else:
            raise Exception(f"Failed to delete priority tag: {response.text}")
    
    # FulfillmentRule methods
    def get_fulfillment_rules(self, rule_type: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve fulfillment rules from Airtable, optionally filtered by type and active status.
        
        Args:
            rule_type (Optional[str]): Filter by rule type
            active_only (bool): Only return active rules
            
        Returns:
            List[Dict[str, Any]]: List of fulfillment rule records
        """
        table_name = "FulfillmentRule"  # Use the actual table name directly
        url = f"{self.base_url}/{table_name}"
        
        # Add filter if parameters are provided
        filters = []
        if rule_type:
            filters.append(f"{{rule_type}} = '{rule_type}'")
        if active_only:
            filters.append("{is_active} = TRUE()")
        
        if filters:
            filter_formula = "AND(" + ",".join(filters) + ")"
            url += f"?filterByFormula={filter_formula}"
        
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            data = response.json()
            return [self._format_record(record) for record in data.get('records', [])]
        else:
            raise Exception(f"Failed to fetch fulfillment rules: {response.text}")
    
    def create_fulfillment_rule(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new fulfillment rule record in Airtable.
        
        Args:
            data (Dict[str, Any]): Fulfillment rule data
            
        Returns:
            Dict[str, Any]: Created record
        """
        table_name = "FulfillmentRule"  # Use the actual table name directly
        response = requests.post(
            f"{self.base_url}/{table_name}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to create fulfillment rule: {response.text}")
    
    def update_fulfillment_rule(self, record_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a fulfillment rule record in Airtable.
        
        Args:
            record_id (str): Airtable record ID
            data (Dict[str, Any]): Data to update
            
        Returns:
            Dict[str, Any]: Updated record
        """
        table_name = "FulfillmentRule"  # Use the actual table name directly
        response = requests.patch(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers,
            json={"fields": data}
        )
        
        if response.status_code == 200:
            return self._format_record(response.json())
        else:
            raise Exception(f"Failed to update fulfillment rule: {response.text}")
    
    def delete_fulfillment_rule(self, record_id: str) -> bool:
        """
        Delete a fulfillment rule record from Airtable.
        
        Args:
            record_id (str): Airtable record ID
            
        Returns:
            bool: True if deletion was successful
        """
        table_name = "FulfillmentRule"  # Use the actual table name directly
        response = requests.delete(
            f"{self.base_url}/{table_name}/{record_id}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            return True
        else:
            raise Exception(f"Failed to delete fulfillment rule: {response.text}")

# Example usage
if __name__ == "__main__":
    handler = AirtableHandler()
    
    # Example: Explore table structure
    try:
        # Explore the structure of the fulfillment center table
        fc_structure = handler.explore_table_structure("fulfillment_center")
        print("\nFulfillment Center Table Structure:")
        print(f"Table ID: {fc_structure['table_id']}")
        print(f"Fields: {fc_structure['fields']}")
        print(f"Sample Values: {fc_structure['sample_values']}")
        
        # Explore the structure of the delivery service table
        ds_structure = handler.explore_table_structure("delivery_service")
        print("\nDelivery Service Table Structure:")
        print(f"Table ID: {ds_structure['table_id']}")
        print(f"Fields: {ds_structure['fields']}")
        
        # Explore the structure of the SKU mapping table
        sku_structure = handler.explore_table_structure("sku_mapping")
        print("\nSKU Mapping Table Structure:")
        print(f"Table ID: {sku_structure['table_id']}")
        print(f"Fields: {sku_structure['fields']}")
    except Exception as e:
        print(f"Error exploring table structure: {e}")
    
    # Example: Get all fulfillment centers
    try:
        fulfillment_centers = handler.get_fulfillment_centers()
        print(f"\nFound {len(fulfillment_centers)} fulfillment centers")
        if fulfillment_centers:
            print(f"First fulfillment center: {fulfillment_centers[0]}")
    except Exception as e:
        print(f"Error getting fulfillment centers: {e}")
    
    # Example: Get zip to zone mapping
    try:
        zip_to_zone = handler.get_zip_to_zone_mapping()
        print(f"\nZip to zone mapping sample: {list(zip_to_zone.items())[:5]}")
    except Exception as e:
        print(f"Error getting zip to zone mapping: {e}")
    
    # Example: Find best fulfillment center for a zip code
    try:
        best_fc = handler.get_best_fulfillment_center("90210")
        if best_fc:
            print(f"\nBest fulfillment center for 90210: {best_fc.get('name')}")
        else:
            print("\nNo fulfillment center found for 90210")
    except Exception as e:
        print(f"Error finding best fulfillment center: {e}")
        
    # Example: Get SKU mappings for a warehouse
    try:
        sku_mappings = handler.get_sku_mappings("Oxnard")
        print(f"\nFound {len(sku_mappings)} SKU mappings for Oxnard")
        if sku_mappings:
            print(f"First SKU mapping: {sku_mappings[0]}")
    except Exception as e:
        print(f"Error getting SKU mappings: {e}")
    
    # Example: Get delivery services
    try:
        delivery_services = handler.get_delivery_services(zip_prefix="900")
        print(f"\nFound {len(delivery_services)} delivery services for zip prefix 900")
        if delivery_services:
            print(f"First delivery service: {delivery_services[0]}")
    except Exception as e:
        print(f"Error getting delivery services: {e}")
