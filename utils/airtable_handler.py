"""
Airtable Handler Module

This module provides functionality to interact with Airtable for storing and retrieving 
zip code mapping data for fulfillment operations.
"""

import os
from typing import Dict, List, Optional, Union, Any
import requests
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

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
        
        # Retrieve all SKU mappings first - this is a fallback if filtering fails
        all_mappings = []
        offset = None
        params = {}
        
        # If we have a warehouse filter, get its ID from FulfillmentCenter table
        fc_id = None
        if warehouse:
            # Get the Fulfillment Center record ID
            fc_table = self.tables.get("fulfillment_center", "FulfillmentCenter")
            fc_url = f"{self.base_url}/{fc_table}"
            fc_params = {'filterByFormula': f"{{name}} = '{warehouse}'"}
            
            fc_response = requests.get(fc_url, headers=self.headers, params=fc_params)
            if fc_response.status_code == 200:
                fc_data = fc_response.json()
                fc_records = fc_data.get('records', [])
                
                if fc_records:
                    fc_id = fc_records[0].get('id')  # Get the Airtable ID of the fulfillment center
                    print(f"Found fulfillment center '{warehouse}' with ID: {fc_id}")
            
        # Loop to handle pagination and retrieve all records
        while True:
            # Add offset parameter if we have one from a previous request
            if offset:
                params['offset'] = offset
                
            # Make the request
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                records = [self._format_record(record) for record in data.get('records', [])]
                all_mappings.extend(records)
                
                # Check if there are more records to fetch
                offset = data.get('offset')
                if not offset:
                    break  # No more pages
            else:
                raise Exception(f"Failed to fetch SKU mappings: {response.text}")
        
        # If we have a warehouse to filter by, do client-side filtering
        # This is more reliable than using complex Airtable formulas for linked records
        if warehouse and fc_id:
            # Filter mappings where fulfillment_center is a list containing the fc_id
            filtered_mappings = []
            for mapping in all_mappings:
                # Check if this mapping has a fulfillment_center that matches our target
                fc_array = mapping.get('fulfillment_center', [])
                
                # Debug output to understand what's in the records
                if len(filtered_mappings) == 0:
                    print(f"Example record structure: {mapping}")
                
                # Different ways the fulfillment center might be referenced
                if isinstance(fc_array, list) and fc_id in fc_array:
                    filtered_mappings.append(mapping)
                elif fc_id == fc_array:  # Direct reference
                    filtered_mappings.append(mapping)
                elif isinstance(fc_array, str) and fc_id in fc_array:  # String containing ID
                    filtered_mappings.append(mapping)
                elif 'name' in mapping and mapping.get('name') == warehouse:  # By name
                    filtered_mappings.append(mapping)
                # Might be nested under another key
                elif isinstance(mapping.get('fields', {}), dict):
                    fields = mapping.get('fields', {})
                    if isinstance(fields.get('fulfillment_center', []), list) and fc_id in fields.get('fulfillment_center', []):
                        filtered_mappings.append(mapping)
            
            print(f"Client-side filtering found {len(filtered_mappings)} SKU mappings for '{warehouse}'")
            return filtered_mappings
        
        print(f"Retrieved {len(all_mappings)} total SKU mappings")
        return all_mappings
    
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
    
    def load_sku_mappings_from_airtable(self, center):
        """
        Load SKU mappings from Airtable for the specified fulfillment center.
        
        Args:
            center (str): The fulfillment center to load mappings for ("Oxnard" or "Wheeling")
            
        Returns:
            dict: Dictionary containing mappings and bundle_info for the specified center
        """
        logger.info(f"Loading SKU mappings for {center} from Airtable...")
        
        # Initialize result structure
        result = {
            "mappings": {center: {}},       # Simple shopify_sku -> inventory_sku mapping
            "bundle_info": {center: {}},   # Bundle component information
            "details": {center: {}}       # All details for each SKU
        }
        
        try:
            # Get SKU mappings from Airtable for this warehouse
            sku_mappings = self.get_sku_mappings(center)
            
            for mapping in sku_mappings:
                # Skip if no order_sku
                if 'order_sku' not in mapping or not mapping['order_sku']:
                    continue
                    
                order_sku = mapping['order_sku']
                
                # Store all fields in the details dictionary
                result["details"][center][order_sku] = {
                    # Common fields - add defaults for any missing fields to prevent errors
                    "picklist_sku": mapping.get('picklist_sku', ''),
                    "actual_qty": mapping.get('actual_qty', 0),
                    "total_pick_weight": mapping.get('total_pick_weight', 0),
                    "pick_type": mapping.get('pick_type', ''),
                    # Include any other fields that might be useful
                    "airtable_id": mapping.get('airtable_id', ''),
                    "created_time": mapping.get('created_time', ''),
                    # Store the entire mapping record for maximum flexibility
                    "raw_data": mapping
                }
                
                # Handle individual SKUs - use dictionary access instead of attribute access
                if 'picklist_sku' in mapping and mapping['picklist_sku']:
                    # Basic SKU mapping (shopify -> inventory)
                    result["mappings"][center][order_sku] = mapping['picklist_sku']
                
                # Handle bundles if they have components
                if 'bundle_components' in mapping and mapping['bundle_components']:
                    bundle_components_str = mapping['bundle_components']
                    # Convert the JSON string to Python objects
                    import json
                    try:
                        # Try to parse the JSON string
                        bundle_components = json.loads(bundle_components_str)
                        
                        # Store bundle components with all available information
                        result["bundle_info"][center][order_sku] = [
                            {
                                "sku": comp.get("sku", ""),
                                "qty": comp.get("qty", 0),
                                "weight": comp.get("weight", 0.0), 
                                "type": comp.get("type", "")
                            }
                            for comp in bundle_components
                        ]
                        
                        # If this is a bundle but has no regular picklist_sku mapping, 
                        # map to first component for backward compatibility
                        if bundle_components and ('picklist_sku' not in mapping or not mapping['picklist_sku']):
                            if bundle_components[0].get("sku"):
                                result["mappings"][center][order_sku] = bundle_components[0].get("sku")
                                logger.debug(f"Mapped bundle {order_sku} to component {bundle_components[0].get('sku')}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing bundle_components JSON for {order_sku}: {e}")
            
            logger.info(f"Loaded {len(result['mappings'][center])} SKU mappings and {len(result['details'][center])} detail records for {center}")
            
        except Exception as e:
            logger.warning(f"Error loading SKU mappings for {center} from Airtable: {e}")
        
        return result

    

if __name__ == "__main__":
    handler = AirtableHandler()
    res = handler.load_sku_mappings_from_airtable("Wheeling")
    print(res)