import os
from typing import Any, Dict, List

import aiohttp
import requests


class LLMHandler:
    """
    Handles interactions with OpenRouter API for LLM-based decision making using session data.
    """

    def __init__(self, model_name="gpt-4"):
        """
        Initialize the LLM handler

        Args:
            model_name: Name of the model to use (default: gpt-4)
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model_name = model_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def get_response(
        self, prompt: str, context: Dict[str, Any], message_history: List[Dict[str, str]]
    ) -> str:
        """
        Get a synchronous response from the LLM

        Args:
            prompt: User prompt
            context: Additional context for the LLM
            message_history: Chat history

        Returns:
            str: LLM response
        """
        if not self.api_key:
            return "Error: OpenRouter API key not found. Please add it to your .env file."

        messages = self._prepare_messages(prompt, context, message_history)
        
        try:
            headers, data = self._prepare_request_data(messages)
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error: {str(e)}"

    async def get_response_async(
        self, prompt: str, context: Dict[str, Any], message_history: List[Dict[str, str]]
    ) -> str:
        """
        Get an asynchronous response from the LLM

        Args:
            prompt: User prompt
            context: Additional context for the LLM
            message_history: Chat history

        Returns:
            str: LLM response
        """
        if not self.api_key:
            return "Error: OpenRouter API key not found. Please add it to your .env file."

        messages = self._prepare_messages(prompt, context, message_history)
        
        try:
            headers, data = self._prepare_request_data(messages)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error: {str(e)}"

    def _prepare_messages(self, prompt: str, context: Dict[str, Any], message_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Prepare messages for API request
        
        Args:
            prompt: User prompt
            context: Additional context for the LLM
            message_history: Chat history
            
        Returns:
            List of formatted messages
        """
        # Format context as a string including session data
        context_str = self._format_context_with_session_data(context)

        # Create system message with instructions
        system_message = f"""You are an AI assistant specializing in fruit order fulfillment.
Your task is to help assign customer fruit orders to fulfillment centers based on rules, inventory, and best practices.

CRITICAL REQUIREMENTS:
- NEVER give general or theoretical answers
- ALWAYS base responses on actual numbers, SKUs, and data from the current session
- ALWAYS include specific data points, quantities, SKU codes, and warehouse names in your responses
- Quote exact numbers from inventory balances, order quantities, shortage amounts
- Reference specific SKU codes and warehouse locations
- Use the actual data provided in the context below

CONTEXT:
{context_str}

INSTRUCTIONS:
1. Answer using ONLY the session data provided above - no general advice
2. Include specific numbers: inventory quantities, order counts, SKU codes, warehouse names
3. Reference exact data points: "SKU apple-10x05 has 150 units at CA-Oxnard-93030"
4. Show calculations: "Order #12345 needs 50 units, inventory shows 200 available"
5. Identify specific issues: "Order #67890 for SKU mango-12x08 cannot be fulfilled - only 5 units available, need 25"
6. If you want to add a new rule, format it as: RULE_UPDATE: {{"type": "zip|bundle|priority", "condition": "condition", "action": "action"}}

DATA-DRIVEN ANALYSIS REQUIREMENTS:
- Always cite specific inventory levels for mentioned SKUs
- Always reference actual order numbers and quantities when discussing orders
- Always mention warehouse locations by name (CA-Oxnard-93030, IL-Wheeling-60090, etc.)
- Always show the math behind recommendations
- Never make assumptions - only work with provided data

If no relevant data exists in the session, state "No data available for this query" rather than giving generic advice.
"""

        # Format messages for API
        messages = [{"role": "system", "content": system_message}]

        # Add message history (excluding system messages)
        for message in message_history:
            if message["role"] != "system":
                messages.append({"role": message["role"], "content": message["content"]})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages

    def _prepare_request_data(self, messages: List[Dict[str, str]]) -> tuple:
        """
        Prepare headers and data for API request
        
        Args:
            messages: Formatted messages for the API
            
        Returns:
            Tuple of (headers, data)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        
        return headers, data

    def _format_context_with_session_data(self, context: Dict[str, Any]) -> str:
        """
        Format context dictionary as a readable string with enhanced session data

        Args:
            context: Context dictionary

        Returns:
            str: Formatted context string with session data
        """
        formatted = []

        # Add session data summary
        formatted.append("=== SESSION DATA SUMMARY ===")
        
        # Count data in session - check for both old and new key names
        data_counts = []
        
        # Check for orders data
        orders_data = context.get("sample_orders") or context.get("orders") or context.get("processed_orders")
        if orders_data:
            count = len(orders_data) if isinstance(orders_data, list) else len(orders_data.get('index', {}))
            data_counts.append(f"Orders: {count}")
        
        # Check for inventory data  
        inventory_data = context.get("sample_inventory") or context.get("inventory") or context.get("inventory_summary")
        if inventory_data:
            count = len(inventory_data) if isinstance(inventory_data, list) else len(inventory_data.get('index', {}))
            data_counts.append(f"Inventory Items: {count}")
        
        # Check for shortage data
        shortage_data = context.get("shortage_summary") or context.get("grouped_shortage_summary")
        if shortage_data:
            count = len(shortage_data) if isinstance(shortage_data, list) else len(shortage_data.get('index', {}))
            total_shortages = context.get("shortage_count", count)
            data_counts.append(f"Shortages: {total_shortages} total")
            
        # Check for staged orders
        if context.get("staged_orders"):
            staged_count = context.get("staged_orders_count", len(context["staged_orders"]))
            data_counts.append(f"Staged Orders: {staged_count}")
            
        # Check for additional data
        if context.get("inventory_comparison"):
            comp_count = context.get("inventory_comparison_count", len(context["inventory_comparison"]))
            data_counts.append(f"Inventory Comparisons: {comp_count}")
            
        if context.get("staging_history"):
            data_counts.append(f"Staging History: {len(context['staging_history'])} actions")
            
        if context.get("processing_stats"):
            data_counts.append(f"Processing Stats: Available")
            
        if context.get("warehouse_performance"):
            data_counts.append(f"Warehouse Performance: Available")
            
        if "rules" in context and context["rules"]:
            data_counts.append(f"Rules: {len(context['rules'])}")
        if "bundles" in context and context["bundles"]:
            data_counts.append(f"Bundles: {len(context['bundles'])}")
            
        # Check for Airtable data
        airtable_counts = []
        if context.get("airtable_sku_mappings"):
            airtable_counts.append(f"SKU Mappings: {context.get('airtable_sku_count', 0)}")
        if context.get("airtable_fulfillment_zones"):
            airtable_counts.append(f"Fulfillment Zones: {context.get('airtable_zones_count', 0)}")
        if context.get("airtable_delivery_services"):
            airtable_counts.append(f"Delivery Services: {context.get('airtable_services_count', 0)}")
        if context.get("airtable_fulfillment_centers"):
            airtable_counts.append(f"Fulfillment Centers: {context.get('airtable_centers_count', 0)}")
            
        # Add debug info if available
        if "debug_session_state" in context:
            formatted.append("DEBUG - Session State Status:")
            for key, status in context["debug_session_state"].items():
                formatted.append(f"  {key}: {status}")
        
        if data_counts:
            formatted.append("Data loaded: " + ", ".join(data_counts))
        if airtable_counts:
            formatted.append("Airtable data: " + ", ".join(airtable_counts))
        if not data_counts and not airtable_counts:
            formatted.append("No session data loaded yet.")

        # Format rules
        if "rules" in context and context["rules"]:
            formatted.append("\n=== CURRENT FULFILLMENT RULES ===")
            for i, rule in enumerate(context["rules"]):
                formatted.append(
                    f"{i+1}. Type: {rule['type']}, Condition: {rule['condition']}, Action: {rule['action']}"
                )

        # Format bundles
        if "bundles" in context and context["bundles"]:
            formatted.append("\n=== FRUIT BUNDLES ===")
            for name, fruits in context["bundles"].items():
                formatted.append(f"{name}: {', '.join(fruits)}")

        # Format shortage data first (most important)
        shortage_data = context.get("shortage_summary") or context.get("grouped_shortage_summary")
        if shortage_data:
            total_shortages = context.get("shortage_count", len(shortage_data) if isinstance(shortage_data, list) else 0)
            formatted.append(f"\n=== CURRENT SHORTAGES ({total_shortages} total) ===")
            
            if isinstance(shortage_data, list):
                # It's a list of records (new format)
                formatted.append(f"Showing all {len(shortage_data)} shortage items:")
                for i, item in enumerate(shortage_data):
                    shortage_details = []
                    # Format all available fields from the shortage record
                    for key, value in item.items():
                        if value is not None and str(value).strip():
                            shortage_details.append(f"{key}: {value}")
                    
                    if shortage_details:
                        formatted.append(f"{i+1}. {' | '.join(shortage_details)}")
                    else:
                        formatted.append(f"{i+1}. Shortage item {i} (no details)")
                        
            elif isinstance(shortage_data, dict) and 'index' in shortage_data:
                # It's a DataFrame converted to dict (old format)
                indices = list(shortage_data['index'].keys())[:10]  # Show first 10
                formatted.append(f"Showing first 10 of {len(shortage_data['index'])} shortage items:")
                
                for i, idx in enumerate(indices):
                    shortage_details = []
                    # Try to get common shortage fields
                    for field in ['sku', 'SKU', 'warehouse_name', 'WarehouseName', 'shortage_qty', 'ShortageQty', 'required_qty', 'available_qty']:
                        if field in shortage_data and str(idx) in shortage_data[field]:
                            shortage_details.append(f"{field}: {shortage_data[field][str(idx)]}")
                    
                    if shortage_details:
                        formatted.append(f"{i+1}. {' | '.join(shortage_details)}")
                    else:
                        formatted.append(f"{i+1}. Shortage item {idx} (details not available)")
            else:
                formatted.append("Shortage data format not recognized")

        # Format sample orders with more detail
        orders_data = context.get("sample_orders") or context.get("orders") or context.get("processed_orders")
        if orders_data:
            if isinstance(orders_data, dict) and 'index' in orders_data:
                # It's a DataFrame converted to dict
                total_orders = len(orders_data['index'])
                formatted.append(f"\n=== PROCESSED ORDERS (showing first 10 of {total_orders}) ===")
                indices = list(orders_data['index'].keys())[:10]
                
                for i, idx in enumerate(indices):
                    order_details = []
                    # Try to get common order fields
                    for field in ['Name', 'SKU Helper', 'ordernumber', 'Shipping: Zip', 'Priority', 'Fulfillment Center', 'FulfillmentCenter']:
                        if field in orders_data and str(idx) in orders_data[field]:
                            order_details.append(f"{field}: {orders_data[field][str(idx)]}")
                    
                    if order_details:
                        formatted.append(f"{i+1}. {' | '.join(order_details)}")
            elif isinstance(orders_data, list):
                # It's a list of orders
                formatted.append(f"\n=== SAMPLE ORDERS (showing first 10 of {len(orders_data)}) ===")
                for i, order in enumerate(orders_data[:10]):
                    order_details = []
                    if order.get('Name'):
                        order_details.append(f"Customer: {order['Name']}")
                    if order.get('SKU Helper'):
                        order_details.append(f"SKU: {order['SKU Helper']}")
                    if order.get('Shipping: Zip'):
                        order_details.append(f"Zip: {order['Shipping: Zip']}")
                    if order.get('Priority'):
                        order_details.append(f"Priority: {order['Priority']}")
                    if order.get('Fulfillment Center'):
                        order_details.append(f"FC: {order['Fulfillment Center']}")
                    
                    formatted.append(f"{i+1}. {' | '.join(order_details)}")

        # Format inventory data
        inventory_data = context.get("sample_inventory") or context.get("inventory") or context.get("inventory_summary")
        if inventory_data:
            if isinstance(inventory_data, dict) and 'index' in inventory_data:
                # It's a DataFrame converted to dict
                total_items = len(inventory_data['index'])
                formatted.append(f"\n=== INVENTORY SUMMARY (showing first 10 of {total_items}) ===")
                indices = list(inventory_data['index'].keys())[:10]
                
                for i, idx in enumerate(indices):
                    inventory_details = []
                    # Try to get common inventory fields
                    for field in ['WarehouseName', 'warehouse_name', 'Sku', 'SKU', 'AvailableQty', 'Balance', 'available_qty']:
                        if field in inventory_data and str(idx) in inventory_data[field]:
                            inventory_details.append(f"{field}: {inventory_data[field][str(idx)]}")
                    
                    if inventory_details:
                        formatted.append(f"{i+1}. {' | '.join(inventory_details)}")
            elif isinstance(inventory_data, list):
                # It's a list of inventory items
                formatted.append(f"\n=== INVENTORY SUMMARY (showing first 10 of {len(inventory_data)}) ===")
                
                # Create warehouse summary
                warehouse_summary = {}
                for item in inventory_data:
                    warehouse = item.get('WarehouseName', 'Unknown')
                    sku = item.get('Sku', 'Unknown')
                    qty = int(item.get('AvailableQty', 0))
                    
                    if warehouse not in warehouse_summary:
                        warehouse_summary[warehouse] = {'total_qty': 0, 'skus': set()}
                    warehouse_summary[warehouse]['total_qty'] += qty
                    warehouse_summary[warehouse]['skus'].add(sku)
                
                formatted.append("Warehouse Summary:")
                for warehouse, data in warehouse_summary.items():
                    formatted.append(f"- {warehouse}: {data['total_qty']} total units, {len(data['skus'])} different SKUs")
                
                formatted.append("\nDetailed Inventory (first 10 items):")
                for i, item in enumerate(inventory_data[:10]):
                    formatted.append(
                        f"{i+1}. {item.get('WarehouseName', 'N/A')} | SKU: {item.get('Sku', 'N/A')} | Available: {item.get('AvailableQty', 'N/A')}"
                    )

        # Format Airtable data
        if context.get("airtable_sku_mappings"):
            formatted.append(f"\n=== AIRTABLE SKU MAPPINGS ({context.get('airtable_sku_count', 0)} total) ===")
            for i, mapping in enumerate(context["airtable_sku_mappings"][:10]):  # Show first 10
                mapping_details = []
                for key in ['order_sku', 'picklist_sku', 'actual_qty', 'fulfillment_center']:
                    if key in mapping and mapping[key]:
                        mapping_details.append(f"{key}: {mapping[key]}")
                if mapping_details:
                    formatted.append(f"{i+1}. {' | '.join(mapping_details)}")
        
        if context.get("airtable_fulfillment_zones"):
            formatted.append(f"\n=== AIRTABLE FULFILLMENT ZONES ({context.get('airtable_zones_count', 0)} total) ===")
            for i, zone in enumerate(context["airtable_fulfillment_zones"][:10]):  # Show first 10
                zone_details = []
                for key in ['zip_prefix', 'zone', 'fulfillment_center']:
                    if key in zone and zone[key]:
                        zone_details.append(f"{key}: {zone[key]}")
                if zone_details:
                    formatted.append(f"{i+1}. {' | '.join(zone_details)}")
        
        if context.get("airtable_delivery_services"):
            formatted.append(f"\n=== AIRTABLE DELIVERY SERVICES ({context.get('airtable_services_count', 0)} total) ===")
            for i, service in enumerate(context["airtable_delivery_services"][:10]):  # Show first 10
                service_details = []
                for key in ['carrier_name', 'service_name', 'days', 'origin']:
                    if key in service and service[key]:
                        service_details.append(f"{key}: {service[key]}")
                if service_details:
                    formatted.append(f"{i+1}. {' | '.join(service_details)}")
        
        if context.get("airtable_fulfillment_centers"):
            formatted.append(f"\n=== AIRTABLE FULFILLMENT CENTERS ({context.get('airtable_centers_count', 0)} total) ===")
            for i, center in enumerate(context["airtable_fulfillment_centers"][:10]):  # Show first 10
                center_details = []
                for key in ['name', 'zip_code', 'active']:
                    if key in center and center[key]:
                        center_details.append(f"{key}: {center[key]}")
                if center_details:
                    formatted.append(f"{i+1}. {' | '.join(center_details)}")

        # Format current staging state (PRIORITY DATA)
        if context.get("staging_workflow_active"):
            formatted.append(f"\n=== CURRENT STAGING WORKFLOW STATUS ===")
            staged_count = context.get("currently_staged_count", 0)
            processing_count = context.get("currently_processing_count", 0)
            formatted.append(f"Orders Currently Staged: {staged_count}")
            formatted.append(f"Orders in Processing: {processing_count}")
            formatted.append(f"Total Orders: {staged_count + processing_count}")
            
            # Show currently staged orders
            if context.get("currently_staged_orders") and staged_count > 0:
                formatted.append(f"\n=== CURRENTLY STAGED ORDERS ({staged_count} total) ===")
                for i, order in enumerate(context["currently_staged_orders"][:10]):  # Show first 10
                    order_details = []
                    for key in ['ordernumber', 'sku', 'Fulfillment Center', 'Transaction Quantity', 'Issues']:
                        if key in order and order[key]:
                            order_details.append(f"{key}: {order[key]}")
                    if order_details:
                        formatted.append(f"{i+1}. {' | '.join(order_details)}")
            
            # Show orders in processing
            if context.get("currently_processing_orders") and processing_count > 0:
                formatted.append(f"\n=== ORDERS IN PROCESSING ({processing_count} total - showing first 10) ===")
                for i, order in enumerate(context["currently_processing_orders"][:10]):  # Show first 10
                    order_details = []
                    for key in ['ordernumber', 'sku', 'Fulfillment Center', 'Transaction Quantity', 'Issues']:
                        if key in order and order[key]:
                            order_details.append(f"{key}: {order[key]}")
                    if order_details:
                        formatted.append(f"{i+1}. {' | '.join(order_details)}")
        
        # Real-time staging processor data
        if context.get("staging_processor_data"):
            proc_data = context["staging_processor_data"]
            formatted.append(f"\n=== STAGING PROCESSOR STATUS ===")
            formatted.append(f"Workflow Initialized: {proc_data.get('workflow_initialized', False)}")
            
            if "staging_summary" in proc_data:
                summary = proc_data["staging_summary"]
                for key, value in summary.items():
                    formatted.append(f"- {key}: {value}")
            
            formatted.append(f"Initial Inventory Items: {proc_data.get('initial_inventory_count', 0)}")
            formatted.append(f"Inventory After Processing: {proc_data.get('inventory_minus_processing_count', 0)}")
            formatted.append(f"Inventory After Staging: {proc_data.get('inventory_minus_staged_count', 0)}")
        
        # Available inventory after staging
        if context.get("inventory_minus_staged"):
            inv_count = context.get("inventory_minus_staged_count", 0)
            formatted.append(f"\n=== AVAILABLE INVENTORY (After Staging - {inv_count} items) ===")
            for i, item in enumerate(context["inventory_minus_staged"][:10]):  # Show first 10
                formatted.append(f"{i+1}. Warehouse: {item.get('warehouse', 'N/A')} | SKU: {item.get('sku', 'N/A')} | Available: {item.get('available_balance', 'N/A')}")
        
        # Legacy staged orders for backward compatibility
        if context.get("staged_orders"):
            staged_count = context.get("staged_orders_count", len(context["staged_orders"]))
            formatted.append(f"\n=== LEGACY STAGED ORDERS ({staged_count} total) ===")
            for i, order in enumerate(context["staged_orders"][:10]):  # Show first 10
                order_details = []
                for key in ['ordernumber', 'customerFirstName', 'Fulfillment Center', 'shopifysku2', 'staged_at']:
                    if key in order and order[key]:
                        order_details.append(f"{key}: {order[key]}")
                if order_details:
                    formatted.append(f"{i+1}. {' | '.join(order_details)}")

        # Format staging history
        if context.get("staging_history"):
            formatted.append(f"\n=== STAGING HISTORY ({len(context['staging_history'])} actions) ===")
            for i, action in enumerate(context["staging_history"][-5:]):  # Show last 5 actions
                action_details = []
                for key in ['timestamp', 'action', 'count', 'removed_from_main']:
                    if key in action:
                        action_details.append(f"{key}: {action[key]}")
                if action_details:
                    formatted.append(f"{i+1}. {' | '.join(action_details)}")

        # Format processing stats
        if context.get("processing_stats"):
            formatted.append(f"\n=== PROCESSING STATISTICS ===")
            stats = context["processing_stats"]
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    formatted.append(f"- {key}: {value:,.2f}")
                else:
                    formatted.append(f"- {key}: {value}")

        # Format warehouse performance
        if context.get("warehouse_performance"):
            formatted.append(f"\n=== WAREHOUSE PERFORMANCE ===")
            perf = context["warehouse_performance"]
            for key, value in perf.items():
                if isinstance(value, (int, float)):
                    formatted.append(f"- {key}: {value:,.2f}")
                else:
                    formatted.append(f"- {key}: {value}")

        # Format override log
        if "override_log" in context and context["override_log"]:
            formatted.append("\n=== RECENT OVERRIDES ===")
            for i, override in enumerate(
                context["override_log"][-5:]
            ):  # Show only the 5 most recent
                formatted.append(
                    f"{i+1}. Field: {override['field']}, Old: {override['old_value']}, New: {override['new_value']}, Reason: {override['reason']}"
                )

        return "\n".join(formatted)

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context dictionary as a readable string (legacy method for compatibility)

        Args:
            context: Context dictionary

        Returns:
            str: Formatted context string
        """
        return self._format_context_with_session_data(context)