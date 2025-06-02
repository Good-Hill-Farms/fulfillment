import os
from typing import Any, Dict, List

import aiohttp
import requests


class LLMHandler:
    """
    Handles interactions with OpenRouter API for LLM-based decision making.
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

        # Format context as a string
        context_str = self._format_context(context)

        # Create system message with instructions
        system_message = f"""You are an AI assistant specializing in fruit order fulfillment.
Your task is to help assign customer fruit orders to fulfillment centers based on rules, inventory, and best practices.

CONTEXT:
{context_str}

INSTRUCTIONS:
1. Help assign orders to fulfillment centers based on zip codes, inventory, and bundle rules
2. Explain your reasoning clearly
3. If you want to add a new rule, format it as: RULE_UPDATE: {{\"type\": \"zip|bundle|priority\", \"condition\": \"condition\", \"action\": \"action\"}}
4. Suggest improvements to the fulfillment process
5. Answer questions about inventory, orders, and fulfillment centers

Remember to consider:
- Zip code proximity to warehouses
- Inventory availability
- Bundle requirements
- Priority tags
- Shipping requirements (e.g., Saturday shipping)
"""

        # Format messages for API
        messages = [{"role": "system", "content": system_message}]

        # Add message history (excluding system messages)
        for message in message_history:
            if message["role"] != "system":
                messages.append({"role": message["role"], "content": message["content"]})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Make API request
        try:
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

        # Format context as a string
        context_str = self._format_context(context)

        # Create system message with instructions
        system_message = f"""You are an AI assistant specializing in fruit order fulfillment.
Your task is to help assign customer fruit orders to fulfillment centers based on rules, inventory, and best practices.

CONTEXT:
{context_str}

INSTRUCTIONS:
1. Help assign orders to fulfillment centers based on zip codes, inventory, and bundle rules
2. Explain your reasoning clearly
3. If you want to add a new rule, format it as: RULE_UPDATE: {{\"type\": \"zip|bundle|priority\", \"condition\": \"condition\", \"action\": \"action\"}}
4. Suggest improvements to the fulfillment process
5. Answer questions about inventory, orders, and fulfillment centers

Remember to consider:
- Zip code proximity to warehouses
- Inventory availability
- Bundle requirements
- Priority tags
- Shipping requirements (e.g., Saturday shipping)
"""

        # Format messages for API
        messages = [{"role": "system", "content": system_message}]

        # Add message history (excluding system messages)
        for message in message_history:
            if message["role"] != "system":
                messages.append({"role": message["role"], "content": message["content"]})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Make API request
        try:
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

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error: {str(e)}"

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context dictionary as a readable string

        Args:
            context: Context dictionary

        Returns:
            str: Formatted context string
        """
        formatted = []

        # Format rules
        if "rules" in context and context["rules"]:
            formatted.append("CURRENT RULES:")
            for i, rule in enumerate(context["rules"]):
                formatted.append(
                    f"{i+1}. Type: {rule['type']}, Condition: {rule['condition']}, Action: {rule['action']}"
                )

        # Format bundles
        if "bundles" in context and context["bundles"]:
            formatted.append("\nBUNDLES:")
            for name, fruits in context["bundles"].items():
                formatted.append(f"{name}: {', '.join(fruits)}")

        # Format override log
        if "override_log" in context and context["override_log"]:
            formatted.append("\nRECENT OVERRIDES:")
            for i, override in enumerate(
                context["override_log"][-5:]
            ):  # Show only the 5 most recent
                formatted.append(
                    f"{i+1}. Field: {override['field']}, Old: {override['old_value']}, New: {override['new_value']}, Reason: {override['reason']}"
                )

        # Format sample orders
        if "sample_orders" in context and context["sample_orders"]:
            formatted.append("\nSAMPLE ORDERS:")
            for i, order in enumerate(context["sample_orders"]):
                formatted.append(
                    f"{i+1}. Order: {order.get('Name', 'N/A')}, SKU: {order.get('SKU Helper', 'N/A')}, Zip: {order.get('Shipping: Zip', 'N/A')}"
                )

        # Format sample inventory
        if "sample_inventory" in context and context["sample_inventory"]:
            formatted.append("\nSAMPLE INVENTORY:")
            for i, item in enumerate(context["sample_inventory"]):
                formatted.append(
                    f"{i+1}. Warehouse: {item.get('WarehouseName', 'N/A')}, SKU: {item.get('Sku', 'N/A')}, Available: {item.get('AvailableQty', 'N/A')}"
                )

        return "\n".join(formatted)
