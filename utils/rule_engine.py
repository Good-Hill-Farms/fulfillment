import json
import re
from typing import Any, Dict, List


class RuleEngine:
    """
    Manages and applies rules for order fulfillment.
    """

    def __init__(self, inventory_df=None):
        """
        Initialize the rule engine

        Args:
            inventory_df: DataFrame containing inventory data (optional)
        """
        self.inventory_df = inventory_df

    def get_initial_rules(self) -> List[Dict[str, str]]:
        """
        Get initial set of rules based on common patterns

        Returns:
            List[Dict[str, str]]: List of rule dictionaries
        """
        # Start with basic rules
        initial_rules = [
            {"type": "zip", "condition": "starts with 9", "action": "warehouse = CA-Oxnard-93030"},
            {
                "type": "zip",
                "condition": "starts with 6",
                "action": "warehouse = IL-Wheeling-60090",
            },
            {
                "type": "zip",
                "condition": "starts with 8",
                "action": "warehouse = IL-Wheeling-60090",
            },
            {
                "type": "zip",
                "condition": "starts with 7",
                "action": "warehouse = IL-Wheeling-60090",
            },
        ]

        return initial_rules

    def get_initial_bundles(self) -> Dict[str, List[str]]:
        """
        Get initial set of fruit bundles

        Returns:
            Dict[str, List[str]]: Dictionary of bundle names and their contents
        """
        # Start with basic bundles
        initial_bundles = {
            "Tropical Deluxe": ["mango", "banana", "pineapple"],
            "Citrus Refresh": ["orange", "lemon", "lime"],
            "Exotic Mix": ["dragonfruit", "mangosteen", "lychee", "rambutan"],
            "Seasonal Favorites": ["cherimoya", "loquat", "passionfruit"],
        }

        return initial_bundles

    def parse_llm_rule_updates(self, llm_response: str) -> List[Dict[str, str]]:
        """
        Parse rule updates from LLM response

        Args:
            llm_response: Response from LLM

        Returns:
            List[Dict[str, str]]: List of new rule dictionaries
        """
        new_rules = []

        # Look for rule updates in the format: RULE_UPDATE: {"type": "...", "condition": "...", "action": "..."}
        pattern = r"RULE_UPDATE:\s*({[^}]+})"
        matches = re.findall(pattern, llm_response)

        for match in matches:
            try:
                # Clean up the match to make it valid JSON
                clean_match = match.replace("'", '"')
                rule_dict = json.loads(clean_match)

                # Validate rule structure
                if all(key in rule_dict for key in ["type", "condition", "action"]):
                    new_rules.append(rule_dict)
            except json.JSONDecodeError:
                # Skip invalid JSON
                continue

        return new_rules

    def apply_rules_to_order(
        self, order: Dict[str, Any], rules: List[Dict[str, str]], bundles: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        Apply rules to a single order

        Args:
            order: Order dictionary
            rules: List of rule dictionaries
            bundles: Dictionary of bundles

        Returns:
            Dict[str, Any]: Updated order with rules applied
        """
        updated_order = order.copy()

        # Apply zip code rules
        if not updated_order.get("Fulfillment Center"):
            zip_code = str(updated_order.get("Shipping: Zip", ""))

            for rule in rules:
                if rule["type"] == "zip":
                    condition = rule["condition"]

                    # Check if zip code matches rule condition
                    if condition.startswith("starts with"):
                        prefix = condition.split("starts with ")[1].strip()
                        if zip_code.startswith(prefix):
                            # Extract warehouse from action
                            action_parts = rule["action"].split("=")
                            if len(action_parts) == 2 and action_parts[0].strip() == "warehouse":
                                updated_order["Fulfillment Center"] = action_parts[1].strip()
                                break

        # Apply bundle rules
        sku = updated_order.get("SKU Helper", "").replace("f.", "")

        for rule in rules:
            if rule["type"] == "bundle":
                condition = rule["condition"]

                # Check if SKU is mentioned in condition
                if sku in condition or any(fruit in sku for fruit in condition.split(",")):
                    # Extract bundle from action
                    action_parts = rule["action"].split("=")
                    if len(action_parts) == 2 and action_parts[0].strip() == "bundle":
                        bundle_name = action_parts[1].strip()
                        updated_order["Bundle"] = bundle_name
                        break

        # Apply priority rules
        for rule in rules:
            if rule["type"] == "priority":
                condition = rule["condition"]
                tags = updated_order.get("NEW Tags", "")

                # Check if tags match condition
                if condition in tags:
                    # Extract priority from action
                    action_parts = rule["action"].split("=")
                    if len(action_parts) == 2 and action_parts[0].strip() == "priority":
                        updated_order["Priority"] = action_parts[1].strip()
                        break

        return updated_order

    def add_rule(self, rule_type: str, condition: str, action: str) -> Dict[str, str]:
        """
        Add a new rule

        Args:
            rule_type: Type of rule (zip, bundle, priority)
            condition: Rule condition
            action: Rule action

        Returns:
            Dict[str, str]: New rule dictionary
        """
        new_rule = {"type": rule_type, "condition": condition, "action": action}

        return new_rule

    def add_bundle(self, name: str, fruits: List[str]) -> Dict[str, List[str]]:
        """
        Add a new bundle

        Args:
            name: Bundle name
            fruits: List of fruits in the bundle

        Returns:
            Dict[str, List[str]]: New bundle dictionary
        """
        new_bundle = {name: fruits}

        return new_bundle
