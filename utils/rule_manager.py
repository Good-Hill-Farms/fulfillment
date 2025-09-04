"""
Rule Manager Module

This module provides functionality to manage fulfillment rules,
including temporary rule modifications for order recalculation.
"""

import json
from typing import Any, Dict, List, Optional

from constants.schemas import FulfillmentRule, SchemaManager


class RuleManager:
    """Manager for handling fulfillment rules and temporary rule modifications"""

    def __init__(self):
        """Initialize the RuleManager"""
        self.schema_manager = SchemaManager()

    def get_rules(
        self, rule_type: Optional[str] = None, active_only: bool = True
    ) -> List[FulfillmentRule]:
        """
        Get fulfillment rules from Airtable

        Args:
            rule_type: Optional filter by rule type
            active_only: Whether to only return active rules

        Returns:
            List of FulfillmentRule objects
        """
        return self.schema_manager.get_fulfillment_rules(rule_type, active_only)

    def create_temp_rules(self, original_rules: List[FulfillmentRule]) -> List[FulfillmentRule]:
        """
        Create a deep copy of rules for temporary modification

        Args:
            original_rules: List of original FulfillmentRule objects

        Returns:
            List of copied FulfillmentRule objects
        """
        # Create deep copies of the rules to avoid modifying the originals
        temp_rules = []
        for rule in original_rules:
            # Convert to dict and back to create a deep copy
            rule_dict = {
                "id": rule.id,
                "rule_name": rule.rule_name,
                "rule_type": rule.rule_type,
                "rule_condition": rule.rule_condition,
                "rule_action": rule.rule_action,
                "priority": rule.priority,
                "is_active": rule.is_active,
                "description": rule.description,
                "created_at": rule.created_at,
                "updated_at": rule.updated_at,
            }
            temp_rules.append(FulfillmentRule(**rule_dict))

        return temp_rules

    def update_bundle_rule(
        self, rules: List[FulfillmentRule], bundle_sku: str, components: List[Dict[str, Any]]
    ) -> List[FulfillmentRule]:
        """
        Update bundle components in bundle_substitution rules

        Args:
            rules: List of FulfillmentRule objects
            bundle_sku: SKU of the bundle to update
            components: New component configuration

        Returns:
            Updated list of FulfillmentRule objects
        """
        for rule in rules:
            if rule.rule_type == "bundle_substitution":
                action = rule.get_action()

                # Check if this rule applies to our target bundle
                if "target_bundle" in action and action["target_bundle"] == bundle_sku:
                    # Update the components
                    action["components"] = components
                    rule.set_action(action)

        return rules

    def update_zone_rule(
        self, rules: List[FulfillmentRule], zip_prefix: str, new_zone: str
    ) -> List[FulfillmentRule]:
        """
        Update zone mapping in zone_override rules

        Args:
            rules: List of FulfillmentRule objects
            zip_prefix: ZIP prefix to update
            new_zone: New zone value

        Returns:
            Updated list of FulfillmentRule objects
        """
        for rule in rules:
            if rule.rule_type == "zone_override":
                condition = rule.get_condition()
                action = rule.get_action()

                # Check if this rule applies to our target zip prefix
                if "zip_prefix" in condition and condition["zip_prefix"] == zip_prefix:
                    # Update the zone
                    if "zone" in action:
                        action["zone"] = new_zone
                        rule.set_action(action)

        return rules

    def update_inventory_threshold(
        self, rules: List[FulfillmentRule], sku: str, new_threshold: float
    ) -> List[FulfillmentRule]:
        """
        Update inventory threshold in inventory_threshold rules

        Args:
            rules: List of FulfillmentRule objects
            sku: SKU to update threshold for
            new_threshold: New threshold value

        Returns:
            Updated list of FulfillmentRule objects
        """
        for rule in rules:
            if rule.rule_type == "inventory_threshold":
                condition = rule.get_condition()

                # Check if this rule applies to our target SKU
                if "sku" in condition and condition["sku"] == sku:
                    # Update the threshold
                    if "threshold" in condition:
                        condition["threshold"] = new_threshold
                        rule.set_condition(condition)

        return rules

    def toggle_rule_active(
        self, rules: List[FulfillmentRule], rule_id: str, is_active: bool
    ) -> List[FulfillmentRule]:
        """
        Toggle a rule's active status

        Args:
            rules: List of FulfillmentRule objects
            rule_id: ID of the rule to update
            is_active: New active status

        Returns:
            Updated list of FulfillmentRule objects
        """
        for rule in rules:
            if str(rule.id) == rule_id:
                rule.is_active = is_active
                break

        return rules

    def format_rule_for_display(self, rule: FulfillmentRule) -> Dict[str, Any]:
        """
        Format a rule for display in the UI

        Args:
            rule: FulfillmentRule object

        Returns:
            Dictionary with formatted rule data
        """
        condition = rule.get_condition()
        action = rule.get_action()

        # Format condition and action for display
        condition_str = json.dumps(condition, indent=2) if condition else "{}"
        action_str = json.dumps(action, indent=2) if action else "{}"

        return {
            "id": str(rule.id),
            "rule_name": rule.rule_name,
            "rule_type": rule.rule_type,
            "condition": condition_str,
            "action": action_str,
            "priority": rule.priority,
            "is_active": "Yes" if rule.is_active else "No",
            "description": rule.description or "",
        }
