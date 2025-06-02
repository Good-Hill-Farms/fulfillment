from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st


def render_header():
    """Render the application header"""
    st.title("🍍 AI-Powered Fulfillment Assistant")
    st.markdown(
        """
    This application helps assign customer fruit orders to fulfillment centers using:
    - AI-enhanced logic
    - Rule-based assignments
    - Inventory optimization
    """
    )


def render_progress_bar(current_step, total_steps, step_name):
    """Render a progress bar for processes"""
    progress = st.progress(0)
    # Avoid division by zero
    if total_steps > 0:
        progress.progress(current_step / total_steps)
    else:
        progress.progress(0)
    st.caption(f"Step {current_step}/{total_steps}: {step_name}")


def render_summary_dashboard(processed_orders, inventory_df):
    """
    Render summary dashboard with charts and metrics

    Args:
        processed_orders: DataFrame of processed orders
        inventory_df: DataFrame of inventory
    """
    if processed_orders is None or inventory_df is None:
        st.warning("No data available for dashboard")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_orders = len(processed_orders["ordernumber"].unique())
        st.metric("Unique Orders", total_orders)

    with col2:
        total_items = len(processed_orders)
        st.metric("Total Items", total_items)

    with col3:
        fulfillment_centers = processed_orders["Fulfillment Center"].nunique()
        st.metric("Fulfillment Centers", fulfillment_centers)

    with col4:
        issues = processed_orders[processed_orders["Issues"] != ""].shape[0]
        st.metric("Issues", issues, delta=None, delta_color="inverse")

    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Show total number of line items processed
        total_line_items = processed_orders.shape[0]
        st.metric("Line Items", total_line_items)

    with col2:
        # Check if Priority column exists
        if "Priority" in processed_orders.columns:
            priority_orders = processed_orders[processed_orders["Priority"] != ""].shape[0]
        else:
            priority_orders = 0
        st.metric("Priority Orders", priority_orders)

    with col3:
        # Check if Bundle column exists
        if "Bundle" in processed_orders.columns:
            bundle_orders = processed_orders[processed_orders["Bundle"] != ""].shape[0]
        else:
            bundle_orders = 0
        st.metric("Bundle Orders", bundle_orders)

    with col4:
        if "Inventory Status" in processed_orders.columns:
            critical_inventory = processed_orders[
                processed_orders["Inventory Status"] == "Critical"
            ].shape[0]
            st.metric(
                "Critical Inventory Items", critical_inventory, delta=None, delta_color="inverse"
            )

    # Orders by fulfillment center
    st.subheader("Orders by Fulfillment Center")
    if "Fulfillment Center" in processed_orders.columns:
        fc_counts = processed_orders["Fulfillment Center"].value_counts().reset_index()
        fc_counts.columns = ["Fulfillment Center", "Count"]
    else:
        # Create empty dataframe if Fulfillment Center column doesn't exist
        fc_counts = pd.DataFrame({"Fulfillment Center": ["No Data"], "Count": [0]})

    fig = px.bar(
        fc_counts,
        x="Fulfillment Center",
        y="Count",
        title="Order Distribution by Fulfillment Center",
        color="Fulfillment Center",
        text="Count",
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Priority distribution
    if "Priority" in processed_orders.columns and processed_orders["Priority"].any():
        st.subheader("Orders by Priority")
        priority_counts = (
            processed_orders[processed_orders["Priority"] != ""]["Priority"]
            .value_counts()
            .reset_index()
        )
        priority_counts.columns = ["Priority", "Count"]

        fig = px.pie(
            priority_counts,
            values="Count",
            names="Priority",
            title="Order Distribution by Priority",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Inventory utilization
    st.subheader("Inventory Utilization")

    # Calculate inventory usage
    inventory_usage = (
        processed_orders.groupby("sku")
        .agg({"Transaction Quantity": "sum", "Starting Balance": "first"})
        .reset_index()
    )
    # Avoid division by zero
    # Convert to numeric first to ensure round() works properly
    inventory_usage["Transaction Quantity"] = pd.to_numeric(
        inventory_usage["Transaction Quantity"], errors="coerce"
    )
    inventory_usage["Starting Balance"] = pd.to_numeric(
        inventory_usage["Starting Balance"], errors="coerce"
    )

    # Calculate usage percentage with safeguards
    inventory_usage["Usage Percentage"] = inventory_usage.apply(
        lambda row: min(
            100, round((row["Transaction Quantity"] / row["Starting Balance"]) * 100, 1)
        )
        if row["Starting Balance"] > 0
        else 0,
        axis=1,
    )

    # Filter out rows with zero starting balance
    inventory_usage = inventory_usage[inventory_usage["Starting Balance"] > 0]

    # Sort by usage percentage
    inventory_usage = inventory_usage.sort_values("Usage Percentage", ascending=False).head(10)

    fig = px.bar(
        inventory_usage,
        x="sku",
        y="Usage Percentage",
        title="Top 10 SKUs by Inventory Usage (%)",
        color="Usage Percentage",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Issues summary
    st.subheader("Issues Summary")

    if processed_orders[processed_orders["Issues"] != ""].shape[0] > 0:
        issues_df = processed_orders[processed_orders["Issues"] != ""]
        # Display only columns that exist in the dataframe
        display_columns = ["ordernumber", "sku", "Fulfillment Center", "Issues"]
        # Check if 'Priority' column exists before adding it
        if "Priority" in issues_df.columns:
            display_columns.append("Priority")
        st.dataframe(issues_df[display_columns])
    else:
        st.info("No issues found in processed orders.")

    # Bundle analysis
    if "Bundle" in processed_orders.columns and processed_orders["Bundle"].any():
        st.subheader("Bundle Analysis")
        bundle_counts = (
            processed_orders[processed_orders["Bundle"] != ""]["Bundle"]
            .value_counts()
            .reset_index()
        )
        bundle_counts.columns = ["Bundle", "Count"]

        fig = px.bar(
            bundle_counts,
            x="Bundle",
            y="Count",
            title="Orders by Bundle Type",
            color="Bundle",
            text="Count",
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Fulfillment Center Distribution
    st.subheader("Fulfillment Center Distribution")
    if "Fulfillment Center" in processed_orders.columns:
        # Create a simple count of orders by fulfillment center
        fc_counts = processed_orders["Fulfillment Center"].value_counts().reset_index()
        fc_counts.columns = ["Fulfillment Center", "Count"]

        # Calculate percentages
        total_orders = fc_counts["Count"].sum()
        if total_orders > 0:  # Avoid division by zero
            fc_counts["Percentage"] = (fc_counts["Count"] / total_orders * 100).round(1)

            fig = px.bar(
                fc_counts,
                x="Fulfillment Center",
                y="Percentage",
                title="Order Distribution by Fulfillment Center (%)",
                color="Fulfillment Center",
                text="Percentage",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No orders data available.")
    else:
        st.info("Fulfillment Center data not available in the processed orders.")

    # Issues summary
    st.subheader("Issues Summary")

    # Extract issues
    issues_df = processed_orders[processed_orders["Issues"] != ""].copy()

    if not issues_df.empty:
        # Count issue types
        issues_df["Issue Type"] = issues_df["Issues"].str.extract(r"([^;]+);")
        issue_counts = issues_df["Issue Type"].value_counts().reset_index()
        issue_counts.columns = ["Issue Type", "Count"]

        fig = px.pie(
            issue_counts, values="Count", names="Issue Type", title="Distribution of Issues"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show issues table
        with st.expander("View All Issues"):
            st.dataframe(
                issues_df[["externalorderid", "sku", "Fulfillment Center", "Issues"]],
                use_container_width=True,
            )
    else:
        st.success("No issues found!")


def render_rule_editor(rules, bundles, override_log):
    """
    Render rule editor interface

    Args:
        rules: List of rule dictionaries
        bundles: Dictionary of bundles
        override_log: List of override logs

    Returns:
        bool: True if rules were updated, False otherwise
    """
    updated = False

    # Create tabs for different rule types
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Zip Code Rules", "Bundle Rules", "Priority Rules", "Override Log"]
    )

    with tab1:
        st.subheader("Zip Code Rules")
        st.write("These rules determine which fulfillment center to use based on zip code.")

        # Show existing zip rules
        zip_rules = [rule for rule in rules if rule["type"] == "zip"]

        if zip_rules:
            zip_df = pd.DataFrame(zip_rules)
            edited_zip_df = st.data_editor(
                zip_df, use_container_width=True, num_rows="dynamic", key="zip_rules_editor"
            )

            # Check if rules were updated
            if not zip_df.equals(edited_zip_df):
                # Update rules
                updated_rules = [rule for rule in rules if rule["type"] != "zip"]
                updated_rules.extend(edited_zip_df.to_dict("records"))

                # Update session state
                st.session_state.rules = updated_rules

                # Log the override
                st.session_state.override_log.append(
                    {
                        "source": "user",
                        "field": "zip rules",
                        "old_value": str(zip_df.shape[0]),
                        "new_value": str(edited_zip_df.shape[0]),
                        "reason": "manual edit",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                updated = True
        else:
            st.info("No zip code rules defined yet.")

        # Add new zip rule
        st.subheader("Add New Zip Rule")
        with st.form("add_zip_rule"):
            condition = st.text_input("Condition (e.g., 'starts with 9')")
            action = st.text_input("Action (e.g., 'warehouse = CA-Oxnard-93030')")

            if st.form_submit_button("Add Rule"):
                if condition and action:
                    new_rule = {"type": "zip", "condition": condition, "action": action}

                    # Add to rules
                    st.session_state.rules.append(new_rule)

                    # Log the override
                    st.session_state.override_log.append(
                        {
                            "source": "user",
                            "field": "zip rule",
                            "old_value": "",
                            "new_value": f"{condition} -> {action}",
                            "reason": "manual addition",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

                    updated = True
                    st.success("Rule added successfully!")
                else:
                    st.error("Please fill in both condition and action.")

    with tab2:
        st.subheader("Bundle Rules")
        st.write("These rules determine which products belong to which bundle.")

        # Show existing bundle rules
        bundle_rules = [rule for rule in rules if rule["type"] == "bundle"]

        if bundle_rules:
            bundle_df = pd.DataFrame(bundle_rules)
            edited_bundle_df = st.data_editor(
                bundle_df, use_container_width=True, num_rows="dynamic", key="bundle_rules_editor"
            )

            # Check if rules were updated
            if not bundle_df.equals(edited_bundle_df):
                # Update rules
                updated_rules = [rule for rule in rules if rule["type"] != "bundle"]
                updated_rules.extend(edited_bundle_df.to_dict("records"))

                # Update session state
                st.session_state.rules = updated_rules

                # Log the override
                st.session_state.override_log.append(
                    {
                        "source": "user",
                        "field": "bundle rules",
                        "old_value": str(bundle_df.shape[0]),
                        "new_value": str(edited_bundle_df.shape[0]),
                        "reason": "manual edit",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                updated = True
        else:
            st.info("No bundle rules defined yet.")

        # Show bundles
        st.subheader("Bundles")

        if bundles:
            bundle_items = []
            for name, fruits in bundles.items():
                bundle_items.append({"Bundle Name": name, "Contents": ", ".join(fruits)})

            bundle_items_df = pd.DataFrame(bundle_items)
            edited_bundle_items_df = st.data_editor(
                bundle_items_df, use_container_width=True, num_rows="dynamic", key="bundles_editor"
            )

            # Check if bundles were updated
            if not bundle_items_df.equals(edited_bundle_items_df):
                # Update bundles
                updated_bundles = {}
                for _, row in edited_bundle_items_df.iterrows():
                    bundle_name = row["Bundle Name"]
                    contents = [item.strip() for item in row["Contents"].split(",")]
                    updated_bundles[bundle_name] = contents

                # Update session state
                st.session_state.bundles = updated_bundles

                # Log the override
                st.session_state.override_log.append(
                    {
                        "source": "user",
                        "field": "bundles",
                        "old_value": str(len(bundles)),
                        "new_value": str(len(updated_bundles)),
                        "reason": "manual edit",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                updated = True
        else:
            st.info("No bundles defined yet.")

        # Add new bundle
        st.subheader("Add New Bundle")
        with st.form("add_bundle"):
            bundle_name = st.text_input("Bundle Name")
            bundle_contents = st.text_input("Contents (comma-separated)")

            if st.form_submit_button("Add Bundle"):
                if bundle_name and bundle_contents:
                    contents = [item.strip() for item in bundle_contents.split(",")]

                    # Add to bundles
                    st.session_state.bundles[bundle_name] = contents

                    # Log the override
                    st.session_state.override_log.append(
                        {
                            "source": "user",
                            "field": "bundle",
                            "old_value": "",
                            "new_value": f"{bundle_name}: {bundle_contents}",
                            "reason": "manual addition",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

                    updated = True
                    st.success("Bundle added successfully!")
                else:
                    st.error("Please fill in both bundle name and contents.")

    with tab3:
        st.subheader("Priority Rules")
        st.write("These rules determine order priority based on tags or other criteria.")

        # Show existing priority rules
        priority_rules = [rule for rule in rules if rule["type"] == "priority"]

        if priority_rules:
            priority_df = pd.DataFrame(priority_rules)
            edited_priority_df = st.data_editor(
                priority_df,
                use_container_width=True,
                num_rows="dynamic",
                key="priority_rules_editor",
            )

            # Check if rules were updated
            if not priority_df.equals(edited_priority_df):
                # Update rules
                updated_rules = [rule for rule in rules if rule["type"] != "priority"]
                updated_rules.extend(edited_priority_df.to_dict("records"))

                # Update session state
                st.session_state.rules = updated_rules

                # Log the override
                st.session_state.override_log.append(
                    {
                        "source": "user",
                        "field": "priority rules",
                        "old_value": str(priority_df.shape[0]),
                        "new_value": str(edited_priority_df.shape[0]),
                        "reason": "manual edit",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

                updated = True
        else:
            st.info("No priority rules defined yet.")

        # Add new priority rule
        st.subheader("Add New Priority Rule")
        with st.form("add_priority_rule"):
            condition = st.text_input("Condition (e.g., 'P1')")
            action = st.text_input("Action (e.g., 'priority = high')")

            if st.form_submit_button("Add Rule"):
                if condition and action:
                    new_rule = {"type": "priority", "condition": condition, "action": action}

                    # Add to rules
                    st.session_state.rules.append(new_rule)

                    # Log the override
                    st.session_state.override_log.append(
                        {
                            "source": "user",
                            "field": "priority rule",
                            "old_value": "",
                            "new_value": f"{condition} -> {action}",
                            "reason": "manual addition",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )

                    updated = True
                    st.success("Rule added successfully!")
                else:
                    st.error("Please fill in both condition and action.")

    with tab4:
        st.subheader("Override Log")
        st.write("History of rule changes and overrides.")

        if override_log:
            log_df = pd.DataFrame(override_log)
            st.dataframe(log_df, use_container_width=True)
        else:
            st.info("No overrides logged yet.")

    return updated
