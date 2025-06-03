from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st


def render_header():
    """Render the application header"""
    st.title("Mixy Matchi Fulfillment Assistant")
    st.markdown(
        """
    This application helps assign customer fruit orders to fulfillment centers using:
    - Inventory optimization
    - Shortage detection and substitution suggestions
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


def render_inventory_analysis(processed_orders, inventory_df):
    """
    Render inventory analysis showing current inventory and projected remaining inventory after orders
    
    Args:
        processed_orders: DataFrame of processed orders
        inventory_df: DataFrame of inventory
    """
    if processed_orders is None or inventory_df is None:
        st.warning("No data available for inventory analysis")
        return
            
    # Only show Inventory Changes
    st.subheader("Inventory Changes")
    # Remove the tabs, we'll only show the changes
    
    # Process inventory data
    # Ensure inventory_df has the necessary columns
    if 'Sku' not in inventory_df.columns or 'WarehouseName' not in inventory_df.columns:
        with inv_tab1:
            st.error("Inventory data is missing required columns (Sku, WarehouseName)")
        return
    
    # Group inventory by SKU and warehouse
    agg_dict = {}
    
    # Check which columns are available for aggregation
    if 'AvailableQty' in inventory_df.columns:
        agg_dict['AvailableQty'] = 'sum'
    
    if 'Balance' in inventory_df.columns:
        # Use max aggregation for Balance to avoid issues with multiple rows per SKU
        agg_dict['Balance'] = 'max'
        
    # Make sure we have at least one aggregation column
    if not agg_dict:
        st.error("Inventory data is missing required quantity columns (AvailableQty or Balance)")
        return
        
    inventory_summary = inventory_df.groupby(['WarehouseName', 'Sku']).agg(agg_dict).reset_index()
    
    # Ensure numeric columns are properly formatted as numbers
    inventory_summary['AvailableQty'] = pd.to_numeric(inventory_summary['AvailableQty'], errors='coerce').fillna(0)
    inventory_summary['Balance'] = pd.to_numeric(inventory_summary['Balance'], errors='coerce').fillna(0)
    
    # Calculate order quantities by SKU
    if 'sku' in processed_orders.columns and 'Transaction Quantity' in processed_orders.columns:
        order_quantities = processed_orders.groupby('sku').agg({
            'Transaction Quantity': 'sum',
            'Fulfillment Center': 'first'
        }).reset_index()
        
        # Create a copy of inventory for projected calculations
        projected_inventory = inventory_summary.copy()
        
        # Map fulfillment centers to warehouse names
        fc_to_warehouse = {
            'Moorpark': 'CA-Moorpark-93021',
            'CA-Moorpark-93021': 'CA-Moorpark-93021',
            'Oxnard': 'CA-Oxnard-93030',
            'CA-Oxnard-93030': 'CA-Oxnard-93030',
            'Wheeling': 'IL-Wheeling-60090',
            'IL-Wheeling-60090': 'IL-Wheeling-60090'
        }
        
        # Create a new column for projected remaining balance
        # Use Balance if available, otherwise use AvailableQty
        if 'Balance' in projected_inventory.columns and projected_inventory['Balance'].sum() > 0:
            projected_inventory['Before Order'] = projected_inventory['Balance']
            projected_inventory['After Order'] = projected_inventory['Balance']
        elif 'AvailableQty' in projected_inventory.columns:
            projected_inventory['Before Order'] = projected_inventory['AvailableQty']
            projected_inventory['After Order'] = projected_inventory['AvailableQty']
        else:
            st.error("No valid quantity columns found in inventory data")
            return
        
        # Update projected remaining based on orders
        for _, order_row in order_quantities.iterrows():
            sku = order_row['sku']
            qty = order_row['Transaction Quantity']
            fc = order_row['Fulfillment Center']
            
            # Convert fulfillment center to warehouse name
            warehouse = fc_to_warehouse.get(fc, fc)
            
            # Find matching inventory row
            matching_rows = projected_inventory[
                (projected_inventory['Sku'] == sku) & 
                (projected_inventory['WarehouseName'] == warehouse)
            ]
            
            if not matching_rows.empty:
                idx = matching_rows.index[0]
                # Update after order quantity
                current = projected_inventory.loc[idx, 'After Order']
                projected_inventory.loc[idx, 'After Order'] = max(0, current - qty)
        
        # Calculate the change in inventory
        projected_inventory['Change'] = projected_inventory['After Order'] - projected_inventory['Before Order']
        
        # Only display inventory changes
        # Filter to only show items with changes
        changes_df = projected_inventory[projected_inventory['Change'] < 0].copy()
        changes_df['Change'] = changes_df['Change'].abs()  # Make positive for display

        if not changes_df.empty:
            st.write("Inventory items affected by orders:")
            st.dataframe(
                changes_df[['WarehouseName', 'Sku', 'AvailableQty', 'Before Order', 'After Order', 'Change']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    'WarehouseName': st.column_config.TextColumn('Warehouse'),
                    'Sku': st.column_config.TextColumn('SKU'),
                    'AvailableQty': st.column_config.NumberColumn('Available Qty'),
                    'Before Order': st.column_config.NumberColumn('Current Balance'),
                    'After Order': st.column_config.NumberColumn('Remaining Balance'),
                    'Change': st.column_config.NumberColumn('Quantity Used')
                }
            )
            
            # Create a bar chart showing the top items with the most changes
            top_changes = changes_df.sort_values('Change', ascending=False).head(10)
            
            if not top_changes.empty:
                fig = px.bar(
                    top_changes,
                    x='Sku',
                    y='Change',
                    title='Top 10 SKUs by Quantity Used',
                    color='WarehouseName',
                    labels={'Change': 'Quantity Used', 'Sku': 'SKU', 'Before Order': 'Before Order', 'After Order': 'After Order'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inventory changes detected from the orders")
    else:
        st.dataframe(inventory_summary[['WarehouseName', 'Sku', 'AvailableQty', 'Balance']])
        st.warning("Cannot calculate projected inventory - order data is missing required columns.")
        st.warning("Cannot calculate inventory changes - order data is missing required columns.")

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
        
    # Create a copy of processed_orders to avoid modifying the original
    processed_orders = processed_orders.copy()
    
    # Ensure externalorderid and id columns are string type to prevent data type mismatch
    for id_col in ['externalorderid', 'id']:
        if id_col in processed_orders.columns:
            processed_orders[id_col] = processed_orders[id_col].astype(str)

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
