import logging

import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode

logger = logging.getLogger(__name__)


def render_header():
    """Render the application header"""
    st.title("Fulfillment Assistant")
    st.markdown(
        """
    This application helps assign customer fruit orders to fulfillment centers using:
    - Inventory optimization
    - Shortage detection and substitution suggestions
    """
    )


def render_workflow_status():
    """Render the workflow status with statistics"""
    # Check workflow state
    has_data = st.session_state.orders_df is not None and st.session_state.inventory_df is not None
    workflow_initialized = st.session_state.get("workflow_initialized", False)
    has_staging = "staging_processor" in st.session_state and st.session_state.staging_processor

    # Get statistics
    if has_staging and workflow_initialized:
        try:
            staging_processor = st.session_state.staging_processor
            inventory_calcs = staging_processor.get_inventory_calculations()
            summary = inventory_calcs.get("staging_summary", {})

            orders_in_processing = summary.get("orders_in_processing", 0)
            orders_staged = summary.get("staged_orders", 0)
            total_orders_processed = orders_in_processing + orders_staged

            len(inventory_calcs.get("inventory_minus_processing", {}))
            len(inventory_calcs.get("inventory_minus_staged", {}))

            # Calculate available inventory statistics
            inventory_minus_staged_dict = inventory_calcs.get("inventory_minus_staged", {})
            available_items = sum(
                1 for balance in inventory_minus_staged_dict.values() if balance > 0
            )
            total_available_balance = sum(
                max(0, balance) for balance in inventory_minus_staged_dict.values()
            )

        except Exception as e:
            logger.warning(f"Error getting inventory calculations: {e}")
            orders_in_processing = 0
            orders_staged = 0
            total_orders_processed = 0
            available_items = 0
            total_available_balance = 0
    else:
        orders_in_processing = 0
        orders_staged = 0
        total_orders_processed = 0
        available_items = 0
        total_available_balance = 0

    # Progress calculation
    completed_steps = 0
    total_steps = 7

    if has_data:
        completed_steps += 1
    if workflow_initialized:
        completed_steps += 2  # Parse + Process
    if orders_in_processing > 0 or orders_staged > 0:
        completed_steps += 1
    if orders_staged > 0:
        completed_steps += 1

    progress_percentage = completed_steps / total_steps

    with st.expander(f"📋 Workflow Status ({progress_percentage:.0%} complete)", expanded=False):
        # Show key metrics first
        if workflow_initialized and (orders_in_processing > 0 or orders_staged > 0):
            st.markdown("**📊 Current Status Overview:**")
            metric_cols = st.columns(5)

            with metric_cols[0]:
                st.metric("📋 Processing", orders_in_processing)
            with metric_cols[1]:
                st.metric("🏷️ Staged", orders_staged)
            with metric_cols[2]:
                st.metric("📦 Available Items", available_items)
            with metric_cols[3]:
                st.metric("📊 Total Available", f"{total_available_balance:.0f}")
            with metric_cols[4]:
                recalculation_ready = orders_staged > 0 and orders_in_processing > 0
                st.metric("🔄 Ready to Recalc", "✅ Yes" if recalculation_ready else "❌ No")

        # Workflow steps with detailed descriptions
        workflow_steps = [
            {
                "step": "📤 Upload Orders/Inventory",
                "status": "✅" if has_data else "⏳",
                "description": f"Orders: {len(st.session_state.orders_df) if st.session_state.orders_df is not None else 0} | Inventory: {len(st.session_state.inventory_df) if st.session_state.inventory_df is not None else 0}",
                "completed": has_data,
            },
            {
                "step": "🔧 Parse/Normalize Data",
                "status": "✅" if workflow_initialized else "⏳",
                "description": f"SKU mappings loaded: {'Yes' if st.session_state.sku_mappings else 'No'}",
                "completed": workflow_initialized,
            },
            {
                "step": "⚙️ Process with SKU Mapping & Inventory",
                "status": "✅" if workflow_initialized else "⏳",
                "description": f"Total orders processed: {total_orders_processed}",
                "completed": workflow_initialized,
            },
            {
                "step": "📊 Orders in Processing & Inventory Calculations",
                "status": "✅" if orders_in_processing > 0 or orders_staged > 0 else "⏳",
                "description": f"Processing: {orders_in_processing} orders | Available inventory: {available_items} items",
                "completed": orders_in_processing > 0 or orders_staged > 0,
            },
            {
                "step": "🏷️ Move Orders to Staging",
                "status": "✅" if orders_staged > 0 else "⏳",
                "description": f"Staged: {orders_staged} orders | Remaining in processing: {orders_in_processing}",
                "completed": orders_staged > 0,
            },
            {
                "step": "🔄 Change Bundle Rules",
                "status": "🔄" if orders_staged > 0 else "⏳",
                "description": "Edit bundle components via warehouse-specific Google Sheets links",
                "completed": False,
                "in_progress": orders_staged > 0,
            },
            {
                "step": "⚡ Apply Changes to Processing Queue",
                "status": "🔄" if orders_staged > 0 else "⏳",
                "description": f"Using Available for Recalculation ({available_items} items with {total_available_balance:.0f} total units)",
                "completed": False,
                "in_progress": orders_staged > 0,
            },
        ]

        # Display workflow in columns with descriptions
        st.markdown("**📋 Workflow Steps:**")
        cols = st.columns(7)

        for i, step_info in enumerate(workflow_steps):
            with cols[i]:
                # Status icon and step name
                status_icon = step_info["status"]
                step_name = step_info["step"].split(" ", 1)[1]  # Remove emoji

                # Color coding
                if step_info["completed"]:
                    st.success(f"{status_icon} {step_name}")
                elif step_info.get("in_progress", False):
                    st.info(f"{status_icon} {step_name}")
                else:
                    st.warning(f"{status_icon} {step_name}")

                # Description
                st.caption(step_info["description"])

        # Progress bar
        st.progress(progress_percentage)

        # Next action guidance with specific links
        if workflow_initialized:
            if orders_staged > 0 and orders_in_processing > 0:
                st.success("🔄 **Ready for Recalculation!**")
                st.info(
                    """
                **Next Steps:**
                1. 📝 Edit bundle rules (optional):
                   - [📊 Edit Oxnard SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=549145618) (Sheet: INPUT_bundles_cvr_oxnard)
                   - [📊 Edit Wheeling SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=0) (Sheet: INPUT_bundles_cvr_wheeling)
                2. 🔄 Go to **📦 Inventory** → **Available for Recalculation** tab
                3. ⚡ Click **'Recalculate All Processing Orders'** button

                This will use your **Available for Recalculation** inventory (Initial - Staged = {available_items} items)
                to recalculate the {orders_in_processing} orders still in processing.
                """.format(
                        available_items=available_items, orders_in_processing=orders_in_processing
                    )
                )
            elif orders_in_processing > 0 and orders_staged == 0:
                st.info(
                    "🏷️ **Next Step:** Select orders in **📜 Orders** tab and move to staging to protect their inventory allocation."
                )
            elif total_orders_processed == 0:
                st.info("📤 **Next Step:** Upload orders and inventory files using the sidebar.")
            else:
                st.success("✅ **Workflow complete!** All orders have been processed.")
        else:
            st.warning(
                "⚠️ **Get Started:** Upload your orders and inventory files to begin the workflow."
            )


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

    st.markdown("**📊 Inventory Analysis**")

    # Create tabs for different inventory views with Grouped Shortages as first tab
    inv_tab_shortages, inv_tab1, inv_tab2 = st.tabs(
        ["⚠️ Grouped Shortages", "📦 Inventory minus Orders", "🎯 Inventory minus Staged Orders"]
    )

    # Initialize staged_inventory with initial inventory data
    staged_inventory = inventory_df.copy()
    if "Balance" in staged_inventory.columns:
        staged_inventory["Before Staged"] = staged_inventory["Balance"]
        staged_inventory["After Staged"] = staged_inventory["Balance"]
    elif "AvailableQty" in staged_inventory.columns:
        staged_inventory["Before Staged"] = staged_inventory["AvailableQty"]
        staged_inventory["After Staged"] = staged_inventory["AvailableQty"]

    # First tab: Grouped Shortages
    with inv_tab_shortages:
        st.markdown("**⚠️ Inventory Shortages by Group**")

        # Check if shortage information is available in session state
        if "shortage_summary" in st.session_state and not st.session_state.shortage_summary.empty:
            shortage_summary = st.session_state.shortage_summary
            shortage_count = len(shortage_summary)

            st.markdown(f"**Found {shortage_count} items with inventory shortages**")

            # Find SKU column using flexible matching
            sku_col = None
            for col in shortage_summary.columns:
                if any(keyword in col.lower() for keyword in ["sku", "item", "product", "part"]):
                    sku_col = col
                    break

            # Find order column using flexible matching
            order_col = None
            for col in shortage_summary.columns:
                if any(keyword in col.lower() for keyword in ["order", "ordernumber", "orderid"]):
                    order_col = col
                    break

            # Display grouped by SKU
            if sku_col:
                st.markdown("**Shortages by SKU**")
                sku_grouped = shortage_summary.groupby(sku_col).size().reset_index()
                sku_grouped.columns = [sku_col, "Count"]
                sku_grouped = sku_grouped.sort_values("Count", ascending=False)

                # Use aggrid for better display
                create_aggrid_table(
                    sku_grouped,
                    height=300,
                    key="shortage_sku_grid",
                    theme="alpine",
                    selection_mode="multiple",
                    show_hints=False,
                    enable_enterprise_modules=True,
                )

            # Display grouped by Order
            if order_col:
                st.markdown("**Shortages by Order**")
                order_grouped = shortage_summary.groupby(order_col).size().reset_index()
                order_grouped.columns = [order_col, "Count"]
                order_grouped = order_grouped.sort_values("Count", ascending=False)

                # Use aggrid for better display
                create_aggrid_table(
                    order_grouped,
                    height=300,
                    key="shortage_order_grid",
                    theme="alpine",
                    selection_mode="multiple",
                    show_hints=False,
                    enable_enterprise_modules=True,
                )

            # Show full shortage details
            st.markdown("**Complete Shortage Details**")
            create_aggrid_table(
                shortage_summary,
                height=400,
                key="full_shortage_grid",
                theme="alpine",
                selection_mode="multiple",
                show_hints=False,
                enable_enterprise_modules=True,
                enable_sidebar=True,
            )
        else:
            st.info("No inventory shortages detected or shortage data is not available.")

    with inv_tab1:
        st.markdown("**📦 Current Inventory minus Orders in Processing**")
        # Process inventory data
        if "Sku" not in inventory_df.columns or "WarehouseName" not in inventory_df.columns:
            st.error("Inventory data is missing required columns (Sku, WarehouseName)")
            return

        # Group inventory by SKU and warehouse
        agg_dict = {}

        # Check which columns are available for aggregation
        if "AvailableQty" in inventory_df.columns:
            agg_dict["AvailableQty"] = "sum"

        if "Balance" in inventory_df.columns:
            agg_dict["Balance"] = "max"

        if not agg_dict:
            st.error(
                "Inventory data is missing required quantity columns (AvailableQty or Balance)"
            )
            return

        # Calculate order quantities by SKU
        if "sku" in processed_orders.columns and "Transaction Quantity" in processed_orders.columns:
            # Get only unstaged orders for this calculation
            orders_in_processing = (
                processed_orders[processed_orders["staged"] == False].copy()
                if "staged" in processed_orders.columns
                else processed_orders.copy()
            )

            order_quantities = (
                orders_in_processing.groupby(["sku", "Fulfillment Center"])
                .agg({"Transaction Quantity": "sum"})
                .reset_index()
            )

            # Map fulfillment centers to warehouse names
            fc_to_warehouse = {
                "Moorpark": "CA-Moorpark-93021",
                "CA-Moorpark-93021": "CA-Moorpark-93021",
                "Oxnard": "CA-Oxnard-93030",
                "CA-Oxnard-93030": "CA-Oxnard-93030",
                "Wheeling": "IL-Wheeling-60090",
                "IL-Wheeling-60090": "IL-Wheeling-60090",
            }

            # Create a new column for projected remaining balance
            if (
                "Balance" in projected_inventory.columns
                and projected_inventory["Balance"].sum() > 0
            ):
                projected_inventory["Before Order"] = projected_inventory["Balance"]
                projected_inventory["After Order"] = projected_inventory["Balance"]
            elif "AvailableQty" in projected_inventory.columns:
                projected_inventory["Before Order"] = projected_inventory["AvailableQty"]
                projected_inventory["After Order"] = projected_inventory["AvailableQty"]

            # Update projected remaining based on orders
            for _, order_row in order_quantities.iterrows():
                sku = order_row["sku"]
                qty = order_row["Transaction Quantity"]
                fc = order_row["Fulfillment Center"]

                # Convert fulfillment center to warehouse name
                warehouse = fc_to_warehouse.get(fc, fc)

                # Find matching inventory row
                matching_rows = projected_inventory[
                    (projected_inventory["Sku"] == sku)
                    & (projected_inventory["WarehouseName"] == warehouse)
                ]

                if not matching_rows.empty:
                    idx = matching_rows.index[0]
                    # Update after order quantity
                    current = projected_inventory.loc[idx, "After Order"]
                    projected_inventory.loc[idx, "After Order"] = max(0, current - qty)

            # Calculate the change in inventory
            projected_inventory["Change"] = (
                projected_inventory["After Order"] - projected_inventory["Before Order"]
            )

            # Filter to only show items with changes
            changes_df = projected_inventory[projected_inventory["Change"] < 0].copy()
            changes_df["Change"] = changes_df["Change"].abs()

            if not changes_df.empty:
                st.write(
                    f"Found {len(changes_df)} items with inventory changes from orders in processing:"
                )

                # Create ag-Grid table
                create_aggrid_table(
                    changes_df,
                    height=400,
                    key="inventory_changes_grid",
                    theme="alpine",
                    selection_mode="multiple",
                    show_hints=False,
                    enable_enterprise_modules=True,
                    enable_sidebar=True,
                    enable_pivot=True,
                    enable_value_aggregation=True,
                    groupable=True,
                    filterable=True,
                )
            else:
                st.info("No inventory changes detected from orders in processing")
        else:
            st.warning(
                "Cannot calculate inventory changes - order data is missing required columns"
            )

    with inv_tab2:
        st.markdown("**🎯 Current Inventory minus Staged Orders**")

        # Get only staged orders
        staged_orders = (
            processed_orders[processed_orders["staged"] == True].copy()
            if "staged" in processed_orders.columns
            else pd.DataFrame()
        )

        if staged_orders.empty:
            st.info("No staged orders to calculate inventory impact")
            return

        # Calculate total quantity per SKU in staged orders
        staged_qty = (
            staged_orders.groupby(["sku", "Fulfillment Center"])
            .agg({"Transaction Quantity": "sum"})
            .reset_index()
        )

        # Update inventory based on staged orders
        for _, row in staged_qty.iterrows():
            sku = row["sku"]
            fc = row["Fulfillment Center"]
            qty = row["Transaction Quantity"]

            # Convert fulfillment center to warehouse name
            warehouse = fc_to_warehouse.get(fc, fc)

            # Find matching inventory row
            matching_rows = staged_inventory[
                (staged_inventory["Sku"] == sku) & (staged_inventory["WarehouseName"] == warehouse)
            ]

            if not matching_rows.empty:
                idx = matching_rows.index[0]
                # Update after staged quantity
                current = staged_inventory.loc[idx, "After Staged"]
                staged_inventory.loc[idx, "After Staged"] = max(0, current - qty)

        # Calculate the change in inventory from staged orders
        staged_inventory["Change from Staged"] = (
            staged_inventory["After Staged"] - staged_inventory["Before Staged"]
        )

        # Filter to only show items affected by staging
        staged_changes = staged_inventory[staged_inventory["Change from Staged"] < 0].copy()
        staged_changes["Change from Staged"] = staged_changes["Change from Staged"].abs()

        if not staged_changes.empty:
            st.write(
                f"Found {len(staged_changes)} items with inventory changes from staged orders:"
            )

            # Create ag-Grid table
            create_aggrid_table(
                staged_changes,
                height=400,
                key="staged_inventory_changes_grid",
                theme="alpine",
                selection_mode="multiple",
                show_hints=False,
                enable_enterprise_modules=True,
                enable_sidebar=True,
                enable_pivot=True,
                enable_value_aggregation=True,
                groupable=True,
                filterable=True,
            )
        else:
            st.info("No inventory changes detected from staged orders")

    # Display complete inventory
    create_aggrid_table(
        full_inventory_df,
        height=400,
        key="complete_inventory_grid",
        theme="alpine",
        selection_mode="multiple",
        show_hints=False,
        enable_enterprise_modules=True,
        enable_sidebar=True,
        enable_pivot=True,
        enable_value_aggregation=True,
        groupable=True,
        filterable=True,
    )


def render_summary_dashboard(
    processed_orders, inventory_df, processing_stats=None, warehouse_performance=None
):
    """
    Enhanced summary dashboard with comprehensive analytics and decision-making metrics

    Args:
        processed_orders: DataFrame of processed orders
        inventory_df: DataFrame of inventory
        processing_stats: Dictionary of processing statistics
        warehouse_performance: Dictionary of warehouse performance metrics (kept for compatibility)
    """
    if processed_orders is None or inventory_df is None:
        st.warning("No data available for dashboard")
        return

    # Note: warehouse_performance parameter kept for compatibility

    # Create a copy of processed_orders to avoid modifying the original
    processed_orders = processed_orders.copy()

    # Ensure externalorderid and id columns are string type to prevent data type mismatch
    for id_col in ["externalorderid", "id"]:
        if id_col in processed_orders.columns:
            processed_orders[id_col] = processed_orders[id_col].astype(str)

    # Enhanced Key Metrics Section
    st.subheader("📊 Key Performance Indicators")

    # First row - Core metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_orders = (
            len(processed_orders["ordernumber"].unique())
            if "ordernumber" in processed_orders.columns
            else 0
        )
        st.metric("Unique Orders", f"{total_orders:,}")

    with col2:
        total_items = len(processed_orders)
        st.metric("Total Line Items", f"{total_items:,}")

    with col3:
        fulfillment_centers = (
            processed_orders["Fulfillment Center"].nunique()
            if "Fulfillment Center" in processed_orders.columns
            else 0
        )
        st.metric("Active Warehouses", fulfillment_centers)

    with col4:
        issues = (
            processed_orders[processed_orders["Issues"] != ""].shape[0]
            if "Issues" in processed_orders.columns
            else 0
        )
        issue_rate = (
            round((issues / len(processed_orders)) * 100, 1) if len(processed_orders) > 0 else 0
        )
        st.metric(
            "Items with Issues",
            f"{issues:,}",
            delta=f"{issue_rate}% of total",
            delta_color="inverse",
        )

    # Second row - Processing efficiency metrics
    with col1:
        if processing_stats and "total_quantity_processed" in processing_stats:
            total_qty = processing_stats["total_quantity_processed"]
            st.metric("Total Quantity", f"{total_qty:,.0f}")
        else:
            quantities = (
                pd.to_numeric(processed_orders["Transaction Quantity"], errors="coerce")
                if "Transaction Quantity" in processed_orders.columns
                else pd.Series([0])
            )
            st.metric("Total Quantity", f"{quantities.sum():,.0f}")

    with col2:
        if processing_stats and "avg_quantity_per_item" in processing_stats:
            avg_qty = processing_stats["avg_quantity_per_item"]
            st.metric("Avg Qty/Item", f"{avg_qty:.1f}")
        else:
            quantities = (
                pd.to_numeric(processed_orders["Transaction Quantity"], errors="coerce")
                if "Transaction Quantity" in processed_orders.columns
                else pd.Series([1])
            )
            st.metric("Avg Qty/Item", f"{quantities.mean():.1f}")

    with col3:
        unique_skus = (
            len(processed_orders["sku"].unique()) if "sku" in processed_orders.columns else 0
        )
        st.metric("Unique SKUs", f"{unique_skus:,}")

    with col4:
        if processing_stats and "primary_fulfillment_center" in processing_stats:
            primary_fc = processing_stats["primary_fulfillment_center"]
            st.metric("Primary Warehouse", primary_fc)
        else:
            if "Fulfillment Center" in processed_orders.columns:
                primary_fc = (
                    processed_orders["Fulfillment Center"].mode().iloc[0]
                    if not processed_orders["Fulfillment Center"].mode().empty
                    else "Unknown"
                )
            else:
                primary_fc = "Unknown"
            st.metric("Primary Warehouse", primary_fc)

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

    # Processing Summary
    if processing_stats:
        with st.expander("📋 Detailed Processing Summary", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Order Processing Stats:**")
                st.write(f"- Total Orders: {processing_stats.get('total_orders', 0):,}")
                st.write(f"- Total Line Items: {processing_stats.get('total_line_items', 0):,}")
                st.write(f"- Unique SKUs: {processing_stats.get('unique_skus', 0):,}")
                st.write(
                    f"- Total Quantity: {processing_stats.get('total_quantity_processed', 0):,.0f}"
                )
                st.write(
                    f"- Average Qty/Item: {processing_stats.get('avg_quantity_per_item', 0):.2f}"
                )

            with col2:
                st.write("**Inventory & Issues:**")
                st.write(f"- Items with Issues: {processing_stats.get('items_with_issues', 0):,}")
                st.write(f"- Issue Rate: {processing_stats.get('issue_rate', 0):.2f}%")
                st.write(
                    f"- Total Inventory Items: {processing_stats.get('total_inventory_items', 0):,}"
                )
                st.write(f"- Zero Balance Items: {processing_stats.get('zero_balance_items', 0):,}")
                st.write(f"- Low Balance Items: {processing_stats.get('low_balance_items', 0):,}")

            # Timestamp
            if "processing_timestamp" in processing_stats:
                timestamp = pd.to_datetime(processing_stats["processing_timestamp"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                st.caption(f"Last updated: {timestamp}")


def render_orders_tab(processed_orders, shortage_summary=None):
    """Renders the Orders tab with data table and metrics"""
    st.markdown("**Orders in Processing**")

    # Get key prefix for dynamic keys to force UI refresh when necessary
    key_prefix = st.session_state.get("key_prefix", 0)

    # Always use the most up-to-date processed_orders from session state if available
    if "processed_orders" in st.session_state:
        processed_orders = st.session_state.processed_orders

    if processed_orders is not None and not processed_orders.empty:
        # First, ensure the staged column exists
        if "staged" not in processed_orders.columns:
            processed_orders["staged"] = False

        # Filter to only show unstaged orders (use boolean value to ensure correct filtering)
        display_orders = processed_orders[processed_orders["staged"] == False].copy()

        # Keep track of counts for reference
        total_orders = len(processed_orders)
        filtered_orders = len(display_orders)
        staged_orders = len(processed_orders[processed_orders["staged"] == True])

        # Don't show the staged column to the user
        if "staged" in display_orders.columns:
            display_orders = display_orders.drop(columns=["staged"])
        # Display key metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            total_orders = (
                len(display_orders["ordernumber"].unique())
                if "ordernumber" in display_orders.columns
                else 0
            )
            st.metric("📋 Unique Orders", f"{total_orders:,}")

        with col2:
            total_items = len(display_orders)
            st.metric("📦 Line Items", f"{total_items:,}")

        with col3:
            issues = (
                display_orders[display_orders["Issues"] != ""].shape[0]
                if "Issues" in display_orders.columns
                else 0
            )
            issue_rate = (
                round((issues / len(display_orders)) * 100, 1) if len(display_orders) > 0 else 0
            )
            st.metric(
                "⚠️ Order Lines with Issues",
                f"{issues:,}",
                delta=f"{issue_rate}% of total",
                delta_color="inverse",
            )

        with col4:
            fulfillment_centers = (
                display_orders["Fulfillment Center"].nunique()
                if "Fulfillment Center" in display_orders.columns
                else 0
            )
            st.metric("🏭 Warehouses Used", fulfillment_centers)

        with col5:
            # Show staged orders count and delta from total
            st.metric(
                "🏷️ Orders Staged",
                f"{staged_orders:,}",
                delta=f"of {total_orders:,} total" if total_orders > 0 else "",
            )

        # Add filters for orders with/without issues
        st.markdown("---")
        orders_without_issues = display_orders[display_orders["Issues"] == ""][
            "ordernumber"
        ].unique()
        orders_with_issues = display_orders[display_orders["Issues"] != ""]["ordernumber"].unique()

        # Initialize filter state in session state if not exists
        if "order_filter" not in st.session_state:
            st.session_state.order_filter = "all"

        # Add filter buttons
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            if st.button("📊 Show All Orders", key="show_all_orders"):
                st.session_state.order_filter = "all"
        with filter_col2:
            if st.button(f"✅ Show Orders Without Issues", key="show_without_issues"):
                st.session_state.order_filter = "without_issues"
        with filter_col3:
            if st.button(f"⚠️ Show Orders With Issues", key="show_with_issues"):
                st.session_state.order_filter = "with_issues"

        # Apply filters based on session state
        filtered_orders = display_orders.copy()
        if st.session_state.order_filter == "without_issues":
            # Get all order numbers
            all_order_numbers = display_orders["ordernumber"].unique()
            # Get order numbers that have any issues
            orders_with_any_issues = display_orders[display_orders["Issues"] != ""][
                "ordernumber"
            ].unique()
            # Orders without issues are those that don't appear in orders_with_any_issues
            orders_without_issues = set(all_order_numbers) - set(orders_with_any_issues)
            # Filter display_orders to show all lines for these orders
            filtered_orders = display_orders[
                display_orders["ordernumber"].isin(orders_without_issues)
            ]
        elif st.session_state.order_filter == "with_issues":
            # Get order numbers with issues
            order_numbers = display_orders[display_orders["Issues"] != ""]["ordernumber"].unique()
            # Filter display_orders to show all lines for these orders
            filtered_orders = display_orders[display_orders["ordernumber"].isin(order_numbers)]

        # Show inventory calculation: Initial - Staged = Available for Recalculation
        if (
            "staging_processor" in st.session_state
            and st.session_state.staging_processor
            and st.session_state.workflow_initialized
        ):
            try:
                inventory_calcs = st.session_state.staging_processor.get_inventory_calculations()
                inventory_minus_staged = inventory_calcs.get(
                    "inventory_minus_staged", pd.DataFrame()
                )

                # Handle DataFrame case
                if (
                    isinstance(inventory_minus_staged, pd.DataFrame)
                    and not inventory_minus_staged.empty
                ):
                    available_items = sum(
                        1
                        for balance in inventory_minus_staged["Balance"]
                        if isinstance(balance, (int, float)) and balance > 0
                    )
                    total_available_balance = sum(
                        max(0, balance)
                        for balance in inventory_minus_staged["Balance"]
                        if isinstance(balance, (int, float))
                    )
                else:
                    # Fallback for other cases
                    available_items = 0
                    total_available_balance = 0
                    logger.warning(
                        "Could not calculate inventory minus staged: Invalid data format"
                    )

                # Calculate initial and used in staging
                initial_inventory = (
                    st.session_state.initial_inventory
                    if "initial_inventory" in st.session_state
                    else None
                )
                used_items = 0
                used_units = 0
                if initial_inventory is not None and not initial_inventory.empty:
                    len(initial_inventory)
                    if "Balance" in initial_inventory.columns:
                        initial_inventory["Balance"].sum()
                    elif "AvailableQty" in initial_inventory.columns:
                        initial_inventory["AvailableQty"].sum()
                    # Used in staging: sum of (initial - available) for each SKU|warehouse
                    initial_lookup = {}
                    for _, row in initial_inventory.iterrows():
                        sku = row.get("Sku") or row.get("sku")
                        warehouse = (
                            row.get("warehouse") or row.get("WarehouseName") or row.get("Warehouse")
                        )
                        balance = (
                            row.get("balance")
                            if "balance" in row
                            else row.get("Balance")
                            if "Balance" in row
                            else row.get("AvailableQty")
                        )
                        if sku is not None and warehouse is not None and balance is not None:
                            initial_lookup[f"{sku}|{warehouse}"] = float(balance)
                    for key, balance in inventory_minus_staged.items():
                        if key in initial_lookup:
                            used = initial_lookup[key] - balance
                            if used > 0:
                                used_items += 1
                                used_units += used
            except Exception as e:
                logger.warning(f"Could not calculate inventory minus staged: {e}")

        # Use the most up-to-date shortage_summary (prefer session state over parameter)
        if "shortage_summary" in st.session_state and st.session_state.shortage_summary is not None:
            # Always use session state if it exists (even if empty - means no shortages after recalculation)
            shortage_summary = st.session_state.shortage_summary
            shortage_count = len(shortage_summary) if not shortage_summary.empty else 0
        elif shortage_summary is not None and not shortage_summary.empty:
            shortage_count = len(shortage_summary)
            # Update session state with shortage_summary for use in other components
            st.session_state.shortage_summary = shortage_summary
        else:
            shortage_count = 0

        if shortage_summary is not None and not shortage_summary.empty:
            # Find SKU column using flexible matching patterns
            sku_keywords = ["sku", "item", "product", "part"]
            sku_col = None

            # Find the first column that contains any of the SKU keywords
            for col in shortage_summary.columns:
                if any(keyword in col.lower() for keyword in sku_keywords):
                    sku_col = col
                    break

            # If still no match, try a case-insensitive exact match for 'sku'
            if not sku_col:
                for col in shortage_summary.columns:
                    if col.lower() == "sku":
                        sku_col = col
                        break

            # Calculate unique SKUs if we found a column
            if sku_col:
                unique_skus = shortage_summary[sku_col].nunique()
            else:
                # If still no match and the DataFrame has data, use the first column as fallback
                if not shortage_summary.empty and len(shortage_summary.columns) > 0:
                    unique_skus = shortage_summary[shortage_summary.columns[0]].nunique()
                else:
                    unique_skus = shortage_count  # Default to shortage_count if no columns found

            # Find order number column using flexible matching patterns
            order_keywords = [
                "order",
                "ordernumber",
                "order number",
                "orderid",
                "order id",
                "external",
            ]
            order_col = None

            # Find the first column that contains any of the order keywords
            for col in shortage_summary.columns:
                if any(keyword in col.lower() for keyword in order_keywords):
                    order_col = col
                    break

            # Calculate affected orders if we found a column
            if order_col:
                affected_orders = shortage_summary[order_col].nunique()
            else:
                # If we didn't find an order column but found a SKU column, assume 1 order per shortage
                affected_orders = shortage_count

            # Calculate items with multiple issues (duplicates)
            if "Issues" in filtered_orders.columns:
                multiple_issues_count = (
                    len(filtered_orders[filtered_orders["Issues"] != ""]) - shortage_count
                    if shortage_count > 0
                    else 0
                )
            else:
                multiple_issues_count = 0

            if shortage_count > 0:
                # Create a clearer, more structured shortage message
                st.info(
                    f"⚠️ INVENTORY SHORTAGES DETECTED:\n"
                    + f"• {shortage_count} inventory shortage instances\n"
                    + f"• {unique_skus} unique Warehouse SKUs affected\n"
                    + f"• {affected_orders} orders impacted"
                    + (
                        f"\n• {multiple_issues_count} additional shortage flags (line items with multiple shortage types)"
                        if multiple_issues_count > 0
                        else ""
                    )
                )

                # Add an expander with detailed shortage information including fulfillment center and order IDs
                with st.expander("📋 View Detailed Shortages by Fulfillment Center", expanded=False):
                    if shortage_summary is not None and not shortage_summary.empty:
                        # Create a detailed view of shortages with fulfillment center info
                        detailed_view = shortage_summary.copy()

                        # Find important columns using flexible matching
                        fc_col = next(
                            (
                                col
                                for col in detailed_view.columns
                                if any(
                                    keyword in col.lower()
                                    for keyword in [
                                        "fulfillment",
                                        "center",
                                        "warehouse",
                                        "location",
                                    ]
                                )
                            ),
                            None,
                        )
                        sku_col = next(
                            (
                                col
                                for col in detailed_view.columns
                                if any(
                                    keyword in col.lower() for keyword in ["sku", "item", "product"]
                                )
                                and not any(kw in col.lower() for kw in ["shopify", "related"])
                            ),
                            None,
                        )
                        order_col = next(
                            (
                                col
                                for col in detailed_view.columns
                                if any(
                                    keyword in col.lower()
                                    for keyword in ["order", "ordernumber", "orderid"]
                                )
                            ),
                            None,
                        )
                        shopify_cols = [
                            col
                            for col in detailed_view.columns
                            if any(keyword in col.lower() for keyword in ["shopify", "related"])
                        ]

                        # Group by fulfillment center and warehouse SKU
                        if fc_col and sku_col and order_col:
                            # Group data by fulfillment center and SKU
                            grouped = detailed_view.groupby([fc_col, sku_col])

                            # Create result rows with grouped data
                            result_rows = []
                            for (fc, sku), group in grouped:
                                row = {"fulfillment_center": fc, "warehouse_sku": sku}

                                # Collect all related Shopify SKUs
                                for col in shopify_cols:
                                    unique_values = group[col].dropna().unique()
                                    if len(unique_values) > 0:
                                        row["shopify_sku"] = ", ".join(
                                            sorted(unique_values.astype(str))
                                        )
                                    else:
                                        row["shopify_sku"] = ""

                                # Add affected order IDs
                                unique_orders = sorted(
                                    group[order_col].dropna().unique().astype(str)
                                )
                                row["order_id"] = ", ".join(unique_orders)

                                # Add counts
                                row["affected_orders"] = len(unique_orders)
                                row["line_items"] = len(group)

                                result_rows.append(row)

                            # Convert to DataFrame and reorder columns
                            if result_rows:
                                grouped_df = pd.DataFrame(result_rows)

                                # Reorder columns for better display
                                column_order = [
                                    "fulfillment_center",
                                    "warehouse_sku",
                                    "shopify_sku",
                                    "order_id",
                                ]

                                # Display the grouped table
                                st.dataframe(
                                    grouped_df,
                                    height=400,
                                    use_container_width=True,
                                    hide_index=True,
                                )
                            else:
                                st.info("No detailed shortage information available")
                        else:
                            # Fallback if we can't find the needed columns
                            st.info(
                                "Could not identify necessary columns for grouping in shortage data"
                            )
                    else:
                        st.info("No shortage data available")

        # Create AgGrid table
        gb = GridOptionsBuilder.from_dataframe(filtered_orders)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb.build()

        # Use a stable key that only changes when data actually changes
        grid_key = f"orders_grid_{key_prefix}_{len(filtered_orders)}"

        grid_response = create_aggrid_table(
            filtered_orders,
            height=600,
            selection_mode="multiple",
            enable_enterprise_modules=True,
            theme="alpine",
            key=grid_key,
        )

        # Show selected rows info and staging option
        selected_rows_data = grid_response["grid_response"]["selected_rows"]
        if selected_rows_data is not None and len(selected_rows_data) > 0:
            # Convert DataFrame to list of dictionaries if needed
            if isinstance(selected_rows_data, pd.DataFrame):
                selected_rows = selected_rows_data.to_dict("records")
            else:
                selected_rows = selected_rows_data
            st.session_state.selected_orders = selected_rows

            # Display info about selection
            st.success(
                f"✅ Selected {len(selected_rows)} rows out of {len(display_orders)} total rows"
            )

            # Calculate detailed metrics for selection
            selected_unique_orders = len(set([row.get("ordernumber", "") for row in selected_rows]))
            selected_line_items = len(selected_rows)
            selected_total_qty = sum(
                [float(row.get("Transaction Quantity", 0)) for row in selected_rows]
            )
            selected_with_issues = len(
                [row for row in selected_rows if row.get("Issues", "") != ""]
            )
            selected_orders_with_issues = len(
                set(
                    [
                        row.get("ordernumber", "")
                        for row in selected_rows
                        if row.get("Issues", "") != ""
                    ]
                )
            )
            selected_warehouses = len(
                set([row.get("Fulfillment Center", "") for row in selected_rows])
            )

            # Show detailed selection stats
            cols = st.columns(4)
            with cols[0]:
                st.metric("📦 Selected Items", selected_line_items)
            with cols[1]:
                st.metric("📋 Unique Orders", selected_unique_orders)
            with cols[2]:
                st.metric("🏭 Warehouses", selected_warehouses)
            with cols[3]:
                st.metric(
                    "⚠️ Items with Issues",
                    selected_with_issues,
                    delta=f"{selected_orders_with_issues} orders",
                )

            # Display additional statistics that the user requested
            st.info(
                f"📊 Summary: {selected_line_items} line items | {selected_unique_orders} unique orders | {selected_with_issues} items with issues | {selected_orders_with_issues} unique orders with issues"
            )

            # Smart Bundle Management section
            bundle_skus = [
                row.get("sku", "") for row in selected_rows if row.get("sku", "").startswith("f.")
            ]
            if bundle_skus:
                with st.expander("🔧 Smart Bundle Management", expanded=False):
                    st.markdown("**Bundle SKUs in Selection:**")
                    unique_bundles = list(set(bundle_skus))
                    for bundle_sku in unique_bundles:
                        st.code(bundle_sku)

                    # Show Available for Recalculation inventory status
                    if (
                        "staging_processor" in st.session_state
                        and st.session_state.staging_processor
                        and st.session_state.workflow_initialized
                    ):
                        try:
                            # Get detailed bundle availability analysis
                            with st.spinner("Analyzing bundle availability..."):
                                bundle_analysis = st.session_state.staging_processor.get_bundle_availability_analysis(
                                    unique_bundles
                                )

                            if "error" not in bundle_analysis:
                                st.markdown(
                                    "**📊 Smart Bundle Analysis (Available for Recalculation):**"
                                )

                                # Show summary metrics
                                summary = bundle_analysis["summary"]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("✅ Available Bundles", summary["available_bundles"])
                                with col2:
                                    st.metric(
                                        "⚠️ Constrained Bundles", summary["constrained_bundles"]
                                    )
                                with col3:
                                    st.metric("📦 Total Bundles", summary["total_bundles"])

                                # Show detailed analysis for each bundle
                                for bundle_sku, analysis in bundle_analysis["bundles"].items():
                                    status_icon = (
                                        "✅"
                                        if analysis["status"] == "available"
                                        else "⚠️"
                                        if analysis["status"] == "constrained"
                                        else "❌"
                                    )

                                    with st.expander(
                                        f"{status_icon} {bundle_sku} - Max possible: {analysis['max_possible_bundles']}",
                                        expanded=False,
                                    ):
                                        if (
                                            analysis["status"] == "constrained"
                                            and analysis["constraints"]
                                        ):
                                            st.error("**Constraints:**")
                                            for constraint in analysis["constraints"]:
                                                st.caption(f"• {constraint}")

                                        # Show component details
                                        if analysis["components"]:
                                            st.markdown("**Component Details:**")
                                            for comp in analysis["components"]:
                                                status_emoji = (
                                                    "🟢" if comp["status"] == "available" else "🔴"
                                                )
                                                st.caption(
                                                    f"{status_emoji} {comp['sku']} @ {comp['warehouse']}: {comp['available_qty']:.0f} available (need {comp['required_qty']:.0f})"
                                                )
                            else:
                                st.warning(f"Bundle analysis error: {bundle_analysis['error']}")

                        except ZeroDivisionError as zde:
                            st.warning(f"Division by zero in bundle analysis: {zde}")
                        except Exception as e:
                            st.warning(f"Could not load bundle analysis: {e}")

                    st.markdown("**Bundle Management Actions:**")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("📝 Edit Bundle Mappings"):
                            st.info(
                                """
                            **Edit Bundle Components:**
                            - [📊 Edit Oxnard SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=549145618) (Sheet: INPUT_bundles_cvr_oxnard)
                            - [📊 Edit Wheeling SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=0) (Sheet: INPUT_bundles_cvr_wheeling)

                            Edit bundle components before recalculation.
                            """
                            )

                    with col2:
                        if st.button("🎯 Stage & Edit Bundles"):
                            st.caption(
                                "Stage selected orders first, then edit bundles via Google Sheets"
                            )
                            # Initialize staging processor if needed
                            if st.session_state.staging_processor is None and st.session_state.get(
                                "workflow_initialized", False
                            ):
                                from utils.data_processor import DataProcessor

                                st.session_state.staging_processor = DataProcessor()
                                st.session_state.staging_processor.initialize_workflow(
                                    st.session_state.orders_df, st.session_state.inventory_df
                                )

                            if (
                                st.session_state.staging_processor is not None
                                and st.session_state.get("workflow_initialized", False)
                            ):
                                try:
                                    # Stage selected orders first
                                    order_indices = []
                                    for selected_row in selected_rows:
                                        for (
                                            idx,
                                            order_row,
                                        ) in (
                                            st.session_state.staging_processor.orders_in_processing.iterrows()
                                        ):
                                            if selected_row.get("ordernumber", "") == order_row.get(
                                                "ordernumber", ""
                                            ) and selected_row.get("sku", "") == order_row.get(
                                                "sku", ""
                                            ):
                                                order_indices.append(idx)
                                                break

                                    if order_indices:
                                        staging_result = st.session_state.staging_processor.stage_selected_orders(
                                            order_indices
                                        )
                                        if "error" not in staging_result:
                                            st.session_state.processed_orders = (
                                                st.session_state.staging_processor.orders_in_processing.copy()
                                            )
                                            st.success(
                                                "✅ Orders staged! Now edit bundle components using these Google Sheets links: \n- [Oxnard](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=549145618) | [Wheeling](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=0)"
                                            )
                                            st.rerun()
                                        else:
                                            st.error(f"Staging error: {staging_result['error']}")
                                    else:
                                        st.warning("No matching orders found for staging")
                                except Exception as e:
                                    st.error(f"Error staging orders: {e}")
                            else:
                                st.error("Staging processor not available")

                    with col3:
                        if st.button("🔄 Smart Recalculate"):
                            st.caption(
                                "Uses Available for Recalculation (Initial - Staged) inventory"
                            )
                            # Initialize staging processor if needed
                            if st.session_state.staging_processor is None and st.session_state.get(
                                "workflow_initialized", False
                            ):
                                from utils.data_processor import DataProcessor

                                st.session_state.staging_processor = DataProcessor()
                                st.session_state.staging_processor.initialize_workflow(
                                    st.session_state.orders_df, st.session_state.inventory_df
                                )

                            if (
                                st.session_state.staging_processor is not None
                                and st.session_state.get("workflow_initialized", False)
                            ):
                                try:
                                    with st.spinner(
                                        "Smart recalculating with Available for Recalculation inventory..."
                                    ):
                                        # Get before state
                                        before_orders = len(
                                            st.session_state.staging_processor.orders_in_processing
                                        )
                                        before_staged = len(
                                            st.session_state.staging_processor.staged_orders
                                        )

                                        recalc_result = (
                                            st.session_state.staging_processor.recalculate_orders_with_updated_inventory()
                                        )

                                        if "error" in recalc_result:
                                            st.error(
                                                f"Recalculation failed: {recalc_result['error']}"
                                            )
                                        else:
                                            # Clear existing shortage data before updating to prevent accumulation
                                            st.session_state.shortage_summary = pd.DataFrame()
                                            st.session_state.grouped_shortage_summary = (
                                                pd.DataFrame()
                                            )

                                            # Update ALL session state data from recalculation results
                                            st.session_state.processed_orders = (
                                                st.session_state.staging_processor.orders_in_processing.copy()
                                            )

                                            # Update shortages and inventory data
                                            if "shortage_summary" in recalc_result:
                                                st.session_state.shortage_summary = recalc_result[
                                                    "shortage_summary"
                                                ]
                                                shortage_count = (
                                                    len(recalc_result["shortage_summary"])
                                                    if not recalc_result["shortage_summary"].empty
                                                    else 0
                                                )
                                                st.info(
                                                    f"📊 Recalculation complete: {shortage_count} shortages found"
                                                )
                                            if "grouped_shortage_summary" in recalc_result:
                                                st.session_state.grouped_shortage_summary = (
                                                    recalc_result["grouped_shortage_summary"]
                                                )
                                            if "inventory_comparison" in recalc_result:
                                                st.session_state.inventory_comparison = (
                                                    recalc_result["inventory_comparison"]
                                                )
                                            if "initial_inventory" in recalc_result:
                                                st.session_state.initial_inventory = recalc_result[
                                                    "initial_inventory"
                                                ]

                                            # Force complete UI refresh
                                            if "key_prefix" not in st.session_state:
                                                st.session_state.key_prefix = 0
                                            st.session_state.key_prefix += 1

                                            # Get after state
                                            after_orders = len(
                                                st.session_state.staging_processor.orders_in_processing
                                            )
                                            after_staged = len(
                                                st.session_state.staging_processor.staged_orders
                                            )

                                            # Update order counts in session state
                                            if not st.session_state.processed_orders.empty:
                                                display_orders = st.session_state.processed_orders[
                                                    st.session_state.processed_orders["staged"]
                                                    == False
                                                ].copy()
                                                if (
                                                    "Issues" in display_orders.columns
                                                    and "ordernumber" in display_orders.columns
                                                ):
                                                    orders_without_issues = display_orders[
                                                        display_orders["Issues"] == ""
                                                    ]["ordernumber"].unique()
                                                    orders_with_issues = display_orders[
                                                        display_orders["Issues"] != ""
                                                    ]["ordernumber"].unique()
                                                    total_orders = len(
                                                        display_orders["ordernumber"].unique()
                                                    )
                                                    st.session_state.orders_without_issues = len(
                                                        orders_without_issues
                                                    )
                                                    st.session_state.orders_with_issues = len(
                                                        orders_with_issues
                                                    )
                                                    st.session_state.total_orders = total_orders

                                            # Show results
                                            st.success("✅ Smart recalculation completed!")
                                            st.info(
                                                f"📊 Orders in processing: {before_orders} → {after_orders} | Staged orders: {before_staged} → {after_staged}"
                                            )
                                            st.caption(
                                                "Recalculation used Available for Recalculation (Initial - Staged) inventory base"
                                            )
                                            st.rerun()
                                except Exception as e:
                                    st.error(f"Error recalculating: {e}")
                            else:
                                st.error("Staging processor not available")

            if st.button("Move Selected to Staging"):
                # Initialize staging processor if needed
                if st.session_state.staging_processor is None and st.session_state.get(
                    "workflow_initialized", False
                ):
                    from utils.data_processor import DataProcessor

                    st.session_state.staging_processor = DataProcessor()
                    st.session_state.staging_processor.initialize_workflow(
                        st.session_state.orders_df, st.session_state.inventory_df
                    )

                # Use staging processor if available
                if st.session_state.staging_processor is not None and st.session_state.get(
                    "workflow_initialized", False
                ):
                    try:
                        # Get the indices of selected rows in orders_in_processing
                        order_indices = []

                        # Find indices by matching selected rows to orders_in_processing
                        for selected_row in selected_rows:
                            for (
                                idx,
                                order_row,
                            ) in st.session_state.staging_processor.orders_in_processing.iterrows():
                                if selected_row.get("ordernumber") == order_row.get(
                                    "ordernumber"
                                ) and selected_row.get("sku") == order_row.get("sku"):
                                    order_indices.append(idx)
                                    break

                        if order_indices:
                            # Use staging processor to stage orders
                            staging_result = (
                                st.session_state.staging_processor.stage_selected_orders(
                                    order_indices
                                )
                            )

                            # Step 1: Get the staged items to mark in processed_orders
                            newly_staged_items = (
                                st.session_state.staging_processor.staged_orders.copy()
                            )

                            # Step 2: Update processed_orders with staged flag
                            # Instead of replacing processed_orders, we'll update the 'staged' flag for moved items
                            for _, row in newly_staged_items.iterrows():
                                if "ordernumber" in row and "sku" in row:
                                    # Find matching rows in processed_orders
                                    mask = (
                                        st.session_state.processed_orders["ordernumber"]
                                        == row["ordernumber"]
                                    ) & (st.session_state.processed_orders["sku"] == row["sku"])
                                    # Mark as staged
                                    st.session_state.processed_orders.loc[mask, "staged"] = True

                            # Step 3: Update staged_orders as a filtered view of processed_orders
                            st.session_state.staged_orders = st.session_state.processed_orders[
                                st.session_state.processed_orders["staged"] == True
                            ].copy()

                            # Increment key_prefix to force UI refresh
                            if "key_prefix" not in st.session_state:
                                st.session_state.key_prefix = 0
                            st.session_state.key_prefix += 1

                            # Display success message
                            st.success(
                                f"✅ Staging completed: {staging_result['staged_count']} items staged | "
                                f"{staging_result['remaining_in_processing']} remaining in processing"
                            )
                        else:
                            st.error("Could not find selected orders in processing queue")

                    except Exception as e:
                        st.error(f"Error using staging processor: {e}")
                        # Fallback to legacy method
                        _legacy_staging_method(
                            selected_rows,
                            processed_orders,
                            selected_line_items,
                            selected_unique_orders,
                            selected_total_qty,
                        )
                else:
                    # Legacy staging method for backward compatibility
                    _legacy_staging_method(
                        selected_rows,
                        processed_orders,
                        selected_line_items,
                        selected_unique_orders,
                        selected_total_qty,
                    )

                # Rerun to update the UI
                st.rerun()

    def _legacy_staging_method(
        self,
        selected_rows,
        processed_orders,
        selected_line_items,
        selected_unique_orders,
        selected_total_qty,
    ):
        """Legacy staging method for backward compatibility"""
        if "staged_orders" not in st.session_state:
            st.session_state.staged_orders = pd.DataFrame()

        # Ensure the 'staged' column exists in processed_orders
        if "staged" not in processed_orders.columns:
            processed_orders["staged"] = False

        # For each selected row, find and mark corresponding rows in processed_orders as staged
        for row in selected_rows:
            # Create match conditions based on available unique identifiers
            if "ordernumber" in row and "sku" in row:
                # Match by order number and SKU
                mask = (processed_orders["ordernumber"] == row["ordernumber"]) & (
                    processed_orders["sku"] == row["sku"]
                )
                # Mark matching rows as staged
                processed_orders.loc[mask, "staged"] = True

        # Update session state with both staged orders and processed orders
        st.session_state.staged_orders = processed_orders[processed_orders["staged"] == True].copy()
        st.session_state.processed_orders = processed_orders

        # Display detailed success message
        st.success(
            f"✅ Moved to staging: {selected_line_items} line items | {selected_unique_orders} unique orders | {selected_total_qty:.0f} total units"
        )

        # Show staging count
        st.metric("📋 Orders in Staging", len(st.session_state.get("staged_orders", pd.DataFrame())))

    # Add recalculation section at the bottom of Orders tab
    st.markdown("---")
    st.markdown("**🔄 Smart Recalculation Workflow**")

    # Check if we have the staging processor available and workflow is initialized
    if (
        st.session_state.get("workflow_initialized", False)
        and "processed_orders" in st.session_state
        and st.session_state.processed_orders is not None
    ):
        # Initialize staging processor if needed
        if st.session_state.staging_processor is None:
            from utils.data_processor import DataProcessor

            st.session_state.staging_processor = DataProcessor()
            # Re-initialize with current data
            result = st.session_state.staging_processor.initialize_workflow(
                st.session_state.orders_df, st.session_state.inventory_df
            )
            # Sync with existing processed_orders if available
            if "orders" in result:
                if st.session_state.processed_orders is None:
                    st.session_state.processed_orders = result["orders"]
                    if "staged" not in st.session_state.processed_orders.columns:
                        st.session_state.processed_orders["staged"] = False
                else:
                    # If we have existing processed_orders, sync the staging processor with them
                    # This ensures the staging processor knows about already staged items
                    st.session_state.staging_processor.orders_in_processing = (
                        st.session_state.processed_orders[
                            st.session_state.processed_orders["staged"] == False
                        ].copy()
                    )
                    st.session_state.staging_processor.staged_orders = (
                        st.session_state.processed_orders[
                            st.session_state.processed_orders["staged"] == True
                        ].copy()
                    )

        # Get current stats
        staging_processor = st.session_state.staging_processor
        inventory_calcs = staging_processor.get_inventory_calculations()
        summary = inventory_calcs["staging_summary"]

        # Get available inventory info
        inventory_minus_staged = inventory_calcs.get("inventory_minus_staged", pd.DataFrame())

        # Handle DataFrame case
        if isinstance(inventory_minus_staged, pd.DataFrame) and not inventory_minus_staged.empty:
            available_items = sum(
                1
                for balance in inventory_minus_staged["Balance"]
                if isinstance(balance, (int, float)) and balance > 0
            )
            total_available_balance = sum(
                max(0, balance)
                for balance in inventory_minus_staged["Balance"]
                if isinstance(balance, (int, float))
            )
        else:
            # Fallback for other cases
            available_items = 0
            logger.warning("Could not calculate inventory minus staged: Invalid data format")

        # Create workflow guidance based on current state
        recalc_col1, recalc_col2 = st.columns([2, 1])
        # staged_orders it's lines from orders_in_processing that are staged
        # orders_in_processing it's lines from orders_in_processing that are not staged
        # summary['staged_orders'] it's the number of lines from orders_in_processing that are staged
        # summary['orders_in_processing'] it's the number of lines from orders_in_processing that are not staged
        # available_items it's the number of lines from orders_in_processing that are not staged
        # total_available_balance it's the total balance of the inventory items that are not staged

        with recalc_col1:
            if summary["staged_orders"] > 0 and summary["orders_in_processing"] > 0:
                st.success("🔄 **Ready for Smart Recalculation!**")
                st.info(
                    f"""
                **Current Status:**
                - ✅ {summary['staged_orders']} order lines are staged (inventory protected)
                - 📋 {summary['orders_in_processing']} order lines ready for recalculation
                - 📦 {available_items} unique SKUs available for recalculation)

                **Recommended Workflow:**
                1. 📝 **Edit Bundle Rules** (optional):
                   - [📊 Edit Oxnard SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=549145618) (Sheet: INPUT_bundles_cvr_oxnard)
                   - [📊 Edit Wheeling SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=0) (Sheet: INPUT_bundles_cvr_wheeling)
                2. 🔄 **Recalculate**: Go to 📦 Inventory → 🔄 Available for Recalculation tab
                3. ⚡ **Apply**: Click 'Recalculate All Processing Orders' to use updated inventory
                """
                )
            elif summary["orders_in_processing"] > 0 and summary["staged_orders"] == 0:
                st.warning("🏷️ **Next Step: Stage Some Order Lines**")
                st.info(
                    f"""
                You have {summary['orders_in_processing']} order lines in processing but none staged yet.

                **To start the smart recalculation workflow:**
                1. ✅ Select order lines above and click 'Move Selected to Staging'
                2. 📝 Edit bundle rules (optional):
                   - [📊 Edit Oxnard SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=549145618) (Sheet: INPUT_bundles_cvr_oxnard)
                   - [📊 Edit Wheeling SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=0) (Sheet: INPUT_bundles_cvr_wheeling)
                3. 🔄 Recalculate remaining order lines using protected inventory
                """
                )
            elif summary["staged_orders"] > 0 and summary["orders_in_processing"] == 0:
                st.info("✅ **All Order Lines Staged**")
                st.caption(
                    f"All {summary['staged_orders']} order lines are currently staged. Move some back to processing if you want to recalculate them."
                )
            else:
                st.info("📊 **No Order Lines Available**")
                st.caption("Upload and process orders to begin the smart recalculation workflow.")

        with recalc_col2:
            # Quick action buttons
            st.markdown("**🚀 Quick Actions:**")

            # Quick recalculate button (simplified version)
            if st.button("⚡ Quick Recalculate Now", key="quick_recalc"):
                try:
                    with st.spinner("Quick recalculating orders with latest SKU mappings..."):
                        recalc_result = (
                            st.session_state.staging_processor.recalculate_orders_with_updated_inventory()
                        )

                        if "error" in recalc_result:
                            st.error(f"Recalculation failed: {recalc_result['error']}")
                        else:
                            # Update ALL session state data from recalculation results
                            st.session_state.processed_orders = (
                                st.session_state.staging_processor.orders_in_processing.copy()
                            )

                            # Update shortages and inventory data
                            if "shortage_summary" in recalc_result:
                                st.session_state.shortage_summary = recalc_result[
                                    "shortage_summary"
                                ]
                            if "grouped_shortage_summary" in recalc_result:
                                st.session_state.grouped_shortage_summary = recalc_result[
                                    "grouped_shortage_summary"
                                ]
                            if "inventory_comparison" in recalc_result:
                                st.session_state.inventory_comparison = recalc_result[
                                    "inventory_comparison"
                                ]
                            if "initial_inventory" in recalc_result:
                                st.session_state.initial_inventory = recalc_result[
                                    "initial_inventory"
                                ]

                            # Add the staged flag back and combine with staged orders
                            if "staged" not in st.session_state.processed_orders.columns:
                                st.session_state.processed_orders["staged"] = False

                            # Add staged orders if they exist, avoiding duplicates
                            if (
                                hasattr(st.session_state.staging_processor, "staged_orders")
                                and not st.session_state.staging_processor.staged_orders.empty
                            ):
                                staged_df = st.session_state.staging_processor.staged_orders.copy()
                                staged_df["staged"] = True

                                # Combine and remove duplicates properly
                                combined = pd.concat(
                                    [st.session_state.processed_orders, staged_df],
                                    ignore_index=True,
                                )
                                # Remove duplicates by ordernumber and sku, keeping staged version if present
                                combined = combined.sort_values("staged", ascending=False)
                                combined = combined.drop_duplicates(
                                    subset=["ordernumber", "sku"], keep="first"
                                )
                                st.session_state.processed_orders = combined.reset_index(drop=True)

                            st.success("✅ Quick recalculation completed!")
                            st.info("📊 Updated: Orders, Shortages, and Projected Inventory")
                            st.rerun()
                except Exception as e:
                    st.error(f"Error during quick recalculation: {e}")

    else:
        st.warning("⚠️ Recalculation not available. Please upload and process files first.")


def render_inventory_tab(
    shortage_summary,
    grouped_shortage_summary,
    initial_inventory=None,
    inventory_comparison=None,
    processed_orders=None,
):
    """Renders the Inventory tab with all inventory-related information"""

    import pandas as pd

    st.header("📦 Inventory & Shortages")

    # Display only useful shortage summary information
    if shortage_summary is not None and not shortage_summary.empty:
        shortage_instances = shortage_summary.shape[0]

        # Find SKU column using flexible matching
        sku_col = next(
            (
                col
                for col in shortage_summary.columns
                if any(keyword in col.lower() for keyword in ["sku", "item", "product"])
            ),
            None,
        )
        unique_skus = shortage_summary[sku_col].nunique() if sku_col else 0

        # Find order column using flexible matching
        order_col = next(
            (
                col
                for col in shortage_summary.columns
                if any(keyword in col.lower() for keyword in ["order", "ordernumber", "orderid"])
            ),
            None,
        )
        affected_orders = shortage_summary[order_col].nunique() if order_col else 0

        st.markdown("**📈 Shortages Summary**")
        cols = st.columns(3)

        with cols[0]:
            st.metric("💼 Line Items", shortage_instances)

        with cols[1]:
            st.metric("💹 Unique SKUs", unique_skus)

        with cols[2]:
            st.metric("💱 Affected Orders", affected_orders)

    # Create updated tabs structure for inventory views - ADD AVAILABLE INVENTORY TAB
    (
        initial_inventory_tab,
        available_inventory_tab,
        inventory_after_processing_tab,
        shortages_tab,
    ) = st.tabs(
        [
            "📋 Initial Inventory",
            "🔄 Available for Recalculation (Initial - Staged)",
            "📊 Projected Inventory (After Full Processing)",
            "⚠️ Shortages",
        ]
    )

    with initial_inventory_tab:
        st.markdown("**Uploaded Initial Inventory State**")

        # Use the most up-to-date initial_inventory (prefer session state over parameter)
        if (
            "initial_inventory" in st.session_state
            and st.session_state.initial_inventory is not None
        ):
            initial_inventory = st.session_state.initial_inventory

        if initial_inventory is not None and not initial_inventory.empty:
            st.dataframe(initial_inventory, height=600, use_container_width=True, hide_index=True)
        else:
            st.info("No initial inventory data available")

    # NEW TAB: Available for Recalculation (Initial - Staged)
    with available_inventory_tab:
        st.markdown(
            "**🔄 Available Inventory for Recalculation (Initial Inventory minus Staged Orders)**"
        )

        # Check if we have staging processor available
        if (
            "staging_processor" in st.session_state
            and st.session_state.staging_processor
            and st.session_state.get("workflow_initialized", False)
        ):
            try:
                # Get inventory calculations from staging processor
                inventory_calcs = st.session_state.staging_processor.get_inventory_calculations()
                inventory_minus_staged = inventory_calcs.get(
                    "inventory_minus_staged", pd.DataFrame()
                )
                staging_summary = inventory_calcs.get("staging_summary", {})

                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if isinstance(inventory_minus_staged, pd.DataFrame):
                        st.metric("📦 Available Items", len(inventory_minus_staged))
                    elif isinstance(inventory_minus_staged, dict):
                        st.metric("📦 Available Items", len(inventory_minus_staged))
                    else:
                        st.metric("📦 Available Items", 0)
                with col2:
                    st.metric("🏷️ Orders Staged", staging_summary.get("staged_orders", 0))
                with col3:
                    st.metric(
                        "📋 Orders in Processing", staging_summary.get("orders_in_processing", 0)
                    )
                with col4:
                    if (
                        isinstance(inventory_minus_staged, pd.DataFrame)
                        and "Balance" in inventory_minus_staged.columns
                    ):
                        positive_inventory = sum(
                            1
                            for balance in inventory_minus_staged["Balance"]
                            if isinstance(balance, (int, float)) and balance > 0
                        )
                    elif isinstance(inventory_minus_staged, dict):
                        positive_inventory = sum(
                            1
                            for balance in inventory_minus_staged.values()
                            if isinstance(balance, (int, float)) and balance > 0
                        )
                    else:
                        positive_inventory = 0
                    st.metric("✅ Items with Stock", positive_inventory)

                if isinstance(inventory_minus_staged, (pd.DataFrame, dict)) and (
                    (
                        isinstance(inventory_minus_staged, pd.DataFrame)
                        and not inventory_minus_staged.empty
                    )
                    or (
                        isinstance(inventory_minus_staged, dict) and len(inventory_minus_staged) > 0
                    )
                ):
                    # Convert to DataFrame for display
                    available_inventory_data = []
                    # Build a lookup for initial inventory
                    initial_inventory_lookup = {}
                    if (
                        initial_inventory is not None
                        and isinstance(initial_inventory, pd.DataFrame)
                        and not initial_inventory.empty
                    ):
                        for _, row in initial_inventory.iterrows():
                            sku = row.get("sku") or row.get("Sku")
                            warehouse = (
                                row.get("warehouse")
                                or row.get("WarehouseName")
                                or row.get("Warehouse")
                            )
                            balance = (
                                row.get("balance")
                                if "balance" in row
                                else row.get("Balance")
                                if "Balance" in row
                                else row.get("AvailableQty")
                            )
                            if sku is not None and warehouse is not None and balance is not None:
                                initial_inventory_lookup[f"{sku}|{warehouse}"] = float(balance)

                    # Handle both DataFrame and dict cases
                    if isinstance(inventory_minus_staged, pd.DataFrame):
                        for _, row in inventory_minus_staged.iterrows():
                            sku = row.get("Sku")
                            warehouse = row.get("WarehouseName")
                            balance = row.get("Balance")
                            if sku is not None and warehouse is not None and balance is not None:
                                key = f"{sku}|{warehouse}"
                                initial = initial_inventory_lookup.get(key, None)
                                used_in_staging = initial - balance if initial is not None else None
                                available_inventory_data.append(
                                    {
                                        "SKU": sku,
                                        "Warehouse": warehouse,
                                        "Initial": initial,
                                        "Used in Staging": used_in_staging,
                                        "Available Balance": balance,
                                        "Status": "✅ Available"
                                        if balance > 0
                                        else "❌ Out of Stock"
                                        if balance == 0
                                        else "⚠️ Oversold",
                                    }
                                )
                    else:  # dict case
                        for key, balance in inventory_minus_staged.items():
                            if "|" in key:
                                sku, warehouse = key.split("|", 1)
                                initial = initial_inventory_lookup.get(key, None)
                                used_in_staging = initial - balance if initial is not None else None
                                available_inventory_data.append(
                                    {
                                        "SKU": sku,
                                        "Warehouse": warehouse,
                                        "Initial": initial,
                                        "Used in Staging": used_in_staging,
                                        "Available Balance": balance,
                                        "Status": "✅ Available"
                                        if balance > 0
                                        else "❌ Out of Stock"
                                        if balance == 0
                                        else "⚠️ Oversold",
                                    }
                                )

                if available_inventory_data:
                    available_df = pd.DataFrame(available_inventory_data)
                    # Sort by available balance (highest first, then by SKU)
                    available_df = available_df.sort_values(
                        ["Available Balance", "SKU"], ascending=[False, True]
                    )

                    # Color coding and filtering options
                    st.markdown("**Filter Options:**")
                    filter_col1, filter_col2, filter_col3 = st.columns(3)

                    with filter_col1:
                        show_available = st.checkbox(
                            "✅ Show Available (>0)", value=True, key="show_available"
                        )
                    with filter_col2:
                        show_out_of_stock = st.checkbox(
                            "❌ Show Out of Stock (=0)", value=True, key="show_out_of_stock"
                        )
                    with filter_col3:
                        show_oversold = st.checkbox(
                            "⚠️ Show Oversold (<0)", value=True, key="show_oversold"
                        )

                    # Apply filters
                    filtered_df = available_df.copy()
                    if not show_available:
                        filtered_df = filtered_df[filtered_df["Available Balance"] <= 0]
                    if not show_out_of_stock:
                        filtered_df = filtered_df[filtered_df["Available Balance"] != 0]
                    if not show_oversold:
                        filtered_df = filtered_df[filtered_df["Available Balance"] >= 0]

                    st.markdown(
                        f"**Showing {len(filtered_df)} of {len(available_df)} inventory items**"
                    )

                    # Display the table
                    create_aggrid_table(
                        filtered_df,
                        height=500,
                        key="available_inventory_grid",
                        theme="alpine",
                        selection_mode="multiple",
                        show_hints=False,
                        enable_enterprise_modules=True,
                        enable_sidebar=True,
                        enable_pivot=True,
                        enable_value_aggregation=True,
                        groupable=True,
                        filterable=True,
                    )

                    # Add action buttons
                    st.markdown("---")
                    st.markdown("**🔄 Recalculation Actions**")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button(
                            "🔄 Recalculate All Processing Orders", key="recalc_all_processing"
                        ):
                            st.info(
                                "This will recalculate all orders currently in processing using the Available for Recalculation inventory above."
                            )

                            # Perform recalculation
                            try:
                                with st.spinner(
                                    "Recalculating orders with Available for Recalculation inventory..."
                                ):
                                    # First reload SKU mappings
                                    if (
                                        hasattr(st.session_state, "staging_processor")
                                        and st.session_state.staging_processor
                                    ):
                                        st.session_state.staging_processor.load_sku_mappings()
                                        st.session_state.sku_mappings = (
                                            st.session_state.staging_processor.sku_mappings
                                        )

                                    # Then perform recalculation
                                    recalc_result = (
                                        st.session_state.staging_processor.recalculate_orders_with_updated_inventory()
                                    )

                                    if "error" in recalc_result:
                                        st.error(f"Recalculation failed: {recalc_result['error']}")
                                    else:
                                        # Clear existing shortage data before updating to prevent accumulation
                                        st.session_state.shortage_summary = pd.DataFrame()
                                        st.session_state.grouped_shortage_summary = pd.DataFrame()

                                        # Update ALL session state data from recalculation results
                                        st.session_state.processed_orders = (
                                            st.session_state.staging_processor.orders_in_processing.copy()
                                        )

                                        # Update shortages and inventory data
                                        if "shortage_summary" in recalc_result:
                                            st.session_state.shortage_summary = recalc_result[
                                                "shortage_summary"
                                            ]
                                        if "grouped_shortage_summary" in recalc_result:
                                            st.session_state.grouped_shortage_summary = (
                                                recalc_result["grouped_shortage_summary"]
                                            )
                                        if "inventory_comparison" in recalc_result:
                                            st.session_state.inventory_comparison = recalc_result[
                                                "inventory_comparison"
                                            ]
                                        if "initial_inventory" in recalc_result:
                                            st.session_state.initial_inventory = recalc_result[
                                                "initial_inventory"
                                            ]

                                        # Add the staged flag back
                                        if (
                                            "staged"
                                            not in st.session_state.processed_orders.columns
                                        ):
                                            st.session_state.processed_orders["staged"] = False

                                        # Update staged orders view with proper deduplication
                                        if (
                                            hasattr(
                                                st.session_state.staging_processor, "staged_orders"
                                            )
                                            and not st.session_state.staging_processor.staged_orders.empty
                                        ):
                                            staged_df = (
                                                st.session_state.staging_processor.staged_orders.copy()
                                            )
                                            staged_df["staged"] = True
                                            # Combine and remove duplicates properly
                                            combined = pd.concat(
                                                [st.session_state.processed_orders, staged_df],
                                                ignore_index=True,
                                            )
                                            combined = combined.sort_values(
                                                "staged", ascending=False
                                            )
                                            combined = combined.drop_duplicates(
                                                subset=["ordernumber", "sku"], keep="first"
                                            )
                                            st.session_state.processed_orders = (
                                                combined.reset_index(drop=True)
                                            )

                                        st.success("✅ Recalculation completed successfully!")
                                        st.info(
                                            "📊 Updated: Orders, Shortages, and Projected Inventory. Navigate to other tabs to see updated results."
                                        )
                                        st.rerun()

                            except Exception as e:
                                st.error(f"Error during recalculation: {e}")

                    with col2:
                        if st.button("📝 Edit SKU Mappings First", key="edit_mappings_hint"):
                            st.info(
                                """
                            **Edit Bundle Components:**
                            - [📊 Edit Oxnard SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=549145618) (Sheet: INPUT_bundles_cvr_oxnard)
                            - [📊 Edit Wheeling SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=0) (Sheet: INPUT_bundles_cvr_wheeling)

                            After editing, return here to recalculate.
                            """
                            )

                        with col3:
                            if st.button("🔄 Refresh Data", key="refresh_inventory"):
                                st.rerun()

                        # DEBUG SECTION
                        st.markdown("---")
                        with st.expander("🔧 Debug: Inventory-Staging Data", expanded=False):
                            st.markdown("**🔍 Debugging Information for Inventory-Staging Process**")

                            try:
                                # Get debug information from staging processor
                                debug_info = (
                                    st.session_state.staging_processor.get_debug_inventory_state()
                                )

                                # Display workflow state
                                st.markdown("**📊 Current Workflow State:**")
                                workflow_state = debug_info.get("workflow_state", {})
                                debug_col1, debug_col2, debug_col3 = st.columns(3)

                                with debug_col1:
                                    st.metric(
                                        "📋 Orders in Processing",
                                        workflow_state.get("orders_in_processing", 0),
                                    )
                                with debug_col2:
                                    st.metric(
                                        "🏷️ Staged Orders", workflow_state.get("staged_orders", 0)
                                    )
                                with debug_col3:
                                    st.metric(
                                        "📦 Initial Inventory Rows",
                                        workflow_state.get("initial_inventory_rows", 0),
                                    )

                                # Display initial inventory debug info
                                if "initial_inventory" in debug_info:
                                    st.markdown("**📦 Initial Inventory Debug:**")
                                    initial_inv = debug_info["initial_inventory"]
                                    if "error" not in initial_inv:
                                        st.write(
                                            f"- Total items: {initial_inv.get('total_items', 0)}"
                                        )
                                        st.write(
                                            f"- Columns: {', '.join(initial_inv.get('columns', []))}"
                                        )
                                        st.write(
                                            f"- Warehouses: {', '.join(initial_inv.get('warehouses', []))}"
                                        )

                                        if initial_inv.get("sample_data"):
                                            st.markdown("**Sample Initial Inventory Data:**")
                                            st.json(
                                                initial_inv["sample_data"][:3]
                                            )  # Show first 3 items
                                    else:
                                        st.warning(
                                            f"Initial inventory error: {initial_inv.get('error')}"
                                        )

                                # Display staged orders debug info
                                if "staged_orders" in debug_info:
                                    st.markdown("**🏷️ Staged Orders Debug:**")
                                    staged_info = debug_info["staged_orders"]
                                    st.write(
                                        f"- Total staged orders: {staged_info.get('total_orders', 0)}"
                                    )
                                    if staged_info.get("total_orders", 0) > 0:
                                        st.write(
                                            f"- Fulfillment centers: {', '.join(staged_info.get('fulfillment_centers', []))}"
                                        )

                                        if staged_info.get("sample_data"):
                                            st.markdown("**Sample Staged Orders Data:**")
                                            st.json(
                                                staged_info["sample_data"][:2]
                                            )  # Show first 2 orders

                                # Display inventory minus staged debug info
                                if "inventory_minus_staged" in debug_info:
                                    st.markdown("**🔄 Inventory Minus Staged Debug:**")
                                    inv_minus = debug_info["inventory_minus_staged"]

                                    (
                                        debug_metrics_col1,
                                        debug_metrics_col2,
                                        debug_metrics_col3,
                                        debug_metrics_col4,
                                    ) = st.columns(4)
                                    with debug_metrics_col1:
                                        st.metric("📦 Total Items", inv_minus.get("total_items", 0))
                                    with debug_metrics_col2:
                                        st.metric(
                                            "✅ Positive Balance",
                                            inv_minus.get("positive_balance_items", 0),
                                        )
                                    with debug_metrics_col3:
                                        st.metric(
                                            "⚖️ Zero Balance",
                                            inv_minus.get("zero_balance_items", 0),
                                        )
                                    with debug_metrics_col4:
                                        st.metric(
                                            "❌ Negative Balance",
                                            inv_minus.get("negative_balance_items", 0),
                                        )

                                    if inv_minus.get("sample_items"):
                                        st.markdown("**Sample Inventory Minus Staged Items:**")
                                        st.json(inv_minus["sample_items"])

                                # Display current inventory state
                                if "current_inventory_state" in debug_info:
                                    st.markdown("**📊 Current Inventory State Debug:**")
                                    current_inv = debug_info["current_inventory_state"]
                                    if "message" not in current_inv:
                                        st.write(
                                            f"- Total items: {current_inv.get('total_items', 0)}"
                                        )
                                        if current_inv.get("sample_items"):
                                            st.markdown("**Sample Current Inventory Items:**")
                                            st.json(current_inv["sample_items"])
                                    else:
                                        st.info(current_inv.get("message"))

                                # Display raw calculation data
                                st.markdown("**🔧 Raw Calculation Access:**")
                                if st.button(
                                    "📄 Show Raw Inventory Calculations", key="show_raw_calcs"
                                ):
                                    try:
                                        raw_calcs = (
                                            st.session_state.staging_processor.get_inventory_calculations()
                                        )
                                        st.json(raw_calcs)
                                    except Exception as e:
                                        st.error(f"Error getting raw calculations: {e}")

                                # Show errors if any
                                if "error" in debug_info:
                                    st.error(f"Debug error: {debug_info['error']}")

                                # Timestamp
                                st.caption(
                                    f"Debug info generated at: {debug_info.get('timestamp', 'unknown')}"
                                )

                            except Exception as e:
                                st.error(f"Error generating debug information: {e}")
                                st.write(
                                    "This usually means the staging processor is not properly initialized."
                                )
                else:
                    st.info(
                        "No available inventory data. This happens when there are no staged orders or no initial inventory."
                    )

            except Exception as e:
                st.error(f"Error calculating available inventory: {e}")
                st.info(
                    "Please ensure orders have been processed and staged before viewing available inventory."
                )
        else:
            st.warning("⚠️ Staging processor not available.")
            st.info(
                "Please upload and process files first, then stage some orders to see available inventory for recalculation."
            )

    with inventory_after_processing_tab:
        st.markdown("**Projected Inventory After Full Current Processing Run**")

        # Use the most up-to-date inventory_comparison (prefer session state over parameter)
        if (
            "inventory_comparison" in st.session_state
            and st.session_state.inventory_comparison is not None
        ):
            # Always use session state if it exists (even if empty - means updated after recalculation)
            inventory_comparison = st.session_state.inventory_comparison
            st.info(f"📊 Using updated inventory comparison with {len(inventory_comparison)} items")

        # Use inventory_comparison if available (shows inventory minus orders)
        if inventory_comparison is not None and not inventory_comparison.empty:
            # Highlight rows where inventory levels are low or negative
            highlight_rows = []

            # Find columns for highlighting
            qty_after_col = next(
                (
                    col
                    for col in inventory_comparison.columns
                    if any(keyword in col.lower() for keyword in ["after", "remaining", "final"])
                ),
                None,
            )

            if qty_after_col:
                # Mark rows with low/negative inventory for highlighting
                for i, row in inventory_comparison.iterrows():
                    if pd.notna(row[qty_after_col]) and row[qty_after_col] <= 0:
                        highlight_rows.append(i)

                # Style the dataframe with conditional formatting
                styled_df = inventory_comparison.style.apply(
                    lambda x: [
                        "background-color: #ffcccc" if i in highlight_rows else ""
                        for i in range(len(x))
                    ],
                    axis=0,
                )

                st.dataframe(
                    inventory_comparison, height=600, use_container_width=True, hide_index=True
                )
            else:
                # Just display without highlighting if we can't find the right column
                st.dataframe(
                    inventory_comparison, height=600, use_container_width=True, hide_index=True
                )
        else:
            st.info("No inventory data available")

    with shortages_tab:
        # Use the most up-to-date grouped_shortage_summary (prefer session state over parameter)
        if (
            "grouped_shortage_summary" in st.session_state
            and st.session_state.grouped_shortage_summary is not None
        ):
            # Always use session state if it exists (even if empty - means updated after recalculation)
            grouped_shortage_summary = st.session_state.grouped_shortage_summary

        if grouped_shortage_summary is not None and not grouped_shortage_summary.empty:
            # Show grouped shortages first using standard dataframe for cleaner display
            st.markdown("**Shortages by SKU**")
            st.dataframe(
                grouped_shortage_summary,
                height=550,  # Increased height for better visibility
                use_container_width=True,
                hide_index=True,
            )

            # Add expander for individual shortage details
            with st.expander("View All Individual Shortage Line Items", expanded=False):
                # Use session state shortage_summary if available
                if (
                    "shortage_summary" in st.session_state
                    and st.session_state.shortage_summary is not None
                ):
                    shortage_summary = st.session_state.shortage_summary

                if shortage_summary is not None and not shortage_summary.empty:
                    create_aggrid_table(
                        shortage_summary,
                        height=400,
                        selection_mode="multiple",
                        key="shortage_table",
                    )
                else:
                    st.info("No individual shortage details available")
        elif shortage_summary is not None and not shortage_summary.empty:
            # If no grouped data, show individual shortages directly
            # Use session state shortage_summary if available
            if (
                "shortage_summary" in st.session_state
                and st.session_state.shortage_summary is not None
            ):
                shortage_summary = st.session_state.shortage_summary

            if shortage_summary is not None and not shortage_summary.empty:
                create_aggrid_table(
                    shortage_summary, height=500, selection_mode="multiple", key="shortage_table"
                )
        else:
            st.info("No shortages detected")


def render_staging_tab():
    """Renders the Staging tab with simplified staging functionality"""
    st.markdown("**Orders Staged for Processing**")
    # Get staged orders directly from processed_orders
    if (
        "processed_orders" not in st.session_state
        or st.session_state.processed_orders is None
        or st.session_state.processed_orders.empty
    ):
        st.info("No orders available for staging")
        return

    # Ensure staged column exists
    if "staged" not in st.session_state.processed_orders.columns:
        st.session_state.processed_orders["staged"] = False

    # Get only staged orders
    staged_orders = st.session_state.processed_orders[
        st.session_state.processed_orders["staged"] == True
    ].copy()

    if staged_orders.empty:
        st.info("No orders currently staged")
        st.write("Move orders from the Orders tab to see them here.")
        return

    # Remove the staged column from display
    display_orders = (
        staged_orders.drop(columns=["staged"])
        if "staged" in staged_orders.columns
        else staged_orders
    )

    # Display key metrics for staged orders
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_orders = (
            len(display_orders["ordernumber"].unique())
            if "ordernumber" in display_orders.columns
            else 0
        )
        st.metric("📋 Staged Orders", f"{total_orders:,}")

    with col2:
        total_items = len(display_orders)
        st.metric("📦 Staged Line Items", f"{total_items:,}")

    with col3:
        if "Transaction Quantity" in display_orders.columns:
            total_quantity = display_orders["Transaction Quantity"].sum()
        else:
            total_quantity = 0
        st.metric("📊 Total Units Staged", f"{total_quantity:,.0f}")

    with col4:
        fulfillment_centers = (
            display_orders["Fulfillment Center"].nunique()
            if "Fulfillment Center" in display_orders.columns
            else 0
        )
        st.metric("🏭 Warehouses Used", fulfillment_centers)

    with col5:
        staged_issues = (
            display_orders[display_orders["Issues"] != ""].shape[0]
            if "Issues" in display_orders.columns
            else 0
        )
        staged_orders_with_issues = (
            display_orders[display_orders["Issues"] != ""]["ordernumber"].nunique()
            if ("Issues" in display_orders.columns and "ordernumber" in display_orders.columns)
            else 0
        )
        st.metric(
            "⚠️ Items with Issues", staged_issues, delta=f"{staged_orders_with_issues} orders"
        )

    # Show summary statistics
    st.info(
        f"📊 Staged Order Summary: {total_items} line items | {total_orders} unique orders | {total_quantity:,.0f} total units | {staged_issues} items with issues | {staged_orders_with_issues} unique orders with issues"
    )

    # Show unique order numbers in an expander for quick reference
    if "ordernumber" in display_orders.columns:
        with st.expander("🔍 View Staged Order Numbers", expanded=False):
            unique_order_numbers = sorted(display_orders["ordernumber"].unique())
            st.write("Unique order numbers in staging:")
            st.code(", ".join(map(str, unique_order_numbers)))

    # Create staging grid
    grid_response = create_aggrid_table(
        display_orders,
        height=600,
        selection_mode="multiple",
        enable_enterprise_modules=True,
        theme="alpine",
        key="staging_grid",
    )

    # Handle selection and removal without rerun
    if (
        grid_response
        and isinstance(grid_response, dict)
        and "grid_response" in grid_response
        and grid_response["grid_response"]
        and "selected_rows" in grid_response["grid_response"]
        and grid_response["grid_response"]["selected_rows"]
    ):
        selected_rows = grid_response["grid_response"]["selected_rows"]
        selected_count = len(selected_rows)

        st.write(f"Selected: {selected_count} items")

        # Display summary of selected items
        if selected_count > 0:
            selected_skus = len(
                set(
                    [
                        row["sku"] if isinstance(row, dict) and "sku" in row else ""
                        for row in selected_rows
                    ]
                )
            )
            selected_orders = len(
                set(
                    [
                        row["ordernumber"] if isinstance(row, dict) and "ordernumber" in row else ""
                        for row in selected_rows
                    ]
                )
            )
            selected_qty = sum(
                [
                    row["Transaction Quantity"]
                    if isinstance(row, dict) and "Transaction Quantity" in row
                    else 0
                    for row in selected_rows
                ]
            )

            st.write(
                f"📦 {selected_qty} units | 🏷️ {selected_skus} unique SKUs | 📝 {selected_orders} orders"
            )


def render_sku_mapping_editor(sku_mappings, data_processor):
    """Renders the SKU Mapping display with warehouse tabs and proper bundle editing"""
    st.markdown("**SKU Mapping**")

    # Add prominent links to Google Sheets for each warehouse
    st.info(
        "📝 **To Edit SKU Mappings:** Use the warehouse-specific Google Sheets links below (changes will be loaded on next app refresh)"
    )

    # Create direct links for each warehouse
    st.markdown("**🔗 Direct Links to Edit SKU Mappings by Warehouse:**")

    # Create columns for warehouse links
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **🏢 Oxnard Warehouse:**

        [📊 Edit Oxnard SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=549145618)

        *Sheet: INPUT_bundles_cvr_oxnard*
        """
        )

    with col2:
        st.markdown(
            """
        **🏭 Wheeling Warehouse:**

        [📊 Edit Wheeling SKU Mappings](https://docs.google.com/spreadsheets/d/19-0HG0voqQkzBfiMwmCC05KE8pO4lQapvrnI_H7nWDY/edit#gid=0)

        *Sheet: INPUT_bundles_cvr_wheeling*
        """
        )

    # Reload button
    link_col1, link_col2 = st.columns([3, 1])

    with link_col2:
        if st.button(
            "🔄 Reload Mappings", help="Reload SKU mappings from Google Sheets after making changes"
        ):
            # Trigger a reload of SKU mappings
            if (
                hasattr(st.session_state, "staging_processor")
                and st.session_state.staging_processor
            ):
                try:
                    st.session_state.staging_processor.sku_mappings = (
                        st.session_state.staging_processor.load_sku_mappings()
                    )
                    st.success("✅ SKU mappings reloaded from Google Sheets!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reloading mappings: {e}")
            else:
                st.warning("No data processor available for reloading")

    st.markdown("---")

    # Debug info
    if sku_mappings is not None:
        if isinstance(sku_mappings, dict):
            warehouses = list(sku_mappings.keys())
            st.info(f"📁 Loaded SKU mappings for warehouses: {', '.join(warehouses)}")
        else:
            st.info(f"📁 Loaded SKU mappings as DataFrame with {len(sku_mappings)} rows")
    else:
        st.warning("⚠️ No SKU mappings loaded. Please check if airtable_sku_mappings.json exists.")
        return

    if not isinstance(sku_mappings, dict):
        st.error("SKU mappings should be a dictionary structure from JSON file.")
        return

    # Create warehouse tabs
    warehouse_names = [w for w in sku_mappings.keys() if w in ["Oxnard", "Wheeling"]]
    if not warehouse_names:
        st.warning("No warehouse data found (looking for 'Oxnard' and 'Wheeling').")
        return

    # Create tabs for each warehouse
    if len(warehouse_names) == 1:
        warehouse_tabs = [st.container()]
        warehouse_names[0]
    else:
        warehouse_tabs = st.tabs([f"🏢 {warehouse}" for warehouse in warehouse_names])

    # Process each warehouse
    for tab_index, warehouse in enumerate(warehouse_names):
        with warehouse_tabs[tab_index] if len(warehouse_names) > 1 else warehouse_tabs[0]:
            st.markdown(f"**{warehouse} Warehouse**")

            warehouse_data = sku_mappings.get(warehouse, {})
            singles_data = warehouse_data.get("singles", {})
            bundles_data = warehouse_data.get("bundles", {})

            # Create sub-tabs for Singles and Bundles within each warehouse
            singles_tab, bundles_tab = st.tabs(["📦 Singles", "🎁 Bundles"])

            # SINGLES TAB
            with singles_tab:
                st.markdown(f"**Single SKU Mappings - {warehouse}**")

                if singles_data:
                    # Convert singles to DataFrame
                    singles_rows = []
                    for order_sku, mapping_data in singles_data.items():
                        singles_rows.append(
                            {
                                "order_sku": order_sku,
                                "picklist_sku": mapping_data.get("picklist_sku", ""),
                                "actualqty": mapping_data.get("actualqty", 1.0),
                                "total_pick_weight": mapping_data.get("total_pick_weight", 0.0),
                                "pick_type": mapping_data.get("pick_type", ""),
                                "notes": mapping_data.get("notes", ""),
                                "airtable_id": mapping_data.get("airtable_id", ""),
                            }
                        )

                    singles_df = pd.DataFrame(singles_rows)

                    # Show summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📦 Total Singles", len(singles_df))
                    with col2:
                        unique_pick_types = singles_df["pick_type"].nunique()
                        st.metric("🏷️ Pick Types", unique_pick_types)
                    with col3:
                        total_weight = singles_df["total_pick_weight"].sum()
                        st.metric("⚖️ Total Weight", f"{total_weight:.1f}")

                    # Format display
                    display_singles = singles_df.copy()
                    display_singles["actualqty"] = display_singles["actualqty"].round(2)
                    display_singles["total_pick_weight"] = display_singles[
                        "total_pick_weight"
                    ].round(3)

                    # Create editable table
                    create_aggrid_table(
                        display_singles,
                        height=500,
                        selection_mode="multiple",
                        enable_enterprise_modules=True,
                        theme="alpine",
                        key=f"singles_{warehouse}",
                        filterable=True,
                        sortable=True,
                        groupable=True,
                        editable=True,
                        show_hints=True,
                        enable_sidebar=True,
                    )
                else:
                    st.info(f"No single SKU mappings found for {warehouse}")

            # BUNDLES TAB
            with bundles_tab:
                st.markdown(f"**Bundle Mappings - {warehouse}**")

                if bundles_data:
                    # Convert bundles to DataFrame showing each component
                    bundle_rows = []
                    for bundle_sku, components in bundles_data.items():
                        if isinstance(components, list):
                            for i, comp in enumerate(components):
                                bundle_rows.append(
                                    {
                                        "bundle_sku": bundle_sku,
                                        "component_index": i + 1,
                                        "component_sku": comp.get("component_sku", ""),
                                        "actualqty": comp.get("actualqty", 1.0),
                                        "weight": comp.get("weight", 0.0),
                                        "pick_type": comp.get("pick_type", ""),
                                        "pick_type_inventory": comp.get("pick_type_inventory", ""),
                                        "total_components": len(components),
                                    }
                                )

                    if bundle_rows:
                        bundles_df = pd.DataFrame(bundle_rows)

                        # Show summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            unique_bundles = bundles_df["bundle_sku"].nunique()
                            st.metric("🎁 Total Bundles", unique_bundles)
                        with col2:
                            total_components = len(bundles_df)
                            st.metric("🧩 Total Components", total_components)
                        with col3:
                            unique_component_skus = bundles_df["component_sku"].nunique()
                            st.metric("📦 Unique Component SKUs", unique_component_skus)
                        with col4:
                            total_weight = bundles_df["weight"].sum()
                            st.metric("⚖️ Total Weight", f"{total_weight:.1f}")

                        # Bundle overview
                        with st.expander("📋 Bundle Overview", expanded=False):
                            bundle_summary = (
                                bundles_df.groupby("bundle_sku")
                                .agg(
                                    {"component_sku": "count", "actualqty": "sum", "weight": "sum"}
                                )
                                .reset_index()
                            )
                            bundle_summary.columns = [
                                "Bundle SKU",
                                "Component Count",
                                "Total Qty",
                                "Total Weight",
                            ]

                            st.dataframe(
                                bundle_summary,
                                height=300,
                                use_container_width=True,
                                hide_index=True,
                            )

                        # Format display
                        display_bundles = bundles_df.copy()
                        display_bundles["actualqty"] = display_bundles["actualqty"].round(2)
                        display_bundles["weight"] = display_bundles["weight"].round(3)

                        # Create editable table for bundle components
                        grid_response = create_aggrid_table(
                            display_bundles,
                            height=500,
                            selection_mode="multiple",
                            enable_enterprise_modules=True,
                            theme="alpine",
                            key=f"bundles_{warehouse}",
                            filterable=True,
                            sortable=True,
                            groupable=True,
                            editable=True,
                            show_hints=True,
                            enable_sidebar=True,
                        )

                        # Bundle editing controls
                        st.markdown("---")
                        st.markdown("**🔧 Bundle Component Editing**")

                        # Show selected components
                        if grid_response["grid_response"]["selected_rows"]:
                            selected = grid_response["grid_response"]["selected_rows"]
                            st.success(f"✅ Selected {len(selected)} component(s)")

                            # Component editing form
                            with st.expander("✏️ Edit Selected Components", expanded=True):
                                if len(selected) == 1:
                                    comp = selected[0]
                                    st.markdown(
                                        f"**Editing component in bundle: {comp.get('bundle_sku')}**"
                                    )

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        edit_component_sku = st.text_input(
                                            "Component SKU",
                                            value=comp.get("component_sku", ""),
                                            key=f"edit_comp_sku_{warehouse}",
                                        )
                                        edit_qty = st.number_input(
                                            "Quantity",
                                            value=float(comp.get("actualqty", 1.0)),
                                            min_value=0.1,
                                            step=0.1,
                                            key=f"edit_qty_{warehouse}",
                                        )
                                        edit_weight = st.number_input(
                                            "Weight",
                                            value=float(comp.get("weight", 0.0)),
                                            min_value=0.0,
                                            step=0.1,
                                            key=f"edit_weight_{warehouse}",
                                        )

                                    with col2:
                                        edit_pick_type = st.text_input(
                                            "Pick Type",
                                            value=comp.get("pick_type", ""),
                                            key=f"edit_pick_type_{warehouse}",
                                        )
                                        edit_pick_type_inv = st.text_input(
                                            "Pick Type Inventory",
                                            value=comp.get("pick_type_inventory", ""),
                                            key=f"edit_pick_inv_{warehouse}",
                                        )

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        if st.button(
                                            "💾 Save Changes", key=f"save_comp_{warehouse}"
                                        ):
                                            st.success(
                                                "✅ Component updated! (Save to JSON will be implemented)"
                                            )
                                    with col2:
                                        if st.button(
                                            "❌ Remove Component", key=f"remove_comp_{warehouse}"
                                        ):
                                            st.warning("⚠️ Component will be removed from bundle")
                                    with col3:
                                        if st.button("🔄 Reset", key=f"reset_comp_{warehouse}"):
                                            st.rerun()
                                else:
                                    st.info(
                                        "Select exactly one component to edit, or use bulk operations below"
                                    )

                                    if st.button(
                                        "❌ Remove All Selected", key=f"remove_selected_{warehouse}"
                                    ):
                                        st.warning(
                                            f"⚠️ Will remove {len(selected)} components from their bundles"
                                        )

                        # Add new component
                        with st.expander("➕ Add Component to Bundle", expanded=False):
                            col1, col2 = st.columns(2)

                            with col1:
                                new_bundle = st.selectbox(
                                    "Select Bundle",
                                    options=list(bundles_data.keys()),
                                    key=f"new_bundle_{warehouse}",
                                )
                                new_comp_sku = st.text_input(
                                    "Component SKU",
                                    placeholder="e.g., apple-10x05",
                                    key=f"new_comp_sku_{warehouse}",
                                )
                                new_qty = st.number_input(
                                    "Quantity",
                                    min_value=0.1,
                                    value=1.0,
                                    step=0.1,
                                    key=f"new_qty_{warehouse}",
                                )

                            with col2:
                                new_weight = st.number_input(
                                    "Weight",
                                    min_value=0.0,
                                    value=0.0,
                                    step=0.1,
                                    key=f"new_weight_{warehouse}",
                                )
                                new_pick_type = st.text_input(
                                    "Pick Type",
                                    placeholder="e.g., Fruit: Apple",
                                    key=f"new_pick_type_{warehouse}",
                                )
                                new_pick_inv = st.text_input(
                                    "Pick Type Inventory", key=f"new_pick_inv_{warehouse}"
                                )

                            if st.button("➕ Add Component", key=f"add_comp_{warehouse}"):
                                if new_bundle and new_comp_sku:
                                    st.success(
                                        f"✅ Ready to add '{new_comp_sku}' to bundle '{new_bundle}'"
                                    )
                                    st.info("💡 Addition will update the JSON file")
                                else:
                                    st.error("❌ Please select bundle and enter component SKU")

                        # Create new bundle
                        with st.expander("🎁 Create New Bundle", expanded=False):
                            new_bundle_sku = st.text_input(
                                "New Bundle SKU",
                                placeholder=f"e.g., m.{warehouse.lower()}_special",
                                key=f"new_bundle_sku_{warehouse}",
                            )
                            new_components_text = st.text_area(
                                "Component SKUs (one per line or comma-separated)",
                                placeholder="apple-10x05\norange-12x06\nor: apple-10x05, orange-12x06",
                                key=f"new_components_{warehouse}",
                            )

                            if st.button("🎁 Create Bundle", key=f"create_bundle_{warehouse}"):
                                if new_bundle_sku and new_components_text:
                                    # Parse components (handle both line-separated and comma-separated)
                                    if "\n" in new_components_text:
                                        components = [
                                            c.strip()
                                            for c in new_components_text.split("\n")
                                            if c.strip()
                                        ]
                                    else:
                                        components = [
                                            c.strip()
                                            for c in new_components_text.split(",")
                                            if c.strip()
                                        ]

                                    st.success(
                                        f"✅ Ready to create bundle '{new_bundle_sku}' with {len(components)} components"
                                    )
                                    st.info("💡 Bundle creation will update the JSON file")
                                else:
                                    st.error("❌ Please enter bundle SKU and components")
                    else:
                        st.info("No valid bundle components found")
                else:
                    st.info(f"No bundle mappings found for {warehouse}")

                    # Create first bundle interface
                    with st.expander("🎁 Create Your First Bundle", expanded=True):
                        st.markdown("**No bundles found. Create your first bundle:**")

                        first_bundle_sku = st.text_input(
                            "Bundle SKU",
                            placeholder=f"e.g., m.{warehouse.lower()}_starter",
                            key=f"first_bundle_{warehouse}",
                        )
                        first_components = st.text_area(
                            "Component SKUs (one per line)",
                            placeholder="apple-10x05\norange-12x06\nmango-09x16",
                            key=f"first_components_{warehouse}",
                        )

                        if st.button("🎁 Create First Bundle", key=f"create_first_{warehouse}"):
                            if first_bundle_sku and first_components:
                                components = [
                                    c.strip() for c in first_components.split("\n") if c.strip()
                                ]
                                st.success(
                                    f"✅ Ready to create first bundle '{first_bundle_sku}' with {len(components)} components"
                                )
                                st.info("💡 Bundle creation will update airtable_sku_mappings.json")
                            else:
                                st.error("❌ Please enter bundle SKU and components")


def create_aggrid_table(
    df,
    height=400,
    selection_mode="multiple",
    enable_enterprise_modules=True,
    fit_columns_on_grid_load=False,
    theme="alpine",
    key=None,
    groupable=True,
    filterable=True,
    sortable=True,
    editable=False,
    show_hints=False,
    enable_download=True,
    enable_sidebar=True,
    enable_pivot=True,
    enable_value_aggregation=True,
    enhanced_menus=True,
    suppress_callback_exceptions=True,
):
    """
    Create an enhanced ag-Grid table with auto-sizing, filtering, grouping, sorting, and download capabilities.

    Args:
        df: DataFrame to display
        height: Table height in pixels
        selection_mode: 'single', 'multiple', or 'disabled'
        enable_enterprise_modules: Enable advanced features (disabled by default to prevent JSON serialization issues)
        fit_columns_on_grid_load: Auto-fit columns to grid width
        theme: ag-Grid theme ('alpine', 'balham', 'material')
        key: Unique key for the table
        groupable: Enable column grouping
        filterable: Enable column filtering
        sortable: Enable column sorting
        editable: Enable cell editing
        show_hints: Show helpful hints to users
        enable_download: Enable download functionality
        enable_sidebar: Enable sidebar for advanced grouping/filtering
        enable_pivot: Enable pivot functionality
        enable_value_aggregation: Enable value aggregation in groups
        enhanced_menus: Enable enhanced context menus and column menu options
        suppress_callback_exceptions: Suppress callback exceptions for better performance

    Returns:
        dict: Contains AgGrid response and additional info
    """
    try:
        # Create GridOptionsBuilder with simplified configuration
        gb = GridOptionsBuilder.from_dataframe(df)

        # Configure default column properties with enhanced filtering and grouping
        gb.configure_default_column(
            filterable=filterable,
            sortable=sortable,
            resizable=True,
            editable=editable,
            groupable=groupable,
            enableRowGroup=groupable,
            enablePivot=enable_pivot,
            enableValue=enable_value_aggregation,
            filter=True,
            floatingFilter=True,  # Enable floating filters for better UX
            suppressMenu=False,
            menuTabs=["filterMenuTab", "generalMenuTab", "columnsMenuTab"] if filterable else [],
            minWidth=120,  # Set minimum column width
            width=150,  # Set default column width
            suppressSizeToFit=False,  # Allow columns to be resized to fit content
        )

        # Configure selection
        if selection_mode != "disabled":
            gb.configure_selection(
                selection_mode,
                use_checkbox=True,
                header_checkbox=True,  # Adds select all checkbox in header
                groupSelectsChildren=True,
                groupSelectsFiltered=True,
            )

        # Configure enhanced column types for better filtering and grouping
        for col in df.columns:
            # Set specific widths for important columns
            col_width = 150  # default
            if any(keyword in col.lower() for keyword in ["sku", "id", "number"]):
                col_width = 180  # wider for SKUs and IDs
            elif any(keyword in col.lower() for keyword in ["name", "description"]):
                col_width = 200  # wider for names
            elif any(keyword in col.lower() for keyword in ["qty", "quantity", "balance"]):
                col_width = 120  # narrower for numbers

            if df[col].dtype in ["int64", "float64"]:
                # Numeric columns with enhanced number filter and aggregation
                gb.configure_column(
                    col,
                    type=["numericColumn"],
                    filter="agNumberColumnFilter",
                    enableValue=True,  # Enable for aggregation
                    aggFunc=["sum", "avg", "min", "max", "count"],
                    floatingFilter=True,
                    width=col_width,
                    minWidth=100,
                )
            elif df[col].dtype == "datetime64[ns]":
                # Date columns with enhanced date filter
                gb.configure_column(
                    col,
                    filter="agDateColumnFilter",
                    type=["dateColumn"],
                    floatingFilter=True,
                    width=col_width,
                    minWidth=120,
                )
            else:
                # Text columns with enhanced text filter and grouping
                gb.configure_column(
                    col,
                    filter="agTextColumnFilter",
                    enableRowGroup=True,  # Enable for grouping
                    floatingFilter=True,
                    width=col_width,
                    minWidth=120,
                )

        # Configure sidebar with working configuration for streamlit-aggrid 1.0.5
        if enable_sidebar:
            try:
                # Use the correct method for streamlit-aggrid 1.0.5
                gb.configure_side_bar()
            except Exception as e:
                print(f"Sidebar configuration failed: {e}")

        # Configure basic grouping (simplified to avoid issues)
        if groupable:
            try:
                gb.configure_grid_options(
                    groupSelectsChildren=True,
                    groupSelectsFiltered=True,
                    suppressRowGroupHidesColumns=False,
                    groupDefaultExpanded=1,
                )
            except Exception as e:
                print(f"Grouping configuration failed: {e}")

        # Configure basic aggregation for numeric columns
        if enable_value_aggregation:
            for col in df.columns:
                if df[col].dtype in ["int64", "float64"]:
                    try:
                        gb.configure_column(col, enableValue=True)
                    except Exception:
                        # Skip if column configuration causes issues
                        pass

        # Build grid options
        grid_options = gb.build()

        # Create the AgGrid with original working configuration
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            height=height,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=fit_columns_on_grid_load,
            theme=theme,
            enable_enterprise_modules=enable_enterprise_modules,
            allow_unsafe_jscode=True,  # Allow JavaScript code to prevent serialization errors
            key=key,
        )

    except Exception as e:
        # Fallback to basic Streamlit dataframe if AgGrid fails
        st.error(f"AgGrid failed to render due to JSON serialization error: {e}")
        st.write("Falling back to basic table view:")

        # Display basic dataframe (without key parameter which isn't supported in older versions)
        st.dataframe(df, height=height, use_container_width=True)

        # Create a mock response for compatibility
        grid_response = {"data": df.to_dict("records"), "selected_rows": [], "selected_data": []}

    # Display enhanced usage instructions and selection summary
    if show_hints:
        with st.expander("💡 AgGrid Usage Tips", expanded=False):
            st.markdown(
                """
            **🔍 Filtering & Searching:**
            - Use the floating filter boxes below column headers for instant filtering
            - Click the ⋮ menu on any column header for advanced filter options
            - Use the sidebar (→) for drag-and-drop grouping and column management

            **📊 Grouping & Analysis:**
            - Drag columns from the sidebar to "Row Groups" to group data
            - Drag numeric columns to "Values" to see aggregations (sum, avg, min, max)
            - Use "Pivot Mode" in the sidebar for cross-tabulation analysis

            **✅ Row Selection:**
            - **Header checkbox**: Click checkbox in header to select/deselect all visible rows
            - **Individual checkboxes**: Use checkboxes to select individual rows
            - **Ctrl+A**: Select all visible rows
            - **Ctrl+Click**: Multi-select rows
            - Selected row count is displayed below the grid

            **📥 Built-in Export Options:**
            - **Right-click on the grid** to access export menu
            - **Export to CSV**: Download filtered/selected data as CSV
            - **Export to Excel**: Download filtered/selected data as Excel
            - **Copy to Clipboard**: Copy data for pasting elsewhere
            - **Print**: Print the current view of the data

            **⌨️ Advanced Features:**
            - Right-click on the grid for context menu options
            - Ctrl+Right-click for additional context menu
            - Tab/Shift+Tab: Navigate between cells and filters
            - Drag column borders to resize columns
            """
            )

    # Return comprehensive response
    return {
        "grid_response": grid_response,
        "selected_count": len(grid_response["selected_rows"])
        if grid_response["selected_rows"] is not None
        else 0,
        "filtered_count": len(grid_response["data"])
        if grid_response["data"] is not None
        else len(df),
    }


def render_progress_bar(current_step, total_steps, step_name):
    """Render a progress bar for processes"""
    progress = st.progress(0)
    # Avoid division by zero
    if total_steps > 0:
        progress.progress(current_step / total_steps)
    else:
        progress.progress(0)
    st.caption(f"Step {current_step}/{total_steps}: {step_name}")
