from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from st_aggrid import (
    AgGrid,
    ColumnsAutoSizeMode,
    DataReturnMode,
    GridOptionsBuilder,
    GridUpdateMode,
)


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

def create_aggrid_table(df, height=400, selection_mode='multiple', enable_enterprise_modules=False, 
                       fit_columns_on_grid_load=False, theme='alpine', key=None, groupable=True, 
                       filterable=True, sortable=True, editable=False, show_hints=True, enable_download=True):
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
    
    Returns:
        dict: Contains AgGrid response and additional info
    """
    try:
        # Create GridOptionsBuilder with simplified configuration
        gb = GridOptionsBuilder.from_dataframe(df)
        
        # Configure default column properties (simplified to avoid JSON serialization issues)
        gb.configure_default_column(
            filterable=filterable,
            sortable=sortable,
            resizable=True,
            editable=editable
        )
        
        # Configure selection
        if selection_mode != 'disabled':
            gb.configure_selection(selection_mode, use_checkbox=True)
        
        # Configure side bar only if groupable and enterprise modules are enabled
        if groupable and enable_enterprise_modules:
            gb.configure_side_bar()
        
        # Build grid options
        grid_options = gb.build()
        
        # Show helpful hints if enabled (but skip if inside an expander to avoid nesting)
        if show_hints:
            try:
                with st.expander("💡 How to use this table", expanded=False):
                    st.markdown("""
                    **🔍 Filtering & Searching:**
                    - Click the ⋮ menu in any column header to filter
                    - Use the search box in filters for text matching
                    - Set number ranges, date ranges, and exact matches
                    
                    **📊 Grouping & Sorting:**
                    - Click column headers to sort (click again to reverse)
                    - Hold Shift + click to sort by multiple columns
                    
                    **✅ Selection:**
                    - Check boxes to select rows
                    - Use Ctrl/Cmd + click for multiple selections
                    - Selected data appears in summary below
                    
                    **🎛️ Column Management:**
                    - Drag column borders to resize
                    - Right-click columns for more options
                    """)
            except Exception:
                # Skip hints if inside an expander (nested expanders not allowed)
                pass
        
        # Create the AgGrid with simplified configuration to avoid JSON serialization issues
        grid_response = AgGrid(
            df,
            gridOptions=grid_options,
            height=height,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=fit_columns_on_grid_load,
            theme=theme,
            enable_enterprise_modules=enable_enterprise_modules,
            key=key
        )
        
    except Exception as e:
        # Fallback to basic Streamlit dataframe if AgGrid fails
        st.error(f"AgGrid failed to render due to JSON serialization error: {e}")
        st.write("Falling back to basic table view:")
        
        # Display basic dataframe (without key parameter which isn't supported in older versions)
        st.dataframe(df, height=height, use_container_width=True)
        
        # Create a mock response for compatibility
        grid_response = {
            'data': df.to_dict('records'),
            'selected_rows': [],
            'selected_data': []
        }
    
    # Add download functionality if enabled
    download_data = {}
    if enable_download:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download all data
            csv_all = df.to_csv(index=False)
            st.download_button(
                label="📥 Download All Data",
                data=csv_all,
                file_name=f"data_all_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"{key}_download_all" if key else "download_all"
            )
        
        with col2:
            # Download filtered data
            if grid_response['data'] is not None and len(grid_response['data']) > 0:
                filtered_df = pd.DataFrame(grid_response['data'])
                csv_filtered = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=csv_filtered,
                    file_name=f"data_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"{key}_download_filtered" if key else "download_filtered"
                )
            else:
                st.button("📥 Download Filtered Data", disabled=True, help="No filtered data available", key=f"{key}_download_filtered_disabled" if key else "download_filtered_disabled")
        
        with col3:
            # Download selected data
            if grid_response['selected_rows'] is not None and len(grid_response['selected_rows']) > 0:
                selected_df = pd.DataFrame(grid_response['selected_rows'])
                csv_selected = selected_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Selected Data",
                    data=csv_selected,
                    file_name=f"data_selected_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"{key}_download_selected" if key else "download_selected"
                )
            else:
                st.button("📥 Download Selected Data", disabled=True, help="No rows selected", key=f"{key}_download_selected_disabled" if key else "download_selected_disabled")
        
        # Store download data for external use
        download_data = {
            'all_data': df,
            'filtered_data': pd.DataFrame(grid_response['data']) if grid_response['data'] is not None and len(grid_response['data']) > 0 else pd.DataFrame(),
            'selected_data': pd.DataFrame(grid_response['selected_rows']) if grid_response['selected_rows'] is not None and len(grid_response['selected_rows']) > 0 else pd.DataFrame()
        }
    
    # Display selection summary
    if grid_response['selected_rows'] is not None and len(grid_response['selected_rows']) > 0:
        st.info(f"✅ Selected {len(grid_response['selected_rows'])} rows out of {len(df)} total rows")
    
    # Return comprehensive response
    return {
        'grid_response': grid_response,
        'download_data': download_data,
        'selected_count': len(grid_response['selected_rows']) if grid_response['selected_rows'] is not None else 0,
        'filtered_count': len(grid_response['data']) if grid_response['data'] is not None else len(df)
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
            st.write("**Inventory items affected by orders:** (Use sidebar to filter and group)")
            
            # Prepare data for ag-Grid
            display_df = changes_df[['WarehouseName', 'Sku', 'AvailableQty', 'Before Order', 'After Order', 'Change']].copy()
            display_df.columns = ['Warehouse', 'SKU', 'Available Qty', 'Current Balance', 'Remaining Balance', 'Quantity Used']
            
            # Create ag-Grid table with filtering and grouping (disable hints to avoid nested expanders)
            inventory_table = create_aggrid_table(
                display_df,
                height=400,
                key="inventory_changes_grid",
                theme="alpine",
                show_hints=False,
                enable_enterprise_modules=False  # Disable to prevent JSON serialization issues
            )
            
            # Show selection summary for inventory changes
            if inventory_table['selected_count'] > 0:
                st.info(f"📊 Selected {inventory_table['selected_count']} inventory items for analysis")
            
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

def render_summary_dashboard(processed_orders, inventory_df, processing_stats=None, warehouse_performance=None):
    """
    Enhanced summary dashboard with comprehensive analytics and decision-making metrics

    Args:
        processed_orders: DataFrame of processed orders
        inventory_df: DataFrame of inventory
        processing_stats: Dictionary of processing statistics
        warehouse_performance: Dictionary of warehouse performance metrics
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

    # Enhanced Key Metrics Section
    st.subheader("📊 Key Performance Indicators")
    
    # First row - Core metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_orders = len(processed_orders["ordernumber"].unique()) if "ordernumber" in processed_orders.columns else 0
        st.metric("Unique Orders", f"{total_orders:,}")

    with col2:
        total_items = len(processed_orders)
        st.metric("Total Line Items", f"{total_items:,}")

    with col3:
        fulfillment_centers = processed_orders["Fulfillment Center"].nunique() if "Fulfillment Center" in processed_orders.columns else 0
        st.metric("Active Warehouses", fulfillment_centers)

    with col4:
        issues = processed_orders[processed_orders["Issues"] != ""].shape[0] if "Issues" in processed_orders.columns else 0
        issue_rate = round((issues / len(processed_orders)) * 100, 1) if len(processed_orders) > 0 else 0
        st.metric("Items with Issues", f"{issues:,}", delta=f"{issue_rate}% of total", delta_color="inverse")

    # Second row - Processing efficiency metrics
    with col1:
        if processing_stats and 'total_quantity_processed' in processing_stats:
            total_qty = processing_stats['total_quantity_processed']
            st.metric("Total Quantity", f"{total_qty:,.0f}")
        else:
            quantities = pd.to_numeric(processed_orders['Transaction Quantity'], errors='coerce') if 'Transaction Quantity' in processed_orders.columns else pd.Series([0])
            st.metric("Total Quantity", f"{quantities.sum():,.0f}")

    with col2:
        if processing_stats and 'avg_quantity_per_item' in processing_stats:
            avg_qty = processing_stats['avg_quantity_per_item']
            st.metric("Avg Qty/Item", f"{avg_qty:.1f}")
        else:
            quantities = pd.to_numeric(processed_orders['Transaction Quantity'], errors='coerce') if 'Transaction Quantity' in processed_orders.columns else pd.Series([1])
            st.metric("Avg Qty/Item", f"{quantities.mean():.1f}")

    with col3:
        unique_skus = len(processed_orders['sku'].unique()) if 'sku' in processed_orders.columns else 0
        st.metric("Unique SKUs", f"{unique_skus:,}")

    with col4:
        if processing_stats and 'primary_fulfillment_center' in processing_stats:
            primary_fc = processing_stats['primary_fulfillment_center']
            st.metric("Primary Warehouse", primary_fc)
        else:
            if "Fulfillment Center" in processed_orders.columns:
                primary_fc = processed_orders["Fulfillment Center"].mode().iloc[0] if not processed_orders["Fulfillment Center"].mode().empty else "Unknown"
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
    
    # Warehouse Performance Comparison
    if warehouse_performance:
        st.subheader("🏭 Warehouse Performance Comparison")
        
        # Create performance comparison dataframe
        perf_data = []
        for warehouse, metrics in warehouse_performance.items():
            perf_data.append({
                'Warehouse': warehouse,
                'Total Orders': metrics.get('total_orders', 0),
                'Line Items': metrics.get('total_line_items', 0),
                'Unique SKUs': metrics.get('unique_skus', 0),
                'Issue Rate (%)': metrics.get('issue_rate', 0),
                'Avg Qty/Item': metrics.get('avg_quantity_per_item', 0),
                'Inventory Balance': metrics.get('inventory_balance', 0),
                'Inventory SKUs': metrics.get('inventory_sku_count', 0)
            })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            
            st.write("**Warehouse Performance Metrics:** (Use sidebar to filter and compare)")
            
            # Create ag-Grid table for warehouse performance (disable hints to avoid nested expanders)
            perf_table = create_aggrid_table(
                perf_df,
                height=300,
                key="warehouse_performance_grid",
                theme="alpine",
                groupable=False,  # Disable grouping for this summary table
                selection_mode='single',
                show_hints=False,
                enable_enterprise_modules=False  # Disable to prevent JSON serialization issues
            )
            
            # Show selection info if warehouse is selected
            if perf_table['selected_count'] > 0:
                selected_warehouse = pd.DataFrame(perf_table['grid_response']['selected_rows'])
                warehouse_name = selected_warehouse.iloc[0]['Warehouse']
                st.info(f"📊 Selected: **{warehouse_name}** - Review detailed metrics above")
            
            # Issue rate comparison chart
            fig_issues = px.bar(
                perf_df,
                x='Warehouse',
                y='Issue Rate (%)',
                title='Issue Rate by Warehouse (Lower is Better)',
                color='Issue Rate (%)',
                color_continuous_scale='RdYlGn_r',
                text='Issue Rate (%)'
            )
            fig_issues.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            st.plotly_chart(fig_issues, use_container_width=True)

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
    
    # Inventory Health & Alerts Section
    st.subheader("🔔 Inventory Health & Alerts")
    
    # Critical inventory alerts
    if processing_stats:
        alert_col1, alert_col2, alert_col3 = st.columns(3)
        
        with alert_col1:
            zero_balance = processing_stats.get('zero_balance_items', 0)
            if zero_balance > 0:
                st.error(f"🚨 {zero_balance} items are OUT OF STOCK")
            else:
                st.success("✅ No out-of-stock items")
        
        with alert_col2:
            low_balance = processing_stats.get('low_balance_items', 0)
            if low_balance > 0:
                st.warning(f"⚠️ {low_balance} items have LOW STOCK (≤10)")
            else:
                st.success("✅ No low stock alerts")
        
        with alert_col3:
            total_shortages = processing_stats.get('total_shortages', 0)
            if total_shortages > 0:
                st.error(f"❌ {total_shortages} shortage instances detected")
            else:
                st.success("✅ No shortages detected")
    
    # Decision Making Insights
    st.subheader("💡 Decision Making Insights")
    
    insights = []
    
    # Warehouse efficiency insights
    if warehouse_performance:
        # Find best and worst performing warehouses
        best_warehouse = min(warehouse_performance.items(), key=lambda x: x[1].get('issue_rate', 100))
        worst_warehouse = max(warehouse_performance.items(), key=lambda x: x[1].get('issue_rate', 0))
        
        if best_warehouse[1].get('issue_rate', 0) != worst_warehouse[1].get('issue_rate', 0):
            insights.append(f"🏆 **Best Performing Warehouse**: {best_warehouse[0]} (Issue Rate: {best_warehouse[1].get('issue_rate', 0):.1f}%)")
            insights.append(f"🔧 **Needs Attention**: {worst_warehouse[0]} (Issue Rate: {worst_warehouse[1].get('issue_rate', 0):.1f}%)")
    
    # Volume distribution insights
    if processing_stats and 'fulfillment_center_distribution' in processing_stats:
        fc_dist = processing_stats['fulfillment_center_distribution']
        total_items = sum(fc_dist.values())
        
        for fc, count in fc_dist.items():
            percentage = (count / total_items) * 100
            if percentage > 70:
                insights.append(f"⚖️ **Load Imbalance**: {fc} is handling {percentage:.1f}% of orders")
            elif percentage < 10 and len(fc_dist) > 1:
                insights.append(f"📉 **Underutilized**: {fc} is only handling {percentage:.1f}% of orders - potential capacity available")
    
    # Shortage insights
    if processing_stats and 'shortages_by_fulfillment_center' in processing_stats:
        shortage_fc = processing_stats['shortages_by_fulfillment_center']
        for fc, shortage_count in shortage_fc.items():
            insights.append(f"📦 **Inventory Alert**: {fc} has {shortage_count} shortage instances - review restocking priorities")
    
    # Display insights
    if insights:
        for insight in insights:
            st.info(insight)
    else:
        st.success("✅ No critical insights at this time - operations appear to be running smoothly")
    
    # Processing Summary
    if processing_stats:
        with st.expander("📋 Detailed Processing Summary", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Order Processing Stats:**")
                st.write(f"- Total Orders: {processing_stats.get('total_orders', 0):,}")
                st.write(f"- Total Line Items: {processing_stats.get('total_line_items', 0):,}")
                st.write(f"- Unique SKUs: {processing_stats.get('unique_skus', 0):,}")
                st.write(f"- Total Quantity: {processing_stats.get('total_quantity_processed', 0):,.0f}")
                st.write(f"- Average Qty/Item: {processing_stats.get('avg_quantity_per_item', 0):.2f}")
            
            with col2:
                st.write("**Inventory & Issues:**")
                st.write(f"- Items with Issues: {processing_stats.get('items_with_issues', 0):,}")
                st.write(f"- Issue Rate: {processing_stats.get('issue_rate', 0):.2f}%")
                st.write(f"- Total Inventory Items: {processing_stats.get('total_inventory_items', 0):,}")
                st.write(f"- Zero Balance Items: {processing_stats.get('zero_balance_items', 0):,}")
                st.write(f"- Low Balance Items: {processing_stats.get('low_balance_items', 0):,}")
                
            # Timestamp
            if 'processing_timestamp' in processing_stats:
                timestamp = pd.to_datetime(processing_stats['processing_timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                st.caption(f"Last updated: {timestamp}")
