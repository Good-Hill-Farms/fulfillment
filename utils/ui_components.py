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
                       filterable=True, sortable=True, editable=False, show_hints=True, enable_download=True,
                       enable_sidebar=True, enable_pivot=True, enable_value_aggregation=True, 
                       enhanced_menus=True):
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
            menuTabs=['filterMenuTab', 'generalMenuTab', 'columnsMenuTab'] if filterable else []
        )
        
        # Configure selection
        if selection_mode != 'disabled':
            gb.configure_selection(
                selection_mode, 
                use_checkbox=True,
                header_checkbox=True,  # Adds select all checkbox in header
                groupSelectsChildren=True,
                groupSelectsFiltered=True
            )
        
        # Configure enhanced column types for better filtering and grouping
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Numeric columns with enhanced number filter and aggregation
                gb.configure_column(
                    col, 
                    type=["numericColumn"],
                    filter="agNumberColumnFilter",
                    enableValue=True,  # Enable for aggregation
                    aggFunc=['sum', 'avg', 'min', 'max', 'count'],
                    floatingFilter=True
                )
            elif df[col].dtype == 'datetime64[ns]':
                # Date columns with enhanced date filter
                gb.configure_column(
                    col,
                    filter="agDateColumnFilter",
                    type=["dateColumn"],
                    floatingFilter=True
                )
            else:
                # Text columns with enhanced text filter and grouping
                gb.configure_column(
                    col,
                    filter="agTextColumnFilter",
                    enableRowGroup=True,  # Enable for grouping
                    floatingFilter=True
                )
        
        # Configure sidebar with working configuration for streamlit-aggrid 1.0.5
        if enable_sidebar:
            try:
                # Use the correct method for streamlit-aggrid 1.0.5
                gb.configure_side_bar()
            except Exception as e:
                print(f"Sidebar configuration failed: {e}")
                pass
        
        # Configure basic grouping (simplified to avoid issues)
        if groupable:
            try:
                gb.configure_grid_options(
                    groupSelectsChildren=True,
                    groupSelectsFiltered=True,
                    suppressRowGroupHidesColumns=False,
                    groupDefaultExpanded=1
                )
            except Exception as e:
                print(f"Grouping configuration failed: {e}")
                pass
        
        # Configure basic aggregation for numeric columns
        if enable_value_aggregation:
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    try:
                        gb.configure_column(
                            col,
                            enableValue=True
                        )
                    except Exception as e:
                        # Skip if column configuration causes issues
                        pass
        
        # Build grid options
        grid_options = gb.build()     
        
        # Create the AgGrid with simplified configuration  
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
    
    
    # Display enhanced usage instructions and selection summary
    if show_hints:
        with st.expander("💡 AgGrid Usage Tips", expanded=False):
            st.markdown("""
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
            """)
    
    
    # Return comprehensive response
    return {
        'grid_response': grid_response,
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
                enable_enterprise_modules=False,  # Disable to prevent JSON serialization issues
                enable_sidebar=True,
                enable_pivot=True,
                enable_value_aggregation=True,
                groupable=True,
                filterable=True
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
        st.warning("Cannot calculate inventory changes - order data is missing required columns.")
    
    # Add complete inventory view
    st.markdown("---")
    st.subheader("📦 Complete Inventory")
    st.write("**All inventory items across warehouses:** (Use sidebar to filter and group)")
    
    # Prepare full inventory for display
    full_inventory_df = inventory_summary.copy()
    
    # Create ag-Grid table for full inventory
    full_inventory_table = create_aggrid_table(
        full_inventory_df,
        height=500,
        key="full_inventory_grid", 
        theme="alpine",
        selection_mode='multiple',
        show_hints=False,
        enable_enterprise_modules=False,
        enable_sidebar=True,
        enable_pivot=True,
        enable_value_aggregation=True,
        groupable=True,
        filterable=True
    )
    
    # Show selection summary for full inventory
    if full_inventory_table['selected_count'] > 0:
        selected_inventory = pd.DataFrame(full_inventory_table['grid_response']['selected_rows'])
        st.info(f"📊 Selected {full_inventory_table['selected_count']} inventory items")
        
        # Show summary of selected inventory
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'AvailableQty' in selected_inventory.columns:
                total_available = pd.to_numeric(selected_inventory['AvailableQty'], errors='coerce').sum()
                st.metric("📦 Total Available Qty", f"{total_available:,.0f}")
        with col2:
            if 'Balance' in selected_inventory.columns:
                total_balance = pd.to_numeric(selected_inventory['Balance'], errors='coerce').sum()
                st.metric("⚖️ Total Balance", f"{total_balance:,.0f}")
        with col3:
            warehouses = selected_inventory['WarehouseName'].nunique() if 'WarehouseName' in selected_inventory.columns else 0
            st.metric("🏭 Warehouses", warehouses)

def render_summary_dashboard(processed_orders, inventory_df, processing_stats=None, warehouse_performance=None):
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
