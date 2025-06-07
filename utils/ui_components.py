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
import plotly.graph_objects as go


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

def create_aggrid_table(df, height=400, selection_mode='multiple', enable_enterprise_modules=True, 
                       fit_columns_on_grid_load=False, theme='alpine', key=None, groupable=True, 
                       filterable=True, sortable=True, editable=False, show_hints=False, enable_download=True,
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
            menuTabs=['filterMenuTab', 'generalMenuTab', 'columnsMenuTab'] if filterable else [],
            minWidth=120,  # Set minimum column width
            width=150,     # Set default column width
            suppressSizeToFit=False  # Allow columns to be resized to fit content
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
            # Set specific widths for important columns
            col_width = 150  # default
            if any(keyword in col.lower() for keyword in ['sku', 'id', 'number']):
                col_width = 180  # wider for SKUs and IDs
            elif any(keyword in col.lower() for keyword in ['name', 'description']):
                col_width = 200  # wider for names
            elif any(keyword in col.lower() for keyword in ['qty', 'quantity', 'balance']):
                col_width = 120  # narrower for numbers
            
            if df[col].dtype in ['int64', 'float64']:
                # Numeric columns with enhanced number filter and aggregation
                gb.configure_column(
                    col, 
                    type=["numericColumn"],
                    filter="agNumberColumnFilter",
                    enableValue=True,  # Enable for aggregation
                    aggFunc=['sum', 'avg', 'min', 'max', 'count'],
                    floatingFilter=True,
                    width=col_width,
                    minWidth=100
                )
            elif df[col].dtype == 'datetime64[ns]':
                # Date columns with enhanced date filter
                gb.configure_column(
                    col,
                    filter="agDateColumnFilter",
                    type=["dateColumn"],
                    floatingFilter=True,
                    width=col_width,
                    minWidth=120
                )
            else:
                # Text columns with enhanced text filter and grouping
                gb.configure_column(
                    col,
                    filter="agTextColumnFilter",
                    enableRowGroup=True,  # Enable for grouping
                    floatingFilter=True,
                    width=col_width,
                    minWidth=120
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
            
    st.markdown("**📊 Inventory Analysis**")
    
    # Create tabs for different inventory views with Grouped Shortages as first tab
    inv_tab_shortages, inv_tab1, inv_tab2 = st.tabs(["⚠️ Grouped Shortages", "📦 Inventory minus Orders", "🎯 Inventory minus Staged Orders"])
    
    # First tab: Grouped Shortages
    with inv_tab_shortages:
        st.markdown("**⚠️ Inventory Shortages by Group**")
        
        # Check if shortage information is available in session state
        if 'shortage_summary' in st.session_state and not st.session_state.shortage_summary.empty:
            shortage_summary = st.session_state.shortage_summary
            shortage_count = len(shortage_summary)
            
            st.markdown(f"**Found {shortage_count} items with inventory shortages**")
            
            # Find SKU column using flexible matching
            sku_col = None
            for col in shortage_summary.columns:
                if any(keyword in col.lower() for keyword in ['sku', 'item', 'product', 'part']):
                    sku_col = col
                    break
                    
            # Find order column using flexible matching
            order_col = None
            for col in shortage_summary.columns:
                if any(keyword in col.lower() for keyword in ['order', 'ordernumber', 'orderid']):
                    order_col = col
                    break
            
            # Display grouped by SKU
            if sku_col:
                st.markdown("**Shortages by SKU**")
                sku_grouped = shortage_summary.groupby(sku_col).size().reset_index()
                sku_grouped.columns = [sku_col, 'Count']
                sku_grouped = sku_grouped.sort_values('Count', ascending=False)
                
                # Use aggrid for better display
                create_aggrid_table(
                    sku_grouped,
                    height=300,
                    key="shortage_sku_grid",
                    theme="alpine",
                    selection_mode='multiple',
                    show_hints=False,
                    enable_enterprise_modules=True
                )
            
            # Display grouped by Order
            if order_col:
                st.markdown("**Shortages by Order**")
                order_grouped = shortage_summary.groupby(order_col).size().reset_index()
                order_grouped.columns = [order_col, 'Count']
                order_grouped = order_grouped.sort_values('Count', ascending=False)
                
                # Use aggrid for better display
                create_aggrid_table(
                    order_grouped,
                    height=300,
                    key="shortage_order_grid",
                    theme="alpine",
                    selection_mode='multiple',
                    show_hints=False,
                    enable_enterprise_modules=True
                )
                
            # Show full shortage details
            st.markdown("**Complete Shortage Details**")
            create_aggrid_table(
                shortage_summary,
                height=400,
                key="full_shortage_grid",
                theme="alpine",
                selection_mode='multiple',
                show_hints=False,
                enable_enterprise_modules=True,
                enable_sidebar=True
            )
        else:
            st.info("No inventory shortages detected or shortage data is not available.")
    
    with inv_tab1:
        st.markdown("**📦 Current Inventory minus Orders in Processing**")
        # Process inventory data
        if 'Sku' not in inventory_df.columns or 'WarehouseName' not in inventory_df.columns:
            st.error("Inventory data is missing required columns (Sku, WarehouseName)")
            return
        
        # Group inventory by SKU and warehouse
        agg_dict = {}
        
        # Check which columns are available for aggregation
        if 'AvailableQty' in inventory_df.columns:
            agg_dict['AvailableQty'] = 'sum'
        
        if 'Balance' in inventory_df.columns:
            agg_dict['Balance'] = 'max'
            
        if not agg_dict:
            st.error("Inventory data is missing required quantity columns (AvailableQty or Balance)")
            return
            
        inventory_summary = inventory_df.groupby(['WarehouseName', 'Sku']).agg(agg_dict).reset_index()
        
        # Calculate order quantities by SKU
        if 'sku' in processed_orders.columns and 'Transaction Quantity' in processed_orders.columns:
            # Get only unstaged orders for this calculation
            orders_in_processing = processed_orders[processed_orders['staged'] == False].copy() if 'staged' in processed_orders.columns else processed_orders.copy()
            
            order_quantities = orders_in_processing.groupby(['sku', 'Fulfillment Center']).agg({
                'Transaction Quantity': 'sum'
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
            if 'Balance' in projected_inventory.columns and projected_inventory['Balance'].sum() > 0:
                projected_inventory['Before Order'] = projected_inventory['Balance']
                projected_inventory['After Order'] = projected_inventory['Balance']
            elif 'AvailableQty' in projected_inventory.columns:
                projected_inventory['Before Order'] = projected_inventory['AvailableQty']
                projected_inventory['After Order'] = projected_inventory['AvailableQty']
            
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
            
            # Filter to only show items with changes
            changes_df = projected_inventory[projected_inventory['Change'] < 0].copy()
            changes_df['Change'] = changes_df['Change'].abs()
            
            if not changes_df.empty:
                st.write(f"Found {len(changes_df)} items with inventory changes from orders in processing:")
                
                # Create ag-Grid table
                create_aggrid_table(
                    changes_df,
                    height=400,
                    key="inventory_changes_grid",
                    theme="alpine",
                    selection_mode='multiple',
                    show_hints=False,
                    enable_enterprise_modules=True,
                    enable_sidebar=True,
                    enable_pivot=True,
                    enable_value_aggregation=True,
                    groupable=True,
                    filterable=True
                )
            else:
                st.info("No inventory changes detected from orders in processing")
        else:
            st.warning("Cannot calculate inventory changes - order data is missing required columns")
    
    with inv_tab2:
        st.markdown("**🎯 Current Inventory minus Staged Orders**")
        
        # Get only staged orders
        staged_orders = processed_orders[processed_orders['staged'] == True].copy() if 'staged' in processed_orders.columns else pd.DataFrame()
        
        if staged_orders.empty:
            st.info("No staged orders to calculate inventory impact")
            return
            
        # Calculate total quantity per SKU in staged orders
        staged_qty = staged_orders.groupby(['sku', 'Fulfillment Center']).agg({
            'Transaction Quantity': 'sum'
        }).reset_index()
        
        # Create a copy of inventory for staged calculations
        staged_inventory = inventory_summary.copy()
        
        # Create columns for before/after staged
        if 'Balance' in staged_inventory.columns and staged_inventory['Balance'].sum() > 0:
            staged_inventory['Before Staged'] = staged_inventory['Balance']
            staged_inventory['After Staged'] = staged_inventory['Balance']
        elif 'AvailableQty' in staged_inventory.columns:
            staged_inventory['Before Staged'] = staged_inventory['AvailableQty']
            staged_inventory['After Staged'] = staged_inventory['AvailableQty']
        
        # Update inventory based on staged orders
        for _, row in staged_qty.iterrows():
            sku = row['sku']
            fc = row['Fulfillment Center']
            qty = row['Transaction Quantity']
            
            # Convert fulfillment center to warehouse name
            warehouse = fc_to_warehouse.get(fc, fc)
            
            # Find matching inventory row
            matching_rows = staged_inventory[
                (staged_inventory['Sku'] == sku) & 
                (staged_inventory['WarehouseName'] == warehouse)
            ]
            
            if not matching_rows.empty:
                idx = matching_rows.index[0]
                # Update after staged quantity
                current = staged_inventory.loc[idx, 'After Staged']
                staged_inventory.loc[idx, 'After Staged'] = max(0, current - qty)
        
        # Calculate the change in inventory from staged orders
        staged_inventory['Change from Staged'] = staged_inventory['After Staged'] - staged_inventory['Before Staged']
        
        # Filter to only show items affected by staging
        staged_changes = staged_inventory[staged_inventory['Change from Staged'] < 0].copy()
        staged_changes['Change from Staged'] = staged_changes['Change from Staged'].abs()
        
        if not staged_changes.empty:
            st.write(f"Found {len(staged_changes)} items with inventory changes from staged orders:")
            
            # Create ag-Grid table
            create_aggrid_table(
                staged_changes,
                height=400,
                key="staged_inventory_changes_grid",
                theme="alpine",
                selection_mode='multiple',
                show_hints=False,
                enable_enterprise_modules=True,
                enable_sidebar=True,
                enable_pivot=True,
                enable_value_aggregation=True,
                groupable=True,
                filterable=True
            )
        else:
            st.info("No inventory changes detected from staged orders")
    
    # Add complete inventory view
    st.markdown("---")
    st.subheader("📦 Complete Inventory")
    st.write("**All inventory items across warehouses:** (Use sidebar to filter and group)")
    
    # Use the already-processed inventory summary from session state if available
    if 'inventory_summary' in st.session_state and not st.session_state.inventory_summary.empty:
        full_inventory_df = st.session_state.inventory_summary.copy()
        st.info(f"📊 Showing {len(full_inventory_df)} inventory items with updated balances from processed orders")
    else:
        # Fallback: use raw inventory data if no processed summary is available
        if inventory_df is not None:
            full_inventory_df = inventory_df.copy()
            
            # Standardize column names
            if 'WarehouseName' in full_inventory_df.columns:
                full_inventory_df['Warehouse'] = full_inventory_df['WarehouseName']
            if 'Sku' in full_inventory_df.columns:
                full_inventory_df['Inventory SKU'] = full_inventory_df['Sku']
            if 'Balance' in full_inventory_df.columns:
                full_inventory_df['Current Balance'] = full_inventory_df['Balance'].apply(
                    lambda x: f"{x:,.0f}" if pd.notna(x) else "0"
                )
            elif 'AvailableQty' in full_inventory_df.columns:
                full_inventory_df['Current Balance'] = full_inventory_df['AvailableQty'].apply(
                    lambda x: f"{x:,.0f}" if pd.notna(x) else "0"
                )
            
            st.warning("📋 Showing raw inventory data - no orders have been processed yet")
        else:
            st.error("❌ No inventory data available")
            return
    
    # Display complete inventory
    create_aggrid_table(
        full_inventory_df,
        height=400,
        key="complete_inventory_grid",
        theme="alpine",
        selection_mode='multiple',
        show_hints=False,
        enable_enterprise_modules=True,
        enable_sidebar=True,
        enable_pivot=True,
        enable_value_aggregation=True,
        groupable=True,
        filterable=True
    )

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

def render_orders_tab(processed_orders, shortage_summary=None):
    """Renders the Orders tab with data table and metrics"""
    st.markdown("**Orders in Processing**")
    
    if processed_orders is not None and not processed_orders.empty:
        # First, ensure the staged column exists
        if 'staged' not in processed_orders.columns:
            processed_orders['staged'] = False
            
        # Filter to only show unstaged orders
        display_orders = processed_orders[processed_orders['staged'] == False].copy()
        
        # Don't show the staged column to the user
        if 'staged' in display_orders.columns:
            display_orders = display_orders.drop(columns=['staged'])
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_orders = len(display_orders["ordernumber"].unique()) if "ordernumber" in display_orders.columns else 0
            st.metric("📋 Unique Orders", f"{total_orders:,}")
        
        with col2:
            total_items = len(display_orders)
            st.metric("📦 Line Items", f"{total_items:,}")
        
        with col3:
            issues = display_orders[display_orders["Issues"] != ""].shape[0] if "Issues" in display_orders.columns else 0
            issue_rate = round((issues / len(display_orders)) * 100, 1) if len(display_orders) > 0 else 0
            st.metric("⚠️ Items with Issues", f"{issues:,}", delta=f"{issue_rate}% of total", delta_color="inverse")
        
        with col4:
            fulfillment_centers = display_orders["Fulfillment Center"].nunique() if "Fulfillment Center" in display_orders.columns else 0
            st.metric("🏭 Warehouses Used", fulfillment_centers)
        
        # Ensure shortage_count matches issues count for consistency
        if shortage_summary is not None and not shortage_summary.empty:
            shortage_count = len(shortage_summary)
            
            # Update session state with shortage_summary for use in other components
            st.session_state.shortage_summary = shortage_summary
        else:
            shortage_count = 0
            
        # Show alerts if there are issues - use shortage_count for consistency
        if issues > 0:
            # Make sure issues count and shortage_count are consistent
            if shortage_count > 0 and shortage_count != issues:
                st.warning(f"⚠️ {issues} items have issues ({shortage_count} inventory shortages) - Check the 📦 Inventory tab for shortage details!")
            else:
                st.warning(f"⚠️ {issues} items have issues - Check the 📦 Inventory tab for shortage details!")
            
        if shortage_summary is not None and not shortage_summary.empty:
            
            # Find SKU column using flexible matching patterns
            sku_keywords = ['sku', 'item', 'product', 'part']
            sku_col = None
            
            # Find the first column that contains any of the SKU keywords
            for col in shortage_summary.columns:
                if any(keyword in col.lower() for keyword in sku_keywords):
                    sku_col = col
                    break
            
            # If still no match, try a case-insensitive exact match for 'sku'
            if not sku_col:
                for col in shortage_summary.columns:
                    if col.lower() == 'sku':
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
            order_keywords = ['order', 'ordernumber', 'order number', 'orderid', 'order id', 'external']
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
            
            # Calculate duplicates
            if 'Issues' in display_orders.columns:
                duplicate_count = len(display_orders[display_orders['Issues'] != '']) - shortage_count if shortage_count > 0 else 0
            else:
                duplicate_count = 0
            
            if shortage_count > 0:
                # Create a clearer, more structured shortage message
                st.error(f"⚠️ INVENTORY SHORTAGES DETECTED:\n" +
                       f"• {shortage_count} line items with shortages\n" +
                       f"• {unique_skus} unique SKUs affected\n" +
                       f"• {affected_orders} orders impacted" +
                       (f"\n• {duplicate_count} items with multiple shortage issues" if duplicate_count > 0 else "") +
                       "\n\nExpand below for detailed shortage information!")
                
                # Add an expander with detailed shortage information including fulfillment center and order IDs
                with st.expander("📋 View Detailed Shortages by Fulfillment Center", expanded=False):
                    if shortage_summary is not None and not shortage_summary.empty:
                        # Create a detailed view of shortages with fulfillment center info
                        detailed_view = shortage_summary.copy()
                        
                        # Find important columns using flexible matching
                        fc_col = next((col for col in detailed_view.columns if any(keyword in col.lower() for keyword in ['fulfillment', 'center', 'warehouse', 'location'])), None)
                        sku_col = next((col for col in detailed_view.columns if any(keyword in col.lower() for keyword in ['sku', 'item', 'product']) and not any(kw in col.lower() for kw in ['shopify', 'related'])), None)
                        order_col = next((col for col in detailed_view.columns if any(keyword in col.lower() for keyword in ['order', 'ordernumber', 'orderid'])), None)
                        shopify_cols = [col for col in detailed_view.columns if any(keyword in col.lower() for keyword in ['shopify', 'related'])]
                        
                        # Group by fulfillment center and warehouse SKU
                        if fc_col and sku_col and order_col:
                            # Group data by fulfillment center and SKU
                            grouped = detailed_view.groupby([fc_col, sku_col])
                            
                            # Create result rows with grouped data
                            result_rows = []
                            for (fc, sku), group in grouped:
                                row = {
                                    "fulfillment_center": fc,
                                    "warehouse_sku": sku
                                }
                                
                                # Collect all related Shopify SKUs
                                for col in shopify_cols:
                                    unique_values = group[col].dropna().unique()
                                    if len(unique_values) > 0:
                                        row["shopify_sku"] = ", ".join(sorted(unique_values.astype(str)))
                                    else:
                                        row["shopify_sku"] = ""
                                
                                # Add affected order IDs
                                unique_orders = sorted(group[order_col].dropna().unique().astype(str))
                                row["order_id"] = ", ".join(unique_orders)
                                
                                # Add counts
                                row["affected_orders"] = len(unique_orders)
                                row["line_items"] = len(group)
                                
                                result_rows.append(row)
                            
                            # Convert to DataFrame and reorder columns
                            if result_rows:
                                grouped_df = pd.DataFrame(result_rows)
                                
                                # Reorder columns for better display
                                column_order = ["fulfillment_center", "warehouse_sku", "shopify_sku", "order_id"]
                                
                                # Display the grouped table
                                st.dataframe(
                                    grouped_df,
                                    height=400,
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                st.info("No detailed shortage information available")
                        else:
                            # Fallback if we can't find the needed columns
                            st.info("Could not identify necessary columns for grouping in shortage data")
                    else:
                        st.info("No shortage data available")
                
                # Add an expander with a single, simplified grouped shortages table
                with st.expander("📦 View Grouped Shortages", expanded=False):
                    if shortage_summary is not None and not shortage_summary.empty:
                        # Find important columns
                        sku_col = next((col for col in shortage_summary.columns if any(keyword in col.lower() for keyword in ['sku', 'item', 'product'])), None)
                        order_col = next((col for col in shortage_summary.columns if any(keyword in col.lower() for keyword in ['order', 'ordernumber', 'orderid'])), None)
                        qty_col = next((col for col in shortage_summary.columns if any(keyword in col.lower() for keyword in ['qty', 'quantity', 'amount'])), None)
                        
                        # Create a clean summary table with only the important columns
                        cols_to_keep = [col for col in [sku_col, order_col, qty_col] if col is not None]
                        
                        if cols_to_keep:
                            clean_summary = shortage_summary[cols_to_keep].copy()
                            
                            # Rename columns for clarity
                            new_names = {}
                            if sku_col: new_names[sku_col] = 'SKU'
                            if order_col: new_names[order_col] = 'Order Number'
                            if qty_col: new_names[qty_col] = 'Shortage Qty'
                            
                            # Apply renames if we have any
                            if new_names:
                                clean_summary = clean_summary.rename(columns=new_names)
                            
                            # Convert to aggregated view with counts
                            if 'SKU' in clean_summary.columns:
                                # Count shortages by SKU
                                agg_dict = {'Order Number': 'nunique'} if 'Order Number' in clean_summary.columns else {}
                                if 'Shortage Qty' in clean_summary.columns:
                                    agg_dict['Shortage Qty'] = 'sum'
                                    
                                # Create the grouped summary with counts
                                grouped_summary = clean_summary.groupby('SKU').agg(agg_dict).reset_index()
                                
                                # Rename aggregated columns
                                col_renames = {}
                                if 'Order Number' in grouped_summary.columns:
                                    col_renames['Order Number'] = 'Orders Affected'
                                
                                if col_renames:
                                    grouped_summary = grouped_summary.rename(columns=col_renames)
                                    
                                # Now add the actual order numbers for each SKU
                                if 'Order Number' in clean_summary.columns:
                                    # Group by SKU and collect all order numbers as a list
                                    orders_by_sku = clean_summary.groupby('SKU')['Order Number'].apply(lambda x: ', '.join(sorted(set(x.astype(str))))).reset_index()
                                    orders_by_sku.rename(columns={'Order Number': 'Affected Order Numbers'}, inplace=True)
                                    
                                    # Merge with the grouped summary
                                    grouped_summary = grouped_summary.merge(orders_by_sku, on='SKU', how='left')
                                    
                                # Add 'Line Items' count
                                items_per_sku = clean_summary.groupby('SKU').size().reset_index()
                                items_per_sku.columns = ['SKU', 'Line Items']
                                grouped_summary = grouped_summary.merge(items_per_sku, on='SKU', how='left')
                                
                                # Reorder columns
                                cols_order = ['SKU', 'Line Items']
                                if 'Orders Affected' in grouped_summary.columns:
                                    cols_order.append('Orders Affected')
                                if 'Shortage Qty' in grouped_summary.columns:
                                    cols_order.append('Shortage Qty')
                                    
                                # Get final columns in right order
                                final_cols = [col for col in cols_order if col in grouped_summary.columns]
                                grouped_summary = grouped_summary[final_cols].sort_values('Line Items', ascending=False)
                                
                                # Display with a standard dataframe for simplified view
                                st.markdown("**Shortage Summary by SKU**")
                                # Use standard dataframe display for a cleaner, simpler look
                                st.dataframe(
                                    grouped_summary,
                                    height=500,  # Increased height for visibility
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                # Fallback to showing the raw shortage summary
                                st.markdown("**Shortage Details**")
                                create_aggrid_table(
                                    shortage_summary,
                                    height=400,
                                    key="shortage_details_table",
                                    theme="alpine",
                                    selection_mode='multiple',
                                    enable_enterprise_modules=True,
                                    show_hints=False
                                )
                        else:
                            st.write("Could not identify key columns in the shortage summary.")
                    else:
                        st.info("No shortage data available.")
        
        # Create AgGrid table
        gb = GridOptionsBuilder.from_dataframe(display_orders)
        gb.configure_selection(selection_mode='multiple', use_checkbox=True)
        gridOptions = gb.build()
        
        grid_response = create_aggrid_table(
            display_orders,
            height=600,
            selection_mode='multiple',
            enable_enterprise_modules=True,
            theme='alpine'
        )
        
        # Show selected rows info and staging option
        if grid_response['grid_response']['selected_rows']:
            selected_rows = grid_response['grid_response']['selected_rows']
            st.session_state.selected_orders = selected_rows
            
            # Display info about selection
            st.success(f"✅ Selected {len(selected_rows)} rows out of {len(display_orders)} total rows")
            
            # Calculate detailed metrics for selection
            selected_unique_orders = len(set([row.get('ordernumber', '') for row in selected_rows]))
            selected_line_items = len(selected_rows)
            selected_total_qty = sum([float(row.get('Transaction Quantity', 0)) for row in selected_rows])
            selected_with_issues = len([row for row in selected_rows if row.get('Issues', '') != ''])
            selected_orders_with_issues = len(set([row.get('ordernumber', '') for row in selected_rows if row.get('Issues', '') != '']))
            
            # Show detailed selection stats
            cols = st.columns(5)
            with cols[0]:
                st.metric("📦 Selected Items", selected_line_items)
            with cols[1]:
                st.metric("📋 Unique Orders", selected_unique_orders)
            with cols[2]:
                st.metric("📊 Total Quantity", f"{selected_total_qty:.0f}")
            with cols[3]:
                warehouses = len(set([row.get('Fulfillment Center', '') for row in selected_rows]))
                st.metric("🏭 Warehouses", warehouses)
            with cols[4]:
                st.metric("⚠️ Items with Issues", selected_with_issues, delta=f"{selected_orders_with_issues} orders")
                
            # Display additional statistics that the user requested
            st.info(f"📊 Summary: {selected_line_items} line items | {selected_unique_orders} unique orders | {selected_with_issues} items with issues | {selected_orders_with_issues} unique orders with issues")
                
            if st.button("Move Selected to Staging"):
                if 'staged_orders' not in st.session_state:
                    st.session_state.staged_orders = pd.DataFrame()
                
                # Ensure the 'staged' column exists in processed_orders
                if 'staged' not in processed_orders.columns:
                    processed_orders['staged'] = False
                    
                # For each selected row, find and mark corresponding rows in processed_orders as staged
                for row in selected_rows:
                    # Create match conditions based on available unique identifiers
                    if 'ordernumber' in row and 'sku' in row:
                        # Match by order number and SKU
                        mask = (
                            (processed_orders['ordernumber'] == row['ordernumber']) & 
                            (processed_orders['sku'] == row['sku'])
                        )
                        # Mark matching rows as staged
                        processed_orders.loc[mask, 'staged'] = True
                
                # Update session state with both staged orders and processed orders
                st.session_state.staged_orders = processed_orders[processed_orders['staged'] == True].copy()
                st.session_state.processed_orders = processed_orders
                
                # Display detailed success message
                st.success(f"✅ Moved to staging: {selected_line_items} line items | {selected_unique_orders} unique orders | {selected_total_qty:.0f} total units")
                
                # Rerun to update the UI
                st.rerun()
        
        # Show staging count
        st.metric("📋 Orders in Staging", len(st.session_state.get('staged_orders', pd.DataFrame())))

def render_inventory_tab(inventory_summary, shortage_summary, grouped_shortage_summary, initial_inventory=None, inventory_comparison=None, processed_orders=None):
    """Renders the Inventory tab with all inventory-related information"""
    st.header("📦 Inventory & Shortages")
    
    # Display only useful shortage summary information
    if shortage_summary is not None and not shortage_summary.empty:
        shortage_instances = shortage_summary.shape[0]
        
        # Find SKU column using flexible matching
        sku_col = next((col for col in shortage_summary.columns if any(keyword in col.lower() for keyword in ['sku', 'item', 'product'])), None)
        unique_skus = shortage_summary[sku_col].nunique() if sku_col else 0
        
        # Find order column using flexible matching
        order_col = next((col for col in shortage_summary.columns if any(keyword in col.lower() for keyword in ['order', 'ordernumber', 'orderid'])), None)
        affected_orders = shortage_summary[order_col].nunique() if order_col else 0
        
        st.markdown("**📈 Shortages Summary**")
        cols = st.columns(3)
        
        with cols[0]:
            st.metric("💼 Line Items", shortage_instances)
            
        with cols[1]:
            st.metric("💹 Unique SKUs", unique_skus)
            
        with cols[2]:
            st.metric("💱 Affected Orders", affected_orders)
    
    # Create updated tabs structure for inventory views
    initial_inventory_tab, inventory_minus_orders_tab, shortages_tab = st.tabs([
        "Initial Inventory",
        "Inventory Minus Orders", 
        "Shortages"
    ])
    
    with initial_inventory_tab:
        st.markdown("**Initial Inventory State**")
        if initial_inventory is not None and not initial_inventory.empty:
            st.dataframe(
                initial_inventory,
                height=600,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No initial inventory data available")
    
    with inventory_minus_orders_tab:
        st.markdown("**Inventory After Processing Orders**")
        
        # Use inventory_comparison if available (shows inventory minus orders)
        if inventory_comparison is not None and not inventory_comparison.empty:
            # Highlight rows where inventory levels are low or negative
            highlight_rows = []
            
            # Find columns for highlighting
            qty_after_col = next((col for col in inventory_comparison.columns 
                               if any(keyword in col.lower() for keyword in ['after', 'remaining', 'final'])), None)
            
            if qty_after_col:
                # Mark rows with low/negative inventory for highlighting
                for i, row in inventory_comparison.iterrows():
                    if pd.notna(row[qty_after_col]) and row[qty_after_col] <= 0:
                        highlight_rows.append(i)
                
                # Style the dataframe with conditional formatting
                styled_df = inventory_comparison.style.apply(
                    lambda x: ['background-color: #ffcccc' if i in highlight_rows else '' for i in range(len(x))], 
                    axis=0
                )
                
                st.dataframe(
                    inventory_comparison,
                    height=600,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                # Just display without highlighting if we can't find the right column
                st.dataframe(
                    inventory_comparison,
                    height=600,
                    use_container_width=True,
                    hide_index=True
                )
        elif inventory_summary is not None and not inventory_summary.empty:
            # Fall back to regular inventory summary if comparison not available
            st.warning("⚠️ Showing raw inventory data - inventory comparison not available")
            
            # Find quantity columns
            qty_col = next((col for col in inventory_summary.columns if any(keyword in col.lower() for keyword in ['qty', 'quantity', 'balance'])), None)
            
            if qty_col:
                # Make a copy to avoid modifying the original
                inventory_display = inventory_summary.copy()
                
                # Rename quantity column for clarity
                inventory_display = inventory_display.rename(columns={qty_col: 'Quantity'})
                
                # Find SKU column
                sku_col = next((col for col in inventory_display.columns if any(keyword in col.lower() for keyword in ['sku', 'item', 'product'])), None)
                
                # Sort the inventory by SKU
                if sku_col:
                    inventory_display = inventory_display.sort_values(by=sku_col)
                
                st.dataframe(
                    inventory_display,
                    height=600,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.dataframe(
                    inventory_summary,
                    height=600,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No inventory data available")
    
    with shortages_tab:
        if grouped_shortage_summary is not None and not grouped_shortage_summary.empty:
            # Show grouped shortages first using standard dataframe for cleaner display
            st.markdown("**Shortages by SKU**")
            st.dataframe(
                grouped_shortage_summary,
                height=550,  # Increased height for better visibility
                use_container_width=True,
                hide_index=True
            )
            
            # Add expander for individual shortage details
            with st.expander("View All Individual Shortage Line Items", expanded=False):
                if shortage_summary is not None and not shortage_summary.empty:
                    create_aggrid_table(shortage_summary, height=400, selection_mode='multiple', key="shortage_table")
                else:
                    st.info("No individual shortage details available")
        elif shortage_summary is not None and not shortage_summary.empty:
            # If no grouped data, show individual shortages directly
            create_aggrid_table(shortage_summary, height=500, selection_mode='multiple', key="shortage_table")
        else:
            st.info("No shortages detected")
    
    # Removed unnecessary tabs to simplify the interface

def render_staging_tab(staged_orders):
    """Renders the Staging tab with staged orders"""
    st.markdown("**Staged Orders**")
    
    # Get only staged orders from processed_orders if available
    if 'processed_orders' in st.session_state and not st.session_state.processed_orders.empty:
        if 'staged' in st.session_state.processed_orders.columns:
            # Use only orders marked as staged=True in processed_orders
            staged_orders = st.session_state.processed_orders[st.session_state.processed_orders['staged'] == True].copy()
            # Remove the staged column as it's not needed in this view
            if 'staged' in staged_orders.columns:
                staged_orders = staged_orders.drop(columns=['staged'])
    
    if not staged_orders.empty:
        # Display key metrics for staged orders with enhanced statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_orders = len(staged_orders["ordernumber"].unique()) if "ordernumber" in staged_orders.columns else 0
            st.metric("📋 Staged Orders", f"{total_orders:,}")
        
        with col2:
            total_items = len(staged_orders)
            st.metric("📦 Staged Line Items", f"{total_items:,}")
        
        with col3:
            if "Transaction Quantity" in staged_orders.columns:
                total_quantity = staged_orders["Transaction Quantity"].sum()
            else:
                total_quantity = 0
            st.metric("📊 Total Units Staged", f"{total_quantity:,.0f}")
        
        with col4:
            fulfillment_centers = staged_orders["Fulfillment Center"].nunique() if "Fulfillment Center" in staged_orders.columns else 0
            st.metric("🏭 Warehouses Used", fulfillment_centers)
            
        with col5:
            staged_issues = staged_orders[staged_orders["Issues"] != ""].shape[0] if "Issues" in staged_orders.columns else 0
            staged_orders_with_issues = staged_orders[staged_orders["Issues"] != ""]["ordernumber"].nunique() if ("Issues" in staged_orders.columns and "ordernumber" in staged_orders.columns) else 0
            st.metric("⚠️ Items with Issues", staged_issues, delta=f"{staged_orders_with_issues} orders")
            
        # Show summary statistics in a highlighted info box
        st.info(f"📊 Staged Order Summary: {total_items} line items | {total_orders} unique orders | {total_quantity:,.0f} total units | {staged_issues} items with issues | {staged_orders_with_issues} unique orders with issues")
        
        # Show unique order numbers in an expander for quick reference
        if "ordernumber" in staged_orders.columns:
            with st.expander("🔍 View Staged Order Numbers", expanded=False):
                unique_order_numbers = sorted(staged_orders["ordernumber"].unique())
                st.write("Unique order numbers in staging:")
                st.code(', '.join(map(str, unique_order_numbers)))
            
        # Add timestamp column if not exists
        if 'staging_timestamp' not in staged_orders.columns:
            staged_orders['staging_timestamp'] = pd.Timestamp.now()
        
        grid_response = create_aggrid_table(
            staged_orders,
            height=600, # Increased from 400 to 600
            selection_mode='multiple',
            enable_enterprise_modules=True,
            theme='alpine'
        )
        
        if grid_response['grid_response']['selected_rows']:
            selected_rows = grid_response['grid_response']['selected_rows']
            selected_count = len(selected_rows)
            
            st.write(f"Selected: {selected_count} items")
            
            # Display summary of selected items
            if selected_count > 0:
                selected_skus = len(set([row.get('sku', '') for row in selected_rows]))
                selected_orders = len(set([row.get('ordernumber', '') for row in selected_rows]))
                selected_qty = sum([row.get('quantity', 0) for row in selected_rows])
                
                st.write(f"📦 {selected_qty} units | 🏷️ {selected_skus} unique SKUs | 📝 {selected_orders} orders")
            
            if st.button("Remove from Staging"):
                selected_indices = [row['_selectedRowNodeInfo']['nodeRowIndex'] 
                                 for row in grid_response['grid_response']['selected_rows']]
                st.session_state.staged_orders = st.session_state.staged_orders.drop(selected_indices)
                st.success(f"✅ {selected_count} items removed from staging")
                st.rerun()

def render_sku_mapping_editor(sku_mappings, data_processor):
    """Renders the SKU Mapping editor interface"""
    st.markdown("**SKU Mapping Adjustments**")
    
    # Convert dictionary to DataFrame if needed
    if sku_mappings is not None:
        if isinstance(sku_mappings, dict):
            # Create a flattened view of the mappings for editing
            rows = []
            for center, mappings in sku_mappings.get('mappings', {}).items():
                for shopify_sku, inventory_sku in mappings.items():
                    rows.append({
                        "fulfillment_center": center,
                        "shopify_sku": shopify_sku,
                        "inventory_sku": inventory_sku
                    })
            
            # Convert to DataFrame if there are mappings
            if rows:
                sku_df = pd.DataFrame(rows)
            else:
                sku_df = pd.DataFrame(columns=["fulfillment_center", "shopify_sku", "inventory_sku"])
        else:
            # If it's already a DataFrame, use it directly
            sku_df = sku_mappings
        
        # Check if DataFrame is not empty
        if not sku_df.empty:
            # Use AgGrid for better filtering and sorting capabilities
            gb = GridOptionsBuilder.from_dataframe(sku_df)
            
            # Configure grid options for better usability
            gb.configure_default_column(
                resizable=True,
                filterable=True,
                sortable=True,
                editable=True
            )
            
            # Add specific filter options for each column
            gb.configure_column("fulfillment_center", filter_params={"filterType": "text"})
            gb.configure_column("shopify_sku", filter_params={"filterType": "text"})
            gb.configure_column("inventory_sku", filter_params={"filterType": "text"})
            
            # Enable pagination and set page size
            gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
            
            # Enable row selection
            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
            
            # Configure sidebar for column filtering
            gb.configure_side_bar()
            
            # Build grid options
            grid_options = gb.build()
            
            # Render the AgGrid table with increased height
            grid_response = AgGrid(
                sku_df,
                gridOptions=grid_options,
                height=600,  # Increased height for better visibility
                data_return_mode="filtered_and_sorted",
                update_mode="value_changed",
                fit_columns_on_grid_load=True,
                theme="alpine",
                allow_unsafe_jscode=True,
                enable_enterprise_modules=True,
                key="sku_mapping_grid"
            )
            
            # Get the updated dataframe
            edited_df = grid_response["data"]
            
            if st.button("Apply SKU Mapping Changes"):
                st.session_state.sku_mappings = edited_df
                
                # Trigger reprocessing of orders with new mappings
                if (st.session_state.orders_df is not None and 
                    st.session_state.inventory_df is not None):
                    with st.spinner("Recalculating allocations using inventory minus staged orders..."):
                        # First, get our inventory dataframe
                        inventory_df = st.session_state.inventory_df.copy()
                        
                        # Extract only staged orders if they exist
                        if ('staged' in st.session_state.processed_orders.columns and 
                            st.session_state.processed_orders[st.session_state.processed_orders['staged'] == True].shape[0] > 0):
                            
                            # Get staged orders
                            staged_orders = st.session_state.processed_orders[st.session_state.processed_orders['staged'] == True].copy()
                            
                            # Create adjusted inventory by subtracting staged orders quantities
                            # Group staged orders by SKU and Fulfillment Center
                            if 'sku' in staged_orders.columns and 'Fulfillment Center' in staged_orders.columns:
                                staged_quantities = staged_orders.groupby(['sku', 'Fulfillment Center'])['quantity'].sum().reset_index()
                                
                                # Only proceed if we have staged quantities
                                if not staged_quantities.empty:
                                    st.info(f"Adjusting inventory based on {len(staged_orders)} staged orders before recalculation")
                                    
                                    # Convert staged quantities to a dictionary for faster lookups
                                    staged_qty_dict = {}
                                    for _, row in staged_quantities.iterrows():
                                        key = (row['sku'], row['Fulfillment Center'])
                                        staged_qty_dict[key] = row['quantity']
                                    
                                    # Adjust inventory quantities
                                    if 'sku' in inventory_df.columns and 'warehouse' in inventory_df.columns and 'quantity' in inventory_df.columns:
                                        for idx, row in inventory_df.iterrows():
                                            sku = row['sku']
                                            warehouse = row['warehouse']
                                            
                                            # Check if this SKU+warehouse combination has staged orders
                                            staged_key = (sku, warehouse)
                                            if staged_key in staged_qty_dict:
                                                # Subtract staged quantity from inventory
                                                current_qty = inventory_df.loc[idx, 'quantity']
                                                adjusted_qty = max(0, current_qty - staged_qty_dict[staged_key])
                                                inventory_df.loc[idx, 'quantity'] = adjusted_qty
                        
                        # Get only unstaged orders for recalculation
                        orders_to_process = st.session_state.orders_df.copy()
                        if 'staged' in st.session_state.processed_orders.columns:
                            # Map staged status from processed orders to original orders based on order number and line item
                            # This requires joining the dataframes to transfer the 'staged' flag
                            if 'ordernumber' in orders_to_process.columns and 'ordernumber' in st.session_state.processed_orders.columns:
                                # Create a unique identifier for each line item (order number + sku)
                                orders_to_process['order_line_key'] = orders_to_process['ordernumber'] + "_" + orders_to_process['sku'].astype(str)
                                
                                # Create the same key in processed orders
                                processed_with_key = st.session_state.processed_orders.copy()
                                processed_with_key['order_line_key'] = processed_with_key['ordernumber'] + "_" + processed_with_key['sku'].astype(str)
                                
                                # Create a mapping of line items that are staged
                                staged_keys = set(processed_with_key[processed_with_key['staged'] == True]['order_line_key'])
                                
                                # Filter out staged orders from orders_to_process
                                orders_to_process = orders_to_process[~orders_to_process['order_line_key'].isin(staged_keys)]
                                
                                # Remove temporary column
                                orders_to_process = orders_to_process.drop(columns=['order_line_key'])
                        
                        # Process orders with new SKU mappings using adjusted inventory
                        result = data_processor.process_orders(
                            orders_to_process,  # Only process unstaged orders
                            inventory_df,       # Use inventory adjusted for staged orders
                            st.session_state.shipping_zones_df,
                            edited_df
                        )
                        
                        # Get the processed orders result
                        new_processed_orders = result['orders']
                        
                        # If we have staged orders, we need to preserve them
                        if 'staged' in st.session_state.processed_orders.columns:
                            # Get existing staged orders
                            staged_orders = st.session_state.processed_orders[st.session_state.processed_orders['staged'] == True].copy()
                            
                            # Add 'staged' column to new processed orders if it doesn't exist
                            if 'staged' not in new_processed_orders.columns:
                                new_processed_orders['staged'] = False
                                
                            # Combine staged orders with new processed orders
                            if not staged_orders.empty:
                                st.session_state.processed_orders = pd.concat([staged_orders, new_processed_orders])
                                st.success(f"✅ SKU mapping applied to {len(new_processed_orders)} orders in processing while preserving {len(staged_orders)} staged orders")
                            else:
                                st.session_state.processed_orders = new_processed_orders
                                st.success(f"✅ SKU mapping applied to all {len(new_processed_orders)} orders in processing")
                        else:
                            # No staged orders, just update all processed orders
                            st.session_state.processed_orders = new_processed_orders
                            st.success(f"✅ SKU mapping applied to all {len(new_processed_orders)} orders in processing")
                        
                        # Update other session state variables
                        st.session_state.inventory_summary = result['inventory_summary']
                        st.session_state.shortage_summary = result['shortage_summary']
                        st.session_state.grouped_shortage_summary = result['grouped_shortage_summary']
                        
                        # Add inventory_comparison for orders minus orders in processing
                        if 'inventory_comparison' in result:
                            st.session_state.inventory_comparison = result['inventory_comparison']
                            # Save the comparison to CSV for reference
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            comparison_filename = f"inventory_comparison_{timestamp}.csv"
                            result['inventory_comparison'].to_csv(comparison_filename, index=False)
                            st.success(f"📊 Inventory comparison saved to {comparison_filename}")
                        else:
                            st.warning("⚠️ Inventory comparison data not available")
                        
                        # Rerun to update the UI
                        st.rerun()
