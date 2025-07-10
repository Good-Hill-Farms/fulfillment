import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.google_sheets import (
    load_agg_orders,
    load_oxnard_inventory,
    load_wheeling_inventory,
    load_pieces_vs_lb_conversion,
    load_all_picklist_v2,
    load_orders_new  # Add new import
)
from utils.scripts_shopify.shopify_orders_report import update_orders_data

st.set_page_config(
    page_title="Fruit Dashboard",
    page_icon="🍎",
    layout="wide"
)

def format_currency(value):
    """Format a number as currency with thousand separators"""
    return f"${value:,.2f}"

def format_number(value):
    """Format a number with thousand separators"""
    return f"{value:,}"

def main():
    st.title("🍎 Fruit Dashboard")
    
    st.markdown("""
    Welcome to the Fruit Dashboard! Use the filters in the sidebar to analyze fruit inventory and orders.
    """)
    
    # Load all data
    if 'agg_orders_df' not in st.session_state:
        with st.spinner("Loading fruit data..."):
            df_orders = load_agg_orders()
            if df_orders is not None:
                # Convert order columns to numeric when loading data
                if 'Oxnard Actual Order' in df_orders.columns:
                    df_orders['Oxnard Actual Order'] = pd.to_numeric(df_orders['Oxnard Actual Order'], errors='coerce').fillna(0)
                if 'Wheeling Actual Order' in df_orders.columns:
                    df_orders['Wheeling Actual Order'] = pd.to_numeric(df_orders['Wheeling Actual Order'], errors='coerce').fillna(0)
                st.session_state['agg_orders_df'] = df_orders
                st.success("✅ Fruit data loaded successfully!")
            else:
                st.error("❌ Failed to load fruit data")
                return

    # Load Orders_new data
    if 'orders_new_df' not in st.session_state:
        with st.spinner("Loading orders history data..."):
            orders_new_df = load_orders_new()
            if orders_new_df is not None:
                st.session_state['orders_new_df'] = orders_new_df
                st.success("✅ Orders history loaded successfully!")
            else:
                st.warning("⚠️ Failed to load orders history")

    # Load inventory and picklist data
    if 'inventory_data' not in st.session_state:
        with st.spinner("Loading inventory and picklist data..."):
            oxnard_df = load_oxnard_inventory()
            wheeling_df = load_wheeling_inventory()
            pieces_vs_lb_df = load_pieces_vs_lb_conversion()
            picklist_df = load_all_picklist_v2()
            
            st.session_state['inventory_data'] = {
                'oxnard': oxnard_df,
                'wheeling': wheeling_df
            }
            
            st.session_state['reference_data'] = {
                'pieces_vs_lb': pieces_vs_lb_df
            }
            
            st.session_state['picklist_data'] = picklist_df
            
            if all(df is not None for df in [oxnard_df, wheeling_df]):
                st.success("✅ Inventory data loaded successfully!")
            else:
                st.warning("⚠️ Some inventory data could not be loaded")

    # Display Picklist Data
    with st.expander("📋 Projections Data", expanded=False):
        picklist_df = st.session_state.get('picklist_data')
        if picklist_df is not None and not picklist_df.empty:
            st.dataframe(
                picklist_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No picklist data available")

    # Display Inventory Data
    with st.expander("📦 Inventory Data", expanded=False):
        inventory_data = st.session_state.get('inventory_data', {})
            
        # Oxnard Inventory
        st.subheader("🏭 Oxnard Inventory")
        oxnard_df = inventory_data.get('oxnard')
        if oxnard_df is not None and not oxnard_df.empty:
            st.dataframe(
                oxnard_df.sort_values('INVENTORY DATE', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "INVENTORY DATE": st.column_config.DatetimeColumn(
                        "Inventory Date",
                        format="MMM DD, YYYY"
                    ),
                    "FRUIT DATE": st.column_config.DatetimeColumn(
                        "Fruit Date",
                        format="MMM DD, YYYY"
                    ),
                    "Total Weight": st.column_config.NumberColumn(
                        "Total Weight",
                        format="%.2f"
                    ),
                    "STATUS": st.column_config.TextColumn(
                        "Status",
                        help="Inventory status (Good/Bad)"
                    )
                }
            )
        else:
            st.warning("No Oxnard inventory data available")
            
        # Wheeling Inventory
        st.subheader("🏭 Wheeling Inventory")
        wheeling_df = inventory_data.get('wheeling')
        if wheeling_df is not None and not wheeling_df.empty:
            st.dataframe(
                wheeling_df.sort_values('INVENTORY DATE', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "INVENTORY DATE": st.column_config.DatetimeColumn(
                        "Inventory Date",
                        format="MMM DD, YYYY"
                    ),
                    "FRUIT DATE": st.column_config.DatetimeColumn(
                        "Fruit Date",
                        format="MMM DD, YYYY"
                    ),
                    "Total Weight": st.column_config.NumberColumn(
                        "Total Weight",
                        format="%.2f"
                    ),
                    "STATUS": st.column_config.TextColumn(
                        "Status",
                        help="Inventory status (Good/Bad)"
                    )
                }
            )
        else:
            st.warning("No Wheeling inventory data available")

    # Display Reference Data
    with st.expander("📚 Reference Data", expanded=False):
        reference_data = st.session_state.get('reference_data', {})
        
        # Pieces vs Lb Conversion
        st.subheader("🔄 Pieces vs Lb Conversion")
        pieces_vs_lb_df = reference_data.get('pieces_vs_lb')
        if pieces_vs_lb_df is not None and not pieces_vs_lb_df.empty:
            st.dataframe(
                pieces_vs_lb_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No pieces vs lb conversion data available")

    df = st.session_state.get('agg_orders_df')
    
    if df is None:
        st.warning("No fruit data available.")
        return

    # Display Combined Warehouse Orders in a collapsible section
    with st.expander("🏭 Warehouse Fruit Orders (from vendors)", expanded=False):
        # --- Oxnard Section ---
        st.subheader("Oxnard Fruit Orders")
        
        # Prepare Oxnard data
        oxnard_cols = [
            'Date_1', 'Vendor', 'Fruit', 'Oxnard Picklist SKU', 'Oxnard Status', 'Oxnard Notes',
            'Oxnard Weight Needed', 'Oxnard Order', 'Oxnard Actual Order', 'Weight Per Pick', 'Oxnard Batchcode'
        ]
        oxnard_cols = [col for col in oxnard_cols if col in df.columns]
        
        # Filter Oxnard data for essential fields
        df_oxnard = df[
            (df['Date_1'].notna()) & 
            (df['Date_1'] != '') & 
            (df['Vendor'].notna()) & 
            (df['Vendor'] != '') & 
            (df['Oxnard Status'].notna())
        ][oxnard_cols].copy()
        
        # Clean numeric columns
        numeric_cols = ['Oxnard Weight Needed', 'Oxnard Order', 'Oxnard Actual Order', 'Weight Per Pick']
        for col in numeric_cols:
            if col in df_oxnard.columns:
                df_oxnard[col] = pd.to_numeric(df_oxnard[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(0)
        
        # Filter by weight > 0
        if 'Oxnard Weight Needed' in df_oxnard.columns:
            df_oxnard = df_oxnard[df_oxnard['Oxnard Weight Needed'] > 0]
        
        # Rename columns for display
        oxnard_rename = {
            'Date_1': 'Date',
            'Oxnard Picklist SKU': 'SKU',
            'Oxnard Status': 'Status',
            'Oxnard Notes': 'Notes',
            'Oxnard Weight Needed': 'Weight Needed (LBS)',
            'Oxnard Order': 'Order Qty (Units)',
            'Oxnard Actual Order': 'Actual Order (Units)',
            'Weight Per Pick': 'Weight/Pick (LBS)',
            'Oxnard Batchcode': 'Batchcode'
        }
        df_oxnard = df_oxnard.rename(columns=oxnard_rename)
        
        # Sort by date
        df_oxnard['Date'] = pd.to_datetime(df_oxnard['Date'])
        df_oxnard = df_oxnard.sort_values('Date', ascending=False)
        
        # Display Oxnard table
        if not df_oxnard.empty:
            st.dataframe(
                df_oxnard,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Date': st.column_config.DatetimeColumn(
                        'Date',
                        format="MMM DD, YYYY"
                    ),
                    'Weight Needed (LBS)': st.column_config.NumberColumn(format="%d"),
                    'Order Qty (Units)': st.column_config.NumberColumn(format="%d"),
                    'Actual Order (Units)': st.column_config.NumberColumn(format="%d"),
                    'Weight/Pick (LBS)': st.column_config.NumberColumn(format="%d")
                }
            )
        else:
            st.warning("No Oxnard orders found!")

        # --- Wheeling Section ---
        st.subheader("Wheeling Fruit Orders")
        
        # Prepare Wheeling data
        wheeling_cols = [
            'Date_1', 'Vendor', 'Fruit', 'Wheeling Picklist SKU', 'Wheeling Status', 'Wheeling Notes',
            'Wheeling Weight Needed', 'Wheeling Order', 'Wheeling Actual Order', 'Wheeling Weight Per Pick', 'Wheeling Batchcode'
        ]
        wheeling_cols = [col for col in wheeling_cols if col in df.columns]
        
        # Filter Wheeling data for essential fields
        df_wheeling = df[
            (df['Date_1'].notna()) & 
            (df['Date_1'] != '') & 
            (df['Vendor'].notna()) & 
            (df['Vendor'] != '') & 
            (df['Wheeling Status'].notna())
        ][wheeling_cols].copy()
        
        # Clean numeric columns
        numeric_cols = ['Wheeling Weight Needed', 'Wheeling Order', 'Wheeling Actual Order', 'Wheeling Weight Per Pick']
        for col in numeric_cols:
            if col in df_wheeling.columns:
                df_wheeling[col] = pd.to_numeric(df_wheeling[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(0)
        
        # Filter by weight > 0
        if 'Wheeling Weight Needed' in df_wheeling.columns:
            df_wheeling = df_wheeling[df_wheeling['Wheeling Weight Needed'] > 0]
        
        # Rename columns for display
        wheeling_rename = {
            'Date_1': 'Date',
            'Wheeling Picklist SKU': 'SKU',
            'Wheeling Status': 'Status',
            'Wheeling Notes': 'Notes',
            'Wheeling Weight Needed': 'Weight Needed (LBS)',
            'Wheeling Order': 'Order Qty (Units)',
            'Wheeling Actual Order': 'Actual Order (Units)',
            'Wheeling Weight Per Pick': 'Weight/Pick (LBS)',
            'Wheeling Batchcode': 'Batchcode'
        }
        df_wheeling = df_wheeling.rename(columns=wheeling_rename)
        
        # Sort by date
        df_wheeling['Date'] = pd.to_datetime(df_wheeling['Date'])
        df_wheeling = df_wheeling.sort_values('Date', ascending=False)
        
        # Display Wheeling table
        if not df_wheeling.empty:
            st.dataframe(
                df_wheeling,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Date': st.column_config.DatetimeColumn(
                        'Date',
                        format="MMM DD, YYYY"
                    ),
                    'Weight Needed (LBS)': st.column_config.NumberColumn(format="%d"),
                    'Order Qty (Units)': st.column_config.NumberColumn(format="%d"),
                    'Actual Order (Units)': st.column_config.NumberColumn(format="%d"),
                    'Weight/Pick (LBS)': st.column_config.NumberColumn(format="%d")
                }
            )
        else:
            st.warning("No Wheeling orders found!")

    # Initialize filtered dataframe
    df_filtered = df.copy()
    
    # Optional Filters
    st.sidebar.title("Optional Filters")
    
    # Add a checkbox to enable/disable filtering
    enable_filters = st.sidebar.checkbox("Enable Filters", value=False)
    
    if enable_filters:
        # 1. Fruit Filter
        st.sidebar.subheader("🍊 Fruit")
        available_fruits = sorted(df['Fruit'].unique())
        selected_fruits = st.sidebar.multiselect(
            "Select Fruits",
            available_fruits,
            default=None,
            placeholder="Choose fruits..."
        )
        if selected_fruits:
            df_filtered = df_filtered[df_filtered['Fruit'].isin(selected_fruits)]

        # 2. Warehouse Filter
        st.sidebar.subheader("🏭 Warehouse")
        warehouse_options = ["Oxnard", "Wheeling"]
        selected_warehouse = st.sidebar.radio(
            "Select Warehouse",
            warehouse_options,
            key="warehouse_selection"
        )

        # 8. Order Status Filter (based on selected warehouse)
        st.sidebar.subheader("📋 Order Status")
        status_column = f'{selected_warehouse} Status'
        
        if status_column in df_filtered.columns:
            # Replace empty status with "❌ No Status" label
            df_filtered[status_column] = df_filtered[status_column].replace(['', np.nan], '❌ No Status')
            status_options = sorted(df_filtered[status_column].unique())
            selected_status = st.sidebar.multiselect(
                "Select Order Status",
                status_options,
                default=None,
                placeholder=f"Choose {selected_warehouse} status..."
            )
            if selected_status:
                df_filtered = df_filtered[df_filtered[status_column].isin(selected_status)]
                
            # Filter by actual orders in selected warehouse
            order_column = f'{selected_warehouse} Actual Order'
            if order_column in df_filtered.columns:
                df_filtered[order_column] = pd.to_numeric(df_filtered[order_column], errors='coerce').fillna(0)
                df_filtered = df_filtered[df_filtered[order_column] > 0]
        else:
            st.sidebar.warning(f"No status information available for {selected_warehouse}")
            
        # 3. Fruit Status Filter
        st.sidebar.subheader("🌱 Fruit Status")
        if 'Fruit Status' in df_filtered.columns:
            status_options = sorted(df_filtered['Fruit Status'].unique())
        else:
            status_options = ["Seasonal", "Abundant"]  # Default options if column doesn't exist
        selected_status = st.sidebar.multiselect(
            "Select Fruit Status",
            status_options,
            default=None,
            placeholder="Choose status..."
        )
        if selected_status and 'Fruit Status' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Fruit Status'].isin(selected_status)]

        # 4. SKU Filter
        st.sidebar.subheader("📦 SKU")
        all_skus = set()
        if 'Oxnard Picklist SKU' in df_filtered.columns:
            all_skus.update(df_filtered['Oxnard Picklist SKU'].dropna().unique())
        if 'Wheeling Picklist SKU' in df_filtered.columns:
            all_skus.update(df_filtered['Wheeling Picklist SKU'].dropna().unique())
        selected_skus = st.sidebar.multiselect(
            "Select SKUs",
            sorted(all_skus),
            default=None,
            placeholder="Choose SKUs..."
        )
        if selected_skus:
            sku_mask = df_filtered['Oxnard Picklist SKU'].isin(selected_skus) | \
                       df_filtered['Wheeling Picklist SKU'].isin(selected_skus)
            df_filtered = df_filtered[sku_mask]

        # 5. Date Period Filter
        st.sidebar.subheader("📅 Date Period")
        if 'Date_1' in df_filtered.columns:
            df_filtered['Date_1'] = pd.to_datetime(df_filtered['Date_1'], errors='coerce')
            df_filtered = df_filtered.dropna(subset=['Date_1'])
            
            if not df_filtered.empty:
                min_date = df_filtered['Date_1'].min()
                max_date = df_filtered['Date_1'].max()
                
                col1, col2 = st.sidebar.columns(2)
                
                # Period Type Selection
                period_type = col1.radio(
                    "Period Type",
                    ["Historical", "Measure"],
                    key="period_type"
                )
                
                # Date Range Selection
                date_range = col2.date_input(
                    "Select Date Range",
                    value=(max_date - timedelta(days=30), max_date),
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1, seconds=-1)
                    df_filtered = df_filtered[
                        (df_filtered['Date_1'] >= start_date) & 
                        (df_filtered['Date_1'] <= end_date)
                    ]

        # 6. Projection Period Filter
        st.sidebar.subheader("📊 Projection Period")
        if 'Projection Period' in df_filtered.columns:
            projection_periods = sorted(df_filtered['Projection Period'].astype(str).unique())
            selected_periods = st.sidebar.multiselect(
                "Select Projection Periods",
                projection_periods,
                default=None,
                placeholder="Choose projection periods..."
            )
            if selected_periods:
                df_filtered = df_filtered[df_filtered['Projection Period'].isin(selected_periods)]

        # 7. Inventory Status Filter
        st.sidebar.subheader("📈 Inventory Status")
        inventory_status_options = ["Good", "Bad"]
        selected_inventory_status = st.sidebar.multiselect(
            "Select Inventory Status",
            inventory_status_options,
            default=None,
            placeholder="Choose inventory status..."
        )
        if selected_inventory_status and 'Inventory Status' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Inventory Status'].isin(selected_inventory_status)]

        # Display filtered data summary
        st.subheader("Filtered Data Summary")
        st.write(f"Number of records after filtering: {len(df_filtered):,}")
    
    # Display Shopify Orders Data in a collapsible section
    with st.expander("📦 View Shopify Orders", expanded=False):
        # Date range selector for Shopify orders
        st.subheader("📅 Select Date Range")
        date_col1, date_col2, refresh_col = st.columns([2, 2, 1])
        
        # Calculate default date range: from last Monday to now
        default_end_date = datetime.now()
        # Get the most recent Monday (0 = Monday, 1 = Tuesday, etc.)
        days_since_monday = default_end_date.weekday()
        default_start_date = default_end_date - timedelta(days=days_since_monday)
        default_start_date = datetime.combine(default_start_date.date(), datetime.min.time())
        
        with date_col1:
            shopify_start_date = st.date_input(
                "Start Date",
                value=default_start_date,
                max_value=datetime.now()
            )
        
        with date_col2:
            shopify_end_date = st.date_input(
                "End Date",
                value=default_end_date,
                max_value=datetime.now()
            )
        
        with refresh_col:
            st.write("")  # Add spacing
            st.write("")  # Add spacing
            if st.button("🔄 Refresh"):
                with st.spinner("Fetching orders data..."):
                    try:
                        orders_df = update_orders_data(
                            start_date=datetime.combine(shopify_start_date, datetime.min.time()),
                            end_date=datetime.combine(shopify_end_date, datetime.max.time())
                        )
                        if orders_df is not None:
                            st.session_state['shopify_orders_df'] = orders_df
                            st.success("✅ Orders data refreshed!")
                        else:
                            st.error("❌ Failed to fetch orders data")
                    except Exception as e:
                        st.error(f"❌ Error fetching orders: {str(e)}")
        
        # Load or refresh Shopify orders data
        if 'shopify_orders_df' not in st.session_state:
            with st.spinner("Loading initial orders data..."):
                try:
                    orders_df = update_orders_data(
                        start_date=datetime.combine(shopify_start_date, datetime.min.time()),
                        end_date=datetime.combine(shopify_end_date, datetime.max.time())
                    )
                    if orders_df is not None:
                        st.session_state['shopify_orders_df'] = orders_df
                    else:
                        st.error("❌ Failed to load orders data")
                except Exception as e:
                    st.error(f"❌ Error loading orders: {str(e)}")
        
        if 'shopify_orders_df' in st.session_state:
            orders_df = st.session_state['shopify_orders_df']
            
            # Filter data based on selected date range
            orders_df['Created At'] = pd.to_datetime(orders_df['Created At'])
            mask = (orders_df['Created At'].dt.date >= shopify_start_date) & (orders_df['Created At'].dt.date <= shopify_end_date)
            filtered_df = orders_df[mask]
            
            # Calculate metrics
            total_orders = len(filtered_df['Order ID'].unique())
            total_revenue = filtered_df['Total'].sum()
            avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
            total_items = filtered_df['Quantity'].sum()
            
            # Display metrics in a nice grid with larger numbers
            st.subheader("📊 Order Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    "Total Orders",
                    format_number(total_orders),
                    help="Total number of unique orders in the selected period"
                )
            
            with metric_col2:
                st.metric(
                    "Total Revenue",
                    format_currency(total_revenue),
                    help="Total revenue from all orders in the selected period"
                )
            
            with metric_col3:
                st.metric(
                    "Avg Order Value",
                    format_currency(avg_order_value),
                    help="Average revenue per order"
                )
            
            with metric_col4:
                st.metric(
                    "Total Items Sold",
                    format_number(total_items),
                    help="Total quantity of items sold"
                )
            
            # Display detailed orders table
            st.subheader("📋 Orders Details")
            st.dataframe(
                filtered_df.sort_values('Created At', ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Created At": st.column_config.DatetimeColumn(
                        "Created At",
                        format="D MMM YYYY, HH:mm"
                    ),
                    "Subtotal": st.column_config.NumberColumn(
                        "Subtotal",
                        format="$%.2f"
                    ),
                    "Shipping": st.column_config.NumberColumn(
                        "Shipping",
                        format="$%.2f"
                    ),
                    "Total": st.column_config.NumberColumn(
                        "Total",
                        format="$%.2f"
                    ),
                    "Discount Amount": st.column_config.NumberColumn(
                        "Discount Amount",
                        format="$%.2f"
                    ),
                    "Unit Price": st.column_config.NumberColumn(
                        "Unit Price",
                        format="$%.2f"
                    ),
                    "Delivery Date": st.column_config.TextColumn(
                        "Delivery Date",
                        help="Scheduled delivery date for the order"
                    ),
                    "Shipping Method": st.column_config.TextColumn(
                        "Shipping Method",
                        help="Method used to ship the order"
                    )
                }
            )
            
            # Add SKU Summary Table
            st.subheader("📊 SKU Summary")
            
            # Calculate summary statistics by SKU
            sku_summary = filtered_df.groupby('SKU').agg({
                'Quantity': 'sum',
                'Unit Price': 'first',  # Get the unit price (assuming it's constant per SKU)
            }).reset_index()
            
            # Calculate total revenue per SKU
            sku_summary['Total Revenue'] = sku_summary['Quantity'] * sku_summary['Unit Price']
            
            # Sort by total revenue descending
            sku_summary = sku_summary.sort_values('Total Revenue', ascending=False)
            
            # Display the summary table
            st.dataframe(
                sku_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "SKU": st.column_config.TextColumn(
                        "SKU",
                        help="Product SKU"
                    ),
                    "Quantity": st.column_config.NumberColumn(
                        "Total Quantity",
                        help="Total quantity sold",
                        format="%d"
                    ),
                    "Unit Price": st.column_config.NumberColumn(
                        "Unit Price",
                        help="Price per unit",
                        format="$%.2f"
                    ),
                    "Total Revenue": st.column_config.NumberColumn(
                        "Total Revenue",
                        help="Total revenue for this SKU",
                        format="$%.2f"
                    )
                }
            )
            
            # Add grand totals
            total_quantity = sku_summary['Quantity'].sum()
            total_revenue = sku_summary['Total Revenue'].sum()
            
            total_col1, total_col2 = st.columns(2)
            with total_col1:
                st.metric(
                    "Total Quantity (All SKUs)",
                    format_number(total_quantity),
                    help="Total quantity across all SKUs"
                )
            with total_col2:
                st.metric(
                    "Total Revenue (All SKUs)",
                    format_currency(total_revenue),
                    help="Total revenue across all SKUs"
                )

    # Display Orders History Data
    with st.expander("📜 Fruit Cost (from invoices)", expanded=False):
        orders_new_df = st.session_state.get('orders_new_df')
        if orders_new_df is not None and not orders_new_df.empty:
            # Get column names
            date_col = orders_new_df.columns[0]  # First column (invoice date)
            vendor_col = orders_new_df.columns[1]  # Second column (Aggregator / Vendor)
            product_col = orders_new_df.columns[2]  # Third column (Product Type)
            price_col = orders_new_df.columns[3]  # Fourth column (Price per lb)
            total_cost_col = orders_new_df.columns[4]  # Fifth column (Actual Total Cost)

            # Ensure date column is datetime
            orders_new_df[date_col] = pd.to_datetime(orders_new_df[date_col], errors='coerce')
            
            # Convert price columns to float without any string manipulation
            orders_new_df[price_col] = pd.to_numeric(orders_new_df[price_col], errors='coerce')
            orders_new_df[total_cost_col] = pd.to_numeric(orders_new_df[total_cost_col], errors='coerce')

            # Add filters for the orders history
            col1, col2 = st.columns(2)
            
            with col1:
                # Date range filter
                try:
                    min_date = orders_new_df[date_col].min()
                    max_date = orders_new_df[date_col].max()
                    if pd.notna(min_date) and pd.notna(max_date):
                        selected_dates = st.date_input(
                            "Select Date Range",
                            value=(min_date.date(), max_date.date()),
                            min_value=min_date.date(),
                            max_value=max_date.date()
                        )
                    else:
                        st.warning("No valid dates found in the data")
                        selected_dates = None
                except Exception as e:
                    st.error(f"Error processing dates: {str(e)}")
                    selected_dates = None

            with col2:
                # Vendor filter
                vendors = sorted(orders_new_df[vendor_col].dropna().unique())
                selected_vendors = st.multiselect(
                    "Select Vendors",
                    vendors,
                    default=None,
                    placeholder="Choose vendors..."
                )

            # Apply filters
            filtered_df = orders_new_df.copy()
            
            if selected_dates and len(selected_dates) == 2:
                start_date, end_date = selected_dates
                filtered_df = filtered_df[
                    (filtered_df[date_col].dt.date >= start_date) &
                    (filtered_df[date_col].dt.date <= end_date)
                ]
            
            if selected_vendors:
                filtered_df = filtered_df[filtered_df[vendor_col].isin(selected_vendors)]

            # Display the filtered data
            st.dataframe(
                filtered_df.sort_values(date_col, ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    date_col: st.column_config.DatetimeColumn(
                        "Invoice Date",
                        format="MMM DD, YYYY"
                    ),
                    vendor_col: "Vendor",
                    product_col: "Product Type",
                    price_col: st.column_config.NumberColumn(
                        "Price per lb",
                        format="$%.2f",
                        help="Price per pound"
                    ),
                    total_cost_col: st.column_config.NumberColumn(
                        "Total Cost",
                        format="$%.2f",
                        help="Total cost of the order"
                    )
                }
            )

            # Display summary metrics
            st.subheader("Summary Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_orders = len(filtered_df)
                st.metric("Total Orders", format_number(total_orders))
            
            with col2:
                unique_vendors = filtered_df[vendor_col].nunique()
                st.metric("Unique Vendors", format_number(unique_vendors))
            
            with col3:
                avg_price = filtered_df[price_col].mean()
                if pd.notna(avg_price):
                    st.metric("Average Price/lb", format_currency(avg_price))
                else:
                    st.metric("Average Price/lb", "N/A")

            with col4:
                total_cost = filtered_df[total_cost_col].sum()
                if pd.notna(total_cost):
                    st.metric("Total Cost", format_currency(total_cost))
                else:
                    st.metric("Total Cost", "N/A")

            # Display price trends by product type
            st.subheader("Price Trends by Product Type")
            # Remove rows with NaN prices before calculating trends
            price_df = filtered_df.dropna(subset=[price_col, product_col])
            if not price_df.empty:
                price_trends = price_df.groupby([product_col]).agg({
                    price_col: ['mean', 'min', 'max', 'count'],
                    total_cost_col: 'sum'
                }).round(2)
                
                price_trends.columns = ['Average Price', 'Min Price', 'Max Price', 'Number of Orders', 'Total Cost']
                price_trends = price_trends.reset_index()
                
                st.dataframe(
                    price_trends.sort_values('Average Price', ascending=False),
                    use_container_width=True,
                    column_config={
                        product_col: "Product Type",
                        "Average Price": st.column_config.NumberColumn(format="$%.2f"),
                        "Min Price": st.column_config.NumberColumn(format="$%.2f"),
                        "Max Price": st.column_config.NumberColumn(format="$%.2f"),
                        "Number of Orders": st.column_config.NumberColumn(format="%d"),
                        "Total Cost": st.column_config.NumberColumn(format="$%.2f")
                    }
                )
            else:
                st.warning("No valid price data available for trends")
        else:
            st.warning("No orders history data available")

if __name__ == "__main__":
    main() 