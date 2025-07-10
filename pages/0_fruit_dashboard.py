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
    load_orders_new,
    get_sku_info
)
from utils.scripts_shopify.shopify_orders_report import update_orders_data, update_unfulfilled_orders
from utils.inventory_api import get_inventory_data
import time
import os

st.set_page_config(
    page_title="Fruit Dashboard",
    page_icon="🍎",
    layout="wide"
)

def format_currency(value):
    """Format a number as currency with thousand separators"""
    return f"${value:,.2f}"

def format_number(value):
    """Format a number with thousand separators and round to 2 decimal places"""
    if isinstance(value, int):
        return f"{value:,}"
    elif isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{round(value, 2):,.2f}"
    return str(value)

def convert_duration_to_minutes(duration_str):
    """Convert duration string to minutes"""
    if not duration_str or pd.isna(duration_str):
        return None
    
    try:
        if 'min' in duration_str:
            return float(duration_str.split(' min')[0])
        elif 'hours' in duration_str:
            return float(duration_str.split(' hours')[0]) * 60
        elif 'days' in duration_str:
            return float(duration_str.split(' days')[0]) * 24 * 60
        return None
    except:
        return None

def format_duration(minutes):
    """Format minutes into a readable duration string"""
    if minutes is None or pd.isna(minutes):
        return ""
    
    if minutes < 60:
        return f"{int(minutes)} min"
    elif minutes < 24 * 60:
        hours = minutes / 60
        return f"{int(hours)} hours"
    else:
        days = minutes / (24 * 60)
        return f"{int(days)} days"

def get_week_range(date=None, previous=False):
    """Get the Monday-Sunday range for a given date or previous week."""
    if date is None:
        date = datetime.now()
    
    # If previous week requested, subtract 7 days from the current date
    if previous:
        date = date - timedelta(days=7)
    
    # Calculate the Monday of the week (weekday() returns 0 for Monday)
    monday = date - timedelta(days=date.weekday())
    monday = datetime.combine(monday.date(), datetime.min.time())
    
    # Calculate the Sunday of the week
    sunday = monday + timedelta(days=6)
    sunday = datetime.combine(sunday.date(), datetime.max.time())
    
    return monday, sunday

def get_safe_date(min_date, max_date, default_date=None):
    """Get a date that's guaranteed to be within the min-max range."""
    if default_date is None:
        default_date = datetime.now()
    
    default_date = pd.to_datetime(default_date)
    min_date = pd.to_datetime(min_date)
    max_date = pd.to_datetime(max_date)
    
    if default_date < min_date:
        return min_date
    elif default_date > max_date:
        return max_date
    return default_date

def check_data_availability(df, date_col, start_date, end_date):
    """Check if there is data available for the selected date range."""
    if df is None or df.empty:
        return False
    
    filtered_df = df[
        (df[date_col] >= start_date) &
        (df[date_col] <= end_date)
    ]
    return not filtered_df.empty

def load_shopify_orders(start_date=None, end_date=None, force_refresh=False):
    """Load Shopify orders data and store in session state"""
    if start_date is None:
        # Default to last Monday
        end_date = datetime.now()
        days_since_monday = end_date.weekday()
        start_date = end_date - timedelta(days=days_since_monday)
        start_date = datetime.combine(start_date.date(), datetime.min.time())
        end_date = datetime.combine(end_date.date(), datetime.max.time())
    
    # Check if we need to refresh the data
    if force_refresh or 'shopify_orders_df' not in st.session_state:
        with st.spinner("Loading orders data..."):
            try:
                orders_df = update_orders_data(start_date=start_date, end_date=end_date)
                unfulfilled_df = update_unfulfilled_orders(start_date=start_date, end_date=end_date)
                
                if orders_df is not None:
                    st.session_state['shopify_orders_df'] = orders_df
                    st.session_state['shopify_orders_start_date'] = start_date
                    st.session_state['shopify_orders_end_date'] = end_date
                
                if unfulfilled_df is not None:
                    st.session_state['unfulfilled_orders_df'] = unfulfilled_df
                
                return orders_df, unfulfilled_df
            except Exception as e:
                st.error(f"Error loading orders: {str(e)}")
                return None, None
    
    # Return cached data if dates match
    elif (st.session_state.get('shopify_orders_start_date') == start_date and 
          st.session_state.get('shopify_orders_end_date') == end_date):
        return (st.session_state.get('shopify_orders_df'), 
                st.session_state.get('unfulfilled_orders_df'))
    
    # Dates changed, need to refresh
    else:
        return load_shopify_orders(start_date, end_date, force_refresh=True)

def main():
    st.title("🍎 Fruit Dashboard")
    
    st.markdown("""
    Welcome to the Fruit Dashboard! Use the filters in the sidebar to analyze fruit inventory and orders.
    """)
    
    # Load all data
    if 'agg_orders_df' not in st.session_state:
        message_placeholder = st.empty()
        with st.spinner("Loading fruit data..."):
            df_orders = load_agg_orders()
            if df_orders is not None:
                # Convert order columns to numeric when loading data
                if 'Oxnard Actual Order' in df_orders.columns:
                    df_orders['Oxnard Actual Order'] = pd.to_numeric(df_orders['Oxnard Actual Order'], errors='coerce').fillna(0)
                if 'Wheeling Actual Order' in df_orders.columns:
                    df_orders['Wheeling Actual Order'] = pd.to_numeric(df_orders['Wheeling Actual Order'], errors='coerce').fillna(0)
                st.session_state['agg_orders_df'] = df_orders
                message_placeholder.success("✅ Fruit data loaded successfully!")
                time.sleep(1)  # Show message for 1 second
                message_placeholder.empty()
            else:
                message_placeholder.error("❌ Failed to load fruit data")
                return

    # Load Orders_new data
    if 'orders_new_df' not in st.session_state:
        message_placeholder = st.empty()
        with st.spinner("Loading orders history data..."):
            orders_new_df = load_orders_new()
            if orders_new_df is not None:
                st.session_state['orders_new_df'] = orders_new_df
                message_placeholder.success("✅ Orders history loaded successfully!")
                time.sleep(1)  # Show message for 1 second
                message_placeholder.empty()
            else:
                message_placeholder.warning("⚠️ Failed to load orders history")

    # Load inventory and picklist data
    if 'inventory_data' not in st.session_state:
        message_placeholder = st.empty()
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
                message_placeholder.success("✅ Inventory data loaded successfully!")
                time.sleep(1)  # Show message for 1 second
                message_placeholder.empty()
            else:
                message_placeholder.warning("⚠️ Some inventory data could not be loaded")

    # Display Picklist Data
    with st.expander("📋 Current Projections Data", expanded=False):
        st.markdown("""
        ### Color Legend:
        - 🟢 **Green** background indicates *under-ordered* (negative needs)
        - 🟠 **Orange** background indicates *over-ordered* (positive needs)
        """)
        
        picklist_df = st.session_state.get('picklist_data')
        if picklist_df is not None and not picklist_df.empty:
            # Add product type filter
            product_types = sorted(picklist_df['Product Type'].unique())
            selected_product_types = st.multiselect(
                "Filter by Product Type",
                options=product_types,
                default=None,
                placeholder="Select product types..."
            )
            
            # Filter data based on selection
            display_df = picklist_df.copy()
            if selected_product_types:
                display_df = display_df[display_df['Product Type'].isin(selected_product_types)]
            
            # Convert all numeric columns to float, except 'Product Type'
            numeric_cols = display_df.columns.difference(['Product Type'])
            for col in numeric_cols:
                # First convert to string to handle any potential formatting
                display_df[col] = display_df[col].astype(str).str.replace(',', '')
                # Then convert to numeric
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0)
                # Round to 1 decimal place
                display_df[col] = display_df[col].round(1)
            
            # Apply styling to needs columns
            needs_cols = [col for col in display_df.columns if 'Needs' in col]
            
            def color_needs(val):
                try:
                    val = float(val)
                    if pd.isna(val):
                        return ''
                    if val < 0:
                        return 'background-color: #e8f5e9'  # Light green background
                    elif val > 0:
                        return 'background-color: #fff3e0'  # Light orange background
                except:
                    return ''
                return ''
            
            styled_df = display_df.style.map(color_needs, subset=needs_cols)
            
            # Create column configuration for all numeric columns
            column_config = {
                'Product Type': st.column_config.TextColumn('Product Type')
            }
            
            # Add number formatting for all numeric columns
            numeric_cols = display_df.columns.difference(['Product Type'])
            for col in numeric_cols:
                column_config[col] = st.column_config.NumberColumn(
                    col,
                    format="%.2g"  # Use general format which will remove unnecessary zeros
                )
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )
            
            # Add summary metrics
            if not display_df.empty:
                st.markdown("### Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    total_products = len(display_df)
                    st.metric("Total Products", format_number(total_products))
                
                with col2:
                    products_with_needs = len(display_df[display_df[needs_cols].abs().sum(axis=1) > 0])
                    st.metric("Products with Needs", format_number(products_with_needs))
        else:
            st.warning("No picklist data available")

    # Display Inventory Data
    with st.expander("📦 Inventory Hardcounts", expanded=False):
        inventory_data = st.session_state.get('inventory_data', {})
            
        # Oxnard Inventory
        st.subheader("🏭 Oxnard Inventory")
        oxnard_df = inventory_data.get('oxnard')
        if oxnard_df is not None and not oxnard_df.empty:
            # Ensure INVENTORY DATE is datetime
            oxnard_df['INVENTORY DATE'] = pd.to_datetime(oxnard_df['INVENTORY DATE'])
            
            # Get the last available date
            last_date = oxnard_df['INVENTORY DATE'].max()
            min_date = oxnard_df['INVENTORY DATE'].min()
            
            # Add date filter at the top
            date_col1, date_col2 = st.columns([1, 3])
            with date_col1:
                week_option = st.radio(
                    "Select Week",
                    ["Previous Week", "Current Week", "Custom Range"],
                    key="oxnard_week_option",
                    horizontal=True
                )
            
            # Initialize date variables
            start_date = None
            end_date = None

            with date_col2:
                if week_option == "Current Week":
                    current_start, current_end = get_week_range()
                    if check_data_availability(oxnard_df, 'INVENTORY DATE', current_start, current_end):
                        start_date, end_date = current_start, current_end
                    else:
                        st.warning("No data available for current week.")
                elif week_option == "Previous Week":
                    prev_start, prev_end = get_week_range(previous=True)
                    if check_data_availability(oxnard_df, 'INVENTORY DATE', prev_start, prev_end):
                        start_date, end_date = prev_start, prev_end
                    else:
                        st.warning("No data available for previous week.")
                else:
                    default_date = get_safe_date(min_date, last_date)
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(default_date.date() - timedelta(days=7), default_date.date()),
                        min_value=min_date.date(),
                        max_value=last_date.date(),
                        key="oxnard_custom_date"
                    )
                    if len(date_range) == 2:
                        start_date = datetime.combine(date_range[0], datetime.min.time())
                        end_date = datetime.combine(date_range[1], datetime.max.time())
            
            # Apply date filter only if we have valid dates
            if start_date and end_date:
                filtered_df = oxnard_df[
                    (oxnard_df['INVENTORY DATE'] >= start_date) &
                    (oxnard_df['INVENTORY DATE'] <= end_date)
                ]
                
                if not filtered_df.empty:
                    # Remove unnecessary columns
                    columns_to_display = [col for col in filtered_df.columns if col not in ['TRUE', 'test status', 'STATUS_1']]
                    display_df = filtered_df[columns_to_display]
                    
                    st.dataframe(
                        display_df.sort_values('INVENTORY DATE', ascending=False),
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

        # Wheeling Inventory
        st.subheader("🏭 Wheeling Inventory")
        wheeling_df = inventory_data.get('wheeling')
        if wheeling_df is not None and not wheeling_df.empty:
            # Ensure INVENTORY DATE is datetime
            wheeling_df['INVENTORY DATE'] = pd.to_datetime(wheeling_df['INVENTORY DATE'])
            
            # Get the last available date
            last_date = wheeling_df['INVENTORY DATE'].max()
            min_date = wheeling_df['INVENTORY DATE'].min()
            
            # Add date filter at the top
            date_col1, date_col2 = st.columns([1, 3])
            with date_col1:
                week_option = st.radio(
                    "Select Week",
                    ["Previous Week", "Current Week", "Custom Range"],
                    key="wheeling_week_option",
                    horizontal=True
                )
            
            # Initialize date variables for Wheeling
            start_date = None
            end_date = None

            with date_col2:
                if week_option == "Current Week":
                    current_start, current_end = get_week_range()
                    if check_data_availability(wheeling_df, 'INVENTORY DATE', current_start, current_end):
                        start_date, end_date = current_start, current_end
                    else:
                        st.warning("No data available for current week.")
                elif week_option == "Previous Week":
                    prev_start, prev_end = get_week_range(previous=True)
                    if check_data_availability(wheeling_df, 'INVENTORY DATE', prev_start, prev_end):
                        start_date, end_date = prev_start, prev_end
                    else:
                        st.warning("No data available for previous week.")
                else:
                    default_date = get_safe_date(min_date, last_date)
                    date_range = st.date_input(
                        "Select Date Range",
                        value=(default_date.date() - timedelta(days=7), default_date.date()),
                        min_value=min_date.date(),
                        max_value=last_date.date(),
                        key="wheeling_custom_date"
                    )
                    if len(date_range) == 2:
                        start_date = datetime.combine(date_range[0], datetime.min.time())
                        end_date = datetime.combine(date_range[1], datetime.max.time())
            
            # Apply date filter only if we have valid dates
            if start_date and end_date:
                filtered_df = wheeling_df[
                    (wheeling_df['INVENTORY DATE'] >= start_date) &
                    (wheeling_df['INVENTORY DATE'] <= end_date)
                ]
                
                if not filtered_df.empty:
                    # Remove unnecessary columns
                    columns_to_display = [col for col in filtered_df.columns if col not in ['TRUE', 'test status', 'STATUS_1']]
                    display_df = filtered_df[columns_to_display]
                    
                    st.dataframe(
                        display_df.sort_values('INVENTORY DATE', ascending=False),
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

    # Display ColdCart Inventory Data
    with st.expander("🧊 ColdCart Inventory", expanded=False):
        # Add a refresh button
        if st.button("🔄 Refresh ColdCart Data"):
            with st.spinner("Fetching latest ColdCart inventory data..."):
                try:
                    coldcart_df = get_inventory_data()
                    if coldcart_df is not None:
                        st.session_state['coldcart_inventory'] = coldcart_df
                        st.success("✅ ColdCart data fetched successfully!")
                    else:
                        st.error("❌ No data received from ColdCart API")
                        st.info("Please check if your API token is correctly set in the environment variables.")
                except Exception as e:
                    st.error(f"❌ Error fetching ColdCart data: {str(e)}")
                    st.info("Please check if your API token is correctly set in the environment variables.")

        # Use cached data if available, otherwise fetch new data
        if 'coldcart_inventory' not in st.session_state:
            with st.spinner("Fetching initial ColdCart inventory data..."):
                try:
                    coldcart_df = get_inventory_data()
                    if coldcart_df is not None:
                        st.session_state['coldcart_inventory'] = coldcart_df
                    else:
                        st.warning("No ColdCart inventory data available")
                except Exception as e:
                    st.error(f"Error fetching ColdCart data: {str(e)}")

        coldcart_df = st.session_state.get('coldcart_inventory')
        if coldcart_df is not None and not coldcart_df.empty:
            # Add filters in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # SKU filter
                skus = sorted(coldcart_df['Sku'].dropna().unique())
                selected_skus = st.multiselect(
                    "Filter by SKU",
                    skus,
                    default=None,
                    placeholder="Choose SKUs...",
                    key="coldcart_sku_filter"
                )
                if selected_skus:
                    coldcart_df = coldcart_df[coldcart_df['Sku'].isin(selected_skus)]

            with col2:
                # Product Type filter
                product_types = sorted(coldcart_df['Type'].dropna().unique())
                selected_types = st.multiselect(
                    "Filter by Product Type",
                    product_types,
                    default=None,
                    placeholder="Choose product types...",
                    key="coldcart_type_filter"
                )
                if selected_types:
                    coldcart_df = coldcart_df[coldcart_df['Type'].isin(selected_types)]

            with col3:
                # Warehouse filter (moved from below)
                warehouses = ['All'] + sorted(coldcart_df['WarehouseName'].unique().tolist())
                selected_warehouse = st.selectbox(
                    'Select Warehouse',
                    warehouses,
                    key="coldcart_warehouse_filter"
                )
                if selected_warehouse != 'All':
                    coldcart_df = coldcart_df[coldcart_df['WarehouseName'] == selected_warehouse]

            # Create tabs for different views
            tab1, tab2 = st.tabs(["Summary by SKU", "By Warehouse"])

            with tab1:
                st.subheader("Total Inventory by SKU")
                # Group by SKU and aggregate quantities
                summary_df = coldcart_df.groupby(['Sku', 'Name', 'Type']).agg({
                    'AvailableQty': 'sum',
                    'DaysOnHand': 'mean'
                }).reset_index()

                st.dataframe(
                    summary_df.sort_values('AvailableQty', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'Sku': st.column_config.TextColumn('SKU'),
                        'Name': st.column_config.TextColumn('Product Name'),
                        'Type': st.column_config.TextColumn('Product Type'),
                        'AvailableQty': st.column_config.NumberColumn(
                            'Available Quantity',
                            format="%d"
                        ),
                        'DaysOnHand': st.column_config.NumberColumn(
                            'Avg Days on Hand',
                            format="%.1f"
                        )
                    }
                )

            with tab2:
                st.subheader("Inventory by Warehouse")
                # Create pivot table for warehouse view
                pivot_df = coldcart_df.pivot_table(
                    index=['Sku', 'Name', 'Type'],
                    columns='WarehouseName',
                    values='AvailableQty',
                    fill_value=0
                ).reset_index()

                st.dataframe(
                    pivot_df.sort_values('Sku'),
                    use_container_width=True,
                    hide_index=True
                )

            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_skus = len(coldcart_df['Sku'].unique())
                st.metric("Total SKUs", format_number(total_skus))
            
            with col2:
                total_qty = coldcart_df['AvailableQty'].sum()
                st.metric("Total Quantity", format_number(int(total_qty)))
            
            with col3:
                avg_days = coldcart_df['DaysOnHand'].mean()
                st.metric("Avg Days on Hand", f"{avg_days:.1f}")
            
            with col4:
                warehouses = coldcart_df['WarehouseName'].nunique()
                st.metric("Warehouses", warehouses)

            # Display inventory by warehouse
            st.subheader("Detailed Inventory by Warehouse")
            
            # Display detailed table
            st.dataframe(
                coldcart_df.sort_values(['WarehouseName', 'Sku']),
                use_container_width=True,
                hide_index=True,
                column_config={
                    'WarehouseName': st.column_config.TextColumn('Warehouse'),
                    'Sku': st.column_config.TextColumn('SKU'),
                    'Name': st.column_config.TextColumn('Product Name'),
                    'Type': st.column_config.TextColumn('Product Type'),
                    'AvailableQty': st.column_config.NumberColumn(
                        'Available Quantity',
                        format="%d"
                    ),
                    'DaysOnHand': st.column_config.NumberColumn(
                        'Days on Hand',
                        format="%.1f"
                    ),
                    'BatchCode': st.column_config.TextColumn('Batch Code')
                }
            )
        else:
            st.warning("No ColdCart inventory data available")

    # Display Reference Data
    with st.expander("📚 Pieces vs Lb Conversion", expanded=False):
        reference_data = st.session_state.get('reference_data', {})
        
        # Pieces vs Lb Conversion
        pieces_vs_lb_df = reference_data.get('pieces_vs_lb')
        if pieces_vs_lb_df is not None and not pieces_vs_lb_df.empty:
            st.dataframe(pieces_vs_lb_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No pieces vs lb conversion data available")

    # Display Orders with Fulfillment Status
    with st.expander("📦 Orders by Fulfillment Status", expanded=False):
        st.subheader("🛍️ Recent Orders")
        
        # Add date range controls
        col1, col2, refresh_col = st.columns([2, 2, 1])
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=datetime.now().weekday()),  # Last Monday
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        with refresh_col:
            st.write("")  # Add spacing
            st.write("")  # Add spacing
            refresh = st.button("🔄 Refresh")
            
        if start_date > end_date:
            st.error("Error: Start date must be before end date")
            return
            
        # Get orders data directly from Shopify API with selected date range
        orders_df, _ = load_shopify_orders(
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time()),
            force_refresh=refresh
        )
        
        if orders_df is not None and not orders_df.empty:
            # Get all unique statuses from the data
            all_statuses = sorted(orders_df['Fulfillment Status'].unique())
            
            # Add fulfillment status filter
            status_filter = st.multiselect(
                "Filter by Fulfillment Status",
                options=all_statuses,
                default=None,  # Show all by default
                help="Select one or more fulfillment statuses to display"
            )
            
            if status_filter:
                filtered_df = orders_df[orders_df['Fulfillment Status'].isin(status_filter)]
            else:
                filtered_df = orders_df
            
            if not filtered_df.empty:
                # Show summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Orders", len(filtered_df))
                with col2:
                    total_value = filtered_df['Total'].sum()
                    st.metric("Total Value", f"${total_value:,.2f}")
                with col3:
                    # Convert duration strings to minutes for calculation
                    duration_minutes = filtered_df['Fulfillment Duration'].apply(convert_duration_to_minutes)
                    avg_minutes = duration_minutes[duration_minutes.notna()].mean()
                    if not pd.isna(avg_minutes):
                        st.metric("Avg Fulfillment Time", format_duration(avg_minutes))
                with col4:
                    # Count orders by status
                    status_counts = filtered_df['Fulfillment Status'].value_counts()
                    status_text = ", ".join([f"{k}: {v}" for k, v in status_counts.items()])
                    st.metric("Status Breakdown", status_text)

                # Add Top SKUs Summary
                st.subheader("📊 Top SKUs")
                
                # Calculate SKU summary
                sku_summary = filtered_df.groupby('SKU').agg({
                    'Quantity': 'sum',
                    'Total': 'sum'
                }).reset_index()
                
                # Sort by quantity and get top 10
                top_skus_qty = sku_summary.nlargest(10, 'Quantity')
                
                # Sort by total value and get top 10
                top_skus_value = sku_summary.nlargest(10, 'Total')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top SKUs by Quantity**")
                    st.dataframe(
                        top_skus_qty,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "SKU": "SKU",
                            "Quantity": st.column_config.NumberColumn(
                                "Total Quantity",
                                format="%d"
                            ),
                            "Total": st.column_config.NumberColumn(
                                "Total Value",
                                format="$%.2f"
                            )
                        }
                    )
                
                with col2:
                    st.markdown("**Top SKUs by Value**")
                    st.dataframe(
                        top_skus_value,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "SKU": "SKU",
                            "Quantity": st.column_config.NumberColumn(
                                "Total Quantity",
                                format="%d"
                            ),
                            "Total": st.column_config.NumberColumn(
                                "Total Value",
                                format="$%.2f"
                            )
                        }
                    )

                # Display full orders table
                st.subheader("📋 All Orders")
                st.dataframe(
                    filtered_df.sort_values('Created At', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Created At": st.column_config.DatetimeColumn(
                            "Order Date",
                            format="MMM DD, YYYY HH:mm"
                        ),
                        "Fulfilled At": st.column_config.DatetimeColumn(
                            "Fulfilled Date",
                            format="MMM DD, YYYY HH:mm"
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
                        "Fulfillment Status": st.column_config.TextColumn(
                            "Status",
                            help="Current fulfillment status of the order"
                        ),
                        "Delivery Date": st.column_config.TextColumn(
                            "Delivery Date",
                            help="Scheduled delivery date"
                        ),
                        "Shipping Method": st.column_config.TextColumn(
                            "Shipping Method",
                            help="Method used to ship the order"
                        ),
                        "Customer Type": st.column_config.TextColumn(
                            "Customer Type",
                            help="New or Returning customer"
                        ),
                        "Tags": st.column_config.TextColumn(
                            "Tags",
                            help="Order tags"
                        )
                    }
                )
            else:
                st.warning("No orders found with selected fulfillment status")
        else:
            st.warning("No orders data available")

    df = st.session_state.get('agg_orders_df')
    
    if df is None:
        st.warning("No fruit data available.")
        return

    # Display Combined Warehouse Orders in a collapsible section
    with st.expander("🏭 Warehouse Fruit Orders (from vendors)", expanded=False):
        # --- Oxnard Section ---
        st.subheader("Oxnard Fruit Orders")
        
        # Add filters in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Status filter for Oxnard Fruit Orders
            if 'Oxnard Status' in df.columns:
                # Replace empty status with "❌ No Status" label
                df['Oxnard Status'] = df['Oxnard Status'].replace(['', np.nan], '❌ No Status')
                oxnard_order_statuses = sorted(df['Oxnard Status'].unique())
                selected_oxnard_order_statuses = st.multiselect(
                    "Filter by Status",
                    oxnard_order_statuses,
                    default=oxnard_order_statuses,
                    key="oxnard_order_status_filter"
                )

        with col2:
            # Product Type filter
            if 'Fruit' in df.columns:
                fruits = sorted(df['Fruit'].dropna().unique())
                selected_fruits = st.multiselect(
                    "Filter by Product Type",
                    fruits,
                    default=None,
                    placeholder="Choose product types...",
                    key="oxnard_order_product_filter"
                )

        with col3:
            # SKU filter
            if 'Oxnard Picklist SKU' in df.columns:
                skus = sorted(df['Oxnard Picklist SKU'].dropna().unique())
                selected_skus = st.multiselect(
                    "Filter by SKU",
                    skus,
                    default=None,
                    placeholder="Choose SKUs...",
                    key="oxnard_order_sku_filter"
                )

        with col4:
            # Vendor filter
            if 'Vendor' in df.columns:
                vendors = sorted(df['Vendor'].dropna().unique())
                selected_vendors = st.multiselect(
                    "Filter by Vendor",
                    vendors,
                    default=None,
                    placeholder="Choose vendors...",
                    key="oxnard_order_vendor_filter"
                )
        
        # Prepare Oxnard data
        oxnard_cols = [
            'Date_1', 'Vendor', 'Fruit', 'Oxnard Picklist SKU', 'Oxnard Status', 'Oxnard Notes',
            'Oxnard Weight Needed', 'Oxnard Order', 'Oxnard Actual Order', 'Weight Per Pick', 'Oxnard Batchcode'
        ]
        oxnard_cols = [col for col in oxnard_cols if col in df.columns]
        
        # Filter Oxnard data for essential fields and selected filters
        df_oxnard = df[
            (df['Date_1'].notna()) & 
            (df['Date_1'] != '') & 
            (df['Vendor'].notna()) & 
            (df['Vendor'] != '') & 
            (df['Oxnard Status'].notna()) &
            (df['Oxnard Status'].isin(selected_oxnard_order_statuses))
        ][oxnard_cols].copy()

        # Apply additional filters
        if selected_fruits:
            df_oxnard = df_oxnard[df_oxnard['Fruit'].isin(selected_fruits)]
        if selected_skus:
            df_oxnard = df_oxnard[df_oxnard['Oxnard Picklist SKU'].isin(selected_skus)]
        if selected_vendors:
            df_oxnard = df_oxnard[df_oxnard['Vendor'].isin(selected_vendors)]
        
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
        
        # Add filters in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Status filter for Wheeling Fruit Orders
            if 'Wheeling Status' in df.columns:
                # Replace empty status with "❌ No Status" label
                df['Wheeling Status'] = df['Wheeling Status'].replace(['', np.nan], '❌ No Status')
                wheeling_order_statuses = sorted(df['Wheeling Status'].unique())
                selected_wheeling_order_statuses = st.multiselect(
                    "Filter by Status",
                    wheeling_order_statuses,
                    default=wheeling_order_statuses,
                    key="wheeling_order_status_filter"
                )

        with col2:
            # Product Type filter
            if 'Fruit' in df.columns:
                fruits = sorted(df['Fruit'].dropna().unique())
                selected_fruits_wheeling = st.multiselect(
                    "Filter by Product Type",
                    fruits,
                    default=None,
                    placeholder="Choose product types...",
                    key="wheeling_order_product_filter"
                )

        with col3:
            # SKU filter
            if 'Wheeling Picklist SKU' in df.columns:
                skus = sorted(df['Wheeling Picklist SKU'].dropna().unique())
                selected_skus_wheeling = st.multiselect(
                    "Filter by SKU",
                    skus,
                    default=None,
                    placeholder="Choose SKUs...",
                    key="wheeling_order_sku_filter"
                )

        with col4:
            # Vendor filter
            if 'Vendor' in df.columns:
                vendors = sorted(df['Vendor'].dropna().unique())
                selected_vendors_wheeling = st.multiselect(
                    "Filter by Vendor",
                    vendors,
                    default=None,
                    placeholder="Choose vendors...",
                    key="wheeling_order_vendor_filter"
                )
        
        # Prepare Wheeling data
        wheeling_cols = [
            'Date_1', 'Vendor', 'Fruit', 'Wheeling Picklist SKU', 'Wheeling Status', 'Wheeling Notes',
            'Wheeling Weight Needed', 'Wheeling Order', 'Wheeling Actual Order', 'Wheeling Weight Per Pick', 'Wheeling Batchcode'
        ]
        wheeling_cols = [col for col in wheeling_cols if col in df.columns]
        
        # Filter Wheeling data for essential fields and selected filters
        df_wheeling = df[
            (df['Date_1'].notna()) & 
            (df['Date_1'] != '') & 
            (df['Vendor'].notna()) & 
            (df['Vendor'] != '') & 
            (df['Wheeling Status'].notna()) &
            (df['Wheeling Status'].isin(selected_wheeling_order_statuses))
        ][wheeling_cols].copy()

        # Apply additional filters
        if selected_fruits_wheeling:
            df_wheeling = df_wheeling[df_wheeling['Fruit'].isin(selected_fruits_wheeling)]
        if selected_skus_wheeling:
            df_wheeling = df_wheeling[df_wheeling['Wheeling Picklist SKU'].isin(selected_skus_wheeling)]
        if selected_vendors_wheeling:
            df_wheeling = df_wheeling[df_wheeling['Vendor'].isin(selected_vendors_wheeling)]
        
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

        # 5. Date Period Filter in sidebar
        st.sidebar.subheader("📅 Date Period")
        if 'Date_1' in df_filtered.columns:
            df_filtered['Date_1'] = pd.to_datetime(df_filtered['Date_1'], errors='coerce')
            df_filtered = df_filtered.dropna(subset=['Date_1'])
            
            if not df_filtered.empty:
                min_date = df_filtered['Date_1'].min()
                max_date = df_filtered['Date_1'].max()
                
                # Period Type Selection
                period_type = st.sidebar.radio(
                    "Select Week",
                    ["Previous Week", "Current Week", "Custom Range"],
                    key="period_type",
                    horizontal=True
                )
                
                # Date Range Selection
                if period_type == "Current Week":
                    current_start, current_end = get_week_range()
                    if check_data_availability(df_filtered, 'Date_1', current_start, current_end):
                        default_date = get_safe_date(min_date, max_date)
                        start_date, end_date = get_week_range(default_date)
                    else:
                        st.sidebar.warning("No data available for current week. Showing previous week instead.")
                        default_date = get_safe_date(min_date, max_date)
                        start_date, end_date = get_week_range(default_date, previous=True)
                elif period_type == "Previous Week":
                    default_date = get_safe_date(min_date, max_date)
                    start_date, end_date = get_week_range(default_date, previous=True)
                else:
                    default_date = get_safe_date(min_date, max_date)
                    date_range = st.sidebar.date_input(
                        "Select Date Range",
                        value=(default_date.date() - timedelta(days=7), default_date.date()),
                        min_value=min_date.date(),
                        max_value=max_date.date()
                    )
                    if len(date_range) == 2:
                        start_date = datetime.combine(date_range[0], datetime.min.time())
                        end_date = datetime.combine(date_range[1], datetime.max.time())
                
                # Apply date filter
                filtered_df = df_filtered[
                    (df_filtered['Date_1'] >= start_date) &
                    (df_filtered['Date_1'] <= end_date)
                ]
                
                if filtered_df.empty:
                    st.sidebar.warning("No data available for the selected date range.")
                else:
                    df_filtered = filtered_df

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
            col1, col2, col3 = st.columns(3)
            
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

            with col3:
                # Product Type filter
                product_types = sorted(orders_new_df[product_col].dropna().unique())
                selected_products = st.multiselect(
                    "Select Product Types",
                    product_types,
                    default=None,
                    placeholder="Choose product types..."
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

            if selected_products:
                filtered_df = filtered_df[filtered_df[product_col].isin(selected_products)]

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