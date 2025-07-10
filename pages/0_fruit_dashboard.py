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
    get_sku_info,
    load_wow_data
)
from utils.scripts_shopify.shopify_orders_report import update_orders_data, update_unfulfilled_orders
from utils.inventory_api import get_inventory_data
import time
import os
import numpy as np

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

def calculate_wow_change(prev_value, current_value):
    """Calculate week-over-week percentage change"""
    if pd.isna(prev_value) or pd.isna(current_value) or prev_value == 0:
        return None
    return ((current_value - prev_value) / abs(prev_value)) * 100

def apply_global_filters(df, filters):
    """Apply global filters to the main dataframe"""
    if df is None or df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Apply fruit filter
    if filters['fruits']:
        filtered_df = filtered_df[filtered_df['Fruit'].isin(filters['fruits'])]
    
    # Apply warehouse filter
    if filters['warehouse'] != 'All':
        # Filter based on warehouse status
        status_col = f"{filters['warehouse']} Status"
        order_col = f"{filters['warehouse']} Actual Order"
        if status_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[status_col].notna()]
            if order_col in filtered_df.columns:
                filtered_df[order_col] = pd.to_numeric(filtered_df[order_col], errors='coerce').fillna(0)
                filtered_df = filtered_df[filtered_df[order_col] > 0]
    
    # Apply fruit status filter (create synthetic status based on seasonality patterns)
    if filters['fruit_status']:
        # Since fruit status doesn't exist, create it based on patterns in the data
        seasonal_fruits = ['Fruit: Cherry', 'Fruit: Peaches', 'Fruit: Persimmon, Fuyu', 'Fruit: Loquat']
        if 'Seasonal' in filters['fruit_status'] and 'Abundant' not in filters['fruit_status']:
            filtered_df = filtered_df[filtered_df['Fruit'].isin(seasonal_fruits)]
        elif 'Abundant' in filters['fruit_status'] and 'Seasonal' not in filters['fruit_status']:
            filtered_df = filtered_df[~filtered_df['Fruit'].isin(seasonal_fruits)]
    
    # Apply SKU filter
    if filters['skus']:
        sku_mask = (
            filtered_df['Oxnard Picklist SKU'].isin(filters['skus']) |
            filtered_df['Wheeling Picklist SKU'].isin(filters['skus'])
        )
        filtered_df = filtered_df[sku_mask]
    
    # Apply date filter
    if filters['date_range'] and 'Date_1' in filtered_df.columns:
        start_date, end_date = filters['date_range']
        filtered_df['Date_1'] = pd.to_datetime(filtered_df['Date_1'], errors='coerce')
        filtered_df = filtered_df[
            (filtered_df['Date_1'] >= start_date) &
            (filtered_df['Date_1'] <= end_date)
        ]
    
    # Apply projection period filter
    if filters['projection_periods'] and 'Projection Period' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Projection Period'].isin(filters['projection_periods'])]
    
    # Apply inventory status filter
    if filters['inventory_status']:
        # This would apply to inventory data sections
        pass  # Handled in individual sections
    
    # Apply order status filter
    if filters['order_status']:
        if filters['warehouse'] != 'All':
            status_col = f"{filters['warehouse']} Status"
            if status_col in filtered_df.columns:
                filtered_df[status_col] = filtered_df[status_col].replace(['', np.nan], '❌ No Status')
                filtered_df = filtered_df[filtered_df[status_col].isin(filters['order_status'])]
        else:
            # Apply to both warehouses
            oxnard_mask = filtered_df['Oxnard Status'].isin(filters['order_status'])
            wheeling_mask = filtered_df['Wheeling Status'].isin(filters['order_status'])
            filtered_df = filtered_df[oxnard_mask | wheeling_mask]
    
    return filtered_df

def load_inventory_and_picklist_data():
    """Load inventory and picklist data individually with progress messages"""
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

def create_sidebar_filters(df):
    """Create compact filter controls in the sidebar"""
    with st.sidebar:
        st.header("🎛️ Filters")
        
        # Initialize filters in session state
        if 'global_filters' not in st.session_state:
            st.session_state.global_filters = {
                'fruits': [],
                'warehouse': 'All',
                'fruit_status': [],
                'skus': [],
                'date_range': None,
                'projection_periods': [],
                'inventory_status': [],
                'order_status': []
            }
        
        # Fruit Filter
        if df is not None and 'Fruit' in df.columns:
            available_fruits = sorted(df['Fruit'].dropna().unique())
            selected_fruits = st.multiselect(
                "🍊 Fruits",
                available_fruits,
                default=st.session_state.global_filters['fruits'],
                placeholder="All fruits",
                key="filter_fruits"
            )
            st.session_state.global_filters['fruits'] = selected_fruits
        
        # Warehouse Filter
        warehouse_options = ["All", "Oxnard", "Wheeling"]
        selected_warehouse = st.selectbox(
            "🏭 Warehouse",
            warehouse_options,
            index=warehouse_options.index(st.session_state.global_filters['warehouse']),
            key="filter_warehouse"
        )
        st.session_state.global_filters['warehouse'] = selected_warehouse
        
        # Fruit Status Filter (synthetic)
        fruit_status_options = ["Seasonal", "Abundant"]
        selected_fruit_status = st.multiselect(
            "🌱 Fruit Status",
            fruit_status_options,
            default=st.session_state.global_filters['fruit_status'],
            placeholder="All statuses",
            key="filter_fruit_status"
        )
        st.session_state.global_filters['fruit_status'] = selected_fruit_status
        
        # SKU Filter
        if df is not None:
            all_skus = set()
            if 'Oxnard Picklist SKU' in df.columns:
                all_skus.update(df['Oxnard Picklist SKU'].dropna().unique())
            if 'Wheeling Picklist SKU' in df.columns:
                all_skus.update(df['Wheeling Picklist SKU'].dropna().unique())
            
            selected_skus = st.multiselect(
                "📦 SKUs",
                sorted(all_skus),
                default=st.session_state.global_filters['skus'],
                placeholder="All SKUs",
                key="filter_skus"
            )
            st.session_state.global_filters['skus'] = selected_skus
        
        # Date Period Filter
        if df is not None and 'Date_1' in df.columns:
            df['Date_1'] = pd.to_datetime(df['Date_1'], errors='coerce')
            df_clean = df.dropna(subset=['Date_1'])
            
            if not df_clean.empty:
                min_date = df_clean['Date_1'].min()
                max_date = df_clean['Date_1'].max()
                
                period_type = st.radio(
                    "📅 Date Period",
                    ["Current Week", "Previous Week", "Custom"],
                    key="filter_period_type"
                )
                
                if period_type == "Current Week":
                    start_date, end_date = get_week_range()
                elif period_type == "Previous Week":
                    start_date, end_date = get_week_range(previous=True)
                else:
                    default_start = get_safe_date(min_date, max_date).date() - timedelta(days=7)
                    default_end = get_safe_date(min_date, max_date).date()
                    
                    date_range = st.date_input(
                        "Custom Range",
                        value=(default_start, default_end),
                        min_value=min_date.date(),
                        max_value=max_date.date(),
                        key="filter_custom_dates"
                    )
                    if len(date_range) == 2:
                        start_date = datetime.combine(date_range[0], datetime.min.time())
                        end_date = datetime.combine(date_range[1], datetime.max.time())
                    else:
                        start_date, end_date = get_week_range()
                
                st.session_state.global_filters['date_range'] = (start_date, end_date)
        
        # Projection Period Filter
        if df is not None and 'Projection Period' in df.columns:
            projection_periods = sorted(df['Projection Period'].dropna().astype(str).unique())
            selected_periods = st.multiselect(
                "📊 Projection Periods",
                projection_periods,
                default=st.session_state.global_filters['projection_periods'],
                placeholder="All periods",
                key="filter_projection_periods"
            )
            st.session_state.global_filters['projection_periods'] = selected_periods
        
        # Inventory Status Filter
        inventory_status_options = ["Good", "Bad"]
        selected_inventory_status = st.multiselect(
            "📈 Inventory Status",
            inventory_status_options,
            default=st.session_state.global_filters['inventory_status'],
            placeholder="All statuses",
            key="filter_inventory_status"
        )
        st.session_state.global_filters['inventory_status'] = selected_inventory_status
        
        # Order Status Filter
        if df is not None:
            order_status_options = set()
            if 'Oxnard Status' in df.columns:
                order_status_options.update(df['Oxnard Status'].dropna().unique())
            if 'Wheeling Status' in df.columns:
                order_status_options.update(df['Wheeling Status'].dropna().unique())
            
            # Map status values to more user-friendly names
            status_mapping = {
                'Imported': 'Imported',
                'Confirmed': 'Confirmed', 
                'N/A': 'Pending',
                '': 'No Status'
            }
            
            order_status_options = sorted([status_mapping.get(status, status) for status in order_status_options if status])
            
            selected_order_status = st.multiselect(
                "📋 Order Status",
                order_status_options,
                default=st.session_state.global_filters['order_status'],
                placeholder="All statuses",
                key="filter_order_status"
            )
            st.session_state.global_filters['order_status'] = selected_order_status
        
        st.divider()
        
        # Reset filters button
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.session_state.global_filters = {
                'fruits': [],
                'warehouse': 'All',
                'fruit_status': [],
                'skus': [],
                'date_range': None,
                'projection_periods': [],
                'inventory_status': [],
                'order_status': []
            }
            st.rerun()
        
        # Show active filters summary
        active_filters = []
        for key, value in st.session_state.global_filters.items():
            if value and value != 'All' and value != []:
                if key == 'date_range' and value:
                    start, end = value
                    active_filters.append(f"📅 {start.strftime('%m/%d')}-{end.strftime('%m/%d')}")
                elif isinstance(value, list) and value:
                    active_filters.append(f"{len(value)} {key}")
                elif value != 'All':
                    active_filters.append(f"{value}")
        
        if active_filters:
            st.info("**Active:** " + " | ".join(active_filters))
    
    return st.session_state.global_filters

def main():
    st.title("🍎 Fruit Dashboard")
    
    # Load all data using the original pattern
    
    # Load WoW data
    if 'wow_df' not in st.session_state:
        message_placeholder = st.empty()
        with st.spinner("Loading WoW data..."):
            wow_df = load_wow_data()
            if wow_df is not None:
                st.session_state['wow_df'] = wow_df
                message_placeholder.success("✅ WoW data loaded successfully!")
                time.sleep(1)  # Show message for 1 second
                message_placeholder.empty()
            else:
                message_placeholder.warning("⚠️ Failed to load WoW data")

    # Load aggregated orders data
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
    load_inventory_and_picklist_data()
    
    # Get main dataframe
    df = st.session_state.get('agg_orders_df')
    
    if df is None:
        st.error("❌ Failed to load main fruit data. Please check your data sources.")
        return
    
    # Create sidebar filters
    filters = create_sidebar_filters(df)
    
    # Apply global filters to main dataframe
    df_filtered = apply_global_filters(df, filters)

    # Display Week over Week Data
    with st.expander("📊 Week over Week Analysis", expanded=True):
        wow_df = st.session_state.get('wow_df')
        if wow_df is not None and not wow_df.empty:
            st.markdown("""
            ### How to use Week over Week Analysis:
            1. Select two date ranges to compare using the dropdown below
            2. The most recent two weeks are selected by default
            3. The "Change" column shows the percentage change from the earlier week (left) to the later week (right)
            4. A positive change (green) means an increase, negative (red) means a decrease
            """)

            # Add date range filter
            def parse_date_for_sorting(date_range):
                try:
                    # Split and clean the date string
                    start_date = date_range.split('-')[0].strip()
                    return pd.to_datetime(start_date, format='%m/%d')
                except:
                    # Return a minimum date for any unparseable dates
                    return pd.Timestamp.min

            def format_date_range(date_range):
                """Convert '6/29-7/5' to 'June 29/July 5'"""
                try:
                    start, end = date_range.split('-')
                    start = pd.to_datetime(start.strip(), format='%m/%d')
                    end = pd.to_datetime(end.strip(), format='%m/%d')
                    return f"{start.strftime('%B')} {start.day}/{end.strftime('%B')} {end.day}"
                except:
                    return date_range

            # Sort date ranges chronologically (oldest to newest)
            date_ranges = sorted(wow_df['Date Range'].unique(), 
                               key=parse_date_for_sorting)
            
            # Create a mapping of formatted dates to original dates
            date_range_mapping = {format_date_range(dr): dr for dr in date_ranges}
            formatted_ranges = list(date_range_mapping.keys())
            
            # Get the last two weeks for default selection
            default_ranges = formatted_ranges[-2:] if len(formatted_ranges) > 1 else formatted_ranges
            
            selected_formatted_ranges = st.multiselect(
                "Select Date Ranges to Compare",
                formatted_ranges,
                default=default_ranges,
                placeholder="Choose two weeks to compare...",
                help="Select two date ranges to compare. Earlier week will be shown on the left, later week on the right."
            )
            
            # Convert back to original date ranges
            selected_ranges = [date_range_mapping[fr] for fr in selected_formatted_ranges]
            
            if selected_ranges:
                filtered_df = wow_df[wow_df['Date Range'].isin(selected_ranges)]
                
                # Sort ranges chronologically (older to newer)
                sorted_ranges = sorted(selected_ranges, key=parse_date_for_sorting)
                
                # Pivot the data for better comparison
                pivot_df = filtered_df.pivot(
                    index='Metric',
                    columns='Date Range',
                    values='Value'
                ).reset_index()
                
                # Reorder columns to show previous week first
                column_order = ['Metric'] + sorted_ranges
                pivot_df = pivot_df[column_order]
                
                # Calculate week-over-week changes if we have at least 2 ranges selected
                if len(sorted_ranges) >= 2:
                    for i in range(len(sorted_ranges)-1):
                        prev_range = sorted_ranges[i]
                        current_range = sorted_ranges[i+1]
                        formatted_prev = format_date_range(prev_range)
                        formatted_curr = format_date_range(current_range)
                        col_name = f"Change ({formatted_prev} → {formatted_curr})"
                        pivot_df[col_name] = pivot_df.apply(
                            lambda row: calculate_wow_change(
                                row[prev_range], 
                                row[current_range]
                            ) if pd.notna(row[prev_range]) and pd.notna(row[current_range]) else None,
                            axis=1
                        )
                
                # Display the data
                column_config = {
                    "Metric": st.column_config.TextColumn(
                        "Metric",
                        help="Performance metric"
                    )
                }
                
                # Add range columns with formatted names
                for range_name in sorted_ranges:
                    formatted_name = format_date_range(range_name)
                    column_config[range_name] = st.column_config.NumberColumn(
                        formatted_name,
                        format="%.2f"
                    )
                
                # Add change columns if we have multiple ranges
                if len(sorted_ranges) >= 2:
                    for i in range(len(sorted_ranges)-1):
                        prev_range = sorted_ranges[i]
                        current_range = sorted_ranges[i+1]
                        formatted_prev = format_date_range(prev_range)
                        formatted_curr = format_date_range(current_range)
                        change_col = f"Change ({formatted_prev} → {formatted_curr})"
                        column_config[f"Change ({prev_range} → {current_range})"] = st.column_config.NumberColumn(
                            change_col,
                            format="+%.1f%%"
                        )
                
                st.dataframe(
                    pivot_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )

                # Display summary metrics
                st.markdown("### Key Insights")
                if len(selected_ranges) >= 2:
                    latest_range = sorted_ranges[-1]
                    prev_range = sorted_ranges[-2]
                    
                    col1, col2, col3 = st.columns(3)
                    metrics_to_show = pivot_df['Metric'].tolist()[:3]  # Show first 3 metrics
                    
                    for i, (col, metric) in enumerate(zip([col1, col2, col3], metrics_to_show)):
                        with col:
                            current_val = pivot_df[pivot_df['Metric'] == metric][latest_range].iloc[0]
                            prev_val = pivot_df[pivot_df['Metric'] == metric][prev_range].iloc[0]
                            delta = calculate_wow_change(prev_val, current_val)
                            
                            st.metric(
                                metric,
                                f"{current_val:,.2f}",
                                f"{delta:+.1f}%" if pd.notna(delta) else None,
                                delta_color="normal"
                            )
            else:
                st.warning("Please select at least one date range to display data.")
        else:
            st.warning("No Week over Week data available")

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
                    display_df = filtered_df[columns_to_display].sort_values('INVENTORY DATE', ascending=False)
                    
                    # Define styling function for STATUS column
                    def style_oxnard_status(df):
                        def color_status(val):
                            if pd.isna(val):
                                return ''
                            val_str = str(val).strip().lower()
                            if val_str == 'good':
                                return 'background-color: #d4edda; color: #155724'  # Green background, dark green text
                            elif val_str == 'bad':
                                return 'background-color: #f8d7da; color: #721c24'  # Red background, dark red text
                            return ''
                        
                        # Apply styling only if STATUS column exists
                        if 'STATUS' in df.columns:
                            return df.style.map(color_status, subset=['STATUS'])
                        return df.style
                    
                    st.dataframe(
                        style_oxnard_status(display_df),
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
                                help="Inventory status: Good (Green) / Bad (Red)"
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
                    display_df = filtered_df[columns_to_display].sort_values('INVENTORY DATE', ascending=False)
                    
                    # Define styling function for STATUS column
                    def style_wheeling_status(df):
                        def color_status(val):
                            if pd.isna(val):
                                return ''
                            val_str = str(val).strip().lower()
                            if val_str == 'good':
                                return 'background-color: #d4edda; color: #155724'  # Green background, dark green text
                            elif val_str == 'bad':
                                return 'background-color: #f8d7da; color: #721c24'  # Red background, dark red text
                            return ''
                        
                        # Apply styling only if STATUS column exists
                        if 'STATUS' in df.columns:
                            return df.style.map(color_status, subset=['STATUS'])
                        return df.style
                    
                    st.dataframe(
                        style_wheeling_status(display_df),
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
                                help="Inventory status: Good (Green) / Bad (Red)"
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
        reference_data = st.session_state.get('pieces_vs_lb_df')
        
        # Pieces vs Lb Conversion
        pieces_vs_lb_df = reference_data
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