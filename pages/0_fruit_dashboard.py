import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import sys
import os
import json

# Add the parent directory to the Python path to fix import issues
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.google_sheets import (
    load_agg_orders,
    load_oxnard_inventory,
    load_wheeling_inventory,
    load_pieces_vs_lb_conversion,
    load_all_picklist_v2,
    load_orders_new,
    get_sku_info,
    load_wow_data,
    load_sku_mappings_from_sheets,
    load_sku_type_data
)
from utils.scripts_shopify.shopify_orders_report import update_orders_data, update_unfulfilled_orders, update_fulfilled_orders_data
from utils.inventory_api import get_inventory_data
import time
import os
import logging

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

def is_digital_item(row):
    """
    Check if an item is digital/virtual and should be excluded from inventory needs.
    
    This function distinguishes between:
    - Digital items (ripening guides, instructions) - EXCLUDED
    - Physical gift items (fruit gifts, spoons) - INCLUDED
    - Free promotional physical items - EXCLUDED (to be conservative)
    """
    sku = str(row.get('SKU', '')).lower()
    product_title = str(row.get('Product Title', '')).lower()
    unit_price = row.get('Unit Price', 0)
    
    # Exclude free items that are likely digital
    if unit_price <= 0:
        # Check if this is a digital item based on SKU patterns
        digital_patterns = [
            'ripening_guide', 'guide', 'e.ripening', 'digital:', 
            'instruction', 'pdf', 'ebook', 'video'
        ]
        for pattern in digital_patterns:
            if pattern in sku or pattern in product_title:
                return True
        # If it's free but doesn't match digital patterns, still exclude it
        # This catches generic free items that aren't clearly digital
        return True
    
    # Allow all paid items (including paid gift items)
    return False

def filter_physical_items(df):
    """
    Filter out digital/free items to get only physical fruit items.
    
    This excludes:
    - Items with $0 unit price (free items like ripening guides)
    - SKUs containing '-gift' or '-bab' (gift box items and free gift items)
    - Any other patterns indicating digital/free items
    
    This ensures fulfillment time calculations only include actual fruit products
    that require physical picking, packing, and shipping.
    """
    if df is None or df.empty:
        return df
    
    # Check if the expected columns exist
    price_col = 'Unit Price' if 'Unit Price' in df.columns else None
    sku_col = 'SKU' if 'SKU' in df.columns else None
    
    if price_col is None or sku_col is None:
        # If we don't have the necessary columns, return the original dataframe
        return df
        
    return df[
        (df[price_col] > 0) &  # Exclude free items
        (~df[sku_col].str.contains('-gift|-bab', case=False, na=False, regex=True))  # Exclude gift items and free gift boxes
    ]

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

def load_shopify_orders(start_date=None, end_date=None, force_refresh=False, use_simplified_query=False):
    """Load Shopify orders data and store in session state"""
    if start_date is None:
        # Default to last Monday
        end_date = datetime.now()
        days_since_monday = end_date.weekday()
        start_date = end_date - timedelta(days=days_since_monday)
        start_date = datetime.combine(start_date.date(), datetime.min.time())
        end_date = datetime.combine(end_date.date(), datetime.max.time())
    
    # Check if we need to refresh the data
    cache_key = f"shopify_orders_df_{'simplified' if use_simplified_query else 'detailed'}"
    
    if force_refresh or cache_key not in st.session_state:
        with st.spinner("Loading orders data..."):
            try:
                if use_simplified_query:
                    orders_df = update_fulfilled_orders_data(start_date=start_date, end_date=end_date)
                else:
                    orders_df = update_orders_data(start_date=start_date, end_date=end_date)
                
                unfulfilled_df = update_unfulfilled_orders(start_date=start_date, end_date=end_date)
                
                if orders_df is not None:
                    st.session_state[cache_key] = orders_df
                    st.session_state[f'shopify_orders_start_date_{"simplified" if use_simplified_query else "detailed"}'] = start_date
                    st.session_state[f'shopify_orders_end_date_{"simplified" if use_simplified_query else "detailed"}'] = end_date
                
                if unfulfilled_df is not None:
                    st.session_state['unfulfilled_orders_df'] = unfulfilled_df
                
                return orders_df, unfulfilled_df
            except Exception as e:
                st.error(f"Error loading orders: {str(e)}")
                return None, None
    
    # Return cached data if dates match
    elif (st.session_state.get(f'shopify_orders_start_date_{"simplified" if use_simplified_query else "detailed"}') == start_date and 
          st.session_state.get(f'shopify_orders_end_date_{"simplified" if use_simplified_query else "detailed"}') == end_date):
        return (st.session_state.get(cache_key), 
                st.session_state.get('unfulfilled_orders_df'))
    
    # Dates changed, need to refresh
    else:
        return load_shopify_orders(start_date, end_date, force_refresh=True, use_simplified_query=use_simplified_query)

def calculate_wow_change(prev_value, current_value):
    """Calculate week-over-week percentage change with better handling of edge cases"""
    if pd.isna(prev_value) or pd.isna(current_value):
        return None
    
    # Handle zero or very small previous values to avoid extreme percentages
    if abs(prev_value) < 0.01:  # Very small number threshold
        if abs(current_value) < 0.01:
            return 0.0  # Both are essentially zero
        else:
            # When previous is ~0 but current isn't, calculate absolute change instead
            return None  # Will be handled as "new value" in display
    
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
    st.title("Fruit Dashboard")
    
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

    # Load SKU mappings if not already loaded (needed for unfulfilled orders processing)
    if 'sku_mappings' not in st.session_state:
        with st.spinner("Loading SKU mappings..."):
            try:
                sku_mappings = load_sku_mappings_from_sheets()
                st.session_state['sku_mappings'] = sku_mappings
                if not sku_mappings or not isinstance(sku_mappings, dict):
                    st.warning("⚠️ SKU mappings loaded but structure is invalid")
            except Exception as e:
                st.error(f"❌ Could not load SKU mappings: {e}")
                st.session_state['sku_mappings'] = None

    # Get main dataframe
    df = st.session_state.get('agg_orders_df')
    
    if df is None:
        st.error("❌ Failed to load main fruit data. Please check your data sources.")
        return

    # ------------------- FAST FRUIT SUMMARY -------------------
    st.markdown("---")
    
    # Load SKU mappings if not already loaded (needed for conversions)
    if 'sku_mappings' not in st.session_state:
        try:
            import json
            with open('sheet_data/sku_mappings.json', 'r') as f:
                sku_mappings = json.load(f)
                st.session_state['sku_mappings'] = sku_mappings
        except Exception as e:
            st.warning(f"Could not load SKU mappings: {e}")
            st.session_state['sku_mappings'] = None

    # Load ColdCart inventory for "What we have" if not already loaded
    if 'coldcart_inventory' not in st.session_state:
        with st.spinner("Loading ColdCart inventory..."):
            try:
                coldcart_df = get_inventory_data()
                if coldcart_df is not None:
                    st.session_state['coldcart_inventory'] = coldcart_df
                else:
                    st.session_state['coldcart_inventory'] = pd.DataFrame()
            except Exception as e:
                st.session_state['coldcart_inventory'] = pd.DataFrame()

    # FAST PRODUCT TYPE LEVEL CALCULATION FOR SASHA
    def create_fast_product_type_summary():
        """Create fast real-time Product Type level summary with automatic lb conversions and cost
        
        CONVERSION LOGIC:
        - ColdCart inventory: pieces → lbs (using pieces_df conversion table)
        - In Transit orders: lbs → pieces (using pieces_df conversion table)
        - This ensures consistent unit tracking across different data sources
        """
        
        # Load fresh data from Google Sheets (not static files)
        try:
            # 1. Get SKU mappings from session state (already loaded in main function)
            sku_mappings = st.session_state.get('sku_mappings', {})
            
            # 2. Load unfulfilled orders from session state (this comes from Shopify API)
            unfulfilled_df = st.session_state.get('unfulfilled_orders_df')
            
            # If not in session state, load fresh data from Shopify API
            if unfulfilled_df is None or unfulfilled_df.empty:
                try:
                    # Use the existing Shopify API functions to get fresh data
                    from utils.scripts_shopify.shopify_orders_report import update_unfulfilled_orders
                    
                    # Load last 7 days of unfulfilled orders
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)
                    
                    unfulfilled_df = update_unfulfilled_orders(start_date=start_date, end_date=end_date)
                    
                    if unfulfilled_df is not None and not unfulfilled_df.empty:
                        # Store in session state for future use
                        st.session_state['unfulfilled_orders_df'] = unfulfilled_df
                    else:
                        unfulfilled_df = None
                except Exception as e:
                    unfulfilled_df = None
            else:
                pass  # Using existing unfulfilled orders from session state
            
            # 3. Load inventory data from Google Sheets
            pieces_df = load_pieces_vs_lb_conversion()
            oxnard_df = load_oxnard_inventory()
            wheeling_df = load_wheeling_inventory()
            agg_orders_df = load_agg_orders()
            orders_new_df = load_orders_new()
            
            # 4. Load SKU Type data for helper lookups
            try:
                sku_type_df = load_sku_type_data()
            except Exception as e:
                sku_type_df = None
            
            # 5. Load ColdCart from live API
            coldcart_df = st.session_state.get('coldcart_inventory')
            
        except Exception as e:
            st.error(f"Error loading Google Sheets data: {e}")
            return pd.DataFrame()
        
        # Helper function to get product type from Shopify SKU
        def get_product_type_from_shopify_sku(sku):
            if not sku or pd.isna(sku):
                return "Unknown"
            
            # Check if SKU mappings are available
            if not sku_mappings or not isinstance(sku_mappings, dict):
                return "Unknown"
            
            # Check SKU mappings for Shopify SKUs
            for location in ['Oxnard', 'Wheeling']:
                if location in sku_mappings and 'singles' in sku_mappings[location]:
                    if sku in sku_mappings[location]['singles']:
                        return sku_mappings[location]['singles'][sku].get('pick_type_inventory', 'Unknown')
                if location in sku_mappings and 'bundles' in sku_mappings[location]:
                    if sku in sku_mappings[location]['bundles']:
                        # For bundles, use the first component's product type
                        components = sku_mappings[location]['bundles'][sku]
                        if components and len(components) > 0:
                            return components[0].get('pick_type_inventory', 'Unknown')
            
            return "Unknown"
        
        # Helper function to get product type from inventory SKU (pieces_df)
        def get_product_type_from_pieces_sku(sku):
            if pieces_df is None or pd.isna(sku) or not sku:
                return "Unknown"
            
            conversion_row = pieces_df[pieces_df['picklist sku'] == sku]
            if not conversion_row.empty:
                return conversion_row.iloc[0].get('Pick Type', 'Unknown')
            return "Unknown"
        
        # Helper function to get weight conversion from pieces_df
        def get_weight_from_pieces_conversion(sku, quantity):
            if pieces_df is None or pd.isna(sku) or not sku:
                return 0
            
            conversion_row = pieces_df[pieces_df['picklist sku'] == sku]
            if not conversion_row.empty:
                weight_per_unit = pd.to_numeric(conversion_row.iloc[0].get('Weight (lbs)', 0), errors='coerce')
                calculated_weight = float(quantity) * float(weight_per_unit) if not pd.isna(weight_per_unit) else 0
                return calculated_weight
            return 0
        
        # Helper function to get weight conversion from SKU mappings
        def get_weight_from_sku_mapping(sku, quantity):
            if not sku or pd.isna(sku) or not sku_mappings or not isinstance(sku_mappings, dict):
                return 0
            
            for location in ['Oxnard', 'Wheeling']:
                if location in sku_mappings and 'singles' in sku_mappings[location]:
                    if sku in sku_mappings[location]['singles']:
                        total_weight = sku_mappings[location]['singles'][sku].get('total_pick_weight', 0)
                        return float(quantity) * float(total_weight) if total_weight else 0
                if location in sku_mappings and 'bundles' in sku_mappings[location]:
                    if sku in sku_mappings[location]['bundles']:
                        # For bundles, sum up all component weights
                        total_weight = 0
                        for component in sku_mappings[location]['bundles'][sku]:
                            comp_weight = component.get('weight', 0)
                            comp_qty = component.get('actualqty', 1)
                            total_weight += float(comp_weight) * float(comp_qty)
                        return float(quantity) * total_weight
            return 0

        # Helper function to get picklist SKU from Shopify SKU
        def get_picklist_sku_from_shopify_sku(sku):
            if not sku or pd.isna(sku) or not sku_mappings or not isinstance(sku_mappings, dict):
                return sku  # fallback to original SKU
            
            for location in ['Oxnard', 'Wheeling']:
                if location in sku_mappings and 'singles' in sku_mappings[location]:
                    if sku in sku_mappings[location]['singles']:
                        return sku_mappings[location]['singles'][sku].get('picklist_sku', sku)
            return sku
        
        # Helper function to convert lbs to pieces using the conversion table
        def get_pieces_from_weight_conversion(sku, weight_lbs):
            """Convert weight in lbs to pieces/units using the pieces vs lb conversion table"""
            if pieces_df is None or pd.isna(sku) or not sku or weight_lbs <= 0:
                return 0
            
            conversion_row = pieces_df[pieces_df['picklist sku'] == sku]
            if not conversion_row.empty:
                weight_per_unit = pd.to_numeric(conversion_row.iloc[0].get('Weight (lbs)', 0), errors='coerce')
                if not pd.isna(weight_per_unit) and weight_per_unit > 0:
                    calculated_pieces = float(weight_lbs) / float(weight_per_unit)
                    return calculated_pieces
            return 0
        
        # Helper function to convert lbs to pieces using product type (for in-transit orders)
        def get_pieces_from_product_type_weight(product_type, weight_lbs):
            """Convert weight in lbs to pieces based on product type by finding a representative SKU"""
            if pieces_df is None or not product_type or weight_lbs <= 0:
                return 0
            
            # Find a representative SKU for this product type
            product_type_rows = pieces_df[pieces_df['Pick Type'] == product_type]
            if product_type_rows.empty:
                # Try without "Fruit: " prefix if it exists
                simplified_type = product_type.replace('Fruit: ', '') if 'Fruit: ' in product_type else product_type
                product_type_rows = pieces_df[pieces_df['Pick Type'].str.contains(simplified_type, case=False, na=False)]
            
            if not product_type_rows.empty:
                # Use the first matching row as representative
                weight_per_unit = pd.to_numeric(product_type_rows.iloc[0].get('Weight (lbs)', 0), errors='coerce')
                if not pd.isna(weight_per_unit) and weight_per_unit > 0:
                    calculated_pieces = float(weight_lbs) / float(weight_per_unit)
                    return calculated_pieces
            return 0

        # Initialize product summary
        product_summary = {}
        processed_count = 0  # Initialize counter  
        honey_debug_count = 0  # Initialize honey debug counter

        # Process NEEDS: Unfulfilled orders from Shopify
        if unfulfilled_df is not None and not unfulfilled_df.empty:
            # Filter out ONLY specific digital items for Fast Summary: e.ripening_guide and expedited-shipping
            # Keep ALL gifts, $0 items, and physical products
            def is_specific_digital_item(row):
                sku = str(row.get('SKU', '')).lower()
                return sku in ['e.ripening_guide', 'expedited-shipping']
            
            needs_df = unfulfilled_df[~unfulfilled_df.apply(is_specific_digital_item, axis=1)].copy()
            
            # Check if we have SKU mappings
            if sku_mappings is None:
                pass  # No SKU mappings, removed warning message
            else:
                pass  # SKU mappings loaded, removed info message
            
            for _, row in needs_df.iterrows():
                sku = row.get('SKU', '')  # Use correct column name
                quantity = row.get('Unfulfilled Quantity', 0)  # Use correct column name
                unit_price = row.get('Unit Price', 0)
                fulfillment_center = row.get('Fulfillment Center', 'Oxnard')
                
                # Convert to numeric
                quantity = pd.to_numeric(quantity, errors='coerce')
                unit_price = pd.to_numeric(unit_price, errors='coerce')
                
                if pd.isna(quantity) or quantity == 0:
                    continue
                
                # Normalize warehouse key
                warehouse_key = 'Oxnard' if 'oxnard' in fulfillment_center.lower() or 'moorpark' in fulfillment_center.lower() else 'Wheeling'
                
                # Check if this is a bundle first
                is_bundle = False
                if (sku_mappings and isinstance(sku_mappings, dict) and 
                    warehouse_key in sku_mappings and 'bundles' in sku_mappings[warehouse_key]):
                    is_bundle = sku in sku_mappings[warehouse_key]['bundles']
                
                if is_bundle:
                    # Process bundle components
                    bundle_components = sku_mappings[warehouse_key]['bundles'][sku]
                    for component in bundle_components:
                        component_sku = component.get('component_sku', '')
                        base_component_qty = float(component.get('actualqty', 1.0))
                        component_qty = base_component_qty * quantity
                        
                        # Get component weight from singles mapping
                        component_weight = 0.0
                        if ('singles' in sku_mappings[warehouse_key] and 
                            component_sku in sku_mappings[warehouse_key]['singles']):
                            singles_data = sku_mappings[warehouse_key]['singles'][component_sku]
                            unit_weight = float(singles_data.get('total_pick_weight', 0.0))
                            component_weight = unit_weight * component_qty
                        else:
                            # Fallback to bundle component weight
                            component_weight = float(component.get('weight', 0.0)) * quantity
                        
                        # Get component product type - with SKU Type helper fallback
                        component_product_type = component.get('pick_type', '') or get_product_type_from_shopify_sku(component_sku)
                        
                        # If component doesn't have a pickup SKU mapping, try to resolve it using helper
                        resolved_component_sku = component_sku
                        if ('singles' not in sku_mappings[warehouse_key] or 
                            component_sku not in sku_mappings[warehouse_key]['singles']) and sku_type_df is not None:
                            # Try to find helper SKU for this component
                            helper_row = sku_type_df[sku_type_df['SKU'] == component_sku]
                            if not helper_row.empty:
                                sku_helper = helper_row.iloc[0].get('SKU_Helper', '')
                                if (sku_helper and sku_helper != component_sku and
                                    'singles' in sku_mappings[warehouse_key] and 
                                    sku_helper in sku_mappings[warehouse_key]['singles']):
                                    resolved_component_sku = sku_helper
                                    # Get weight from the helper SKU mapping
                                    singles_data = sku_mappings[warehouse_key]['singles'][sku_helper]
                                    unit_weight = float(singles_data.get('total_pick_weight', 0.0))
                                    component_weight = unit_weight * component_qty
                                    # Get product type from helper mapping
                                    component_product_type = singles_data.get('pick_type', '') or component_product_type
                        
                        # If still unknown and we have SKU type data, try the helper
                        if component_product_type == "Unknown" and sku_type_df is not None and component_sku:
                            # Try to find component SKU using the helper
                                helper_row = sku_type_df[sku_type_df['SKU'] == component_sku]
                                if not helper_row.empty:
                                    component_product_type = helper_row.iloc[0].get('PRODUCT_TYPE', 'Unknown')
                            
                        if component_product_type == "Unknown":
                            component_product_type = f"Bundle Component: {component_sku}"
                        
                        if component_product_type not in product_summary:
                            product_summary[component_product_type] = {
                                'Needs (lbs)': 0,
                                'Needs (ea)': 0,  # Changed from 'units' to 'ea'
                                'Inventory (lbs)': 0,
                                'Inventory Coldcart (ea)': 0,
                                'In Transit (lbs)': 0,
                                'In Transit (ea)': 0,  # Changed from 'units' to 'ea'
                                'Total Inventory (lbs)': 0,
                                'Total Inventory (ea)': 0,  # Changed from 'units' to 'ea'
                                'Difference (lbs)': 0,
                                'Difference (ea)': 0,  # Changed from 'units' to 'ea'
                                'Cost ($)': 0,
                                'Weight per Unit (lbs)': 0,
                                'Picklist SKUs': set(),
                                'Shopify SKUs': set()
                            }
                        
                        product_summary[component_product_type]['Needs (lbs)'] += component_weight
                        product_summary[component_product_type]['Needs (ea)'] += component_qty
                        product_summary[component_product_type]['Shopify SKUs'].add(sku)  # Original bundle SKU
                        product_summary[component_product_type]['Picklist SKUs'].add(resolved_component_sku)
                        
                        processed_count += 1
                        
                        # Calculate average weight per unit
                        if product_summary[component_product_type]['Needs (ea)'] > 0:
                            product_summary[component_product_type]['Weight per Unit (lbs)'] = product_summary[component_product_type]['Needs (lbs)'] / product_summary[component_product_type]['Needs (ea)']
                else:
                    # Process single SKU
                    # Get mapping data for weight and product type
                    actualqty = quantity
                    weight_lbs = quantity * 1.0  # Default fallback
                    product_type = get_product_type_from_shopify_sku(sku)
                    picklist_sku = sku  # Default fallback
                    
                    if (sku_mappings and isinstance(sku_mappings, dict) and 
                        warehouse_key in sku_mappings and 
                        'singles' in sku_mappings[warehouse_key] and
                        sku in sku_mappings[warehouse_key]['singles']):
                        singles_data = sku_mappings[warehouse_key]['singles'][sku]
                        base_actualqty = float(singles_data.get('actualqty', 1.0))
                        base_weight = float(singles_data.get('total_pick_weight', 0.0))
                        
                        # Calculate actual quantities based on order quantity
                        actualqty = base_actualqty * quantity
                        weight_lbs = base_weight * quantity
                        
                        # Get product type and picklist SKU from mapping
                        product_type = singles_data.get('pick_type', '') or product_type
                        picklist_sku = singles_data.get('picklist_sku', '') or sku
                    else:
                        # Fallback to existing weight calculation method
                        weight_lbs = get_weight_from_sku_mapping(sku, quantity)
                        picklist_sku = get_picklist_sku_from_shopify_sku(sku)
                    
                    # Try SKU Type helper if product type is still unknown
                    if product_type == "Unknown" and sku_type_df is not None:
                        helper_row = sku_type_df[sku_type_df['SKU'] == sku]
                        if not helper_row.empty:
                            product_type = helper_row.iloc[0].get('PRODUCT_TYPE', 'Unknown')
                    
                    if product_type == "Unknown":
                        continue
                    
                    if product_type not in product_summary:
                        product_summary[product_type] = {
                            'Needs (lbs)': 0,
                            'Needs (ea)': 0,  # Changed from 'units' to 'ea'
                            'Inventory (lbs)': 0,
                            'Inventory Coldcart (ea)': 0,
                            'In Transit (lbs)': 0,
                            'In Transit (ea)': 0,  # Changed from 'units' to 'ea'
                            'Total Inventory (lbs)': 0,
                            'Total Inventory (ea)': 0,  # Changed from 'units' to 'ea'
                            'Difference (lbs)': 0,
                            'Difference (ea)': 0,  # Changed from 'units' to 'ea'
                            'Cost ($)': 0,
                            'Weight per Unit (lbs)': 0,
                            'Picklist SKUs': set(),
                            'Shopify SKUs': set()
                        }
                    
                    product_summary[product_type]['Needs (lbs)'] += weight_lbs
                    product_summary[product_type]['Needs (ea)'] += actualqty  # Use calculated actualqty instead of quantity
                    product_summary[product_type]['Shopify SKUs'].add(sku)
                    product_summary[product_type]['Picklist SKUs'].add(picklist_sku)
                    
                    processed_count += 1
                    
                    # Calculate average weight per unit
                    if product_summary[product_type]['Needs (ea)'] > 0:
                        product_summary[product_type]['Weight per Unit (lbs)'] = product_summary[product_type]['Needs (lbs)'] / product_summary[product_type]['Needs (ea)']

            # Processing complete - removed notification messages
            
            # Debug final product summary for honey
            for product_type, data in product_summary.items():
                if 'honey' in product_type.lower():
                    pass  # Removed debug messages
        else:
            pass  # No unfulfilled orders data available for needs calculation

        # Process IN TRANSIT: Vendor orders with transit statuses
        if agg_orders_df is not None and not agg_orders_df.empty:
            # Define in-transit statuses (not yet in inventory but confirmed/shipped)
            in_transit_statuses = ['Confirmed', 'InTransit', 'Delivered']
            
            # Filter for recent orders (last 2 weeks instead of just this week)
            if 'Date_1' in agg_orders_df.columns:
                # Get current week range and extend to last 2 weeks
                current_week_start, current_week_end = get_week_range()
                # Extend to 2 weeks back
                extended_start = current_week_start - timedelta(weeks=1)
                
                agg_orders_df['Date_1'] = pd.to_datetime(agg_orders_df['Date_1'], errors='coerce')
                
                # Filter for recent orders (last 2 weeks)
                recent_orders = agg_orders_df[
                    (agg_orders_df['Date_1'] >= extended_start) & 
                    (agg_orders_df['Date_1'] <= current_week_end)
                ]
            else:
                recent_orders = agg_orders_df
            
            in_transit_processed = 0
            for _, row in recent_orders.iterrows():
                fruit_type = row.get('Fruit', '')
                
                # Process Oxnard orders
                oxnard_status = row.get('Oxnard Status', '')
                oxnard_order = row.get('Oxnard Order', 0)  # Use Oxnard Order instead of Weight Needed
                oxnard_sku = row.get('Oxnard Picklist SKU', '')
                
                # Better order quantity conversion
                try:
                    if pd.notna(oxnard_order) and str(oxnard_order).strip() not in ['', '_', 'nan']:
                        # Extract numeric value from strings like "17 cs", "33 cs", "1179 lb"
                        order_str = str(oxnard_order).replace(',', '').strip()
                        # Extract number using regex (handles "17 cs", "33", "1179 lb", etc.)
                        import re
                        match = re.search(r'(\d+\.?\d*)', order_str)
                        oxnard_order_qty = float(match.group(1)) if match else 0
                    else:
                        oxnard_order_qty = 0
                except (ValueError, TypeError):
                    oxnard_order_qty = 0
                
                # Oxnard Order is already in pounds, no conversion needed
                oxnard_weight = oxnard_order_qty  # Already in pounds
                
                # Check if this is an in-transit order with valid data
                status_match = pd.notna(oxnard_status) and oxnard_status in in_transit_statuses
                has_valid_data = oxnard_order_qty > 0
                has_fruit = pd.notna(fruit_type) and fruit_type.strip()
                
                if has_fruit and status_match and has_valid_data:
                    product_type = fruit_type  # Use fruit type directly as it's already formatted
                    
                    if product_type not in product_summary:
                        product_summary[product_type] = {
                            'Needs (lbs)': 0,
                            'Needs (ea)': 0,
                            'Inventory (lbs)': 0,
                            'Inventory Coldcart (ea)': 0,
                            'In Transit (lbs)': 0,
                            'In Transit (ea)': 0,
                            'Total Inventory (lbs)': 0,
                            'Total Inventory (ea)': 0,
                            'Difference (lbs)': 0,
                            'Difference (ea)': 0,
                            'Cost ($)': 0,
                            'Weight per Unit (lbs)': 0,
                            'Picklist SKUs': set(),
                            'Shopify SKUs': set()
                        }
                    
                    # Convert lbs to pieces for proper unit tracking
                    oxnard_pieces = 0
                    if oxnard_sku:
                        # Try to convert using the specific SKU
                        oxnard_pieces = get_pieces_from_weight_conversion(oxnard_sku, oxnard_weight)
                    if oxnard_pieces == 0:
                        # Fallback to product type conversion
                        oxnard_pieces = get_pieces_from_product_type_weight(product_type, oxnard_weight)
                    
                    product_summary[product_type]['In Transit (lbs)'] += oxnard_weight
                    product_summary[product_type]['In Transit (ea)'] += oxnard_pieces
                    product_summary[product_type]['Picklist SKUs'].add(f"OX-{oxnard_sku or fruit_type}")
                    in_transit_processed += 1
                
                # Process Wheeling orders
                wheeling_status = row.get('Wheeling Status', '')
                wheeling_order = row.get('Wheeling Order', 0)  # Use Wheeling Order instead of Weight Needed
                wheeling_sku = row.get('Wheeling Picklist SKU', '')
                
                # Better order quantity conversion for Wheeling
                try:
                    if pd.notna(wheeling_order) and str(wheeling_order).strip() not in ['', '_', 'nan']:
                        # Extract numeric value from strings like "70 cs", "34 cs", "1872.0"
                        order_str = str(wheeling_order).replace(',', '').strip()
                        # Extract number using regex (handles "70 cs", "34", "1872.0", etc.)
                        import re
                        match = re.search(r'(\d+\.?\d*)', order_str)
                        wheeling_order_qty = float(match.group(1)) if match else 0
                    else:
                        wheeling_order_qty = 0
                except (ValueError, TypeError):
                    wheeling_order_qty = 0
                
                # Wheeling Order is already in pounds, no conversion needed
                wheeling_weight = wheeling_order_qty  # Already in pounds
                
                # Check if this is an in-transit order with valid data
                status_match = pd.notna(wheeling_status) and wheeling_status in in_transit_statuses
                has_valid_data = wheeling_order_qty > 0
                has_fruit = pd.notna(fruit_type) and fruit_type.strip()
                
                if has_fruit and status_match and has_valid_data:
                    product_type = fruit_type  # Use fruit type directly as it's already formatted
                    
                    if product_type not in product_summary:
                        product_summary[product_type] = {
                            'Needs (lbs)': 0,
                            'Needs (ea)': 0,
                            'Inventory (lbs)': 0,
                            'Inventory Coldcart (ea)': 0,
                            'In Transit (lbs)': 0,
                            'In Transit (ea)': 0,
                            'Total Inventory (lbs)': 0,
                            'Total Inventory (ea)': 0,
                            'Difference (lbs)': 0,
                            'Difference (ea)': 0,
                            'Cost ($)': 0,
                            'Weight per Unit (lbs)': 0,
                            'Picklist SKUs': set(),
                            'Shopify SKUs': set()
                        }
                    
                    # Convert lbs to pieces for proper unit tracking
                    wheeling_pieces = 0
                    if wheeling_sku:
                        # Try to convert using the specific SKU
                        wheeling_pieces = get_pieces_from_weight_conversion(wheeling_sku, wheeling_weight)
                    if wheeling_pieces == 0:
                        # Fallback to product type conversion
                        wheeling_pieces = get_pieces_from_product_type_weight(product_type, wheeling_weight)
                    
                    product_summary[product_type]['In Transit (lbs)'] += wheeling_weight
                    product_summary[product_type]['In Transit (ea)'] += wheeling_pieces
                    product_summary[product_type]['Picklist SKUs'].add(f"WH-{wheeling_sku or fruit_type}")
                    in_transit_processed += 1

            logging.debug(f"Total in-transit processed: {in_transit_processed}")

        # Process INVENTORY: ColdCart API ONLY (accurate live data)
        # Process ColdCart inventory (live API) - aggregate by SKU first
        if coldcart_df is not None and not coldcart_df.empty:
            # Aggregate ColdCart data by SKU to get total available quantities
            # This properly handles positive and negative batch quantities
            coldcart_aggregated = coldcart_df.groupby('Sku')['AvailableQty'].sum().reset_index()
            
            for _, row in coldcart_aggregated.iterrows():
                sku = row.get('Sku', '')
                total_available_qty = row.get('AvailableQty', 0)
                
                # Convert to numeric
                total_available_qty = pd.to_numeric(total_available_qty, errors='coerce')
                
                # Skip items with zero or negative total inventory
                if pd.isna(total_available_qty) or total_available_qty <= 0:
                    continue
                
                # Get product type from pieces conversion
                product_type = get_product_type_from_pieces_sku(sku)
                
                # Get weight conversion from pieces_df
                weight_lbs = get_weight_from_pieces_conversion(sku, total_available_qty)
                
                if product_type not in product_summary:
                    product_summary[product_type] = {
                        'Needs (lbs)': 0,
                        'Needs (ea)': 0,
                        'Inventory (lbs)': 0,
                        'Inventory Coldcart (ea)': 0,
                        'In Transit (lbs)': 0,
                        'In Transit (ea)': 0,
                        'Total Inventory (lbs)': 0,
                        'Total Inventory (ea)': 0,
                        'Difference (lbs)': 0,
                        'Difference (ea)': 0,
                        'Cost ($)': 0,
                        'Weight per Unit (lbs)': 0,
                        'Picklist SKUs': set(),
                        'Shopify SKUs': set()
                    }
                
                product_summary[product_type]['Inventory (lbs)'] += weight_lbs
                product_summary[product_type]['Inventory Coldcart (ea)'] += total_available_qty
                product_summary[product_type]['Picklist SKUs'].add(sku)

        # DISABLED: Process Oxnard static inventory 
        # Commented out to prevent double-counting since ColdCart API already includes all warehouses
        # if oxnard_df is not None and not oxnard_df.empty:
        #     for _, row in oxnard_df.iterrows():
        #         product_name = row.get('PRODUCT', '')
        #         total_weight = row.get('Total Weight', 0)
        #         status = row.get('STATUS', 'Good')
        #         
        #         # Convert to numeric
        #         total_weight = pd.to_numeric(total_weight, errors='coerce')
        #         
        #         if pd.isna(total_weight) or total_weight <= 0 or status != 'Good':
        #             continue
        #         
        #         # Use the product name as product type
        #         if product_name and not pd.isna(product_name):
        #             product_type = str(product_name).strip()
        #         
        #         if product_type not in product_summary:
        #             product_summary[product_type] = {
        #                 'Needs (lbs)': 0,
        #                 'Needs (units)': 0,
        #                 'Inventory (lbs)': 0,
        #                 'Inventory Coldcart (ea)': 0,
        #                 'In Transit (lbs)': 0,
        #                 'In Transit (units)': 0,
        #                 'Total Inventory (lbs)': 0,
        #                 'Total Inventory (units)': 0,
        #                 'Difference (lbs)': 0,
        #                 'Difference (units)': 0,
        #                 'Cost ($)': 0,
        #                 'Weight per Unit (lbs)': 0,
        #                 'Picklist SKUs': set(),
        #                 'Shopify SKUs': set()
        #             }
        #         
        #         product_summary[product_type]['Inventory (lbs)'] += total_weight
        #         product_summary[product_type]['Inventory Coldcart (ea)'] += 1  # Count as 1 product unit
        #         product_summary[product_type]['Picklist SKUs'].add(product_name)

        # DISABLED: Process Wheeling static inventory 
        # Commented out to prevent double-counting since ColdCart API already includes all warehouses
        # if wheeling_df is not None and not wheeling_df.empty:
        #     for _, row in wheeling_df.iterrows():
        #         product_name = row.get('PRODUCT', '')
        #         total_weight = row.get('Total Weight', 0)
        #         status = row.get('STATUS', 'Good')
        #         
        #         # Convert to numeric
        #         total_weight = pd.to_numeric(total_weight, errors='coerce')
        #         
        #         if pd.isna(total_weight) or total_weight <= 0 or status != 'Good':
        #             continue
        #         
        #         # Use the product name as product type
        #         if product_name and not pd.isna(product_name):
        #             product_type = str(product_name).strip()
        #         
        #         if product_type not in product_summary:
        #             product_summary[product_type] = {
        #                 'Needs (lbs)': 0,
        #                 'Needs (units)': 0,
        #                 'Inventory (lbs)': 0,
        #                 'Inventory Coldcart (ea)': 0,
        #                 'In Transit (lbs)': 0,
        #                 'In Transit (units)': 0,
        #                 'Total Inventory (lbs)': 0,
        #                 'Total Inventory (units)': 0,
        #                 'Difference (lbs)': 0,
        #                 'Difference (units)': 0,
        #                 'Cost ($)': 0,
        #                 'Weight per Unit (lbs)': 0,
        #                 'Picklist SKUs': set(),
        #                 'Shopify SKUs': set()
        #             }
        #         
        #         product_summary[product_type]['Inventory (lbs)'] += total_weight
        #         product_summary[product_type]['Inventory Coldcart (ea)'] += 1  # Count as 1 product unit
        #         product_summary[product_type]['Picklist SKUs'].add(product_name)

        # Process COST from orders_new (get LATEST cost per fruit type, not sum)
        if orders_new_df is not None and not orders_new_df.empty:
            # Sort by date to get the most recent entries first
            orders_new_df_sorted = orders_new_df.copy()
            
            # Ensure date column is datetime for proper sorting
            orders_new_df_sorted.iloc[:, 0] = pd.to_datetime(orders_new_df_sorted.iloc[:, 0], errors='coerce')
            
            # Sort by date descending (most recent first)
            orders_new_df_sorted = orders_new_df_sorted.sort_values(orders_new_df_sorted.columns[0], ascending=False, na_position='last')
            
            # Track which product types we've already processed (to get only the latest)
            processed_product_types = set()
            
            for _, row in orders_new_df_sorted.iterrows():
                # Columns: invoice date, Aggregator/Vendor, Product Type, Price per lb, Actual Total Cost
                invoice_date = row.iloc[0] if len(row) > 0 else None  # Invoice date
                product_type = row.iloc[2] if len(row) > 2 else None  # Product Type column
                price_per_lb = row.iloc[3] if len(row) > 3 else 0     # Price per lb column
                total_cost = row.iloc[4] if len(row) > 4 else 0       # Actual Total Cost column
                
                if pd.isna(product_type) or not product_type:
                    continue
                
                product_type = str(product_type).strip()
                
                # Skip if we already processed this product type (we want only the latest)
                if product_type in processed_product_types:
                    continue
                
                # Convert cost to numeric
                total_cost = pd.to_numeric(total_cost, errors='coerce')
                price_per_lb = pd.to_numeric(price_per_lb, errors='coerce')
                
                if pd.isna(total_cost):
                    total_cost = 0
                if pd.isna(price_per_lb):
                    price_per_lb = 0
                
                # Create product summary entry if it doesn't exist
                if product_type not in product_summary:
                    product_summary[product_type] = {
                        'Needs (lbs)': 0,
                        'Needs (ea)': 0,
                        'Inventory (lbs)': 0,
                        'Inventory Coldcart (ea)': 0,
                        'In Transit (lbs)': 0,
                        'In Transit (ea)': 0,
                        'Total Inventory (lbs)': 0,
                        'Total Inventory (ea)': 0,
                        'Difference (lbs)': 0,
                        'Difference (ea)': 0,
                        'Cost ($)': 0,
                        'Latest Price ($/lb)': 0,
                        'Latest Invoice Date': None,
                        'Weight per Unit (lbs)': 0,
                        'Picklist SKUs': set(),
                        'Shopify SKUs': set()
                    }
                
                # Use the LATEST total cost for this product type (not sum)
                product_summary[product_type]['Cost ($)'] = total_cost
                product_summary[product_type]['Latest Price ($/lb)'] = price_per_lb
                product_summary[product_type]['Latest Invoice Date'] = invoice_date
                
                # Mark this product type as processed
                processed_product_types.add(product_type)

        # Calculate totals and differences for each product type
        for product_type in product_summary:
            needs_lbs = product_summary[product_type]['Needs (lbs)']
            needs_ea = product_summary[product_type]['Needs (ea)']
            inventory_lbs = product_summary[product_type]['Inventory (lbs)']
            inventory_ea = product_summary[product_type]['Inventory Coldcart (ea)']
            in_transit_lbs = product_summary[product_type]['In Transit (lbs)']
            in_transit_ea = product_summary[product_type]['In Transit (ea)']
            
            total_inventory_lbs = inventory_lbs + in_transit_lbs
            total_inventory_ea = inventory_ea + in_transit_ea
            
            # Properly format numbers - integers for ea, clean decimals for weights
            product_summary[product_type]['Needs (lbs)'] = round(needs_lbs, 1) if needs_lbs % 1 != 0 else int(needs_lbs)
            product_summary[product_type]['Needs (ea)'] = int(round(needs_ea, 0))
            product_summary[product_type]['Inventory (lbs)'] = round(inventory_lbs, 1) if inventory_lbs % 1 != 0 else int(inventory_lbs)
            product_summary[product_type]['Inventory Coldcart (ea)'] = int(round(inventory_ea, 0))
            product_summary[product_type]['In Transit (lbs)'] = round(in_transit_lbs, 1) if in_transit_lbs % 1 != 0 else int(in_transit_lbs)
            product_summary[product_type]['In Transit (ea)'] = int(round(in_transit_ea, 0))
            product_summary[product_type]['Total Inventory (lbs)'] = round(total_inventory_lbs, 1) if total_inventory_lbs % 1 != 0 else int(total_inventory_lbs)
            product_summary[product_type]['Total Inventory (ea)'] = int(round(total_inventory_ea, 0))
            
            difference_lbs = total_inventory_lbs - needs_lbs
            difference_ea = total_inventory_ea - needs_ea
            product_summary[product_type]['Difference (lbs)'] = round(difference_lbs, 1) if difference_lbs % 1 != 0 else int(difference_lbs)
            product_summary[product_type]['Difference (ea)'] = int(round(difference_ea, 0))
            product_summary[product_type]['Cost ($)'] = round(product_summary[product_type]['Cost ($)'], 2)
            
            # Update weight per unit calculation if we have both pounds and ea
            if total_inventory_ea > 0 and total_inventory_lbs > 0:
                weight_per_unit = total_inventory_lbs / total_inventory_ea
                product_summary[product_type]['Weight per Unit (lbs)'] = round(weight_per_unit, 2) if weight_per_unit % 1 != 0 else int(weight_per_unit)

        # Convert to DataFrame
        if product_summary:
            # Convert sets to comma-separated strings before creating DataFrame
            for product_type in product_summary:
                product_summary[product_type]['Picklist SKUs'] = ', '.join(sorted(product_summary[product_type]['Picklist SKUs'])) if product_summary[product_type]['Picklist SKUs'] else ''
                product_summary[product_type]['Shopify SKUs'] = ', '.join(sorted(product_summary[product_type]['Shopify SKUs'])) if product_summary[product_type]['Shopify SKUs'] else ''
            
            df = pd.DataFrame.from_dict(product_summary, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Product Type'}, inplace=True)
            
            # Format the Latest Invoice Date column
            if 'Latest Invoice Date' in df.columns:
                df['Latest Invoice Date'] = df['Latest Invoice Date'].apply(
                    lambda x: x.strftime('%m/%d/%Y') if pd.notna(x) and x is not None else ''
                )
            
            # Convert ALL numeric columns to float for proper formatting and sorting
            numeric_cols = ['Needs (lbs)', 'Needs (ea)', 'Inventory (lbs)', 'Inventory Coldcart (ea)', 
                          'In Transit (lbs)', 'In Transit (ea)', 'Total Inventory (lbs)', 
                          'Total Inventory (ea)', 'Difference (lbs)', 'Difference (ea)', 
                          'Weight per Unit (lbs)', 'Cost ($)', 'Latest Price ($/lb)']
            
            # First convert all to numeric and ensure float type
            for col in numeric_cols:
                if col in df.columns:
                    # Convert to numeric and keep as float (not int) for consistent formatting
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
            
            # Calculate Total Latest Cost = Total Inventory (lbs) × Latest Price ($/lb)
            if 'Total Inventory (lbs)' in df.columns and 'Latest Price ($/lb)' in df.columns:
                df['Total Latest Cost ($)'] = (df['Total Inventory (lbs)'] * df['Latest Price ($/lb)']).astype(float)
            
            # Keep ALL numeric columns as float for consistent formatting - no rounding to int
            # The formatting will be handled by Streamlit's NumberColumn format parameter
            for col in numeric_cols + ['Total Latest Cost ($)']:
                if col in df.columns:
                    # Convert to float to ensure consistent 2-decimal precision formatting
                    df[col] = df[col].astype(float)
            
            return df
        else:
            return pd.DataFrame()

    # Display the FAST Product Type Summary
    with st.container():
        st.subheader("🚀 Need/Have Summary")
        st.markdown("**⚡ Real-time calculation**: Product Type level summary")
        
        # Add refresh button for unfulfilled orders
        if st.button("🔄 Refresh Shopify Unfulfilled Orders", key="refresh_unfulfilled"):
            with st.spinner("Refreshing unfulfilled orders..."):
                try:
                    # Load last 7 days of unfulfilled orders
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)
                    
                    orders_df, unfulfilled_df = load_shopify_orders(start_date=start_date, end_date=end_date, force_refresh=True)
                    
                    if unfulfilled_df is not None and not unfulfilled_df.empty:
                        pass  # Refreshed successfully
                    else:
                        pass  # No unfulfilled orders found
                except Exception as e:
                    st.error(f"⚠️ Could not refresh Shopify data: {str(e)}")
        
        try:
            product_summary_df = create_fast_product_type_summary()
            
            if not product_summary_df.empty:
                # Add filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    product_search = st.text_input("🔍 Search Product Type", placeholder="Type to search...", key="fast_summary_search")
                
                with col2:
                    show_only_shortages = st.checkbox("Show only shortages", value=False, key="fast_summary_shortages")
                
                with col3:
                    show_all_columns = st.checkbox("Show all columns", value=False, key="fast_summary_all_columns")
                
                # Apply filters
                display_df = product_summary_df.copy()
                
                # Helper function to convert values to numbers for calculations (now values are already numeric)
                def parse_number(val):
                    if pd.isna(val) or val == '':
                        return 0
                    # Values are already numeric, just ensure float conversion
                    return float(val)
                
                if product_search:
                    mask = display_df['Product Type'].str.contains(product_search, case=False, na=False)
                    display_df = display_df[mask]
                
                if show_only_shortages:
                    # Values are already numeric for comparison
                    display_df = display_df[display_df['Difference (lbs)'] < 0]
                
                # Sort by Needs (lbs) descending (highest needs first) - values are already numeric
                display_df = display_df.sort_values('Needs (lbs)', ascending=False)
                
                # Style the dataframe - work with numeric values
                def color_difference(val):
                    try:
                        # Values are already numeric
                        if val < 0:
                            return 'background-color: #ffcdd2; color: #c62828; font-weight: bold'
                        elif val == 0:
                            return 'background-color: #fff3e0; color: #f57c00'
                        else:
                            return 'background-color: #c8e6c9; color: #2e7d32'
                    except:
                        return ''
                
                # Determine which columns to show
                if show_all_columns:
                    # Show all columns except SKU columns
                    all_columns = display_df.columns.tolist()
                    columns_to_show = [col for col in all_columns if col not in ['Picklist SKUs', 'Shopify SKUs']]
                    column_config = {
                        'Product Type': st.column_config.TextColumn('Product Type', width=200),
                        'Needs (lbs)': st.column_config.NumberColumn('Needs LB'),
                        'Needs (ea)': st.column_config.NumberColumn('Needs (ea)'),
                        'Inventory (lbs)': st.column_config.NumberColumn('Inventory Coldcart LB'),
                        'Inventory Coldcart (ea)': st.column_config.NumberColumn('Inventory Coldcart (ea)'),
                        'In Transit (lbs)': st.column_config.NumberColumn('Inventory In Transit LB'),
                        'In Transit (ea)': st.column_config.NumberColumn('In Transit (ea)'),
                        'Total Inventory (lbs)': st.column_config.NumberColumn('Total Inventory LB'),
                        'Total Inventory (ea)': st.column_config.NumberColumn('Total Inventory (ea)'),
                        'Difference (lbs)': st.column_config.NumberColumn('Difference'),
                        'Difference (ea)': st.column_config.NumberColumn('Difference (ea)'),
                        'Weight per Unit (lbs)': st.column_config.NumberColumn('Weight per Unit (lbs)'),
                        'Cost ($)': st.column_config.NumberColumn('Latest Total Cost', format="$%.0f"),
                        'Latest Price ($/lb)': st.column_config.NumberColumn('Latest Price ($/lb)', format="$%.0f"),
                        'Total Latest Cost ($)': st.column_config.NumberColumn('Total Latest Cost ($)', format="$%.0f"),
                        'Latest Invoice Date': st.column_config.TextColumn('Latest Invoice Date')
                    }
                else:
                    # Show only essential columns by default
                    essential_columns = [
                        'Product Type',
                        'Needs (lbs)',
                        'Inventory (lbs)', 
                        'In Transit (lbs)',
                        'Total Inventory (lbs)',
                        'Difference (lbs)'
                    ]
                    columns_to_show = [col for col in essential_columns if col in display_df.columns]
                    column_config = {
                        'Product Type': st.column_config.TextColumn('Product Type', width=200),
                        'Needs (lbs)': st.column_config.NumberColumn('Needs LB'),
                        'Inventory (lbs)': st.column_config.NumberColumn('Inventory Coldcart LB'),
                        'In Transit (lbs)': st.column_config.NumberColumn('Inventory In Transit LB'),
                        'Total Inventory (lbs)': st.column_config.NumberColumn('Total Inventory LB'),
                        'Difference (lbs)': st.column_config.NumberColumn('Difference')
                    }
                
                # Filter display dataframe to show only selected columns
                display_df_filtered = display_df[columns_to_show]
                
                # Apply styling with precision=0 formatting for all numeric columns
                numeric_columns = [col for col in display_df_filtered.columns if col != 'Product Type' and col != 'Latest Invoice Date']
                styled_df = display_df_filtered.style.map(color_difference, subset=['Difference (lbs)']).format(precision=0, subset=numeric_columns)
                
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )
                
                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    total_products = len(display_df)
                    st.metric("Product Types", f"{total_products:,}")
                
                with col2:
                    # Values are already numeric for comparison
                    shortage_products = len(display_df[display_df['Difference (lbs)'] < 0])
                    st.metric("Shortages", f"{shortage_products:,}")
                
                with col3:
                    # Values are already numeric for sum
                    total_needs = display_df['Needs (lbs)'].sum()
                    st.metric("Total Needs", f"{total_needs:,.0f} lbs")
                
                with col4:
                    # Values are already numeric for sum
                    total_inventory = display_df['Total Inventory (lbs)'].sum()
                    st.metric("Total Available", f"{total_inventory:,.0f} lbs")
                
                with col5:
                    # Use the calculated Total Latest Cost column (Total Inventory × Latest Price)
                    if 'Total Latest Cost ($)' in display_df.columns:
                        total_cost = display_df['Total Latest Cost ($)'].sum()
                    elif 'Cost ($)' in display_df.columns:
                        # Fallback to old cost column if new one doesn't exist
                        total_cost = display_df['Cost ($)'].sum()
                    else:
                        total_cost = 0
                    st.metric("Total Value at Latest Prices", f"${total_cost:,.0f}")
                    
            else:
                st.info("No data available for summary. Make sure all data sources are loaded.")
                
                # Debug information
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    unfulfilled_df = st.session_state.get('unfulfilled_orders_df')
                    if unfulfilled_df is not None and not unfulfilled_df.empty:
                        pass  # Has Shopify data
                    else:
                        pass  # No Shopify data
                
                with col2:
                    inventory_count = 0
                    sources = []
                    coldcart_df = st.session_state.get('coldcart_inventory')
                    if coldcart_df is not None and not coldcart_df.empty:
                        inventory_count += len(coldcart_df)
                        sources.append("ColdCart")
                    
                    oxnard_df = st.session_state.get('inventory_data', {}).get('oxnard')
                    if oxnard_df is not None and not oxnard_df.empty:
                        inventory_count += len(oxnard_df)
                        sources.append("Oxnard")
                    
                    wheeling_df = st.session_state.get('inventory_data', {}).get('wheeling')
                    if wheeling_df is not None and not wheeling_df.empty:
                        inventory_count += len(wheeling_df)
                        sources.append("Wheeling")
                    
                    if inventory_count > 0:
                        pass  # Has inventory data
                    else:
                        pass  # No inventory data
                
                with col3:
                    agg_orders_df = st.session_state.get('agg_orders_df')
                    if agg_orders_df is not None and not agg_orders_df.empty:
                        pass  # Has vendor orders
                    else:
                        pass  # No vendor orders
                
                with col4:
                    sku_mappings_file = 'airtable_sku_mappings.json'
                    total_skus = 0
                    if os.path.exists(sku_mappings_file):
                        try:
                            with open(sku_mappings_file, 'r') as f:
                                sku_mappings = json.load(f)
                                
                            if isinstance(sku_mappings, dict):
                                for warehouse in ['Oxnard', 'Wheeling']:
                                    if warehouse in sku_mappings:
                                        if 'singles' in sku_mappings[warehouse]:
                                            total_skus += len(sku_mappings[warehouse]['singles'])
                                        if 'bundles' in sku_mappings[warehouse]:
                                            total_skus += len(sku_mappings[warehouse]['bundles'])
                        except Exception:
                            pass
                    
                    if total_skus > 0:
                        pass  # Has SKU mappings
                    else:
                        pass  # No SKU mappings
                        
        except Exception as e:
            st.error(f"Error creating summary: {str(e)}")
            logging.error(f"Error in create_fast_product_type_summary: {e}")

    with st.expander("📋 Data Sources & Descriptions", expanded=False):
        st.markdown("""
        ### Data Sources Overview
        
        **📦 Needs (Customer Orders)**
        - **Source**: Shopify Admin API (Unfulfilled Orders)
        - **Content**: Customer orders awaiting fulfillment
        - **Period**: Last 7 days from today (unfulfilled orders created within this period)
        - **Date Range**: {datetime.now().strftime('%m/%d/%Y')} back to {(datetime.now() - timedelta(days=7)).strftime('%m/%d/%Y')}
        - **Units**: Both pounds (lbs) and pieces (ea)
        - **Filter**: Excludes ONLY e.ripening_guide and expedited-shipping. INCLUDES all gifts, $0 items, and physical products
        - **Processing**: Bundles are decomposed into individual components
        
        **🏪 Current Inventory**
        - **Source**: ColdCart Live API + Google Sheets inventory data
        - **Content**: Available inventory in warehouses (real-time)
        - **Period**: Current moment (live data as of {datetime.now().strftime('%m/%d/%Y %I:%M %p')})
        - **Conversion**: ColdCart pieces → lbs using conversion tables
        - **Locations**: Oxnard & Wheeling facilities
        - **Frequency**: Real-time data when available
        
        **🚛 In Transit Inventory**
        - **Source**: Google Sheets (Aggregation Orders from vendors)
        - **Content**: Confirmed/shipped orders from vendors
        - **Statuses**: Confirmed, InTransit, Delivered
        - **Period**: Last 2 weeks of vendor orders (fixed lookback)
        - **Date Range**: {(datetime.now() - timedelta(weeks=2)).strftime('%m/%d/%Y')} to {datetime.now().strftime('%m/%d/%Y')}
        - **Conversion**: lbs → pieces for consistent unit tracking
        
        **💰 Cost Data**
        - **Source**: Google Sheets (orders_new - Latest Invoices)
        - **Content**: Most recent price per lb by product type
        - **Period**: Latest available invoice per product type (no time limit)
        - **Method**: Latest invoice date takes precedence (not averaged)
        - **Display**: Shows latest total cost and price per lb
        
        **🔄 Unit Conversions & Mappings**
        - **ColdCart**: pieces → lbs (using INPUT_picklist_sku table)
        - **In Transit**: lbs → pieces (for consistent tracking)
        - **SKU Mappings**: Shopify SKUs ↔ Warehouse Pick SKUs
        - **Bundles**: Automatic decomposition using bundle mappings
        - **Product Types**: SKU → Product Type classification
        
        ### Key Metrics Explained
        - **Difference**: Total Inventory - Needs (negative = shortage)
        - **Total Inventory**: Current + In Transit inventory
        - **Weight per Unit**: Calculated average from mappings
        - **Related SKUs**: Both Shopify and warehouse SKUs tracked
        """)
    

    st.markdown("---")
    # ------------------- END: FAST FRUIT SUMMARY -------------------

    # Create sidebar filters
    filters = create_sidebar_filters(df)
    
    # Apply global filters to main dataframe
    df_filtered = apply_global_filters(df, filters)

    # Display Picklist Data - MOVED TO FIRST POSITION
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
            
            # Create column configuration for all numeric columns with renamed labels
            column_config = {
                'Product Type': st.column_config.TextColumn('Product Type')
            }
            
            # Define column name mappings for display
            column_name_mappings = {
                'Total Weight': 'Needed for Shopify orders',
                'Inventory': 'Coldcard inventory', 
                'Total': 'Total Needed inventory',
                'Total Needs (LBS)': 'GHF NEEDS'
            }
            
            # Add number formatting for all numeric columns
            numeric_cols = display_df.columns.difference(['Product Type'])
            for col in numeric_cols:
                # Use renamed label if available, otherwise use original column name
                display_name = column_name_mappings.get(col, col)
                column_config[col] = st.column_config.NumberColumn(
                    display_name,
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

    # Display Week over Week Data
    with st.expander("📊 Week over Week Analysis", expanded=False):
        wow_df = st.session_state.get('wow_df')
        if wow_df is not None and not wow_df.empty:

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
            
            # Add metric filter with logical groupings
            if selected_ranges:  # Only show metric filter if date ranges are selected
                all_metrics = sorted(wow_df['Metric'].unique())
                
                # Define the exact metric order as specified
                preferred_metric_order = [
                    "Delivered Weight - OX",
                    "Delivered Weight - WH", 
                    "TOTAL delivered weight ( )",
                    "Estimated cost - OX",
                    "Estimated cost - WH",
                    "TOTAL fruit cost",
                    "Last week inventory - OX",
                    "Last week inventory - WH",
                    "Total inventory ( )",
                    "Estimated Inventory cost - OX",
                    "Estimated Inventory cost - WH",
                    "Total inventory cost",
                    "TOTAL delivered + last inventory ( )",
                    "TOTAL delivered + last inventory cost",
                    "EOW Inventory - OX",
                    "EOW Inventory - WH",
                    "TOTAL EOW Inventory",
                    "TOTAL EOW Inventory cost",
                    "New Culls - OX",
                    "New Culls - WH",
                    "TOTAL New culls",
                    "New cull cost",
                    "Aged Culls - OX",
                    "Aged Culls - WH",
                    "TOTAL Aged culls",
                    "Aged cull cost",
                    "TOTAL CULLS",
                    "TOTAL CULLS Cost",
                    "CULL RATE",
                    "INVENTORY LEFT",
                    "TOTAL FULFILLED WEIGHT ( )",
                    "FULFILLED FRUIT RATE",
                    "FULFILLED NET SALES",
                    "TOTAL FULFILLED cost",
                    "FULFILLED FRUIT EXPENSE %",
                    "OVER-UNDER PACK ",
                    "OVERPACK Cost estimated",
                    "OVERPACK RATE",
                    "TRANSPORT Cost",
                    "TRANSPORT %",
                    "AGG INVOICED",
                    "AGG INVOICED %"
                ]
                
                # Search box for metrics
                search_term = st.text_input(
                    "🔍 Search Metrics",
                    placeholder="Type to search metrics (e.g., 'cull', 'inventory', 'rate')...",
                    help="Search through metric names to quickly find what you need"
                )
                
                # Quick action buttons for metric selection
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("✅ All Metrics", help="Select all metrics"):
                        st.session_state['metric_action'] = 'select_all'
                        st.rerun()
                with col2:
                    if st.button("📊 Rates Only", help="Show only rate/percentage metrics"):
                        st.session_state['metric_action'] = 'rates_only'
                        st.rerun()
                with col3:
                    if st.button("💰 Costs Only", help="Show only cost/financial metrics"):
                        st.session_state['metric_action'] = 'costs_only'
                        st.rerun()
                with col4:
                    if st.button("📦 Weights Only", help="Show only weight/inventory metrics"):
                        st.session_state['metric_action'] = 'weights_only'
                        st.rerun()
                
                # Apply search filter first
                if search_term:
                    # Filter metrics based on search term
                    search_filtered_metrics = [
                        m for m in preferred_metric_order 
                        if m in all_metrics and search_term.lower() in m.lower()
                    ]
                else:
                    # No search, use all available metrics
                    search_filtered_metrics = [m for m in preferred_metric_order if m in all_metrics]
                
                # Handle quick actions and determine selected metrics
                action = st.session_state.get('metric_action', None)
                
                if action == 'select_all':
                    selected_metrics = search_filtered_metrics
                elif action == 'rates_only':
                    selected_metrics = [m for m in search_filtered_metrics if any(word in m.lower() for word in ['rate', '%', 'remaining'])]
                elif action == 'costs_only':
                    selected_metrics = [m for m in search_filtered_metrics if any(word in m.lower() for word in ['cost', 'expense', 'invoiced', 'sales'])]
                elif action == 'weights_only':
                    selected_metrics = [m for m in search_filtered_metrics if any(word in m.lower() for word in ['weight', 'inventory', 'delivered', 'fulfilled', 'waste', 'pack'])]
                else:
                    # Default to search filtered metrics (or all if no search)
                    selected_metrics = search_filtered_metrics
                
                # Clear the action after use
                if action:
                    del st.session_state['metric_action']
                
                # Show selection summary
                if search_term:
                    if selected_metrics:
                        st.info(f"🔍 **{len(selected_metrics)} metrics found** for '{search_term}' | Click buttons above to filter further")
                    else:
                        st.warning(f"🔍 No metrics found for '{search_term}'. Try a different search term.")
                else:
                    st.info(f"📈 **{len(selected_metrics)} metrics selected** | Use search box or buttons above to filter")
                
            if selected_ranges and selected_metrics:
                filtered_df = wow_df[
                    (wow_df['Date Range'].isin(selected_ranges)) & 
                    (wow_df['Metric'].isin(selected_metrics))
                ]
                
                # Sort ranges chronologically (older to newer)
                sorted_ranges = sorted(selected_ranges, key=parse_date_for_sorting)
                
                # Use the preferred metric order (exact order as specified)
                available_metrics_in_data = filtered_df['Metric'].unique()
                original_metric_order = [m for m in preferred_metric_order if m in available_metrics_in_data]
                
                # Pivot the data for better comparison
                pivot_df = filtered_df.pivot(
                    index='Metric',
                    columns='Date Range',
                    values='Value'
                ).reset_index()
                
                # Restore original metric order from the Google Sheet
                pivot_df['Metric'] = pd.Categorical(pivot_df['Metric'], categories=original_metric_order, ordered=True)
                pivot_df = pivot_df.sort_values('Metric').reset_index(drop=True)
                
                # Reorder columns to show previous week first
                column_order = ['Metric'] + sorted_ranges
                pivot_df = pivot_df[column_order]
                
                # Format values with appropriate units based on metric type
                display_pivot_df = pivot_df.copy()
                
                # Convert numeric columns to object type to allow string formatting
                for date_range in sorted_ranges:
                    if date_range in display_pivot_df.columns:
                        display_pivot_df[date_range] = display_pivot_df[date_range].astype('object')
                
                for idx, row in display_pivot_df.iterrows():
                    metric = row['Metric']
                    
                    # Get formatting info for this metric from the original data
                    metric_info = filtered_df[filtered_df['Metric'] == metric].iloc[0] if not filtered_df[filtered_df['Metric'] == metric].empty else None
                    
                    if metric_info is not None:
                        is_percentage = metric_info['Is Percentage']
                        is_currency = metric_info['Is Currency']
                        
                        # Format all date range columns for this metric
                        for date_range in sorted_ranges:
                            if date_range in display_pivot_df.columns:
                                value = display_pivot_df.at[idx, date_range]
                                if pd.notna(value):
                                    if is_percentage:
                                        display_pivot_df.at[idx, date_range] = f"{value:.1f}%"
                                    elif is_currency:
                                        display_pivot_df.at[idx, date_range] = f"${value:,.0f}"
                                    else:
                                        # If it's just a number (no $ or %), it's weight in lb
                                        display_pivot_df.at[idx, date_range] = f"{value:,.0f} lb"
                
                # Calculate week-over-week changes if we have at least 2 ranges selected
                if len(sorted_ranges) >= 2:
                    for i in range(len(sorted_ranges)-1):
                        prev_range = sorted_ranges[i]
                        current_range = sorted_ranges[i+1]
                        formatted_prev = format_date_range(prev_range)
                        formatted_curr = format_date_range(current_range)
                        col_name = f"Change ({formatted_prev} → {formatted_curr})"
                        
                        # Calculate changes using original numeric values
                        changes = pivot_df.apply(
                            lambda row: calculate_wow_change(
                                row[prev_range], 
                                row[current_range]
                            ) if pd.notna(row[prev_range]) and pd.notna(row[current_range]) else None,
                            axis=1
                        )
                        
                        # Format changes as percentages with arrows and add to display dataframe
                        def format_change_with_arrow(x):
                            if pd.isna(x):
                                return ""
                            if x > 0:
                                return f"↗️ +{x:.1f}%"
                            elif x < 0:
                                return f"↘️ {x:.1f}%"
                            else:
                                return f"→ {x:.1f}%"
                        
                        display_pivot_df[col_name] = changes.apply(format_change_with_arrow).astype('object')
                
                # WoW Metric Name Mapping for Better Display Names
                wow_metric_mapping = {
                    # Weight & Delivery Metrics
                    "Delivered Weight - OX": "Weight Delivered to Oxnard",
                    "Delivered Weight - WH": "Weight Delivered to Wheeling", 
                    "TOTAL delivered weight ( )": "Total Weight Delivered (Both FC)",
                    
                    # Cost Metrics
                    "Estimated cost - OX": "Estimated Fruit Cost - Oxnard",
                    "Estimated cost - WH": "Estimated Fruit Cost - Wheeling",
                    "TOTAL fruit cost": "Total Estimated Fruit Cost",
                    
                    # Previous Inventory
                    "Last week inventory - OX": "Previous Week Inventory - Oxnard",
                    "Last week inventory - WH": "Previous Week Inventory - Wheeling",
                    "Total inventory ( )": "Total Previous Week Inventory",
                    "Estimated Inventory cost - OX": "Previous Inventory Cost - Oxnard",
                    "Estimated Inventory cost - WH": "Previous Inventory Cost - Wheeling", 
                    "Total inventory cost": "Total Previous Inventory Cost",
                    
                    # Combined Metrics
                    "TOTAL delivered + last inventory ( )": "Total Available (Delivered + Previous)",
                    "TOTAL delivered + last inventory cost": "Total Available Cost",
                    
                    # End of Week Inventory
                    "EOW Inventory - OX": "Current Week Inventory - Oxnard",
                    "EOW Inventory - WH": "Current Week Inventory - Wheeling",
                    "TOTAL EOW Inventory": "Total Current Week Inventory",
                    "TOTAL EOW Inventory cost": "Current Week Inventory Cost",
                    
                    # Culls (Waste)
                    "New Culls - OX": "New Waste - Oxnard",
                    "New Culls - WH": "New Waste - Wheeling",
                    "TOTAL New culls": "Total New Waste",
                    "New cull cost": "New Waste Cost",
                    "Aged Culls - OX": "Aged Waste - Oxnard", 
                    "Aged Culls - WH": "Aged Waste - Wheeling",
                    "TOTAL Aged culls": "Total Aged Waste",
                    "Aged cull cost": "Aged Waste Cost",
                    "TOTAL CULLS": "Total Waste",
                    "TOTAL CULLS Cost": "Total Waste Cost",
                    
                    # Rate Metrics
                    "CULL RATE": "Waste Rate",
                    "INVENTORY LEFT": "Inventory Remaining Rate",
                    "FULFILLED FRUIT RATE": "Fulfillment Rate",
                    "FULFILLED FRUIT EXPENSE %": "Fruit Cost as % of Sales",
                    "OVERPACK RATE": "Overpack Rate",
                    "TRANSPORT %": "Transport Cost as % of Sales",
                    "AGG INVOICED %": "Invoiced Fruit Cost as % of Sales",
                    
                    # Fulfillment Metrics
                    "TOTAL FULFILLED WEIGHT ( )": "Total Weight Fulfilled",
                    "FULFILLED NET SALES": "Net Sales Fulfilled",
                    "TOTAL FULFILLED cost": "Fulfilled Fruit Cost",
                    
                    # Packing & Transport
                    "OVER-UNDER PACK ": "Over/Under Pack Weight",
                    "OVERPACK Cost estimated": "Estimated Overpack Cost",
                    "TRANSPORT Cost": "Transport Cost",
                    "AGG INVOICED": "Total Invoiced Fruit Cost"
                }
                
                # Apply metric name mapping
                def apply_wow_metric_mapping(metric):
                    """Apply user-friendly names to WoW metrics"""
                    return wow_metric_mapping.get(metric, metric)
                
                # Make metric names more visual with icons
                def add_metric_icon(metric):
                    metric = metric.replace('**', '')  # Remove any existing markdown
                    
                    # Apply the WoW metric mapping first
                    metric = apply_wow_metric_mapping(metric)
                    
                    # Prioritize rate/percentage icon for anything with % or rate
                    if 'rate' in metric.lower() or '%' in metric.lower() or 'left' in metric.lower():
                        return f"📊 {metric}"
                    # Cost icon for anything with cost, expense, sales, invoiced
                    elif 'cost' in metric.lower() or 'expense' in metric.lower() or 'sales' in metric.lower() or 'invoiced' in metric.lower():
                        return f"💰 {metric}"
                    # Weight icon for weight, delivered, fulfilled weight, pack
                    elif 'weight' in metric.lower() or 'delivered' in metric.lower() or ('fulfilled' in metric.lower() and 'weight' in metric.lower()) or 'pack' in metric.lower():
                        return f"📦 {metric}"
                    # Inventory icon for inventory (but not costs)
                    elif 'inventory' in metric.lower():
                        return f"📦 {metric}"
                    # Waste icon for culls
                    elif 'cull' in metric.lower():
                        return f"🗑️ {metric}"
                    # Transport icon
                    elif 'transport' in metric.lower():
                        return f"🚛 {metric}"
                    # Fulfilled (non-weight) gets checkmark
                    elif 'fulfilled' in metric.lower():
                        return f"✅ {metric}"
                    else:
                        return f"📈 {metric}"
                
                display_pivot_df['Metric'] = display_pivot_df['Metric'].apply(add_metric_icon)
                
                # Apply color blocking to group related metrics
                def style_color_blocks(df):
                    def apply_color_blocks(row):
                        metric_name = row['Metric']
                        
                        # Define color blocks for metric groups (updated for new naming)
                        if any(keyword in metric_name for keyword in ['Weight Delivered', 'Total Weight Delivered']):
                            return ['background-color: #f3e5f5; color: #4a148c'] * len(row)  # Light purple - Delivered
                        elif any(keyword in metric_name for keyword in ['Estimated Fruit Cost', 'Total Estimated Fruit Cost']):
                            return ['background-color: #e8f5e8; color: #1b5e20'] * len(row)  # Light green - Costs
                        elif any(keyword in metric_name for keyword in ['Previous Week Inventory', 'Total Previous Week Inventory']):
                            return ['background-color: #fff3e0; color: #e65100'] * len(row)  # Light orange - Previous Inventory
                        elif any(keyword in metric_name for keyword in ['Previous Inventory Cost', 'Total Previous Inventory Cost']):
                            return ['background-color: #e0f2f1; color: #00695c'] * len(row)  # Light teal - Previous Inventory Costs
                        elif any(keyword in metric_name for keyword in ['Total Available', 'Total Available Cost']):
                            return ['background-color: #e1f5fe; color: #01579b'] * len(row)  # Light cyan - Combined
                        elif any(keyword in metric_name for keyword in ['Current Week Inventory', 'Total Current Week Inventory']):
                            return ['background-color: #fce4ec; color: #880e4f'] * len(row)  # Light pink - Current Inventory
                        elif any(keyword in metric_name for keyword in ['New Waste', 'Total New Waste', 'New Waste Cost']):
                            return ['background-color: #ffebee; color: #b71c1c'] * len(row)  # Light red - New Waste
                        elif any(keyword in metric_name for keyword in ['Aged Waste', 'Total Aged Waste', 'Aged Waste Cost']):
                            return ['background-color: #fff8e1; color: #f57f17'] * len(row)  # Light yellow - Aged Waste
                        elif any(keyword in metric_name for keyword in ['Total Waste', 'Waste Rate', 'Inventory Remaining Rate']):
                            return ['background-color: #f1f8e9; color: #33691e'] * len(row)  # Light lime - Waste Summary
                        elif any(keyword in metric_name for keyword in ['Total Weight Fulfilled', 'Fulfillment Rate', 'Net Sales Fulfilled', 'Fruit Cost as % of Sales']):
                            return ['background-color: #e8eaf6; color: #1a237e'] * len(row)  # Light indigo - Fulfillment
                        elif any(keyword in metric_name for keyword in ['Over/Under Pack', 'Overpack']):
                            return ['background-color: #f9fbe7; color: #827717'] * len(row)  # Light lime green - Packing
                        elif any(keyword in metric_name for keyword in ['Transport Cost', 'Invoiced Fruit Cost']):
                            return ['background-color: #efebe9; color: #3e2723'] * len(row)  # Light brown - Transport/Invoicing
                        else:
                            return [''] * len(row)  # No styling for unmatched metrics
                    
                    return df.style.apply(apply_color_blocks, axis=1)
                
                # Display the data
                column_config = {
                    "Metric": st.column_config.TextColumn(
                        "📊 Metric",
                        help="Performance metrics with visual indicators"
                    )
                }
                
                # Add range columns with formatted names (now as text since they include units)
                for range_name in sorted_ranges:
                    formatted_name = format_date_range(range_name)
                    column_config[range_name] = st.column_config.TextColumn(
                        formatted_name,
                        help="Values formatted with appropriate units ($, %, or numeric)"
                    )
                
                # Add change columns if we have multiple ranges
                if len(sorted_ranges) >= 2:
                    for i in range(len(sorted_ranges)-1):
                        prev_range = sorted_ranges[i]
                        current_range = sorted_ranges[i+1]
                        formatted_prev = format_date_range(prev_range)
                        formatted_curr = format_date_range(current_range)
                        change_col = f"Change ({formatted_prev} → {formatted_curr})"
                        column_config[change_col] = st.column_config.TextColumn(
                            f"📈 {change_col}",
                            help="Percentage change between periods (↗️ increase, ↘️ decrease)"
                        )
                
                st.dataframe(
                    style_color_blocks(display_pivot_df),
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )

                # Display summary metrics - Focus on Key Rate Metrics
                st.markdown("### 📊 Key Rate Insights")
                if len(selected_ranges) >= 2:
                    latest_range = sorted_ranges[-1]
                    prev_range = sorted_ranges[-2]
                    
                    # Find rate metrics to display
                    rate_metrics = []
                    for metric in pivot_df['Metric'].tolist():
                        metric_lower = metric.lower()
                        if 'rate' in metric_lower or '%' in metric_lower or 'left' in metric_lower:
                            rate_metrics.append(metric)
                    
                    # Show top rate metrics (up to 4) - updated for new naming
                    priority_rates = []
                    priority_names = ['Waste Rate', 'Fulfillment Rate', 'Overpack Rate', 'Invoiced Fruit Cost as % of Sales', 'Transport Cost as % of Sales', 'Fruit Cost as % of Sales']
                    for priority in priority_names:
                        for metric in rate_metrics:
                            # Check if the mapped metric name contains the priority
                            mapped_metric = apply_wow_metric_mapping(metric.replace('📊 ', '').replace('📈 ', ''))
                            if priority in mapped_metric:
                                priority_rates.append(metric)
                                break
                    
                    # Take up to 4 rate metrics
                    metrics_to_show = priority_rates[:4] if priority_rates else rate_metrics[:4]
                    
                    if metrics_to_show:
                        # Create columns based on number of metrics (2-4)
                        if len(metrics_to_show) == 2:
                            cols = st.columns(2)
                        elif len(metrics_to_show) == 3:
                            cols = st.columns(3)
                        else:
                            cols = st.columns(4)
                        
                        for i, (col, metric) in enumerate(zip(cols, metrics_to_show)):
                            with col:
                                current_val = pivot_df[pivot_df['Metric'] == metric][latest_range].iloc[0]
                                prev_val = pivot_df[pivot_df['Metric'] == metric][prev_range].iloc[0]
                                delta = calculate_wow_change(prev_val, current_val)
                                
                                # Get formatting info for this metric
                                metric_info = filtered_df[filtered_df['Metric'] == metric].iloc[0] if not filtered_df[filtered_df['Metric'] == metric].empty else None
                                
                                # Format the metric name for display (remove icons if present)
                                clean_metric = metric.replace('📊 ', '').replace('📈 ', '')
                                
                                if metric_info is not None:
                                    is_percentage = metric_info['Is Percentage']
                                    is_currency = metric_info['Is Currency']
                                    
                                    if is_percentage:
                                        formatted_val = f"{current_val:.1f}%"
                                    elif is_currency:
                                        formatted_val = f"${current_val:,.0f}"
                                    else:
                                        # If it's just a number (no $ or %), it's weight in lb
                                        formatted_val = f"{current_val:,.0f} lb"
                                else:
                                    formatted_val = f"{current_val:,.1f}%"  # Default to percentage for rates
                                
                                # Create delta display with clear business performance indicators
                                if pd.notna(delta):
                                    # Determine if this is a "lower is better" metric
                                    clean_metric_lower = clean_metric.lower()
                                    lower_is_better = any(word in clean_metric_lower for word in [
                                        'waste rate', 'overpack rate', 'expense', 'cost', 'transport cost', 'fruit cost as %'
                                    ])
                                    
                                    # Cap extreme percentage changes to make them more readable
                                    if abs(delta) > 999:
                                        # For very large changes, show absolute change instead
                                        abs_change = current_val - prev_val
                                        is_good_change = (abs_change < 0 if lower_is_better else abs_change > 0)
                                        
                                        if abs_change > 0:
                                            if is_good_change:
                                                delta_display = f"✅ +{abs_change:.1f}pts"
                                            else:
                                                delta_display = f"⚠️ +{abs_change:.1f}pts"
                                            delta_color = "normal"
                                        elif abs_change < 0:
                                            if not is_good_change:
                                                delta_display = f"⚠️ {abs_change:.1f}pts"
                                            else:
                                                delta_display = f"✅ {abs_change:.1f}pts"
                                            delta_color = "normal"
                                        else:
                                            delta_display = f"➡️ {abs_change:.1f}pts"
                                            delta_color = "normal"
                                    else:
                                        # Normal percentage change display with clear good/bad indicators
                                        is_good_change = (delta < 0 if lower_is_better else delta > 0)
                                        
                                        if delta > 0:
                                            if is_good_change:
                                                delta_display = f"✅ +{delta:.1f}%"
                                            else:
                                                delta_display = f"⚠️ +{delta:.1f}%"
                                            delta_color = "normal"
                                        elif delta < 0:
                                            if not is_good_change:
                                                delta_display = f"⚠️ {delta:.1f}%"
                                            else:
                                                delta_display = f"✅ {delta:.1f}%"
                                            delta_color = "normal"
                                        else:
                                            delta_display = f"➡️ {delta:.1f}%"
                                            delta_color = "normal"
                                else:
                                    # Handle cases where previous value was near zero
                                    if pd.notna(prev_val) and pd.notna(current_val):
                                        if abs(prev_val) < 0.01 and current_val > 0.01:
                                            delta_display = "🆕 New"
                                            delta_color = "normal"
                                        elif abs(current_val) < 0.01 and prev_val > 0.01:
                                            delta_display = "📉 Stopped"
                                            delta_color = "normal"
                                        else:
                                            delta_display = None
                                            delta_color = "normal"
                                    else:
                                        delta_display = None
                                        delta_color = "normal"
                                
                                # Display metric without Streamlit's automatic delta arrows
                                st.metric(
                                    f"📊 {clean_metric}",
                                    formatted_val,
                                    help=f"Change from {format_date_range(prev_range)} to {format_date_range(latest_range)}"
                                )
                                
                                # Show our custom change indicator below
                                if delta_display:
                                    st.markdown(f"<div style='margin-top: -10px; font-size: 14px;'>{delta_display}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No rate metrics found in the selected data.")
            elif selected_ranges and not selected_metrics:
                st.warning("Please select at least one metric to display.")
            else:
                st.warning("Please select at least one date range to display data.")
        else:
            st.warning("No Week over Week data available")



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
                                "Total Weight (lb)",
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
                                "Total Weight (lb)",
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
        reference_data = st.session_state.get('reference_data', {})
        
        # Pieces vs Lb Conversion
        pieces_vs_lb_df = reference_data.get('pieces_vs_lb')
        if pieces_vs_lb_df is not None and not pieces_vs_lb_df.empty:
            st.dataframe(pieces_vs_lb_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No pieces vs lb conversion data available")

    # Display Unfulfilled Orders (same date range as needs/haves)
    with st.expander("🚨 Unfulfilled Orders", expanded=False):
        st.subheader("📋 Unfulfilled Orders (Last 7 Days)")
        st.markdown("*Using the same date range as the Need/Have Summary above*")
        
        # Use the same 7-day period as the needs/haves summary
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Add refresh button
        if st.button("🔄 Refresh Unfulfilled Orders", key="unfulfilled_orders_refresh"):
            with st.spinner("Refreshing unfulfilled orders..."):
                try:
                    orders_df, unfulfilled_df = load_shopify_orders(start_date=start_date, end_date=end_date, force_refresh=True)
                    if unfulfilled_df is not None:
                        st.session_state['unfulfilled_orders_df'] = unfulfilled_df
                        st.success("✅ Unfulfilled orders refreshed!")
                    else:
                        st.warning("⚠️ No unfulfilled orders found")
                except Exception as e:
                    st.error(f"❌ Could not refresh unfulfilled orders: {str(e)}")
        
        # Get unfulfilled orders from session state or load fresh
        unfulfilled_df = st.session_state.get('unfulfilled_orders_df')
        if unfulfilled_df is None:
            with st.spinner("Loading unfulfilled orders..."):
                try:
                    orders_df, unfulfilled_df = load_shopify_orders(start_date=start_date, end_date=end_date)
                    if unfulfilled_df is not None:
                        st.session_state['unfulfilled_orders_df'] = unfulfilled_df
                except Exception as e:
                    st.error(f"Error loading unfulfilled orders: {str(e)}")
        
        if unfulfilled_df is not None and not unfulfilled_df.empty:
            # Filter out digital items for display (same logic as Need/Have summary)
            display_unfulfilled = unfulfilled_df[~unfulfilled_df.apply(is_digital_item, axis=1)].copy()
            
            if not display_unfulfilled.empty:
                # Add filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # SKU filter
                    skus = sorted(display_unfulfilled['SKU'].dropna().unique())
                    selected_skus = st.multiselect(
                        "Filter by SKU",
                        skus,
                        default=None,
                        placeholder="Choose SKUs...",
                        key="unfulfilled_sku_filter"
                    )
                    if selected_skus:
                        display_unfulfilled = display_unfulfilled[display_unfulfilled['SKU'].isin(selected_skus)]
                
                with col2:
                    # Shipping Method filter
                    shipping_methods = sorted(display_unfulfilled['Shipping Method'].dropna().unique())
                    selected_shipping = st.multiselect(
                        "Filter by Shipping Method",
                        shipping_methods,
                        default=None,
                        placeholder="Choose shipping methods...",
                        key="unfulfilled_shipping_filter"
                    )
                    if selected_shipping:
                        display_unfulfilled = display_unfulfilled[display_unfulfilled['Shipping Method'].isin(selected_shipping)]
                
                with col3:
                    # Order Name filter
                    order_names = sorted(display_unfulfilled['Order Name'].dropna().unique())
                    selected_orders = st.multiselect(
                        "Filter by Order",
                        order_names,
                        default=None,
                        placeholder="Choose orders...",
                        key="unfulfilled_order_filter"
                    )
                    if selected_orders:
                        display_unfulfilled = display_unfulfilled[display_unfulfilled['Order Name'].isin(selected_orders)]
                

                # Display the unfulfilled orders table
                st.dataframe(
                    display_unfulfilled.sort_values('Created At', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Created At": st.column_config.DatetimeColumn(
                            "Order Date",
                            format="MMM DD, YYYY HH:mm"
                        ),
                        "Order Name": st.column_config.TextColumn("Order #"),
                        "SKU": st.column_config.TextColumn("SKU"),
                        "Product Title": st.column_config.TextColumn("Product"),
                        "Variant Title": st.column_config.TextColumn("Variant"),
                        "Unfulfilled Quantity": st.column_config.NumberColumn(
                            "Qty Needed",
                            format="%d"
                        ),
                        "Unit Price": st.column_config.NumberColumn(
                            "Unit Price",
                            format="$%.2f"
                        ),
                        "Line Item Total": st.column_config.NumberColumn(
                            "Line Total",
                            format="$%.2f"
                        ),
                        "Delivery Date": st.column_config.TextColumn("Delivery Date"),
                        "Shipping Method": st.column_config.TextColumn("Shipping Method"),
                        "Tags": st.column_config.TextColumn("Tags")
                    }
                )
                

            else:
                st.info("No unfulfilled orders found (excluding digital items like ripening guides)")
        else:
            st.warning("No unfulfilled orders data available")

    # Display Fulfilled Orders
    with st.expander("📦 Fulfilled Orders", expanded=False):
        st.subheader("🛍️ Fulfilled Orders")
        
        # Add date range controls - now focused on fulfilled orders
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
            
        # Get fulfilled orders data using the simplified query
        with st.spinner("Loading fulfilled orders..."):
            try:
                orders_df = update_fulfilled_orders_data(
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.max.time())
                )
            except Exception as e:
                st.error(f"Error loading fulfilled orders: {str(e)}")
                orders_df = None
        
        if orders_df is not None and not orders_df.empty:
            # Since we're only showing fulfilled orders, no status filter needed
            filtered_df = orders_df
            
            # Add basic filters since customer data is removed
            if 'SKU' in orders_df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    # SKU filter
                    skus = sorted([sku for sku in orders_df['SKU'].dropna().unique() if sku])
                    selected_skus = st.multiselect(
                        "Filter by SKU",
                        skus,
                        default=None,
                        placeholder="Choose SKUs...",
                        help="Filter orders by SKU"
                    )
                    if selected_skus:
                        filtered_df = filtered_df[filtered_df['SKU'].isin(selected_skus)]
                
                with col2:
                    # Order tags filter
                    if 'Tags' in orders_df.columns:
                        all_tags = set()
                        for tag_list in orders_df['Tags'].dropna():
                            if isinstance(tag_list, str) and tag_list:
                                all_tags.update([tag.strip() for tag in tag_list.split(',') if tag.strip()])
                        
                        if all_tags:
                            selected_tags = st.multiselect(
                                "Filter by Order Tags",
                                sorted(all_tags),
                                default=None,
                                placeholder="Choose order tags...",
                                help="Filter orders by order tags"
                            )
                            if selected_tags:
                                # Filter rows where order tags contain any of the selected tags
                                mask = orders_df['Tags'].apply(
                                    lambda tags: any(tag in str(tags) for tag in selected_tags) if pd.notna(tags) else False
                                )
                                filtered_df = filtered_df[mask]
            
            if not filtered_df.empty:
                # Show summary statistics for fulfilled orders
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Fulfilled Orders", len(filtered_df))
                with col2:
                    # Use Total column for simplified query
                    total_value = filtered_df['Total'].sum() if 'Total' in filtered_df.columns else 0
                    st.metric("Total Value", f"${total_value:,.2f}")
                with col3:
                    # Add average fulfillment time for physical items (if we have fulfilled orders with timing data)
                    if 'Fulfilled At' in filtered_df.columns and 'Created At' in filtered_df.columns:
                        # Filter out digital items for fulfillment time calculation
                        physical_items = filter_physical_items(filtered_df)
                        
                        if not physical_items.empty:
                            # Calculate fulfillment time for physical items only
                            physical_with_times = physical_items.dropna(subset=['Fulfilled At', 'Created At'])
                            if not physical_with_times.empty:
                                # Convert string dates to datetime if needed
                                if physical_with_times['Created At'].dtype == 'object':
                                    physical_with_times = physical_with_times.copy()
                                    physical_with_times['Created At'] = pd.to_datetime(physical_with_times['Created At'])
                                if physical_with_times['Fulfilled At'].dtype == 'object':
                                    physical_with_times = physical_with_times.copy()
                                    physical_with_times['Fulfilled At'] = pd.to_datetime(physical_with_times['Fulfilled At'])
                                
                                # Calculate fulfillment duration in hours
                                fulfillment_durations = (physical_with_times['Fulfilled At'] - physical_with_times['Created At']).dt.total_seconds() / 3600
                                avg_hours = fulfillment_durations.mean()
                                
                                if pd.notna(avg_hours):
                                    if avg_hours >= 24:
                                        avg_display = f"{avg_hours/24:.1f} days"
                                    else:
                                        avg_display = f"{avg_hours:.1f} hours"
                                    st.metric("Avg Fulfillment Time (Physical)", avg_display)
                                else:
                                    st.metric("Avg Fulfillment Time (Physical)", "No data")
                            else:
                                st.metric("Avg Fulfillment Time (Physical)", "No timing data")
                        else:
                            st.metric("Avg Fulfillment Time (Physical)", "No physical items")
                    else:
                        # Show average order value for fulfilled orders (fallback if no timing data)
                        total_value = filtered_df['Total'].sum() if 'Total' in filtered_df.columns else 0
                        avg_order_value = total_value / len(filtered_df) if len(filtered_df) > 0 else 0
                        st.metric("Avg Order Value", f"${avg_order_value:,.2f}")
                with col4:
                    # Show unique SKUs count since customer data is removed
                    unique_skus = len(filtered_df['SKU'].dropna().unique()) if 'SKU' in filtered_df.columns else 0
                    st.metric("Unique SKUs", f"{unique_skus:,}")

                # Add Top SKUs Summary
                st.subheader("📊 Top SKUs (Physical Items Only)")
                
                # Calculate SKU summary for physical items only
                physical_items = filter_physical_items(filtered_df)
                
                if not physical_items.empty:
                    # Calculate line item total properly: Unit Price × Quantity
                    physical_items_calc = physical_items.copy()
                    physical_items_calc['Line Item Total'] = physical_items_calc['Unit Price'] * physical_items_calc['Quantity']
                    
                    sku_summary = physical_items_calc.groupby('SKU').agg({
                        'Quantity': 'sum',
                        'Line Item Total': 'sum'
                    }).reset_index()
                    
                    # Rename the total column for consistency
                    sku_summary = sku_summary.rename(columns={'Line Item Total': 'Total Value'})
                    
                    # Sort by quantity and get top 10
                    top_skus_qty = sku_summary.nlargest(10, 'Quantity')
                    
                    # Sort by total value and get top 10
                    top_skus_value = sku_summary.nlargest(10, 'Total Value')
                    
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
                                "Total Value": st.column_config.NumberColumn(
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
                                "Total Value": st.column_config.NumberColumn(
                                    "Total Value",
                                    format="$%.2f"
                                )
                            }
                        )
                else:
                    st.info("No physical items found in the selected orders. Only digital/free items are present.")

                # Display full orders table
                st.subheader("📋 All Orders")
                
                # Configure columns for fulfilled orders (no customer data)
                column_config = {
                    "Created At": st.column_config.DatetimeColumn(
                        "Order Date",
                        format="MMM DD, YYYY HH:mm"
                    ),
                    "Order Name": st.column_config.TextColumn("Order #"),
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
                    "Delivery Date": st.column_config.TextColumn(
                        "Delivery Date",
                        help="Scheduled delivery date"
                    ),
                    "SKU": st.column_config.TextColumn(
                        "SKU",
                        help="Product SKU"
                    ),
                    "Quantity": st.column_config.NumberColumn(
                        "Quantity",
                        format="%d"
                    ),
                    "Unit Price": st.column_config.NumberColumn(
                        "Unit Price",
                        format="$%.2f"
                    ),
                    "Tags": st.column_config.TextColumn(
                        "Order Tags",
                        help="Order tags"
                    )
                }
                
                st.dataframe(
                    filtered_df.sort_values('Created At', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )
            else:
                st.warning("No fulfilled orders found with selected filters")
        else:
            st.warning("No fulfilled orders data available")

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
                status_help_text = """
                - **Imported:** Added to ColdCart inventory system
                - **Confirmed:** Vendor confirmed they will fulfill this order
                - **InTransit:** On its way to warehouse
                - **Delivered:** Arrived at warehouse, but not yet in ColdCart
                - **Pending:** In discussion with vendor, waiting for confirmation
                - **N/A:** Not able to get fruit because of some reasons
                """
                selected_oxnard_order_statuses = st.multiselect(
                    "Filter by Status",
                    oxnard_order_statuses,
                    default=oxnard_order_statuses,
                    key="oxnard_order_status_filter",
                    help=status_help_text
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
                status_help_text = """
                - **Imported:** Added to ColdCart inventory system
                - **Confirmed:** Vendor confirmed they will fulfill this order
                - **InTransit:** On its way to warehouse
                - **Delivered:** Arrived at warehouse, but not yet in ColdCart
                - **Pending:** In discussion with vendor, waiting for confirmation
                - **N/A:** Not able to get fruit because of some reasons
                """
                selected_wheeling_order_statuses = st.multiselect(
                    "Filter by Status",
                    wheeling_order_statuses,
                    default=wheeling_order_statuses,
                    key="wheeling_order_status_filter",
                    help=status_help_text
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