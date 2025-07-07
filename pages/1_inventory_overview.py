import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.inventory_api import get_inventory_data
from utils.google_sheets import load_agg_orders

st.set_page_config(
    page_title="Inventory Overview",
    page_icon="📦",
    layout="wide"
)

def create_order_units_by_sku_chart(df):
    """Create a bar chart showing total actual fruit order (from vendors) units by SKU"""
    # Combine Oxnard and Wheeling actual orders
    df_combined = pd.DataFrame()
    
    if 'Oxnard Actual Order' in df.columns:
        oxnard_orders = df.groupby(['Fruit', 'Oxnard Picklist SKU'])['Oxnard Actual Order'].sum().reset_index()
        oxnard_orders.columns = ['Fruit', 'SKU', 'Total Units']
        df_combined = pd.concat([df_combined, oxnard_orders])
        
    if 'Wheeling Actual Order' in df.columns:
        wheeling_orders = df.groupby(['Fruit', 'Wheeling Picklist SKU'])['Wheeling Actual Order'].sum().reset_index()
        wheeling_orders.columns = ['Fruit', 'SKU', 'Total Units']
        df_combined = pd.concat([df_combined, wheeling_orders])
    
    # Aggregate total units by Fruit/SKU
    df_combined = df_combined.groupby(['Fruit', 'SKU'])['Total Units'].sum().reset_index()
    df_combined['Fruit / SKU'] = df_combined['Fruit'] + ' - ' + df_combined['SKU']
    
    # Sort and get top 15
    df_combined = df_combined.sort_values('Total Units', ascending=False).head(15)
    
    fig = px.bar(
        df_combined,
        x='Fruit / SKU',
        y='Total Units',
        title='Top 15: Total Actual Fruit Order Units by Fruit/SKU',
        labels={'Fruit / SKU': 'Fruit / SKU', 'Total Units': 'Total Fruit Order Units'},
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=500
    )
    return fig

def create_order_units_by_vendor_chart(df):
    """Create a bar chart showing total actual fruit order units by vendor"""
    df_combined = pd.DataFrame()
    
    if 'Oxnard Actual Order' in df.columns:
        oxnard_orders = df.groupby('Vendor')['Oxnard Actual Order'].sum().reset_index()
        df_combined = pd.concat([df_combined, oxnard_orders])
        
    if 'Wheeling Actual Order' in df.columns:
        wheeling_orders = df.groupby('Vendor')['Wheeling Actual Order'].sum().reset_index()
        df_combined = pd.concat([df_combined, wheeling_orders])
    
    # Aggregate total units by Vendor
    df_combined = df_combined.groupby('Vendor').sum().reset_index()
    
    # Sort and filter out vendors with 0 orders
    df_combined = df_combined[df_combined['Oxnard Actual Order'] > 0].sort_values('Oxnard Actual Order', ascending=False)
    
    fig = px.bar(
        df_combined,
        x='Vendor',
        y='Oxnard Actual Order',
        title='Total Actual Fruit Order Units by Vendor',
        labels={'Vendor': 'Vendor', 'Oxnard Actual Order': 'Total Fruit Order Units'},
        color_discrete_sequence=['lightblue']
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=500
    )
    return fig

def main():
    st.title("📦 Inventory Overview")

    st.markdown("""
    This page provides a summary of inventory across all warehouses.
    """)

    # Load aggregated orders data first
    if 'agg_orders_df' not in st.session_state:
        with st.spinner("Fetching aggregated orders data..."):
            df_orders = load_agg_orders()
            if df_orders is not None:
                st.session_state['agg_orders_df'] = df_orders
                st.success("✅ Order data fetched successfully!")
            else:
                st.error("❌ No order data received from Google Sheets")

    df_orders = st.session_state.get('agg_orders_df')

    if df_orders is None:
        st.warning("No aggregated order data available.")
    else:
        # Pre-filter to remove rows with empty Status or Projection Period
        if 'Status' in df_orders.columns:
            df_orders['Status'] = df_orders['Status'].replace('', np.nan)
            df_orders.dropna(subset=['Status'], inplace=True)
        
        if 'Projection Period' in df_orders.columns:
            df_orders['Projection Period'] = df_orders['Projection Period'].replace('', np.nan)
            df_orders.dropna(subset=['Projection Period'], inplace=True)

        # Show filters first
        st.header("Fruit Order Filters (from vendors)")
        
        df_filtered = df_orders.copy()

        # Create columns for global filters
        col1, col2, col3 = st.columns(3)

        # Combined Fruit/SKU/Batchcode filter
        filter_type = col1.selectbox(
            "Filter by",
            ["Fruit", "SKU", "Batchcode"],
            key="filter_type"
        )

        if filter_type == "Fruit" and 'Fruit' in df_filtered.columns:
            fruits = sorted(df_filtered['Fruit'].astype(str).unique())
            selected_values = col2.multiselect("Select Fruits", fruits)
            if selected_values:
                df_filtered = df_filtered[df_filtered['Fruit'].isin(selected_values)]
                st.session_state['selected_fruits'] = selected_values
        
        elif filter_type == "SKU":
            all_skus = set()
            if 'Oxnard Picklist SKU' in df_filtered.columns:
                all_skus.update(df_filtered['Oxnard Picklist SKU'].dropna().unique())
            if 'Wheeling Picklist SKU' in df_filtered.columns:
                all_skus.update(df_filtered['Wheeling Picklist SKU'].dropna().unique())
            selected_values = col2.multiselect("Select SKUs", sorted(all_skus))
            if selected_values:
                sku_mask = df_filtered['Oxnard Picklist SKU'].isin(selected_values) | \
                          df_filtered['Wheeling Picklist SKU'].isin(selected_values)
                df_filtered = df_filtered[sku_mask]
                st.session_state['selected_skus'] = selected_values

        elif filter_type == "Batchcode":
            all_batchcodes = set()
            if 'Oxnard Batchcode' in df_filtered.columns:
                all_batchcodes.update(df_filtered['Oxnard Batchcode'].dropna().unique())
            if 'Wheeling Batchcode' in df_filtered.columns:
                all_batchcodes.update(df_filtered['Wheeling Batchcode'].dropna().unique())
            selected_values = col2.multiselect("Select Batchcodes", sorted(all_batchcodes))
            if selected_values:
                batchcode_mask = df_filtered['Oxnard Batchcode'].isin(selected_values) | \
                                df_filtered['Wheeling Batchcode'].isin(selected_values)
                df_filtered = df_filtered[batchcode_mask]
                st.session_state['selected_batchcodes'] = selected_values

        # Create second row for date and projection period filters
        date_col, proj_col, _ = st.columns(3)

        # Date filter
        if 'Date_1' in df_filtered.columns:
            df_filtered['Date_1'] = pd.to_datetime(df_filtered['Date_1'], errors='coerce')
            df_filtered.dropna(subset=['Date_1'], inplace=True)

            if not df_filtered.empty:
                min_date = df_filtered['Date_1'].min()
                max_date = df_filtered['Date_1'].max()

                default_start_date = max_date - pd.Timedelta(days=6)
                if default_start_date < min_date:
                    default_start_date = min_date

                selected_date_range = date_col.date_input(
                    "Filter by Order Date",
                    value=(default_start_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="date_filter"
                )
                if len(selected_date_range) == 2:
                    start_date, end_date = selected_date_range
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1, seconds=-1)
                    df_filtered = df_filtered[(df_filtered['Date_1'] >= start_date) & (df_filtered['Date_1'] <= end_date)]
            else:
                date_col.warning("No valid dates found in 'Date_1' column to filter.")
        else:
            date_col.warning("Date column ('Date_1') not found.")

        # Projection Period filter
        if 'Projection Period' in df_filtered.columns:
            projection_periods = sorted(df_filtered['Projection Period'].astype(str).unique())
            selected_periods = proj_col.multiselect("Filter by Projection Period", projection_periods, default=projection_periods)
            df_filtered = df_filtered[df_filtered['Projection Period'].isin(selected_periods)]
        else:
            proj_col.warning("Column 'Projection Period' not found.")

        # Add Status Summary Table
        st.subheader("Fruit Order Status Summary (from vendors)")
        
        # Create status summary for Oxnard
        if 'Oxnard Status' in df_filtered.columns:
            # Clean numeric columns first
            df_filtered['Oxnard Actual Order'] = pd.to_numeric(df_filtered['Oxnard Actual Order'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(0)
            df_filtered['Oxnard Weight Needed'] = pd.to_numeric(df_filtered['Oxnard Weight Needed'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(0)
            
            # Replace empty status with "❌ No Status" label
            df_filtered['Oxnard Status'] = df_filtered['Oxnard Status'].replace(['', np.nan], '❌ No Status')
            
            oxnard_status_summary = df_filtered.groupby('Oxnard Status').agg({
                'Oxnard Actual Order': ['count', 'sum'],
                'Oxnard Weight Needed': 'sum'
            }).reset_index()
            
            # Flatten column names
            oxnard_status_summary.columns = ['Status', 'Number of Orders', 'Total Units', 'Total Weight (LBS)']
            oxnard_status_summary['Warehouse'] = 'Oxnard'
        else:
            oxnard_status_summary = pd.DataFrame()

        # Create status summary for Wheeling
        if 'Wheeling Status' in df_filtered.columns:
            # Clean numeric columns first
            df_filtered['Wheeling Actual Order'] = pd.to_numeric(df_filtered['Wheeling Actual Order'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(0)
            df_filtered['Wheeling Weight Needed'] = pd.to_numeric(df_filtered['Wheeling Weight Needed'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(0)
            
            # Replace empty status with "❌ No Status" label
            df_filtered['Wheeling Status'] = df_filtered['Wheeling Status'].replace(['', np.nan], '❌ No Status')
            
            wheeling_status_summary = df_filtered.groupby('Wheeling Status').agg({
                'Wheeling Actual Order': ['count', 'sum'],
                'Wheeling Weight Needed': 'sum'
            }).reset_index()
            
            # Flatten column names
            wheeling_status_summary.columns = ['Status', 'Number of Orders', 'Total Units', 'Total Weight (LBS)']
            wheeling_status_summary['Warehouse'] = 'Wheeling'
        else:
            wheeling_status_summary = pd.DataFrame()

        # Combine both summaries
        status_summary = pd.concat([oxnard_status_summary, wheeling_status_summary])
        
        # Clean up numeric columns and ensure they're integers
        numeric_cols = ['Number of Orders', 'Total Units', 'Total Weight (LBS)']
        for col in numeric_cols:
            status_summary[col] = pd.to_numeric(status_summary[col], errors='coerce').fillna(0).astype(int)

        # Sort by Warehouse and then by Status, but keep "❌ No Status" at the end of each warehouse group
        def custom_sort(df):
            # Create a temporary column for sorting where "❌ No Status" gets a high value
            df['sort_key'] = df['Status'].apply(lambda x: 'z' if x == '❌ No Status' else x)
            # Sort by Warehouse and then by the sort key
            df = df.sort_values(['Warehouse', 'sort_key'])
            # Drop the temporary column
            df = df.drop('sort_key', axis=1)
            return df

        status_summary = custom_sort(status_summary)
        
        # Display the summary table
        st.dataframe(
            status_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Number of Orders': st.column_config.NumberColumn(format="%d"),
                'Total Units': st.column_config.NumberColumn(format="%d"),
                'Total Weight (LBS)': st.column_config.NumberColumn(format="%d"),
                'Status': st.column_config.Column(width="medium"),
                'Warehouse': st.column_config.Column(width="small")
            }
        )

        # Clean numeric columns
        for col in ['Oxnard Actual Order', 'Wheeling Actual Order']:
            if col in df_filtered.columns:
                df_filtered[col] = pd.to_numeric(df_filtered[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(0)

        # Create visualization section
        st.header("Fruit Order Analytics (from vendors)")
        
        # Display total actual order units by SKU chart
        fig_units = create_order_units_by_sku_chart(df_filtered)
        st.plotly_chart(fig_units, use_container_width=True)
        
        # Display vendor chart
        fig_vendor = create_order_units_by_vendor_chart(df_filtered)
        st.plotly_chart(fig_vendor, use_container_width=True)

        st.header("Fruit Orders (from vendors)")

        # --- Oxnard Section ---
        st.subheader("Oxnard Fruit Orders (from vendors)")
        
        # Oxnard-specific status filter
        if 'Oxnard Status' in df_filtered.columns:
            oxnard_statuses = sorted(df_filtered['Oxnard Status'].astype(str).unique())
            selected_oxnard_statuses = st.multiselect("Filter by Oxnard Status", oxnard_statuses, default=oxnard_statuses)
            df_filtered = df_filtered[df_filtered['Oxnard Status'].isin(selected_oxnard_statuses)]
        
        oxnard_cols_to_display = [
            'Date_1', 'Vendor', 'Fruit', 'Oxnard Picklist SKU', 'Oxnard Status', 'Oxnard Notes',
            'Oxnard Weight Needed', 'Oxnard Order', 'Oxnard Actual Order', 'Weight Per Pick', 'Oxnard Batchcode'
        ]
        oxnard_display_cols = [col for col in oxnard_cols_to_display if col in df_filtered.columns]
        df_oxnard = df_filtered[df_filtered['Oxnard Status'].notna()][oxnard_display_cols].copy()
        
        # Clean numeric columns for this specific dataframe
        oxnard_numeric_cols = ['Oxnard Weight Needed', 'Oxnard Order', 'Oxnard Actual Order', 'Weight Per Pick']
        for col in oxnard_numeric_cols:
            if col in df_oxnard.columns:
                df_oxnard[col] = pd.to_numeric(df_oxnard[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(0)

        # Filter by weight > 0
        if 'Oxnard Weight Needed' in df_oxnard.columns:
            df_oxnard = df_oxnard[df_oxnard['Oxnard Weight Needed'] > 0]

        # Rename columns for final display
        oxnard_rename_map = {
            "Date_1": "Agg_Date", "Oxnard Status": "Status", "Oxnard Notes": "Notes",
            "Oxnard Picklist SKU": "SKU", "Oxnard Batchcode": "Batchcode",
            "Oxnard Weight Needed": "Weight Needed (LBS)", "Oxnard Order": "Fruit Order Qty (Units)",
            "Oxnard Actual Order": "Actual Fruit Order (Units)", "Weight Per Pick": "Weight/Pick (LBS)"
        }
        st.dataframe(df_oxnard.rename(columns=oxnard_rename_map), use_container_width=True, hide_index=True)

        # --- Wheeling Section ---
        st.subheader("Wheeling Fruit Orders (from vendors)")
        
        # Wheeling-specific status filter
        if 'Wheeling Status' in df_filtered.columns:
            wheeling_statuses = sorted(df_filtered['Wheeling Status'].astype(str).unique())
            selected_wheeling_statuses = st.multiselect("Filter by Wheeling Status", wheeling_statuses, default=wheeling_statuses)
            df_filtered = df_filtered[df_filtered['Wheeling Status'].isin(selected_wheeling_statuses)]
        
        wheeling_cols_to_display = [
            'Date_1', 'Vendor', 'Fruit', 'Wheeling Picklist SKU', 'Wheeling Status', 'Wheeling Notes',
            'Wheeling Weight Needed', 'Wheeling Order', 'Wheeling Actual Order', 'Wheeling Weight Per Pick', 'Wheeling Batchcode'
        ]
        wheeling_display_cols = [col for col in wheeling_cols_to_display if col in df_filtered.columns]
        df_wheeling = df_filtered[df_filtered['Wheeling Status'].notna()][wheeling_display_cols].copy()

        # Clean numeric columns for this specific dataframe
        wheeling_numeric_cols = ['Wheeling Weight Needed', 'Wheeling Order', 'Wheeling Actual Order', 'Wheeling Weight Per Pick']
        for col in wheeling_numeric_cols:
            if col in df_wheeling.columns:
                df_wheeling[col] = pd.to_numeric(df_wheeling[col].astype(str).str.extract(r'(\d+\.?\d*)', expand=False), errors='coerce').fillna(0)
        
        # Filter by weight > 0
        if 'Wheeling Weight Needed' in df_wheeling.columns:
            df_wheeling = df_wheeling[df_wheeling['Wheeling Weight Needed'] > 0]
        
        # Rename columns for final display
        wheeling_rename_map = {
            "Date_1": "Agg_Date", "Wheeling Status": "Status", "Wheeling Notes": "Notes",
            "Wheeling Picklist SKU": "SKU", "Wheeling Batchcode": "Batchcode",
            "Wheeling Weight Needed": "Weight Needed (LBS)", "Wheeling Order": "Fruit Order Qty (Units)",
            "Wheeling Actual Order": "Actual Fruit Order (Units)", "Wheeling Weight Per Pick": "Weight/Pick (LBS)"
        }
        st.dataframe(df_wheeling.rename(columns=wheeling_rename_map), use_container_width=True, hide_index=True)

    # Load and display current inventory data at the end
    if 'inventory_df' not in st.session_state:
        with st.spinner("Fetching initial inventory data..."):
            try:
                df = get_inventory_data()
                if df is not None:
                    st.session_state['inventory_df'] = df
                else:
                    st.error("❌ No data received from the API")
                    st.info("Please check if your API token is correctly set in the environment variables.")
                    return
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.info("Please check if your API token is correctly set in the environment variables.")
                return
    
    df = st.session_state.get('inventory_df')

    if df is not None:
        # Consolidate Moorpark and Oxnard warehouses
        df['WarehouseName'] = df['WarehouseName'].replace({
            'CA-Moorpark-93021': 'Oxnard',
            'CA-Oxnard-93030': 'Oxnard'
        })

        # Extract fruit from SKU if possible
        df['Fruit'] = df['Sku'].str.extract(r'^([^_]+)(?:_|$)').fillna(df['Name'].str.split().str[0])

        # Apply filters based on selection
        filter_type = st.session_state.get('filter_type')
        if filter_type == "Fruit" and 'selected_fruits' in st.session_state:
            selected_fruits = st.session_state['selected_fruits']
            if selected_fruits:
                df = df[df['Fruit'].isin(selected_fruits)]
        elif filter_type == "SKU" and 'selected_skus' in st.session_state:
            selected_skus = st.session_state['selected_skus']
            if selected_skus:
                df = df[df['Sku'].isin(selected_skus)]

        st.header("Current Inventory (ColdCart)")
        st.subheader("Total Inventory by SKU (ColdCart)")
        
        if df.empty:
            st.warning("No inventory data found for the selected filters.")
            return
            
        sku_inventory = df.groupby(['Sku', 'Name', 'Fruit'])['AvailableQty'].sum().reset_index()
        sku_inventory = sku_inventory[sku_inventory['AvailableQty'] > 0]
        
        # Sort by Fruit and then by SKU
        sku_inventory = sku_inventory.sort_values(['Fruit', 'Sku'])
        
        # Display with fruit column
        st.dataframe(
            sku_inventory,
            use_container_width=True,
            hide_index=True,
            column_config={
                'AvailableQty': st.column_config.NumberColumn(
                    'Available Quantity',
                    format="%d"
                )
            }
        )

        st.subheader("Inventory by SKU and Warehouse (ColdCart)")
        # Filter original df to only include SKUs that are in stock somewhere
        in_stock_skus = sku_inventory['Sku'].unique()
        df_in_stock = df[df['Sku'].isin(in_stock_skus)]
        
        warehouse_sku_inventory = df_in_stock.groupby(['WarehouseName', 'Sku', 'Name', 'Fruit'])['AvailableQty'].sum().reset_index()
        
        # Filter out zero-quantity rows before pivoting
        warehouse_sku_inventory = warehouse_sku_inventory[warehouse_sku_inventory['AvailableQty'] > 0]

        # Pivot table for better readability, leave non-existent values as blank
        if not warehouse_sku_inventory.empty:
            # Sort by Fruit and SKU before pivoting
            warehouse_sku_inventory = warehouse_sku_inventory.sort_values(['Fruit', 'Sku'])
            
            # Create pivot table with Fruit included in the index
            pivot_table = warehouse_sku_inventory.pivot(
                index=['Fruit', 'Sku', 'Name'],
                columns='WarehouseName',
                values='AvailableQty'
            )
            
            # Fill NaN values with empty string for better display
            pivot_table = pivot_table.fillna('')
            
            st.dataframe(
                pivot_table,
                use_container_width=True
            )
        else:
            st.info("No warehouse-specific stock found for the items listed above.")

if __name__ == "__main__":
    main()
