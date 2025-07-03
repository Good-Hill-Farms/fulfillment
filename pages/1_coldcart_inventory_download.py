import streamlit as st
import os
import pandas as pd
from datetime import datetime
from utils.inventory_api import get_inventory_data, save_as_excel, save_as_csv

st.set_page_config(
    page_title="ColdCart Inventory Download",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 ColdCart Inventory Download")
    
    st.markdown("""
    This page allows you to download and analyze the current inventory data from ColdCart.
    The data includes warehouse locations, SKUs, quantities, and other relevant information.
    """)
    
    # Add a refresh button
    if st.button("🔄 Refresh Inventory Data"):
        with st.spinner("Fetching latest inventory data..."):
            try:
                df = get_inventory_data()
                if df is not None:
                st.session_state['inventory_df'] = df
                st.success("✅ Data fetched successfully!")
                else:
                    st.error("❌ No data received from the API")
                    st.info("Please check if your API token is correctly set in the environment variables.")
                    return
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.info("Please check if your API token is correctly set in the environment variables.")
                return
    
    # Use cached data if available, otherwise fetch new data
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
    if df is None:
        st.error("❌ No inventory data available")
        st.info("Please try refreshing the data using the button above.")
        return
    
    # Data Filtering Section
    st.subheader("Data Filters")
    
    # Create columns for filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Warehouse filter
        warehouses = ['All'] + sorted(df['WarehouseName'].unique().tolist())
        selected_warehouse = st.selectbox('Filter by Warehouse', warehouses)
        
        # Type filter
        types = ['All'] + sorted(df['Type'].unique().tolist())
        selected_type = st.selectbox('Filter by Type', types)
    
    with filter_col2:
        # SKU search
        sku_search = st.text_input('Search by SKU')
        
        # Name search
        name_search = st.text_input('Search by Name')
    
    with filter_col3:
        # Show only items with stock
        show_in_stock = st.checkbox('Show Only Items in Stock', value=False)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_warehouse != 'All':
        filtered_df = filtered_df[filtered_df['WarehouseName'] == selected_warehouse]
    
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['Type'] == selected_type]
    
    if sku_search:
        filtered_df = filtered_df[filtered_df['Sku'].str.contains(sku_search, case=False, na=False)]
    
    if name_search:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(name_search, case=False, na=False)]
    
    if show_in_stock:
        filtered_df = filtered_df[filtered_df['AvailableQty'] > 0]
    
    # Display filtered data statistics
    st.write(f"Showing {len(filtered_df)} out of {len(df)} items")
    
    # Sort options
    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_by = st.selectbox(
            'Sort by',
            ['WarehouseName', 'Sku', 'Name', 'Type', 'AvailableQty', 'DaysOnHand']
        )
    with sort_col2:
        sort_order = st.selectbox('Order', ['Ascending', 'Descending'])
    
    # Apply sorting
    filtered_df = filtered_df.sort_values(
        by=sort_by,
        ascending=(sort_order == 'Ascending')
    )
    
    # Display the data
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download buttons for filtered data
    st.subheader("Download Filtered Data")
    col1, col2 = st.columns(2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with col1:
        csv_filename = f"inventory_export_{timestamp}.csv"
        csv_path = save_as_csv(filtered_df, csv_filename)
        with open(csv_path, 'rb') as f:
            st.download_button(
                label=f"📥 Download Filtered Data as CSV ({len(filtered_df)} items)",
                data=f,
                file_name=csv_filename,
                mime="text/csv"
            )
        os.remove(csv_path)
    
    with col2:
        excel_filename = f"inventory_export_{timestamp}.xlsx"
        excel_path = save_as_excel(filtered_df, excel_filename)
        with open(excel_path, 'rb') as f:
            st.download_button(
                label=f"📥 Download Filtered Data as Excel ({len(filtered_df)} items)",
                data=f,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        os.remove(excel_path)
    
    # Add some helpful information
    st.markdown("""
    ---
    ### Notes:
    - The data is fetched in real-time from ColdCart's API
    - Use the filters above to narrow down the data
    - Sort the data by clicking on column headers
    - Downloads will include only the filtered data you see in the table
    - Files are generated with timestamps to avoid conflicts
    - Make sure your API token is set in the environment variables as `COLDCART_API_TOKEN`
    """)

if __name__ == "__main__":
    main() 