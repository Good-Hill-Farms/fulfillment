import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime
from utils.inventory_api import get_inventory_data, save_as_excel, save_as_csv
import pytz
import re

st.set_page_config(
    page_title="Oxnard Inventory | ColdCart",
    page_icon="📊",
    layout="wide"
)

def extract_date_from_batch(batch_code):
    if pd.isna(batch_code):
        return ''
    
    # Try to find date in new format after delivered_
    new_pattern = r'delivered_(\d{6})'
    match = re.search(new_pattern, str(batch_code))
    if match:
        date_str = match.group(1)
        # Keep as MMDDYY format
        return date_str
    
    # Try to find date in format MM/DD/YY
    date_pattern = r'delivered__(\d{2}/\d{2}/\d{2})'
    match = re.search(date_pattern, str(batch_code))
    if match:
        return match.group(1)
    
    # Try to find date after #
    hash_pattern = r'#.*?(\d{2}/\d{2}/\d{2})'
    match = re.search(hash_pattern, str(batch_code))
    if match:
        return match.group(1)
    
    return ''

def main():
    st.title("📊 Oxnard Inventory | ColdCart")
    
    # Add current date in Los Angeles timezone
    la_tz = pytz.timezone('America/Los_Angeles')
    current_time_la = datetime.now(la_tz)
    st.markdown(f"### Data as of {current_time_la.strftime('%Y-%m-%d %H:%M:%S')} (Los Angeles)")
    
    st.markdown("""
    This page allows you to download and analyze the current inventory data for the Oxnard warehouse.
    The data includes SKUs, quantities, and other relevant information.
    """)
    
    # Add a refresh button
    if st.button("🔄 Refresh Inventory Data"):
        with st.spinner("Fetching latest inventory data..."):
            try:
                df = get_inventory_data()
                if df is not None:
                    # Filter for Oxnard warehouse and rename column
                    df = df[df['WarehouseName'] == 'CA-Oxnard-93030'].copy()
                    
                    # Add empty columns for QTY, Unit, and Notes
                    df['Counted QTY'] = ''
                    df['Unit'] = ''  # Empty unit
                    df['Notes'] = ''
                    
                    # Extract date from BatchCode to LOT
                    df['LOT'] = df['BatchCode'].apply(extract_date_from_batch)
                    
                    # Convert ItemId to string
                    df['ItemId'] = df['ItemId'].astype(str)
                    
                    # Ensure Expected AvailableQty is numeric and rename it
                    df['Expected AvailableQty (ea)'] = pd.to_numeric(df['AvailableQty'], errors='coerce')
                    df = df.drop(columns=['AvailableQty', 'BatchCode'])  # Remove unused columns
                    
                    # Reorder columns
                    df = df[['ItemId', 'Sku', 'Name', 'LOT', 'Expected AvailableQty (ea)', 'Counted QTY', 'Unit', 'Notes']]
                    
                    st.session_state['oxnard_df'] = df
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
    if 'oxnard_df' not in st.session_state:
        with st.spinner("Fetching initial inventory data..."):
            try:
                df = get_inventory_data()
                if df is not None:
                    # Filter for Oxnard warehouse and rename column
                    df = df[df['WarehouseName'] == 'CA-Oxnard-93030'].copy()
                    
                    # Add empty columns for QTY, Unit, and Notes
                    df['Counted QTY'] = ''
                    df['Unit'] = ''  # Empty unit
                    df['Notes'] = ''
                    
                    # Extract date from BatchCode to LOT
                    df['LOT'] = df['BatchCode'].apply(extract_date_from_batch)
                    
                    # Convert ItemId to string
                    df['ItemId'] = df['ItemId'].astype(str)
                    
                    # Ensure Expected AvailableQty is numeric and rename it
                    df['Expected AvailableQty (ea)'] = pd.to_numeric(df['AvailableQty'], errors='coerce')
                    df = df.drop(columns=['AvailableQty', 'BatchCode'])  # Remove unused columns
                    
                    # Reorder columns
                    df = df[['ItemId', 'Sku', 'Name', 'LOT', 'Expected AvailableQty (ea)', 'Counted QTY', 'Unit', 'Notes']]
                    
                    st.session_state['oxnard_df'] = df
                else:
                    st.error("❌ No data received from the API")
                    st.info("Please check if your API token is correctly set in the environment variables.")
                    return
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.info("Please check if your API token is correctly set in the environment variables.")
                return
    
    df = st.session_state.get('oxnard_df')
    if df is None:
        st.error("❌ No inventory data available")
        st.info("Please try refreshing the data using the button above.")
        return
    
    # Data Filtering Section
    st.subheader("Data Filters")
    
    # Create columns for filters
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # SKU search
        sku_search = st.text_input('Search by SKU')
    
    with filter_col2:
        # Name search
        name_search = st.text_input('Search by Name')
        
        # Show only items with stock
        show_in_stock = st.checkbox('Show Only Items in Stock', value=False)
    
    # Apply filters
    filtered_df = df.copy()
    
    if sku_search:
        filtered_df = filtered_df[filtered_df['Sku'].str.contains(sku_search, case=False, na=False)]
    
    if name_search:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(name_search, case=False, na=False)]
    
    if show_in_stock:
        filtered_df = filtered_df[filtered_df['Expected AvailableQty (ea)'].fillna(0) > 0]
    
    # Display filtered data statistics
    st.write(f"Showing {len(filtered_df)} out of {len(df)} items")
    
    # Sort options
    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_by = st.selectbox(
            'Sort by',
            ['ItemId', 'Sku', 'Name', 'LOT', 'Expected AvailableQty (ea)']
        )
    with sort_col2:
        sort_order = st.selectbox('Order', ['Ascending', 'Descending'])
    
    # Apply sorting
    filtered_df = filtered_df.sort_values(
        by=sort_by,
        ascending=(sort_order == 'Ascending')
    )
    
    # Apply styling to the dataframe
    def highlight_columns(x):
        df_styled = pd.DataFrame('', index=x.index, columns=x.columns)
        df_styled['Counted QTY'] = 'background-color: #e6ffe6'
        df_styled['Unit'] = 'background-color: #e6ffe6'
        df_styled['Notes'] = 'background-color: #e6ffe6'
        return df_styled

    # Display the data with styling
    st.dataframe(
        filtered_df.style.apply(highlight_columns, axis=None),
        use_container_width=True,
        hide_index=True
    )
    
    # Download buttons for filtered data
    st.subheader("Download Filtered Data")
    
    # Add note about export filtering
    st.caption("Note: Export includes items with positive Expected AvailableQty and empty rows for manual input")
    
    col1, col2 = st.columns(2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with col1:
        csv_filename = f"oxnard_inventory_{timestamp}.csv"
        # Filter out items with 0 or negative quantity but keep empty rows
        export_df = filtered_df.copy()
        mask = (export_df['Expected AvailableQty (ea)'].fillna(-1) <= 0) & (export_df['Expected AvailableQty (ea)'].notna())
        export_df = export_df[~mask]
        
        # Add 50 empty rows for export
        empty_rows = pd.DataFrame({
            'ItemId': [''] * 50,
            'Sku': [''] * 50,
            'Name': [''] * 50,
            'LOT': [''] * 50,
            'Expected AvailableQty (ea)': [np.nan] * 50,
            'Counted QTY': [''] * 50,
            'Unit': [''] * 50,
            'Notes': [''] * 50
        })
        export_df = pd.concat([export_df, empty_rows], ignore_index=True)
        
        csv_path = save_as_csv(export_df, csv_filename)
        with open(csv_path, 'rb') as f:
            st.download_button(
                label=f"📥 Download Filtered Data as CSV ({len(export_df)} items)",
                data=f,
                file_name=csv_filename,
                mime="text/csv"
            )
        os.remove(csv_path)
    
    with col2:
        excel_filename = f"oxnard_inventory_{timestamp}.xlsx"
        # Filter out items with 0 or negative quantity but keep empty rows
        export_df = filtered_df.copy()
        mask = (export_df['Expected AvailableQty (ea)'].fillna(-1) <= 0) & (export_df['Expected AvailableQty (ea)'].notna())
        export_df = export_df[~mask]
        
        # Add 50 empty rows for export
        empty_rows = pd.DataFrame({
            'ItemId': [''] * 50,
            'Sku': [''] * 50,
            'Name': [''] * 50,
            'LOT': [''] * 50,
            'Expected AvailableQty (ea)': [np.nan] * 50,
            'Counted QTY': [''] * 50,
            'Unit': [''] * 50,
            'Notes': [''] * 50
        })
        export_df = pd.concat([export_df, empty_rows], ignore_index=True)
        
        excel_path = save_as_excel(export_df, excel_filename, colorful=True)
        with open(excel_path, 'rb') as f:
            st.download_button(
                label=f"📥 Download Filtered Data as Excel ({len(export_df)} items)",
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