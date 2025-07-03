import streamlit as st
import os
import pandas as pd
from datetime import datetime
from utils.inventory_api import get_inventory_data, save_as_excel, save_as_csv
import pytz

st.set_page_config(
    page_title="Wheeling Inventory | ColdCart | SKU grouped",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Wheeling Inventory | ColdCart | SKU grouped")
    
    # Add current date in Los Angeles timezone
    la_tz = pytz.timezone('America/Los_Angeles')
    current_time_la = datetime.now(la_tz)
    st.markdown(f"### Data as of {current_time_la.strftime('%Y-%m-%d %H:%M:%S')} (Los Angeles)")
    
    st.markdown("""
    This page allows you to download and analyze the current inventory data for the Wheeling warehouse.
    The data includes SKUs, quantities, and other relevant information.
    """)
    
    # Add a refresh button
    if st.button("🔄 Refresh Inventory Data"):
        with st.spinner("Fetching latest inventory data..."):
            try:
                df = get_inventory_data()
                if df is not None:
                    # Filter for Wheeling warehouse
                    df = df[df['WarehouseName'] == 'IL-Wheeling-60090'].copy()
                    # Convert ItemId to integer
                    df['ItemId'] = df['ItemId'].astype(int)
                    
                    # Create a summary DataFrame with total quantities and additional columns
                    summary_df = df.groupby(['ItemId', 'Sku', 'Name', 'Type'], as_index=False, group_keys=False).agg({
                        'AvailableQty': 'sum'
                    }).rename(columns={'AvailableQty': 'Expected AvailableQty'})
                    
                    # Add empty columns for QTY/LOT groups
                    summary_df['QTY_1'] = ''
                    summary_df['LOT_1'] = ''
                    summary_df['QTY_2'] = ''
                    summary_df['LOT_2'] = ''
                    summary_df['QTY_3'] = ''
                    summary_df['LOT_3'] = ''
                    summary_df['Notes'] = ''
                    
                    # Create a DataFrame with batch details
                    batch_df = df.copy()
                    # Include all quantities, not just positive ones
                    batch_df = batch_df.sort_values('BatchCode', ascending=True)
                    
                    # Add batch information as a list to the summary
                    def format_batch_info(group):
                        batch_details = []
                        for _, row in group.iterrows():
                            batch_code = row['BatchCode'] if pd.notna(row['BatchCode']) and row['BatchCode'].strip() != '' else '#noBatchCode'
                            # Show all quantities, including negative ones
                            batch_details.append(f"{batch_code}: {int(row['AvailableQty'])}")
                        return ', '.join(batch_details)

                    batch_info = batch_df.groupby(['ItemId', 'Sku'], group_keys=False).apply(format_batch_info).reset_index()
                    batch_info.columns = ['ItemId', 'Sku', 'BatchDetails']
                    
                    # Merge batch information with summary
                    summary_df = summary_df.merge(batch_info, on=['ItemId', 'Sku'], how='left')
                    
                    # Format ItemId as integer without commas
                    summary_df['ItemId'] = summary_df['ItemId'].astype(str)
                    
                    # Reorder columns
                    summary_df = summary_df[['ItemId', 'Sku', 'Name', 'Type', 'Expected AvailableQty', 
                                          'QTY_1', 'LOT_1', 'QTY_2', 'LOT_2', 'QTY_3', 'LOT_3', 'Notes',
                                          'BatchDetails']]
                    
                    # Store DataFrame in session state
                    st.session_state['wheeling_df'] = summary_df
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
    if 'wheeling_df' not in st.session_state:
        with st.spinner("Fetching initial inventory data..."):
            try:
                df = get_inventory_data()
                if df is not None:
                    # Filter for Wheeling warehouse
                    df = df[df['WarehouseName'] == 'IL-Wheeling-60090'].copy()
                    # Convert ItemId to integer
                    df['ItemId'] = df['ItemId'].astype(int)
                    
                    # Create a summary DataFrame with total quantities and additional columns
                    summary_df = df.groupby(['ItemId', 'Sku', 'Name', 'Type'], as_index=False, group_keys=False).agg({
                        'AvailableQty': 'sum'
                    }).rename(columns={'AvailableQty': 'Expected AvailableQty'})
                    
                    # Add empty columns for QTY/LOT groups
                    summary_df['QTY_1'] = ''
                    summary_df['LOT_1'] = ''
                    summary_df['QTY_2'] = ''
                    summary_df['LOT_2'] = ''
                    summary_df['QTY_3'] = ''
                    summary_df['LOT_3'] = ''
                    summary_df['Notes'] = ''
                    
                    # Create a DataFrame with batch details
                    batch_df = df.copy()
                    # Include all quantities, not just positive ones
                    batch_df = batch_df.sort_values('BatchCode', ascending=True)
                    
                    # Add batch information as a list to the summary
                    def format_batch_info(group):
                        batch_details = []
                        for _, row in group.iterrows():
                            batch_code = row['BatchCode'] if pd.notna(row['BatchCode']) and row['BatchCode'].strip() != '' else '#noBatchCode'
                            # Show all quantities, including negative ones
                            batch_details.append(f"{batch_code}: {int(row['AvailableQty'])}")
                        return ', '.join(batch_details)

                    batch_info = batch_df.groupby(['ItemId', 'Sku'], group_keys=False).apply(format_batch_info).reset_index()
                    batch_info.columns = ['ItemId', 'Sku', 'BatchDetails']
                    
                    # Merge batch information with summary
                    summary_df = summary_df.merge(batch_info, on=['ItemId', 'Sku'], how='left')
                    
                    # Format ItemId as integer without commas
                    summary_df['ItemId'] = summary_df['ItemId'].astype(str)
                    
                    # Reorder columns
                    summary_df = summary_df[['ItemId', 'Sku', 'Name', 'Type', 'Expected AvailableQty', 
                                          'QTY_1', 'LOT_1', 'QTY_2', 'LOT_2', 'QTY_3', 'LOT_3', 'Notes',
                                          'BatchDetails']]
                    
                    # Store DataFrame in session state
                    st.session_state['wheeling_df'] = summary_df
                else:
                    st.error("❌ No data received from the API")
                    st.info("Please check if your API token is correctly set in the environment variables.")
                    return
            except Exception as e:
                st.error(f"❌ Error fetching data: {str(e)}")
                st.info("Please check if your API token is correctly set in the environment variables.")
                return
    
    df = st.session_state.get('wheeling_df')
    if df is None:
        st.error("❌ No inventory data available")
        st.info("Please try refreshing the data using the button above.")
        return
    
    # Data Filtering Section
    st.subheader("Data Filters")
    
    # Create columns for filters
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # Type filter
        types = ['All'] + sorted(df['Type'].unique().tolist())
        selected_type = st.selectbox('Filter by Type', types)
        
        # SKU search
        sku_search = st.text_input('Search by SKU')
    
    with filter_col2:
        # Name search
        name_search = st.text_input('Search by Name')
        
        # Show only items with stock
        show_in_stock = st.checkbox('Show Only Items in Stock', value=False)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['Type'] == selected_type]
    
    if sku_search:
        filtered_df = filtered_df[filtered_df['Sku'].str.contains(sku_search, case=False, na=False)]
    
    if name_search:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(name_search, case=False, na=False)]
    
    if show_in_stock:
        filtered_df = filtered_df[filtered_df['Expected AvailableQty'] > 0]
    
    # Display filtered data statistics
    st.write(f"Showing {len(filtered_df)} out of {len(df)} items")
    
    # Sort options
    sort_col1, sort_col2 = st.columns(2)
    with sort_col1:
        sort_by = st.selectbox(
            'Sort by',
            ['ItemId', 'Sku', 'Name', 'Type', 'Expected AvailableQty']
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
    
    # Add note about export filtering
    st.caption("Note: Export includes only items with Expected AvailableQty > 0")
    
    col1, col2 = st.columns(2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with col1:
        csv_filename = f"wheeling_inventory_{timestamp}.csv"
        # Filter for available items
        export_df = filtered_df[filtered_df['Expected AvailableQty'] > 0].copy()
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
        excel_filename = f"wheeling_inventory_{timestamp}.xlsx"
        # Filter for available items
        export_df = filtered_df[filtered_df['Expected AvailableQty'] > 0].copy()
        excel_path = save_as_excel(export_df, excel_filename, colorful=True)  # Add colorful parameter
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