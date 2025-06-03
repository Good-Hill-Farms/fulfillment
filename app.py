import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from constants.shipping_zones import load_shipping_zones

# Import utility modules
from utils.data_processor import DataProcessor
from utils.ui_components import render_header, render_summary_dashboard, render_inventory_analysis

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="🍍 AI-Powered Fulfillment Assistant",
    page_icon="🍍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dashboard layout
st.markdown(
    """
<style>
.reportview-container .main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 2rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "orders_df" not in st.session_state:
    st.session_state.orders_df = None
if "inventory_df" not in st.session_state:
    st.session_state.inventory_df = None
if "shipping_zones_df" not in st.session_state:
    st.session_state.shipping_zones_df = None
if "processed_orders" not in st.session_state:
    st.session_state.processed_orders = None
if "inventory_summary" not in st.session_state:
    st.session_state.inventory_summary = pd.DataFrame()
if "shortage_summary" not in st.session_state:
    st.session_state.shortage_summary = pd.DataFrame()
if "grouped_shortage_summary" not in st.session_state:
    st.session_state.grouped_shortage_summary = pd.DataFrame()
if "rules" not in st.session_state:
    st.session_state.rules = []
if "bundles" not in st.session_state:
    st.session_state.bundles = {}
if "override_log" not in st.session_state:
    st.session_state.override_log = []
if "sku_mappings" not in st.session_state:
    st.session_state.sku_mappings = None


def main():
    """Main application function"""

    # Render header
    render_header()
    
    # Sidebar for configuration
    with st.sidebar:
        # API key configuration
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            st.warning("⚠️ OpenRouter API key not found. Please add it to your .env file.")

        # Upload section
        st.subheader("📤 Upload Files")
        orders_file = st.file_uploader("Upload Orders CSV", type="csv", key="orders_upload")
        inventory_file = st.file_uploader(
            "Upload Inventory CSV", type="csv", key="inventory_upload"
        )

        if orders_file and inventory_file:
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    # Initialize data processor
                    data_processor = DataProcessor()

                    # Load and process data
                    st.session_state.orders_df = data_processor.load_orders(orders_file)
                    st.session_state.inventory_df = data_processor.load_inventory(inventory_file)

                    # Load shipping zones data from constants directory
                    try:
                        st.session_state.shipping_zones_df = load_shipping_zones()
                    except Exception as e:
                        st.error(f"Error loading shipping zones: {str(e)}")
                        st.session_state.shipping_zones_df = pd.DataFrame(
                            columns=["zip_prefix", "moorpark_zone", "wheeling_zip", "wheeling_zone"]
                        )

                    # Process orders if all required data is available
                    if (
                        st.session_state.orders_df is not None
                        and st.session_state.inventory_df is not None
                    ):
                        # Load shipping zones if not already loaded
                        if st.session_state.shipping_zones_df is None:
                            shipping_zones_path = os.path.join(
                                os.path.dirname(__file__), "docs", "shipping_zones.csv"
                            )
                            if os.path.exists(shipping_zones_path):
                                st.session_state.shipping_zones_df = pd.read_csv(
                                    shipping_zones_path
                                )
                                st.success(
                                    f"✅ Loaded shipping zones data: {len(st.session_state.shipping_zones_df)} zones"
                                )
                            else:
                                st.warning(
                                    "⚠️ Shipping zones data not found. Using default shipping logic."
                                )

                        # Load SKU mappings if not already loaded
                        if st.session_state.sku_mappings is None:
                            st.session_state.sku_mappings = data_processor.load_sku_mappings()
                            # Only show warning if mappings couldn't be loaded
                            if not st.session_state.sku_mappings:
                                st.warning(
                                    "⚠️ SKU mappings could not be loaded. Some SKUs may not be properly matched."
                                )

                        # Process orders
                        result = data_processor.process_orders(
                            st.session_state.orders_df,
                            st.session_state.inventory_df,
                            st.session_state.shipping_zones_df,
                            st.session_state.sku_mappings,
                        )
                        
                        # Store the results in session state
                        st.session_state.processed_orders = result['orders']
                        st.session_state.inventory_summary = result['inventory_summary']
                        st.session_state.shortage_summary = result['shortage_summary']
                        st.session_state.grouped_shortage_summary = result['grouped_shortage_summary']

                st.success("✅ Files processed successfully!")
                st.rerun()

    # Main content area
    if st.session_state.orders_df is not None and st.session_state.inventory_df is not None:
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📜 Orders", "📦 Inventory", "📈 Dashboard", "⚙️ Rules"])

        with tab1:
            st.header("📜 Processed Orders")
            if st.session_state.processed_orders is not None:
                # Convert processed orders to CSV
                csv = st.session_state.processed_orders.to_csv(index=False)

                # Provide download button
                st.download_button(
                    label="Download Processed Orders CSV",
                    data=csv,
                    file_name="processed_orders.csv",
                    mime="text/csv",
                )

                # Display all processed orders in an editable table
                st.subheader("All Processed Orders (Editable)")

                # Create a unique key for the edited dataframe
                if "edited_orders" not in st.session_state:
                    st.session_state.edited_orders = st.session_state.processed_orders.copy()

                # Use the original dataframe without custom filtering
                # Streamlit's built-in filtering will handle this
                filtered_df = st.session_state.edited_orders.copy()

                # Create column configuration for data editor with improved formatting
                column_config = {}

                # Ensure externalorderid and id are treated as string to avoid type conflicts
                for id_col in ['externalorderid', 'id']:
                    if id_col in filtered_df.columns:
                        filtered_df[id_col] = filtered_df[id_col].astype(str)
                
                # Configure columns with appropriate types and formats
                for col in filtered_df.columns:
                    # Format date columns
                    if "date" in col.lower():
                        column_config[col] = st.column_config.DatetimeColumn(
                            col,
                            format="YYYY-MM-DD HH:mm",
                            required=False,
                            help=f"Date/time for {col}",
                        )
                    # Handle order ID columns as text
                    elif col == 'externalorderid' or col == 'id':
                        column_config[col] = st.column_config.TextColumn(
                            col,
                            required=False,
                            help=f"Order ID: {col}"
                        )
                    # Format numeric columns
                    elif (
                        filtered_df[col].dtype in ["int64", "float64"] or "quantity" in col.lower()
                    ):
                        column_config[col] = st.column_config.NumberColumn(
                            col,
                            format="%d" if filtered_df[col].dtype == "int64" else "%.2f",
                            required=False,
                        )

                # Make the dataframe editable with Streamlit's built-in column configuration
                edited_df = st.data_editor(
                    filtered_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    hide_index=False,
                    column_config=column_config,
                    height=500,
                    disabled=False,
                    key="main_table_editor",
                )

                # Save changes button
                if st.button("Save Changes"):
                    st.session_state.processed_orders = edited_df.copy()
                    st.session_state.edited_orders = edited_df.copy()
                    st.success("✅ Changes saved successfully!")

                    # Update CSV download with the edited data
                    csv = st.session_state.processed_orders.to_csv(index=False)
                    st.download_button(
                        label="Download Updated Orders CSV",
                        data=csv,
                        file_name="updated_processed_orders.csv",
                        mime="text/csv",
                        key="download_updated",
                    )
        with tab2:            
            # Move inventory shortages to the inventory tab
            if "shortage_summary" in st.session_state and not st.session_state.shortage_summary.empty:
                with st.expander(f"⚠️ INVENTORY SHORTAGES DETECTED: {len(st.session_state.shortage_summary)} items", expanded=True):
                    if "grouped_shortage_summary" in st.session_state and not st.session_state.grouped_shortage_summary.empty:
                        grouped_df = st.session_state.grouped_shortage_summary.copy()
                        
                        # Format the order_ids list to be more readable
                        if 'order_ids' in grouped_df.columns:
                            grouped_df['order_ids'] = grouped_df['order_ids'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
                        
                        if 'issue' in grouped_df.columns:
                            # Highlight rows with issues
                            st.markdown("**⚠️ Rows with issues are highlighted**")
                            # Create a styled dataframe
                            styled_df = grouped_df.style.apply(
                                lambda row: ['background-color: #ffcccc' if row['issue'] else '' for _ in row],
                                axis=1
                            )
                            st.dataframe(styled_df)
                        else:
                            st.dataframe(grouped_df)
                        
                        # Provide download button for grouped shortage summary
                        csv = st.session_state.grouped_shortage_summary.to_csv(index=False)
                        st.download_button(
                            label="Download Grouped Shortage Summary",
                            data=csv,
                            file_name="grouped_shortage_summary.csv",
                            mime="text/csv",
                        )
            
            # Add inventory analysis section
            render_inventory_analysis(
                st.session_state.processed_orders, st.session_state.inventory_df
            )
            
        with tab3:
            
            # Original dashboard content
            render_summary_dashboard(
                st.session_state.processed_orders, st.session_state.inventory_df
            )
    else:
        # Welcome screen
        st.header("🍍 Welcome to the AI-Powered Fulfillment Assistant")
        st.write(
            """
        This application helps you assign customer fruit orders to fulfillment centers using:
        - Uploaded CSVs (orders_placed.csv, inventory.csv)
        - LLM-enhanced logic (OpenRouter: Claude, GPT)
        - Rules (zip code → warehouse, fruit bundles, priority)
        - Editable dashboard with explanations
        - Final exportable CSV in structured format

        To get started, please upload your order and inventory CSV files using the sidebar.
        """
        )

if __name__ == "__main__":
    main()
