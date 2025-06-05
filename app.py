import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from st_aggrid import (
    AgGrid,
    ColumnsAutoSizeMode,
    DataReturnMode,
    GridOptionsBuilder,
    GridUpdateMode,
)

from constants.shipping_zones import load_shipping_zones

# Import utility modules
from utils.data_processor import DataProcessor
from utils.ui_components import render_header, render_inventory_analysis, render_summary_dashboard, create_aggrid_table

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
if "inventory_comparison" not in st.session_state:
    st.session_state.inventory_comparison = pd.DataFrame()
if "processing_stats" not in st.session_state:
    st.session_state.processing_stats = {}
if "warehouse_performance" not in st.session_state:
    st.session_state.warehouse_performance = {}
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
                        st.session_state.inventory_comparison = result.get('inventory_comparison', pd.DataFrame())
                        
                        # Calculate processing statistics for decision making
                        st.session_state.processing_stats = data_processor.calculate_processing_stats(
                            st.session_state.processed_orders,
                            st.session_state.inventory_summary,
                            st.session_state.shortage_summary
                        )
                        
                        # Calculate warehouse performance metrics
                        st.session_state.warehouse_performance = data_processor.calculate_warehouse_performance(
                            st.session_state.processed_orders,
                            st.session_state.inventory_summary
                        )

                st.success("✅ Files processed successfully!")
                st.rerun()

    # Main content area
    if st.session_state.orders_df is not None and st.session_state.inventory_df is not None:
        # Create tabs for different sections
        tab1, tab2, tab3, tab4= st.tabs(["📜 Orders", "📦 Inventory", "📈 Dashboard", "⚙️ Rules"])

        with tab1:
            st.header("📜 Processed Orders")
            if st.session_state.processed_orders is not None:
                # Calculate order statistics
                orders_df = st.session_state.processed_orders
                total_orders = len(orders_df['ordernumber'].unique()) if 'ordernumber' in orders_df.columns else 0
                total_line_items = len(orders_df)
                
                # Calculate issues
                issues_count = 0
                if 'Issues' in orders_df.columns:
                    issues_count = len(orders_df[orders_df['Issues'] != ""])
                
                # Display key statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📋 Unique Orders", f"{total_orders:,}")
                with col2:
                    st.metric("📦 Line Items", f"{total_line_items:,}")
                with col3:
                    if issues_count > 0:
                        st.metric("⚠️ Items with Issues", f"{issues_count:,}", delta=f"{(issues_count/total_line_items*100):.1f}% of total", delta_color="inverse")
                    else:
                        st.metric("✅ Items with Issues", "0", delta="Perfect!", delta_color="normal")
                with col4:
                    fulfillment_centers = orders_df['Fulfillment Center'].nunique() if 'Fulfillment Center' in orders_df.columns else 0
                    st.metric("🏭 Warehouses Used", fulfillment_centers)
                
                # Enhanced navigation hints
                if issues_count > 0:
                    st.warning(f"⚠️ **{issues_count} items have issues** - Check the **📦 Inventory** tab for shortage details and **📈 Dashboard** for performance insights!")
                else:
                    st.success("✅ **All orders processed successfully!** - Visit **📈 Dashboard** for performance analytics and **📦 Inventory** for balance tracking.")
                
                # Usage hints
                with st.expander("💡 How to analyze your orders", expanded=False):
                    st.markdown("""
                    **🔍 Quick Actions:**
                    - **Filter by Issues**: Click the Issues column filter to see only problematic orders
                    - **Group by Warehouse**: Drag "Fulfillment Center" to the Row Groups area to see distribution
                    - **Sort by Quantity**: Click "Transaction Quantity" header to find largest orders
                    
                    **📊 Next Steps:**
                    - **📦 Inventory Tab**: View shortage details and inventory balance changes
                    - **📈 Dashboard**: Analyze warehouse performance and get decision-making insights
                    - **Download Data**: Use the download buttons below the table for further analysis
                    
                    **🎯 Pro Tips:**
                    - Use Ctrl/Cmd + click to select multiple rows for bulk analysis
                    - Right-click column headers for advanced options
                    - Filter by warehouse to focus on specific fulfillment center performance
                    """)

                # Display all processed orders with advanced filtering and grouping
                st.subheader("All Processed Orders")

                # Prepare the dataframe for ag-Grid
                display_df = st.session_state.processed_orders.copy()
                
                # Convert ID columns to string to prevent issues
                for id_col in ['externalorderid', 'id', 'ordernumber']:
                    if id_col in display_df.columns:
                        display_df[id_col] = display_df[id_col].astype(str)
                
                # Create ag-Grid table with advanced features
                table_result = create_aggrid_table(
                    display_df,
                    height=600,
                    key="orders_main_grid",
                    theme="alpine",
                    editable=False,  # Disable editing to prevent JSON serialization issues
                    selection_mode='multiple',
                    enable_enterprise_modules=False  # Disable to prevent JSON serialization issues
                )
                
                grid_response = table_result['grid_response']
                
                # Display additional information about selected rows
                if table_result['selected_count'] > 0:
                    # Show selected rows summary
                    with st.expander("Selected Rows Summary", expanded=False):
                        selected_df = pd.DataFrame(grid_response['selected_rows'])
                        if 'Fulfillment Center' in selected_df.columns:
                            fc_counts = selected_df['Fulfillment Center'].value_counts()
                            st.write("**Selected rows by Fulfillment Center:**")
                            for fc, count in fc_counts.items():
                                st.write(f"- {fc}: {count} items")
                        
                        # Show quantity summary if available
                        if 'Transaction Quantity' in selected_df.columns:
                            total_qty = pd.to_numeric(selected_df['Transaction Quantity'], errors='coerce').sum()
                            st.write(f"**Total Quantity in Selection:** {total_qty:,.0f} units")

        with tab2:            
            # Move inventory shortages to the inventory tab
            if "shortage_summary" in st.session_state and not st.session_state.shortage_summary.empty:
                with st.expander(f"⚠️ INVENTORY SHORTAGES DETECTED: {len(st.session_state.shortage_summary)} items", expanded=True):
                    if "grouped_shortage_summary" in st.session_state and not st.session_state.grouped_shortage_summary.empty:
                        grouped_df = st.session_state.grouped_shortage_summary.copy()
                        
                        # Format the order_ids list to be more readable
                        if 'order_ids' in grouped_df.columns:
                            grouped_df['order_ids'] = grouped_df['order_ids'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x)
                        
                        st.markdown("**Grouped Shortage Summary** (Use sidebar to filter and group)")
                        
                        # Use ag-Grid for shortage summary (disable hints to avoid nested expanders)
                        shortage_table = create_aggrid_table(
                            grouped_df,
                            height=400,
                            key="shortage_summary_grid",
                            theme="alpine",
                            selection_mode='multiple',
                            show_hints=False,
                            enable_enterprise_modules=False  # Disable to prevent JSON serialization issues
                        )
                        
                        # Show summary of selected shortage items
                        if shortage_table['selected_count'] > 0:
                            st.info(f"🔍 Selected {shortage_table['selected_count']} shortage groups for detailed analysis")
                    
                    # Also show detailed shortage summary
                    st.markdown("---")
                    st.markdown("**Detailed Shortage Summary**")
                    detailed_shortage_df = st.session_state.shortage_summary.copy()
                    
                    # Create ag-Grid for detailed shortages (disable hints to avoid nested expanders)
                    detailed_table = create_aggrid_table(
                        detailed_shortage_df,
                        height=500,
                        key="detailed_shortage_grid",
                        theme="alpine",
                        selection_mode='multiple',
                        show_hints=False,
                        enable_enterprise_modules=False  # Disable to prevent JSON serialization issues
                    )
                    
                    # Show detailed shortage analysis
                    if detailed_table['selected_count'] > 0:
                        selected_details = pd.DataFrame(detailed_table['grid_response']['selected_rows'])
                        if 'shortage_qty' in selected_details.columns:
                            total_shortage = pd.to_numeric(selected_details['shortage_qty'], errors='coerce').sum()
                            st.warning(f"⚠️ Selected items show {total_shortage:,.0f} units in shortage")
            
            # Add inventory analysis section
            render_inventory_analysis(
                st.session_state.processed_orders, st.session_state.inventory_df
            )
            
        with tab3:
            # Enhanced dashboard content with analytics
            render_summary_dashboard(
                st.session_state.processed_orders, 
                st.session_state.inventory_df,
                st.session_state.get('processing_stats', {}),
                st.session_state.get('warehouse_performance', {})
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
