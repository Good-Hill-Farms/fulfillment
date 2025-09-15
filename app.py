import logging
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from constants.shipping_zones import load_shipping_zones

# Import utility modules
from utils.data_processor import DataProcessor
from utils.ui_components import (
    render_header,
    render_inventory_tab,
    render_orders_tab,
    render_sku_mapping_editor,
    render_staging_tab,
    render_workflow_status,
    render_summary_dashboard,
)

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

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

# Initialize staging system and workflow - LIGHTWEIGHT INITIALIZATION
if "staged_orders" not in st.session_state:
    st.session_state.staged_orders = pd.DataFrame()
if "staging_history" not in st.session_state:
    st.session_state.staging_history = []
if "workflow_initialized" not in st.session_state:
    st.session_state.workflow_initialized = False
if "staging_processor" not in st.session_state:
    st.session_state.staging_processor = None  # Initialize lazily when needed
if "sku_mappings" not in st.session_state:
    st.session_state.sku_mappings = None  # Load only when SKU Mapping tab is accessed

if "shortage_summary" not in st.session_state:
    st.session_state.shortage_summary = pd.DataFrame()
if "grouped_shortage_summary" not in st.session_state:
    st.session_state.grouped_shortage_summary = pd.DataFrame()
if "inventory_comparison" not in st.session_state:
    st.session_state.inventory_comparison = pd.DataFrame()
if "initial_inventory" not in st.session_state:
    st.session_state.initial_inventory = pd.DataFrame()
if "processing_stats" not in st.session_state:
    st.session_state.processing_stats = {}
if "warehouse_performance" not in st.session_state:
    st.session_state.warehouse_performance = {}
if "rules" not in st.session_state:
    st.session_state.rules = []
if "temp_rules" not in st.session_state:
    st.session_state.temp_rules = None
if "bundles" not in st.session_state:
    st.session_state.bundles = {}
if "override_log" not in st.session_state:
    st.session_state.override_log = []
# SKU mappings are now initialized above with staging_processor


def optimize_memory():
    """Optimize memory usage and clear unnecessary data"""

    # Clear large temporary DataFrames that aren't needed in session state
    temp_keys = [
        "temp_orders",
        "temp_inventory",
        "temp_processing",
        "large_dataframes",
        "cached_calculations",
    ]

    for key in temp_keys:
        if key in st.session_state:
            del st.session_state[key]

    # Limit staging history to last 50 entries
    if "staging_history" in st.session_state and len(st.session_state.staging_history) > 50:
        st.session_state.staging_history = st.session_state.staging_history[-50:]

    # Limit override log to last 100 entries
    if "override_log" in st.session_state and len(st.session_state.override_log) > 100:
        st.session_state.override_log = st.session_state.override_log[-100:]


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames duplicate columns by appending a suffix."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            f"{dup}_{i}" if i != 0 else dup for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df


def main():
    """Main application function - OPTIMIZED FOR FAST LOADING"""

    # Optimize memory usage
    optimize_memory()

    # Render header immediately
    render_header()

    # Initialize data processor once and cache it
    if "data_processor" not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    data_processor = st.session_state.data_processor

    # Sidebar for file uploads and configuration
    with st.sidebar:
        st.subheader("📤 Data Sources")

        # File uploaders
        orders_file = st.file_uploader(
            "Upload Orders CSV (e.g. from Shopify)", type="csv", key="orders_upload"
        )
        
        # Inventory source selection
        st.subheader("📦 Inventory Source")
        inventory_source = st.radio(
            "Choose inventory source:",
            ["Upload File", "ColdCart API"],
            key="inventory_source",
            index=0  # Default to Upload File
        )
        
        inventory_file = None
        if inventory_source == "Upload File":
            inventory_file = st.file_uploader(
                "Upload Inventory CSV (from warehouse)", type="csv", key="inventory_upload"
            )
        else:  # ColdCart API
            st.info("✅ Using automatic token extractor for real-time inventory")
            st.caption("🔄 Token automatically refreshed every 4 days")

        # Auto-process when orders are uploaded, but keep manual button option
        can_process = orders_file and (inventory_file or inventory_source == "ColdCart API")
        
        # Auto-process logic
        auto_process = False
        if can_process:
            # Check if this is a new orders file (different from what's already processed)
            if (orders_file is not None and 
                (st.session_state.get('last_processed_orders_file') != orders_file.name or
                 st.session_state.get('orders_df') is None)):
                auto_process = True
        
        # Manual process button (always show if can process)
        manual_process = False
        if can_process:
            if st.button("🔄 Reprocess Data", key="manual_process_files"):
                    manual_process = True
        
        # Process data (either auto or manual)
        if auto_process or manual_process:
            process_type = "Auto-processing" if auto_process else "Reprocessing"
            with st.spinner(f"{process_type} data..."):
                try:
                    # Step 1: Load orders
                    st.session_state.orders_df = data_processor.load_orders(orders_file)
                    
                    # Remember this file as processed
                    if orders_file:
                        st.session_state.last_processed_orders_file = orders_file.name
                    
                    # Step 2: Load inventory based on source
                    if inventory_source == "ColdCart API":
                        try:
                            st.session_state.inventory_df = data_processor.load_inventory(
                                source="coldcart"
                            )
                            st.success("✅ Successfully fetched inventory from ColdCart API")
                        except Exception as e:
                            st.error(f"❌ Failed to fetch from ColdCart API: {str(e)}")
                            st.stop()
                    else:  # Upload File
                        st.session_state.inventory_df = data_processor.load_inventory(
                            inventory_file, source="file"
                        )

                    # Initialize staging processor only when processing files
                    if st.session_state.staging_processor is None:
                        st.session_state.staging_processor = DataProcessor()

                    # Load shipping zones and cache them
                    if (
                        "shipping_zones_df" not in st.session_state
                        or st.session_state.shipping_zones_df is None
                    ):
                        st.session_state.shipping_zones_df = load_shipping_zones()

                    # Initialize staging workflow
                    result = st.session_state.staging_processor.initialize_workflow(
                        st.session_state.orders_df, st.session_state.inventory_df
                    )

                    # Store results in session state - with error checking
                    if "orders" in result:
                        st.session_state.processed_orders = result["orders"]

                        # Initialize staged flag if not exists to ensure orders appear in processing tab
                        if "staged" not in st.session_state.processed_orders.columns:
                            st.session_state.processed_orders["staged"] = False

                        # Update staged_orders as a filtered view of processed_orders
                        st.session_state.staged_orders = st.session_state.processed_orders[
                            st.session_state.processed_orders["staged"] == True
                        ].copy()

                    # Store other results - check if keys exist before accessing
                    if "inventory_summary" in result:
                        st.session_state.inventory_summary = result["inventory_summary"]

                    if "shortage_summary" in result:
                        st.session_state.shortage_summary = result["shortage_summary"]

                    if "grouped_shortage_summary" in result:
                        st.session_state.grouped_shortage_summary = result[
                            "grouped_shortage_summary"
                        ]

                    if "initial_inventory" in result:
                        st.session_state.initial_inventory = result["initial_inventory"]
                        # Always sync to staging_processor
                        if st.session_state.staging_processor is not None:
                            st.session_state.staging_processor.initial_inventory = (
                                st.session_state.initial_inventory
                            )

                    if "inventory_comparison" in result:
                        st.session_state.inventory_comparison = result["inventory_comparison"]

                    # Mark workflow as initialized
                    st.session_state.workflow_initialized = True

                    # Load SKU mappings only when workflow is initialized
                    if st.session_state.sku_mappings is None:
                        logger.info("Loading SKU mappings after workflow initialization...")
                        st.session_state.staging_processor.load_sku_mappings()
                        st.session_state.sku_mappings = (
                            st.session_state.staging_processor.sku_mappings
                        )
                        if (
                            st.session_state.sku_mappings is None
                            or not st.session_state.sku_mappings
                        ):
                            logger.warning(
                                "Failed to load SKU mappings. Using default empty structure."
                            )
                            st.session_state.sku_mappings = {
                                "Oxnard": {"singles": {}, "bundles": {}},
                                "Wheeling": {"singles": {}, "bundles": {}},
                                "Walnut": {"singles": {}, "bundles": {}},
                                "Northlake": {"singles": {}, "bundles": {}},
                            }

                    # Show warnings, errors, and skipped records if any were collected during processing
                    processor = st.session_state.staging_processor
                    if hasattr(processor, 'data_warnings') or hasattr(processor, 'data_errors') or hasattr(processor, 'skipped_records'):
                        warning_count = len(processor.data_warnings) if hasattr(processor, 'data_warnings') else 0
                        error_count = len(processor.data_errors) if hasattr(processor, 'data_errors') else 0
                        skipped_count = len(processor.skipped_records) if hasattr(processor, 'skipped_records') else 0
                        
                        total_issues = warning_count + error_count + skipped_count
                        
                        if total_issues > 0:
                            with st.expander(f"⚠️ {total_issues} Data Issue(s) - Click to view details", expanded=False):
                                
                                # Show skipped records first (most important)
                                if skipped_count > 0:
                                    st.error(f"🚫 {skipped_count} Records Skipped")
                                    skipped_df = pd.DataFrame([{
                                        'Order ID': record['record'].get('order_id', ''),
                                        'Customer': record['record'].get('customer', ''),
                                        'SKU': record['record'].get('sku', ''),
                                        'Fulfillment Center': record['record'].get('fulfillment_center', ''),
                                        'ZIP': record['record'].get('zip', ''),
                                        'Reason': record['reason']
                                    } for record in processor.skipped_records])
                                    
                                    st.dataframe(skipped_df, height=200, use_container_width=True, hide_index=True)
                                
                                # Show errors
                                if error_count > 0:
                                    st.error(f"❌ {error_count} Processing Errors")
                                    for error in processor.data_errors:
                                        st.error(f"• {error['error']}")
                                        if error.get('context'):
                                            st.caption(f"Context: {error['context']}")
                                
                                # Show general warnings
                                if warning_count > 0:
                                    st.warning(f"⚠️ {warning_count} Data Warnings")
                                    for warning in processor.data_warnings:
                                        st.warning(f"• {warning}")
                                
                                st.caption("Processing continued with the remaining valid data.")

                    if inventory_source == "Upload File":
                        if auto_process:
                            st.success("🚀 Auto-processed files successfully!")
                        else:
                            st.success("✅ Files reprocessed successfully!")
                    else:  # ColdCart API
                        if auto_process:
                            st.success("🚀 Auto-processed successfully with ColdCart inventory!")
                        else:
                            st.success("✅ Data reprocessed successfully with ColdCart inventory!")

                except Exception as e:
                    error_msg = f"Error processing data: {str(e)}"
                    st.error(error_msg)
                    # Also track the error if processor exists
                    if hasattr(st.session_state, 'staging_processor') and st.session_state.staging_processor:
                        st.session_state.staging_processor.add_data_error(error_msg, "File processing")

    # Always show workflow status
    # render_workflow_status()

    # Main content area
    if st.session_state.orders_df is not None and st.session_state.inventory_df is not None:
        # Create tabs for different sections
        orders_tab, staging_tab, inventory_tab, mapping_tab, dashboard_tab = st.tabs(
            ["📜 Orders", "📋 Staging", "📦 Inventory", "⚙️ SKU Mapping", "📊 Dashboard"]
        )

        # Orders Tab
        with orders_tab:
            render_orders_tab(st.session_state.processed_orders, st.session_state.shortage_summary)

        # Ensure staged column exists when processing files
        if (
            "processed_orders" in st.session_state
            and st.session_state.processed_orders is not None
            and not st.session_state.processed_orders.empty
        ):
            if "staged" not in st.session_state.processed_orders.columns:
                st.session_state.processed_orders["staged"] = False
                # Initial sync of staged_orders based on the staged flag
                st.session_state.staged_orders = st.session_state.processed_orders[
                    st.session_state.processed_orders["staged"] == True
                ].copy()

        # Initialize staged column if not present
        if (
            "processed_orders" in st.session_state
            and st.session_state.processed_orders is not None
            and not st.session_state.processed_orders.empty
        ):
            if "staged" not in st.session_state.processed_orders.columns:
                st.session_state.processed_orders["staged"] = False

        # Staging Tab
        with staging_tab:
            render_staging_tab()

        # Inventory Tab
        with inventory_tab:
            render_inventory_tab(
                st.session_state.shortage_summary,
                st.session_state.grouped_shortage_summary,
                st.session_state.initial_inventory
                if "initial_inventory" in st.session_state
                else None,
                st.session_state.inventory_comparison
                if "inventory_comparison" in st.session_state
                else None,
                st.session_state.processed_orders
                if "processed_orders" in st.session_state
                else None,
            )

        # SKU Mapping Tab
        with mapping_tab:
            # Load SKU mappings only when accessing this tab
            if (
                st.session_state.sku_mappings is None
                and st.session_state.staging_processor is not None
            ):
                with st.spinner("Loading SKU mappings..."):
                    st.session_state.staging_processor.load_sku_mappings()
                    st.session_state.sku_mappings = st.session_state.staging_processor.sku_mappings
                    if st.session_state.sku_mappings is None:
                        st.session_state.sku_mappings = {
                            "Oxnard": {"singles": {}, "bundles": {}},
                            "Wheeling": {"singles": {}, "bundles": {}},
                            "Walnut": {"singles": {}, "bundles": {}},
                            "Northlake": {"singles": {}, "bundles": {}},
                        }

            render_sku_mapping_editor(st.session_state.sku_mappings, st.session_state.data_processor)

        # Dashboard Tab
        with dashboard_tab:
            # OPTIMIZATION: Only load dashboard when user explicitly requests it
            # This prevents expensive dashboard calculations from running on every page load
            
            if st.button("📊 Load Dashboard Analytics", type="primary", key="load_dashboard"):
                st.session_state['dashboard_loaded'] = True
            
            if st.session_state.get('dashboard_loaded', False):
                render_summary_dashboard(
                    st.session_state.processed_orders,
                    st.session_state.inventory_df,
                    st.session_state.processing_stats if "processing_stats" in st.session_state else None,
                    st.session_state.warehouse_performance if "warehouse_performance" in st.session_state else None,
                )
            else:
                st.info("Click 'Load Dashboard Analytics' to view comprehensive order and inventory analytics.")
                
                # Show basic stats without expensive calculations
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.session_state.processed_orders is not None:
                        order_count = len(st.session_state.processed_orders)
                        st.metric("📋 Processed Orders", f"{order_count:,}")
                    else:
                        st.metric("📋 Processed Orders", "0")
                
                with col2:
                    if st.session_state.inventory_df is not None:
                        inventory_count = len(st.session_state.inventory_df)
                        st.metric("📦 Inventory Items", f"{inventory_count:,}")
                    else:
                        st.metric("📦 Inventory Items", "0")
                
                with col3:
                    if st.session_state.get('shortage_summary') is not None and not st.session_state.shortage_summary.empty:
                        shortage_count = len(st.session_state.shortage_summary)
                        st.metric("⚠️ Shortages", f"{shortage_count:,}")
                    else:
                        st.metric("⚠️ Shortages", "0")
    else:
        # Show minimal welcome message for faster loading
        st.info("👋 **Welcome to the AI-Powered Fulfillment Assistant!**")
        st.warning(
            "⚠️ **No data loaded yet**. Upload Orders and Inventory files in the sidebar - processing will happen automatically!"
        )

        # Show detailed instructions only if user clicks to expand
        if st.button("📖 Show Getting Started Guide"):
            st.markdown(
                """
            ### 🚀 Getting Started

            To begin using the smart fulfillment system:

            1. **📁 Upload Orders**: Use the sidebar to upload your Orders file - processing starts automatically!
            2. **📦 Inventory Source**: File upload is selected by default - upload your inventory CSV
            3. **🚀 Auto-Processing**: The system automatically processes your data when both files are uploaded
            4. **📊 Explore Results**: Navigate through the tabs to see orders, staging, inventory, and analytics

            ### 🎯 Key Features

            - **Smart Bundle Management**: Choose bundles in orders, change components, and apply with Available for Recalculation inventory
            - **Staging Workflow**: Stage orders to protect inventory allocations
            - **Real-time Recalculation**: Uses Initial - Staged inventory for smart recalculation
            - **Interactive Analytics**: Comprehensive dashboard with inventory insights
            """
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📜 Orders File Requirements:**")
                st.caption("• CSV format with order details")
                st.caption("• Must include order numbers and SKUs")
                st.caption("• Quantity and customer information")

            with col2:
                st.markdown("**📦 Inventory Source Options:**")
                st.caption("• **File Upload**: CSV with SKU and quantity columns (Default)")
                st.caption("• **ColdCart API**: Real-time inventory alternative option")
                st.caption("• Automatic warehouse normalization for both sources")


if __name__ == "__main__":
    main()
