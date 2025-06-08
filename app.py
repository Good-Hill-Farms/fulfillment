import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from constants.shipping_zones import load_shipping_zones

# Import utility modules
from utils.data_processor import DataProcessor
from utils.ui_components import (
    render_header,
    render_workflow_status,
    render_orders_tab,
    render_inventory_tab,
    render_staging_tab,
    render_sku_mapping_editor,
    render_summary_dashboard
)
from constants.schemas import SchemaManager, FulfillmentRule
from utils.airtable_handler import AirtableHandler
from utils.rule_manager import RuleManager

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

# Initialize staging system and workflow
if "staged_orders" not in st.session_state:
    st.session_state.staged_orders = pd.DataFrame()
if "staging_history" not in st.session_state:
    st.session_state.staging_history = []
if "workflow_initialized" not in st.session_state:
    st.session_state.workflow_initialized = False
if "staging_processor" not in st.session_state:
    st.session_state.staging_processor = DataProcessor()
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
if "sku_mappings" not in st.session_state:
    # Try to load SKU mappings from JSON file at startup
    try:
        import json
        with open('/Users/olenaliubynetska/fulfillment-mixy-matchi/airtable_sku_mappings.json', 'r') as f:
            st.session_state.sku_mappings = json.load(f)
    except Exception:
        st.session_state.sku_mappings = None


def main():
    """Main application function"""
    # Render header
    render_header()
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Sidebar for file uploads and configuration
    with st.sidebar:
        st.subheader("📤 Upload Files")
        
        # File uploaders
        orders_file = st.file_uploader("Upload Orders CSV (e.g. from Shopify)", type="csv", key="orders_upload")
        inventory_file = st.file_uploader("Upload Inventory CSV (from warehouse)", type="csv", key="inventory_upload")
        
        if orders_file and inventory_file:
            if st.button("Process Files", key="process_files"):
                with st.spinner("Processing files..."):
                    try:
                        # Step 1 & 2: Load and parse files
                        st.session_state.orders_df = data_processor.load_orders(orders_file)
                        st.session_state.inventory_df = data_processor.load_inventory(inventory_file)
                        
                        # Load shipping zones
                        st.session_state.shipping_zones_df = load_shipping_zones()
                        
                        # Initialize staging workflow
                        result = st.session_state.staging_processor.initialize_workflow(
                            st.session_state.orders_df,
                            st.session_state.inventory_df
                        )
                        
                        # Store results in session state - with error checking
                        if 'orders' in result:
                            st.session_state.processed_orders = result['orders']
                            
                            # Initialize staged flag if not exists to ensure orders appear in processing tab
                            if 'staged' not in st.session_state.processed_orders.columns:
                                st.session_state.processed_orders['staged'] = False
                            
                            # Update staged_orders as a filtered view of processed_orders
                            st.session_state.staged_orders = st.session_state.processed_orders[st.session_state.processed_orders['staged'] == True].copy()
                        
                        # Store other results - check if keys exist before accessing
                        if 'inventory_summary' in result:
                            st.session_state.inventory_summary = result['inventory_summary']
                        
                        if 'shortage_summary' in result:
                            st.session_state.shortage_summary = result['shortage_summary']
                        
                        if 'grouped_shortage_summary' in result:
                            st.session_state.grouped_shortage_summary = result['grouped_shortage_summary']
                        
                        if 'initial_inventory' in result:
                            st.session_state.initial_inventory = result['initial_inventory']
                        
                        if 'inventory_comparison' in result:
                            st.session_state.inventory_comparison = result['inventory_comparison']
                        
                        # Mark workflow as initialized
                        st.session_state.workflow_initialized = True
                        
                        # Load shipping zones for compatibility
                        st.session_state.shipping_zones_df = load_shipping_zones()
                        
                        # Load SKU mappings directly from JSON file
                        try:
                            import json
                            with open('/Users/olenaliubynetska/fulfillment-mixy-matchi/airtable_sku_mappings.json', 'r') as f:
                                st.session_state.sku_mappings = json.load(f)
                        except Exception as e:
                            st.warning(f"Could not load SKU mappings from JSON: {e}")
                            st.session_state.sku_mappings = st.session_state.staging_processor.sku_mappings
                        
                        st.success("✅ Files processed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
    
    # Always show workflow status
    render_workflow_status()
    
    # Main content area - only show if data is loaded
    if st.session_state.orders_df is not None and st.session_state.inventory_df is not None:
        # Create tabs for different sections
        orders_tab, staging_tab, inventory_tab, dashboard_tab, mapping_tab = st.tabs([
            "📜 Orders",
            "📋 Staging",
            "📦 Inventory",
            "📈 Dashboard",
            "⚙️ SKU Mapping"
        ])
        
        # Orders Tab
        with orders_tab:
            render_orders_tab(
                st.session_state.processed_orders,
                st.session_state.shortage_summary
            )
        
        # Ensure staged column exists when processing files
        if 'processed_orders' in st.session_state and not st.session_state.processed_orders.empty:
            if 'staged' not in st.session_state.processed_orders.columns:
                st.session_state.processed_orders['staged'] = False
                # Initial sync of staged_orders based on the staged flag
                st.session_state.staged_orders = st.session_state.processed_orders[st.session_state.processed_orders['staged'] == True].copy()
        
        # Ensure staged_orders is up-to-date before rendering tab
        if 'processed_orders' in st.session_state and 'staged' in st.session_state.processed_orders.columns:
            # Always sync staged_orders from processed_orders before showing the tab
            st.session_state.staged_orders = st.session_state.processed_orders[st.session_state.processed_orders['staged'] == True].copy()
            
            # Increment the key_prefix if it exists to ensure UI refresh
            if 'key_prefix' in st.session_state:
                st.session_state.key_prefix += 1
        
        # Staging Tab
        with staging_tab:
            render_staging_tab(
                st.session_state.staged_orders if 'staged_orders' in st.session_state 
                else pd.DataFrame(),
                st.session_state.staging_processor if 'workflow_initialized' in st.session_state and st.session_state.workflow_initialized else None
            )
        
        # Inventory Tab
        with inventory_tab:
            render_inventory_tab(
                st.session_state.shortage_summary,
                st.session_state.grouped_shortage_summary,
                st.session_state.initial_inventory if 'initial_inventory' in st.session_state else None,
                st.session_state.inventory_comparison if 'inventory_comparison' in st.session_state else None
            )
        
        # Dashboard Tab
        with dashboard_tab:
            render_summary_dashboard(
                st.session_state.processed_orders,
                st.session_state.inventory_df,
                st.session_state.processing_stats,
                st.session_state.warehouse_performance
            )
        
        # SKU Mapping Tab
        with mapping_tab:
            render_sku_mapping_editor(
                st.session_state.sku_mappings,
                data_processor
            )

if __name__ == "__main__":
    main()
