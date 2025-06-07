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

# Initialize staging system
if "staged_orders" not in st.session_state:
    st.session_state.staged_orders = pd.DataFrame()
if "staging_history" not in st.session_state:
    st.session_state.staging_history = []
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
                        
                        # Load SKU mappings
                        if st.session_state.sku_mappings is None:
                            st.session_state.sku_mappings = data_processor.load_sku_mappings()
                        
                        # Process orders with all available data
                        result = data_processor.process_orders(
                            st.session_state.orders_df,
                            st.session_state.inventory_df,
                            st.session_state.shipping_zones_df,
                            st.session_state.sku_mappings
                        )
                        
                        # Store results in session state
                        st.session_state.processed_orders = result['orders']
                        st.session_state.inventory_summary = result['inventory_summary']
                        st.session_state.shortage_summary = result['shortage_summary']
                        st.session_state.grouped_shortage_summary = result['grouped_shortage_summary']
                        st.session_state.initial_inventory = result['initial_inventory']
                        st.session_state.inventory_comparison = result['inventory_comparison']
                        
                        # Calculate additional metrics
                        st.session_state.processing_stats = data_processor.calculate_processing_stats(
                            st.session_state.processed_orders,
                            st.session_state.inventory_summary,
                            st.session_state.shortage_summary
                        )
                        
                        st.session_state.warehouse_performance = data_processor.calculate_warehouse_performance(
                            st.session_state.processed_orders,
                            st.session_state.inventory_summary
                        )
                        
                        st.success("✅ Files processed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
    
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
        
        # Staging Tab
        with staging_tab:
            render_staging_tab(
                st.session_state.staged_orders if 'staged_orders' in st.session_state 
                else pd.DataFrame()
            )
        
        # Inventory Tab
        with inventory_tab:
            render_inventory_tab(
                st.session_state.inventory_summary,
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
