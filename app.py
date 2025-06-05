import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from constants.shipping_zones import load_shipping_zones

# Import utility modules
from utils.data_processor import DataProcessor
from utils.ui_components import render_header, render_inventory_analysis, render_summary_dashboard, create_aggrid_table
from constants.schemas import SchemaManager
from utils.airtable_handler import AirtableHandler

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
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📜 Orders", "📋 Staging", "📦 Inventory", "📈 Dashboard", "⚙️ Rules"])

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
 

                # Display all processed orders with advanced filtering and grouping
                st.subheader("All Processed Orders")

                # Prepare the dataframe for ag-Grid
                display_df = st.session_state.processed_orders.copy()
                
                # Add staged column to processed_orders if it doesn't exist
                if 'staged' not in st.session_state.processed_orders.columns:
                    st.session_state.processed_orders['staged'] = False
                
                # Add staged column to display_df if it doesn't exist
                if 'staged' not in display_df.columns:
                    display_df['staged'] = False
                
                # Filter out staged orders for main display
                original_count = len(display_df)
                staged_orders = display_df[display_df['staged'] == True].copy()
                display_df = display_df[display_df['staged'] == False].copy()
                filtered_count = len(display_df)
                staged_count = len(staged_orders)
                
                # Debug info (can be removed later)
                if staged_count > 0:
                    st.success(f"✅ **Staging Active**: {staged_count} orders moved to staging. Main table now shows {filtered_count} orders (was {original_count})")
                    
                    # Show which orders were staged
                    if 'ordernumber' in staged_orders.columns:
                        staged_order_nums = staged_orders['ordernumber'].head(5).tolist()
                        remaining_staged = len(staged_orders) - 5
                        if remaining_staged > 0:
                            st.info(f"📋 Staged order IDs: {staged_order_nums} ...and {remaining_staged} more")
                        else:
                            st.info(f"📋 Staged order IDs: {staged_order_nums}")
                
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
                    enable_enterprise_modules=True,  # Try enabling for sidebar functionality
                    enable_sidebar=True,
                    enable_pivot=True,
                    enable_value_aggregation=True,
                    groupable=True,
                    filterable=True,
                    enable_download=False
                )
                
                grid_response = table_result['grid_response']
                
                # Display selection summary outside of expander
                if table_result['selected_count'] > 0:
                    st.info(f"✅ Selected {table_result['selected_count']} rows out of {len(display_df)} total rows")
                
                # Display additional information about selected rows
                if table_result['selected_count'] > 0:
                    selected_df = pd.DataFrame(grid_response['selected_rows'])
                    
                    # Quick metrics always visible
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📦 Selected Items", len(selected_df))
                    with col2:
                        if 'ordernumber' in selected_df.columns:
                            unique_orders = selected_df['ordernumber'].nunique()
                            st.metric("📋 Unique Orders", unique_orders)
                    with col3:
                        if 'Transaction Quantity' in selected_df.columns:
                            total_qty = pd.to_numeric(selected_df['Transaction Quantity'], errors='coerce').sum()
                            st.metric("📊 Total Quantity", f"{total_qty:,.0f}")
                    with col4:
                        if 'Fulfillment Center' in selected_df.columns:
                            unique_warehouses = selected_df['Fulfillment Center'].nunique()
                            st.metric("🏭 Warehouses", unique_warehouses)
                    
                    # Staging functionality always visible
                    st.markdown("**📋 Order Staging**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("🎯 Add Selected to Staging", key="add_to_staging"):
                            # Add selected orders to staging using boolean column approach
                            if len(selected_df) > 0:
                                # Find the best ID column to use for matching
                                id_col = None
                                for col in ['ordernumber', 'externalorderid', 'id']:
                                    if col in selected_df.columns and col in st.session_state.processed_orders.columns:
                                        id_col = col
                                        break
                                
                                if id_col:
                                    # Add staged column if it doesn't exist
                                    if 'staged' not in st.session_state.processed_orders.columns:
                                        st.session_state.processed_orders['staged'] = False
                                    
                                    # Convert ID columns to strings for consistent comparison
                                    selected_df[id_col] = selected_df[id_col].astype(str)
                                    st.session_state.processed_orders[id_col] = st.session_state.processed_orders[id_col].astype(str)
                                    
                                    staged_ids = selected_df[id_col].tolist()
                                    
                                    # Since selected_df comes from filtered display (non-staged only), 
                                    # all selected orders should be stageable
                                    mask = st.session_state.processed_orders[id_col].isin(staged_ids)
                                    
                                    # Double-check: make sure none are already staged
                                    already_staged_mask = mask & (st.session_state.processed_orders['staged'] == True)
                                    new_to_stage_mask = mask & (st.session_state.processed_orders['staged'] == False)
                                    
                                    already_staged_count = already_staged_mask.sum()
                                    new_to_stage_count = new_to_stage_mask.sum()
                                    
                                    if already_staged_count > 0:
                                        st.warning(f"⚠️ {already_staged_count} orders are already staged (skipped)")
                                    
                                    if new_to_stage_count > 0:
                                        # Mark new orders as staged
                                        st.session_state.processed_orders.loc[new_to_stage_mask, 'staged'] = True
                                        st.session_state.processed_orders.loc[new_to_stage_mask, 'staged_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                        
                                        # Add to staging history
                                        st.session_state.staging_history.append({
                                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            'action': 'staged',
                                            'count': new_to_stage_count,
                                            'orders': staged_ids
                                        })
                                        
                                        st.success(f"✅ Added {new_to_stage_count} new orders to staging!")
                                    else:
                                        st.info("ℹ️ All selected orders are already staged.")
                                    
                                    st.rerun()
                                else:
                                    st.error("❌ Cannot stage orders: No suitable ID column found for matching.")
                            else:
                                st.warning("⚠️ No orders selected for staging.")
                    
                    with col2:
                        if st.session_state.processed_orders is not None and 'staged' in st.session_state.processed_orders.columns:
                            staged_count = (st.session_state.processed_orders['staged'] == True).sum()
                        else:
                            staged_count = 0
                        st.metric("📋 Orders in Staging", staged_count)
                    
                    with col3:
                        if staged_count > 0:
                            # Download staged orders using boolean column
                            staged_orders_df = st.session_state.processed_orders[st.session_state.processed_orders['staged'] == True]
                            staged_csv = staged_orders_df.to_csv(index=False)
                            st.download_button(
                                label="💾 Download Staged Orders",
                                data=staged_csv,
                                file_name=f"staged_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="download_staged_from_orders_tab"
                            )
                    
                    # Detailed information in collapsible expander
                    with st.expander("📋 Detailed Selection Summary", expanded=False):
                        # Order ID Summary
                        if 'ordernumber' in selected_df.columns:
                            st.write("**📋 Order IDs in Selection:**")
                            unique_orders = selected_df['ordernumber'].unique()
                            if len(unique_orders) <= 10:
                                # Show all order IDs if 10 or fewer
                                order_list = ", ".join([str(order) for order in sorted(unique_orders)])
                                st.write(f"`{order_list}`")
                            else:
                                # Show first 10 and count of remaining
                                first_10 = sorted(unique_orders)[:10]
                                order_list = ", ".join([str(order) for order in first_10])
                                remaining = len(unique_orders) - 10
                                st.write(f"`{order_list}` ... and {remaining} more orders")
                        
                        # Bundle Information
                        bundle_cols = [col for col in selected_df.columns if 'bundle' in col.lower()]
                        if bundle_cols:
                            st.write("**📦 Bundle Information:**")
                            for col in bundle_cols:
                                if selected_df[col].notna().any():
                                    bundle_items = selected_df[selected_df[col].notna()][col].value_counts()
                                    if not bundle_items.empty:
                                        st.write(f"*{col}:*")
                                        for bundle, count in bundle_items.head(5).items():
                                            st.write(f"  - {bundle}: {count} items")
                                            
                        # Fulfillment Center breakdown
                        if 'Fulfillment Center' in selected_df.columns:
                            st.write("**🏭 Fulfillment Center Breakdown:**")
                            fc_counts = selected_df['Fulfillment Center'].value_counts()
                            for fc, count in fc_counts.items():
                                percentage = (count / len(selected_df)) * 100
                                st.write(f"  - **{fc}**: {count} items ({percentage:.1f}%)")
                        
                        # Issues/Problems summary
                        issue_cols = [col for col in selected_df.columns if any(keyword in col.lower() for keyword in ['issue', 'problem', 'error', 'warning'])]
                        if issue_cols:
                            for col in issue_cols:
                                if col in selected_df.columns:
                                    issues = selected_df[selected_df[col].notna() & (selected_df[col] != "")]
                                    if not issues.empty:
                                        st.warning(f"**⚠️ {col}**: {len(issues)} items have issues")
                                        # Show unique issues
                                        unique_issues = issues[col].value_counts().head(3)
                                        for issue, count in unique_issues.items():
                                            st.write(f"  - {issue}: {count} items")
                
                # Health & Insights section for Orders tab
                if st.session_state.get('processing_stats'):
                    with st.expander("🔔 Inventory Health & Alerts", expanded=False):
                        processing_stats = st.session_state.processing_stats
                        
                        # Critical inventory alerts
                        alert_col1, alert_col2, alert_col3 = st.columns(3)
                        
                        with alert_col1:
                            zero_balance = processing_stats.get('zero_balance_items', 0)
                            if zero_balance > 0:
                                st.error(f"🚨 {zero_balance} items are OUT OF STOCK")
                            else:
                                st.success("✅ No out-of-stock items")
                        
                        with alert_col2:
                            low_balance = processing_stats.get('low_balance_items', 0)
                            if low_balance > 0:
                                st.warning(f"⚠️ {low_balance} items have LOW STOCK (≤10)")
                            else:
                                st.success("✅ No low stock alerts")
                        
                        with alert_col3:
                            total_shortages = processing_stats.get('total_shortages', 0)
                            if total_shortages > 0:
                                st.error(f"❌ {total_shortages} shortage instances detected")
                            else:
                                st.success("✅ No shortages detected")
                
                    with st.expander("💡 Decision Making Insights", expanded=False):
                        insights = []
                        
                        # Warehouse efficiency insights
                        if st.session_state.get('warehouse_performance'):
                            warehouse_performance = st.session_state.warehouse_performance
                            # Find best and worst performing warehouses
                            best_warehouse = min(warehouse_performance.items(), key=lambda x: x[1].get('issue_rate', 100))
                            worst_warehouse = max(warehouse_performance.items(), key=lambda x: x[1].get('issue_rate', 0))
                            
                            if best_warehouse[1].get('issue_rate', 0) != worst_warehouse[1].get('issue_rate', 0):
                                insights.append(f"🏆 **Best Performing Warehouse**: {best_warehouse[0]} (Issue Rate: {best_warehouse[1].get('issue_rate', 0):.1f}%)")
                                insights.append(f"🔧 **Needs Attention**: {worst_warehouse[0]} (Issue Rate: {worst_warehouse[1].get('issue_rate', 0):.1f}%)")
                        
                        # Volume distribution insights
                        if processing_stats and 'fulfillment_center_distribution' in processing_stats:
                            fc_dist = processing_stats['fulfillment_center_distribution']
                            total_items = sum(fc_dist.values())
                            
                            for fc, count in fc_dist.items():
                                percentage = (count / total_items) * 100
                                if percentage > 70:
                                    insights.append(f"⚖️ **Load Imbalance**: {fc} is handling {percentage:.1f}% of orders")
                                elif percentage < 10 and len(fc_dist) > 1:
                                    insights.append(f"📉 **Underutilized**: {fc} is only handling {percentage:.1f}% of orders - potential capacity available")
                        
                        # Shortage insights
                        if processing_stats and 'shortages_by_fulfillment_center' in processing_stats:
                            shortage_fc = processing_stats['shortages_by_fulfillment_center']
                            for fc, shortage_count in shortage_fc.items():
                                insights.append(f"📦 **Inventory Alert**: {fc} has {shortage_count} shortage instances - review restocking priorities")
                        
                        # Display insights
                        if insights:
                            for insight in insights:
                                st.info(insight)
                        else:
                            st.success("✅ No critical insights at this time - operations appear to be running smoothly")

        with tab2:
            # Staging Management Tab
            st.header("📋 Order Staging Management")
            
            # Staging overview metrics
            if st.session_state.processed_orders is not None:
                if 'staged' not in st.session_state.processed_orders.columns:
                    st.session_state.processed_orders['staged'] = False
                staged_count = (st.session_state.processed_orders['staged'] == True).sum()
                remaining_count = (st.session_state.processed_orders['staged'] == False).sum()
                
                # Calculate unique orders for staged and remaining
                if 'ordernumber' in st.session_state.processed_orders.columns:
                    staged_unique = st.session_state.processed_orders[st.session_state.processed_orders['staged'] == True]['ordernumber'].nunique()
                    remaining_unique = st.session_state.processed_orders[st.session_state.processed_orders['staged'] == False]['ordernumber'].nunique()
                else:
                    staged_unique = 0
                    remaining_unique = 0
            else:
                staged_count = 0
                remaining_count = 0
                staged_unique = 0
                remaining_unique = 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📋 Staged Line Items", staged_count)
                st.metric("📋 Unique Orders", staged_unique)
            with col2:
                st.metric("📜 Remaining Line Items", remaining_count) 
                st.metric("📜 Unique Orders", remaining_unique)
            with col3:
                total_original = staged_count + remaining_count
                progress = (staged_count / total_original * 100) if total_original > 0 else 0
                st.metric("🎯 Staging Progress", f"{progress:.1f}%")
            with col4:
                history_count = len(st.session_state.staging_history)
                st.metric("📝 Actions Taken", history_count)
            
            # Staging controls
            if staged_count > 0:
                st.markdown("### 📋 Staged Orders")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download staged orders
                    staged_csv = st.session_state.staged_orders.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Staged Orders",
                        data=staged_csv,
                        file_name=f"staged_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_staged_main"
                    )
                
                with col2:
                    if st.button("🔄 Clear All Staging", key="clear_staging"):
                        # Move all staged orders back to main using boolean column
                        if st.session_state.processed_orders is not None and 'staged' in st.session_state.processed_orders.columns:
                            staged_mask = st.session_state.processed_orders['staged'] == True
                            staged_count = staged_mask.sum()
                            
                            if staged_count > 0:
                                # Mark all staged orders as not staged
                                st.session_state.processed_orders.loc[staged_mask, 'staged'] = False
                                # Remove staging timestamps
                                if 'staged_at' in st.session_state.processed_orders.columns:
                                    st.session_state.processed_orders.loc[staged_mask, 'staged_at'] = None
                                
                                # Add to history
                                st.session_state.staging_history.append({
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'action': 'cleared_all',
                                    'count': staged_count,
                                    'orders': []
                                })
                                
                                st.success(f"✅ Moved {staged_count} orders back to main queue!")
                                st.rerun()
                            else:
                                st.info("No staged orders to clear.")
                
                with col3:
                    if st.button("🗑️ Delete Staged Orders", key="delete_staging"):
                        # Delete staged orders using boolean column
                        if st.session_state.processed_orders is not None and 'staged' in st.session_state.processed_orders.columns:
                            staged_mask = st.session_state.processed_orders['staged'] == True
                            deleted_count = staged_mask.sum()
                            
                            if deleted_count > 0:
                                # Remove staged orders from the main dataframe
                                st.session_state.processed_orders = st.session_state.processed_orders[~staged_mask].copy()
                                
                                # Add to history
                                st.session_state.staging_history.append({
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'action': 'deleted',
                                    'count': deleted_count,
                                    'orders': []
                                })
                                
                                st.warning(f"🗑️ Deleted {deleted_count} staged orders!")
                                st.rerun()
                            else:
                                st.info("No staged orders to delete.")
                
                # Display staged orders table using boolean column
                st.markdown("### 🔍 Staged Orders Preview")
                if st.session_state.processed_orders is not None and 'staged' in st.session_state.processed_orders.columns:
                    staged_orders_df = st.session_state.processed_orders[st.session_state.processed_orders['staged'] == True].copy()
                    if not staged_orders_df.empty:
                        staging_table = create_aggrid_table(
                            staged_orders_df,
                            height=400,
                            key="staging_table",
                            theme="alpine",
                            selection_mode='multiple',
                            show_hints=False,
                            enable_enterprise_modules=False,
                            enable_sidebar=True,
                            enable_pivot=True,
                            enable_value_aggregation=True,
                            groupable=True,
                            filterable=True
                        )
                    else:
                        staging_table = {'selected_count': 0}
                        st.info("No staged orders to display")
                else:
                    staging_table = {'selected_count': 0}
                    st.info("No staged orders to display")
                
                # Option to remove selected items from staging
                if staging_table['selected_count'] > 0:
                    if st.button("↩️ Move Selected Back to Orders", key="move_back_selected"):
                        selected_staging = pd.DataFrame(staging_table['grid_response']['selected_rows'])
                        
                        # Find ID column for matching
                        id_col = None
                        for col in ['ordernumber', 'externalorderid', 'id']:
                            if col in selected_staging.columns:
                                id_col = col
                                break
                        
                        if id_col:
                            selected_ids = selected_staging[id_col].astype(str).tolist()
                            
                            # Mark selected orders as not staged (move back to main)
                            mask = st.session_state.processed_orders[id_col].astype(str).isin(selected_ids)
                            st.session_state.processed_orders.loc[mask, 'staged'] = False
                            # Remove staging timestamp
                            if 'staged_at' in st.session_state.processed_orders.columns:
                                st.session_state.processed_orders.loc[mask, 'staged_at'] = None
                            
                            st.success(f"✅ Moved {len(selected_staging)} orders back to main queue!")
                            st.rerun()
                        else:
                            st.error("❌ Cannot move orders back: No suitable ID column found.")
            
            else:
                st.info("📋 No orders in staging yet. Go to the Orders tab to select and stage orders.")
            
            # Staging history
            if st.session_state.staging_history:
                st.markdown("### 📝 Staging History")
                history_df = pd.DataFrame(st.session_state.staging_history)
                
                # Add row numbers for better display
                history_df = history_df.reset_index(drop=True)
                history_df.insert(0, '#', range(1, len(history_df) + 1))
                
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    height=300
                )
                
                if st.button("🗑️ Clear History", key="clear_history"):
                    st.session_state.staging_history = []
                    st.success("✅ Staging history cleared!")
                    st.rerun()

        with tab3:            
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
                            enable_enterprise_modules=False,  # Disable to prevent JSON serialization issues
                            enable_sidebar=True,
                            enable_pivot=True,
                            enable_value_aggregation=True,
                            groupable=True,
                            filterable=True
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
                        enable_enterprise_modules=False,  # Disable to prevent JSON serialization issues
                        enable_sidebar=True,
                        enable_pivot=True,
                        enable_value_aggregation=True,
                        groupable=True,
                        filterable=True
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
            
        with tab4:
            # Enhanced dashboard content with analytics
            render_summary_dashboard(
                st.session_state.processed_orders, 
                st.session_state.inventory_df,
                st.session_state.get('processing_stats', {}),
                st.session_state.get('warehouse_performance', {})
            )
        
        with tab5:
            # Rules and Configuration tab
            st.header("⚙️ Rules & Configuration")
            
            # Initialize Airtable handler
            airtable_handler = AirtableHandler()
            schema_manager = SchemaManager()
            
            # Create tabs for different configuration sections
            rules_tab1, rules_tab2, rules_tab3 = st.tabs([
                "🔄 SKU Mappings", 
                "📍 Zip Codes & Zones", 
                "🚚 Delivery Services",
            ])
            
            with rules_tab1:
                st.subheader("🔄 SKU Mappings Management")
                st.info("View, add, edit, or remove SKU mappings from Airtable.")
                
                # Fetch SKU mappings and fulfillment centers from Airtable
                try:
                    with st.spinner("Loading SKU mappings from Airtable..."):
                        sku_mappings = airtable_handler.get_sku_mappings()
                        fulfillment_centers = airtable_handler.get_fulfillment_centers()
                        
                        # Create a lookup for fulfillment center IDs to names
                        fc_lookup = {fc.get("airtable_id"): fc.get("name") for fc in fulfillment_centers}
                        
                        if sku_mappings:
                            # Convert to DataFrame for display
                            mappings_data = []
                            
                            for m in sku_mappings:
                                # Create a new entry with reordered fields
                                mapping_entry = {
                                    "airtable_id": m.get("airtable_id", ""),
                                    "order_sku": m.get("order_sku", ""),
                                    "picklist_sku": m.get("picklist_sku", "")
                                }
                            
                                # Handle fulfillment_center field with proper type checking
                                fc_value = m.get("fulfillment_center", [])
                                if isinstance(fc_value, list):
                                    # Convert IDs to names using the lookup dictionary
                                    fc_names = [fc_lookup.get(fc_id, fc_id) for fc_id in fc_value]
                                    mapping_entry["fulfillment_center"] = ", ".join(fc_names)
                                else:
                                    mapping_entry["fulfillment_center"] = str(fc_value) if fc_value else ""
                                    
                                # Add the remaining fields after fulfillment_center
                                mapping_entry.update({
                                    "actual_qty": m.get("actual_qty", 0),
                                    "total_pick_weight": m.get("total_pick_weight", 0),
                                    "pick_type": m.get("pick_type", ""),
                                    "bundle_components": m.get("bundle_components", ""),
                                    "created_at": m.get("created_at", ""),
                                    "updated_at": m.get("updated_at", "")
                                })
                                
                                mappings_data.append(mapping_entry)
                            
                            mappings_df = pd.DataFrame(mappings_data)
                            
                            # Hide the airtable_id column from display
                            display_cols = [col for col in mappings_df.columns if col != 'airtable_id']
                            filtered_df = mappings_df[display_cols]
                            
                            # Display SKU mappings
                            st.write(f"Found {len(filtered_df)} SKU mappings")
                            
                            # Display the filtered dataframe
                            if not filtered_df.empty:
                                create_aggrid_table(
                                    filtered_df,
                                    height=400,
                                    key="sku_mappings_grid",
                                    theme="alpine",
                                    selection_mode='multiple',
                                    show_hints=False,
                                    enable_enterprise_modules=False,
                                    enable_sidebar=True,
                                    enable_pivot=True,
                                    enable_value_aggregation=True,
                                    groupable=True,
                                    filterable=True
                                )
                            else:
                                st.warning("No SKU mappings found matching your search criteria.")
                        else:
                            st.warning("No SKU mappings found in Airtable.")
                            
                    # Add new SKU mapping
                    with st.expander("➕ Add New SKU Mapping", expanded=False):
                        st.info("This feature will be available in the next update.")
                        
                except Exception as e:
                    st.error(f"Error loading SKU mappings: {str(e)}")
            
            with rules_tab2:
                st.subheader("📍 Zip Codes & Zones Management")
                st.info("View, add, edit, or remove zip code to zone mappings from Airtable.")
                
                # Fetch fulfillment zones from Airtable
                try:
                    with st.spinner("Loading fulfillment zones from Airtable..."):
                        all_zones = airtable_handler.get_fulfillment_zones()
                        fulfillment_centers = airtable_handler.get_fulfillment_centers()
                        
                        # Create a lookup for fulfillment center IDs to names
                        fc_lookup = {fc.get("airtable_id"): fc.get("name") for fc in fulfillment_centers}
                        
                        if all_zones:
                            # Convert to DataFrame for display
                            zones_df = pd.DataFrame([
                                {
                                    "airtable_id": z.get("airtable_id", ""),  # Keep for reference but hide from display
                                    "zip_prefix": z.get("zip_prefix", ""),
                                    "zone": z.get("zone", ""),
                                    "fulfillment_center": ", ".join([fc_lookup.get(fc_id, fc_id) for fc_id in z.get("FulfillmentCenter", [])]) 
                                        if isinstance(z.get("FulfillmentCenter"), list) else ""
                                } for z in all_zones
                            ])
                            
                            # Hide the IDs columns from display but keep them for reference
                            zones_df = zones_df[['zip_prefix', 'zone', 'fulfillment_center']]
                            
                            # Display zones
                            st.write(f"Found {len(zones_df)} zip code to zone mappings")
                            
                            # Use the full dataframe for display
                            filtered_df = zones_df
                            
                            # Display the filtered dataframe
                            if not filtered_df.empty:
                                create_aggrid_table(
                                    filtered_df,
                                    height=400,
                                    key="zones_grid",
                                    theme="alpine",
                                    selection_mode='multiple',
                                    show_hints=False,
                                    enable_enterprise_modules=False,
                                    enable_sidebar=True,
                                    enable_pivot=True,
                                    enable_value_aggregation=True,
                                    groupable=True,
                                    filterable=True
                                )
                            else:
                                st.warning("No zone mappings found matching your search criteria.")
                        else:
                            st.warning("No zone mappings found in Airtable.")
                    
                                        
                except Exception as e:
                    st.error(f"Error loading fulfillment zones: {str(e)}")
                    
                # Add bulk upload option
                with st.expander("📁 Bulk Upload Zones", expanded=False):
                    st.info("Upload a CSV file with zip code to zone mappings.")
                    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="zone_csv_upload")
                    
                    if uploaded_file is not None:
                        try:
                            # Read CSV
                            df = pd.read_csv(uploaded_file)
                            st.write("Preview of uploaded data:")
                            st.dataframe(df.head())
                            
                            # Check for required columns
                            required_cols = ["zip_prefix", "zone", "fulfillment_center"]
                            if all(col in df.columns for col in required_cols):
                                if st.button("Process Bulk Upload"):
                                    st.info("Bulk upload functionality will be implemented in the next update.")
                            else:
                                st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
                        except Exception as e:
                            st.error(f"Error processing CSV: {str(e)}")
                            
                # Add download option
                if 'zones_df' in locals() and not zones_df.empty:
                    zones_csv = zones_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Zone Mappings",
                        data=zones_csv,
                        file_name=f"zone_mappings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with rules_tab3:
                st.subheader("🚚 Delivery Services Management")
                st.info("View, add, edit, or remove delivery services from Airtable.")
                
                # Fetch delivery services from Airtable
                try:
                    with st.spinner("Loading delivery services from Airtable..."):
                        all_services = airtable_handler.get_delivery_services()
                        
                        if all_services:
                            # Convert to DataFrame for display
                            services_df = pd.DataFrame([
                                {
                                    "airtable_id": ds.get("airtable_id", ""),  # Keep for reference but don't display
                                    "id": ds.get("id", ""),
                                    "destination_zip_short": ds.get("destination_zip_short", ""),
                                    "origin": ds.get("origin", ""),
                                    "carrier_name": ds.get("carrier_name", ""),
                                    "service_name": ds.get("service_name", ""),
                                    "days": ds.get("days", 0),
                                    "created_at": ds.get("created_at", ""),
                                    "updated_at": ds.get("updated_at", "")
                                } for ds in all_services
                            ])
                            
                            # Hide the airtable_id column from display
                            display_cols = [col for col in services_df.columns if col != 'airtable_id']
                            services_df = services_df[display_cols]
                            
                            # Display delivery services
                            st.write(f"Found {len(services_df)} delivery services")
                            
                            # Use the full dataframe for display
                            filtered_df = services_df
                            
                            # Display the filtered dataframe
                            if not filtered_df.empty:
                                create_aggrid_table(
                                    filtered_df,
                                    height=400,
                                    key="services_grid",
                                    theme="alpine",
                                    selection_mode='multiple',
                                    show_hints=False,
                                    enable_enterprise_modules=False,
                                    enable_sidebar=True,
                                    enable_pivot=True,
                                    enable_value_aggregation=True,
                                    groupable=True,
                                    filterable=True
                                )
                            else:
                                st.warning("No delivery services found matching your search criteria.")
                        else:
                            st.warning("No delivery services found in Airtable.")
                    
                    # Add new delivery service or edit existing one
                    if "editing_service" in st.session_state:
                        # Edit existing delivery service
                        with st.expander("✏️ Edit Delivery Service", expanded=True):
                            service_data = st.session_state.editing_service
                            
                            # Form for editing
                            with st.form("edit_service_form"):
                                edited_name = st.text_input("Service Name", value=service_data.get("name", ""))
                                edited_code = st.text_input("Service Code", value=service_data.get("service_code", ""))
                                edited_desc = st.text_area("Description", value=service_data.get("description", ""))
                                edited_priority = st.number_input("Priority", min_value=0, value=int(service_data.get("priority", 0)))
                                edited_active = st.checkbox("Active", value=service_data.get("active") == "Yes")
                                
                                submit_edit = st.form_submit_button("Save Changes")
                                cancel_edit = st.form_submit_button("Cancel")
                                
                                if submit_edit:
                                    try:
                                        # Update in Airtable
                                        airtable_handler.update_delivery_service(
                                            service_data["airtable_id"],
                                            {
                                                "name": edited_name,
                                                "service_code": edited_code,
                                                "description": edited_desc,
                                                "priority": edited_priority,
                                                "active": edited_active
                                            }
                                        )
                                        
                                        st.success("Delivery service updated successfully!")
                                        # Clear editing state and cache
                                        del st.session_state.editing_service
                                        schema_manager.clear_cache()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error updating delivery service: {str(e)}")
                                
                                if cancel_edit:
                                    del st.session_state.editing_service
                                    st.rerun()
                    else:
                        # Add new delivery service
                        with st.expander("➕ Add New Delivery Service", expanded=False):
                            # Form for adding new service
                            with st.form("add_service_form"):
                                new_name = st.text_input("Service Name")
                                new_code = st.text_input("Service Code")
                                new_desc = st.text_area("Description")
                                new_priority = st.number_input("Priority", min_value=0, value=0)
                                new_active = st.checkbox("Active", value=True)
                                
                                submit_new = st.form_submit_button("Add Delivery Service")
                                
                                if submit_new:
                                    if new_name and new_code:
                                        try:
                                            # Create in Airtable
                                            airtable_handler.create_delivery_service({
                                                "name": new_name,
                                                "service_code": new_code,
                                                "description": new_desc,
                                                "priority": new_priority,
                                                "active": new_active
                                            })
                                            
                                            st.success("New delivery service created successfully!")
                                            # Clear cache
                                            schema_manager.clear_cache()
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error creating delivery service: {str(e)}")
                                    else:
                                        st.warning("Please provide at least a name and service code.")
                                        
                except Exception as e:
                    st.error(f"Error loading delivery services: {str(e)}")
                    
                # Add bulk upload option
                with st.expander("📁 Bulk Upload Services", expanded=False):
                    st.info("Upload a CSV file with delivery services.")
                    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="service_csv_upload")
                    
                    if uploaded_file is not None:
                        try:
                            # Read CSV
                            df = pd.read_csv(uploaded_file)
                            st.write("Preview of uploaded data:")
                            st.dataframe(df.head())
                            
                            # Check for required columns
                            required_cols = ["name", "service_code"]
                            if all(col in df.columns for col in required_cols):
                                if st.button("Process Bulk Upload", key="process_service_upload"):
                                    st.info("Bulk upload functionality will be implemented in the next update.")
                            else:
                                st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
                        except Exception as e:
                            st.error(f"Error processing CSV: {str(e)}")
            # Add refresh button for all Airtable data
            if st.button("🔄 Refresh All Airtable Data", key="refresh_airtable"):
                try:
                    schema_manager.clear_cache()
                    st.success("✅ Airtable data cache cleared. Data will be refreshed on next access.")
                except Exception as e:
                    st.error(f"Error refreshing Airtable data: {str(e)}")

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
