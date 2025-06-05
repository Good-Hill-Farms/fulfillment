import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from constants.models import MODEL_DISPLAY_NAMES, MODEL_GROUPS
from utils.data_processor import DataProcessor
from utils.llm_handler import LLMHandler

# Load environment variables
load_dotenv()

# Initialize session state and processors
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
if "sku_mappings" not in st.session_state:
    st.session_state.sku_mappings = None

# Initialize processors
data_processor = DataProcessor()
llm_handler = LLMHandler()


def get_model_response(messages, model):
    """Get response using LLMHandler"""
    try:
        # Extract the prompt from the last user message
        prompt = next(msg["content"] for msg in reversed(messages) if msg["role"] == "user")

        # Get context from session state
        context = {}
        
        # Debug: Check what's actually in session state
        debug_info = {}
        for key in ['inventory_df', 'orders_df', 'inventory_summary', 'processed_orders', 'shortage_summary', 'grouped_shortage_summary']:
            if key in st.session_state:
                val = st.session_state[key]
                if hasattr(val, 'empty'):  # DataFrame
                    debug_info[key] = f"DataFrame with {len(val)} rows" if not val.empty else "Empty DataFrame"
                elif val is not None:
                    debug_info[key] = f"Data available ({type(val).__name__})"
                else:
                    debug_info[key] = "None"
            else:
                debug_info[key] = "Missing from session state"
        
        # Add debug info to context
        context["debug_session_state"] = debug_info
        
        # Add ALL data - the LLM needs complete access for proper analysis
        if "inventory_df" in st.session_state and st.session_state.inventory_df is not None:
            df = st.session_state.inventory_df
            context["inventory"] = df.to_dict('records')
            context["inventory_total_count"] = len(df)
        
        if "orders_df" in st.session_state and st.session_state.orders_df is not None:
            df = st.session_state.orders_df
            context["orders"] = df.to_dict('records')
            context["orders_total_count"] = len(df)
            
        if "inventory_summary" in st.session_state and not st.session_state.inventory_summary.empty:
            df = st.session_state.inventory_summary
            context["inventory_summary"] = df.to_dict('records')
            context["inventory_summary_total_count"] = len(df)
            
        if "processed_orders" in st.session_state and st.session_state.processed_orders is not None:
            df = st.session_state.processed_orders
            context["processed_orders"] = df.to_dict('records')
            context["processed_orders_total_count"] = len(df)
            
        if "shortage_summary" in st.session_state and not st.session_state.shortage_summary.empty:
            df = st.session_state.shortage_summary
            context["shortage_summary"] = df.to_dict('records')
            context["shortage_count"] = len(df)
            
        if "grouped_shortage_summary" in st.session_state and not st.session_state.grouped_shortage_summary.empty:
            df = st.session_state.grouped_shortage_summary
            context["grouped_shortage_summary"] = df.to_dict('records')
            context["grouped_shortage_total_count"] = len(df)
            
        # STAGED ORDERS - Critical data!
        if "staged_orders" in st.session_state and not st.session_state.staged_orders.empty:
            df = st.session_state.staged_orders
            context["staged_orders"] = df.to_dict('records')
            context["staged_orders_count"] = len(df)
        else:
            context["staged_orders"] = []
            context["staged_orders_count"] = 0
            
        # STAGING HISTORY
        if "staging_history" in st.session_state:
            context["staging_history"] = st.session_state.staging_history
        else:
            context["staging_history"] = []
            
        # PROCESSING STATS
        if "processing_stats" in st.session_state:
            context["processing_stats"] = st.session_state.processing_stats
        else:
            context["processing_stats"] = {}
            
        # WAREHOUSE PERFORMANCE
        if "warehouse_performance" in st.session_state:
            context["warehouse_performance"] = st.session_state.warehouse_performance
        else:
            context["warehouse_performance"] = {}
            
        # INVENTORY COMPARISON
        if "inventory_comparison" in st.session_state and not st.session_state.inventory_comparison.empty:
            df = st.session_state.inventory_comparison
            context["inventory_comparison"] = df.to_dict('records')
            context["inventory_comparison_count"] = len(df)
        else:
            context["inventory_comparison"] = []
            context["inventory_comparison_count"] = 0
            
        # Reference data
        if "shipping_zones_df" in st.session_state and st.session_state.shipping_zones_df is not None:
            context["shipping_zones"] = st.session_state.shipping_zones_df.to_dict('records')
            
        if "sku_mappings" in st.session_state and st.session_state.sku_mappings is not None:
            context["sku_mappings"] = st.session_state.sku_mappings
            
        context["rules"] = st.session_state.get("rules", [])
        context["bundles"] = st.session_state.get("bundles", {})
        context["override_log"] = st.session_state.get("override_log", [])
        
        # Add Airtable data
        try:
            from utils.airtable_handler import AirtableHandler
            airtable_handler = AirtableHandler()
            
            # Get ALL Airtable data - complete access needed
            try:
                sku_mappings = airtable_handler.get_sku_mappings()
                if sku_mappings:
                    context["airtable_sku_mappings"] = sku_mappings  # ALL records
                    context["airtable_sku_count"] = len(sku_mappings)
            except Exception as e:
                context["airtable_sku_mappings"] = []
                context["airtable_error_sku"] = str(e)
            
            # Get fulfillment zones from Airtable
            try:
                fulfillment_zones = airtable_handler.get_fulfillment_zones()
                if fulfillment_zones:
                    context["airtable_fulfillment_zones"] = fulfillment_zones  # ALL records
                    context["airtable_zones_count"] = len(fulfillment_zones)
            except Exception as e:
                context["airtable_fulfillment_zones"] = []
                context["airtable_error_zones"] = str(e)
            
            # Get delivery services from Airtable
            try:
                delivery_services = airtable_handler.get_delivery_services()
                if delivery_services:
                    context["airtable_delivery_services"] = delivery_services  # ALL records
                    context["airtable_services_count"] = len(delivery_services)
            except Exception as e:
                context["airtable_delivery_services"] = []
                context["airtable_error_services"] = str(e)
                
            # Get fulfillment centers from Airtable
            try:
                fulfillment_centers = airtable_handler.get_fulfillment_centers()
                if fulfillment_centers:
                    context["airtable_fulfillment_centers"] = fulfillment_centers  # ALL records
                    context["airtable_centers_count"] = len(fulfillment_centers)
            except Exception as e:
                context["airtable_fulfillment_centers"] = []
                context["airtable_error_centers"] = str(e)
                
        except Exception as e:
            context["airtable_error_general"] = f"Could not load Airtable handler: {str(e)}"

        # Update LLM handler model
        llm_handler.model_name = model

        # Get response
        return llm_handler.get_response(prompt, context, messages)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_data_summary():
    """Get summary of available data for the AI assistant"""
    summary = []

    if "inventory_summary" in st.session_state and not st.session_state.inventory_summary.empty:
        inv = st.session_state.inventory_summary
        summary.append(f"Inventory: {len(inv)} items")
        low_stock = inv[inv["Balance"] < 10] if "Balance" in inv.columns else pd.DataFrame()
        if not low_stock.empty:
            summary.append(f"Low stock items: {len(low_stock)}")

    if "processed_orders" in st.session_state and st.session_state.processed_orders is not None:
        orders = st.session_state.processed_orders
        summary.append(f"Orders: {len(orders)} total")

    if "shortage_summary" in st.session_state and not st.session_state.shortage_summary.empty:
        shortages = st.session_state.shortage_summary
        summary.append(f"Shortages: {len(shortages)} items need attention")

    return (
        "\n".join(summary)
        if summary
        else "No data loaded yet. Please upload inventory and orders files."
    )


def main():
    st.set_page_config(page_title="Inventory Chat", page_icon="📊", layout="wide")
    
    # Force session state reinitialization if data is missing
    # This ensures data persists between page navigation
    for key in ['orders_df', 'inventory_df', 'shipping_zones_df', 'processed_orders', 
               'inventory_summary', 'shortage_summary', 'grouped_shortage_summary', 'sku_mappings']:
        if key not in st.session_state:
            st.session_state[key] = None if 'df' in key or key == 'processed_orders' or key == 'sku_mappings' else pd.DataFrame()

    # Sidebar for model selection
    with st.sidebar:
        st.title("🤖 Chat Assistant")

        # Model selection
        st.header("🤖 Select Model")
        provider = st.selectbox(
            "Provider",
            options=list(MODEL_GROUPS.keys()),
            index=list(MODEL_GROUPS.keys()).index("Google"),
        )
        model_id = st.selectbox(
            "Model",
            options=MODEL_GROUPS[provider],
            format_func=lambda x: MODEL_DISPLAY_NAMES[x],
            index=MODEL_GROUPS[provider].index("google/gemini-2.0-flash-lite-001")
            if provider == "Google"
            else 0,
        )

        # Show data status
        st.header("📊 Data Status")
        if "inventory_summary" in st.session_state and not st.session_state.inventory_summary.empty:
            st.success("✅ Inventory data loaded")
        else:
            st.warning("⚠️ No inventory data")

        if "processed_orders" in st.session_state and st.session_state.processed_orders is not None:
            st.success("✅ Orders processed")
        else:
            st.warning("⚠️ No orders data")

        if "shortage_summary" in st.session_state and not st.session_state.shortage_summary.empty:
            st.error(f"⚠️ {len(st.session_state.shortage_summary)} items with shortages")

        # Add help text
        if (st.session_state.inventory_df is None or st.session_state.orders_df is None):
            st.warning("⚠️ No data found! Please upload files in the main app first.")
            if st.button("🏠 Go to Main App"):
                st.switch_page("app.py")
        else:
            st.info("💡 All data loaded successfully!")

    # Main chat interface
    st.title("📊 Inventory Assistant")
    
    # DEBUG: Show session state
    with st.expander("🔧 Debug Session State", expanded=False):
        debug_info = {}
        
        # Check all data sources
        data_keys = [
            'shortage_summary', 'inventory_summary', 'processed_orders', 'grouped_shortage_summary',
            'staged_orders', 'inventory_comparison', 'orders_df', 'inventory_df', 'shipping_zones_df'
        ]
        
        for key in data_keys:
            if key in st.session_state:
                val = st.session_state[key]
                if hasattr(val, 'empty'):
                    debug_info[key] = f"DataFrame: {len(val)} rows" if not val.empty else "Empty DataFrame"
                    if not val.empty:
                        debug_info[f"{key}_columns"] = list(val.columns)
                elif isinstance(val, list):
                    debug_info[key] = f"List: {len(val)} items"
                elif isinstance(val, dict):
                    debug_info[key] = f"Dict: {len(val)} keys"
                else:
                    debug_info[key] = f"{type(val).__name__}: {val}"
            else:
                debug_info[key] = "Missing"
        
        # Check other important session state
        other_keys = ['staging_history', 'processing_stats', 'warehouse_performance', 'sku_mappings', 'rules', 'bundles']
        for key in other_keys:
            if key in st.session_state:
                val = st.session_state[key]
                if isinstance(val, list):
                    debug_info[key] = f"List: {len(val)} items"
                elif isinstance(val, dict):
                    debug_info[key] = f"Dict: {len(val)} keys"
                else:
                    debug_info[key] = f"{type(val).__name__}"
            else:
                debug_info[key] = "Missing"
                
        st.json(debug_info)
    
    # Show data availability status
    if (st.session_state.inventory_df is None or st.session_state.orders_df is None):
        st.error("❌ **No data available for chat assistant**")
        st.info("Please go to the main app and upload your inventory and orders files first.")
        st.stop()

    # Initialize chat
    if "messages" not in st.session_state:
        system_msg = """You are an AI assistant with direct access to live inventory and order data through the system.
        You can analyze current inventory levels, process orders, and help with fulfillment decisions.

        Your capabilities include:
        1. View and analyze real-time inventory data
        2. Process and optimize order fulfillment
        3. Track inventory shortages and suggest solutions
        4. Analyze shipping zones for optimal fulfillment
        5. Work with SKU mappings and bundle configurations
        6. Review fulfillment rules and override history
        """

        system_msg += "\n\nAvailable Data:\n"

        if "inventory_df" in st.session_state and st.session_state.inventory_df is not None:
            inv = st.session_state.inventory_df
            system_msg += f"\nRaw Inventory Data:\n"
            system_msg += f"- {len(inv)} inventory records\n"
            system_msg += f"- Columns: {', '.join(inv.columns)}\n"

        if "orders_df" in st.session_state and st.session_state.orders_df is not None:
            orders = st.session_state.orders_df
            system_msg += f"\nRaw Order Data:\n"
            system_msg += f"- {len(orders)} order records\n"
            system_msg += f"- Columns: {', '.join(orders.columns)}\n"

        if (
            "shipping_zones_df" in st.session_state
            and st.session_state.shipping_zones_df is not None
        ):
            zones = st.session_state.shipping_zones_df
            system_msg += f"\nShipping Zones:\n"
            system_msg += f"- {len(zones)} ZIP code mappings\n"
            system_msg += f"- Columns: {', '.join(zones.columns)}\n"

        if "sku_mappings" in st.session_state and st.session_state.sku_mappings is not None:
            system_msg += f"\nSKU Mappings:\n"
            system_msg += f"- Cross-reference mappings between fulfillment centers\n"

        if "rules" in st.session_state and st.session_state.rules:
            system_msg += f"\nFulfillment Rules:\n"
            system_msg += f"- {len(st.session_state.rules)} active rules\n"

        if "bundles" in st.session_state and st.session_state.bundles:
            system_msg += f"\nProduct Bundles:\n"
            system_msg += f"- {len(st.session_state.bundles)} configured bundles\n"

        system_msg += f"\nCurrent Status:\n{get_data_summary()}"

        system_msg += "\n\nProvide data-driven responses using the actual numbers and details from the available data. You can directly reference specific inventory levels, orders, and other data points."

        st.session_state.messages = [{"role": "system", "content": system_msg}]

    # Display messages
    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't show system messages
            with st.chat_message(message["role"]):
                st.markdown(
                    f'<div class="{message["role"]}-message">{message["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # Chat input
    if prompt := st.chat_input("💬 Ask about inventory, orders, or shortages..."):
        # Update system message with current data status
        system_msg = (
            "You are an AI assistant helping with inventory management and order fulfillment. "
        )
        system_msg += "\nYou have access to the following data:\n"

        # Raw Data
        if "inventory_df" in st.session_state and st.session_state.inventory_df is not None:
            inv = st.session_state.inventory_df
            system_msg += f"\nRaw Inventory Data:\n"
            system_msg += f"- {len(inv)} inventory records\n"
            system_msg += f"- Columns: {', '.join(inv.columns)}\n"

        if "orders_df" in st.session_state and st.session_state.orders_df is not None:
            orders = st.session_state.orders_df
            system_msg += f"\nRaw Order Data:\n"
            system_msg += f"- {len(orders)} order records\n"
            system_msg += f"- Columns: {', '.join(orders.columns)}\n"

        # Processed Data
        if "inventory_summary" in st.session_state and not st.session_state.inventory_summary.empty:
            inv = st.session_state.inventory_summary
            system_msg += f"\nInventory Summary:\n"
            system_msg += f"- {len(inv)} SKUs\n"
            if "Balance" in inv.columns:
                low_stock = inv[inv["Balance"] < 10]
                system_msg += f"- {len(low_stock)} low stock items\n"
                system_msg += f"- Total balance: {inv['Balance'].sum()}\n"

        if "processed_orders" in st.session_state and st.session_state.processed_orders is not None:
            orders = st.session_state.processed_orders
            system_msg += f"\nProcessed Orders:\n"
            system_msg += f"- {len(orders)} orders\n"
            if "FulfillmentCenter" in orders.columns:
                by_center = orders["FulfillmentCenter"].value_counts()
                for center, count in by_center.items():
                    system_msg += f"- {center}: {count} orders\n"

        if "shortage_summary" in st.session_state and not st.session_state.shortage_summary.empty:
            shortages = st.session_state.shortage_summary
            system_msg += f"\nShortage Summary:\n"
            system_msg += f"- {len(shortages)} items with shortages\n"
            if "ShortageQty" in shortages.columns:
                system_msg += f"- Total shortage quantity: {shortages['ShortageQty'].sum()}\n"

        if (
            "grouped_shortage_summary" in st.session_state
            and not st.session_state.grouped_shortage_summary.empty
        ):
            grouped = st.session_state.grouped_shortage_summary
            system_msg += f"\nGrouped Shortages:\n"
            system_msg += f"- {len(grouped)} shortage groups\n"

        # Reference Data
        system_msg += "\nReference Data:\n"
        if (
            "shipping_zones_df" in st.session_state
            and st.session_state.shipping_zones_df is not None
        ):
            zones = st.session_state.shipping_zones_df
            system_msg += f"- Shipping zones: {len(zones)} ZIP code mappings\n"

        if "sku_mappings" in st.session_state and st.session_state.sku_mappings is not None:
            system_msg += f"- SKU mappings available\n"

        if "rules" in st.session_state and st.session_state.rules:
            system_msg += f"- {len(st.session_state.rules)} fulfillment rules\n"

        if "bundles" in st.session_state and st.session_state.bundles:
            system_msg += f"- {len(st.session_state.bundles)} product bundles\n"

        if "override_log" in st.session_state and st.session_state.override_log:
            system_msg += f"- {len(st.session_state.override_log)} fulfillment overrides\n"

        st.session_state.messages[0] = {"role": "system", "content": system_msg}

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # DEBUG: Show what data is being passed
                with st.expander("🔍 Debug: Data Being Passed to LLM", expanded=False):
                    debug_context = {}
                    total_context_size = 0
                    
                    # Check session state data
                    for key in ['shortage_summary', 'inventory_summary', 'processed_orders']:
                        if key in st.session_state:
                            val = st.session_state[key]
                            if hasattr(val, 'empty') and not val.empty:
                                records = val.to_dict('records')
                                size_bytes = len(str(records))
                                debug_context[key] = f"DataFrame: {len(val)} rows → {len(records)} records → {size_bytes:,} bytes"
                                total_context_size += size_bytes
                            elif val is not None:
                                debug_context[key] = f"Available ({type(val).__name__})"
                            else:
                                debug_context[key] = "None"
                        else:
                            debug_context[key] = "Missing"
                    
                    # Check if we're hitting limits
                    if total_context_size > 100000:  # 100KB threshold
                        debug_context["⚠️ Warning"] = f"Large context size: {total_context_size:,} bytes - may hit API limits"
                    
                    debug_context["Total Context Size"] = f"{total_context_size:,} bytes"
                    st.json(debug_context)
                
                response = get_model_response(st.session_state.messages, model_id)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Failed to get response from AI")

    # Add custom CSS
    st.markdown(
        """
    <style>
        /* Adjust sidebar width */
        [data-testid="stSidebar"] {
            min-width: 300px;
            max-width: 300px;
        }

        /* Center main content */
        .main .block-container {
            max-width: 800px;
            padding-top: 2rem;
            padding-right: 1rem;
            padding-left: 1rem;
            margin: 0 auto;
        }

        /* Style chat messages */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            padding: 0.5rem 0;
        }

        /* User message style */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"]:has(.user-message)) {
            justify-content: flex-end;
        }

        /* Message content style */
        [data-testid="stMarkdownContainer"] p {
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0;
            line-height: 1.4;
        }

        /* User message color */
        .user-message {
            background-color: #e3f2fd;
            display: inline-block;
            max-width: 80%;
            margin-left: auto;
        }

        /* Assistant message color */
        .assistant-message {
            background-color: #f5f5f5;
            display: inline-block;
            max-width: 80%;
            margin-right: auto;
        }

        /* Chat input style */
        .stChatInputContainer {
            padding-bottom: 2rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
