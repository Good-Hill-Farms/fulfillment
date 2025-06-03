import streamlit as st
import json
import os
import pandas as pd
from dotenv import load_dotenv
from constants.models import MODEL_GROUPS, MODEL_DISPLAY_NAMES
from constants.shipping_zones import load_shipping_zones
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
        context = {
            # Raw data
            "inventory": st.session_state.inventory_df.to_dict() if "inventory_df" in st.session_state and st.session_state.inventory_df is not None else {},
            "orders": st.session_state.orders_df.to_dict() if "orders_df" in st.session_state and st.session_state.orders_df is not None else {},
            
            # Processed data
            "inventory_summary": st.session_state.inventory_summary.to_dict() if "inventory_summary" in st.session_state and not st.session_state.inventory_summary.empty else {},
            "processed_orders": st.session_state.processed_orders.to_dict() if "processed_orders" in st.session_state and st.session_state.processed_orders is not None else {},
            "shortage_summary": st.session_state.shortage_summary.to_dict() if "shortage_summary" in st.session_state and not st.session_state.shortage_summary.empty else {},
            "grouped_shortage_summary": st.session_state.grouped_shortage_summary.to_dict() if "grouped_shortage_summary" in st.session_state and not st.session_state.grouped_shortage_summary.empty else {},
            
            # Reference data
            "shipping_zones": st.session_state.shipping_zones_df.to_dict() if "shipping_zones_df" in st.session_state and st.session_state.shipping_zones_df is not None else {},
            "sku_mappings": st.session_state.sku_mappings if "sku_mappings" in st.session_state and st.session_state.sku_mappings is not None else {},
            "rules": st.session_state.rules if "rules" in st.session_state else [],
            "bundles": st.session_state.bundles if "bundles" in st.session_state else {},
            "override_log": st.session_state.override_log if "override_log" in st.session_state else []
        }
        
        # Update LLM handler model
        llm_handler.model_name = model
        
        # Get response
        return llm_handler.get_response(prompt, context, messages)
    except Exception as e:
        st.error(f'Error: {str(e)}')
        return None

def get_data_summary():
    """Get summary of available data for the AI assistant"""
    summary = []
    
    if 'inventory_summary' in st.session_state and not st.session_state.inventory_summary.empty:
        inv = st.session_state.inventory_summary
        summary.append(f"Inventory: {len(inv)} items")
        low_stock = inv[inv['Balance'] < 10] if 'Balance' in inv.columns else pd.DataFrame()
        if not low_stock.empty:
            summary.append(f"Low stock items: {len(low_stock)}")
    
    if 'processed_orders' in st.session_state and st.session_state.processed_orders is not None:
        orders = st.session_state.processed_orders
        summary.append(f"Orders: {len(orders)} total")
    
    if 'shortage_summary' in st.session_state and not st.session_state.shortage_summary.empty:
        shortages = st.session_state.shortage_summary
        summary.append(f"Shortages: {len(shortages)} items need attention")
    
    return "\n".join(summary) if summary else "No data loaded yet. Please upload inventory and orders files."
    
def main():
    st.set_page_config(
        page_title="Inventory Chat",
        page_icon="📊",
        layout="wide"
    )
    
    # Sidebar for model selection
    with st.sidebar:
        st.title("🤖 Chat Assistant")
        
        # Model selection
        st.header("🤖 Select Model")
        provider = st.selectbox(
            "Provider", 
            options=list(MODEL_GROUPS.keys()),
            index=list(MODEL_GROUPS.keys()).index("Google")
        )
        model_id = st.selectbox(
            "Model", 
            options=MODEL_GROUPS[provider],
            format_func=lambda x: MODEL_DISPLAY_NAMES[x],
            index=MODEL_GROUPS[provider].index("google/gemini-2.0-flash-lite-001") if provider == "Google" else 0
        )
        
        # Show data status
        st.header("📊 Data Status")
        if 'inventory_summary' in st.session_state and not st.session_state.inventory_summary.empty:
            st.success("✅ Inventory data loaded")
        else:
            st.warning("⚠️ No inventory data")
            
        if 'processed_orders' in st.session_state and st.session_state.processed_orders is not None:
            st.success("✅ Orders processed")
        else:
            st.warning("⚠️ No orders data")
            
        if 'shortage_summary' in st.session_state and not st.session_state.shortage_summary.empty:
            st.error(f"⚠️ {len(st.session_state.shortage_summary)} items with shortages")
            
        # Add help text
        st.info("💡 Upload data in the main app to use the chat assistant")
    
    # Main chat interface
    st.title("📊 Inventory Assistant")
    
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
        
        if 'inventory_df' in st.session_state and st.session_state.inventory_df is not None:
            inv = st.session_state.inventory_df
            system_msg += f"\nRaw Inventory Data:\n"
            system_msg += f"- {len(inv)} inventory records\n"
            system_msg += f"- Columns: {', '.join(inv.columns)}\n"
        
        if 'orders_df' in st.session_state and st.session_state.orders_df is not None:
            orders = st.session_state.orders_df
            system_msg += f"\nRaw Order Data:\n"
            system_msg += f"- {len(orders)} order records\n"
            system_msg += f"- Columns: {', '.join(orders.columns)}\n"
            
        if 'shipping_zones_df' in st.session_state and st.session_state.shipping_zones_df is not None:
            zones = st.session_state.shipping_zones_df
            system_msg += f"\nShipping Zones:\n"
            system_msg += f"- {len(zones)} ZIP code mappings\n"
            system_msg += f"- Columns: {', '.join(zones.columns)}\n"
            
        if 'sku_mappings' in st.session_state and st.session_state.sku_mappings is not None:
            system_msg += f"\nSKU Mappings:\n"
            system_msg += f"- Cross-reference mappings between fulfillment centers\n"
            
        if 'rules' in st.session_state and st.session_state.rules:
            system_msg += f"\nFulfillment Rules:\n"
            system_msg += f"- {len(st.session_state.rules)} active rules\n"
            
        if 'bundles' in st.session_state and st.session_state.bundles:
            system_msg += f"\nProduct Bundles:\n"
            system_msg += f"- {len(st.session_state.bundles)} configured bundles\n"
            
        system_msg += f"\nCurrent Status:\n{get_data_summary()}"
        
        system_msg += "\n\nProvide data-driven responses using the actual numbers and details from the available data. You can directly reference specific inventory levels, orders, and other data points."
        
        st.session_state.messages = [{
            "role": "system",
            "content": system_msg
        }]
    
    # Display messages
    for message in st.session_state.messages:
        if message["role"] != "system":  # Don't show system messages
            with st.chat_message(message["role"]):
                st.markdown(f'<div class="{message["role"]}-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("💬 Ask about inventory, orders, or shortages..."):
        # Update system message with current data status
        system_msg = "You are an AI assistant helping with inventory management and order fulfillment. "        
        system_msg += "\nYou have access to the following data:\n"
        
        # Raw Data
        if 'inventory_df' in st.session_state and st.session_state.inventory_df is not None:
            inv = st.session_state.inventory_df
            system_msg += f"\nRaw Inventory Data:\n"
            system_msg += f"- {len(inv)} inventory records\n"
            system_msg += f"- Columns: {', '.join(inv.columns)}\n"
        
        if 'orders_df' in st.session_state and st.session_state.orders_df is not None:
            orders = st.session_state.orders_df
            system_msg += f"\nRaw Order Data:\n"
            system_msg += f"- {len(orders)} order records\n"
            system_msg += f"- Columns: {', '.join(orders.columns)}\n"
        
        # Processed Data
        if 'inventory_summary' in st.session_state and not st.session_state.inventory_summary.empty:
            inv = st.session_state.inventory_summary
            system_msg += f"\nInventory Summary:\n"
            system_msg += f"- {len(inv)} SKUs\n"
            if 'Balance' in inv.columns:
                low_stock = inv[inv['Balance'] < 10]
                system_msg += f"- {len(low_stock)} low stock items\n"
                system_msg += f"- Total balance: {inv['Balance'].sum()}\n"
        
        if 'processed_orders' in st.session_state and st.session_state.processed_orders is not None:
            orders = st.session_state.processed_orders
            system_msg += f"\nProcessed Orders:\n"
            system_msg += f"- {len(orders)} orders\n"
            if 'FulfillmentCenter' in orders.columns:
                by_center = orders['FulfillmentCenter'].value_counts()
                for center, count in by_center.items():
                    system_msg += f"- {center}: {count} orders\n"
        
        if 'shortage_summary' in st.session_state and not st.session_state.shortage_summary.empty:
            shortages = st.session_state.shortage_summary
            system_msg += f"\nShortage Summary:\n"
            system_msg += f"- {len(shortages)} items with shortages\n"
            if 'ShortageQty' in shortages.columns:
                system_msg += f"- Total shortage quantity: {shortages['ShortageQty'].sum()}\n"
        
        if 'grouped_shortage_summary' in st.session_state and not st.session_state.grouped_shortage_summary.empty:
            grouped = st.session_state.grouped_shortage_summary
            system_msg += f"\nGrouped Shortages:\n"
            system_msg += f"- {len(grouped)} shortage groups\n"
        
        # Reference Data
        system_msg += "\nReference Data:\n"
        if 'shipping_zones_df' in st.session_state and st.session_state.shipping_zones_df is not None:
            zones = st.session_state.shipping_zones_df
            system_msg += f"- Shipping zones: {len(zones)} ZIP code mappings\n"
        
        if 'sku_mappings' in st.session_state and st.session_state.sku_mappings is not None:
            system_msg += f"- SKU mappings available\n"
        
        if 'rules' in st.session_state and st.session_state.rules:
            system_msg += f"- {len(st.session_state.rules)} fulfillment rules\n"
        
        if 'bundles' in st.session_state and st.session_state.bundles:
            system_msg += f"- {len(st.session_state.bundles)} product bundles\n"
        
        if 'override_log' in st.session_state and st.session_state.override_log:
            system_msg += f"- {len(st.session_state.override_log)} fulfillment overrides\n"
        
        st.session_state.messages[0] = {
            "role": "system",
            "content": system_msg
        }
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_model_response(st.session_state.messages, model_id)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Failed to get response from AI")
    
    # Add custom CSS
    st.markdown("""
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
    """, unsafe_allow_html=True)
    

    


if __name__ == "__main__":
    main()
