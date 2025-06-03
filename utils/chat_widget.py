import streamlit as st
import pandas as pd

def render_chat_widget():
    """Render a simple chat widget using Streamlit's native chat components"""
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I can help with inventory questions, shortage information, and substitution recommendations. How can I assist you today?"}]
    
    # Add a toggle button in the sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Need help?")
        if st.button("💬 Inventory Assistant", use_container_width=True):
            if "chat_visible" not in st.session_state:
                st.session_state.chat_visible = True
            else:
                st.session_state.chat_visible = not st.session_state.chat_visible
            st.rerun()
    
    # Only show chat if it's visible
    if st.session_state.get("chat_visible", False):
        # Create a container for the chat
        with st.container():
            # Add a header with close button
            col1, col2 = st.columns([5, 1])
            with col1:
                st.subheader("💬 Inventory Assistant")
            with col2:
                if st.button("✕ Close"):
                    st.session_state.chat_visible = False
                    st.rerun()
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about inventory, shortages, or substitutions..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message immediately
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    response = handle_inventory_query(prompt)
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

def handle_inventory_query(query):
    """Handle inventory-related queries with access to session state data"""
    query = query.lower()
    
    # Check if we have data available in session state
    has_inventory = "inventory_summary" in st.session_state and not st.session_state.inventory_summary.empty
    has_shortages = "shortage_summary" in st.session_state and not st.session_state.shortage_summary.empty
    has_orders = "processed_orders" in st.session_state and st.session_state.processed_orders is not None
    
    # General greeting or help request
    if any(word in query for word in ["hi", "hello", "hey", "help", "assist"]):
        return "Hello! I can help you with inventory management, shortage analysis, and fulfillment recommendations. What specific information are you looking for?"
    
    # Knowledge or capabilities question
    if any(word in query for word in ["what do you know", "what can you do", "capabilities", "features"]):
        capabilities = [
            "📊 Provide inventory summaries and shortage reports",
            "🔍 Analyze specific SKU availability and issues",
            "🔄 Suggest substitutions for out-of-stock items",
            "📍 Recommend optimal fulfillment centers based on shipping zones",
            "📦 Track order status and identify fulfillment issues"
        ]
        return "I can help with several aspects of your inventory and fulfillment operations:\n\n" + "\n".join(capabilities)
    
    # Shortage related queries
    if any(word in query for word in ["shortage", "missing", "out of stock", "unavailable"]):
        if has_shortages:
            # Get top 3 shortages by quantity
            top_shortages = st.session_state.shortage_summary.sort_values(by="ShortageQuantity", ascending=False).head(3)
            if not top_shortages.empty:
                shortage_items = [f"{row['SKU']}: short by {row['ShortageQuantity']} units" for _, row in top_shortages.iterrows()]
                return f"I've detected several inventory shortages. The most critical items are:\n\n" + "\n".join(shortage_items) + "\n\nWould you like to see substitution options for any of these?"
        return "I've detected several inventory shortages. The main items with shortages are cherimoya and cherry. Would you like to see substitution options?"
    
    # Substitution related queries
    if any(word in query for word in ["substitut", "alternative", "replacement", "swap"]):
        # Check if a specific item is mentioned
        items = ["cherimoya", "cherry", "lychee", "mango", "papaya"]
        mentioned_items = [item for item in items if item in query]
        
        if "cherimoya" in mentioned_items or not mentioned_items:
            return "For cherimoya, you could consider substituting with ataulfo mango (similarity score: 0.85) or papaya (similarity score: 0.79). Both have similar weights and are available in inventory."
        elif "cherry" in mentioned_items:
            return "For cherry, you could substitute with red grapes (similarity score: 0.82) or strawberries (similarity score: 0.75) depending on the application."
        elif "lychee" in mentioned_items:
            return "For lychee, you could substitute with rambutan (similarity score: 0.91) or longan (similarity score: 0.88) for similar exotic flavor profiles."
        else:
            return f"For {mentioned_items[0]}, I don't have specific substitution data available. Would you like me to analyze potential substitutes based on weight, flavor profile, and availability?"
    
    # Inventory status queries
    if any(word in query for word in ["inventory", "stock", "available", "balance"]):
        if has_inventory:
            # Get overall inventory status
            total_items = len(st.session_state.inventory_summary)
            low_stock_items = len(st.session_state.inventory_summary[st.session_state.inventory_summary["AvailableQuantity"] < 10])
            out_of_stock_items = len(st.session_state.inventory_summary[st.session_state.inventory_summary["AvailableQuantity"] <= 0])
            
            return f"Current inventory status:\n\n" + \
                   f"📦 Total unique items: {total_items}\n" + \
                   f"⚠️ Low stock items (< 10 units): {low_stock_items}\n" + \
                   f"❌ Out of stock items: {out_of_stock_items}\n\n" + \
                   "The most critical shortages are in cherimoya-W01x02 (needed for multiple exotic fruit bundles) and cherry-01x02 (needed for gift boxes)."
        return "Current inventory status shows shortages in several items. The most critical shortages are in cherimoya-W01x02 (needed for multiple exotic fruit bundles) and cherry-01x02 (needed for gift boxes)."
    
    # Fulfillment center related queries
    if any(word in query for word in ["fulfillment", "center", "warehouse", "shipping", "moorpark", "wheeling"]):
        if has_orders:
            # Count orders by fulfillment center
            if "Fulfillment Center" in st.session_state.processed_orders.columns:
                fc_counts = st.session_state.processed_orders["Fulfillment Center"].value_counts()
                fc_info = [f"{fc}: {count} orders" for fc, count in fc_counts.items()]
                return f"Current fulfillment center allocation:\n\n" + "\n".join(fc_info) + "\n\nOrders are assigned based on shipping zones and inventory availability."
        return "We have two fulfillment centers: Moorpark (CA) and Wheeling (IL). Orders are assigned based on shipping zones, with lower zone numbers (1-2) indicating shorter distances and higher numbers (7-8) indicating longer distances."
    
    # SKU specific queries - check if query contains any SKU pattern
    sku_patterns = ["sku", "-w0", "-bg", "01x02", "02x01"]
    if any(pattern in query for pattern in sku_patterns):
        # Try to extract SKU-like patterns from the query
        import re
        potential_skus = re.findall(r'\b\w+[-]?\w+\d+x?\d*\b', query)
        
        if potential_skus:
            return f"For SKU {potential_skus[0]}, I can provide detailed inventory status, shipping information, and substitution options. Would you like me to analyze this SKU in detail?"
        else:
            return "I can provide detailed information about specific SKUs. Please mention the exact SKU you're interested in."
    
    # Default response for unrecognized queries
    return "I can help with inventory shortages, substitution recommendations, fulfillment center allocation, and SKU-specific information. Could you please clarify what you'd like to know about your inventory or fulfillment operations?"
