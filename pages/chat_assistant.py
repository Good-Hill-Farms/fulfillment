import streamlit as st
import os
import requests
from typing import Optional

from constants.models import MODEL_DISPLAY_NAMES, MODEL_GROUPS

# Page config
st.set_page_config(
    page_title="Simple Chat Assistant",
    page_icon="💬",
    layout="centered"
)

# Initialize session state for messages
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Initialize model selection
if "chat_model" not in st.session_state:
    st.session_state.chat_model = "google/gemini-2.0-flash-lite-001"

def get_ai_response(messages: list, model: str) -> Optional[str]:
    """
    Get response from AI model using direct API call
    """
    try:
        if not messages:
            return "Hello! How can I help you today?"
        
        # Get API key
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            return "Error: OpenRouter API key not found. Please add it to your .env file."
        
        # Prepare messages for direct API call
        api_messages = [
            {
                "role": "system", 
                "content": "You are a helpful AI assistant. Provide friendly, helpful responses to user questions. You can help with general conversation, answering questions, creative writing, problem solving, and providing assistance on various topics."
            }
        ]
        
        # Add conversation history
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Make direct API call
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": model,
            "messages": api_messages,
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: API request failed with status {response.status_code}"
        
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        return "I'm sorry, I'm having trouble responding right now. Please try again."

def clear_chat():
    """Clear the chat history"""
    st.session_state.chat_messages = []

def main():
    # Header
    st.title("💬 AI Chat Assistant")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("🤖 AI Settings")
        
        # Model selection
        provider = st.selectbox(
            "Provider",
            options=list(MODEL_GROUPS.keys()),
            index=list(MODEL_GROUPS.keys()).index("Google"),
        )
        
        selected_model = st.selectbox(
            "Model",
            options=MODEL_GROUPS[provider],
            format_func=lambda x: MODEL_DISPLAY_NAMES[x],
            index=MODEL_GROUPS[provider].index("google/gemini-2.0-flash-lite-001")
            if provider == "Google" and "google/gemini-2.0-flash-lite-001" in MODEL_GROUPS[provider]
            else 0,
        )
        st.session_state.chat_model = selected_model
        
        # Show current model
        st.info(f"🤖 Using: {MODEL_DISPLAY_NAMES[selected_model]}")
        
        st.divider()
        
        # Chat controls
        st.header("🎮 Controls")
        
        if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
            clear_chat()
            st.rerun()
        
        # Chat stats
        message_count = len(st.session_state.chat_messages)
        if message_count > 0:
            st.info(f"💬 Messages: {message_count}")
        

    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                response = get_ai_response(st.session_state.chat_messages, st.session_state.chat_model)
                
                if response:
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Sorry, I couldn't generate a response. Please try again.")

    # Welcome message for first-time users
    if len(st.session_state.chat_messages) == 0:
        st.markdown("""
        ### 👋 Welcome!
        
        Start chatting by typing a message above.
        """)

    # Custom CSS for clean look
    st.markdown("""
    <style>
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        
        /* Main content styling */
        .main .block-container {
            max-width: 800px;
            padding: 1rem;
        }
        
        /* Chat message styling */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            padding: 0.5rem 0;
        }
        
        /* Welcome section styling */
        .stMarkdown h3 {
            color: #1f77b4;
        }
        
        /* Button styling */
        .stButton > button {
            width: 100%;
            border-radius: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
