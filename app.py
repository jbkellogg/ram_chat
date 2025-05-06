import os
import streamlit as st
import traceback
from datetime import datetime, timedelta
from langchain_community.chat_message_histories import ChatMessageHistory

# Import configuration, agent factory, and the new update function
import config
from agent_factory import create_conversational_agent

# Import the specific update function and embedding getter
from vector_store import update_email_vector_store_manual, get_embedding_model

# --- Check API Key ---
if not config.GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it.")
    st.stop()

# --- Caching Agent Creation ---
@st.cache_resource # Caches the entire agent setup process
def get_cached_agent():
    """Calls the agent factory and caches the result."""
    print("--- Initializing Agent Chain (will be cached) ---")
    try:
        # Initialize session state for context here if needed by agent creation
        if 'last_used_context' not in st.session_state:
            st.session_state['last_used_context'] = None
        agent = create_conversational_agent()
        print("--- Agent Chain Initialized ---")
        return agent
    except Exception as e:
        st.error(f"Failed to initialize the agent chain: {e}")
        print("--- Error during Agent Chain Initialization ---")
        traceback.print_exc()
        st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="RamChat", layout="wide") # Optional: Set page config

# --- Sidebar for Controls ---
with st.sidebar:
    st.image("public/logo-dark.png", width=150) # Adjust width as needed
    st.header("Email Index")

    # Button to trigger email update
    if st.button("Update Email Index"):
        st.info("Checking for new emails and updating the index...")
        with st.spinner("Processing... This may take a minute."):
            try:
                # Need the embedding model to load the store for update
                embedding_model = get_embedding_model()
                # Call the update function
                success, message = update_email_vector_store_manual(
                    persist_dir=config.EMAIL_DB_DIRECTORY,
                    embedding_model=embedding_model,
                    days_to_check=config.EMAIL_UPDATE_WINDOW_DAYS # Use config value
                )
                if success:
                    st.success(message)
                    # IMPORTANT: Clear the agent cache so it reloads with the updated store
                    get_cached_agent.clear()
                    st.info("Agent cache cleared. The next query will use the updated index.")
                else:
                    st.error(message)
            except Exception as btn_err:
                st.error(f"An unexpected error occurred during update: {btn_err}")
                traceback.print_exc()

    st.divider()
    # Add other sidebar elements if needed (e.g., clear history button)
    if st.button("Clear Chat History"):
        if "lc_history" in st.session_state:
            st.session_state.lc_history.clear()
            st.rerun() # Rerun to clear the displayed chat

# --- Main Chat Interface ---
st.title("Ram Chat")
st.caption("Ask me about school info, menus, athletics, and more!")

# Initialize chat history in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lc_history" not in st.session_state:
    st.session_state.lc_history = ChatMessageHistory()

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
prompt = st.chat_input("What's up?")
if prompt:
    # Add user message to Streamlit display state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the cached agent
    try:
        conversational_agent_chain = get_cached_agent()

        # Display thinking indicator and invoke agent
        with st.chat_message("assistant"):
            with st.spinner("Scratching my horns..."):
                # Invoke the agent using the Langchain history mechanism
                response = conversational_agent_chain.invoke(
                    {"input": prompt},
                    # Pass session_id config for RunnableWithMessageHistory
                    config={"configurable": {"session_id": "streamlit_session"}}
                )
                # Extract the actual response content
                ai_response = response.get("output", "Sorry, I encountered an issue.")
                st.markdown(ai_response)

        # Add AI response to Streamlit display state
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

        # --- Store context for the 'get_last_context' tool ---
        # The ReAct agent doesn't easily expose intermediate retriever results.
        # If you need context storage, you might need to modify the email_tool
        # or handbook_tool to store their retrieved docs in session state *before* returning.
        # For now, we'll assume context storage might not work perfectly with ReAct this way.
        # st.session_state['last_used_context'] = response.get("context") # This likely won't work directly with AgentExecutor

    except Exception as e:
        st.error(f"An error occurred while processing your request: {e}")
        print("--- Error during Agent Invocation ---")
        traceback.print_exc()