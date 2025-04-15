import streamlit as st
import requests
import os

# Streamlit app configuration
st.set_page_config(page_title="Plugin AI Assist", page_icon="🚀")

# Add custom CSS
st.markdown("""
<style>
img[alt="Logo"] {
    height: 100px;
    width: auto;
}
</style>
""", unsafe_allow_html=True)

# Display logo
st.image("logo-plugin-color.png", width=100)
st.title("Plugin(Decentralized Oracle) - AI Assist")
st.write("Ask Anything about Plugin & it's Eco System")

# Backend URL from environment variable
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000/query")  # Fallback for non-Docker

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🚀"):
        st.markdown(message["content"])

# Input box for user query
if user_input := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    try:
        payload = {"query": user_input}
        if st.session_state.session_id:
            payload["session_id"] = st.session_state.session_id
        response = requests.post(BACKEND_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        bot_response = result.get("response", "No response received")
        st.session_state.session_id = result.get("session_id")
    except requests.RequestException as e:
        bot_response = f"Error: Could not connect to backend ({str(e)})"

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant", avatar="🚀"):
        st.markdown(bot_response)