import asyncio
import os
import uuid

import requests
import streamlit as st
import websockets
from dotenv import load_dotenv

from shared import ServiceLogger

logger = ServiceLogger("ui-service")


load_dotenv()

# Configuration
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8000")

st.set_page_config(page_title="💬 AI Doradca Inwestycyjny", page_icon="💼")

# Custom CSS for chat interface
st.markdown(
    """
<style>
.chat-container {
    height: 500px;
    overflow-y: auto;
    padding: 20px;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    margin-bottom: 20px;
}

.user-message {
    background-color: #e3f2fd;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 10px 0;
    max-width: 80%;
    float: right;
    clear: both;
}

.assistant-message {
    background-color: #f5f5f5;
    padding: 10px 15px;
    border-radius: 15px;
    margin: 10px 0;
    max-width: 80%;
    float: left;
    clear: both;
}

.clearfix::after {
    content: "";
    clear: both;
    display: table;
}
</style>
""",
    unsafe_allow_html=True,
)


def process_documents(files):
    """Send documents to document processing service"""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", file))

        response = requests.post(
            f"{API_GATEWAY_URL}/documents/process", files=files_data
        )
        return response.json()
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return None


def get_chat_response(question):
    """Get response from chat service"""
    try:
        response = requests.post(f"{API_GATEWAY_URL}/chat", json={"question": question})
        return response.json()
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return None


async def real_time_chat(user_input):
    uri = f"wss://{API_GATEWAY_URL}/chat"
    async with websockets.connect(uri) as websocket:
        await websocket.send(user_input)

        response_text = ""
        response_container = st.empty()

        while True:
            chunk = await websocket.recv()
            if not chunk:
                break
            
            response_text += chunk
            response_container.markdown(response_text)


def main():
    st.title("💬 AI Investment Advisor")

    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Documents")
        uploaded_files = st.file_uploader(
            "Upload documents for analysis", accept_multiple_files=True, type=["pdf"]
        )

        if uploaded_files:
            if st.button("Process documents"):
                with st.spinner("Processing documents..."):
                    result = process_documents(uploaded_files)
                    if result:
                        st.success("Documents processed")
                    else:
                        st.error("Error processing documents")

    # --- Session State Initialization ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        # Generate a unique session ID for this user session if it doesn't exist
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"Generated new session_id: {st.session_state.session_id}")
    # ------------------------------------

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-message">{message["content"]}</div><div class="clearfix"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-message">{message["content"]}</div><div class="clearfix"></div>',
                unsafe_allow_html=True,
            )

    # Chat input
    if prompt := st.chat_input("Ask a question about investments..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Add user message to UI immediately
        st.markdown(
            f'<div class="user-message">{prompt}</div><div class="clearfix"></div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Thinking..."):
            try:
                message_placeholder = st.empty()
                response_text = ""

                async def stream_chat_response():
                    nonlocal response_text
                    ws_url = API_GATEWAY_URL.replace("http://", "ws://").replace(
                        "https://", "wss://"
                    )
                    # Use the session_id stored in Streamlit session state
                    current_session_id = st.session_state.session_id
                    connect_url = f"{ws_url}/chat?session_id={current_session_id}"  # <-- Fixed the =
                    logger.info(f"Connecting to: {connect_url}")
                    async with websockets.connect(connect_url) as websocket:
                        logger.info(
                            f"Sending prompt for session {current_session_id}: {prompt}"
                        )
                        await websocket.send(prompt)

                        try:
                            while True:
                                chunk = await websocket.recv()
                                # Handle potential bytes and decode
                                if isinstance(chunk, bytes):
                                    chunk_text = chunk.decode("utf-8")
                                else:
                                    chunk_text = str(chunk)

                                if chunk_text == "[END]":
                                    logger.info(
                                        f"Received [END] marker for session {current_session_id}"
                                    )
                                    break

                                logger.info(
                                    f"Received chunk for session {current_session_id}: {chunk_text}"
                                )
                                response_text += chunk_text
                        except websockets.exceptions.ConnectionClosed as e:
                            logger.warning(
                                f"WebSocket connection closed unexpectedly for session {current_session_id}: {e}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error during WebSocket communication for session {current_session_id}: {e}"
                            )

                # Run the async processing
                asyncio.run(stream_chat_response())

                # After streaming is complete, add the full response to session state
                if response_text:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )
                    # Display the final assistant message INSTEAD of the placeholder
                    st.markdown(
                        f'<div class="assistant-message">{response_text}</div><div class="clearfix"></div>',
                        unsafe_allow_html=True,
                    )
                    # Clear the placeholder if it was used
                    message_placeholder.empty()
                else:
                    st.error("Failed to get a response")
                    # Add an error message to the chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "Sorry, an error occurred and I can't answer right now.",
                        }
                    )
                    # Display the error in the chat area
                    st.markdown(
                        f'<div class="assistant-message">Sorry, an error occurred and I can\'t answer right now.</div><div class="clearfix"></div>',
                        unsafe_allow_html=True,
                    )
                    message_placeholder.empty()

            except Exception as e:
                st.error(f"Error during communication: {str(e)}")
                message_placeholder.empty()

        # Rerun to clear the input box and potentially update message display if needed
        st.rerun()


if __name__ == "__main__":
    main()
