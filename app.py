import os

import streamlit as st
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    download_loader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms import OpenAI
import openai

st.set_page_config(
    layout="centered", page_icon=":knot:", page_title="Chat with your data"
)

openai.api_key = st.secrets["OPENAI_API_KEY"]
openai_api_key_user = st.sidebar.text_input("OpenAI API Key")
if openai_api_key_user:
    openai.api_key = openai_api_key_user


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(
        text="Loading and indexing the documents – hang tight! This should take 1-2 minutes."
    ):
        SimpleWebPageReader = download_loader("SimpleWebPageReader")

        loader = SimpleWebPageReader()
        documents = loader.load_data(urls=['https://google.com'])
        index = VectorStoreIndex.from_documents(documents)
    return index


# check if storage already exists
if not os.path.exists("./storage"):
    index = load_data()
    # store it for later
    index.storage_context.persist()
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)

st.header("📚 Chat with this bot. 🦙")

st.divider()

# If no chat history, create one
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about HITAS evaluations! Example: What would it cost to build a 6 story building in Kalasatama?",
        }
    ]

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
