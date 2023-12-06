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
st.title("🦙 Llama Chat 🦙")

openai.api_key = st.secrets["OPENAI_API_KEY"]
openai_api_key_user = st.sidebar.text_input("OpenAI API Key")
if openai_api_key_user:
    openai.api_key = openai_api_key_user


@st.cache_resource(show_spinner=False)
def load_data(url_list):
    with st.spinner(
        text="Loading and indexing the documents – hang tight! This should take 1-2 minutes."
    ):
        SimpleWebPageReader = download_loader("SimpleWebPageReader")

        loader = SimpleWebPageReader(html_to_text=True)
        documents = loader.load_data(urls=url_list)
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.5,
                system_prompt="You are an expert on the reading web pages and answering questions about the contents. Keep your answers short and based on facts – do not hallucinate features.",
            )
        )
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
    return index

url_list = st.text_input(label="input_urls", placeholder="url1,url2,...,urlN")
if url_list:
    url_list = url_list.split(",")

if not url_list:
    st.stop()

index = load_data(url_list=url_list)

# # check if storage already exists <- use this ifelse to avoid reindexing
# if not os.path.exists("./storage"):
#     index = load_data(url_list=url_list)
#     # store it for later
#     index.storage_context.persist()
# else:
#     # load the existing index
#     storage_context = StorageContext.from_defaults(persist_dir="./storage")
#     index = load_index_from_storage(storage_context)

st.divider()

# If no chat history, create one
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about your webpages!",
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
