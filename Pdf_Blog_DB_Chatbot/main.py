import streamlit as st
import asyncio
import pickle
import os
import io

from langchain_community.chat_models import ChatOpenAI
from pypdf import PdfReader
from altair import Chart

from langchain.document_loaders import TextLoader
from langchain_community.llms import openai
from streamlit_chat import message
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.text import TextLoader

import html2text
import requests

st.set_page_config(
    page_title="PDF_CHAT",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a Bug': 'https://github.com/streamlit/streamlit/issues',
        'About': 'This is a PDF Chat app built with Streamlit.'
    }
)

api_key = os.getenv("OPEN_API_KEY")
if not api_key:
    st.error("API key not found. Please set the OPEN_API_KEY environment variable.")
    st.stop()

openai.api_key = api_key


async def main():
    async def storeDocEmbeds(file, filename): # for PDF
        reader = PdfReader(file)
        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        chunks = splitter.split_text(corpus)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.from_texts(chunks, embeddings)

        try:
            with open(filename + '.pkl', 'wb') as f:
                pickle.dump(vectors, f)
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            return None


    async def getDocEmbbs(file, filename): # for PDF
        if not os.path.isfile((filename + '.pkl')):
            await storeDocEmbeds(file, filename)

        try:
            with open(filename + '.pkl', 'rb') as f:
                vectors = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            return None

        return vectors

    async def storeStringEmbeds(input_string, filename): # for blog
        try:
            with open(filename, 'w') as f:
                f.write(input_string)
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            return None

        loader = TextLoader(filename)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        chunks = splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectors = FAISS.from_texts(chunks, embeddings)

        try:
            with open(filename + '.pkl', 'wb') as f:
                pickle.dump(vectors, f)
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            return None

    async def getStringEmbbs(file, filename): # for BLOG
        if not os.path.isfile((filename + '.pkl')):
            await storeStringEmbeds(file, filename)

        try:
            with open(filename + '.pkl', 'rb') as f:
                vectors = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            return None

        return vectors

    def extract_text_from_url(url):
        response = requests.get(url)
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        text = converter.handle(response.text)
        return text

    async def conversational_chat(query):
        result = qa({"question": query, 'chat_history': st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    st.title("PDFCHAT: ")

    option = st.selectbox("select option", ("PDF", "Blog", "Database"))

    if option == "PDF":
        uploaded_file = st.file_uploader("choose a file", type="PDF")
        if uploaded_file is not None:
            with st.spinner("processing..."):
                uploaded_file.seek(0)
                file = uploaded_file.read()
                vectors = await getDocEmbbs(io.BytesIO(file), uploaded_file.name)
                try:
                    qa = ConversationalRetrievalChain.from_llm(
                        ChatOpenAI(model_name="gpt-3.5-turbo"),
                        retriever=vectors.as_retriever(),
                        return_source_documents=True
                    )
                except Exception as e:
                    st.error(f"Error creating retrieval chain: {str(e)}")


            st.session_state["ready"] = True

    elif option == "Blog":
        url = st.text_input("enter the URL of the blog")

        if url:
            with st.spinner("Processing..."):
                content = extract_text_from_url(url)
                vectors = await getStringEmbbs(content, "blog.txt")
                try:
                    qa = ConversationalRetrievalChain.from_llm(
                        ChatOpenAI(model_name="gpt-3.5-turbo"),
                        retriever=vectors.as_retriever(),
                        return_source_documents=True
                    )
                except Exception as e:
                    st.error(f"Error creating retrieval chain: {str(e)}")


            st.session_state["ready"] = True

    elif option == "Database":
        uploaded_file = st.file_uploader("choose a DB file", type="db")
        if uploaded_file is not None:
            with st.spinner("Proccessing.."):
                uploaded_file.seek(0)

            st.session_state["ready"] = True

    if st.session_state.get('ready', False):
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcome! you can ask any questions"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey"]

        container = st.container()
        response_container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="e.g: summarize the document", key='input')
                submit_button = st.form_submit_button(label='send')

            if submit_button and user_input:
                output = await conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    if i < len(st.session_state['past']):
                        st.markdown(
                            "<div style='background-color: #90caf9; color: black; padding: 10px; border-radius: 5px; width: 70%; float: right; margin: 5px;'>" +
                            st.session_state["past"][i] + "</div>",
                            unsafe_allow_html=True
                        )
                    message(st.session_state['generated'][i], f"generated_{i}")


if __name__ == "__main__":
    asyncio.run(main())
