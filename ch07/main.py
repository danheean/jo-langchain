# Chroma sqlite3 에러 대응
# __import__("pysqlite3")
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import streamlit as st
from streamlit_extras.buy_me_a_coffee import button
import pandas as pd
from io import StringIO
import tempfile

model_name = os.getenv("LLM_MODEL") or "gpt-4o-mini"
model_provider = os.getenv("LLM_MODEL_PROVIDER") or "openai"

# 제목
st.title(f"ChatPDF ({model_name})")
st.write("---")

openai_key = st.text_input("OpenAI API 키를 입력해주세요.", type="password")

uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=["pdf"])
st.write("---")

# Buy me a coffee
button(username="jurepi", floating=True, width=201)

def pdf_to_document(uploaded_file):
    """ Streamlit 예시 코드
    if uploaded_file is not None: 
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        st.write(stringio)

        string_data = stringio.getvalue()
        st.write(string_data)
        dataframe = pd.read_dsv(uploaded_file)
        st.write(dataframe)
    """

    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()

    return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    print(len(pages))

    # Splittter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_documents(pages)
    print(texts[0])

    # Embeddings
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_key
    )

    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    # Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text
        
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text += token
            self.container.markdown(self.text)

    # User Input
    st.header("PDF에게 질문을 해보세요!")
    question = st.text_input("질문을 입력해주세요.")

    if st.button("질문하기"):
        with st.spinner("답변 생성 중..."):
            # LLM
            llm = ChatOpenAI(model=model_name, temperature=0, openai_api_key=openai_key)

            # Retriever
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=db.as_retriever(),
                llm=llm
            )

            def format_docs(docs):
                search_results = "\n\n".join([doc.page_content for doc in docs])
                # print(search_results)
                return search_results

            prompt = hub.pull("rlm/rag-prompt")

            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box)

            generate_llm = ChatOpenAI(
                model=model_name, 
                temperature=0, 
                openai_api_key=openai_key, 
                streaming=True, 
                callbacks=[stream_handler]
            )

            # RAG
            rag_chain = (
                {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
                | prompt 
                | generate_llm 
                | StrOutputParser()
            )

            print(question)
            result = rag_chain.invoke({"input": question})

            # st.write(result)
            # print(result)