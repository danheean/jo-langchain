import os
import tempfile
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from streamlit_extras.buy_me_a_coffee import button
from langchain.load import dumps, loads
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

# 'Buy me a coffee'
button(username="jurepi", floating=True, width=221)

st.title("ChatPDF with Multiquery+hybridSearch+RagFusion")
st.write("---")
st.write("PDF 파일을 업로드하고 내용을 기반으로 질문하세요.")

openai_key = st.text_input("OpenAI API 키를 입력해주세요.", type="password")

model_choice = st.selectbox(
    "사용할 GPT 모델을 선택하세요:",
    ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4o']
)

uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=["pdf"])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    return docs

def format_docs(docs):
    formatted = "\n\n".join(doc.page_content for doc in docs)
    return formatted

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(pages)

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_key
    )

    embedding_dimension = len(embedding_model.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dimension)

    vectorstore = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vectorstore.add_documents(documents=splits, ids=range(len(splits)))

    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=dict(
            k=1,
            fetch_k=4
        )
    )

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 2

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.8, 0.2]
    )

    template = """
        당신은 AI 언어 모델 조수입니다. 당신의 임무는 주어진 사용자 질문에 대해 벡터 데이터베이스에서 관련 문서를 검색할 수 있도록 다섯 가지 다른 버전을 생성하는 것이빈다.
        사용자 질문에 대한 여러 관점을 생성함으로써, 거리 기반 유사성 검색의 한계를 극복하는 데 도움을 주는 것이 목표입니다.
        각 질문은 새로운 줄로 구분하여 제공하세요. 원본 질문: {question}
    """

    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | ChatOpenAI(model=model_choice, temperature=0, openai_api_key=openai_key)
        | StrOutputParser()
        |(lambda x: x.split("\n"))
    )

    def reciprocal_rank_fusion(results: list[list], k=60, top_n=2):
        """
        여러 개의 순위가 매겨진 문서 리스트를 받아, RRF(Reciprocal Rank Fusion) 공식을 사용하여 문서의 최종 순위를 계산하는 함수입니다. 
        k는 RRF 공식에서 사용되는 선택적 파라미터이며, top_n은 반환할 우선순위가 높은 문서의 개수입니다.
        """

        fused_scores = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                previous_score = fused_scores[doc_str]
                fused_scores[doc_str] = 1 / (rank + 1)

        fused_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return fused_results[:top_n]

    retrieval_chain_rag_fusion = generate_queries | ensemble_retriever.map() | reciprocal_rank_fusion

    template = """다음 맥락을 바탕으로 질문에 답변하세요:
        {context}

        질문: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model=model_choice, temperature=0, openai_api_key=openai_key)

    final_rag_chain = (
        {
            "context": retrieval_chain_rag_fusion, 
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    st.header("PDF에 질문하세요!")
    question = st.text_input("질문을 입력해주세요.")

    if st.button("질문하기(ASK)"):
        with st.spinner("답변 생성 중..."):
            result = final_rag_chain.invoke(question)
            st.write(result)
