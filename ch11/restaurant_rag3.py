from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model_name = os.getenv("LLM_MODEL") or "gpt-4o-mini"
model_provider = os.getenv("LLM_MODEL_PROVIDER") or "openai"

current_dir = Path(__file__).parent
#print(current_dir, current_dir.parent, Path.cwd())

data_dir = current_dir.parent / "data"
index_dir = current_dir.parent / "index"

restaurant_faiss = index_dir / "restaurant-faiss"
restaurant_text = data_dir / "restaurants.txt"

def create_faiss_index():
    loader = TextLoader(str(restaurant_text))
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.create_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(restaurant_faiss))
    
    print("Faiss Index created and saved")

def load_faiss_index():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    load_db = FAISS.load_local(
        str(restaurant_faiss),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return load_db

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def answer_question(db, query):
    llm = ChatOpenAI(model=model_name)
    #llm = OpenAI(model=model_name)
    prompt_template = """
    당신은 유능한 AI 비서입니다. 주어진 맥락 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해야 합니다.
    맥락: {context}
    질문: {question}
    답변을 작성할 때 다음 지침을 따르세요:
    1. 주어진 맥락 정보에 있는 내용만을 사용하여 답변하세요.
    2. 맥락 정보에 없는 내용은 답변에 포함하지 마세요.
    3. 질문과 관련이 없는 정보는 제외하세요
    4. 답변은 간결하고 명확하게 작성하세요.
    5. 불확실한 경우, "주어진 정보로는 정확한 답변을 드릴 수 없습니다."라고 답변하세요.
    답변: 
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    qa_chain = (
        {
            "context": db.as_retriever() | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    #result = qa_chain.invoke({"input": query})
    result = qa_chain.invoke(query)
    return result

def main():
    if not os.path.exists(str(restaurant_faiss)):
        create_faiss_index()

    db = load_faiss_index()
    while True:
        query = input("레스토랑에 대해서 궁금한 점을 물어보세요 (종료하려면 'quit' 입력): ")
        if query.lower() == "quit":
            print("프로그램을 종료합니다.")
            break
        answer = answer_question(db, query)
        print(f"답변: {answer}\n")
        
if __name__ == "__main__":
    main()