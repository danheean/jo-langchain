from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from pprint import pprint

load_dotenv(find_dotenv())

model_name = os.getenv("LLM_MODEL") or "gpt-4o-mini"
model_provider = os.getenv("LLM_MODEL_PROVIDER") or "openai"

# https://github.com/sw-woo/hanbit-langchain/blob/main/chapter%2009~11/restaurants.txt

current_dir = Path(__file__).parent
data_dir = current_dir.parent / "data"
index_dir = current_dir.parent / "index"

restaurant_faiss = index_dir / "restaurant-faiss"
loader = TextLoader(str(data_dir / "restaurants.txt"))

documents = loader.load()
#print(len(documents))
#print(documents[0])

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)

docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

db = FAISS.from_documents(docs, embeddings)
db.save_local(str(restaurant_faiss))
print("레스토랑 임베딩 저장 완료")