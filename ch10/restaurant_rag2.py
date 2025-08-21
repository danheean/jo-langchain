
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

load_dotenv(find_dotenv())

model_name = os.getenv("LLM_MODEL") or "gpt-4o-mini"
model_provider = os.getenv("LLM_MODEL_PROVIDER") or "openai"

current_dir = Path(__file__).parent
data_dir = current_dir.parent / "data"
index_dir = current_dir.parent / "index"
restaurant_faiss = index_dir / "restaurant-faiss"

async def main():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    load_db = FAISS.load_local(
        str(restaurant_faiss),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    query = "음식점의 룸 서비스는 어떻게 운영되나요?"

    results = load_db.similarity_search(query, k=2)

    print(results, "\n")

    embedding_vector_query = embeddings.embed_query(query)
    print("Query vector: ", embedding_vector_query, "\n")

    docs = await load_db.asimilarity_search_by_vector(embedding_vector_query)
    print(docs[0], "\n")

if __name__ == "__main__":
    asyncio.run(main())