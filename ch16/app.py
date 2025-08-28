from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langserve import add_routes
import uvicorn

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model_name = os.getenv("MODEL_NAME") or "gpt-4o-mini"
model_provider = os.getenv("MODEL_PROVIDER") or "openai"

# print(os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="Langchain Server",
    version="0.1",
    description="Simple langchain API Server",
)

openAiModel = ChatOpenAI(
    model=model_name,
    temperature=0.7
)

ollamaModel = OllamaLLM(model="gemma3:12b")

prompt = ChatPromptTemplate.from_template("한국어로 답변을 작성해줘 {input}")
prompt2 = ChatPromptTemplate.from_template("주제에 맞는 소설을 작성해줘 500자 이내로 작성해줘 {topic}")
prompt3 = ChatPromptTemplate.from_template("주제에 맞는 시를 작성해줘 200자 이내로 작성해줘 {topic}")

try:
    add_routes(
        app,
        prompt | openAiModel,
        path="/openai"
    )

    add_routes(
        app,
        prompt2 | openAiModel,
        path="/openai/novel"
    )

    add_routes(
        app,
        prompt3 | openAiModel,
        path="/openai/poem"
    )

    add_routes(
        app,
        prompt | ollamaModel,
        path="/ollama"
    )

    add_routes(
        app,
        prompt2 | ollamaModel,
        path="/ollama/novel"
    )

    add_routes(
        app,
        prompt3 | ollamaModel,
        path="/ollama/poem"
    )
except Exception as e:
    print(f"Error adding routes: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)