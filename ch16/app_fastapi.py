from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
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

class ChatRequest(BaseModel):
    input: str

class TopicRequest(BaseModel):
    topic: str

prompt = ChatPromptTemplate.from_template("한국어로 답변을 작성해줘 {input}")
prompt2 = ChatPromptTemplate.from_template("주제에 맞는 소설을 작성해줘 500자 이내로 작성해줘 {topic}")
prompt3 = ChatPromptTemplate.from_template("주제에 맞는 시를 작성해줘 200자 이내로 작성해줘 {topic}")

@app.get("/")
async def root():
    return {"message": "Langchain Server is running"}

@app.post("/openai")
async def openai_chat(request: ChatRequest):
    chain = prompt | openAiModel
    response = chain.invoke({"input": request.input})
    return {"response": response.content}

@app.post("/openai/novel")
async def openai_novel(request: TopicRequest):
    chain = prompt2 | openAiModel
    response = chain.invoke({"topic": request.topic})
    return {"response": response.content}

@app.post("/openai/poem")
async def openai_poem(request: TopicRequest):
    chain = prompt3 | openAiModel
    response = chain.invoke({"topic": request.topic})
    return {"response": response.content}

@app.post("/ollama")
async def ollama_chat(request: ChatRequest):
    chain = prompt | ollamaModel
    response = chain.invoke({"input": request.input})
    return {"response": response}

@app.post("/ollama/novel")
async def ollama_novel(request: TopicRequest):
    chain = prompt2 | ollamaModel
    response = chain.invoke({"topic": request.topic})
    return {"response": response}

@app.post("/ollama/poem")
async def ollama_poem(request: TopicRequest):
    chain = prompt3 | ollamaModel
    response = chain.invoke({"topic": request.topic})
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)