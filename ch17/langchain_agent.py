from langchain.agents import AgentExecutor 

from langchain.agents import create_openai_tools_agent

from langchain import hub
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader 

from langchain_community.utilities import WikipediaAPIWrapper 
from langchain_community.tools import WikipediaQueryRun

from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model_name = os.getenv("MODEL_NAME") or "gpt-4o-mini"
model_provider = os.getenv("MODEL_PROVIDER") or "openai"

openai = ChatOpenAI(
    model=model_name,
    temperature=0.1
)

prompt = hub.pull("hwchase17/openai-functions-agent")

api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=400
)

wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
print(wiki.name)

loader = WebBaseLoader(
    "https://news.naver.com",
    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

docs = loader.load()

documents = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
).split_documents(docs)

vectordb = FAISS.from_documents(
    documents,
    OpenAIEmbeddings(model="text-embedding-3-large")
)

retriever = vectordb.as_retriever()
print(retriever)

retriever_tool = create_retriever_tool(
    retriever,
    "naver_news_search",
    "네이버 뉴스 정보가 저장된 벡터 DB, 당일 기사에 대해 궁금하면 이 툴을 사용하세요!"
)
print(retriever_tool.name)

arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=400,
    load_all_available_meta=False,
)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
print(arxiv.name)

tools = [wiki, arxiv, retriever_tool]

agent = create_openai_tools_agent(llm=openai, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_result = agent_executor.invoke({"input": "오늘 이재명 대통령 관련 주요 소식을 알려줘"})

print(agent_result)
