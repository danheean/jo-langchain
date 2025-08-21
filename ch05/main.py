from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

import streamlit as st

# 프로젝트 루트의 .env 파일 로드 (자동 탐색)
load_dotenv(find_dotenv())

model_name = os.getenv("LLM_MODEL") or "gpt-4o-mini"
model_provider = os.getenv("LLM_MODEL_PROVIDER") or "openai"

llm = init_chat_model(model_name, model_provider=model_provider)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

# 제목
st.title('인공지능 시인')

# 시 주제 입력 필드
content = st.text_input('시의 주제를 제시해주세요.')

if content:
    st.write("시의 주제는", content)
    if st.button("시 작성 요청하기"):
        with st.spinner("시를 작성중입니다..."):
            result = chain.invoke({"input": content + "에 대한 시를 써줘"})
            st.write(result)