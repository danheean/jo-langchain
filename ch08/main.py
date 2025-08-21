import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY_UNSU")
client = OpenAI(api_key=openai_api_key)

import streamlit as st

# 제목과 초기화 버튼을 같은 줄에 배치
if st.button("🔄 초기화", type="secondary"):
    st.session_state.chat_history = []
    if 'response_id' in st.session_state:
        del st.session_state.response_id
    st.rerun()
st.title("현진건 작가님과의 대화")
st.write("---")

# 대화 히스토리 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# 대화 히스토리 표시
if 'response_id' in st.session_state:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# 질문 입력 (Streamlit의 chat_input은 자동으로 페이지 하단에 고정됨)
prompt = st.chat_input("물어보고 싶은 것을 입력하세요!")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    if 'response_id' not in st.session_state:
        with st.spinner("Wait for it..."):
            response = client.responses.create(
                model="gpt-4o-mini",
                instructions="당신은 소설 운수 좋은 날을 집필한 현진건 작가님입니다.",
                # input=[
                #     {
                #         "role": "user",
                #         "content": [
                #             {
                #                 "type": "input_text",
                #                 "text": "아내가 먹고 싶어한 음식이 뭐야?"
                #             }
                #         ]
                #     }
                # ],
                input=prompt,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [
                            "vs_68a6d3e61f648191bfbf3395f226fb02"
                        ]
                    }
                ],
                temperature=1,
                max_output_tokens=2048,
                top_p=1,
                store=True
            )
    else:
        with st.spinner("Wait for it..."):
            response = client.responses.create(
                previous_response_id=st.session_state.response_id,
                model="gpt-4o-mini",
                instructions="당신은 소설 운수 좋은 날을 집필한 현진건 작가님입니다.",
                input=prompt,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": ["vs_68a6d3e61f648191bfbf3395f226fb02"]
                    }
                ],
            )

    with st.chat_message("assistant"):
        st.write(response.output_text)
    
    st.session_state.chat_history.append({"role": "assistant", "content": response.output_text})
    st.session_state.response_id = response.id