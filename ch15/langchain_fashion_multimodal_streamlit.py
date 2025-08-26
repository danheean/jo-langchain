import os
from pathlib import Path
import base64
import streamlit as st
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

model_name = os.getenv("MODEL_NAME") or "gpt-4o-mini"
model_provider = os.getenv("MODEL_PROVIDER") or "openai"

import chromadb

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SCRIPT_DIR = str(Path(__file__).parent.parent)
# SCRIPT_DIR = os.path.dirname(os.path.abspath(Path.cwd()))
# print(SCRIPT_DIR)
def setup_dataset():
    dataset = load_dataset("detection-datasets/fashionpedia")
    dataset_folder = os.path.join(SCRIPT_DIR, 'data', 'fashionpedia')
    os.makedirs(dataset_folder, exist_ok=True)
    return dataset, dataset_folder


def save_images(dataset, dataset_folder, num_images=500):
    for i in range(num_images):
        image = dataset['train'][i]['image']
        image.save(os.path.join(dataset_folder, f'image_{i+1}.png'))

    print(f"Saved {num_images} images to {dataset_folder}")


def setup_chroma_db():
    vdb_path = os.path.join(SCRIPT_DIR, 'index', 'img_vdb')

    chroma_client = chromadb.PersistentClient(path=vdb_path)
    image_loader = ImageLoader()
    CLIP = OpenCLIPEmbeddingFunction()

    image_vdb = chroma_client.get_or_create_collection(
        name="image",
        embedding_function=CLIP,
        data_loader=image_loader,
    )

    return image_vdb


def get_existing_ids(image_vdb, dataset_folder):
    existing_ids = set()
    try:
        num_images = len([name for name in os.listdir(dataset_folder)])
        print(f"데이터 폴더 전체 이미지수: {num_images}")
        records = image_vdb.query(
            query_texts=[""], 
            n_results=num_images, 
            include=['ids']
        )
        for record in records["ids"]:
            existing_ids.update(record)
            print(f"{len(record)} 존재 IDs")
    except Exception as e:
        print(f"{len(record)}개의 기존 IDs가 있습니다.")
    return existing_ids


def add_images_to_db(image_vdb, dataset_folder):
    existing_ids = get_existing_ids(image_vdb, dataset_folder)

    ids = []
    uris = []
    for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
        if filename.endswith('.png'):
            img_id = str(i)
            if img_id not in existing_ids:
                file_path = os.path.join(dataset_folder, filename)
                ids.append(img_id)
                uris.append(file_path)
    if ids:
        image_vdb.add(ids=ids, uris=uris)
        print("이미지가 데이터베이스에 추가되었습니다.")
    else:
        print("추가할 새로운 이미지가 없습니다.")

def query_db(image_vdb, query, results=2):
    return image_vdb.query(
        query_texts=[query],
        n_results=results,
        include=['uris', 'distances']
    )


def translate(text, target_lang):

    translation_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )
    translation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a translator Translate the following text to {target_lang}"),
        ("user", "{text}")
    ])

    # print(text, target_lang)
    translation_chain = translation_prompt | translation_model | StrOutputParser()
    return translation_chain.invoke({"text": text})

def setup_vision_chain():
    gpt4 = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )
    parser = StrOutputParser()
    image_prompt = ChatPromptTemplate.from_messages([
        (
            "system", """You are a helpful fashion and styling assistant. 
            Answer the user's question using the given image context with direct references to parts of the images provided.
            Maintain a more conversational tone, don't make too many list. Use markdown formatting for hightlights, emphasis, and structure"""
        ),
        ("user", 
            [
                {
                    "type": "text",
                    "text": "What are some ideas for styling {user_query}"
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_1}"
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_2}"
                }
            ]
        )
    ])
    
    return image_prompt | gpt4 | parser

def format_prompt_inputs(data, user_query):
    inputs = {}
    inputs['user_query'] = user_query

    # print(inputs)

    image_path_1 = data['uris'][0][0]
    image_path_2 = data['uris'][0][1]

    # print(image_path_1, image_path_2)
    with open(image_path_1, 'rb') as image_file:
        image_data_1 = image_file.read()
    inputs['image_data_1'] = base64.b64encode(image_data_1).decode('utf-8')
        
    with open(image_path_2, 'rb') as image_file:
        image_data_2 = image_file.read()
    inputs['image_data_2'] = base64.b64encode(image_data_2).decode('utf-8')

    #print(inputs)
    return inputs

def load_image_from_db(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def main():
    st.set_page_config(page_title="FashionRAG", page_icon=":fashion:", layout="wide")
    st.title("FashionRAG: 패션 및 스타일링 어시스턴트")

    dataset_folder = os.path.join(SCRIPT_DIR, 'data', 'fashionpedia')
    if not os.path.exists(dataset_folder) or not any(fname.endswith('.png') for fname in os.listdir(dataset_folder)):
        with st.spinner("데이터셋 설정 및 이미지 저장중..."):
            dataset, dataset_folder = setup_dataset()
            save_images(dataset, dataset_folder)
        st.success("데이터셋 설징 및 이미지 저장 중...")
    else:
        st.info("이미지셋이 설정되고 이미지가 저장되었습니다.")

    vdb_path = os.path.join(SCRIPT_DIR, 'index', 'img_vdb')
    if not os.path.exists(vdb_path) or not os.listdir(vdb_path):
        with st.spinner("벡터 데이터베이스 설정 및 이미지 추가 중..."):
            image_vdb = setup_chroma_db()
            add_images_to_db(image_vdb, dataset_folder)
        st.success("벡터 데이터베이스 설정 및 이미지 추가가 완료되었습니다.")
    else:
        st.info("벡터 데이터베이스가 이미 설정되어 잇습니다. 데이터베이스 설정을 건너뜁니다.")
        image_vdb = setup_chroma_db()

    vision_chain = setup_vision_chain() 
    st.header("스타일링 조언을 받아보세요")
    query_ko = st.text_input("스타일링에 대한 질문을 입력하세요:")

    if query_ko:
        with st.spinner("번역 및 쿼리 진행 중..."):
            query_en = translate(query_ko, "English")
            results = query_db(image_vdb, query_en, results=2)
            prompt_input = format_prompt_inputs(results, query_en)
            response_en = vision_chain.invoke(prompt_input)
            response_ko = translate(response_en, "Korean")
            
        st.subheader("검색된 이미지:")
        for idx, uri in enumerate(results['uris'][0]):
            img_base64 = load_image_from_db(uri)
            img_data_url = f"data:image/png;base64,{img_base64}"
            st.image(img_data_url, caption=f"ID: {results['ids'][0][idx]}", width=300)
            
        st.subheader("FashionRAG의 응답:")
        st.markdown(response_ko)
        

if __name__ == "__main__":
    main()