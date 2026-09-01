# src/data/ingestion.py
# local_data 폴더의 문서를 읽고, 의미 단위로 잘게 쪼갠(Chunking) 뒤, 벡터로 변환하여 DB에 저장하는 '데이터 주입' 파이프라인입니다.

import os
os.environ["GRPC_DNS_RESOLVER"] = "native"
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader

# 현재 파일의 상위 모듈을 import 하기 위한 경로 설정
# 이 스크립트를 단독으로 실행할 때 필요합니다.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.storage import get_vector_store

DATA_PATH = "local_data"

def ingest_documents():
    """local_data 폴더의 문서를 로드, 분할하여 Vector DB에 저장합니다."""
    
    # 1. 문서 로드 (Markdown 파일만 대상으로)
    loader = DirectoryLoader(
        DATA_PATH, 
        glob="**/*.md", 
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True
    )
    documents = loader.load()
    
    if not documents:
        print("No new documents to ingest.")
        return

    # 2. 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    # 3. DB에 저장
    print(f"Ingesting {len(splits)} document splits...")
    vector_store = get_vector_store()
    vector_store.add_documents(documents=splits)
    print("Ingestion complete.")

# 이 파일을 직접 실행하여 데이터 주입을 수행합니다.
if __name__ == '__main__':
    ingest_documents()