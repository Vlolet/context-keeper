# src/data/storage.py
# ChromaDB를 초기화하고, 전체 시스템에서 단 하나의 DB 인스턴스만 사용하도록 설정(Singleton 패턴)하는 역할을 합니다.

import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src import config

# Vector DB가 저장될 경로
CHROMA_PATH = "chroma_db"
# 임베딩 모델 정의
embedding_function = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=config.GOOGLE_API_KEY
)

def get_vector_store() -> Chroma:
    """
    ChromaDB 클라이언트를 초기화하고 Vector Store 객체를 반환합니다.
    Streamlit의 cache_resource를 사용하여 앱 전체에서 단일 인스턴스를 유지합니다.
    """
    # 디스크 기반 클라이언트 생성
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Vector Store 객체 생성 (기존 collection이 있으면 로드, 없으면 생성)
    vector_store = Chroma(
        client=client,
        collection_name="context_keeper_docs",
        embedding_function=embedding_function,
    )
    print(f"Vector store initialized with {vector_store._collection.count()} documents.")
    return vector_store

# 이 파일을 직접 실행할 경우, DB 초기화 테스트
if __name__ == '__main__':
    db = get_vector_store()
    print("Vector store connection successful.")
    print(f"Number of documents in collection: {db._collection.count()}")