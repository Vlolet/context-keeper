# src/data/retriever.py
# Vector DB에서 가장 관련성 높은 문서를 찾아주는 '검색기'를 생성하는 역할을 합니다.

from langchain_core.vectorstores import VectorStoreRetriever
from src.data.storage import get_vector_store

def get_retriever() -> VectorStoreRetriever:
    """
    Vector DB로부터 Retriever 객체를 생성하여 반환합니다.
    검색 시 상위 3개의 문서를 가져오도록 설정합니다 (k=3).
    """
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": 3})