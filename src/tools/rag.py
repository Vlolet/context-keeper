# src/tools/rag.py

from langchain_core.tools import Tool
from src.data.retriever import get_retriever

def _search_internal_documents(query: str) -> str:
    """내부 문서 저장소(Vector DB)에서 정보를 검색하는 실제 로직"""
    print(f"[Tool Called] internal_document_search with query: '{query}'")
    
    retriever = get_retriever()
    docs = retriever.invoke(query)
    
    # 검색된 문서 내용을 하나의 문자열로 합쳐서 반환
    return "\n\n".join([doc.page_content for doc in docs])

# LangChain이 이해할 수 있는 'Tool' 형태로 포장
# name과 description이 라우터(LLM)의 판단에 가장 중요한 역할을 합니다.
internal_document_search_tool = Tool(
    name="internal_document_search",
    func=_search_internal_documents,
    description="프로젝트 가이드라인, 핵심 철학, 아키텍처 원칙 등 내부 문서에 저장된 정보를 검색할 때 사용합니다. 사용자가 프로젝트의 내부 정보에 대해 질문할 때 가장 먼저 사용해야 하는 도구입니다."
)