# 웹 검색 등 커스텀 도구 정의

from langchain_tavily import TavilySearch
from src import config
from .rag import internal_document_search_tool

# 도구 정의 부분
search_tool = TavilySearch(max_results=3, api_key=config.TAVILY_API_KEY)
search_tool.name = "web_search"
search_tool.description = "최신 정보나 일반적인 주제에 대해 웹 검색이 필요할 때 사용합니다."

# 모든 도구를 리스트로 묶어 제공합니다.
# 나중에 데이터베이스 검색 등의 도구가 추가되면 이 리스트에 추가하면 됩니다.
tools = [search_tool, internal_document_search_tool]