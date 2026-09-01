# src/graph/utils.py

from langchain_core.messages import AIMessage

def parse_llm_response(response: AIMessage) -> str:
    """
    LangChain 모델의 AIMessage 응답을 파싱하여 순수한 텍스트만 추출합니다.
    - content가 단순 문자열인 경우 그대로 반환합니다.
    - content가 복잡한 list/dict 구조인 경우, 'text' 키를 찾아 반환합니다.
    
    Args:
        response: LLM으로부터 받은 AIMessage 객체.
        
    Returns:
        정제된 순수 텍스트(string).
    """
    raw_content = response.content
    
    if isinstance(raw_content, str):
        # Case 1: 이미 순수한 문자열인 경우
        return raw_content
    elif isinstance(raw_content, list) and raw_content and isinstance(raw_content[0], dict):
        # Case 2: 복잡한 구조인 경우 'text' 값 추출
        return raw_content[0].get("text", "")
    
    # 예외 처리: 예상치 못한 형식일 경우 빈 문자열 반환
    print("\n⚠️ [Warning] 예상치 못한 형식의 LLM 응답이 감지되었습니다. 응답 객체를 출력합니다.")
    print("------- AIMessage -------")
    print(response)
    print("------- --------- -------\n")
    return ""