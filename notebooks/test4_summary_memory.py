import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

# --- 0. utils ---
# AI 응답에서 텍스트를 안전하게 추출하는 함수
def get_ai_response_content(ai_message: AIMessage) -> str:
    """AIMessage의 content가 문자열이거나 리스트 형태일 수 있으므로, 이를 처리하여 텍스트만 반환합니다."""
    content = ai_message.content
    if isinstance(content, list) and content and isinstance(content[0], dict):
        # [{'type': 'text', 'text': '...'}] 와 같은 구조를 처리
        return content[0].get('text', '')
    elif isinstance(content, str):
        return content
    return str(content) # 예외적인 경우를 대비해 문자열로 변환


# --- 1. 설정 및 도구 정의 (이전과 동일) ---
load_dotenv()

@tool
def web_search(query: str) -> str:
    """사용자가 최신 정보나 특정 주제에 대해 물어볼 때 웹을 검색하여 결과를 반환합니다."""
    print(f"--- 웹 검색 수행: {query} ---")
    if "gemini-2.5" in query.lower():
        return "Gemini 2.5는 Google의 최신 고성능 모델로, Flash와 Pro 버전이 존재합니다."
    return f"'{query}'에 대한 일반 검색 결과입니다."

tools = [web_search]

# --- 2. AgentState 정의 (메모리 관련 필드 추가) ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    # ** 신규 추가: 요약된 메모리를 저장할 필드 **
    memory_summary: str

# --- 3. 모델 및 메모리 관리자 정의 ---
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
model_with_tools = model.bind_tools(tools)


# --- 4. LangGraph 노드 및 그래프 설계 (이전과 거의 동일) ---
# AgentState에 memory_summary가 추가되었지만, 그래프 로직 자체는 변경되지 않습니다.
def call_model(state: AgentState):
    # ** 변경점: 입력으로 들어온 messages를 그대로 사용 **
    # 메모리 관리는 그래프에 들어오기 전에 이미 처리되었다고 가정합니다.
    response = model_with_tools.invoke(state['messages'])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "call_tool"
    else:
        return "__end__"

workflow = StateGraph(AgentState)
workflow.add_node("llm", call_model)
workflow.add_node("call_tool", tool_node)
workflow.set_entry_point("llm")
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {"call_tool": "call_tool", "__end__": END},
)
workflow.add_edge("call_tool", "llm")
app = workflow.compile()


# --- 5. 실행 및 테스트 (메모리 관리 로직 추가) ---
print("### LangGraph 요약 메모리 테스트 ###")
print("대화를 12번 이상 진행하여 메모리 요약 기능을 테스트하세요.")
MEMORY_THRESHOLD = 12  # 8개 메시지를 초과하면 요약을 실행합니다.

# 대화 기록과 요약본을 함께 관리
conversation_history: List[BaseMessage] = []
summary_text = ""

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    conversation_history.append(HumanMessage(content=user_input))
    
    inputs = {"messages": conversation_history}
    final_state = app.invoke(inputs)
    
    conversation_history = final_state['messages']
    ai_response_message = conversation_history[-1]
    response_text = get_ai_response_content(ai_response_message)
    print(f"AI: {response_text}")
    
    # ** 턴이 끝나고 메모리 상태 점검 **
    if len(conversation_history) >= MEMORY_THRESHOLD:
        print(f"\n--- 메모리 관리자 작동 (현재 {len(conversation_history)}개 메시지) ---")
        
        # 메시지 객체를 대화형으로 재구성
        dialog_text = ""
        for msg in conversation_history:
            if isinstance(msg, HumanMessage):
                dialog_text += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                dialog_text += f"AI: {get_ai_response_content(msg)}\n"
        
        # 요약 프롬프트 구성
        summarizer_prompt = [
            SystemMessage(content="You are a conversation summarizer. Summarize the following dialogue into one concise Korean sentence, incorporating the previous summary if provided."),
            HumanMessage(content=dialog_text)
        ]
        
        # 요약 실행 및 업데이트
        summary_response = model.invoke(summarizer_prompt)
        summary_text = get_ai_response_content(summary_response)
        print(f"생성된 요약: {summary_text}")

        # 다음 턴부터 요약본을 대화 기록으로 사용
        conversation_history = [
            SystemMessage(content=f"This is a summary of the previous conversasion: {summary_text}")
        ]
        print("---메모리가 요약본으로 교체되었습니다. ---")