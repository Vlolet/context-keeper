import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

# --- (도구 정의, AgentState 정의, 모델 정의 부분은 test2와 동일) ---
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """사용자가 최신 정보나 특정 주제에 대해 물어볼 때 웹을 검색하여 결과를 반환합니다."""
    print(f"--- 웹 검색 수행: {query} ---")
    return f"'{query}'에 대한 검색 결과: 오늘은 맑고 화창한 날씨입니다. 미세먼지 농도는 '좋음'입니다."

tools = [web_search]

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
model_with_tools = model.bind_tools(tools)

def call_model(state: AgentState):
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "call_tool"
    else:
        return "__end__"

# --- (그래프 설계 부분은 test2와 동일) ---
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


# --- 6. 실행 및 테스트 (핵심 변경 부분) ---
print("### LangGraph 대화 기록 유지 테스트 ###")
print("질문을 입력하세요 (종료하려면 'exit' 입력):")

# ** 변경점 1: 대화 기록을 저장할 리스트를 루프 바깥에 생성 **
conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # ** 변경점 2: 현재 대화 기록을 기반으로 inputs를 구성 **
    # HumanMessage 객체를 사용하여 사용자 입력을 기록에 추가합니다.
    conversation_history.append(HumanMessage(content=user_input))
    
    inputs = {"messages": conversation_history}
    
    # 그래프 실행
    final_state = app.invoke(inputs)
    
    # ** 변경점 3: 실행 후, 전체 대화 기록을 다시 가져와서 업데이트 **
    # final_state['messages']에는 HumanMessage와 AIMessage가 모두 포함된 전체 대화록이 들어있습니다.
    conversation_history = final_state['messages']
    
    # 최종 AI 답변만 출력
    ai_response = conversation_history[-1].content
    print(f"AI: {ai_response}")