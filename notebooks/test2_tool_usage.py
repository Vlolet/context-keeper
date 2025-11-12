import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode  # ToolNode를 임포트합니다.
from typing import TypedDict, Annotated
import operator

# --- 1. 도구(Tool) 정의 ---
# 에이전트가 사용할 수 있는 함수를 만듭니다.
# @tool 데코레이터를 사용하면 LangChain이 이 함수를 '도구'로 인식합니다.
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """
    사용자가 최신 정보나 특정 주제에 대해 물어볼 때 웹을 검색하여 결과를 반환합니다.
    예: '오늘 서울 날씨 어때?' -> web_search(query='오늘 서울 날씨')
    """
    print(f"--- 웹 검색 수행: {query} ---")
    # 실제로는 여기서 웹 크롤링이나 검색 API를 호출해야 합니다.
    # 지금은 테스트를 위해 가짜 결과물을 반환합니다.
    return f"'{query}'에 대한 검색 결과: 오늘은 맑고 화창한 날씨입니다."

# 에이전트가 사용할 도구들을 리스트로 묶어줍니다.
tools = [web_search]

# --- 2. AgentState (상태) 정의 ---
# 이전과 동일합니다. messages는 대화 기록을 계속 추가(add)하는 방식으로 업데이트됩니다.
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# --- 3. 모델과 노드(Node) 정의 ---
# 모델에 우리가 만든 도구를 사용할 수 있다고 알려줍니다.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
model_with_tools = model.bind_tools(tools)

# LLM을 호출하는 노드는 이제 model_with_tools를 사용합니다.
def call_model(state: AgentState):
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# 도구를 실행하는 노드를 만듭니다. ToolNode는 이 과정을 매우 쉽게 해줍니다.
# ToolNode가 알아서 LLM의 tool_calls를 보고, 맞는 도구를 실행해줍니다.
tool_node = ToolNode(tools)

# --- 4. 라우터(Router) 함수 정의 ---
# LLM의 답변을 보고, 다음에 어떤 노드로 가야 할지 결정하는 함수입니다.
def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    # LLM이 도구를 사용하라는 명령(tool_calls)을 내렸는지 확인합니다.
    if last_message.tool_calls:
        # 그렇다면, 다음 목적지는 'tool_node' (도구 실행기) 입니다.
        return "call_tool"
    else:
        # 아니라면, 그냥 답변한 것이므로 '종료'합니다.
        return "__end__"

# --- 5. 그래프(Graph) 설계 및 연결 ---
# 이제 모든 조각들을 조립합니다.
workflow = StateGraph(AgentState)

# 2개의 노드를 추가합니다: LLM 호출 노드, 도구 실행 노드
workflow.add_node("llm", call_model)
workflow.add_node("call_tool", tool_node)

# 진입점(Entry Point)은 항상 LLM입니다.
workflow.set_entry_point("llm")

# ** 여기가 핵심: 조건부 엣지 (Conditional Edge) **
# 'llm' 노드가 끝난 후, 'should_continue' 함수를 실행하여 다음 경로를 결정합니다.
# - should_continue가 "call_tool"을 반환하면 -> 'call_tool' 노드로 이동
# - should_continue가 "__end__"를 반환하면 -> 그래프 종료(END)
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "call_tool": "call_tool",
        "__end__": END,
    },
)

# 도구 실행이 끝나면, 다시 LLM으로 돌아가서 결과를 보고 최종 답변을 생성하도록 합니다.
workflow.add_edge("call_tool", "llm")

# 그래프 컴파일
app = workflow.compile()

# --- 6. 실행 및 테스트 ---
print("### LangGraph 도구 사용 테스트 ###")
print("질문을 입력하세요 (종료하려면 'exit' 입력):")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    inputs = {"messages": [("user", user_input)]}
    # stream 대신 invoke를 사용하여 최종 결과만 확인
    final_state = app.invoke(inputs)
    
    # 최종 답변은 마지막 메시지에 들어있습니다.
    ai_response = final_state['messages'][-1].content
    print(f"AI: {ai_response}")