# GUI 애플리케이션 실행 파일

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool


# --- 1. 설정 및 에이전트 로직 ---

load_dotenv()
MODEL_NAME = "gemini-2.5-flash"

@tool
def web_search(query: str) -> str:
    """
    최신 정보, 특정 인물, 장소, 이벤트, 기술 용어(예: Gemini 2.5) 또는 실시간 정보(예: 날씨, 뉴스)에 대한 질문에 답변하기 위해 사용합니다.
    사용자의 질문이 AI의 내부 지식만으로 답변하기 어렵다고 판단될 때, 반드시 이 도구를 사용해야 합니다.
    """
    # 실제로는 여기서 API를 호출해야 합니다. 지금은 가짜 결과물을 반환합니다.
    st.sidebar.info(f"🔎 웹 검색 수행: {query}") # UI에 검색 과정을 표시
    if "gemini-2.5" in query.lower():
        return "Gemini 2.5는 Google의 최신 고성능 모델로, Flash와 Pro 버전이 존재합니다."
    return f"'{query}'에 대한 일반 검색 결과입니다."

tools = [web_search]

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.7)
model_with_tools = model.bind_tools(tools)

def call_model(state: AgentState):
    response = model_with_tools.invoke(state['messages'])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    if state['messages'][-1].tool_calls:
        return "call_tool"
    return "__end__"

workflow = StateGraph(AgentState)
workflow.add_node("llm", call_model)
workflow.add_node("call_tool", tool_node)
workflow.set_entry_point("llm")
workflow.add_conditional_edges("llm", should_continue)
workflow.add_edge("call_tool", "llm")
app = workflow.compile()


# --- 2. Streamlit UI 구현 ---

st.set_page_config(page_title="Context Keeper", page_icon="🧠")
st.title("🧠 Context Keeper")

SYSTEM_PROMPT = """당신은 유능하고 적극적인 AI 비서 'Context Keeper'입니다. 당신의 임무는 다음과 같습니다:
1. 사용자의 질문에 최대한 정확하고 친절하게 답변합니다.
2. 모르는 정보나 최신 정보가 필요하다고 판단되면, 주저하지 말고 당신이 가진 'web_search' 도구를 사용합니다.
3. 대화의 전체 맥락을 항상 기억하고, 사용자가 모호하게 말하더라도 이전 대화를 참고하여 의도를 파악해야 합니다."""

# ** Streamlit의 세션 상태(Session State)를 이용한 대화 기록 유지 **
# st.session_state는 웹페이지가 새로고침 되어도 값을 유지해주는 마법 같은 딕셔너리입니다.
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# 이전 대화 기록을 화면에 표시
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            # AIMessage의 content가 복잡한 구조일 수 있으므로 텍스트만 추출
            content = message.content
            if isinstance(content, list) and content and isinstance(content[0], dict):
                st.markdown(content[0].get('text', ''))
            else:
                st.markdown(content)

# 사용자 입력을 받는 채팅 입력창
if prompt := st.chat_input("무엇이든 물어보세요."):
    # 사용자가 입력한 내용을 기록하고 화면에 표시
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 로딩 스피너 표시
    with st.spinner("생각 중..."):
        # 에이전트 실행
        inputs = {"messages": st.session_state.messages}
        final_state = app.invoke(inputs)
        
        # 실행 후의 전체 메시지 기록으로 세션 상태를 업데이트
        st.session_state.messages = final_state['messages']
        
        # 마지막 AI 응답만 가져와서 화면에 새로 표시
        ai_response_message = st.session_state.messages[-1]

        with st.chat_message("assistant"):
            content = ai_response_message.content
            if isinstance(content, list) and content and isinstance(content[0], dict):
                st.markdown(content[0].get('text', ''))
            else:
                st.markdown(content)