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
from langchain_tavily import TavilySearch
from google.api_core import exceptions


# --- 1. 설정 및 에이전트 로직 ---

load_dotenv()
MODEL_NAME = "gemini-2.5-flash"

search_tool = TavilySearch(max_results=3)
search_tool.name = "web_search" # 기본 도구 이름은 'tavily_search'
tools = [search_tool]

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.85)
model_with_tools = model.bind_tools(tools)

def call_model(state: AgentState):
    response = model_with_tools.invoke(state['messages'])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tool"
    return "__end__"

workflow = StateGraph(AgentState)
workflow.add_node("llm", call_model)
workflow.add_node("call_tool", tool_node)
workflow.set_entry_point("llm")
workflow.add_conditional_edges("llm", should_continue)
workflow.add_edge("call_tool", "llm")
app = workflow.compile()


# --- 2. LangGraph 스트림을 소비하고, 텍스트 청크만 변환하는 함수

def get_content_from_message(message: BaseMessage) -> str:
    """모든 종류의 메시지 객체에서 안전하게 텍스트 내용만 추출합니다."""
    if not isinstance(message, AIMessage):
        return message.content
    
    content = message.content
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return content[0].get('text', '')
    return str(content) # 문자열이거나 예외 상황 처리

def run_agent(user_input: list):
    inputs = {"messages": user_input}
    
    # app.stream()은 복잡한 이벤트 딕셔너리를 생성합니다.
    for event in app.stream(inputs, stream_mode="values"):
        # 각 이벤트에서 'messages' 키의 값을 가져옵니다.
        message_chunk_list = event.get("messages", [])
        if message_chunk_list:
            # messages는 항상 리스트이므로 마지막 항목을 확인합니다.
            last_message_chunk = message_chunk_list[-1]
            if isinstance(last_message_chunk, AIMessage):
                # AIMessage 청크의 content만 st.write_stream으로 보냅니다.
                yield last_message_chunk.content


# --- 3. Streamlit UI 구현 ---

st.set_page_config(page_title="Context Keeper", page_icon="🧠")
st.title("🧠 Context Keeper")
st.sidebar.title("Agent Status")
st.sidebar.markdown("에이전트의 생각 과정이나 도구 사용 내역이 여기에 표시됩니다.")

SYSTEM_PROMPT = """당신은 유능하고 적극적인 AI 비서 'Context Keeper'입니다. 당신의 임무는 다음과 같습니다:
1. 사용자의 질문에 최대한 정확하고 친절하게 답변합니다.
2. 모르는 정보나 최신 정보가 필요하다고 판단되면, 주저하지 말고 당신이 가진 'web_search' 도구를 사용합니다.
3. 대화의 전체 맥락을 항상 기억하고, 사용자가 모호하게 말하더라도 이전 대화를 참고하여 의도를 파악해야 합니다."""

# ** Streamlit의 세션 상태(Session State)를 이용한 대화 기록 유지 **
# st.session_state는 웹페이지가 새로고침 되어도 값을 유지해주는 마법 같은 딕셔너리입니다.
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# 이전 대화 기록 표시 함수
def display_messages():
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
                    
display_messages()

# 사용자 입력을 받는 채팅 입력창
if prompt := st.chat_input("무엇이든 물어보세요."):
    # 사용자가 입력한 내용을 기록하고 화면에 표시
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("생각 중..."):
                final_state = app.invoke({"messages": st.session_state.messages})
            final_ai_message = final_state['messages'][-1]
            
            # 행동 분기
            # Case A: 만약 첫 행동이 '도구 호출'이라면
            if final_ai_message.tool_calls:
                tool_call = final_ai_message.tool_calls[0]
                st.sidebar.info(f"{tool_call['name']} 호출\n- 검색어: {tool_call['args']['query']}")
                
                with st.spinner("웹 검색 결과를 바탕으로 답변을 생성 중..."):
                    response_stream = model.stream(final_state['messages'])
                    full_response = st.write_stream(
                        (chunk.content for chunk in response_stream if isinstance(chunk, AIMessage))
                    )
            
            # Case B: 도구 사용x
            else:
                # 가짜 스트리밍 효과
                full_response = st.write_stream(
                    (char for char in final_ai_message.content)
                )
            
            st.session_state.messages = final_state['messages']
            
        except exceptions.ServiceUnavailable as e:
            st.error("모델 서버가 일시적으로 응답하지 않습니다. 잠시 후 다시 시도해주세요.")
        except Exception as e:
            st.error(f"예상치 못한 오류가 발생했습니다: {e}")