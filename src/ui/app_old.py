# src/ui/app.py
# GUI 애플리케이션 실행 파일

import json
import operator
from typing import TypedDict, Annotated, List, Literal
from pathlib import Path

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from google.api_core import exceptions

from src import config

# --- 1. 설정 및 에이전트 로직 ---

MODEL_NAME = "gemini-2.5-flash"

search_tool = TavilySearch(max_results=3)
search_tool.name = "web_search" # 기본 도구 이름은 'tavily_search'
tools = [search_tool]

model_router = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, max_retries=0,)
model_agent = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.85, max_retries=0,)
model_with_tools = model_agent.bind_tools(tools)

class RouteQuery(BaseModel):
    """사용자 질문의 의도를 'general_chat'과 'search_required'로 분류합니다."""
    intent: Literal["general_chat", "search_required"] = Field(
        ...,
        description="인사, 잡담, 자기소개, 일반 상식은 'general_chat', 최신 정보나 구체적 사실 확인이 필요하면 'search_required'로 분류합니다."
    )

router_chain = model_router.with_structured_output(RouteQuery)

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    intent: str

def router_node(state: AgentState):
    last_msg = state["messages"][-1]
    
    # HumanMessage만 router_chain에 전달
    if isinstance(last_msg, HumanMessage):
        try:
            result = router_chain.invoke([last_msg])
            intent = result.intent
        except Exception:
            # 실패 시 안전하게 일반 대화로
            intent = "general_chat"
    else:
        # 시스템 메시지 등이 들어오면 무조건 일반 대화로 처리
        intent = "general_chat"
    
    return {"intent": intent}

# def general_chat_node(state: AgentState):
#     response = model_router.invoke(state["messages"])
#     return {"messages": [response]}

# def search_agent_node(state: AgentState):
#     response = model_with_tools.invoke(state['messages'])
#     return {"messages": [response]}

# 지식 베이스 관리 로직
KNOWLEDGE_FILE = Path("data/knowledge.json")

@st.cache_data
def load_knowledge() -> dict:
    """knowledge.json 파일에서 장기기억 로드"""
    if KNOWLEDGE_FILE.exists():
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_knowledge(knowledge: dict):
    """knowledge.json 파일에 장기 기억 저장"""
    KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True) # data 폴더가 없으면 생성
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)

AGENT_KNOWLEDGE = load_knowledge()

def general_chat_node(state: AgentState):
    # --- 이 함수 전체가 수정되었습니다 ---
    
    # 지식 베이스(JSON)를 LLM이 읽기 좋은 문자열로 변환
    knowledge_str = json.dumps(AGENT_KNOWLEDGE, ensure_ascii=False, indent=2)
    
    # 기본 시스템 프롬프트와 지식 베이스를 결합하여 최종 지침 생성
    final_prompt = f"{SYSTEM_PROMPT}\n\n<참고 정보 (사용자 및 PC)>\n{knowledge_str}"
    
    # 기존 대화 기록 맨 앞에 최종 지침을 담은 SystemMessage를 추가
    messages_with_prompt = [SystemMessage(content=final_prompt)] + state["messages"]
    
    # 수정된 메시지 리스트로 모델 호출
    response = model_router.invoke(messages_with_prompt)
    return {"messages": [response]}

def search_agent_node(state: AgentState):
    # --- 이 함수 전체가 수정되었습니다 ---

    knowledge_str = json.dumps(AGENT_KNOWLEDGE, ensure_ascii=False, indent=2)
    
    # 검색 노드에는 도구 사용에 대한 더 구체적인 규칙을 추가
    search_prompt = f"""{SYSTEM_PROMPT}

<도구 사용 규칙>
- 당신의 현재 임무는 'web_search' 도구를 사용하여 질문에 답하는 것입니다.
- 만약 도구 사용에 필수적인 정보(예: 날씨 검색을 위한 '도시 이름')가 명확하지 않다면, 도구를 사용하지 말고 사용자에게 해당 정보를 요청하는 질문을 하세요.

<참고 정보 (사용자 및 PC)>\n{knowledge_str}"""

    messages_with_prompt = [SystemMessage(content=search_prompt)] + state["messages"]
    
    response = model_with_tools.invoke(messages_with_prompt)
    return {"messages": [response]}

# def call_model(state: AgentState):
#     response = model_with_tools.invoke(state['messages'])
#     return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tool"
    return "__end__"

@st.cache_resource
def get_langgraph_app():
    print("LangGraph App 컴파일 중...")
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("general_chat", general_chat_node)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", lambda x: x['intent'], {
        "general_chat": "general_chat",
        "search_required": "search_agent",
    })
    workflow.add_edge("general_chat", END)
    workflow.add_conditional_edges("search_agent", should_continue, {"call_tool": "tools", "__end__": END})
    workflow.add_edge("tools", "search_agent")
    return workflow.compile()

app = get_langgraph_app()

@st.cache_resource
def draw_graph():
    from IPython.display import Image, display
    from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
    display(
        Image(
            app.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )
    
    graph_img_path = "langgraph_graph.png"
    image_bytes = app.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    try:
        with open(graph_img_path, "wb") as f: # 덮어쓰기 모드이므로 기존 파일이 있든 없든 괜찮음
            f.write(image_bytes)
        print(f"'{graph_img_path}' 그래프 이미지가 성공적으로 저장되었습니다.")
    except PermissionError:
        print(f"오류: '{graph_img_path}' 경로에 파일을 쓸 권한이 없습니다.")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
draw_graph()

# workflow = StateGraph(AgentState)
# workflow.add_node("llm", call_model)
# workflow.add_node("call_tool", tool_node)
# workflow.set_entry_point("llm")
# workflow.add_conditional_edges("llm", should_continue)
# workflow.add_edge("call_tool", "llm")
# app = workflow.compile()


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

# 사이드바에 체크박스를 추가합니다. (기본값은 해제 상태로 두어 API를 아낍니다)
enable_learning = st.sidebar.checkbox("🧠 지식 추출(학습) 활성화", value=False)
st.sidebar.divider()

# 앱 시작 시 장기 기억(knowledge)를 로드하여 세션 상태에 저장
if "knowledge" not in st.session_state:
    st.session_state.knowledge = load_knowledge()

# 사이드바에 현재 저장된 지식 표시
st.sidebar.markdown("### 장기 기억 (Knowledge Base)")
# st.sidebar.json(st.session_state.knowledge, expanded=False)
for category, details in st.session_state.knowledge.items():
    with st.sidebar.expander(f"▼ {category}"):
        # expander 내부에는 각 항목을 표 형식으로 깔끔하게 표시
        st.table(details)

# SYSTEM_PROMPT = """당신은 유능하고 적극적인 AI 비서 'Context Keeper'입니다. 당신의 임무는 다음과 같습니다:
# 1. 사용자의 질문에 최대한 정확하고 친절하게 답변합니다.
# 2. 모르는 정보나 최신 정보가 필요하다고 판단되면, 주저하지 말고 당신이 가진 'web_search' 도구를 사용합니다.
# 3. 대화의 전체 맥락을 항상 기억하고, 사용자가 모호하게 말하더라도 이전 대화를 참고하여 의도를 파악해야 합니다."""

SYSTEM_PROMPT = """당신은 정중하고 친절한 AI 에이전트 'Context Keeper'입니다.

<규칙>
- ChatGPT처럼 친구에게 하는 말투 혹은 과도한 아부는 절대 사용하지 않습니다.
- 모르거나 확실하지 않은 정보에 대해서는 반드시 'web_search' 도구를 사용해 정확성과 근거를 확보해야 합니다. 정책이나 방법에 대한 질문에는 공식 문서를 참조하고 링크를 제공해야 합니다.
- 해결책 제시에 필요한 정보가 부족할 경우, 즉시 해결책을 제시하지 말고 사용자에게 추가 정보를 요청해야 합니다.
- 결과물을 출력하기 전, 항상 내용을 검토하고 잘못된 점이 있다면 수정합니다.
- 사용자가 참고한 자료가 당신의 지식과 다를 경우, 두 정보를 비교하고 'web_search'를 통해 사실을 확인하여 정확한 정보와 근거를 제시해야 합니다.
"""

# ** Streamlit의 세션 상태(Session State)를 이용한 대화 기록 유지 **
# st.session_state는 웹페이지가 새로고침 되어도 값을 유지해주는 마법 같은 딕셔너리입니다.
if "messages" not in st.session_state:
    # st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.messages = [] # 처음에는 비워둠

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
        num_messages_before = len(st.session_state.messages)
        
        with st.spinner("답변 생성 중..."):
            try:
                final_state = app.invoke({"messages": st.session_state.messages})
                st.session_state.messages = final_state['messages']    
            except Exception as e:
                st.error(f"에이전트 실행 중 오류 발생: {e}")
                st.session_state.messages.append(AIMessage(content=f"오류: {e}"))
                
        if final_state:
            with st.status("에이전트의 생각 과정 보기", expanded=False) as status:
                # [핵심 수정 2] 이번 턴에서 '새로 생성된' 메시지만큼만 잘라서 확인합니다.
                newly_generated_messages = final_state['messages'][num_messages_before:]
                
                # 라우터의 결정은 final_state에서 직접 가져옵니다.
                intent = final_state.get('intent')
                st.write(f"🧭 **경로 결정:** {intent}")

                tool_used = False
                # [핵심 수정 3] 이제 '새로운' 메시지 리스트만 순회하여 정확한 로그를 만듭니다.
                for message in newly_generated_messages:
                    if isinstance(message, AIMessage) and message.tool_calls:
                        tool_call = message.tool_calls[0]
                        st.write(f"🔎 **도구 호출:** `{tool_call['name']}`")
                        st.json(tool_call['args'])
                        tool_used = True
                    if isinstance(message, ToolMessage):
                        st.write("✅ **도구 실행 완료**")
                        st.code(f"{message.content[:100]}...", language=None) # language=None으로 자동 하이라이팅 방지
                
                if not tool_used and intent == "search_required":
                    st.write("⚠️ **정보 부족:** 도구 사용에 필요한 정보가 없어 사용자에게 질문합니다.")

                status.update(label="도구 로그 보기", state="complete")

        # 5. 최종 AI 응답 메시지를 가져와 화면에 출력합니다.
        final_ai_message = st.session_state.messages[-1]
        response_text = get_content_from_message(final_ai_message)
        st.write(response_text)
                
    # except exceptions.ServiceUnavailable as e:
    #     st.error("모델 서버가 일시적으로 응답하지 않습니다. 잠시 후 다시 시도해주세요.")
    # except Exception as e:
    #     st.error(f"예상치 못한 오류가 발생했습니다: {e}")
    
    # 지능형 기억 로직
    if enable_learning:
        with st.sidebar:
            with st.spinner("대화를 복기하며 기억할 내용을 선별하는 중..."):
                try:
                    # 대화 기록을 텍스트로 변환
                    dialogue_text = ""
                    for msg in st.session_state.messages[-4:]: # 최근 4개 메시지만 복기
                        role = 'Human' if isinstance(msg, HumanMessage) else "AI"
                        content = msg.content
                        if content: dialogue_text += f"{role}: {content}\n"
                        
                    # 지식 추출 프롬프트
                    knowledge_extraction_prompt = f"""
                    당신은 대화에서 핵심 정보를 추출하는 분석가입니다.
                    아래 대화 내용에서 사용자의 이름, 목표, 선호도, 특정 프로젝트 이름 등
                    '미래의 대화에 도움이 될 만한 구체적인 사실'이 있다면 JSON 형식으로 추출해주세요.
                    추출할 정보가 없다면 빈 JSON 객체(`{{}}`)를 반환하세요.
                    
                    추출 규칙:
                    - key는 정보의 종류 (예: "user_name", "project_name", "user_goal")
                    - value는 정보의 내용
                    
                    [대화 내용]
                    {dialogue_text}
                    
                    [추출된 정보 (JSON)]
                    """
                    
                    extractor_model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
                    response = extractor_model.invoke(knowledge_extraction_prompt)
                    
                    # LLM의 응답에서 JSON만 추출
                    extracted_json_str = response.content.strip().replace("```json", "").replace("```", "")
                    if extracted_json_str:
                        new_knowledge = json.loads(extracted_json_str)
                        
                        if new_knowledge: # 추출된 정보가 있을 경우에만
                            st.sidebar.success("✨ 새로운 지식을 학습했습니다!")
                            st.sidebar.write("학습 내용:")
                            st.sidebar.json(new_knowledge)
                            
                            # 기존 지식에 새로운 지식 업데이트
                            st.session_state.knowledge.update(new_knowledge)
                            save_knowledge(st.session_state.knowledge)
                
                except Exception as e:
                    st.sidebar.error(f"지식 추출 중 오류 발생: {e}")
