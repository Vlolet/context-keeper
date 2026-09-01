# src/ui/app.py

import streamlit as st
import os
os.environ["GRPC_DNS_RESOLVER"] = "native"
from langchain_core.messages import HumanMessage

# 새롭게 설계된 에이전트의 두뇌(그래프)와 기억(메모리)을 불러옵니다.
from src.graph.workflow import create_graph
from src.graph.state import AgentState
# from src.data.memory import load_knowledge, save_knowledge

st.markdown("""
<style>
/* Streamlit의 코드 블록 스타일을 재정의합니다 */
pre code {
    white-space: pre-wrap !important;       /* 공백은 유지하되, 필요시 자동 줄바꿈 */
    word-break: break-all !important;      /* 매우 긴 단어도 강제로 줄바꿈 */
}
</style>
""", unsafe_allow_html=True)

# --- content를 안전하게 파싱하는 헬퍼 함수 ---
def parse_message_content(content) -> str:
    """
    AIMessage.content의 다양한 형태(str, list of dicts, list of strs)를
    모두 안전하게 처리하여 단일 텍스트 문자열로 반환합니다.
    """
    text_content = ""
    if isinstance(content, str):
        text_content = content
    
    # content가 리스트인 경우
    if isinstance(content, list):
        text_parts = []
        for part in content:
            # 리스트의 항목이 딕셔너리인 경우
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            # 리스트의 항목이 그냥 문자열인 경우
            elif isinstance(part, str):
                text_parts.append(part)
        text_content = "".join(text_parts)
    
    # 그 외 예외적인 모든 경우, 일단 문자열로 변환하여 반환
    else:
        text_content = str(content)
    
    # 1. '\\n' (두 글자) -> '\n' (줄바꿈)
    # 2. '\\`' (두 글자) -> '`' (백틱)
    return text_content.replace('\\n', '\n').replace('\\`', '`')

# --- 1. 에이전트 및 세션 상태 초기화 ---

# 캐싱된 그래프를 로드합니다.
app = create_graph()

# Streamlit의 세션 상태를 초기화합니다.
if "state" not in st.session_state:
    st.session_state.state: AgentState = {
        "messages": [],
        "awaiting_feedback": False,
        "project_plan": {
            "goal": "사용자가 LangGraph와 Streamlit으로 AI 에이전트를 만들 수 있도록 돕기",
            "steps": ["기본 아키텍처 설명", "UI 코드 작성", "디버깅 지원"],
            "completed_steps": []
        },
        "current_task": "Idle",
        "working_memory": "",
    }

# --- 2. 사이드바 UI 렌더링 ---

def render_agent_status_in_sidebar(state: AgentState):
    """현재 에이전트의 상태를 사이드바에 시각화합니다."""
    st.sidebar.header("Agent's Brain 🧠")
    st.sidebar.subheader("Current Task")
    st.sidebar.info(f"**{state.get('current_task', 'Idle')}**")
    
    st.sidebar.subheader("Project Plan")
    plan = state.get('project_plan')
    if plan and plan.get("goal"):
        st.sidebar.write(f"**Goal:** {plan['goal']}")
        for step in plan.get('completed_steps', []):
            st.sidebar.markdown(f"- [x] ~~{step}~~")
            
        if plan.get('steps'):
            st.sidebar.markdown(f"- 👉 **{plan['steps'][0]}**")
            for step in plan.get('steps', [])[1:]:
                st.sidebar.markdown(f"- [ ] {step}")
        else:
            st.sidebar.success("All steps completed!")
    else:
        st.sidebar.caption("No active project.")

render_agent_status_in_sidebar(st.session_state.state)

# --- 3. 메인 채팅 UI ---

st.title("🧠 Context Keeper (v2)")

# 대화 기록 표시
for msg in st.session_state.state["messages"]:
    with st.chat_message(msg.type):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_name = msg.tool_calls[0]['name']
            st.markdown(f"```log\n🔍 {tool_name} 도구를 사용하여 정보를 찾음...\n```")
        else:
            st.markdown(parse_message_content(msg.content))

# 사용자 입력 처리
if prompt := st.chat_input("프로젝트에 대해 이야기하세요..."):
    # 사용자 메시지를 상태에 추가
    st.session_state.state["messages"].append(HumanMessage(content=prompt))
    
    # UI 즉시 업데이트
    st.rerun()

# 마지막 메시지가 사용자일 때만 에이전트 실행
if st.session_state.state["messages"] and st.session_state.state["messages"][-1].type == "human":
    with st.chat_message("ai"):
        with st.spinner("생각 중..."):
            # invoke 호출 직전 AgentState 디버깅용 코드
            print("\n" + "="*50)
            print("invoke() 호출 직전 AgentState:")
            print(f"  - Current Task: {st.session_state.state.get('current_task')}")
            print(f"  - Project Goal: {st.session_state.state['project_plan'].get('goal')}")
            print(f"  - Remaining Steps: {st.session_state.state['project_plan'].get('steps')}")
            print(f"  - Last User Message: {st.session_state.state['messages'][-1].content}")
            print("="*50 + "\n")
            
            # 에이전트 실행
            new_state = app.invoke(st.session_state.state)
            
            # 반환된 새로운 상태로 세션 상태를 업데이트
            st.session_state.state = new_state
            
            # UI 다시 그려서 결과 표시
            st.rerun()