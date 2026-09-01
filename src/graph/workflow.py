# 그래프 구성 및 컴파일
# LangGraph의 `StateGraph`를 정의하고, 노드들을 연결(엣지)하여 최종적으로 `app`을 컴파일하는 역할만 담당합니다.

import streamlit as st
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END

# --- 변경된 import ---
from .state import AgentState
from .nodes import (
    classify_task_node,
    execute_step_node,
    status_check_node,
    reflect_and_correct_node,
    update_plan_node,
    general_chat_node,
    prepare_tool_call_node,
    process_tool_result_node,
    execute_tools_node,
)

@st.cache_resource
def create_graph():
    """상태 머신(State Machine) 그래프를 생성하고 컴파일합니다."""
    
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("classify_task", classify_task_node)
    workflow.add_node("execute_step", execute_step_node)
    workflow.add_node("status_check", status_check_node)
    workflow.add_node("reflect_and_correct", reflect_and_correct_node)
    workflow.add_node("update_plan", update_plan_node)
    workflow.add_node("general_chat", general_chat_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("prepare_tool_call", prepare_tool_call_node)
    workflow.add_node("process_tool_result", process_tool_result_node)

    # 엣지(흐름) 정의
    workflow.set_entry_point("classify_task")
    
    def decide_next_node(state: AgentState):
        # 만약 '피드백 대기' 상태라면, classify_task의 판단과 상관없이 무조건 '반성'으로 보냅니다.
        if state.get("awaiting_feedback"):
            return "reflect_and_correct"
        return state["current_task"]
    
    workflow.add_conditional_edges(
        "classify_task",
        decide_next_node,
        {
            "execute_plan": "execute_step",
            "reflect_and_correct": "reflect_and_correct",
            "continue_discussion": "execute_step",
            "status_check": "status_check",
            "general_chat": "general_chat",
            "internal_document_search": "prepare_tool_call",
            # "start_new_project": "create_plan_node", # 추후 추가할 노드
            # "debugging": "call_debugging_subgraph", # 추후 추가할 노드
        }
    )
    
    workflow.add_edge("prepare_tool_call", "execute_tools")
    workflow.add_edge("execute_tools", "process_tool_result")
    
    # 도구 사용 후나 계획 실행 후에는 턴을 종료하여 사용자 피드백을 기다립니다.
    workflow.add_edge("process_tool_result", END)
    workflow.add_edge("execute_step", END)
    
    def decide_after_reflection(state: AgentState):
        state["awaiting_feedback"] = False
        if state["working_memory"] == "True":
            return "update_plan" # 단계가 끝났으면 계획 업데이트
        else:
            return END
    
    workflow.add_conditional_edges("reflect_and_correct", decide_after_reflection)
    
    workflow.add_edge("update_plan", END) # 한 턴이 끝나면 종료
    workflow.add_edge("status_check", END)
    workflow.add_edge("general_chat", END)

    # 그래프 컴파일
    app = workflow.compile()
    return app