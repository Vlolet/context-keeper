# src/agent/graph.py
# LangGraph 워크플로우 정의
# 다른 파일들로 코드 분할하여 리팩토링함. 사용하지 않는 파일

import json
import operator
from typing import TypedDict, Annotated, Literal

import streamlit as st
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# 다른 모듈에서 필요한 전문가들을 불러옵니다.
from ..tools.search import tools
from ..data.memory import load_knowledge
from src import config

# --- 1. 모델 및 상태 정의 ---

# 모델 정의
MODEL_NAME = "gemini-2.5-flash"
model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, max_retries=0)
model_with_tools = model.bind_tools(tools)

# 우리가 설계한 새로운 AgentState
class ProjectPlan(TypedDict):
    goal: str
    steps: list[str]
    completed_steps: list[str]

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    project_plan: ProjectPlan
    current_task: str
    working_memory: str

# --- 2. 노드(Node) 함수 정의: 에이전트의 각 행동 단계 ---

class TaskClassification(BaseModel):
    """
    사용자의 메시지와 현재 프로젝트 상태를 기반으로, 수행해야 할 가장 적절한 작업 유형을 선택합니다.
    LLM이 이 클래스의 docstring과 각 필드의 description을 보고 판단의 근거로 삼습니다.
    """
    task: Literal["execute_plan", "continue_discussion", "general_chat", "start_new_project", "debugging"] = Field(
        description=(
            "- execute_plan: 사용자가 '다음 단계를 진행해', '계속해줘', '코드를 짜줘', '실행해' 처럼, 계획을 '능동적으로 진행'시키라는 명확한 '지시'나 '명령'을 내릴 때.\n"
            "- continue_discussion: 사용자가 현재 진행 중인 단계에 대해 추가 질문을 하거나, 더 자세한 설명을 요구하는 등 '현재 단계를 계속 논의'하고자 할 때.\n"
            "- status_check: 사용자가 '지금 뭐해?', '다음 단계는 뭐야?', '계획 보여줘' 처럼, 계획을 진행시키라는 의도 없이 단순히 현재 상황을 '확인'하거나 '질문'만 할 때.\n"
            "- general_chat: 프로젝트 계획과 직접적인 관련이 없는 일반적인 질문이나 대화일 때.\n"
            "- start_new_project: 사용자가 명시적으로 새로운 프로젝트를 시작하고 싶다고 말할 때.\n"
            "- debugging: 사용자가 코드 에러나 기술적인 문제 해결을 요청할 때."
        )
    )
    user_request: str = Field(
        description="분류된 작업에 필요한 사용자의 핵심 요청사항을 간결하게 요약합니다. general_chat의 경우, 원래 메시지를 그대로 전달해도 좋습니다."
    )

def classify_task_node(state: AgentState):
    """사용자 입력과 현재 상태를 보고 어떤 작업을 해야할지 결정하는 '슈퍼 라우터'"""
    user_message = state["messages"][-1].content
    plan = state["project_plan"]
    
    # 라우팅을 위한 프롬프트
    prompt = f"""당신은 사용자의 요청을 분석하여 다음에 수행할 작업을 결정하는 AI 라우터입니다.
    
# 현재 프로젝트 상황
- 최종 목표: {plan.get('goal', '아직 설정되지 않음')}
- 다음 단계: {plan.get('steps', ['계획 없음'])[0]}

# 사용자의 최근 메시지
"{user_message}"

위 상황을 종합적으로 고려하여, 사용자의 요청에 가장 적합한 작업 유형(`task`)과 요청 내용 요약(`user_request`)을 분류해주세요.
"""
    
    # 구조화된 출력을 사용해 LLM이 명확한 결정을 내리도록 함
    structured_router = model.with_structured_output(TaskClassification)
    classification_result = structured_router.invoke(prompt)
    
    # 디버깅을 위해 터미널에 결과 출력
    print("\n--- Router Decision ---")
    print(f"Task: {classification_result.task}")
    print(f"Request: {classification_result.user_request}")
    print("-----------------------\n")
    
    # 결정된 작업을 AgentState에 업데이트하여 다음 노드로 전달
    return {
        "current_task": classification_result.task,
        "working_memory": classification_result.user_request
    }


def execute_step_node(state: AgentState):
    """계획에 따라 실제 작업을 수행하는 노드"""
    print(f"\n[Node Entered] execute_step_node")
    plan = state["project_plan"]
    if not plan or not plan.get("steps"):
        # 계획이 없으면 일반 대화로 처리
        system_prompt = "You are a helpful assistant. Respond to the user's message."
        messages_with_prompt = [SystemMessage(content=system_prompt)] + state["messages"]
        response = model.invoke(messages_with_prompt)
        return {"messages": [response]}

    current_step = plan["steps"][0]
    goal = plan["goal"]
    
    # 작업 실행을 위한 프롬프트
    execution_prompt = f"""
    Current Goal: {goal}
    Current Step: {current_step}
    
    Based on the current conversation, execute this step. Use tools if necessary.
    """
    messages_with_prompt = [SystemMessage(content=execution_prompt)] + state["messages"]
    
    response = model_with_tools.invoke(messages_with_prompt)
    
    # 결과물을 working_memory에 저장하여 다음 노드(reflect)로 전달
    return {"messages": [response], "working_memory": response.content}

def status_check_node(state: AgentState):
    """
    프로젝트의 현재 상태를 읽고, 사용자에게 자연스러운 문장으로 보고하는 노드.
    이 노드는 절대로 프로젝트 계획을 수정하지 않습니다.
    """
    plan = state["project_plan"]
    
    if not plan or not plan.get("goal"):
        response_text = "현재 진행 중인 프로젝트가 없습니다."
    else:
        goal = plan['goal']
        completed = plan['completed_steps']
        next_step = plan['steps'][0] if plan['steps'] else "모든 계획이 완료되었습니다."
        
        response_text = (
            f"네, 현재 프로젝트의 상태를 알려드릴게요.\n\n"
            f"**최종 목표:** {goal}\n"
            f"**다음 단계:** {next_step}\n"
            f"**완료된 단계:** {', '.join(completed) if completed else '아직 없습니다.'}"
        )
    
    # 이 노드는 LLM을 호출할 필요 없이, 상태만으로 답변을 생성할 수 있습니다.
    # 만약 더 자연스러운 문장을 원한다면 이 response_text를 model.invoke()에 넘겨도 됩니다.
    return {"messages": [AIMessage(content=response_text)]}

class Reflection(BaseModel):
    """대화와 작업 결과물을 검토하여, 현재 계획 단계를 완료할지 여부를 결정합니다."""
    is_step_complete: bool = Field(description="사용자와의 대화와 AI의 답변을 종합했을 때, 현재 계획 단계의 목표가 완전히 달성되었으면 True, 아직 논의가 더 필요하면 False.")
    reasoning: str = Field(description="왜 그렇게 판단했는지에 대한 간결한 근거.")

def reflect_and_correct_node(state: AgentState):
    """결과물을 스스로 검토하고, 현재 단계를 계속할지 아니면 끝낼지를 결정합니다."""
    print(f"\n[Node Entered] reflect_and_correct_node")
    
    # 이 노드는 LLM을 사용하여 '반성'을 수행합니다.
    reflection_prompt = f"""
# Context
- Project Goal: {state['project_plan']['goal']}
- Current Step: {state['project_plan']['steps'][0]}
- Last few messages: {state['messages'][-3:]}

# Task
Analyze the conversation. Is the goal of the current step fully achieved and ready to be marked as complete?
Or does the user still have questions or require more work on this step?
"""
    
    reflection_model = model.with_structured_output(Reflection)
    reflection = reflection_model.invoke(reflection_prompt)
    
    print(f"--- Reflection ---")
    print(f"Is step complete? {reflection.is_step_complete}")
    print(f"Reasoning: {reflection.reasoning}")
    print(f"--------------------")
    
    # 결정된 결과를 working_memory에 저장하여 조건부 분기에 사용
    return {"working_memory": str(reflection.is_step_complete)}

def update_plan_node(state: AgentState):
    """한 단계가 끝나면 계획을 업데이트하는 노드"""
    print(f"\n[Node Entered] update_plan_node")
    plan = state["project_plan"].copy()
    if plan and plan.get("steps"):
        completed_step = plan["steps"].pop(0)
        plan["completed_steps"].append(completed_step)
    return {"project_plan": plan}

def general_chat_node(state: AgentState):
    """프로젝트와 관련 없는 일반 대화를 처리하는 노드"""
    # 에이전트의 기본 페르소나를 알려주는 간단한 시스템 프롬프트를 추가하고,
    # 지금까지의 '진짜' 대화 기록 전체를 전달하여 자연스러운 답변을 유도합니다.
    
    system_prompt = "당신은 'Context Keeper'라는 이름을 가진 친절하고 도움이 되는 AI 어시스턴트입니다."
    
    messages_for_chat = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = model.invoke(messages_for_chat)
    return {"messages": [response]}

# --- 3. 그래프 생성 함수 ---

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

    # 엣지(흐름) 정의
    workflow.set_entry_point("classify_task")
    
    def decide_next_node(state: AgentState):
        return state["current_task"]
    
    workflow.add_conditional_edges(
        "classify_task",
        decide_next_node,
        {
            "execute_plan": "execute_step",
            "continue_discussion": "execute_step",
            "status_check": "status_check",
            "general_chat": "general_chat",
            # "start_new_project": "create_plan_node", # 추후 추가할 노드
            # "debugging": "call_debugging_subgraph", # 추후 추가할 노드
        }
    )
    
    workflow.add_edge("execute_step", "reflect_and_correct")
    
    def decide_after_reflection(state: AgentState):
        if state["working_memory"] == "True":
            return "update_plan" # 단계가 끝났으면 계획 업데이트
        else:
            return END # 아직 안 끝났으면, 대화를 계속하기 위해 턴을 종료
    
    workflow.add_conditional_edges("reflect_and_correct", decide_after_reflection)
    
    workflow.add_edge("update_plan", END) # 한 턴이 끝나면 종료
    workflow.add_edge("status_check", END)
    workflow.add_edge("general_chat", END)

    # 그래프 컴파일
    app = workflow.compile()
    return app