# 모든 노드 함수
# classify_task, execute_step 등 그래프를 구성하는 모든 '노드' 함수들을 이 파일에 모아 관리합니다.

import uuid
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolCall
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode as InternalToolNode

from .utils import parse_llm_response
from .state import AgentState
from src.tools.search import tools
from src import config

# 모델 정의
MODEL_NAME = "gemini-2.5-flash"
model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0, max_retries=0, google_api_key=config.GOOGLE_API_KEY)
model_with_tools = model.bind_tools(tools)

# 모델 정의 (로컬)
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
MODEL_NAME = "qwen-3.5-9-3072"
model = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", # 로컬이라 api key 아무거나 써도 됨
    model=MODEL_NAME,
    temperature=1,
    max_retries=0,
    # streaming=True,
    # callbacks=[StreamingStdOutCallbackHandler()]
)

# 참고용) ollama에서 qwen3.5:9b 받으면 기본 세팅
'''PARAMETER presence_penalty 1.5
PARAMETER temperature 1
PARAMETER top_k 20
PARAMETER top_p 0.95'''


# --- 2. 노드(Node) 함수 정의: 에이전트의 각 행동 단계 ---

class TaskClassification(BaseModel):
    """
    사용자의 메시지와 현재 프로젝트 상태를 기반으로, 수행해야 할 가장 적절한 작업 유형을 선택합니다.
    LLM이 이 클래스의 docstring과 각 필드의 description을 보고 판단의 근거로 삼습니다.
    """
    task: Literal["execute_plan", "continue_discussion", "general_chat", "internal_document_search", "start_new_project", "debugging"] = Field(
        description=(
            "- execute_plan: 사용자가 '다음 단계를 진행해', '계속해줘', '코드를 짜줘', '실행해' 처럼, 계획을 '능동적으로 진행'시키라는 명확한 '지시'나 '명령'을 내릴 때.\n"
            "- continue_discussion: 사용자가 현재 진행 중인 단계에 대해 추가 질문을 하거나, 더 자세한 설명을 요구하는 등 '현재 단계를 계속 논의'하고자 할 때.\n"
            "- status_check: 사용자가 '지금 뭐해?', '다음 단계는 뭐야?', '계획 보여줘' 처럼, 계획을 진행시키라는 의도 없이 단순히 현재 상황을 '확인'하거나 '질문'만 할 때.\n"
            "- general_chat: 프로젝트 계획과 직접적인 관련이 없는 일반적인 질문이나 대화일 때.\n"
            "- internal_document_search: 사용자가 프로젝트의 가이드라인, 핵심 철학, 아키텍처 원칙 등 '내부 문서'에 있을 법한 정보를 질문할 때.\n"
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

[중요 지침]
1. 위 상황을 종합적으로 고려하여, 사용자의 요청에 가장 적합한 작업 유형(`task`)과 요청 내용 요약(`user_request`)을 분류해주세요.
2. `user_request` 요약 시 반드시 한국어(Korean)로만 작성하세요. 
3. 다른 언어(중국어 등)를 섞거나, 동일한 단어(예: 생성, 생성...)를 무의미하게 반복하는 것을 절대 금지합니다.
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

def prepare_tool_call_node(state: AgentState):
    """
    라우터의 결정을 바탕으로 ToolNode가 이해할 수 있는 AIMessage를 생성합니다.
    """
    tool_name = state["current_task"]
    tool_query = state["working_memory"]
    
    # ToolCall 생성 시, 고유한 id를 반드시 포함해야 합니다.
    tool_call_id = str(uuid.uuid4())
    
    # ToolNode는 tool_calls 리스트를 포함하는 AIMessage를 기대합니다.
    tool_call_message = AIMessage(
        content="", # AI의 말이 아니라 도구 호출이므로 내용은 비워둡니다.
        tool_calls=[
            ToolCall(
                name=tool_name, 
                args={"query": tool_query}, # 우리 도구는 'query'라는 인자를 받습니다.
                id=tool_call_id
            )
        ]
    )
    
    # 이 메시지를 상태의 messages 리스트에 추가하여 ToolNode로 전달합니다.
    return {"messages": [tool_call_message]}

_tool_executor = InternalToolNode(tools)
def execute_tools_node(state: AgentState):
    """
    내부 ToolNode를 호출하고, 그 결과를 'messages'가 아닌 'working_memory'에 저장합니다.
    이것이 Wrapper Node 패턴입니다.
    """
    # ToolNode는 state 딕셔너리를 입력으로 받아, messages 리스트 안의
    # tool_calls를 찾아 자동으로 실행합니다.
    output_state = _tool_executor.invoke(state)
    
    # ToolNode는 결과를 'messages' 키에 담아 반환합니다.
    # 우리는 이 결과(ToolMessage 리스트)를 추출합니다.
    tool_messages = output_state["messages"]
    
    # 추출한 결과를 working_memory에 저장하여 반환합니다.
    return {"working_memory": tool_messages}

def process_tool_result_node(state: AgentState):
    """도구 실행 결과를 바탕으로 사용자에게 보여줄 최종 답변을 생성합니다."""
    # 도구 실행 결과는 항상 messages 리스트의 마지막에 ToolMessage 형태로 추가됩니다.
    tool_messages = state["working_memory"]
    
    # ToolMessage의 내용(content)을 추출합니다.
    tool_output = tool_messages[0].content
    
    # LLM에게 도구 결과를 요약하고 자연스러운 답변을 만들도록 요청합니다.
    # 사용자의 원래 질문을 함께 제공하여 더 맥락에 맞는 답변을 유도합니다.
    user_question = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    prompt = f"""당신은 AI 어시스턴트입니다. 사용자의 질문에 답하기 위해 방금 도구를 사용하여 아래와 같은 정보를 얻었습니다. 
이 정보를 바탕으로 사용자의 원래 질문에 대해 친절하고 자연스러운 답변을 생성해주세요. 원본 내용을 그대로 복사하지 말고, 핵심 내용을 요약하여 문장 형태로 설명해야 합니다.

# 사용자의 원래 질문
"{user_question}"

# 도구 실행 결과 (정보)
"{tool_output}"
"""
    
    response = model.invoke(prompt)
    return {"messages": [response]}

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
    user_request = state.get("working_memory", "")
    
    # 작업 실행을 위한 프롬프트
    execution_prompt = f"""당신은 'Context Keeper' 프로젝트를 관리하는 유능한 AI 프로젝트 매니저입니다.
당신의 임무는 주어진 계획의 현재 단계를 수행하고, 그 결과를 사용자에게 명확하게 보고한 후, 다음 행동을 제안하는 것입니다.

# 현재 프로젝트 정보
- 최종 목표: {goal}
- 전체 계획: {plan.get('steps')}
- 현재 단계: {current_step}
- 사용자의 최근 요청: "{user_request}"

# 당신의 임무
1.  위 정보를 바탕으로 '현재 단계'를 완수하기 위한 구체적인 결과물(설명, 코드 등)을 생성하세요.
2.  작업이 끝나면, 아래 <응답 형식>을 **반드시** 준수하여 사용자에게 보고하세요.

# <응답 형식>
1.  **작업 완료 보고:** "네, 요청하신 '{current_step}' 단계를 완료했습니다." 와 같이 성공적으로 작업을 마쳤음을 알립니다.
2.  **핵심 결과물 제시:** 생성한 설명이나 코드를 명확하게 제시합니다. 
    - **중요 규칙:** 모든 코드 스니펫은 반드시 ` ```bash ` 또는 ` ```python `과 같은 마크다운 코드 블록으로 감싸야 합니다. 리스트(`*`)나 헤더(`#`) 안에 코드를 넣지 마세요.
3.  **다음 단계 제안:** 현재 단계가 끝났으니, 다음 계획은 무엇인지 사용자에게 알려주고, 계속 진행할지 질문하며 답변을 유도합니다. (예: "다음 단계는 '디버깅 지원'입니다. 계속해서 진행할까요?")

---
이제, 위 지시에 따라 응답을 생성하세요.
"""
    
    response = model_with_tools.invoke(execution_prompt)
    
    return {"messages": [response], "awaiting_feedback": True, "working_memory": "step execution completed."}

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
    return {"working_memory": str(reflection.is_step_complete), "awaiting_feedback": False}

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
    clean_content = parse_llm_response(response)
    return {"messages": [AIMessage(content=clean_content)]}