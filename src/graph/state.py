# AgentState 정의

import operator
from typing import TypedDict, Annotated

# AgentState 정의를 별도 파일로 분리하여 관리합니다.
# 다른 모든 모듈이 이 파일을 참조하게 됩니다.

class ProjectPlan(TypedDict):
    goal: str
    steps: list[str]
    completed_steps: list[str]

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    awaiting_feedback: bool
    project_plan: ProjectPlan
    current_task: str
    working_memory: str