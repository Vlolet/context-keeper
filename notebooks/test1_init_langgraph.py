import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# .env 파일에서 API 키 등 환경 변수 로드
load_dotenv()

# LangGraph의 상태 정의
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# LLM 모델 초기화
# stream=True로 설정하면 실시간으로 답변을 받아볼 수 있습니다.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, stream=True)

# 그래프의 노드(함수) 정의
def call_model(state: AgentState):
    messages = state['messages']
    response = model.invoke(messages)
    # 상태에 LLM의 응답을 추가
    return {"messages": [response]}

# LangGraph 워크플로우 정의
workflow = StateGraph(AgentState)
workflow.add_node("llm", call_model)
workflow.set_entry_point("llm")
workflow.add_edge("llm", END)

# 그래프 컴파일
app = workflow.compile()

# 실행 및 스트리밍 출력
print("### LangGraph 테스트 ###")
print("질문을 입력하세요 (종료하려면 'exit' 입력):")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # 그래프 실행
    inputs = {"messages": [("user", user_input)]}
    for output in app.stream(inputs):
        # 스트리밍 출력을 위해 마지막 메시지만 출력
        for key, value in output.items():
            if key == 'llm':
                print(f"AI: {value['messages'][-1].content}")