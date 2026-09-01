# check_graph.py 

import streamlit as st 
from src.graph.workflow import create_graph

print("에이전트 그래프를 생성하고 이미지 파일로 저장합니다...")

# 1. 그래프 생성 함수를 호출하여 컴파일된 app 객체를 가져옵니다.
#    (내부적으로 캐싱되지만, 이 스크립트에서는 한 번만 실행됩니다.)
app = create_graph()

# 2. 그래프 객체에서 PNG 이미지 데이터를 바이트 형태로 추출합니다.
image_bytes = app.get_graph().draw_mermaid_png()

# 3. 바이트 데이터를 'langgraph_v2_graph.png' 파일로 저장합니다.
file_path = "langgraph_v2_graph.png"
with open(file_path, "wb") as f:
    f.write(image_bytes)

print(f"'{file_path}' 파일이 성공적으로 생성되었습니다.")