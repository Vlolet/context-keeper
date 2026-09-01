# src/agent/memory.py
# 메모리 관리(요약, KG 등) 로직

import json
from pathlib import Path
import streamlit as st # 캐싱을 위해 streamlit import

# 지식 베이스 관리 로직
KNOWLEDGE_FILE = Path("data/knowledge.json")

@st.cache_data
def load_knowledge() -> dict:
    """knowledge.json 파일에서 장기기억 로드"""
    if KNOWLEDGE_FILE.exists():
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {} # 파일이 비어있거나 형식이 잘못된 경우
    return {}

def save_knowledge(knowledge: dict):
    """knowledge.json 파일에 장기 기억 저장"""
    KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=4)