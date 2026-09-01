# src/config.py
# 값 처리 및 인터페이스

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# --- API Keys ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Debugging & Logging
# (os.getenv는 문자열을 반환하므로, Bool로 변환)
VISUALIZE_GRAPH = os.getenv("VISUALIZE_GRAPH", "False").lower() in ("true", "1", "t")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- App Settings ---
APP_TITLE = os.getenv("APP_TITLE", "Defualt App Title")

# 값 추가 확인 로직 추가 가능
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY가 .env 파일에 설정되지 않았습니다.")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")