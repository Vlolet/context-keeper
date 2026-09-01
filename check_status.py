import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# 환경변수 로드
load_dotenv()

def test_model(model_name):
    print(f"--- Testing Model: {model_name} ---")
    try:
        # 가장 기초적인 설정으로 호출
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            max_retries=0 # 재시도 없이 한번만 찌름
        )
        
        start = time.time()
        response = llm.invoke("hi")
        end = time.time()
        
        print(f"✅ 성공! (Latency: {end - start:.2f}s)")
        print(f"답변: {response.content}\n")
        return True
        
    except Exception as e:
        print(f"❌ 실패! (에러 메시지 확인)")
        print(f"Error: {e}\n")
        return False

if __name__ == "__main__":
    print("API 상태 진단을 시작합니다...\n")
    
    # 1. 안정적인 1.5 Flash 모델 테스트
    is_15_alive = test_model("gemini-2.0-flash")
    
    # 2. 문제의 2.5 Flash 모델 테스트
    is_25_alive = test_model("gemini-2.5-flash")
    
    print("="*30)
    print("📢 진단 결과")
    if is_15_alive and is_25_alive:
        print("결론: API 키와 모델 모두 정상입니다. (코드 로직 문제)")
    elif is_15_alive and not is_25_alive:
        print("결론: 'gemini-2.5-flash' 모델만 쿼터가 잠겼습니다. (Cool-down 필요)")
        print("해결: 1시간 정도 기다리거나, 잠시 1.5 모델을 사용하세요.")
    elif not is_15_alive and not is_25_alive:
        print("결론: API Key 자체가 차단되었습니다.")
        print("해결: 구글 클라우드 콘솔에서 새 API Key를 발급받아야 합니다.")