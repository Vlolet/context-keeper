import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GRPC_DNS_RESOLVER"] = "native"

print("네트워크 테스트를 시작합니다...")

try:
    # 1. .env 파일에서 API 키를 로드합니다.
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("'.env' 파일에서 GOOGLE_API_KEY를 찾을 수 없습니다.")
    print("API 키를 성공적으로 로드했습니다.")

    # 2. 임베딩 클라이언트 초기화를 시도합니다.
    #    이 과정에서 Google 서버와의 첫 연결이 발생합니다.
    print("Google 임베딩 서비스에 연결을 시도합니다...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
    )
    print("✅ 성공! Google 서버와 성공적으로 통신했습니다.")

    # 3. 간단한 임베딩 테스트를 수행합니다.
    print("간단한 텍스트 임베딩을 요청합니다...")
    vector = embeddings.embed_query("Hello, world!")
    print(f"✅ 성공! 벡터를 수신했습니다. (차원: {len(vector)})")
    print("\n🎉 모든 네트워크 테스트를 통과했습니다.")

except Exception as e:
    print("\n❌ 테스트 실패.")
    print("오류 유형:", type(e).__name__)
    print("오류 메시지:", e)