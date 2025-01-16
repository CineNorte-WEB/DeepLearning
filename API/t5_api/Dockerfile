# 베이스 이미지
FROM python:3.8-slim

# 작업 디렉터리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt .
COPY main.py .
COPY model ./model

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
