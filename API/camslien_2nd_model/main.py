from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

# 환경변수 로드
load_dotenv()

# OpenAI API Key 및 모델 ID 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model_id = os.getenv("MODEL_ID")  # 모델 ID 가져오기

# FastAPI 앱 초기화
app = FastAPI()

#Input Data model 정의
class KeywordsInput(BaseModel):
    store_name: str #가게명
    keywords: list # 키워드 리스트

def parse_model_output(raw_output):
    """
    OpenAI 모델의 출력을 파싱하여 필요한 정보를 추출하는 함수
    """
    try:
        # JSON 문자열을 파이썬 객체로 변환
        parsed_output = json.loads(raw_output)

        # 결과 저장용 리스트
        parsed_results = []

        # 배열 형식일 경우 모든 항목 순회
        if isinstance(parsed_output, list):
            for category_data in parsed_output:
                category = category_data.get("category", "N/A")
                group_keywords = category_data.get("group_keywords", "N/A")
                representative_sentence = category_data.get("representative_sentence", "N/A")
                parsed_results.append({
                    "Category": category,
                    "Group Keywords": group_keywords,
                    "Representative Sentence": representative_sentence
                })
            return parsed_results
        else:
            raise ValueError("Expected a list in the model output")

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse model output: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_keywords(input_data: KeywordsInput):
    """
    OpenAI Fine-tuned 모델 API
    - Input: StoreName, Keywords
    - Output: Category, Group Keywords, Representative Sentence
    """
    store_name = input_data.store_name
    keywords = input_data.keywords

    # 키워드를 문자열로 변환
    keywords_str = ", ".join(keywords)

    try:
        # OpenAI Fine-tuned 모델 호출
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "Analyze restaurant review keywords and classify them into appropriate categories among the five categories (맛, 서비스, 위생, 가성비, 분위기) based on their meanings and characteristics. "
                               "For each category, generate a natural Korean sentence that summarizes the key points while maintaining the original sentiment and including specific details from the keywords."
                },
                {"role": "user", "content": f"Keywords: [{keywords_str}]"}
            ]
        )
        # 모델의 응답 처리
        raw_output = response.choices[0].message.content

        # 디버깅 출력
        print(f"Outputs: {raw_output}")

        # 모든 카테고리 데이터를 파싱
        parsed_results = parse_model_output(raw_output)

        # 결과 반환
        return {
            "StoreName": store_name,
            "Results": parsed_results
        }

    except Exception as e:  # 변경: OpenAIError 대신 Exception 사용
        raise HTTPException(status_code=500, detail=f"OpenAI model failed: {str(e)}")

