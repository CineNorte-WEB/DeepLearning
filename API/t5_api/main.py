from fastapi import FastAPI
from pydantic import BaseModel
from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch

app = FastAPI()

# 모델 및 토크나이저 로드
model_dir = "./model"
tokenizer = T5TokenizerFast.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# GPU 또는 CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class ReviewInput(BaseModel):
    storename: str
    review: str

@app.post("/generate")
async def generate_keywords_and_sentiments(input_data: ReviewInput):
    """
    Storename과 Review를 입력받아 긍정/부정 키워드를 반환합니다.
    """
    storename = input_data.storename
    review = input_data.review

    # Prompt 추가
    input_text = (
        "You are an assistant responsible for extracting meaningful keywords from reviews and assigning a sentiment label (positive/negative) to each keyword. "
        "The meaningful keywords should be related to aspects such as taste, service, ambiance, cleanliness, and cost-effectiveness. "
        "You should extract 3-5 key phrases from the review, but additional phrases can be included if necessary. "
        "The output format should be as follows: keyword1(sentiment1), keyword2(sentiment2), ... "
        "Review: " + review
    )

    # 입력 텍스트를 토큰화
    inputs = tokenizer(
        input_text,
        max_length=350,  # 재학습 설정에 맞춰 최대 토큰 길이를 350으로 변경
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=128,
            num_beams=5,
            early_stopping=True
        )

    keywords = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 디버깅 출력
    print(f"Generated Keywords: {keywords}")

    # 키워드 분류
    positive_keywords, negative_keywords = classify_keywords(keywords)

    # 디버깅 출력
    print(f"Positive Keywords: {positive_keywords}")
    print(f"Negative Keywords: {negative_keywords}")

    return {
        "StoreName": storename,
        "Positive_Keywords": positive_keywords,
        "Negative_Keywords": negative_keywords
    }

def classify_keywords(keywords):
    """
    Keywords 필드에서 긍정적/부정적 키워드를 분리합니다.
    """
    positive_keywords = []
    negative_keywords = []

    if keywords:  # None 또는 빈 문자열 방지
        # 키워드와 Sentiment를 분리
        pairs = keywords.split(", ")  # 쉼표와 공백 기준으로 분리
        for pair in pairs:
            if "(Positive)" in pair:
                positive_keywords.append(pair.replace("(Positive)", "").strip())
            elif "(Negative)" in pair:
                negative_keywords.append(pair.replace("(Negative)", "").strip())

    return positive_keywords, negative_keywords