# DeepLearing Module

## 프로젝트 개요
딥러닝 파트는 [플랫폼 크롤링 리뷰]와 [사용자 리뷰]를 바탕으로, 가게에 대한 **긍/부정 분류 및 키워드 추출**과 **키워드 기반 카테고리별 문장 생성** 기능을 제공합니다. 

- **Model 1** : 리뷰에서 주요 키워드 추출 및 긍/부정 감성 분석
- **Model 2** : 키워드를 카테고리(맛, 서비스, 분위기, 위생, 가성비)로 분류하고, 대표 문장 생성

---

## 1. 주요 기능

### **Model 1 : 키워드 추출 및 감성 분석**
- 리뷰 텍스트를 입력받아 **키워드 추출** 및 **긍/부정 감성 분석** 수행
- ![image](https://github.com/user-attachments/assets/c72e27ce-e62c-4617-a5d1-6d4bec51b8d3)

- **학습 모델** : `pko-t5-base` (Hugging Face)
- **결과** :
  - 키워드와 감성 라벨링 (`Positive`/`Negative`)
  - `StoreName`, `Review`, `Keywords_Sentiments` 형식으로 반환
 
### **Model 2: 키워드 분류 및 문장 생성**
- 추출된 키워드를 **맛, 서비스, 위생, 분위기, 가성비**로 분류
- 각 카테고리에 대한 **대표 문장 생성**
- **학습 모델** : OpenAI Fine-Tuned GPT 모델
- **결과**:
  - `StoreName`, `Category`, `Group Keywords`, `Representative Sentence` 형식으로 반환

---
## 2. 설치 및 실행

### **2-1 레포지토리 클론**
```bash
git clone https://github.com/your-username/your-repository.git
cd DeepLearning
```
### **2-2 가상환경 설정 및 종속성 설치**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **2-3. Docker 이미지 빌드 및 실행**

#### **Model 1**
```bash
cd API/t5_api
docker build -t fastapi-t5-api:v2 .
docker run -d -p 8000:8000 fastapi-t5-api:v2
```
#### **Model 2**
```bash
cd API/camslien_2nd_model
docker build -t camslien-2nd-model .
docker run -d -p 8000:8000 camslien-2nd-model
```


---
## **3. API 사용법**

#### **Model 1: 키워드 추출 및 감성 분석**

**Endpoint**
- URL: `POST /generate`
- Input:
  ```
  {
  "storename": "Sample Store",
  "review": "대표메뉴는 소고기보신탕. 굴국밥도 괜찮고 여름철에 콩국수가 별미."
  }
  ```
- Output:
  ```
  {
  "StoreName": "Sample Store",
  "Positive_Keywords": ["소고기보신탕", "굴국밥", "콩국수"],
  "Negative_Keywords": []
  }
  ```
  
#### **Model 2: 키워드 분류 및 문장 생성**

**Endpoint**
- URL: `POST /analyze`
- Input:
  ```
  {
  "store_name": "153 스트리트",
  "keywords": ["패티 두께", "패티 추가", "두툼한 수제패티", "브리오슈번"]
  }
  ```
- Output:
  ```
  {
  "StoreName": "153 스트리트",
  "Results": [
    {
      "Category": "맛",
      "Group Keywords": "패티 두께, 두툼한 수제패티, 패티 두툼하고 육즙 좋음",
      "Representative Sentence": "두툼하고 육즙이 가득한 수제버거와 조화로운 소스 맛이 일품입니다."
    }
  ]
  }
  ```
---

## **4. 디렉토리 구조**

```
DeepLearning/
├── API/
│   ├── t5_api/                     # Model 1 코드
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── model/
│   │   └── requirements.txt
│   ├── camslien_2nd_model/         # Model 2 코드
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── .env
├── README.md
```
