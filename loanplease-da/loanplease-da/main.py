from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 앱과 모델 초기화
app = FastAPI(title="Iris Classifier API")

# 모델 로드
model = joblib.load("model.pkl")

# 입력 데이터 모델
class Item(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 예측을 위한 경로
@app.post("/predict")
async def predict(item: Item):
    data = [[item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]]
    prediction = model.predict(data)
    probability = model.predict_proba(data).max()
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability)
    }