from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import os
from typing import List, Optional
import json
import sys
import io

# Принудительно устанавливаем UTF-8 кодировку для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
# Инициализация FastAPI приложения
app = FastAPI(
    title="Credit Scoring API",
    description="REST API для предсказания вероятности дефолта клиента",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Загрузка модели
MODEL_PATH = "models/best_model"
model_predictor = None

class CreditData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    default_probability: float
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str

def load_model():
    """Загрузка модели при старте приложения"""
    global model_predictor
    try:
        from src.models.predict import ModelPredictor
        model_predictor = ModelPredictor(MODEL_PATH)
        if model_predictor.model is not None:
            print("✅ Модель успешно загружена")
        else:
            print("⚠️ Модель не загружена - режим демонстрации")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        model_predictor = None

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте приложения"""
    load_model()

@app.get("/", summary="Корневой endpoint", tags=["General"])
async def root():
    """Корневой endpoint API"""
    return {
        "message": "Credit Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["General"])
async def health_check():
    """Проверка здоровья API и загрузки модели"""
    model_loaded = model_predictor is not None and model_predictor.model is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        message="API работает" if model_loaded else "API в режиме демонстрации (модель не загружена)"
    )

@app.post("/predict", response_model=PredictionResponse, summary="Предсказание дефолта", tags=["Prediction"])
async def predict(data: CreditData):
    """
    Предсказание вероятности дефолта клиента
    
    - **LIMIT_BAL**: Кредитный лимит
    - **SEX**: Пол (1=мужчина, 2=женщина)  
    - **EDUCATION**: Образование (1=аспирант, 2=университет, 3=школа, 4=другое)
    - **MARRIAGE**: Семейное положение (1=женат/замужем, 2=холост/не замужем, 3=другое)
    - **AGE**: Возраст
    - **PAY_X**: История платежей (от -2 до 8)
    - **BILL_AMTX**: Суммы счетов
    - **PAY_AMTX**: Суммы платежей
    """
    
    if model_predictor is None or model_predictor.model is None:
        # Демонстрационный режим если модель не загружена
        return PredictionResponse(
            prediction=0,
            probability=0.1,
            default_probability=0.1,
            status="demo_mode"
        )
    
    try:
        input_dict = data.dict()
        result = model_predictor.predict(input_dict)
        
        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            default_probability=result['default_probability'],
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Ошибка предсказания: {str(e)}")

@app.post("/batch_predict", summary="Пакетное предсказание", tags=["Prediction"])
async def batch_predict(data: List[CreditData]):
    """Пакетное предсказание для нескольких клиентов"""
    
    if model_predictor is None or model_predictor.model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        input_list = [item.dict() for item in data]
        results = model_predictor.batch_predict(input_list)
        
        return {
            "predictions": results,
            "count": len(results),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка пакетного предсказания: {str(e)}")

@app.get("/model_info", summary="Информация о модели", tags=["Model"])
async def model_info():
    """Получение информации о загруженной модели"""
    try:
        if os.path.exists('reports/best_model_info.json'):
            with open('reports/best_model_info.json', 'r') as f:
                model_info = json.load(f)
            return {
                "status": "success",
                "model_info": model_info
            }
        else:
            return {
                "status": "no_info",
                "message": "Информация о модели не найдена"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения информации: {str(e)}")

# Пример данных для тестирования
@app.get("/sample_data", summary="Пример данных для тестирования", tags=["Testing"])
async def sample_data():
    """Возвращает пример данных для тестирования API"""
    sample = {
        "LIMIT_BAL": 50000,
        "SEX": 1,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 1000,
        "BILL_AMT2": 1000,
        "BILL_AMT3": 1000,
        "BILL_AMT4": 1000,
        "BILL_AMT5": 1000,
        "BILL_AMT6": 1000,
        "PAY_AMT1": 1000,
        "PAY_AMT2": 1000,
        "PAY_AMT3": 1000,
        "PAY_AMT4": 1000,
        "PAY_AMT5": 1000,
        "PAY_AMT6": 1000
    }
    return {"sample_data": sample}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)