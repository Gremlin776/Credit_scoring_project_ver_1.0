import pytest
import requests
import json
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

class TestAPI:
    """Тесты для API endpoints"""
    
    def test_root_endpoint(self):
        """Тест корневого endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_check(self):
        """Тест health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_sample_data_endpoint(self):
        """Тест endpoint с примером данных"""
        response = client.get("/sample_data")
        assert response.status_code == 200
        data = response.json()
        assert "sample_data" in data
        assert "LIMIT_BAL" in data["sample_data"]
    
    def test_model_info_endpoint(self):
        """Тест endpoint информации о модели"""
        response = client.get("/model_info")
        assert response.status_code == 200
    
    def test_predict_endpoint_structure(self):
        """Тест структуры predict endpoint"""
        sample_data = {
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
        
        response = client.post("/predict", json=sample_data)
        
        # Даже если модель не загружена, должен быть корректный ответ
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "default_probability" in data
            assert "status" in data
            
            # Проверяем типы данных
            assert isinstance(data["prediction"], int)
            assert isinstance(data["probability"], float)
            assert isinstance(data["default_probability"], float)
            
            # Проверяем диапазоны значений
            assert data["prediction"] in [0, 1]
            assert 0 <= data["probability"] <= 1
            assert 0 <= data["default_probability"] <= 1

def test_batch_predict_structure():
    """Тест структуры batch predict endpoint"""
    sample_data_list = [
        {
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
    ]
    
    response = client.post("/batch_predict", json=sample_data_list)
    
    # Проверяем структуру ответа
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert "status" in data

def test_invalid_data_validation():
    """Тест валидации невалидных данных"""
    invalid_data = {
        "LIMIT_BAL": "invalid",  # Неправильный тип
        "SEX": 3,  # Неправильное значение
        "AGE": 15  # Слишком молодой
    }
    
    response = client.post("/predict", json=invalid_data)
    # Должен вернуть ошибку валидации
    assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v"])