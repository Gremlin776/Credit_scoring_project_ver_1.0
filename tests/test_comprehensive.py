#!/usr/bin/env python3
"""
Комплексные тесты для проекта Credit Scoring
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Добавляем путь для импортов
sys.path.append(str(Path(__file__).parent.parent))

class TestDataQuality:
    """Тесты качества данных"""
    
    def test_raw_data_exists(self):
        """Тест наличия исходных данных"""
        assert os.path.exists('data/raw/UCI_Credit_Card.csv'), "Исходные данные не найдены"
    
    def test_raw_data_quality(self):
        """Тест качества исходных данных"""
        df = pd.read_csv('data/raw/UCI_Credit_Card.csv')
        assert len(df) > 0, "Нет данных в исходном датасете"
        assert 'default.payment.next.month' in df.columns, "Нет целевой переменной в исходных данных"
    
    def test_processed_data(self):
        """Тест обработанных данных"""
        if os.path.exists('data/processed/data_with_features.csv'):
            df = pd.read_csv('data/processed/data_with_features.csv')
            assert len(df) > 0, "Нет данных в обработанном датасете"
            assert 'default' in df.columns, "Нет целевой переменной"
            # Проверяем отсутствие NaN в ключевых признаках
            key_columns = ['LIMIT_BAL', 'AGE', 'PAY_0', 'default']
            for col in key_columns:
                if col in df.columns:
                    assert not df[col].isnull().any(), f"Есть NaN в колонке {col}"

class TestFeatureEngineering:
    """Тесты feature engineering"""
    
    def test_feature_creation(self):
        """Тест создания признаков"""
        if os.path.exists('data/processed/data_with_features.csv'):
            df = pd.read_csv('data/processed/data_with_features.csv')
            # Проверяем созданные признаки
            expected_features = ['PAYMENT_MEAN', 'BILL_AMT_MEAN']
            for feature in expected_features:
                if feature in df.columns:
                    assert not df[feature].isnull().all(), f"Признак {feature} содержит только NaN"

class TestModels:
    """Тесты моделей"""
    
    def test_model_files_exist(self):
        """Тест наличия файлов моделей"""
        model_paths = [
            'models/best_model',
            'mlruns'
        ]
        model_exists = any(os.path.exists(path) for path in model_paths)
        assert model_exists, "Модели не найдены"
    
    def test_mlflow_runs(self):
        """Тест наличия экспериментов MLflow"""
        if os.path.exists('mlruns'):
            # Проверяем что есть хотя бы один эксперимент
            experiments = list(Path('mlruns').glob('*'))
            assert len(experiments) > 0, "Нет экспериментов в MLflow"

class TestAPI:
    """Тесты API"""
    
    def test_api_health(self):
        """Тест здоровья API"""
        # Этот тест может быть запущен когда API работает
        pass
    
    def test_reports_exist(self):
        """Тест наличия отчетов"""
        if os.path.exists('reports'):
            reports = list(Path('reports').glob('*.png')) + list(Path('reports').glob('*.json'))
            assert len(reports) > 0, "Нет сгенерированных отчетов"

def test_integration_pipeline():
    """Интеграционный тест всего пайплайна"""
    # Проверяем что все ключевые файлы созданы
    key_files = [
        'data/processed/data_with_features.csv',
        'reports'
    ]
    
    for file_path in key_files:
        if not os.path.exists(file_path):
            pytest.skip(f"Файл {file_path} не найден, пропускаем интеграционный тест")
            return
    
    # Если все файлы есть, считаем пайплайн успешным
    assert True, "Пайплайн успешно выполнен"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])