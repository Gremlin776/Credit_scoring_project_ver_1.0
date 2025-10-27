import pytest
import pandas as pd
import numpy as np
import sys
import os

# Добавляем путь к src для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.make_dataset import clean_data, preprocess_data
from features.build_features import FeatureEngineer

class TestPreprocessing:
    """Тесты для модуля предобработки данных"""
    
    def test_clean_data(self):
        """Тест функции очистки данных"""
        # Создаем тестовые данные с "грязными" значениями
        test_data = pd.DataFrame({
            'EDUCATION': [1, 2, 0, 5, 6],  # Некорректные значения
            'MARRIAGE': [1, 2, 0, 3, 1],   # Некорректные значения
            'LIMIT_BAL': [10000, 20000, 30000, 40000, 50000],
            'default.payment.next.month': [0, 1, 0, 1, 0]
        })
        
        cleaned_data = clean_data(test_data)
        
        # Проверяем что некорректные значения заменены
        assert set(cleaned_data['EDUCATION'].unique()) <= {1, 2, 3, 4}
        assert set(cleaned_data['MARRIAGE'].unique()) <= {1, 2, 3}
    
    def test_preprocess_data(self):
        """Тест основного препроцессинга"""
        test_data = pd.DataFrame({
            'ID': [1, 2, 3],
            'LIMIT_BAL': [10000, 20000, 30000],
            'default.payment.next.month': [0, 1, 0]
        })
        
        processed_data = preprocess_data(test_data)
        
        # Проверяем что ID удален и целевая переменная переименована
        assert 'ID' not in processed_data.columns
        assert 'default' in processed_data.columns
    
    def test_feature_engineering(self):
        """Тест feature engineering"""
        test_data = pd.DataFrame({
            'PAY_0': [0, -1, -2],
            'PAY_2': [0, -1, -2],
            'PAY_3': [0, -1, -2],
            'BILL_AMT1': [1000, 2000, 3000],
            'BILL_AMT2': [1000, 2000, 3000],
            'LIMIT_BAL': [50000, 60000, 70000],
            'AGE': [25, 35, 45],
            'SEX': [1, 2, 1],
            'EDUCATION': [2, 3, 1],
            'MARRIAGE': [1, 2, 1],
            'default': [0, 1, 0]
        })
        
        feature_engineer = FeatureEngineer()
        df_with_features = feature_engineer.fit_transform(test_data)
        
        # Проверяем что созданы новые признаки
        expected_features = ['PAYMENT_MEAN', 'BILL_AMT_MEAN', 'PAY_AMT_MEAN', 'RISK_SCORE_1']
        for feature in expected_features:
            assert feature in df_with_features.columns
        
        # Проверяем корректность вычислений
        assert 'PAYMENT_MEAN' in df_with_features.columns
        assert df_with_features['PAYMENT_MEAN'].isna().sum() == 0

def test_data_validation():
    """Тест валидации данных"""
    # Этот тест может быть расширен для проверки Great Expectations
    valid_data = pd.DataFrame({
        'LIMIT_BAL': [50000, 100000],
        'SEX': [1, 2],
        'EDUCATION': [2, 3],
        'MARRIAGE': [1, 2],
        'AGE': [25, 35],
        'default': [0, 1]
    })
    
    # Проверяем что данные проходят базовые проверки
    assert valid_data['SEX'].isin([1, 2]).all()
    assert valid_data['EDUCATION'].isin([1, 2, 3, 4]).all()
    assert valid_data['default'].isin([0, 1]).all()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])