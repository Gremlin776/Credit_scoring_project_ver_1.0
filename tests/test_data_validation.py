import pytest
import pandas as pd
import numpy as np
from src.data.make_dataset import load_and_clean_data
from src.data.validation import DataValidator

class TestDataValidation:
    """Тесты для валидации данных"""
    
    def test_data_loading(self):
        """Тест загрузки данных"""
        df = load_and_clean_data()
        assert df.shape[0] > 0, "Данные не загружены"
        assert 'default' in df.columns, "Отсутствует целевая переменная"
        
    def test_data_validator(self):
        """Тест валидатора данных"""
        validator = DataValidator()
        sample_data = pd.DataFrame({
            'AGE': [25, 30, 35],
            'LIMIT_BAL': [10000, 20000, 30000],
            'default': [0, 1, 0]
        })
        
        is_valid, report = validator.validate(sample_data)
        assert is_valid == True
        assert 'missing_values' in report

class TestFeatureEngineering:
    """Тесты для feature engineering"""
    
    def test_feature_engineer_creation(self):
        """Тест создания фич"""
        from src.features.build_features import FeatureEngineer
        
        engineer = FeatureEngineer()
        sample_data = pd.DataFrame({
            'PAY_0': [-1, 0, 1],
            'PAY_2': [-1, 0, 1],
            'BILL_AMT1': [1000, 2000, 3000],
            'default': [0, 1, 0]
        })
        
        result = engineer.fit_transform(sample_data)
        assert 'PAYMENT_MEAN' in result.columns
        assert 'BILL_AMT_MEAN' in result.columns