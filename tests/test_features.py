import pytest
import pandas as pd
import numpy as np
from src.features.build_features import FeatureEngineer

class TestFeatureEngineering:
    """Тесты для инженерии признаков"""
    
    def setup_method(self):
        """Настройка тестовых данных"""
        self.sample_data = pd.DataFrame({
            'LIMIT_BAL': [10000, 20000, 30000, 40000],
            'AGE': [25, 30, 35, 40],
            'PAY_0': [-1, 0, 1, 2],
            'PAY_2': [-1, 0, 1, 2],
            'PAY_3': [-1, 0, 1, 2],
            'BILL_AMT1': [1000, 2000, 3000, 4000],
            'BILL_AMT2': [1500, 2500, 3500, 4500],
            'PAY_AMT1': [500, 1000, 1500, 2000],
            'PAY_AMT2': [600, 1100, 1600, 2100],
            'default': [0, 1, 0, 1]
        })
        
    def test_feature_engineer_initialization(self):
        """Тест инициализации FeatureEngineer"""
        engineer = FeatureEngineer()
        assert engineer is not None
        
    def test_feature_creation(self):
        """Тест создания новых признаков"""
        engineer = FeatureEngineer()
        transformed_data = engineer.fit_transform(self.sample_data)
        
        # Проверяем создание новых признаков
        expected_features = ['PAYMENT_MEAN', 'BILL_AMT_MEAN', 'PAY_AMT_MEAN']
        for feature in expected_features:
            assert feature in transformed_data.columns
            
    def test_no_data_loss(self):
        """Тест что данные не теряются при трансформации"""
        engineer = FeatureEngineer()
        transformed_data = engineer.fit_transform(self.sample_data)
        
        assert len(transformed_data) == len(self.sample_data)
        assert 'default' in transformed_data.columns
        
    def test_handle_missing_values(self):
        """Тест обработки пропущенных значений"""
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[0, 'PAY_0'] = np.nan
        
        engineer = FeatureEngineer()
        transformed_data = engineer.fit_transform(data_with_nan)
        
        # Проверяем что нет NaN в результатах
        assert not transformed_data.isnull().any().any()