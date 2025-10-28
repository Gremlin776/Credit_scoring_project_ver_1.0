import pytest
import pandas as pd
import numpy as np
from src.models.pipeline import CreditScoringPipeline
from src.models.train import ModelTrainer
from src.models.predict import ModelPredictor

class TestModelPipeline:
    """Тесты для ML пайплайна"""
    
    def setup_method(self):
        """Настройка тестовых данных"""
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'LIMIT_BAL': np.random.randint(10000, 50000, 100),
            'AGE': np.random.randint(20, 60, 100),
            'PAYMENT_MEAN': np.random.uniform(-2, 2, 100),
            'BILL_AMT_MEAN': np.random.uniform(1000, 50000, 100),
            'default': np.random.randint(0, 2, 100)
        })
        
    def test_pipeline_initialization(self):
        """Тест инициализации пайплайна"""
        pipeline = CreditScoringPipeline()
        assert pipeline is not None
        
    def test_model_trainer(self):
        """Тест обучения модели"""
        trainer = ModelTrainer()
        X = self.sample_data.drop('default', axis=1)
        y = self.sample_data['default']
        
        model, metrics = trainer.train(X, y)
        
        assert model is not None
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
    def test_model_predictor(self):
        """Тест предсказаний модели"""
        predictor = ModelPredictor()
        
        # Создаем простую модель для тестов
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        X = self.sample_data.drop('default', axis=1)
        y = self.sample_data['default']
        model.fit(X, y)
        
        predictor.model = model
        
        # Тестируем предсказание
        sample_input = X.iloc[0:1]
        result = predictor.predict(sample_input)
        
        assert 'prediction' in result
        assert 'probability' in result
        assert 0 <= result['probability'] <= 1
        
    def test_prediction_format(self):
        """Тест формата выходных данных предсказания"""
        predictor = ModelPredictor()
        
        sample_input = pd.DataFrame({
            'LIMIT_BAL': [20000],
            'AGE': [30], 
            'PAYMENT_MEAN': [0.5],
            'BILL_AMT_MEAN': [25000]
        })
        
        # Мокаем модель для теста
        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.7, 0.3]])
                
        predictor.model = MockModel()
        result = predictor.predict(sample_input)
        
        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'probability' in result
        assert result['prediction'] in [0, 1]