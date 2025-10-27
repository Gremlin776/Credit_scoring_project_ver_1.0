import pandas as pd
import mlflow.sklearn
import os
import numpy as np
from .pipeline import create_model_pipeline
import sys
import io

# Принудительно устанавливаем UTF-8 кодировку для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

class ModelPredictor:
    """Класс для предсказаний модели"""
    
    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = mlflow.sklearn.load_model(model_path)
                print(f"✅ Модель загружена из: {model_path}")
            except Exception as e:
                print(f"❌ Ошибка загрузки модели: {e}")
        else:
            # Попытка загрузки последней обученной модели
            self.model = self.load_latest_model()
    
    def load_latest_model(self):
        """Загрузка последней обученной модели"""
        try:
            # Поиск в MLflow или локальной директории
            if os.path.exists('models/best_model'):
                return mlflow.sklearn.load_model('models/best_model')
            else:
                print("⚠️ Обученная модель не найдена")
                return None
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return None
    
    def preprocess_input(self, input_data):
        """Препроцессинг входных данных для предсказания"""
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input data must be dict or DataFrame")
        
        # Применение Feature Engineering (как при обучении)
        from src.features.build_features import FeatureEngineer
        feature_engineer = FeatureEngineer()
        input_df = feature_engineer.fit_transform(input_df)
        
        return input_df
    
    def predict(self, input_data):
        """Предсказание дефолта"""
        if self.model is None:
            raise ValueError("❌ Модель не загружена")
        
        try:
            input_df = self.preprocess_input(input_data)
            prediction = self.model.predict(input_df)
            probability = self.model.predict_proba(input_df)
            
            return {
                'prediction': int(prediction[0]),
                'probability': float(probability[0][1]),
                'default_probability': float(probability[0][1]),
                'class_probabilities': probability[0].tolist()
            }
        except Exception as e:
            raise ValueError(f"❌ Ошибка предсказания: {e}")
    
    def batch_predict(self, input_data_list):
        """Пакетное предсказание"""
        results = []
        for data in input_data_list:
            try:
                results.append(self.predict(data))
            except Exception as e:
                results.append({'error': str(e)})
        return results

def main():
    """Пример использования"""
    predictor = ModelPredictor()
    
    if predictor.model is None:
        print("❌ Не удалось загрузить модель для предсказаний")
        return
    
    # Пример данных для предсказания
    sample_data = {
        'LIMIT_BAL': 50000,
        'SEX': 1,
        'EDUCATION': 2,
        'MARRIAGE': 1,
        'AGE': 35,
        'PAY_0': 0,
        'PAY_2': 0,
        'PAY_3': 0,
        'PAY_4': 0,
        'PAY_5': 0,
        'PAY_6': 0,
        'BILL_AMT1': 1000,
        'BILL_AMT2': 1000,
        'BILL_AMT3': 1000,
        'BILL_AMT4': 1000,
        'BILL_AMT5': 1000,
        'BILL_AMT6': 1000,
        'PAY_AMT1': 1000,
        'PAY_AMT2': 1000,
        'PAY_AMT3': 1000,
        'PAY_AMT4': 1000,
        'PAY_AMT5': 1000,
        'PAY_AMT6': 1000
    }
    
    try:
        result = predictor.predict(sample_data)
        print(f"✅ Результат предсказания: {result}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()