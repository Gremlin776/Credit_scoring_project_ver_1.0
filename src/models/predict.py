#!/usr/bin/env python3
"""
Модуль для предсказания с использованием обученной модели
"""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np


class ModelPredictor:
    """Класс для предсказаний модели"""

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.load_model()

    def load_model(self):
        """Загружает модель"""
        print("ЗАГРУЗКА МОДЕЛИ...")

        model_path = Path("models/best_model/model.pkl")

        if model_path.exists():
            print(f"Файл модели найден: {model_path}")
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.model_loaded = True
                print("МОДЕЛЬ УСПЕШНО ЗАГРУЖЕНА!")
                print(f"Тип модели: {type(self.model)}")
            except Exception as e:
                print(f"Ошибка загрузки: {e}")
        else:
            print("Файл модели не найден!")

    def create_features(self, input_df):
        """Создание ВСЕХ необходимых фич как при обучении"""
        df = input_df.copy()

        # 1. Признаки из истории платежей
        pay_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

        df["PAYMENT_MEAN"] = df[pay_columns].mean(axis=1)
        df["PAYMENT_MAX_DELAY"] = df[pay_columns].max(axis=1)
        df["PAYMENT_DELAY_COUNT"] = (df[pay_columns] > 0).sum(axis=1)
        df["PAYMENT_SEVERITY"] = (df[pay_columns] > 1).sum(axis=1)

        # 2. Признаки из сумм счетов
        bill_columns = [
            "BILL_AMT1",
            "BILL_AMT2",
            "BILL_AMT3",
            "BILL_AMT4",
            "BILL_AMT5",
            "BILL_AMT6",
        ]

        df["BILL_AMT_MEAN"] = df[bill_columns].mean(axis=1)
        df["BILL_AMT_MAX"] = df[bill_columns].max(axis=1)
        df["BILL_AMT_TO_LIMIT_RATIO"] = df["BILL_AMT_MEAN"] / (df["LIMIT_BAL"] + 1e-6)
        df["BILL_AMT_UTILIZATION"] = df["BILL_AMT_MAX"] / (df["LIMIT_BAL"] + 1e-6)

        # 3. Демографические признаки
        df["LIMIT_BAL_PER_AGE"] = df["LIMIT_BAL"] / (df["AGE"] + 1e-6)
        df["LIMIT_BAL_LOG"] = np.log1p(df["LIMIT_BAL"])

        # 4. Риск-признаки
        df["RISK_SCORE_1"] = (
            df["PAY_0"] * 0.3
            + df["PAYMENT_DELAY_COUNT"] * 0.2
            + df["BILL_AMT_TO_LIMIT_RATIO"] * 0.2
            + (df["AGE"] < 30).astype(int) * 0.1
            + (df["EDUCATION"] == 4).astype(int) * 0.1
            + (df["LIMIT_BAL"] < 50000).astype(int) * 0.1
        )

        df["CRITICAL_DEBT"] = (
            (df["PAYMENT_MAX_DELAY"] > 2) & (df["BILL_AMT_TO_LIMIT_RATIO"] > 0.8)
        ).astype(int)

        print(f"Создано фич: {len(df.columns)}")
        return df

    def preprocess_input(self, input_data):
        """Препроцессинг с созданием ВСЕХ фич"""
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()

        # Создаем ВСЕ фичи как при обучении
        input_df = self.create_features(input_df)
        return input_df

    def predict(self, input_data):
        """Предсказание с правильными фичами"""
        if not self.model_loaded or self.model is None:
            print("Используется ДЕМО-режим!")
            return self.demo_predict(input_data)

        try:
            # Препроцессинг с созданием фич
            input_df = self.preprocess_input(input_data)
            print(f"Входные данные после feature engineering: {input_df.shape}")
            print(f"Колонки: {list(input_df.columns)}")

            # Предсказание
            prediction = self.model.predict(input_df)
            probability = self.model.predict_proba(input_df)

            result = {
                "prediction": int(prediction[0]),
                "probability": float(probability[0][1]),
                "default_probability": float(probability[0][1]),
                "risk_level": self.get_risk_level(float(probability[0][1])),
                "model_status": "trained",
            }

            print(f"Результат предсказания: {result}")
            return result

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            import traceback

            traceback.print_exc()
            return self.demo_predict(input_data)

    def demo_predict(self, input_data):
        """Демо-режим"""
        print("АКТИВИРОВАН ДЕМО-РЕЖИМ!")

        if isinstance(input_data, dict):
            pay_0 = input_data.get("PAY_0", 0)
            limit_bal = input_data.get("LIMIT_BAL", 0)
        else:
            pay_0 = 0
            limit_bal = 0

        if pay_0 <= 0 and limit_bal > 100000:
            default_prob = 0.05
        elif pay_0 <= 1 and limit_bal > 50000:
            default_prob = 0.15
        elif pay_0 >= 2 or limit_bal < 20000:
            default_prob = 0.75
        else:
            default_prob = 0.3

        result = {
            "prediction": 1 if default_prob > 0.5 else 0,
            "probability": default_prob,
            "default_probability": default_prob,
            "risk_level": self.get_risk_level(default_prob),
            "model_status": "demo",
        }

        return result

    def get_risk_level(self, probability):
        """Определение уровня риска"""
        if probability < 0.1:
            return "очень низкий"
        elif probability < 0.2:
            return "низкий"
        elif probability < 0.4:
            return "средний"
        elif probability < 0.6:
            return "высокий"
        else:
            return "очень высокий"


# ТЕСТИРУЕМ
if __name__ == "__main__":
    print("=" * 50)
    print("ТЕСТ МОДЕЛИ С ФИЧАМИ")
    print("=" * 50)

    predictor = ModelPredictor()
    print(f"Модель загружена: {predictor.model_loaded}")

    if predictor.model_loaded:
        test_data = {
            "LIMIT_BAL": 200000,
            "SEX": 2,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 35,
            "PAY_0": -1,
            "PAY_2": -1,
            "PAY_3": -1,
            "PAY_4": -1,
            "PAY_5": -1,
            "PAY_6": -1,
            "BILL_AMT1": 35000,
            "BILL_AMT2": 34000,
            "BILL_AMT3": 33000,
            "BILL_AMT4": 32000,
            "BILL_AMT5": 31000,
            "BILL_AMT6": 30000,
            "PAY_AMT1": 5000,
            "PAY_AMT2": 5000,
            "PAY_AMT3": 5000,
            "PAY_AMT4": 5000,
            "PAY_AMT5": 5000,
            "PAY_AMT6": 5000,
        }

        result = predictor.predict(test_data)
        print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: {result}")
    else:
        print("МОДЕЛЬ НЕ ЗАГРУЖЕНА!")