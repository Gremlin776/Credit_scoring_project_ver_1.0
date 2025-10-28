import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import requests
import json
import os
from pathlib import Path
import sys
import io

def calculate_psi(expected, actual, buckets=10):
    """Calculate Population Stability Index"""
    # Определяем границы бинов на основе тренировочных данных
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    
    # Избегаем дублирующихся границ
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        breakpoints = np.linspace(np.min(expected), np.max(expected), buckets + 1)
    
    # Вычисляем проценты для expected и actual
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Добавляем небольшое значение чтобы избежать деления на 0
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Вычисляем PSI
    psi = np.sum((expected_percents - actual_percents) * 
                 np.log(expected_percents / actual_percents))
    return psi

def detect_drift():
    """Обнаружение дрифта в данных"""
    print("Запуск мониторинга дрифта...")
    
    try:
        # Загрузка тренировочных данных (эталон)
        train_data = pd.read_csv('data/processed/train.csv')
        
        # Имитация новых данных (в продакшне здесь был бы реальный поток данных)
        test_data = pd.read_csv('data/processed/test.csv').sample(1000, random_state=42)
        
        drift_metrics = {}
        
        # PSI для ключевых признаков на основе EDA
        key_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAYMENT_MEAN', 'BILL_AMT_TO_LIMIT_RATIO']
        
        for feature in key_features:
            if feature in train_data.columns and feature in test_data.columns:
                try:
                    psi = calculate_psi(train_data[feature].dropna(), test_data[feature].dropna())
                    drift_metrics[f'psi_{feature}'] = float(psi)
                    
                    # KS test для статистической значимости
                    ks_stat, ks_pvalue = ks_2samp(train_data[feature].dropna(), test_data[feature].dropna())
                    drift_metrics[f'ks_{feature}'] = float(ks_stat)
                    drift_metrics[f'ks_pvalue_{feature}'] = float(ks_pvalue)
                    
                    print(f"   {feature}: PSI={psi:.4f}, KS-pvalue={ks_pvalue:.4f}")
                    
                except Exception as e:
                    print(f"Ошибка для {feature}: {e}")
                    continue
        
        # Дрифт в распределении целевой переменной
        if 'default' in train_data.columns and 'default' in test_data.columns:
            target_drift = abs(train_data['default'].mean() - test_data['default'].mean())
            drift_metrics['target_drift'] = float(target_drift)
            print(f"   Target drift: {target_drift:.4f}")
        
        # Оценка общего дрифта
        high_drift_features = [k for k, v in drift_metrics.items() 
                             if k.startswith('psi_') and v > 0.2]
        medium_drift_features = [k for k, v in drift_metrics.items() 
                               if k.startswith('psi_') and 0.1 < v <= 0.2]
        
        drift_metrics['high_drift_features_count'] = len(high_drift_features)
        drift_metrics['medium_drift_features_count'] = len(medium_drift_features)
        drift_metrics['drift_status'] = 'high' if len(high_drift_features) > 0 else 'medium' if len(medium_drift_features) > 0 else 'low'
        
        # Сохранение метрик дрифта
        os.makedirs('reports', exist_ok=True)
        with open('reports/drift_metrics.json', 'w') as f:
            json.dump(drift_metrics, f, indent=2)
        
        print("\n Результаты мониторинга дрифта:")
        print(f"   Статус дрифта: {drift_metrics['drift_status']}")
        print(f"   Признаков с высоким дрифтом: {len(high_drift_features)}")
        print(f"   Признаков со средним дрифтом: {len(medium_drift_features)}")
        
        return drift_metrics
        
    except Exception as e:
        print(f" Ошибка мониторинга дрифта: {e}")
        return {}

def test_api_predictions():
    """Тестирование API и мониторинг дрифта предсказаний"""
    print("\n🧪 Тестирование API предсказаний...")
    
    try:
        # Создание тестовых данных
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
        
        # Отправка запроса к API
        response = requests.post('http://localhost:8000/predict', 
                               json=sample_data, 
                               timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   API тест: Успешно")
            print(f"   Предсказание: {result}")
            return True
        else:
            print(f"   API тест: Ошибка {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   API тест: {e}")
        return False

def main():
    """Основная функция мониторинга"""
    print("ЗАПУСК СИСТЕМЫ МОНИТОРИНГА")
    print("=" * 50)
    
    # Детекция дрифта данных
    drift_results = detect_drift()
    
    # Тестирование API
    api_test_result = test_api_predictions()
    
    # Генерация отчета
    monitoring_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_drift": drift_results,
        "api_status": "healthy" if api_test_result else "degraded",
        "recommendations": []
    }
    
    # Добавление рекомендаций
    if drift_results.get('drift_status') == 'high':
        monitoring_report["recommendations"].append(
            "Обнаружен высокий дрифт данных. Рекомендуется переобучение модели."
        )
    elif drift_results.get('drift_status') == 'medium':
        monitoring_report["recommendations"].append(
            "Обнаружен средний дрифт данных. Рекомендуется мониторинг."
        )
    
    if not api_test_result:
        monitoring_report["recommendations"].append(
            "API недоступен. Проверьте запущено ли приложение."
        )
    
    # Сохранение полного отчета
    with open('reports/monitoring_report.json', 'w') as f:
        json.dump(monitoring_report, f, indent=2)
    
    print("\nОТЧЕТ МОНИТОРИНГА СОХРАНЕН")
    print(f"   Файл: reports/monitoring_report.json")
    print(f"   Рекомендации: {len(monitoring_report['recommendations'])}")

if __name__ == "__main__":
    main()