import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
import sys
import io

# Принудительно устанавливаем UTF-8 кодировку для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')


def load_raw_data():
    """Загрузка исходных данных"""
    df = pd.read_csv('data/raw/UCI_Credit_Card.csv')
    print(f"Загружено данных: {df.shape}")
    return df

def clean_data(df):
    """Очистка данных на основе EDA insights"""
    print("Очистка данных...")
    
    # Обработка EDUCATION на основе EDA
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})  # Группировка 'Других'
    df['EDUCATION'] = df['EDUCATION'].clip(1, 4)  # Ограничение диапазона
    
    # Обработка MARRIAGE на основе EDA
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})  # Замена 0 на 'Другое'
    df['MARRIAGE'] = df['MARRIAGE'].clip(1, 3)  # Ограничение диапазона
    
    print("Очистка данных завершена")
    return df

def preprocess_data(df):
    """Основной препроцессинг данных"""
    # Удаление ID
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Переименование целевой переменной для удобства
    df = df.rename(columns={'default.payment.next.month': 'default'})
    
    return df

def save_processed_data(df, train_df, test_df):
    """Сохранение обработанных данных"""
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/processed_data.csv', index=False)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    # Сохранение информации о данных
    data_info = {
        'original_shape': df.shape,
        'train_shape': train_df.shape,
        'test_shape': test_df.shape,
        'default_rate_train': train_df['default'].mean(),
        'default_rate_test': test_df['default'].mean()
    }
    
    os.makedirs('reports', exist_ok=True)
    with open('reports/data_info.json', 'w') as f:
        json.dump(data_info, f, indent=2)

def main():
    """Основная функция подготовки данных с учетом EDA insights"""
    print("=== ПОДГОТОВКА ДАННЫХ ===")
    
    # Загрузка
    df = load_raw_data()
    
    # Очистка на основе EDA
    df = clean_data(df)
    
    # Препроцессинг
    df = preprocess_data(df)
    
    # Стратифицированное разделение с учетом дисбаланса
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['default']
    )
    
    # Сохранение
    save_processed_data(df, train_df, test_df)
    
    # Отчет
    print(f"✅ Итоговый размер данных: {df.shape}")
    print(f"✅ Размер тренировочной выборки: {train_df.shape}")
    print(f"✅ Размер тестовой выборки: {test_df.shape}")
    print(f"✅ Процент дефолтов в тренировочной: {train_df['default'].mean():.3f}")
    print(f"✅ Процент дефолтов в тестовой: {test_df['default'].mean():.3f}")
    print("✅ Подготовка данных завершена!")

if __name__ == "__main__":
    main()