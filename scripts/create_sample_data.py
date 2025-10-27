import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
import io

# Принудительно устанавливаем UTF-8 кодировку для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')


def create_sample_data():
    """Создание реалистичных sample данных кредитного скоринга"""
    print("Создание sample данных для Credit Scoring...")
    
    np.random.seed(42)
    n_samples = 5000  # Увеличим для более реалистичных результатов
    
    # Генерация реалистичных данных
    data = {
        'ID': range(1, n_samples + 1),
        'LIMIT_BAL': np.random.lognormal(11, 0.5, n_samples).astype(int),
        'SEX': np.random.choice([1, 2], n_samples, p=[0.45, 0.55]),
        'EDUCATION': np.random.choice([1, 2, 3, 4], n_samples, p=[0.1, 0.5, 0.3, 0.1]),
        'MARRIAGE': np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.4, 0.1]),
        'AGE': np.random.normal(35, 10, n_samples).astype(int).clip(20, 70),
    }
    
    # История платежей (более реалистичная)
    pay_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    for i, col in enumerate(pay_columns):
        # Создаем коррелированные данные платежей
        base = np.random.choice([-2, -1, 0, 1, 2, 3], n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05])
        data[col] = base + np.random.randint(-1, 2, n_samples)
        data[col] = data[col].clip(-2, 8)
    
    # Суммы счетов (коррелированные с лимитом)
    bill_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    for i, col in enumerate(bill_columns):
        utilization = np.random.uniform(0.1, 0.9, n_samples)
        noise = np.random.normal(0, 0.1, n_samples)
        data[col] = (data['LIMIT_BAL'] * utilization * (1 + noise)).astype(int).clip(0)
    
    # Суммы платежей (коррелированные со счетами)
    pay_amt_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    for i, col in enumerate(pay_amt_columns):
        bill_col = bill_columns[i]
        payment_ratio = np.random.uniform(0.05, 0.3, n_samples)
        data[col] = (data[bill_col] * payment_ratio).astype(int).clip(0, data[bill_col])
    
    # Целевая переменная (дефолт) с реалистичной логикой
    default_prob = (
        (data['PAY_0'] > 0) * 0.3 +                    # Просрочки увеличивают риск
        (data['AGE'] < 30) * 0.1 +                     # Молодой возраст
        (data['EDUCATION'] == 4) * 0.2 +               # Низкое образование
        (data['LIMIT_BAL'] < 50000) * 0.1 +           # Низкий лимит
        np.random.uniform(0, 0.3, n_samples)           # Случайный шум
    )
    
    data['default.payment.next.month'] = (default_prob > 0.5).astype(int)
    
    df = pd.DataFrame(data)
    
    # Сохранение
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'UCI_Credit_Card.csv', index=False)
    
    print(f"✅ Созданы sample данные: {len(df)} записей")
    print(f"   Распределение дефолтов: {df['default.payment.next.month'].mean():.2%}")
    print(f"   Файл: {output_dir / 'UCI_Credit_Card.csv'}")
    
    return df

if __name__ == "__main__":
    create_sample_data()