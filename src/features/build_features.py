import pandas as pd
import numpy as np

import sys
import io

# Принудительно устанавливаем UTF-8 кодировку для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

class FeatureEngineer:
    """Класс для Feature Engineering с учетом insights из EDA"""
    
    def __init__(self):
        self.features_created = []
    
    def create_payment_features(self, df):
        """Создание признаков из истории платежей с учетом EDA корреляций"""
        pay_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        
        # Признаки на основе EDA (высокая корреляция с целевой)
        df['PAYMENT_MEAN'] = df[pay_columns].mean(axis=1)
        df['PAYMENT_MAX_DELAY'] = df[pay_columns].max(axis=1)
        df['PAYMENT_MIN_DELAY'] = df[pay_columns].min(axis=1)
        df['PAYMENT_STD'] = df[pay_columns].std(axis=1)
        
        # Количество просроченных платежей (на основе EDA)
        df['PAYMENT_DELAY_COUNT'] = (df[pay_columns] > 0).sum(axis=1)
        df['PAYMENT_SEVERITY'] = (df[pay_columns] > 1).sum(axis=1)
        
        self.features_created.extend([
            'PAYMENT_MEAN', 'PAYMENT_MAX_DELAY', 'PAYMENT_MIN_DELAY',
            'PAYMENT_STD', 'PAYMENT_DELAY_COUNT', 'PAYMENT_SEVERITY'
        ])
        return df
    
    def create_bill_amount_features(self, df):
        """Создание признаков из сумм счетов с учетом EDA"""
        bill_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
                       'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
        
        # Признаки на основе EDA insights
        df['BILL_AMT_MEAN'] = df[bill_columns].mean(axis=1)
        df['BILL_AMT_STD'] = df[bill_columns].std(axis=1)
        df['BILL_AMT_MAX'] = df[bill_columns].max(axis=1)
        df['BILL_AMT_MIN'] = df[bill_columns].min(axis=1)
        
        # Использование кредитного лимита (важно по EDA)
        df['BILL_AMT_TO_LIMIT_RATIO'] = df['BILL_AMT_MEAN'] / (df['LIMIT_BAL'] + 1e-6)
        df['BILL_AMT_UTILIZATION'] = df['BILL_AMT_MAX'] / (df['LIMIT_BAL'] + 1e-6)
        
        self.features_created.extend([
            'BILL_AMT_MEAN', 'BILL_AMT_STD', 'BILL_AMT_MAX', 'BILL_AMT_MIN',
            'BILL_AMT_TO_LIMIT_RATIO', 'BILL_AMT_UTILIZATION'
        ])
        return df
    
    def create_payment_amount_features(self, df):
        """Создание признаков из сумм платежей с учетом EDA"""
        pay_amt_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 
                          'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        
        # Признаки на основе EDA
        df['PAY_AMT_MEAN'] = df[pay_amt_columns].mean(axis=1)
        df['PAY_AMT_SUM'] = df[pay_amt_columns].sum(axis=1)
        df['PAY_AMT_STD'] = df[pay_amt_columns].std(axis=1)
        df['PAY_AMT_MAX'] = df[pay_amt_columns].max(axis=1)
        
        # Отношения на основе EDA insights
        df['PAYMENT_TO_BILL_RATIO'] = df['PAY_AMT_MEAN'] / (df['BILL_AMT_MEAN'] + 1e-6)
        df['PAYMENT_TO_LIMIT_RATIO'] = df['PAY_AMT_MEAN'] / (df['LIMIT_BAL'] + 1e-6)
        
        self.features_created.extend([
            'PAY_AMT_MEAN', 'PAY_AMT_SUM', 'PAY_AMT_STD', 'PAY_AMT_MAX',
            'PAYMENT_TO_BILL_RATIO', 'PAYMENT_TO_LIMIT_RATIO'
        ])
        return df
    
    def create_demographic_features(self, df):
        """Создание демографических признаков с учетом EDA"""
        # Биннинг возраста на основе EDA
        df['AGE_GROUP'] = pd.cut(df['AGE'], bins=[0, 25, 35, 45, 55, 100], 
                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # Признаки на основе EDA insights
        df['LIMIT_BAL_PER_AGE'] = df['LIMIT_BAL'] / (df['AGE'] + 1e-6)
        df['LIMIT_BAL_LOG'] = np.log1p(df['LIMIT_BAL'])
        
        self.features_created.extend([
            'AGE_GROUP', 'LIMIT_BAL_PER_AGE', 'LIMIT_BAL_LOG'
        ])
        return df
    
    def create_risk_features(self, df):
        """Создание риск-ориентированных признаков на основе EDA"""
        # Композитный риск-скор на основе высококоррелирующих признаков
        df['RISK_SCORE_1'] = (
            df['PAY_0'] * 0.3 + 
            df['PAYMENT_DELAY_COUNT'] * 0.2 +
            df['BILL_AMT_TO_LIMIT_RATIO'] * 0.2 +
            (df['AGE'] < 30).astype(int) * 0.1 +
            (df['EDUCATION'] == 4).astype(int) * 0.1 +
            (df['LIMIT_BAL'] < 50000).astype(int) * 0.1
        )
        
        # Признак "критической задолженности"
        df['CRITICAL_DEBT'] = (
            (df['PAYMENT_MAX_DELAY'] > 2) & 
            (df['BILL_AMT_TO_LIMIT_RATIO'] > 0.8)
        ).astype(int)
        
        self.features_created.extend(['RISK_SCORE_1', 'CRITICAL_DEBT'])
        return df
    
    def fit_transform(self, df):
        """Применение всех преобразований с учетом EDA insights"""
        print("🔧 Feature Engineering...")
        
        df = self.create_payment_features(df)
        df = self.create_bill_amount_features(df)
        df = self.create_payment_amount_features(df)
        df = self.create_demographic_features(df)
        df = self.create_risk_features(df)
        
        print(f"✅ Создано {len(self.features_created)} новых признаков")
        
        return df

def main():
    """Основная функция для создания фич с учетом EDA"""
    try:
        # Загрузка данных
        df = pd.read_csv('data/processed/processed_data.csv')
        
        # Feature Engineering с учетом EDA
        feature_engineer = FeatureEngineer()
        df_with_features = feature_engineer.fit_transform(df)
        
        # Анализ новых признаков
        correlation_with_target = df_with_features.corr()['default'].sort_values(
            key=abs, ascending=False
        )
        
        print("\n📊 Топ-10 признаков по корреляции с целевой переменной:")
        for feature, corr in correlation_with_target.head(10).items():
            print(f"   {feature}: {corr:.4f}")
        
        # Сохранение
        df_with_features.to_csv('data/processed/data_with_features.csv', index=False)
        
        # Сохранение информации о фичах
        features_info = {
            'total_features': len(df_with_features.columns),
            'new_features_created': len(feature_engineer.features_created),
            'top_correlated_features': correlation_with_target.head(10).to_dict()
        }
        
        import json
        with open('reports/features_info.json', 'w') as f:
            json.dump(features_info, f, indent=2)
        
        print("✅ Feature engineering завершен!")
        
    except Exception as e:
        print(f"❌ Ошибка feature engineering: {e}")

if __name__ == "__main__":
    main()