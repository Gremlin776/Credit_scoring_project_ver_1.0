#!/usr/bin/env python3
"""
EDA (Exploratory Data Analysis) для Credit Scoring проекта
Адаптивный анализ с автоматическими выводами на основе данных
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import io
import warnings
warnings.filterwarnings('ignore')

# Принудительно устанавливаем UTF-8 кодировку для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

def calculate_imbalance_ratio(target_series):
    """Рассчитывает коэффициент дисбаланса классов"""
    class_counts = target_series.value_counts()
    majority_class = class_counts.max()
    minority_class = class_counts.min()
    imbalance_ratio = minority_class / majority_class
    return imbalance_ratio, class_counts

def analyze_correlations(df, target_column, top_n=10):
    """Анализирует корреляции и возвращает значимые признаки"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_with_target = df[numeric_columns].corr()[target_column].abs().sort_values(ascending=False)
    
    # Исключаем целевую переменную из топ корреляций
    top_correlations = correlation_with_target[correlation_with_target.index != target_column].head(top_n)
    
    # Определяем высококоррелированные признаки (порог > 0.1)
    high_corr_features = top_correlations[top_correlations > 0.1]
    medium_corr_features = top_correlations[(top_correlations > 0.05) & (top_correlations <= 0.1)]
    
    return top_correlations, high_corr_features, medium_corr_features

def detect_data_issues(df):
    """Обнаруживает проблемы в данных"""
    issues = []
    
    # Пропущенные значения
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        issues.append(f"Пропущенные значения: {missing_data[missing_data > 0].to_dict()}")
    
    # Выбросы в числовых признаках
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_percent = len(outliers) / len(df) * 100
        
        if outlier_percent > 5:  # Если больше 5% - значительные выбросы
            outlier_info[col] = f"{outlier_percent:.1f}% выбросов"
    
    if outlier_info:
        issues.append(f"Значительные выбросы: {outlier_info}")
    
    # Некорректные значения в категориальных признаках
    categorical_checks = {
        'SEX': [1, 2],
        'EDUCATION': [1, 2, 3, 4],
        'MARRIAGE': [1, 2, 3]
    }
    
    for col, allowed_values in categorical_checks.items():
        if col in df.columns:
            unexpected_values = set(df[col].unique()) - set(allowed_values)
            if unexpected_values:
                issues.append(f"Некорректные значения в {col}: {unexpected_values}")
    
    return issues

def analyze_feature_distributions(df, target_column):
    """Анализирует распределения признаков относительно целевой переменной"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    feature_analysis = {}
    
    for col in numeric_columns:
        if col != target_column:
            # Статистический тест на различие распределений
            group_0 = df[df[target_column] == 0][col].dropna()
            group_1 = df[df[target_column] == 1][col].dropna()
            
            if len(group_0) > 0 and len(group_1) > 0:
                # T-тест для нормальных распределений, U-тест для остальных
                if col in ['LIMIT_BAL', 'AGE']:  # Предполагаем нормальность для этих признаков
                    stat, p_value = stats.ttest_ind(group_0, group_1, nan_policy='omit')
                else:
                    stat, p_value = stats.mannwhitneyu(group_0, group_1, nan_policy='omit')
                
                feature_analysis[col] = {
                    'p_value': p_value,
                    'mean_0': group_0.mean(),
                    'mean_1': group_1.mean(),
                    'significant': p_value < 0.05
                }
    
    return feature_analysis

def generate_insights(eda_results):
    """Генерирует автоматические выводы на основе анализа"""
    insights = []
    
    # Анализ дисбаланса
    imbalance_ratio = eda_results['imbalance_ratio']
    if imbalance_ratio < 0.3:
        insights.append("🔴 ВЫСОКИЙ ДИСБАЛАНС КЛАССОВ - рекомендуется использовать class_weight='balanced' или oversampling")
    elif imbalance_ratio < 0.5:
        insights.append("🟡 СРЕДНИЙ ДИСБАЛАНС КЛАССОВ - можно использовать стандартные методы")
    else:
        insights.append("🟢 НИЗКИЙ ДИСБАЛАНС КЛАССОВ - данные сбалансированы")
    
    # Анализ корреляций
    high_corr_count = len(eda_results['high_corr_features'])
    medium_corr_count = len(eda_results['medium_corr_features'])
    
    if high_corr_count > 0:
        insights.append(f"🔍 Обнаружено {high_corr_count} признаков с ВЫСОКОЙ корреляцией с целевой переменной")
    if medium_corr_count > 0:
        insights.append(f"📊 Обнаружено {medium_corr_count} признаков со СРЕДНЕЙ корреляцией с целевой переменной")
    
    if high_corr_count == 0 and medium_corr_count == 0:
        insights.append("⚠️ Нет признаков с существенной корреляцией - возможно, требуется более сложное feature engineering")
    
    # Анализ проблем данных
    if len(eda_results['data_issues']) > 0:
        insights.append("🚨 ОБНАРУЖЕНЫ ПРОБЛЕМЫ В ДАННЫХ - требуется предобработка")
    else:
        insights.append("✅ Данные в хорошем состоянии")
    
    # Анализ значимых признаков
    significant_features = [feature for feature, info in eda_results['feature_analysis'].items() 
                          if info['significant']]
    insights.append(f"📈 {len(significant_features)} признаков имеют статистически значимые различия между классами")
    
    return insights

def run_eda():
    """Запуск полного EDA анализа с автоматическими выводами"""
    print("🎯 ЗАПУСК АДАПТИВНОГО EDA АНАЛИЗА")
    print("=" * 60)

    # Создаем необходимые папки
    os.makedirs('reports', exist_ok=True)
    
    # Настройки отображения
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    pd.set_option('display.max_columns', None)
    
    # Загрузка данных
    try:
        df = pd.read_csv('data/raw/UCI_Credit_Card.csv')
        print(f"✅ Данные загружены: {df.shape}")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # Определяем целевую переменную
    target_column = 'default.payment.next.month'
    if target_column not in df.columns:
        # Пытаемся найти целевую переменную по другим возможным названиям
        possible_targets = ['default', 'target', 'y', 'Default']
        for col in possible_targets:
            if col in df.columns:
                target_column = col
                break
        else:
            print("❌ Не удалось найти целевую переменную")
            return
    
    # Базовая информация
    print(f"\n📊 БАЗОВАЯ ИНФОРМАЦИЯ:")
    print(f"Размер данных: {df.shape}")
    print(f"Целевая переменная: '{target_column}'")
    print(f"Типы данных:\n{df.dtypes.value_counts()}")
    
    # Анализ целевой переменной
    print(f"\n🎯 АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ:")
    target_counts = df[target_column].value_counts()
    target_percent = df[target_column].value_counts(normalize=True) * 100
    
    for val, count in target_counts.items():
        print(f"Класс {val}: {count} записей ({target_percent[val]:.2f}%)")
    
    # Расчет дисбаланса
    imbalance_ratio, class_counts = calculate_imbalance_ratio(df[target_column])
    print(f"Коэффициент дисбаланса: {imbalance_ratio:.3f}")
    
    # Визуализация распределения целевой переменной
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x=target_column)
    plt.title(f'Распределение целевой переменной\n(дисбаланс: {imbalance_ratio:.3f})')
    plt.xlabel('Класс')
    plt.ylabel('Количество')
    
    plt.subplot(1, 2, 2)
    plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
            colors=['lightblue', 'lightcoral'])
    plt.title('Соотношение классов')
    
    plt.tight_layout()
    plt.savefig('reports/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Корреляционный анализ
    print(f"\n🔗 КОРРЕЛЯЦИОННЫЙ АНАЛИЗ:")
    top_correlations, high_corr_features, medium_corr_features = analyze_correlations(df, target_column)
    
    print("Топ-10 признаков по корреляции с целевой переменной:")
    for feature, corr in top_correlations.head(10).items():
        significance = "🔴 ВЫСОКАЯ" if corr > 0.1 else "🟡 СРЕДНЯЯ" if corr > 0.05 else "⚪ НИЗКАЯ"
        print(f"  {significance} {feature}: {corr:.4f}")
    
    # Визуализация матрицы корреляций
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        plt.figure(figsize=(12, 8))
        corr_matrix = df[numeric_columns].corr()
        
        # Создаем маску для верхнего треугольника
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Матрица корреляций числовых признаков', fontsize=16)
        plt.tight_layout()
        plt.savefig('reports/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Анализ распределений числовых признаков
    print(f"\n📈 АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ:")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_data = df[numeric_columns].describe()
    print(numeric_data)
    
    # Визуализация распределений ключевых признаков
    key_features = list(high_corr_features.index[:6]) if len(high_corr_features) > 0 else numeric_columns[:6]
    
    if len(key_features) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(key_features[:6]):
            if i < len(axes):
                df[col].hist(bins=50, alpha=0.7, color='skyblue', edgecolor='black', ax=axes[i])
                axes[i].set_title(f'{col}\n(корр: {top_correlations.get(col, 0):.3f})')
                axes[i].set_xlabel(col)
        
        plt.tight_layout()
        plt.savefig('reports/key_features_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Анализ категориальных признаков
    print(f"\n📊 АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ:")
    categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']
    available_categorical = [col for col in categorical_columns if col in df.columns]
    
    for col in available_categorical:
        if col in df.columns:
            value_counts = df[col].value_counts().sort_index()
            print(f"\n{col}:")
            for val, count in value_counts.items():
                print(f"  {val}: {count} ({count/len(df)*100:.1f}%)")
    
    # Обнаружение проблем в данных
    print(f"\n🔍 ДЕТЕКТИРОВАНИЕ ПРОБЛЕМ ДАННЫХ:")
    data_issues = detect_data_issues(df)
    if len(data_issues) > 0:
        for issue in data_issues:
            print(f"  ⚠️ {issue}")
    else:
        print("  ✅ Серьезных проблем не обнаружено")
    
    # Статистический анализ признаков
    print(f"\n📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ ПРИЗНАКОВ:")
    feature_analysis = analyze_feature_distributions(df, target_column)
    significant_features = [feature for feature, info in feature_analysis.items() if info['significant']]
    
    print(f"Признаков со значимыми различиями между классами: {len(significant_features)}")
    if len(significant_features) > 0:
        print("Самые значимые признаки:")
        for feature in significant_features[:5]:
            info = feature_analysis[feature]
            print(f"  {feature}: p-value={info['p_value']:.6f}")
    
    # Сбор результатов для автоматических выводов
    eda_results = {
        'dataset_shape': df.shape,
        'target_distribution': target_counts.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'top_correlations': top_correlations.to_dict(),
        'high_corr_features': high_corr_features.to_dict(),
        'medium_corr_features': medium_corr_features.to_dict(),
        'data_issues': data_issues,
        'feature_analysis': feature_analysis,
        'numeric_summary': numeric_data.to_dict()
    }
    
    # Генерация автоматических выводов
    print(f"\n🎯 АВТОМАТИЧЕСКИЕ ВЫВОДЫ НА ОСНОВЕ ДАННЫХ:")
    print("=" * 50)
    
    insights = generate_insights(eda_results)
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Детальные рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ ДЛЯ МОДЕЛИРОВАНИЯ:")
    if imbalance_ratio < 0.3:
        print("  • Используйте class_weight='balanced' в моделях")
        print("  • Рассмотрите SMOTE или другие oversampling методы")
        print("  • Используйте метрики: ROC-AUC, Precision-Recall")
    
    if len(high_corr_features) > 0:
        print("  • Включите высококоррелированные признаки в модель")
        print("  • Рассмотрите взаимодействия между ключевыми признаками")
    
    if len(data_issues) > 0:
        print("  • Выполните предобработку данных перед обучением")
        print("  • Обработайте выбросы и пропущенные значения")
    
    # Сохранение отчета EDA
    import json
    os.makedirs('reports', exist_ok=True)
    with open('reports/eda_report.json', 'w', encoding='utf-8') as f:
        json.dump(eda_results, f, indent=2, default=str)
    
    # Сохранение выводов
    with open('reports/eda_insights.txt', 'w', encoding='utf-8') as f:
        f.write("АВТОМАТИЧЕСКИЕ ВЫВОДЫ EDA\n")
        f.write("=" * 40 + "\n\n")
        for insight in insights:
            f.write(f"• {insight}\n")
    
    print(f"\n✅ EDA АНАЛИЗ ЗАВЕРШЕН!")
    print(f"📁 Отчеты сохранены в папке reports/")
    print(f"   - eda_report.json (детальный анализ)")
    print(f"   - eda_insights.txt (автоматические выводы)")
    print(f"   - *.png (визуализации)")
    
    return eda_results

if __name__ == "__main__":
    import os
    run_eda()