#!/usr/bin/env python3
"""
Скрипт обучения моделей с MLflow tracking
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold 
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import json

# Исправляем импорт - абсолютный путь
try:
    from src.models.pipeline import create_model_pipeline, get_feature_names
except ImportError:
    # Альтернативный вариант импорта
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.pipeline import create_model_pipeline, get_feature_names

def load_data_with_features():
    """Загрузка данных с фичами"""
    try:
        df = pd.read_csv('data/processed/data_with_features.csv')
        
        # Целевая переменная
        y = df['default']
        
        # Признаки - исключаем целевую и временные колонки
        exclude_columns = ['default', 'AGE_GROUP']
        X = df.drop(columns=exclude_columns, errors='ignore')
        
        print(f"Данные загружены: {X.shape}")
        return X, y
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return None, None

def hyperparameter_tuning(pipeline, model_type, X_train, y_train):
    """Автоматический подбор гиперпараметров с помощью GridSearchCV"""
    print(f"GridSearchCV для {model_type}...")
    
    param_grids = {
        'logistic': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__class_weight': ['balanced']
        },
        'random_forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 15],
            'classifier__class_weight': ['balanced']
        },
        'gradient_boosting': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.1, 0.2],
            'classifier__max_depth': [3, 5]
        }
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    if model_type in param_grids:
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[model_type],
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Лучшие параметры для {model_type}: {grid_search.best_params_}")
        print(f"Лучший ROC-AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    else:
        print(f"Параметры для {model_type} не найдены, используется модель по умолчанию")
        pipeline.fit(X_train, y_train)
        return pipeline, {}

def evaluate_model(model, X_test, y_test, feature_names=None):
    """Расширенная оценка модели с визуализациями"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Базовые метрики
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'accuracy': (y_pred == y_test).mean()
    }
    
    roc_curve_path = None
    
    try:
        # Создаем папку reports если не существует
        os.makedirs('reports', exist_ok=True)
        
        # ROC curve
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Feature importance (если доступно)
        plt.subplot(1, 3, 2)
        if hasattr(model.named_steps['classifier'], 'feature_importances_') and feature_names:
            importances = model.named_steps['classifier'].feature_importances_
            indices = np.argsort(importances)[::-1][:10]  # Топ-10 фич
            
            plt.barh(range(len(indices)), importances[indices], color='skyblue')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importances')
            plt.gca().invert_yaxis()
        
        # Distribution of predictions
        plt.subplot(1, 3, 3)
        plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='No Default', color='green')
        plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Default', color='red')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predictions')
        plt.legend()
        
        plt.tight_layout()
        roc_curve_path = 'reports/model_evaluation.png'
        plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Предупреждение: не удалось создать графики: {e}")
        roc_curve_path = None
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return metrics, roc_curve_path, class_report

def safe_mlflow_log_artifact(file_path):
    """Безопасное логирование артефактов в MLflow"""
    try:
        if file_path and os.path.exists(file_path):
            mlflow.log_artifact(file_path)
            return True
        else:
            print(f"Предупреждение: файл не существует: {file_path}")
            return False
    except Exception as e:
        print(f"Предупреждение: не удалось залогировать артефакт {file_path}: {e}")
        return False

def train_experiment():
    """Проведение экспериментов с учетом EDA insights и GridSearchCV"""
    # Загрузка данных
    X, y = load_data_with_features()
    if X is None:
        return
    
    # Стратифицированное разделение с учетом дисбаланса
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Default rate in train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")
    
    # Эксперименты с GridSearchCV
    experiments = [
        {
            'name': 'Logistic Regression - GridSearch',
            'model_type': 'logistic',
            'use_grid_search': True,
            'base_params': {'random_state': 42, 'max_iter': 1000}
        },
        {
            'name': 'Random Forest - GridSearch', 
            'model_type': 'random_forest',
            'use_grid_search': True,
            'base_params': {'random_state': 42}
        },
        {
            'name': 'Gradient Boosting - GridSearch',
            'model_type': 'gradient_boosting', 
            'use_grid_search': True,
            'base_params': {'random_state': 42}
        },
        {
            'name': 'Logistic Regression - Balanced',
            'model_type': 'logistic',
            'use_grid_search': False,
            'base_params': {'C': 0.1, 'class_weight': 'balanced', 'random_state': 42, 'max_iter': 1000}
        },
        {
            'name': 'Random Forest - Default',
            'model_type': 'random_forest',
            'use_grid_search': False, 
            'base_params': {'n_estimators': 100, 'random_state': 42}
        }
    ]
    
    best_score = 0
    best_model = None
    best_experiment = None
    
    # Настройка MLflow
    try:
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("Credit Scoring Experiments")
        mlflow_available = True
    except Exception as e:
        print(f"Предупреждение: MLflow недоступен: {e}")
        mlflow_available = False
    
    for exp in experiments:
        # Используем контекст MLflow только если он доступен
        if mlflow_available:
            mlflow_context = mlflow.start_run(run_name=exp['name'])
            mlflow_context.__enter__()
        else:
            mlflow_context = None
        
        try:
            print(f"\nЭксперимент: {exp['name']}")
            
            # Создание базового пайплайна
            pipeline = create_model_pipeline(exp['model_type'], **exp['base_params'])
            
            # Подбор гиперпараметров или обучение с базовыми параметрами
            if exp['use_grid_search']:
                model, best_params = hyperparameter_tuning(
                    pipeline, exp['model_type'], X_train, y_train
                )
                # Логирование лучших параметров
                if mlflow_available:
                    for param, value in best_params.items():
                        try:
                            mlflow.log_param(param, value)
                        except Exception as e:
                            print(f"Предупреждение: не удалось залогировать параметр {param}: {e}")
            else:
                model = pipeline
                model.fit(X_train, y_train)
                # Логирование базовых параметров
                if mlflow_available:
                    for param, value in exp['base_params'].items():
                        try:
                            mlflow.log_param(param, value)
                        except Exception as e:
                            print(f"Предупреждение: не удалось залогировать параметр {param}: {e}")
            
            # Логирование типа модели
            if mlflow_available:
                try:
                    mlflow.log_param('model_type', exp['model_type'])
                    mlflow.log_param('use_grid_search', exp['use_grid_search'])
                except Exception as e:
                    print(f"Предупреждение: не удалось залогировать параметры модели: {e}")
            
            # Получение имен фич для анализа
            try:
                feature_names = get_feature_names(model, X_train.columns)
            except:
                feature_names = None
            
            # Оценка модели
            metrics, roc_curve_path, class_report = evaluate_model(
                model, X_test, y_test, feature_names
            )
            
            # Логирование метрик
            if mlflow_available:
                for metric_name, metric_value in metrics.items():
                    try:
                        mlflow.log_metric(metric_name, metric_value)
                    except Exception as e:
                        print(f"Предупреждение: не удалось залогировать метрику {metric_name}: {e}")
            
            print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1-Score: {metrics['f1']:.4f}")
            
            # Логирование артефактов
            if mlflow_available:
                safe_mlflow_log_artifact(roc_curve_path)
                
                # Логирование classification report
                try:
                    with open('reports/classification_report.json', 'w') as f:
                        json.dump(class_report, f, indent=2)
                    safe_mlflow_log_artifact('reports/classification_report.json')
                except Exception as e:
                    print(f"Предупреждение: не удалось сохранить classification report: {e}")
                
                # Логирование модели
                try:
                    mlflow.sklearn.log_model(model, "model")
                except Exception as e:
                    print(f"Предупреждение: не удалось залогировать модель: {e}")
            
            print(f"{exp['name']} - ROC AUC: {metrics['roc_auc']:.4f}")
            
            # Сохранение лучшей модели
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = model
                best_experiment = exp['name']
                
        finally:
            # Всегда закрываем контекст MLflow
            if mlflow_context:
                mlflow_context.__exit__(None, None, None)
    
    # Сохранение лучшей модели
    if best_model is not None:
        try:
            os.makedirs('models', exist_ok=True)
            mlflow.sklearn.save_model(best_model, "models/best_model")
            
            # Сохранение информации о лучшей модели
            model_info = {
                'best_experiment': best_experiment,
                'best_score': best_score,
                'model_type': best_model.named_steps['classifier'].__class__.__name__,
                'feature_importance': None
            }
            
            # Если есть feature importance, сохраняем
            if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
                try:
                    feature_names = get_feature_names(best_model, X_train.columns)
                    importances = best_model.named_steps['classifier'].feature_importances_
                    feature_importance = dict(zip(feature_names, importances))
                    model_info['feature_importance'] = feature_importance
                    
                    # Визуализация важности признаков
                    plt.figure(figsize=(10, 8))
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
                    features, importance_vals = zip(*top_features)
                    
                    plt.barh(range(len(features)), importance_vals, color='lightblue')
                    plt.yticks(range(len(features)), features)
                    plt.xlabel('Feature Importance')
                    plt.title('Top 15 Feature Importances - Best Model')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig('reports/best_model_feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    print(f"Предупреждение: ошибка получения feature importance: {e}")
            
            with open('reports/best_model_info.json', 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"\nЛУЧШАЯ МОДЕЛЬ")
            print(f"   Эксперимент: {best_experiment}")
            print(f"   ROC AUC: {best_score:.4f}")
            print(f"   Тип модели: {model_info['model_type']}")
            print(f"   Сохранена в: models/best_model")
            
        except Exception as e:
            print(f"Предупреждение: не удалось сохранить лучшую модель: {e}")

if __name__ == "__main__":
    train_experiment()