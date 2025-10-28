# scripts/create_feature_importance.py
#!/usr/bin/env python3
"""
Скрипт для создания графика важности признаков
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import json
import pickle
import glob

# Добавляем путь для импортов
sys.path.append(str(Path(__file__).parent.parent))

def load_best_model():
    """Загрузка лучшей модели"""
    print(" Поиск модели для анализа важности признаков...")
    
    # Все возможные пути
    possible_paths = [
        "models/best_model",
        "models/best_model/model.pkl",
        "mlruns/0/*/artifacts/model",
        "mlruns/*/models/*/artifacts/model",
        "mlruns/254278752131644154/*/artifacts/model"
    ]
    
    # Расширяем пути
    expanded_paths = []
    for path_pattern in possible_paths:
        if '*' in path_pattern:
            matches = glob.glob(path_pattern)
            expanded_paths.extend(matches)
        else:
            expanded_paths.append(path_pattern)
    
    for path in expanded_paths:
        try:
            model_path = Path(path)
            if not model_path.exists():
                continue
                
            print(f" Пробуем загрузить из: {path}")
            
            # Пробуем разные способы загрузки
            if model_path.is_dir() and (model_path / "MLmodel").exists():
                import mlflow.sklearn
                model = mlflow.sklearn.load_model(str(model_path))
                print(f" MLflow модель загружена из {path}")
                return model
                
            elif model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print(f" Pickle модель загружена из {path}")
                return model
                
            elif model_path.is_dir() and (model_path / "model.pkl").exists():
                with open(model_path / "model.pkl", 'rb') as f:
                    model = pickle.load(f)
                print(f" Модель загружена из {path}/model.pkl")
                return model
                
        except Exception as e:
            print(f" Ошибка загрузки из {path}: {e}")
            continue
    
    print(" Не удалось загрузить модель для анализа")
    return None

def create_feature_importance_plot():
    """Создание графика важности признаков"""
    model = load_best_model()
    if model is None:
        print(" Не удалось загрузить модель для анализа")
        return False
    
    try:
        # Проверяем есть ли feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        else:
            print(" У модели нет атрибута feature_importances_")
            return False
        
        # Создаем generic имена признаков
        feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Создаем DataFrame с важностью признаков
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Создаем график
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        
        plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Важность признака')
        plt.title('Топ-15 наиболее важных признаков')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Сохраняем график
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/best_model_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Сохраняем данные в JSON
        importance_data = {
            'top_features': top_features.to_dict('records'),
            'total_features': len(importances),
            'max_importance': float(importances.max()),
            'min_importance': float(importances.min())
        }
        
        with open('reports/feature_importance.json', 'w', encoding='utf-8') as f:
            json.dump(importance_data, f, indent=2, ensure_ascii=False)
        
        print(" График важности признаков создан: reports/best_model_feature_importance.png")
        print(" Данные важности признаков сохранены: reports/feature_importance.json")
        
        return True
            
    except Exception as e:
        print(f" Ошибка создания графика важности признаков: {e}")
        return False

def main():
    """Основная функция"""
    print(" Создание графика важности признаков...")
    success = create_feature_importance_plot()
    
    if success:
        print(" График важности признаков успешно создан!")
    else:
        print(" Не удалось создать график важности признаков")

if __name__ == "__main__":
    main()