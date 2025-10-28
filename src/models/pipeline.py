from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

def create_feature_selection_pipeline(k=20):
    """Создание пайплайна с feature selection на основе EDA"""
    
    # Признаки на основе EDA insights
    high_corr_features = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    payment_features = ['PAYMENT_MEAN', 'PAYMENT_MAX_DELAY', 'PAYMENT_DELAY_COUNT', 'PAYMENT_SEVERITY']
    bill_features = ['BILL_AMT_TO_LIMIT_RATIO', 'BILL_AMT_UTILIZATION']
    risk_features = ['RISK_SCORE_1', 'CRITICAL_DEBT']
    demographic_features = ['LIMIT_BAL_LOG', 'LIMIT_BAL_PER_AGE']
    
    numeric_features = [
        'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
        'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
        'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ] + high_corr_features + payment_features + bill_features + risk_features + demographic_features
    
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    # Препроцессинг на основе EDA
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Feature selection на основе EDA
    feature_selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_features) + 10))
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selector', feature_selector)
    ])
    
    return full_pipeline

def create_model_pipeline(model_type='logistic', **params):
    """Создание полного пайплайна с моделью"""
    
    preprocessor = create_feature_selection_pipeline()
    
    # Выбор модели на основе EDA экспериментов
    if model_type == 'logistic':
        model = LogisticRegression(**params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**params)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

def get_feature_names(pipeline, original_features):
    """Получение имен фич после преобразований"""
    try:
        preprocessor = pipeline.named_steps['preprocessor']
        feature_selector = pipeline.named_steps['feature_selector']
        
        # Получение имен после препроцессинга
        numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        all_features = list(numeric_features) + list(categorical_features)
        
        # Применение feature selection
        selected_mask = feature_selector.get_support()
        selected_features = [all_features[i] for i in range(len(selected_mask)) if selected_mask[i]]
        
        return selected_features
    except Exception as e:
        print(f"Ошибка получения имен фич: {e}")
        return None