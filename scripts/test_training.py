import sys
import io
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_training_components():
    """Быстрый тест компонентов обучения для CI/CD"""
    try:
        # Тест импортов
        from src.models.pipeline import create_model_pipeline
        from src.features.build_features import FeatureEngineer
        from src.data.validation import validate_data
        
        # Тест создания пайплайнов
        for model_type in ['logistic', 'random_forest']:
            pipeline = create_model_pipeline(model_type, random_state=42)
            assert pipeline is not None
            print(f" {model_type} pipeline created")
        
        # Тест feature engineering
        import pandas as pd
        sample_data = pd.DataFrame({
            'LIMIT_BAL': [50000],
            'SEX': [1],
            'EDUCATION': [2],
            'MARRIAGE': [1],
            'AGE': [35],
            'PAY_0': [0],
            'PAY_2': [0],
            'PAY_3': [0]
        })
        
        feature_engineer = FeatureEngineer()
        result = feature_engineer.fit_transform(sample_data)
        assert result is not None
        print(" Feature engineering works")
        
        print(" All training components tested successfully")
        return True
        
    except Exception as e:
        print(f" Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_training_components()
    sys.exit(0 if success else 1)