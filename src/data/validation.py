import great_expectations as ge
import pandas as pd
import numpy as np
from great_expectations.core.expectation_configuration import ExpectationConfiguration
import json
import os
import sys
import io

def create_validation_suite():
    """Создает расширенный набор правил для валидации данных"""
    context = ge.get_context()
    
    suite = context.create_expectation_suite(
        "credit_data_suite", 
        overwrite_existing=True
    )
    
    # Проверки структуры данных
    expectations = [
        # Структурные проверки
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={
                "column_list": [
                    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                    'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
                    'PAY_AMT5', 'PAY_AMT6',
                    'default'
                ]
            }
        ),
        
        # Проверки на отсутствие null значений в ключевых полях
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "LIMIT_BAL"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null", 
            kwargs={"column": "default"}
        ),
        
        # Проверки категориальных признаков
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "SEX", "value_set": [1, 2]}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "EDUCATION", "value_set": [1, 2, 3, 4]}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "MARRIAGE", "value_set": [1, 2, 3]}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "default", "value_set": [0, 1]}
        ),
        
        # Проверки диапазонов значений
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "AGE", "min_value": 18, "max_value": 100}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "LIMIT_BAL", "min_value": 0, "max_value": 1000000}
        ),
    ]
    
    for expectation in expectations:
        suite.add_expectation(expectation)
    
    context.save_expectation_suite(suite)
    return suite

def validate_data(df: pd.DataFrame) -> bool:
    """Валидирует данные с помощью Great Expectations"""
    suite = create_validation_suite()
    results = ge.from_pandas(df).validate(expectation_suite=suite)
    
    # Сохранение результатов валидации
    validation_results = {
        'success': results.success,
        'statistics': {
            'evaluated_expectations': results.statistics['evaluated_expectations'],
            'successful_expectations': results.statistics['successful_expectations'],
            'unsuccessful_expectations': results.statistics['unsuccessful_expectations'],
            'success_percent': results.statistics['success_percent']
        },
        'failed_expectations': []
    }
    
    if not results.success:
        print("Валидация данных провалена:")
        for result in results.results:
            if not result.success:
                failed_expectation = {
                    'expectation_type': result.expectation_config.expectation_type,
                    'column': result.expectation_config.kwargs.get('column', 'N/A'),
                    'details': str(result.result)
                }
                validation_results['failed_expectations'].append(failed_expectation)
                print(f"   - {result.expectation_config.expectation_type} для колонки {failed_expectation['column']}")
    else:
        print("Валидация данных пройдена успешно!")
    
    # Сохранение отчета о валидации
    os.makedirs('reports', exist_ok=True)
    with open('reports/validation_report.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    return results.success

def main():
    """Основная функция для тестирования валидации"""
    try:
        # Загрузка данных
        df = pd.read_csv('data/processed/processed_data.csv')
        result = validate_data(df)
        
        if result:
            print("Все проверки пройдены!")
        else:
            print("Требуется исправление данных")
            
    except Exception as e:
        print(f"Ошибка валидации: {e}")

if __name__ == "__main__":
    main()