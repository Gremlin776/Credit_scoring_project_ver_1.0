#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
import io

# Принудительно устанавливаем UTF-8 кодировку для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/launch.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Создание структуры директорий проекта"""
    logger.info("Создание структуры проекта...")
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'reports',
        'logs',
        'notebooks',
        'scripts',
        'src/data',
        'src/features',
        'src/models',
        'src/api',
        'src/monitoring',
        'tests',
        '.github/workflows'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("Структура проекта создана!")

def run_script(script_name, description):
    """Запуск скрипта с обработкой ошибок и логированием"""
    logger.info(f"Запуск {description}...")
    
    if not Path(script_name).exists():
        logger.warning(f"Скрипт {script_name} не найден")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=300
        )
        
        if result.returncode == 0:
            logger.info(f"{description} завершен успешно")
            # Логируем успешный вывод
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-5:]:  # Последние 5 строк
                    if line.strip():
                        logger.debug(f"{description}: {line.strip()}")
            return True
        else:
            logger.error(f"{description} завершен с ошибкой (код: {result.returncode})")
            # Логируем ошибки
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[:10]:  # Первые 10 строк ошибки
                    if line.strip():
                        logger.error(f"{description} ошибка: {line.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"{description} - превышено время выполнения")
        return False
    except Exception as e:
        logger.error(f"{description} - исключение: {e}")
        return False

def run_tests():
    """Запуск unit-тестов"""
    logger.info("Запуск unit-тестов...")
    
    try:
        # Используем sys.executable для гарантии запуска из venv
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/', '-v'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("Все тесты прошли успешно!")
            return True
        else:
            logger.warning("Некоторые тесты не прошли")
            return False
            
    except Exception as e:
        logger.error(f"Ошибка запуска тестов: {e}")
        return False

def run_linting():
    """Запуск линтинга кода"""
    logger.info("Проверка стиля кода...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'flake8', 'src/', 'tests/', '--count', '--max-line-length=100', '--exit-zero'
        ], capture_output=True, text=True, timeout=120)
        
        # flake8 всегда возвращает код ошибки если есть проблемы, используем --exit-zero
        error_count = len(result.stdout.strip().split('\n')) if result.stdout else 0
        if error_count > 0:
            logger.warning(f"Flake8: найдено проблем - {error_count}")
        else:
            logger.info("Flake8 проверка пройдена")
        return True
            
    except Exception as e:
        logger.error(f"Ошибка линтинга: {e}")
        return False

def run_drift_monitoring():
    """Запуск мониторинга дрифта"""
    logger.info(" Запуск мониторинга дрифта...")
    
    try:
        # Проверяем существование скрипта мониторинга
        if not Path('scripts/monitor_drift.py').exists():
            logger.warning("Скрипт мониторинга дрифта не найден")
            return None
            
        # Запускаем мониторинг дрифта
        result = subprocess.run([
            sys.executable, 'scripts/monitor_drift.py'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(" Мониторинг дрифта завершен успешно")
            # Ищем информацию о дрифте в выводе
            output_lines = result.stdout.split('\n')
            drift_info = [line for line in output_lines if 'дрифт' in line.lower() or 'drift' in line.lower()]
            for info in drift_info[:3]:  # Первые 3 строки с информацией о дрифте
                logger.info(f"Мониторинг: {info}")
            return True
        else:
            logger.warning(" Мониторинг дрифта завершен с предупреждениями")
            return False
            
    except Exception as e:
        logger.error(f" Ошибка мониторинга дрифта: {e}")
        return None

def setup_monitoring():
    """Настройка системы мониторинга"""
    logger.info(" Настройка мониторинга...")
    
    try:
        # Создаем базовые скрипты мониторинга если их нет
        if not Path('scripts/monitor_drift.py').exists():
            logger.info("Создание базового скрипта мониторинга...")
            Path('scripts').mkdir(exist_ok=True)
            
            # Создаем простой скрипт мониторинга (исправлены кавычки)
            monitor_script = '''#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_drift_check():
    """Простая проверка дрифта"""
    try:
        # Проверяем наличие данных
        data_path = Path('data/processed/data_with_features.csv')
        if not data_path.exists():
            logger.warning("Данные для мониторинга не найдены")
            return {"status": "no_data"}
            
        df = pd.read_csv(data_path)
        logger.info(f"Проверка дрифта для {len(df)} записей")
        
        # Простая проверка статистик
        stats = {
            "records_count": len(df),
            "default_rate": df['default'].mean() if 'default' in df.columns else 0,
            "features_count": len(df.columns)
        }
        
        logger.info(f"Статистики: {stats}")
        return {"status": "checked", "stats": stats}
        
    except Exception as e:
        logger.error(f"Ошибка мониторинга: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    result = simple_drift_check()
    print(f"Результат мониторинга: {result}")
'''
            with open('scripts/monitor_drift.py', 'w', encoding='utf-8') as f:
                f.write(monitor_script)
            logger.info(" Базовый скрипт мониторинга создан")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка настройки мониторинга: {e}")
        return False

def setup_testing():
    """Настройка системы тестирования"""
    logger.info(" Настройка тестирования...")
    
    try:
        # Создаем базовые тесты если их нет
        if not Path('tests').exists():
            Path('tests').mkdir(exist_ok=True)
            
        # Создаем простой тестовый файл если нет тестов (исправлены кавычки)
        if not any(Path('tests').glob('test_*.py')):
            logger.info("Создание базовых тестов...")
            
            basic_test = '''#!/usr/bin/env python3

import pytest
import pandas as pd
import os

def test_data_exists():
    """Тест наличия данных"""
    assert os.path.exists('data/raw/UCI_Credit_Card.csv'), "Исходные данные не найдены"

def test_processed_data():
    """Тест обработанных данных"""
    if os.path.exists('data/processed/data_with_features.csv'):
        df = pd.read_csv('data/processed/data_with_features.csv')
        assert len(df) > 0, "Нет данных в processed data"
        assert 'default' in df.columns, "Нет целевой переменной"

def test_model_files():
    """Тест наличия моделей"""
    # Проверяем наличие любой модели
    model_files = [
        'models/best_model',
        'models/trained_model.pkl'
    ]
    model_exists = any(os.path.exists(f) for f in model_files)
    assert model_exists, "Модели не найдены"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
            with open('tests/test_basic.py', 'w', encoding='utf-8') as f:
                f.write(basic_test)
            logger.info(" Базовые тесты созданы")
        
        return True
    except Exception as e:
        logger.error(f"Ошибка настройки тестирования: {e}")
        return False

def main():
    """Основная функция запуска"""
    parser = argparse.ArgumentParser(description='Credit Scoring ML Pipeline')
    parser.add_argument('--mode', choices=['full', 'train', 'test', 'monitor', 'api', 'ci'], 
                       default='full', help='Режим запуска')
    parser.add_argument('--skip-tests', action='store_true', 
                       help='Пропустить тесты')
    parser.add_argument('--setup-only', action='store_true',
                       help='Только настройка без запуска')
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info(f"ЗАПУСК ПРОЕКТА CREDIT SCORING - Режим: {args.mode}")
    logger.info("=" * 50)
    
    # Проверка окружения
    if sys.prefix == sys.base_prefix:
        logger.error("Виртуальное окружение не активировано")
        return
    
    logger.info("Виртуальное окружение активировано")
    
    # Создание структуры директорий
    create_directories()
    
    # Настройка тестирования и мониторинга
    if args.mode in ['full', 'test', 'monitor', 'ci']:
        setup_testing()
        setup_monitoring()
    
    if args.setup_only:
        logger.info("Настройка завершена")
        return
    
    # Проверка данных
    if Path('data/raw/UCI_Credit_Card.csv').exists():
        logger.info("Данные найдены")
    else:
        logger.warning("Данные не найдены в data/raw/UCI_Credit_Card.csv")
    
    # РАЗНЫЕ РЕЖИМЫ ЗАПУСКА
    
    if args.mode == 'full':
        """Полный пайплайн с тестированием и мониторингом"""
        logger.info(" Режим: Полный пайплайн")
        
        # Тестирование (если не пропущено)
        if not args.skip_tests:
            tests_ok = run_tests()
            lint_ok = run_linting()
            if not (tests_ok and lint_ok):
                logger.warning(" Проблемы с тестами/линтингом, но продолжаем...")
        
        # Мониторинг дрифта
        drift_result = run_drift_monitoring()
        
        # Основной пайплайн
        scripts_to_run = [
            ("notebooks/01_eda.py", "EDA анализ"),
            ("src/data/make_dataset.py", "Подготовка данных"),
            ("src/features/build_features.py", "Feature Engineering"),
            ("src/models/train.py", "Обучение моделей"),
        ]
        
        for script, description in scripts_to_run:
            success = run_script(script, description)
            if not success:
                logger.warning(f"Пропускаем {description} из-за ошибок")
                continue
        
        # Запуск сервисов
        start_services()
        
    elif args.mode == 'train':
        """Только обучение"""
        logger.info(" Режим: Обучение моделей")
        scripts_to_run = [
            ("src/data/make_dataset.py", "Подготовка данных"),
            ("src/features/build_features.py", "Feature Engineering"),
            ("src/models/train.py", "Обучение моделей"),
        ]
        for script, description in scripts_to_run:
            run_script(script, description)
            
    elif args.mode == 'test':
        """Только тестирование"""
        logger.info(" Режим: Тестирование")
        if not args.skip_tests:
            run_tests()
            run_linting()
        
    elif args.mode == 'monitor':
        """Только мониторинг"""
        logger.info(" Режим: Мониторинг")
        run_drift_monitoring()
        
    elif args.mode == 'api':
        """Только API"""
        logger.info(" Режим: API сервер")
        start_services(api_only=True)
        
    elif args.mode == 'ci':
        """CI/CD режим"""
        logger.info(" Режим: CI/CD")
        tests_ok = run_tests()
        lint_ok = run_linting()
        if not (tests_ok and lint_ok):
            logger.error("CI/CD пайплайн не прошел")
            sys.exit(1)
        logger.info(" CI/CD пайплайн успешно завершен")
    
    logger.info("Завершение работы launch.py")

def start_services(api_only=False):
    """Запуск MLflow и FastAPI сервисов"""
    logger.info("Запуск сервисов...")
    try:
        processes = []
        
        if not api_only:
            # MLflow
            mlflow_process = subprocess.Popen([
                sys.executable, "-m", "mlflow", "ui", 
                "--backend-store-uri", "mlruns",
                "--host", "127.0.0.1", 
                "--port", "5000"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(mlflow_process)
            logger.info("MLflow запущен: http://127.0.0.1:5000")
        
        # FastAPI
        api_process = subprocess.Popen([
            sys.executable, "src/api/app.py"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(api_process)
        logger.info("FastAPI запущен: http://127.0.0.1:8000")
        logger.info(" Документация: http://127.0.0.1:8000/docs")
        logger.info(" Дашборд: http://127.0.0.1:8000/dashboard")
        
        logger.info("Для остановки нажмите Ctrl+C")
        
        # Ожидаем завершения процессов
        for process in processes:
            process.wait()
            
    except KeyboardInterrupt:
        logger.info("Остановка сервисов...")
        for process in processes:
            process.terminate()
    except Exception as e:
        logger.error(f"Ошибка запуска сервисов: {e}")

if __name__ == "__main__":
    main()