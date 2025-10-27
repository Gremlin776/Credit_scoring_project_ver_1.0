#!/usr/bin/env python3
"""
LAUNCH.PY - Основной скрипт запуска проекта
"""

import os
import sys
import subprocess
import logging
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

def main():
    """Основная функция запуска"""
    logger.info("=" * 50)
    logger.info("ЗАПУСК ПРОЕКТА CREDIT SCORING")
    logger.info("=" * 50)
    
    # Проверка окружения
    if sys.prefix == sys.base_prefix:
        logger.error("Виртуальное окружение не активировано")
        return
    
    logger.info("Виртуальное окружение активировано")
    
    # Создание структуры директорий
    create_directories()
    
    # Проверка данных
    if Path('data/raw/UCI_Credit_Card.csv').exists():
        logger.info("Данные найдены")
    else:
        logger.warning("Данные не найдены в data/raw/UCI_Credit_Card.csv")
    
    # Запуск скриптов с продолжением при ошибках
    scripts_to_run = [
        ("notebooks/01_eda.py", "EDA анализ"),
        ("src/data/make_dataset.py", "Подготовка данных"),
        ("src/features/build_features.py", "Feature Engineering"),
        ("src/models/train_model.py", "Обучение моделей"),
        ("src/models/train_pipeline.py", "Обучение пайплайна")
    ]
    
    for script, description in scripts_to_run:
        success = run_script(script, description)
        if not success:
            logger.warning(f"Пропускаем {description} из-за ошибок")
            continue
    
    # Запуск сервисов
    logger.info("Запуск MLflow и FastAPI...")
    try:
        # MLflow
        mlflow_process = subprocess.Popen([
            sys.executable, "-m", "mlflow", "ui", 
            "--backend-store-uri", "mlruns",
            "--host", "127.0.0.1", 
            "--port", "5000"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # FastAPI
        api_process = subprocess.Popen([
            sys.executable, "src/api/app.py"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        logger.info("MLflow запущен: http://127.0.0.1:5000")
        logger.info("FastAPI запущен: http://127.0.0.1:8000")
        logger.info("Для остановки нажмите Ctrl+C")
        
        mlflow_process.wait()
        api_process.wait()
        
    except KeyboardInterrupt:
        logger.info("Остановка проекта...")
        mlflow_process.terminate()
        api_process.terminate()
    except Exception as e:
        logger.error(f"Ошибка запуска сервисов: {e}")
    
    logger.info("Завершение работы launch.py")

if __name__ == "__main__":
    main()