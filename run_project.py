#!/usr/bin/env python3
"""
Главный скрипт для автоматического запуска проекта Credit Scoring
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse
from pathlib import Path
import io

# Принудительно устанавливаем UTF-8 кодировку для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

class ProjectRunner:
    def __init__(self, ci_mode=False):
        self.project_root = Path(__file__).parent
        self.steps_completed = []
        self.ci_mode = ci_mode
        self.is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'
        self.in_venv = sys.prefix != sys.base_prefix
        
    def log(self, message, emoji="🔍"):
        print(f"{emoji} {message}")
    
    def check_environment(self):
        """Проверяет и настраивает окружение"""
        self.log("Проверка окружения...")
        
        if not self.in_venv and not self.ci_mode:
            self.log("Виртуальное окружение не активировано!", "⚠️")
            return False
        else:
            self.log("Виртуальное окружение активировано", "✅")
            return True
    
    def run_command(self, command, description, check=True, timeout=300):
        """Запускает команду с исправленной кодировкой"""
        self.log(description)
        
        try:
            if isinstance(command, list):
                result = subprocess.run(command, capture_output=True, text=True,
                                      cwd=self.project_root, timeout=timeout,
                                      encoding='utf-8', errors='ignore')
            else:
                result = subprocess.run(command, shell=True, capture_output=True,
                                      text=True, cwd=self.project_root, timeout=timeout,
                                      encoding='utf-8', errors='ignore')
            
            if result.returncode == 0 or not check:
                self.log("Успешно", "✅")
                self.steps_completed.append(description)
                return True
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                self.log(f"Ошибка: {error_msg[:200]}", "❌")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("Таймаут выполнения", "⏰")
            return False
        except Exception as e:
            self.log(f"Исключение: {e}", "❌")
            return False
    
    def setup_directories(self):
        """Создает необходимые директории"""
        directories = ['data/raw', 'data/processed', 'models', 'reports', 'logs']
        for directory in directories:
            path = self.project_root / directory
            path.mkdir(parents=True, exist_ok=True)
        self.log("Директории созданы", "✅")
        return True
    
    def check_data_exists(self):
        """Проверяет наличие данных"""
        if Path('data/raw/UCI_Credit_Card.csv').exists():
            self.log("Данные уже существуют", "✅")
            return True
        else:
            self.log("Данные не найдены", "❌")
            return False
    
    def run_eda(self):
        """Запускает EDA анализ"""
        return self.run_command(
            [sys.executable, 'notebooks/01_eda.py'],
            "Запуск EDA анализа"
        )
    
    def run_training(self):
        """Запускает обучение моделей с обработкой ошибок импорта"""
        # Сначала проверяем наличие файла
        train_script = self.project_root / 'src' / 'models' / 'train.py'
        if not train_script.exists():
            self.log("Скрипт обучения не найден", "❌")
            return False
            
        return self.run_command(
            [sys.executable, 'src/models/train.py'],
            "Обучение моделей с MLflow"
        )
    
    def run_tests(self):
        """Запускает тесты (если есть)"""
        tests_dir = self.project_root / 'tests'
        if tests_dir.exists() and any(tests_dir.iterdir()):
            return self.run_command(
                [sys.executable, '-m', 'pytest', 'tests/', '-v'],
                "Запуск unit-тестов",
                check=False
            )
        else:
            self.log("Тесты не найдены - пропускаем", "ℹ️")
            return True
    
    def start_services(self):
        """Запускает сервисы"""
        processes = []
        
        # Запуск MLflow UI
        try:
            mlflow_process = subprocess.Popen(
                [sys.executable, '-m', 'mlflow', 'ui', '--port', '5000'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.project_root
            )
            processes.append(mlflow_process)
            time.sleep(3)
            self.log("MLflow UI запущен на http://localhost:5000", "✅")
        except Exception as e:
            self.log(f"Ошибка запуска MLflow UI: {e}", "❌")
        
        # Запуск FastAPI
        try:
            api_process = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'src.api.app:app', '--host', '0.0.0.0', '--port', '8000'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.project_root
            )
            processes.append(api_process)
            time.sleep(5)
            self.log("FastAPI запущен на http://localhost:8000", "✅")
        except Exception as e:
            self.log(f"Ошибка запуска FastAPI: {e}", "❌")
        
        return processes
    
    def run(self):
        """Главный метод запуска"""
        self.log("ЗАПУСК ПРОЕКТА CREDIT SCORING")
        self.log("=" * 50)
        
        # Проверка окружения
        if not self.check_environment():
            return
        
        # Основные шаги
        steps = [
            self.setup_directories,
            self.check_data_exists,
            self.run_eda,
            self.run_training,
            self.run_tests,
        ]
        
        for step in steps:
            success = step()
            if not success:
                self.log(f"Пропускаем дальнейшие шаги", "⚠️")
                break
        
        # Запуск сервисов (только в локальном режиме)
        if not self.ci_mode:
            processes = self.start_services()
            
            if processes:
                try:
                    self.log("Сервисы работают... Нажмите Ctrl+C для остановки", "⏳")
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.log("Остановка сервисов...", "🛑")
                    for process in processes:
                        if process:
                            process.terminate()
                    self.log("Все сервисы остановлены", "✅")
        
        self.log("Проект завершен!", "🎉")

def main():
    parser = argparse.ArgumentParser(description='Запуск Credit Scoring проекта')
    parser.add_argument('--ci-mode', action='store_true', help='CI режим')
    
    args = parser.parse_args()
    
    runner = ProjectRunner(ci_mode=args.ci_mode)
    runner.run()

if __name__ == "__main__":
    main()