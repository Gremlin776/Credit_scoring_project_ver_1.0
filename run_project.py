#!/usr/bin/env python3
"""
Главный скрипт для автоматического запуска проекта Credit Scoring
С автоматическим определением виртуального окружения
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse
from pathlib import Path

class ProjectRunner:
    def __init__(self, ci_mode=False):
        self.project_root = Path(__file__).parent
        self.steps_completed = []
        self.ci_mode = ci_mode
        self.is_github_actions = os.getenv('GITHUB_ACTIONS') == 'true'
        
        # Автоматическое определение виртуального окружения
        self.in_venv = sys.prefix != sys.base_prefix
        self.venv_path = Path(sys.prefix)
        
    def log(self, message, emoji="🔍"):
        print(f"{emoji} {message}")
    
    def check_environment(self):
        """Проверяет и настраивает окружение"""
        self.log("Проверка окружения...")
        
        if not self.in_venv and not self.ci_mode:
            self.log("Виртуальное окружение не активировано!", "⚠️")
            self.log("Запуск автоматической настройки...", "🔄")
            
            # Запускаем настройку окружения
            setup_script = self.project_root / 'setup_environment.py'
            if setup_script.exists():
                try:
                    result = subprocess.run([
                        sys.executable, str(setup_script)
                    ], cwd=self.project_root)
                    
                    if result.returncode == 0:
                        self.log("Настройка завершена успешно!", "✅")
                        self.log("Пожалуйста, активируйте виртуальное окружение и запустите снова", "💡")
                        
                        # Показываем инструкции
                        if os.name == 'nt':  # Windows
                            self.log("Запустите: activate_env.bat")
                        else:  # Linux/Mac
                            self.log("Запустите: source activate_env.sh")
                            
                        return False
                    else:
                        self.log("Настройка не удалась", "❌")
                        return False
                except Exception as e:
                    self.log(f"Ошибка настройки: {e}", "❌")
                    return False
            else:
                self.log("Файл настройки не найден", "❌")
                return False
        else:
            if self.in_venv:
                self.log("Виртуальное окружение активировано", "✅")
            else:
                self.log("CI режим - проверка окружения пропущена", "ℹ️")
            
            return True
    
    def run_command(self, command, description, check=True, timeout=300):
        """Запускает команду"""
        self.log(description)
        
        try:
            if isinstance(command, list):
                result = subprocess.run(command, capture_output=True, text=True, 
                                      cwd=self.project_root, timeout=timeout)
            else:
                result = subprocess.run(command, shell=True, capture_output=True, 
                                      text=True, cwd=self.project_root, timeout=timeout)
            
            if result.returncode == 0 or not check:
                self.log("Успешно", "✅")
                if result.stdout and not self.ci_mode:
                    print(f"   Вывод: {result.stdout[:200]}...")
                self.steps_completed.append(description)
                return True
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                self.log(f"Ошибка: {error_msg[:200]}...", "❌")
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
    
    def create_sample_data(self):
        """Создает sample данные"""
        if not list((self.project_root / 'data/raw').glob('*.csv')):
            return self.run_command(
                [sys.executable, 'scripts/create_sample_data.py'],
                "Создание sample данных"
            )
        else:
            self.log("Данные уже существуют", "✅")
            return True
    
    def run_eda(self):
        """Запускает EDA анализ"""
        return self.run_command(
            [sys.executable, 'notebooks/01_eda.py'],
            "Запуск EDA анализа"
        )
    
    def run_training(self):
        """Запускает обучение моделей"""
        if self.ci_mode:
            return self.run_command(
                [sys.executable, '-c', """
import sys
sys.path.append('src')
try:
    from models.pipeline import create_model_pipeline
    print('✅ Пайплайн создан успешно')
except Exception as e:
    print(f'❌ Ошибка: {e}')
    sys.exit(1)
"""],
                "Тестирование пайплайна обучения"
            )
        
        return self.run_command(
            [sys.executable, 'src/models/train.py'],
            "Обучение моделей с MLflow"
        )
    
    def run_tests(self):
        """Запускает тесты"""
        return self.run_command(
            [sys.executable, '-m', 'pytest', 'tests/', '-v'],
            "Запуск unit-тестов",
            check=False
        )
    
    def start_services(self):
        """Запускает сервисы"""
        if self.ci_mode:
            return True
            
        # Запуск MLflow UI
        try:
            mlflow_process = subprocess.Popen(
                [sys.executable, '-m', 'mlflow', 'ui', '--port', '5000'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.project_root
            )
            time.sleep(3)
            self.log("MLflow UI запущен на http://localhost:5000", "✅")
        except Exception as e:
            self.log(f"Ошибка запуска MLflow UI: {e}", "❌")
            mlflow_process = None
        
        # Запуск FastAPI
        try:
            api_process = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'src.api.app:app', '--host', '0.0.0.0', '--port', '8000'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.project_root
            )
            time.sleep(5)
            self.log("FastAPI запущен на http://localhost:8000", "✅")
        except Exception as e:
            self.log(f"Ошибка запуска FastAPI: {e}", "❌")
            api_process = None
        
        # Открываем браузеры
        if not self.ci_mode:
            urls = [
                ("MLflow Experiments", "http://localhost:5000"),
                ("FastAPI Documentation", "http://localhost:8000/docs"),
            ]
            
            for name, url in urls:
                try:
                    webbrowser.open(url)
                    self.log(f"Открыто: {name}", "✅")
                    time.sleep(1)
                except Exception as e:
                    self.log(f"Не удалось открыть {url}: {e}", "❌")
        
        return mlflow_process, api_process
    
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
            self.create_sample_data,
            self.run_eda,
            self.run_training,
            self.run_tests,
        ]
        
        for step in steps:
            if not step():
                self.log(f"Прервано на шаге: {step.__name__}", "❌")
                return
        
        # Запуск сервисов (только в локальном режиме)
        if not self.ci_mode:
            mlflow_process, api_process = self.start_services()
            
            if api_process:
                try:
                    self.log("Сервисы работают... Нажмите Ctrl+C для остановки", "⏳")
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.log("Остановка сервисов...", "🛑")
                    if mlflow_process:
                        mlflow_process.terminate()
                    if api_process:
                        api_process.terminate()
                    self.log("Все сервисы остановлены", "✅")
        
        self.log("Проект успешно завершен!", "🎉")

def main():
    parser = argparse.ArgumentParser(description='Запуск Credit Scoring проекта')
    parser.add_argument('--ci-mode', action='store_true', help='CI режим')
    
    args = parser.parse_args()
    ci_mode = args.ci_mode or os.getenv('GITHUB_ACTIONS') == 'true'
    
    runner = ProjectRunner(ci_mode=ci_mode)
    runner.run()

if __name__ == "__main__":
    main()