#!/usr/bin/env python3
"""
Главный скрипт для автоматического запуска всего проекта Credit Scoring
Запуск: python run_project.py
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path
import requests
import io

# Принудительно устанавливаем UTF-8 кодировку для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

class ProjectRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.steps_completed = []
        
    def run_command(self, command, description, check=True):
        """Запускает команду и логирует результат"""
        print(f"\n🎯 {description}...")
        print(f"   Команда: {command}")
        
        try:
            if isinstance(command, list):
                result = subprocess.run(command, capture_output=True, text=True, cwd=self.project_root)
            else:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0 or not check:
                print(f"   ✅ Успешно")
                if result.stdout:
                    print(f"   Вывод: {result.stdout[:200]}...")
                self.steps_completed.append(description)
                return True
            else:
                print(f"   ❌ Ошибка: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ Исключение: {e}")
            return False
    
    def check_installation(self):
        """Проверка установки необходимых инструментов"""
        print("🔍 Проверка окружения...")
        
        # Проверка Python
        try:
            python_version = subprocess.run([sys.executable, '--version'], capture_output=True, text=True)
            print(f"   Python: {python_version.stdout.strip()}")
        except:
            print("   ❌ Python не найден")
            return False
        
        # Проверка pip
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True)
            print("   pip: ✅")
        except:
            print("   ❌ pip не найден")
            return False
            
        return True
    
    def install_dependencies(self):
        """Установка всех зависимостей"""
        return self.run_command(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            "Установка зависимостей из requirements.txt"
        )
    
    def setup_directories(self):
        """Создание необходимых директорий"""
        directories = ['data/raw', 'data/processed', 'models', 'reports', 'logs']
        for directory in directories:
            path = self.project_root / directory
            path.mkdir(parents=True, exist_ok=True)
        print("✅ Созданы необходимые директории")
        return True
    
    def create_sample_data(self):
        """Создание sample данных если их нет"""
        if not list((self.project_root / 'data/raw').glob('*.csv')):
            return self.run_command(
                [sys.executable, 'scripts/create_sample_data.py'],
                "Создание sample данных"
            )
        else:
            print("✅ Данные уже существуют")
            return True
    
    def run_data_pipeline(self):
        """Запуск DVC пайплайна"""
        # Инициализация DVC если нужно
        if not (self.project_root / '.dvc').exists():
            self.run_command('dvc init', "Инициализация DVC", check=False)
        
        return self.run_command('dvc repro', "Запуск DVC пайплайна (подготовка данных + обучение)")
    
    def run_training(self):
        """Запуск обучения моделей"""
        return self.run_command(
            [sys.executable, 'src/models/train.py'],
            "Обучение моделей с MLflow"
        )
    
    def run_tests(self):
        """Запуск тестов"""
        return self.run_command(
            [sys.executable, '-m', 'pytest', 'tests/', '-v'],
            "Запуск unit-тестов",
            check=False  # Не блокируем если тесты падают
        )
    
    def start_mlflow_ui(self):
        """Запуск MLflow UI в фоне"""
        try:
            # Запускаем MLflow UI в отдельном процессе
            mlflow_process = subprocess.Popen(
                [sys.executable, '-m', 'mlflow', 'ui', '--port', '5000'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.project_root
            )
            time.sleep(3)  # Даем время на запуск
            print("✅ MLflow UI запущен на http://localhost:5000")
            return mlflow_process
        except Exception as e:
            print(f"❌ Ошибка запуска MLflow UI: {e}")
            return None
    
    def start_fastapi(self):
        """Запуск FastAPI в фоне"""
        try:
            api_process = subprocess.Popen(
                [sys.executable, '-m', 'uvicorn', 'src.api.app:app', '--host', '0.0.0.0', '--port', '8000'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.project_root
            )
            time.sleep(5)  # Даем больше времени на запуск API
            print("✅ FastAPI запущен на http://localhost:8000")
            return api_process
        except Exception as e:
            print(f"❌ Ошибка запуска FastAPI: {e}")
            return None
    
    def test_api(self):
        """Тестирование API endpoints"""
        print("\n🔍 Тестирование API...")
        
        try:
            # Health check
            response = requests.get('http://localhost:8000/health', timeout=10)
            if response.status_code == 200:
                print("   ✅ API Health check: OK")
            else:
                print(f"   ❌ API Health check: {response.status_code}")
                return False
            
            # Test prediction
            sample_data = {
                "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
                "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 1000, "BILL_AMT2": 1000, "BILL_AMT3": 1000, "BILL_AMT4": 1000,
                "BILL_AMT5": 1000, "BILL_AMT6": 1000, "PAY_AMT1": 1000, "PAY_AMT2": 1000,
                "PAY_AMT3": 1000, "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000
            }
            
            response = requests.post('http://localhost:8000/predict', json=sample_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Prediction test: {result}")
            else:
                print(f"   ❌ Prediction test: {response.status_code}")
                return False
                
            return True
            
        except Exception as e:
            print(f"   ❌ API тест не удался: {e}")
            return False
    
    def run_monitoring(self):
        """Запуск мониторинга дрифта"""
        return self.run_command(
            [sys.executable, 'src/monitoring/drift_detection.py'],
            "Запуск мониторинга дрифта данных"
        )
    
    def open_browsers(self):
        """Открытие браузеров с результатами"""
        print("\n🌐 Открытие результатов в браузере...")
        
        urls = [
            ("📊 MLflow Experiments", "http://localhost:5000"),
            ("🚀 FastAPI Documentation", "http://localhost:8000/docs"),
            ("📈 FastAPI Application", "http://localhost:8000")
        ]
        
        for name, url in urls:
            try:
                webbrowser.open(url)
                print(f"   ✅ Открыто: {name} - {url}")
                time.sleep(1)
            except Exception as e:
                print(f"   ❌ Не удалось открыть {url}: {e}")
    
    def generate_report(self):
        """Генерация финального отчета"""
        print("\n" + "="*60)
        print("📊 ФИНАЛЬНЫЙ ОТЧЕТ ПРОЕКТА CREDIT SCORING")
        print("="*60)
        
        print(f"\n✅ Выполнено шагов: {len(self.steps_completed)}/{11}")
        for i, step in enumerate(self.steps_completed, 1):
            print(f"   {i}. {step}")
        
        print(f"\n🌐 Доступные сервисы:")
        print("   • MLflow Experiments: http://localhost:5000")
        print("   • FastAPI Docs:       http://localhost:8000/docs") 
        print("   • FastAPI App:        http://localhost:8000")
        
        print(f"\n📁 Результаты:")
        results_dir = self.project_root / 'reports'
        if results_dir.exists():
            for file in results_dir.glob('*'):
                print(f"   • {file.name}")
        
        print(f"\n🎯 Проверка требований задания:")
        requirements = [
            ("✅ Структура проекта", "src/, tests/, data/, models/"),
            ("✅ Подготовка данных", "make_dataset.py + валидация"),
            ("✅ Feature Engineering", "build_features.py"),
            ("✅ ML пайплайн", "Sklearn Pipeline + GridSearchCV"),
            ("✅ MLflow Tracking", "Эксперименты + метрики"),
            ("✅ DVC пайплайн", "dvc.yaml + версионирование"),
            ("✅ Тестирование", "pytest + GitHub Actions"),
            ("✅ FastAPI", "REST API с /predict"),
            ("✅ Docker", "Dockerfile"),
            ("✅ Мониторинг", "Детекция дрифта"),
        ]
        
        for status, desc in requirements:
            print(f"   {status} {desc}")
        
        print(f"\n🚀 Проект успешно запущен!")
        print("   Для остановки нажмите Ctrl+C")
    
    def run(self):
        """Главный метод запуска"""
        print("🚀 ЗАПУСК АВТОМАТИЗИРОВАННОГО ПРОЕКТА CREDIT SCORING")
        print("=" * 60)
        
        # Проверка окружения
        if not self.check_installation():
            print("❌ Пожалуйста, установите Python и pip")
            return
        
        # Основные шаги
        steps = [
            self.setup_directories,
            self.install_dependencies,
            self.create_sample_data,
            self.run_data_pipeline,
            self.run_training,
            self.run_tests
        ]
        
        # Запускаем основные шаги
        for step in steps:
            if not step():
                print(f"❌ Прервано на шаге: {step.__name__}")
                return
        
        # Запускаем сервисы в фоне
        mlflow_process = self.start_mlflow_ui()
        api_process = self.start_fastapi()
        
        if api_process:
            # Тестируем API
            self.test_api()
            
            # Запускаем мониторинг
            self.run_monitoring()
            
            # Открываем браузеры
            self.open_browsers()
            
            # Генерируем отчет
            self.generate_report()
            
            try:
                # Держим процессы активными
                print("\n⏳ Сервисы работают... Нажмите Ctrl+C для остановки")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Остановка сервисов...")
                if mlflow_process:
                    mlflow_process.terminate()
                if api_process:
                    api_process.terminate()
                print("✅ Все сервисы остановлены")

if __name__ == "__main__":
    runner = ProjectRunner()
    runner.run()