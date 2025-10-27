#!/usr/bin/env python3
"""
Автоматическая настройка виртуального окружения для проекта
Исправленная версия для Windows с проблемами кодировки
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

class EnvironmentSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / 'venv'
        self.is_windows = platform.system() == 'Windows'
        
        # Исправление кодировки для Windows
        if self.is_windows:
            try:
                # Пробуем установить UTF-8 кодировку
                sys.stdout.reconfigure(encoding='utf-8')
            except:
                pass
    
    def safe_print(self, message, emoji="🔧"):
        """Безопасный вывод с обработкой Unicode"""
        try:
            # Пробуем вывести с emoji
            print(f"{emoji} {message}")
        except UnicodeEncodeError:
            # Если не получается - выводим без emoji
            try:
                print(f">>> {message}")
            except:
                # Последняя попытка - простой вывод
                print(message)
    
    def log(self, message, emoji="🔧"):
        """Логирование с обработкой ошибок кодировки"""
        self.safe_print(message, emoji)
    
    def check_python_version(self):
        """Проверяет версию Python"""
        version = sys.version_info
        self.log(f"Python {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.log("Требуется Python 3.8 или выше!", "ERROR")
            return False
        return True
    
    def create_venv(self):
        """Создает виртуальное окружение"""
        if self.venv_path.exists():
            self.log("Виртуальное окружение уже существует", "OK")
            return True
            
        self.log("Создание виртуального окружения...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', 'venv'],  
                         check=True, cwd=self.project_root,
                         capture_output=True, text=True, encoding='utf-8')
            self.log("Виртуальное окружение создано", "OK")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Ошибка создания venv: {e}", "ERROR")
            return False
    
    def get_venv_python(self):
        """Возвращает путь к Python в виртуальном окружении"""
        if self.is_windows:
            return self.venv_path / 'Scripts' / 'python.exe'
        else:
            return self.venv_path / 'bin' / 'python'
    
    def get_venv_pip(self):
        """Возвращает путь к pip в виртуальном окружении"""
        if self.is_windows:
            return self.venv_path / 'Scripts' / 'pip.exe'
        else:
            return self.venv_path / 'bin' / 'pip'
    
    def upgrade_pip(self):
        """Обновляет pip в виртуальном окружении"""
        python = self.get_venv_python()
        self.log("Обновление pip...")
        try:
            result = subprocess.run(
                [str(python), '-m', 'pip', 'install', '--upgrade', 'pip'], 
                check=True, 
                capture_output=True, 
                text=True,
                encoding='utf-8', 
                errors='ignore',
                timeout=120
            )
            self.log("pip обновлен", "OK")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Не удалось обновить pip: {e.stderr if e.stderr else e.stdout}", "WARN")
            return True
        except subprocess.TimeoutExpired:
            self.log("Таймаут обновления pip", "WARN")
            return True
        except Exception as e:
            self.log(f"Ошибка обновления pip: {e}", "WARN")
            return True
    
    def install_dependencies(self):
        """Устанавливает зависимости в виртуальном окружении"""
        pip = self.get_venv_pip()
        requirements_file = self.project_root / 'requirements.txt'
        
        self.log("Установка зависимостей...")
        try:
            # Сначала пробуем установить основные пакеты по одному
            core_packages = [
                'pip', 'setuptools', 'wheel',
                'numpy', 'pandas', 'scikit-learn',
                'matplotlib', 'seaborn', 'requests',
                'fastapi', 'uvicorn', 'pydantic'
            ]
            
            for package in core_packages:
                try:
                    result = subprocess.run([
                    str(pip), 'install', '-r', str(requirements_file)
                    ], 
                    capture_output=True, 
                    text=True, 
                    timeout=300,
                    encoding='utf-8', 
                    errors='ignore'
                    )
                    if result.returncode == 0:
                        self.log(f"Установлен: {package}", "OK")
                    else:
                        self.log(f"Ошибка установки {package}: {result.stderr[:100]}", "WARN")
                except subprocess.TimeoutExpired:
                    self.log(f"Таймаут установки: {package}", "WARN")
                except Exception as e:
                    self.log(f"Ошибка установки {package}: {e}", "WARN")

            # Затем пробуем установить из requirements.txt если он есть
            if requirements_file.exists():
                try:
                    result = subprocess.run([
                    str(pip), 'install', '-r', str(requirements_file)
                    ], 
                    capture_output=True, 
                    text=True, 
                    timeout=300,
                    encoding='utf-8', 
                    errors='ignore'
                    )

                    if result.returncode == 0:
                        self.log("Зависимости установлены", "OK")
                        return True
                    else:
                        self.log(f"Ошибка установки из requirements.txt: {result.stderr[:200]}", "WARN")
                except subprocess.TimeoutExpired:
                    self.log("Таймаут установки зависимостей", "WARN")
            
            return True
                
        except Exception as e:
            self.log(f"Ошибка установки: {e}", "ERROR")
            return False
    
    def setup_project_structure(self):
        """Создает структуру проекта"""
        directories = [
            'data/raw',
            'data/processed', 
            'models',
            'reports',
            'logs',
            'src/data',
            'src/features', 
            'src/models',
            'src/api',
            'src/monitoring',
            'tests',
            'notebooks',
            'scripts'
        ]
        
        for directory in directories:
            path = self.project_root / directory
            path.mkdir(parents=True, exist_ok=True)
        
        self.log("Структура проекта создана", "OK")
        return True
    
    def create_activation_scripts(self):
        """Создает скрипты для активации окружения"""
        # Windows batch file (без emoji)
        bat_content = """@echo off
chcp 65001 >nul
echo ========================================
echo  Credit Scoring Project - Activation
echo ========================================
echo.
echo Activating virtual environment...
call venv\\Scripts\\activate.bat
echo.
echo Virtual environment activated!
echo To deactivate run: deactivate
echo To run project: python run_project.py
echo.
cmd /k
"""
        
        # Linux/Mac shell script (без emoji)
        sh_content = """#!/bin/bash
echo "========================================"
echo " Credit Scoring Project - Activation"
echo "========================================"
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo ""
echo "Virtual environment activated!"
echo "To deactivate run: deactivate"
echo "To run project: python run_project.py"
exec bash
"""
        
        try:
            (self.project_root / 'activate_env.bat').write_text(bat_content, encoding='utf-8')
            (self.project_root / 'activate_env.sh').write_text(sh_content, encoding='utf-8')
            
            if not self.is_windows:
                subprocess.run(['chmod', '+x', 'activate_env.sh'], cwd=self.project_root)
            
            self.log("Скрипты активации созданы", "OK")
            return True
        except Exception as e:
            self.log(f"Ошибка создания скриптов: {e}", "WARN")
            return True
    
    def create_simple_requirements(self):
        """Создает упрощенный requirements.txt если его нет"""
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            simple_requirements = """# Core data science
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.3.0
scipy>=1.7.0

# ML tools
mlflow>=2.0.0

# API
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.0

# Data validation
great-expectations>=0.15.0

# Version control
dvc>=3.0.0

# Testing
pytest>=6.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
requests>=2.25.0
python-dotenv>=0.19.0
"""
            requirements_file.write_text(simple_requirements, encoding='utf-8')
            self.log("Файл requirements.txt создан", "OK")
        
        return True
    
    def verify_installation(self):
        """Проверяет установку ключевых пакетов"""
        python = self.get_venv_python()
        test_script = """
import sys
try:
    import pandas as pd
    import numpy as np
    print("SUCCESS: pandas and numpy imported")
    
    try:
        import sklearn
        print("SUCCESS: scikit-learn imported")
    except:
        print("WARNING: scikit-learn not available")
    
    try:
        import fastapi
        print("SUCCESS: fastapi imported")
    except:
        print("WARNING: fastapi not available")
        
    sys.exit(0)
except ImportError as e:
    print(f"FAILED: {e}")
    sys.exit(1)
"""
        
        try:
            result = subprocess.run([
                str(python), '-c', test_script
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.log("Проверка зависимостей пройдена", "OK")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        self.log(line.strip(), "INFO")
                return True
            else:
                self.log(f"Проверка не пройдена: {result.stdout}", "WARN")
                return False
        except Exception as e:
            self.log(f"Ошибка проверки: {e}", "WARN")
            return False
    
    def run(self):
        """Запускает полную настройку"""
        self.safe_print("НАСТРОЙКА ВИРТУАЛЬНОГО ОКРУЖЕНИЯ")
        self.safe_print("==========================================")
        
        if not self.check_python_version():
            return False
        
        steps = [
            ("Создание структуры проекта", self.setup_project_structure),
            ("Создание requirements.txt", self.create_simple_requirements),
            ("Создание виртуального окружения", self.create_venv),
            ("Обновление pip", self.upgrade_pip),
            ("Установка зависимостей", self.install_dependencies),
            ("Создание скриптов активации", self.create_activation_scripts),
            ("Проверка установки", self.verify_installation)
        ]
        
        for step_name, step_func in steps:
            self.log(f"{step_name}...")
            if not step_func():
                self.log(f"Прервано на шаге: {step_name}", "ERROR")
                return False
        
        self.safe_print("==========================================")
        self.safe_print("НАСТРОЙКА ЗАВЕРШЕНА УСПЕШНО!", "SUCCESS")
        self.safe_print("")
        self.safe_print("Следующие шаги:")
        self.safe_print("1. Активируйте виртуальное окружение:")
        
        if self.is_windows:
            self.safe_print("   - Запустите: activate_env.bat")
            self.safe_print("   - Или: venv\\Scripts\\activate")
        else:
            self.safe_print("   - Запустите: source activate_env.sh") 
            self.safe_print("   - Или: source venv/bin/activate")
        
        self.safe_print("2. Запустите проект: python run_project.py")
        self.safe_print("")
        self.safe_print("Для деактивации: deactivate")
        
        return True

def main():
    setup = EnvironmentSetup()
    success = setup.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()