#!/usr/bin/env python3
"""
Автоматическая настройка виртуального окружения для проекта
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
        
    def log(self, message, status="INFO"):
        """Логирование"""
        print(f"{status} {message}")
    
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
            subprocess.run([
                sys.executable, '-m', 'venv', 'venv'
            ], check=True, cwd=self.project_root, capture_output=True, text=True)
            self.log("Виртуальное окружение создано", "OK")
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Ошибка создания venv: {e}", "ERROR")
            return False
    
    def get_venv_python(self):
        """Возвращает путь к Python в виртуальном окружении"""
        if self.is_windows:
            python_path = self.venv_path / 'Scripts' / 'python.exe'
        else:
            python_path = self.venv_path / 'bin' / 'python'
        
        if not python_path.exists():
            self.log(f"Python в venv не найден: {python_path}", "ERROR")
            return None
        return python_path
    
    def install_dependencies(self):
        """Устанавливает зависимости в виртуальном окружении"""
        python = self.get_venv_python()
        if python is None:
            return False
            
        requirements_file = self.project_root / 'requirements.txt'
        
        self.log("Установка зависимостей из requirements.txt...")
        
        try:
            result = subprocess.run([
                str(python), '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True, timeout=300, cwd=self.project_root)
            
            if result.returncode == 0:
                self.log("Основные зависимости установлены", "OK")
                
                # Проверяем установку ключевых пакетов
                check_result = subprocess.run([
                    str(python), '-c', 
                    "import pytest, flake8, scipy, mlflow, fastapi; print('SUCCESS: Все пакеты установлены')"
                ], capture_output=True, text=True, cwd=self.project_root)
                
                if "SUCCESS" in check_result.stdout:
                    self.log(" Все ключевые пакеты установлены корректно", "OK")
                else:
                    self.log(" Некоторые пакеты могут быть не установлены", "WARN")
                    
                return True
            else:
                self.log(f"Ошибка установки зависимостей: {result.stderr[:200]}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Ошибка установки зависимостей: {e}", "ERROR")
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
            'scripts',
            '.github/workflows'
        ]
        
        for directory in directories:
            path = self.project_root / directory
            path.mkdir(parents=True, exist_ok=True)
        
        self.log("Структура проекта создана", "OK")
        return True
    
    def create_activation_scripts(self):
        """Создает скрипты для активации окружения"""
        # Windows batch file
        bat_content = """@echo off
echo ========================================
echo  Credit Scoring Project - Activation
echo ========================================
echo.
echo Activating virtual environment...
call venv\\Scripts\\activate.bat
echo.
echo Virtual environment activated!
echo To deactivate run: deactivate
echo To run project: python launch.py
echo.
cmd /k
"""
        
        # Linux/Mac shell script
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
echo "To run project: python launch.py"
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
    
    def verify_installation(self):
        """Проверяет установку ключевых пакетов"""
        python = self.get_venv_python()
        if python is None:
            return False
            
        test_script = """
try:
    import pandas as pd
    print("SUCCESS: pandas imported")
    
    import numpy as np
    print("SUCCESS: numpy imported")
    
    import sklearn
    print("SUCCESS: scikit-learn imported")
    
    import mlflow
    print("SUCCESS: mlflow imported")
    
    import pytest
    print("SUCCESS: pytest imported")
    
    import flake8
    print("SUCCESS: flake8 imported")
    
    import scipy
    print("SUCCESS: scipy imported")
    
    print("ALL_CHECKS_PASSED")
except ImportError as e:
    print(f"FAILED: {e}")
"""
        
        try:
            result = subprocess.run([
                str(python), '-c', test_script
            ], capture_output=True, text=True, timeout=30, cwd=self.project_root)
            
            if "ALL_CHECKS_PASSED" in result.stdout:
                self.log(" Проверка зависимостей пройдена", "OK")
                return True
            else:
                self.log(f" Проверка не пройдена: {result.stdout}", "WARN")
                return False
        except Exception as e:
            self.log(f"Ошибка проверки: {e}", "WARN")
            return False
    
    def run(self):
        """Запускает полную настройку"""
        self.log("НАСТРОЙКА ВИРТУАЛЬНОГО ОКРУЖЕНИЯ")
        self.log("==========================================")
        
        if not self.check_python_version():
            return False
        
        steps = [
            ("Создание структуры проекта", self.setup_project_structure),
            ("Создание виртуального окружения", self.create_venv),
            ("Установка зависимостей", self.install_dependencies),
            ("Создание скриптов активации", self.create_activation_scripts),
            ("Проверка установки", self.verify_installation)
        ]
        
        for step_name, step_func in steps:
            self.log(f"{step_name}...")
            success = step_func()
            if not success and step_name == "Установка зависимостей":
                self.log("Продолжаем без некоторых зависимостей", "WARN")
                continue
            elif not success:
                self.log(f"Прервано на шаге: {step_name}", "ERROR")
                return False
        
        self.log("==========================================")
        self.log(" НАСТРОЙКА ЗАВЕРШЕНА!", "SUCCESS")
        self.log("Следующие шаги:")
        self.log("1. Активируйте виртуальное окружение:")
        
        if self.is_windows:
            self.log("   - Запустите: activate_env.bat")
            self.log("   - Или: source venv\\Scripts\\activate")
        else:
            self.log("   - Запустите: source activate_env.sh") 
            self.log("   - Или: source venv/bin/activate")
        
        self.log("2. Запустите проект: python launch.py")
        self.log("3. Откройте в браузере:")
        self.log("   -  Документация: http://127.0.0.1:8000/docs")
        self.log("   -  Дашборд: http://127.0.0.1:8000/dashboard")
        self.log("   -  Эксперименты: http://127.0.0.1:5000")
        
        return True

def main():
    setup = EnvironmentSetup()
    success = setup.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()