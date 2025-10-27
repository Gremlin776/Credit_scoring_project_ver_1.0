#!/usr/bin/env python3
"""
Скрипт для исправления проблем с зависимостями
"""

import subprocess
import sys
import importlib

def check_import(package_name):
    """Проверяет может ли пакет быть импортирован"""
    try:
        importlib.import_module(package_name.replace('-', '_'))
        print(f"✅ {package_name} - импорт работает")
        return True
    except Exception as e:
        print(f"❌ {package_name} - ошибка импорта: {e}")
        return False

def fix_sklearn():
    """Исправляет проблемы с scikit-learn"""
    print("🔧 Исправление scikit-learn...")
    
    # Переустанавливаем scikit-learn
    packages = [
        'scikit-learn==1.5.0',
        'numpy==1.26.4',
        'scipy==1.13.1',
        'joblib==1.4.2',
        'threadpoolctl==3.5.0'
    ]
    
    for package in packages:
        try:
            print(f"Установка {package}...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--force-reinstall', package
            ], check=True, capture_output=True)
            print(f"✅ {package} установлен")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Ошибка установки {package}: {e.stderr.decode()}")
    
    # Проверяем импорт
    return check_import('sklearn')

def install_core_packages():
    """Устанавливает основные пакеты по одному"""
    core_packages = [
        'pandas==2.2.2',
        'numpy==1.26.4',
        'scikit-learn==1.5.0',
        'matplotlib==3.8.4',
        'seaborn==0.13.2',
        'requests==2.31.0',
        'fastapi==0.111.0',
        'uvicorn==0.30.1',
        'pydantic==2.7.1',
        'mlflow==2.14.2',
        'pytest==8.2.1',
        'great-expectations==0.18.14',
        'dvc==3.50.1'
    ]
    
    successful = []
    failed = []
    
    for package in core_packages:
        try:
            print(f"📦 Установка {package}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                successful.append(package)
                print(f"✅ {package} - успешно")
            else:
                failed.append(package)
                print(f"❌ {package} - ошибка: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            failed.append(package)
            print(f"⏰ {package} - таймаут")
        except Exception as e:
            failed.append(package)
            print(f"💥 {package} - исключение: {e}")
    
    return successful, failed

def check_all_imports():
    """Проверяет импорт всех ключевых пакетов"""
    packages = [
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'requests',
        'fastapi',
        'uvicorn',
        'pydantic',
        'mlflow',
        'pytest',
        'great_expectations',
        'dvc'
    ]
    
    print("\n🔍 Проверка импорта пакетов...")
    working = []
    broken = []
    
    for package in packages:
        if check_import(package):
            working.append(package)
        else:
            broken.append(package)
    
    return working, broken

def main():
    print("🚀 ЗАПУСК ИСПРАВЛЕНИЯ ЗАВИСИМОСТЕЙ")
    print("=" * 50)
    
    # Сначала пробуем исправить scikit-learn
    if not fix_sklearn():
        print("\n🔄 Пробуем полную переустановку...")
        successful, failed = install_core_packages()
        
        print(f"\n📊 Результаты:")
        print(f"✅ Успешно: {len(successful)} пакетов")
        print(f"❌ Не удалось: {len(failed)} пакетов")
        
        if failed:
            print("\nНеудачные пакеты:")
            for pkg in failed:
                print(f"  - {pkg}")
    
    # Финальная проверка
    working, broken = check_all_imports()
    
    print(f"\n🎯 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
    print(f"✅ Работают: {len(working)} пакетов")
    print(f"❌ Сломаны: {len(broken)} пакетов")
    
    if broken:
        print("\nСледующие пакеты требуют внимания:")
        for pkg in broken:
            print(f"  - {pkg}")
        
        print("\n💡 Рекомендации:")
        if 'sklearn' in broken or 'scikit-learn' in broken:
            print("  • Попробуйте: pip install --force-reinstall scikit-learn")
            print("  • Или: conda install scikit-learn (если используете conda)")
        
        if 'great_expectations' in broken:
            print("  • great-expectations может требовать дополнительных зависимостей")
        
        return False
    else:
        print("\n🎉 Все зависимости работают корректно!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)