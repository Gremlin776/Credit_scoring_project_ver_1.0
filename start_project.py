#!/usr/bin/env python3
"""
Универсальный скрипт запуска для всех операционных систем
Работает в: Windows Command Prompt, Git Bash, Linux Terminal, Mac Terminal
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def detect_environment():
    """Определяет окружение и подсказывает правильную команду"""
    print("🔍 Определение окружения...")
    
    is_windows = platform.system() == 'Windows'
    is_git_bash = 'MINGW' in platform.system() or 'Git' in os.environ.get('SHELL', '')
    
    print(f"ОС: {platform.system()}")
    print(f"Терминал: {'Git Bash' if is_git_bash else 'Стандартный терминал'}")
    
    return is_windows, is_git_bash

def check_python():
    """Проверяет наличие Python"""
    try:
        result = subprocess.run([sys.executable, '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Python: {result.stdout.strip()}")
            return True
    except:
        pass
    
    print("❌ Python не найден в PATH")
    print("💡 Установите Python с официального сайта: https://python.org")
    return False

def check_venv():
    """Проверяет активировано ли виртуальное окружение"""
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        print("✅ Виртуальное окружение активировано")
    else:
        print("❌ Виртуальное окружение не активировано")
    return in_venv

def setup_environment():
    """Запускает настройку окружения"""
    print("\n🔧 Запуск автоматической настройки...")
    try:
        result = subprocess.run([sys.executable, 'setup_environment.py'], 
                              cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Ошибка настройки: {e}")
        return False

def show_activation_instructions(is_windows, is_git_bash):
    """Показывает инструкции по активации для текущего окружения"""
    project_dir = Path(__file__).parent
    
    print("\n🎯 ИНСТРУКЦИЯ ДЛЯ ВАШЕГО ОКРУЖЕНИЯ:")
    print("=" * 50)
    
    if is_git_bash:
        print("Вы используете Git Bash (MINGW64)")
        print("")
        print("1. Активируйте виртуальное окружение:")
        print("   source venv/Scripts/activate")
        print("")
        print("2. Запустите проект:")
        print("   python run_project.py")
        print("")
        print("ИЛИ одной командой:")
        print("   source venv/Scripts/activate && python run_project.py")
        
    elif is_windows:
        print("Вы используете Windows Command Prompt")
        print("")
        print("1. Активируйте виртуальное окружение:")
        print("   venv\\Scripts\\activate")
        print("")
        print("2. Запустите проект:")
        print("   python run_project.py")
        print("")
        print("ИЛИ запустите: activate_env.bat")
        
    else:  # Linux/Mac
        print("Вы используете Linux/Mac Terminal")
        print("")
        print("1. Активируйте виртуальное окружение:")
        print("   source venv/bin/activate")
        print("")
        print("2. Запустите проект:")
        print("   python run_project.py")
        print("")
        print("ИЛИ запустите: ./activate_env.sh")
    
    print("")
    print("Для деактивации: deactivate")

def main():
    print("🚀 CREDIT SCORING PROJECT - УНИВЕРСАЛЬНЫЙ ЗАПУСК")
    print("=" * 60)
    
    # Определяем окружение
    is_windows, is_git_bash = detect_environment()
    
    # Проверяем Python
    if not check_python():
        return
    
    # Проверяем виртуальное окружение
    if not check_venv():
        print("\n🔄 Настройка необходима...")
        
        # Запускаем настройку
        if setup_environment():
            print("\n✅ Настройка завершена успешно!")
            
            # Показываем инструкции
            show_activation_instructions(is_windows, is_git_bash)
        else:
            print("\n❌ Настройка не удалась")
            print("💡 Попробуйте запустить вручную: python setup_environment.py")
        
        return
    
    # Если venv активирован - запускаем проект
    print("\n🎯 Запуск проекта...")
    try:
        subprocess.run([sys.executable, 'run_project.py'], 
                      cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n🛑 Проект остановлен пользователем")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")

if __name__ == "__main__":
    main()