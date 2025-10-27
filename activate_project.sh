#!/bin/bash
echo "🚀 Credit Scoring Project - Активация для Git Bash"

# Проверка Python
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python не найден"
    exit 1
fi

# Определяем Python
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

# Проверка виртуального окружения
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Виртуальное окружение уже активировано"
    echo "🎯 Запуск проекта..."
    $PYTHON_CMD run_project.py
    exit 0
fi

# Проверка существования venv
if [ -d "venv" ]; then
    echo "🔧 Активация виртуального окружения..."
    source venv/Scripts/activate
    echo "🎯 Запуск проекта..."
    $PYTHON_CMD run_project.py
else
    echo "🔧 Виртуальное окружение не найдено"
    echo "🔄 Запуск настройки..."
    $PYTHON_CMD setup_environment.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Настройка завершена!"
        echo "🔧 Активируйте окружение и запустите снова:"
        echo "   source venv/Scripts/activate"
        echo "   python run_project.py"
    else
        echo "❌ Настройка не удалась"
        exit 1
    fi
fi