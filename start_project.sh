#!/bin/bash
echo "🚀 Запуск Credit Scoring Project..."

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не установлен"
    exit 1
fi

# Проверка виртуального окружения
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 Виртуальное окружение не активировано"
    echo "Запуск автоматической настройки..."
    python3 setup_environment.py
    if [ $? -ne 0 ]; then
        echo "❌ Настройка не удалась"
        exit 1
    fi
    echo ""
    echo "✅ Настройка завершена!"
    echo "Запустите: source activate_env.sh"
    exit 0
fi

echo "✅ Виртуальное окружение активировано"
echo "🎯 Запуск проекта..."
python3 run_project.py