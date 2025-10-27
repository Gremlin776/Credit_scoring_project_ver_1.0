@echo off
chcp 65001 >nul
echo 🚀 Запуск Credit Scoring Project...

:: Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python не установлен или не добавлен в PATH
    pause
    exit /b 1
)

:: Проверка виртуального окружения
python -c "import sys; print('VENV_ACTIVE' if sys.prefix != sys.base_prefix else 'NO_VENV')" | find "VENV_ACTIVE" >nul
if errorlevel 1 (
    echo 🔧 Виртуальное окружение не активировано
    echo Запуск автоматической настройки...
    python setup_environment.py
    if errorlevel 1 (
        echo ❌ Настройка не удалась
        pause
        exit /b 1
    )
    echo.
    echo ✅ Настройка завершена!
    echo Запустите activate_env.bat для активации окружения
    pause
    exit /b 0
)

echo ✅ Виртуальное окружение активировано
echo 🎯 Запуск проекта...
python run_project.py

pause