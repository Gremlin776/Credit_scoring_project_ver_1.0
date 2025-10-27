import os
from pathlib import Path
import sys
import io

# Принудительно устанавливаем UTF-8 кодировку для Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')



def create_project_structure():
    """Создание структуры директорий проекта"""
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'reports',
        'logs',
        'notebooks',
        'scripts',
        'src/data',
        'src/features',
        'src/models',
        'src/api',
        'src/monitoring',
        'tests',
        '.github/workflows'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Создана директория: {directory}")
    
    print("🎯 Структура проекта создана!")

if __name__ == "__main__":
    create_project_structure()