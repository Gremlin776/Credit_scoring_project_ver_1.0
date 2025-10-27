FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Создание необходимых директорий
RUN mkdir -p data/raw data/processed models reports logs

# Создание пользователя для безопасности
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Экспорт портов
EXPOSE 8000  # FastAPI
EXPOSE 5000  # MLflow

# Команда запуска - запускаем основной скрипт
CMD ["python", "run_project.py"]