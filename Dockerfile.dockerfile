FROM python:3.9-slim

WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p data/raw data/processed models reports

# Экспорт порта
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]