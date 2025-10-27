# Credit Scoring ML Pipeline

Проект для автоматизации разработки и тестирования скоринговой модели предсказания дефолта (PD-модель).

## О проекте

Этот проект реализует сквозной (end-to-end) автоматизированный пайплайн для:
- Разработки ML моделей для предсказания дефолта клиентов
- Валидации данных и мониторинга дрифта
- Развертывания моделей через REST API
- Автоматического тестирования и CI/CD

**Домен**: Финансы / Кредитный скоринг  
**Данные**: Default of Credit Card Clients Dataset с UCI Machine Learning Repository

## Структура проекта
credit-scoring-project/
├── data/ # Данные (версионируются DVC)
│ ├── raw/ # Исходные данные
│ ├── processed/ # Обработанные данные
│ └── expectations/ # Правила валидации Great Expectations
├── notebooks/ # EDA и эксперименты
│ └── 01_eda.ipynb # Детальный анализ данных
├── src/ # Исходный код
│ ├── data/
│ │ ├── make_dataset.py # Загрузка и очистка данных
│ │ └── validation.py # Валидация с Great Expectations
│ ├── features/
│ │ └── build_features.py # Feature Engineering
│ ├── models/
│ │ ├── train.py # Обучение моделей с MLflow
│ │ ├── predict.py # Предсказания
│ │ └── pipeline.py # Sklearn pipelines
│ ├── api/
│ │ └── app.py # FastAPI приложение
│ └── monitoring/
│ └── drift_detection.py # Мониторинг дрифта
├── tests/ # Unit-тесты
│ ├── test_preprocessing.py
│ ├── test_models.py
│ └── test_api.py
├── models/ # Сохраненные модели
├── reports/ # Отчеты и графики
├── .github/workflows/ # GitHub Actions
│ └── ci-cd.yml
├── Dockerfile
├── requirements.txt
├── dvc.yaml # DVC pipeline
├── pyproject.toml # Конфигурация инструментов
└── README.md

text

## Установка и настройка

### Предварительные требования

- Python 3.9+
- Git
- DVC
- Docker (опционально)

### Установка

1. **Клонирование репозитория**
```bash
git clone https://github.com/your-username/credit-scoring-project
cd credit-scoring-project
Установка зависимостей

bash
pip install -r requirements.txt
Инициализация DVC

bash
dvc init
Загрузка данных

bash
# Добавьте данные в data/raw/ или настройте DVC remote storage
🚀 Использование
Запуск полного пайплайна
bash
# Запуск всего DVC пайплайна
dvc repro

# Просмотр результатов
dvc metrics show
Обучение моделей
bash
# Запуск экспериментов с MLflow
python src/models/train.py

# Просмотр экспериментов в MLflow UI
mlflow ui
Запуск API
bash
# Локальный запуск
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Или через Docker
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
Тестирование API
bash
# Health check
curl http://localhost:8000/health

# Предсказание
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "LIMIT_BAL": 50000,
       "SEX": 1,
       "EDUCATION": 2,
       "MARRIAGE": 1,
       "AGE": 35,
       "PAY_0": 0,
       "PAY_2": 0,
       "PAY_3": 0,
       "PAY_4": 0,
       "PAY_5": 0,
       "PAY_6": 0,
       "BILL_AMT1": 1000,
       "BILL_AMT2": 1000,
       "BILL_AMT3": 1000,
       "BILL_AMT4": 1000,
       "BILL_AMT5": 1000,
       "BILL_AMT6": 1000,
       "PAY_AMT1": 1000,
       "PAY_AMT2": 1000,
       "PAY_AMT3": 1000,
       "PAY_AMT4": 1000,
       "PAY_AMT5": 1000,
       "PAY_AMT6": 1000
     }'
Мониторинг дрифта
bash
python src/monitoring/drift_detection.py
🧪 Тестирование
Запуск тестов
bash
# Все тесты
pytest tests/ -v

# С покрытием кода
pytest tests/ -v --cov=src --cov-report=html

# Конкретный модуль
pytest tests/test_models.py -v
Линтинг и форматирование
bash
# Проверка стиля кода
flake8 src/ tests/

# Форматирование кода
black src/ tests/

# Проверка форматирования
black --check src/ tests/
🔧 CI/CD
Проект использует GitHub Actions для автоматического:

Запуска тестов при каждом пуше

Проверки качества кода (flake8, black)

Валидации данных с Great Expectations

Сборки Docker образа

📊 Модели и метрики
Поддерживаемые алгоритмы
Logistic Regression

Random Forest

Gradient Boosting

Ключевые метрики
ROC-AUC

Precision, Recall, F1-Score

Accuracy

Feature Engineering
Агрегация истории платежей

Биннинг возраста и других признаков

Создание риск-ориентированных признаков

Временные тренды

📈 Мониторинг
Валидация данных
Great Expectations для проверки качества данных

Автоматическое обнаружение аномалий

Валидация новых входящих данных

Детекция дрифта
Population Stability Index (PSI)

Статистические тесты распределений

Мониторинг ключевых признаков

🐳 Docker
Сборка образа
bash
docker build -t credit-scoring-api .
Запуск контейнера
bash
docker run -p 8000:8000 credit-scoring-api
📝 Лицензия
Этот проект создан в учебных целях.