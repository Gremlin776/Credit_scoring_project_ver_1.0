Credit Scoring ML Pipeline
Автоматизированный ML пайплайн для предсказания дефолта клиентов.

# Оглавление
- Быстрый старт
- Структура проекта
- Что выполняется автоматически
- Результаты
- Требования задания
- Ручной запуск
- Docker запуск
- API документация
- Мониторинг
- Разработка

⚡ Быстрый старт
Всего 2 шага:

## 1. Скачать проект

`git clone https://github.com/Gremlin776/Credit_scoring_project_ver_1.0`

## 2. Перейти в скачанную директорию
`cd Credit_scoring_project_ver_1.0`

## 3. Запустить одну команду
`python run_project.py`

Что произойдет:
 Автоматически выполнится:

Установка всех зависимостей

Создание реалистичных sample данных

Запуск полного ML пайплайна

Обучение 5+ моделей с GridSearchCV

Валидация данных с Great Expectations

Запуск REST API

Мониторинг дрифта данных

# Генерация отчетов

 Автоматически откроются в браузере:

- MLflow Experiments - http://localhost:5000

- FastAPI Documentation - http://localhost:8000/docs

- Web Interface - http://localhost:8000

# Структура проекта

credit-scoring-project/
├──  run_project.py                 # запуск всего проекта
│   
├──  README.md                      # документация
│   
├──  requirements.txt               # Зависимости Python

├──  Dockerfile                     # Конфигурация Docker
│   
├──  dvc.yaml                       # DVC пайплайн
│   
├──  pyproject.toml                 # Конфигурация инструментов
│   
├──  .github/workflows/ci-cd.yml    # GitHub Actions CI/CD
│   
├──  data/                          # Данные (версионируются DVC)
│  
│   ├── raw/                         # Исходные данные
│   
│   └── processed/                   # Обработанные данные
│   
├──  src/                           # Исходный код
│   
│   ├──  data/
│   
│   │   ├── make_dataset.py          # Загрузка и очистка данных
│   
│   │   └── validation.py            # Валидация с Great Expectations
│   
│   ├──  features/
│   │ 
│   │   └── build_features.py        # Feature Engineering
│   │
│   ├──  models/
│   │ 
│   │   ├── train.py                 # Обучение моделей с MLflow
│   │ 
│   │   ├── predict.py               # Предсказания
│   │ 
│   │   └── pipeline.py              # Sklearn Pipeline
│   │
│   ├──  api/
│   │   │  
│   │   └── app.py                   # FastAPI приложение
│   │
│   └──  monitoring/
│       └── drift_detection.py       # Мониторинг дрифта
│
├──  tests/                         # Unit-тесты
│   │ 
│   ├── test_preprocessing.py
│   │ 
│   ├── test_models.py
│   │ 
│   └── test_api.py
│
├──  notebooks/                     # EDA и эксперименты
│   └── 01_eda.py                    # Exploratory Data Analysis
│
├──  scripts/                       # Вспомогательные скрипты
│   │ 
│   ├── create_sample_data.py        # Генерация тестовых данных
│   │ 
│   └── create_directories.py        # Создание структуры проекта
│
├──  models/                        # Обученные модели
│   
├──  reports/                       # Отчеты и графики
│   
└──  logs/                          # Логи приложения

# Что выполняется автоматически
При запуске python run_project.py:
Этап	                   Статус	             Описание
Установка зависимостей	Автоматически	   Все Python пакеты из requirements.txt
Подготовка данных	Автоматически	   Создание sample данных + очистка
Feature Engineering	Автоматически	   15+ новых признаков на основе EDA
Валидация данных	Автоматически	   Great Expectations + отчеты
Обучение моделей	5+ эксперименто    Logistic, Random Forest, Gradient Boosting
GridSearchCV	        Автоматически	   Подбор гиперпараметров
MLflow Tracking	        Автоматически      Метрики, параметры, артефакты
Тестирование	        Автоматически	   Pytest + покрытие кода
REST API	        Автоматически	   FastAPI с Swagger документацией
Мониторинг дрифта	Автоматически	   PSI + статистические тесты
Генерация отчетов	Автоматически	   Метрики, графики, аналитика

# Результаты
После запуска вы получите:

1. MLflow Experiments (http://localhost:5000)
- Сравнение 5+ моделей

- Метрики: ROC-AUC, Precision, Recall, F1-Score

- ROC-кривые и Feature Importance

- Параметры моделей

- Анализ экспериментов

2. REST API (http://localhost:8000/docs)
- Endpoint /predict для предсказаний

- Swagger документация

- Health checks (/health)

- Информация о модели (/model_info)

- Batch predictions (/batch_predict)

3. Отчеты (папка reports/)
- best_model_info.json - информация о лучшей модели

- model_evaluation.png - ROC-кривые и графики

- validation_report.json - результаты валидации данных

- drift_metrics.json - мониторинг дрифта

- eda_report.json - анализ данных

# Ручной запуск 
Если нужен контроль над отдельными этапами:

# Только обучение моделей
python src/models/train.py

# Только API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Только MLflow UI
mlflow ui --port 5000

# Только тесты
pytest tests/ -v

# Только валидация данных
python src/data/validation.py

# Только мониторинг дрифта
python src/monitoring/drift_detection.py

# Полный DVC пайплайн
dvc repro

# Docker запуск
Сборка и запуск:
bash
## Сборка образа
docker build -t credit-scoring .

## Запуск контейнера
docker run -p 8000:8000 -p 5000:5000 credit-scoring

Проверка работы:

## Health check
curl http://localhost:8000/health

# Документация API
## Откройте: http://localhost:8000/docs

# MLflow UI
## Откройте: http://localhost:5000

# API документация
Основные endpoints:
Method	Endpoint	Description
GET	/	Информация о API
GET	/health	Проверка здоровья системы
POST	/predict	Предсказание дефолта
POST	/batch_predict	Пакетное предсказание
GET	/model_info	Информация о модели
GET	/sample_data	Пример данных для тестирования
Пример запроса к API:
bash
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
📈 Мониторинг
Детекция дрифта данных:
python src/monitoring/drift_detection.py

#Метрики мониторинга:

* PSI (Population Stability Index) - для ключевых признаков
* Kolmogorov-Smirnov test - статистическая значимость
* Target drift - изменение распределения целевой переменной
* Feature drift - дрифт отдельных признаков

Отчеты: reports/drift_metrics.json, reports/monitoring_report.json

# Разработка
Установка для разработки:

## Клонирование 
git clone https://github.com/Gremlin776/Credit_scoring_project_ver_1.0
cd credit-scoring-project

## Установка зависимостей
pip install -r requirements.txt

## Запуск тестов
pytest tests/ -v

## Проверка качества кода
flake8 src/ tests/
black --check src/ tests/

#CI/CD пайплайн:

Проект включает GitHub Actions для автоматического:

* Запуска тестов при каждом пуше
* Проверки качества кода (flake8, black)
* Валидации данных с Great Expectations
* Сборки Docker образа
* Деплоя на staging

#  Технологии
Категория	 --  Технологии
ML Framework	 --  Scikit-learn, MLflow
API	         -- FastAPI, Uvicorn, Pydantic
Data Validation	 -- Great Expectations
Version Control	 -- DVC, Git
Testing	         -- Pytest, Coverage
Containerization -- Docker
CI/CD	         -- GitHub Actions
Monitoring	 -- PSI, Statistical Tests
Visualization	 -- Matplotlib, Seaborn

# Поддержка
Если возникли проблемы:
Проверьте установку Python 3.9+
python --version

Проверьте доступность портов 8000 и 5000
netstat -an | findstr :8000
netstat -an | findstr :5000

Запустите проверку проекта
python check_project.py

Посмотрите логи в папке logs/

Быстрое решение:
# Если порты заняты
lsof -ti:8000 | xargs kill -9
lsof -ti:5000 | xargs kill -9

# Если ошибки зависимостей
pip install --force-reinstall -r requirements.txt

# Если данные не создаются
python scripts/create_sample_data.py
