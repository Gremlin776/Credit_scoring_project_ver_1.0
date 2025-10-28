#  Credit Scoring ML Pipeline

##  Оглавление
- [Описание проекта](#-описание-проекта)
- [Основные возможности](#-основные-возможности)
- [Архитектура проекта](#️-архитектура-проекта)
- [Установка и запуск](#️-установка-и-запуск)
- [Структура проекта](#-структура-проекта)
- [Использование](#-использование)
- [API Endpoints](#-api-endpoints)
- [Модели и метрики](#-модели-и-метрики)
- [Технологический стек](#️-технологический-стек)
- [Разработка](#-разработка)
- [Быстрый старт](#-быстрый-старт)

## Описание проекта

**Credit Scoring ML Pipeline** - это комплексное end-to-end решение для автоматизированного предсказания дефолта клиентов банка. Проект представляет собой полный ML пайплайн от исследования данных до готового REST API, готового к интеграции в продакшн-среду.

**Бизнес-задача**: Автоматизация процесса разработки и обслуживания PD-модели (Probability of Default) для кредитного отдела банка.

**Данные**: Default of Credit Card Clients Dataset (UCI) - 30,000 записей с 24 признаками, включая демографическую информацию, историю платежей, данные о счетах и платежах.

### **Автоматическое тестирование**
- Unit-тесты для всех компонентов пайплайна
- Интеграционные тесты API
- CI/CD пайплайн с GitHub Actions
- Проверка качества кода с flake8

### **Мониторинг дрифта данных**
- PSI (Population Stability Index) метрики
- Автоматическое обнаружение дрифта признаков
- Визуализация изменений распределений
- Алертинг при значительном дрифте

### **Улучшенный дашборд**
- Интерактивные графики важности признаков
- Real-time мониторинг состояния системы
- Кнопки для управления пайплайном
- Автоматическое обновление отчетов

## Основные возможности

### **Адаптивный EDA анализ**
- Автоматическое обнаружение проблем данных (выбросы, пропуски, аномалии)
- Статистический анализ признаков и их значимости
- Визуализация распределений и корреляций
- Генерация автоматических выводов и рекомендаций

### **Автоматический Feature Engineering**
- Создание агрегированных признаков из истории платежей
- Биннинг возраста и создание возрастных групп
- Расчет риск-скоров на основе EDA insights
- Создание отношений и пропорций между признаками

### **Обучение моделей**
- Multiple алгоритмы: Logistic Regression, Random Forest, Gradient Boosting
- Автоматический подбор гиперпараметров через GridSearchCV
- Sklearn Pipeline с полной предобработкой данных
- Стратифицированное разделение данных с учетом дисбаланса

### **MLflow Tracking**
- Логирование параметров, метрик и артефактов
- Сравнение экспериментов в UI интерфейсе
- Версионирование моделей и воспроизводимость
- Автоматическое сохранение лучшей модели

### **REST API с FastAPI**
- Endpoint для одиночных и пакетных предсказаний
- Автоматическая документация Swagger/OpenAPI
- Health checks и мониторинг состояния
- Валидация входных данных через Pydantic

### **Интерактивный дашборд**
- Визуализация матрицы корреляций
- Графики распределения целевой переменной
- ROC-кривые и оценка качества моделей
- Feature importance анализ

### **Валидация данных**
- Great Expectations для создания suite правил
- Проверка на наличие null значений
- Валидация диапазонов и типов данных
- Автоматическое обнаружение аномалий

### **DVC для версионирования**
- Управление версиями датасетов и моделей
- Воспроизводимость экспериментов
- DVC пайплайн с этапами prepare → features → validate → train

## Архитектура проекта
credit-scoring-project/
├── launch.py # Главный скрипт запуска всего пайплайна
-
├── setup_environment.py # Автоматическая настройка окружения
-
├── requirements.txt # Зависимости проекта
-
├── dvc.yaml # Конфигурация DVC пайплайна
-
├── Dockerfile # Конфигурация Docker контейнера

├── .github/workflows/ # GitHub Actions для CI/CD

│ └── ci-cd.yml
│
├── src/ # Исходный код проекта
│ ├── data/ # Загрузка и подготовка данных
│ │ ├── make_dataset.py # Создание processed_data.csv
│ │ └── validation.py # Валидация данных
│ │
│ ├── features/ # Feature Engineering
│ │ └── build_features.py # Создание признаков
│ │
│ ├── models/ # Обучение моделей
│ │ ├── train.py # Основное обучение с MLflow
│ │ ├── pipeline.py # Sklearn пайплайны
│ │ └── predict.py # Класс для предсказаний
│ │
│ ├── api/ # FastAPI приложение
│ │ └── app.py # Основное API приложение
│ │
│ └️── monitoring/ # Мониторинг дрифта
│ └── drift_detection.py # Детекция дрифта данных
│
├── tests/ # Unit-тесты
│ ├── test_data.py # Тесты обработки данных
│ ├── test_features.py # Тесты feature engineering
│ └── test_models.py # Тесты моделей
│
├── notebooks/ # Jupyter ноутбуки
│ └── 01_eda.py # Автоматический EDA анализ
│
├── scripts/ # Вспомогательные скрипты
│ ├── create_directories.py # Создание структуры проекта
│ ├── create_sample_data.py # Создание тестовых данных
│ ├── monitor_drift.py # Мониторинг дрифта
│ └── create_feature_importance.py # Создание графиков
│
├── data/ # Данные (управляются через DVC)
│ ├── raw/ # Исходные данные
│ │ └── UCI_Credit_Card.csv # Основной датасет
│ └── processed/ # Обработанные данные
│ ├── processed_data.csv # Данные после очистки
│ └── data_with_features.csv # Данные с фичами
│
├── models/ # Обученные модели
│ └── best_model/ # Лучшая модель
│ └── model.pkl # Сериализованная модель
│
├── reports/ # Отчеты и визуализации
│ ├── target_distribution.png # Распределение целевой переменной
│ ├── correlation_matrix.png # Матрица корреляций
│ ├── model_evaluation.png # Оценка модели
│ ├── best_model_feature_importance.png # Важность признаков
│ ├── eda_report.json # Детальный отчет EDA
│ └── feature_importance.json # Данные важности признаков
│
├── logs/ # Логи приложения
│ └── launch.log # Лог запуска проекта
│
└── mlruns/ # MLflow эксперименты
└── .../ # Запуски и артефакты

text

## Установка и запуск

### Предварительные требования

**Обязательные:**
- Python 3.8 или выше
- Git
- pip (менеджер пакетов Python)

**Рекомендуемые:**
- 4GB+ оперативной памяти
- 2GB+ свободного места на диске
- Стабильное интернет-соединение

**Поддерживаемые ОС:**
- Windows 10/11
- Linux (Ubuntu 18.04+, CentOS 7+)
- macOS 10.14+

### Быстрый запуск

```bash
# 1. Клонирование репозитория
git clone https://github.com/Gremlin776/Credit_scoring_project_ver_1.0.git
cd Credit_scoring_project_ver_1.0

# 2. Автоматическая настройка окружения
python setup_environment.py

# 3. Активация виртуального окружения
# Для Windows:
source venv\Scripts\activate
# Для Linux/Mac:
source venv/bin/activate

# 4. Запуск полного пайплайна
python launch.py
После выполнения этих команд откроются:

 FastAPI Documentation: http://127.0.0.1:8000/docs

 MLflow UI: http://127.0.0.1:5000

 Дашборд с графиками: http://127.0.0.1:8000/dashboard

 Ручная настройка
bash
# Создание виртуального окружения
python -m venv venv

# Активация окружения
source venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Установка зависимостей
pip install -r requirements.txt

# Запуск проекта
python launch.py

# Запуск через Docker
bash
# Сборка и запуск контейнера
docker build -t credit-scoring .
docker run -p 8000:8000 -p 5000:5000 credit-scoring

# Использование
Полный цикл работы
Запуск пайплайна → python launch.py

Анализ результатов через веб-интерфейсы

Тестирование API через Swagger документацию

Мониторинг через MLflow и health checks

Режимы запуска
bash
# Полный пайплайн
python launch.py

# Только обучение моделей
python launch.py --mode train

# Только тестирование
python launch.py --mode test

# Только мониторинг дрифта
python launch.py --mode monitor

# Только API сервер
python launch.py --mode api

# CI/CD режим
python launch.py --mode ci
 API Endpoints
Method	Endpoint	    Description
GET	/	            Информация о API
GET	/health	            Статус системы
GET	/docs	            Swagger документация
POST	/predict	    Предсказание дефолта
POST	/batch_predict      Пакетное предсказание
GET	/model_info	    Информация о модели
GET	/dashboard	    Интерактивный дашборд
GET	/monitoring/drift   Отчет о дрифте данных
POST	/testing/run	    Запуск тестов
# Тестирование API
bash
# Health check
curl http://127.0.0.1:8000/health

# Предсказание для хорошего заемщика
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 200000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": -1,
    "PAY_2": -1,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -1,
    "PAY_6": -1,
    "BILL_AMT1": 35000,
    "BILL_AMT2": 34000,
    "BILL_AMT3": 33000,
    "BILL_AMT4": 32000,
    "BILL_AMT5": 31000,
    "BILL_AMT6": 30000,
    "PAY_AMT1": 5000,
    "PAY_AMT2": 5000,
    "PAY_AMT3": 5000,
    "PAY_AMT4": 5000,
    "PAY_AMT5": 5000,
    "PAY_AMT6": 5000
  }'
📈 Модели и метрики
Обученные алгоритмы
Logistic Regression - быстрая, интерпретируемая

Random Forest - устойчивость к выбросам, feature importance

Gradient Boosting - высокая точность

Метрики качества
ROC-AUC: 0.75+

Precision: 0.65+

Recall: 0.70+

F1-Score: 0.67+

 Feature Importance
Топ-5 наиболее важных признаков:

PAY_0 - статус последнего платежа

PAY_2 - статус платежа 2 месяца назад

PAY_3 - статус платежа 3 месяца назад

LIMIT_BAL - кредитный лимит

AGE - возраст клиента

# Технологический стек
Machine Learning
Scikit-learn - ML алгоритмы

MLflow - трекинг экспериментов

DVC - версионирование данных

Pandas & NumPy - обработка данных

Web & API
FastAPI - современный REST API

Uvicorn - ASGI сервер

Pydantic - валидация данных

DevOps & Monitoring
Docker - контейнеризация

GitHub Actions - CI/CD

pytest - тестирование

flake8 - линтинг кода

# Заключение
Credit Scoring ML Pipeline - это решение для автоматизации кредитного скоринга

 Ключевые преимущества:
* Полная автоматизация - от данных до API

* Профессиональные метрики - ROC-AUC 0.75+

* Готовность к продакшн - Docker, CI/CD, мониторинг

* Комплексное тестирование - unit-тесты, линтинг

* Мониторинг дрифта - PSI метрики, алертинг

* Исчерпывающая документация - для разработчиков и пользователей
