#!/usr/bin/env python3
"""
FastAPI приложение для Credit Scoring ML Pipeline
"""

import json
import os
import subprocess
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Добавляем пути для импорта
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Глобальная переменная для модели
model_predictor = None


def load_model():
    """Загрузка модели"""
    global model_predictor
    try:
        print("Пытаемся загрузить модель...")

        # Пробуем разные варианты импорта
        try:
            from models.predict import ModelPredictor
            print("Импорт через 'models.predict' успешен")
        except ImportError:
            try:
                from src.models.predict import ModelPredictor
                print("Импорт через 'src.models.predict' успешен")
            except ImportError:
                # Прямой импорт по пути
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "predict", project_root / "src" / "models" / "predict.py"
                )
                predict_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(predict_module)
                ModelPredictor = predict_module.ModelPredictor
                print("Прямой импорт из файла успешен")

        model_predictor = ModelPredictor()

        if model_predictor.model_loaded:
            print("=" * 50)
            print("МОДЕЛЬ УСПЕШНО ЗАГРУЖЕНА В API!")
            print("=" * 50)
        else:
            print("=" * 50)
            print("МОДЕЛЬ НЕ ЗАГРУЖЕНА В API!")
            print("=" * 50)

    except Exception as e:
        print(f"Критическая ошибка загрузки модели: {e}")
        import traceback

        traceback.print_exc()
        model_predictor = None


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager для замены on_event"""
    # Startup
    load_model()
    yield
    # Shutdown


# Инициализация FastAPI приложения с lifespan
app = FastAPI(
    title="Credit Scoring API",
    description="REST API для предсказания вероятности дефолта клиента",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Монтирование статических файлов
app.mount("/static", StaticFiles(directory="reports"), name="static")


class CreditData(BaseModel):
    """Модель данных для кредитного скоринга"""

    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


class PredictionResponse(BaseModel):
    """Модель ответа для предсказания"""

    prediction: int
    probability: float
    default_probability: float
    status: str


class HealthResponse(BaseModel):
    """Модель ответа для health check"""

    status: str
    model_loaded: bool
    message: str


@app.get("/", summary="Корневой endpoint", tags=["General"])
async def root():
    """Корневой endpoint API"""
    return {
        "message": "Credit Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "reports": "/reports",
        "dashboard": "/dashboard",
        "monitoring": "/monitoring/drift",
        "testing": "/testing/status",
    }


@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["General"])
async def health_check():
    """Проверка здоровья API и загрузки модели"""
    model_loaded = model_predictor is not None and model_predictor.model_loaded

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        message=(
            "API работает с обученной моделью"
            if model_loaded
            else "API в режиме демонстрации (модель не загружена)"
        ),
    )


@app.post("/predict", response_model=PredictionResponse, summary="Предсказание дефолта", tags=["Prediction"])
async def predict(data: CreditData):
    """
    Предсказание вероятности дефолта клиента

    - **LIMIT_BAL**: Кредитный лимит
    - **SEX**: Пол (1=мужчина, 2=женщина)
    - **EDUCATION**: Образование (1=аспирант, 2=университет, 3=школа, 4=другое)
    - **MARRIAGE**: Семейное положение (1=женат/замужем, 2=холост/не замужем, 3=другое)
    - **AGE**: Возраст
    - **PAY_X**: История платежей (от -2 до 8)
    - **BILL_AMTX**: Суммы счетов
    - **PAY_AMTX**: Суммы платежей
    """
    if model_predictor is None or not model_predictor.model_loaded:
        # Демонстрационный режим если модель не загружена
        return PredictionResponse(
            prediction=0,
            probability=0.1,
            default_probability=0.1,
            status="demo_mode",
        )

    try:
        input_dict = data.dict()
        result = model_predictor.predict(input_dict)

        return PredictionResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            default_probability=result["default_probability"],
            status="success",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


@app.post("/batch_predict", summary="Пакетное предсказание", tags=["Prediction"])
async def batch_predict(data: List[CreditData]):
    """Пакетное предсказание для нескольких клиентов"""
    if model_predictor is None or not model_predictor.model_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        input_list = [item.dict() for item in data]
        results = []

        for input_data in input_list:
            result = model_predictor.predict(input_data)
            results.append(result)

        return {
            "predictions": results,
            "count": len(results),
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка пакетного предсказания: {str(e)}")


@app.get("/model_info", summary="Информация о модели", tags=["Model"])
async def model_info():
    """Получение информации о загруженной модели"""
    try:
        model_info_paths = [
            "reports/best_model_info.json",
            "reports/feature_importance.json",
        ]

        info_data = {}

        for path in model_info_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    key = Path(path).stem
                    info_data[key] = json.load(f)

        if info_data:
            return {
                "status": "success",
                "model_info": info_data,
            }
        else:
            return {
                "status": "no_info",
                "message": "Информация о модели не найдена",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения информации: {str(e)}")


@app.get("/reports", summary="Доступ к отчетам и графикам", tags=["Reports"])
async def get_reports():
    """Возвращает список доступных отчетов и графиков"""
    reports = []

    if os.path.exists("reports"):
        for file in os.listdir("reports"):
            if file.endswith((".png", ".jpg", ".jpeg", ".pdf", ".txt", ".json", ".html")):
                file_path = f"reports/{file}"
                file_size = os.path.getsize(file_path)
                reports.append({
                    "name": file,
                    "url": f"/static/{file}",
                    "size": file_size,
                    "type": file.split(".")[-1],
                })

    return {
        "reports": reports,
        "count": len(reports),
    }


@app.get("/reports/{report_name}", summary="Получить конкретный отчет", tags=["Reports"])
async def get_report(report_name: str):
    """Возвращает конкретный отчет по имени файла"""
    report_path = f"reports/{report_name}"

    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Отчет не найден")

    # Проверяем тип файла для правильного Content-Type
    if report_name.endswith(".png"):
        media_type = "image/png"
    elif report_name.endswith(".jpg") or report_name.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif report_name.endswith(".pdf"):
        media_type = "application/pdf"
    elif report_name.endswith(".json"):
        media_type = "application/json"
    elif report_name.endswith(".txt"):
        media_type = "text/plain"
    elif report_name.endswith(".html"):
        media_type = "text/html"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        report_path,
        media_type=media_type,
        filename=report_name,
    )


@app.get("/dashboard", summary="Дашборд с графиками", tags=["Reports"])
async def get_dashboard():
    """Возвращает HTML дашборд со всеми графиками"""
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Scoring Dashboard</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 2px solid #eee;
            }
            .chart { 
                margin: 20px 0; 
                border: 1px solid #ddd; 
                padding: 15px;
                border-radius: 8px;
                background: white;
            }
            img { 
                max-width: 100%; 
                height: auto; 
                border: 1px solid #eee;
                border-radius: 4px;
            }
            .report-list { 
                background: #f8f9fa; 
                padding: 15px; 
                margin: 10px 0;
                border-radius: 8px;
            }
            .actions {
                margin: 20px 0;
                padding: 15px;
                background: #e9ecef;
                border-radius: 8px;
            }
            .actions h3 {
                margin-top: 0;
            }
            button {
                background: #007bff;
                color: white;
                border: none;
                padding: 10px 15px;
                margin: 5px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
            }
            button:hover {
                background: #0056b3;
            }
            .status {
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                font-weight: bold;
            }
            .status.healthy {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.degraded {
                background: #fff3cd;
                color: #856404;
                border: 1px solid #ffeaa7;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Credit Scoring Dashboard</h1>
                <div id="api-status" class="status"></div>
            </div>

            <div class="actions">
                <h3>🚀 Действия</h3>
                <button onclick="generateFeatureImportance()">
                    📈 Создать график важности признаков
                </button>
                <button onclick="runTests()">🧪 Запустить тесты</button>
                <button onclick="checkDrift()">📊 Проверить дрифт данных</button>
                <button onclick="reloadModel()">🔄 Перезагрузить модель</button>
            </div>

            <div class="report-list">
                <h3>📁 Доступные отчеты:</h3>
                <div id="reports-list"></div>
            </div>

            <div class="chart">
                <h2>🎯 Распределение целевой переменной</h2>
                <img src="/static/target_distribution.png" 
                     alt="Target Distribution" 
                     onerror="this.alt='График не доступен'">
            </div>

            <div class="chart">
                <h2>📈 Матрица корреляций</h2>
                <img src="/static/correlation_matrix.png" 
                     alt="Correlation Matrix"
                     onerror="this.alt='График не доступен'">
            </div>

            <div class="chart">
                <h2>📊 Распределение ключевых признаков</h2>
                <img src="/static/key_features_distributions.png" 
                     alt="Key Features"
                     onerror="this.alt='График не доступен'">
            </div>

            <div class="chart">
                <h2>📉 Оценка модели</h2>
                <img src="/static/model_evaluation.png" 
                     alt="Model Evaluation"
                     onerror="this.alt='График не доступен'">
            </div>

            <div class="chart">
                <h2>⭐ Важность признаков</h2>
                <img src="/static/best_model_feature_importance.png" 
                     alt="Feature Importance" 
                     onerror="this.style.display='none'; 
                     document.getElementById('feature-importance-warning').style.display='block';">
                <div id="feature-importance-warning" 
                     style="display: none; color: #856404; background: #fff3cd; 
                     padding: 10px; border-radius: 5px; margin: 10px 0;">
                    ⚠️ График важности признаков не сгенерирован. 
                    Нажмите "Создать график важности признаков" выше.
                </div>
            </div>

            <div class="chart">
                <h2>🔍 Детали модели</h2>
                <div id="model-info"></div>
            </div>
        </div>

        <script>
            // Загрузка статуса API
            fetch('/health')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('api-status');
                    statusDiv.textContent = `Статус API: ${data.status.toUpperCase()} | ${data.message}`;
                    statusDiv.className = `status ${data.status === 'healthy' ? 'healthy' : 'degraded'}`;
                });

            // Динамическая загрузка списка отчетов
            fetch('/reports')
                .then(response => response.json())
                .then(data => {
                    const reportsList = document.getElementById('reports-list');
                    if (data.reports.length === 0) {
                        reportsList.innerHTML = '<p>Нет доступных отчетов</p>';
                    } else {
                        data.reports.forEach(report => {
                            const div = document.createElement('div');
                            div.style.margin = '5px 0';
                            div.style.padding = '8px';
                            div.style.background = '#fff';
                            div.style.borderRadius = '4px';
                            div.style.border = '1px solid #ddd';
                            
                            const link = document.createElement('a');
                            link.href = report.url;
                            link.textContent = `📄 ${report.name} (${(report.size/1024).toFixed(1)} KB)`;
                            link.target = '_blank';
                            link.style.textDecoration = 'none';
                            link.style.color = '#007bff';
                            
                            div.appendChild(link);
                            reportsList.appendChild(div);
                        });
                    }
                });

            // Загрузка информации о модели
            fetch('/model_info')
                .then(response => response.json())
                .then(data => {
                    const modelInfoDiv = document.getElementById('model-info');
                    if (data.status === 'success') {
                        let html = '<div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">';
                        if (data.model_info.best_model_info) {
                            const info = data.model_info.best_model_info;
                            html += `<p><strong>Лучшая модель:</strong> ${info.best_experiment || 'Неизвестно'}</p>`;
                            html += `<p><strong>ROC AUC:</strong> ${info.best_score ? info.best_score.toFixed(4) : 'Неизвестно'}</p>`;
                        }
                        if (data.model_info.feature_importance) {
                            const fi = data.model_info.feature_importance;
                            html += `<p><strong>Всего признаков:</strong> ${fi.total_features}</p>`;
                            html += `<p><strong>Максимальная важность:</strong> ${fi.max_importance ? fi.max_importance.toFixed(4) : 'Неизвестно'}</p>`;
                        }
                        html += '</div>';
                        modelInfoDiv.innerHTML = html;
                    } else {
                        modelInfoDiv.innerHTML = '<p>Информация о модели не доступна</p>';
                    }
                });

            function generateFeatureImportance() {
                showLoading('Создание графика важности признаков...');
                fetch('/generate_feature_importance', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        hideLoading();
                        alert(data.message);
                        if (data.success) {
                            setTimeout(() => location.reload(), 1000);
                        }
                    })
                    .catch(error => {
                        hideLoading();
                        alert('Ошибка: ' + error);
                    });
            }
            
            function runTests() {
                showLoading('Запуск тестов...');
                fetch('/testing/run', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        hideLoading();
                        alert('Тесты завершены. Код возврата: ' + data.return_code);
                        if (data.output) {
                            console.log('Вывод тестов:', data.output);
                        }
                    })
                    .catch(error => {
                        hideLoading();
                        alert('Ошибка запуска тестов: ' + error);
                    });
            }
            
            function checkDrift() {
                showLoading('Проверка дрифта данных...');
                fetch('/monitoring/drift')
                    .then(response => response.json())
                    .then(data => {
                        hideLoading();
                        if (data.status === 'success') {
                            const driftCount = data.report.summary.features_with_drift;
                            const maxPsi = data.report.summary.max_psi;
                            alert(`Проверка дрифта завершена. Признаков с дрифтом: ${driftCount}, Максимальный PSI: ${maxPsi.toFixed(3)}`);
                        } else {
                            alert('Не удалось проверить дрифт: ' + data.message);
                        }
                    })
                    .catch(error => {
                        hideLoading();
                        alert('Ошибка проверки дрифта: ' + error);
                    });
            }

            function reloadModel() {
                showLoading('Перезагрузка модели...');
                fetch('/reload_model', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        hideLoading();
                        alert(data.message);
                        if (data.success) {
                            location.reload();
                        }
                    })
                    .catch(error => {
                        hideLoading();
                        alert('Ошибка перезагрузки модели: ' + error);
                    });
            }

            function showLoading(message) {
                alert(message + ' Пожалуйста, подождите...');
            }

            function hideLoading() {
                // Можно добавить более сложную логику скрытия loading
            }
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=dashboard_html)


@app.get("/sample_data", summary="Пример данных для тестирования", tags=["Testing"])
async def sample_data():
    """Возвращает пример данных для тестирования API"""
    sample = {
        "LIMIT_BAL": 150000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 42189,
        "BILL_AMT2": 41723,
        "BILL_AMT3": 40881,
        "BILL_AMT4": 39554,
        "BILL_AMT5": 38675,
        "BILL_AMT6": 37734,
        "PAY_AMT1": 3000,
        "PAY_AMT2": 3000,
        "PAY_AMT3": 3000,
        "PAY_AMT4": 3000,
        "PAY_AMT5": 3000,
        "PAY_AMT6": 3000,
    }
    return {"sample_data": sample}


@app.get("/sample_data/{scenario}", summary="Примеры данных по сценариям", tags=["Testing"])
async def sample_data_scenario(scenario: str = "good"):
    """Возвращает пример данных для разных сценариев"""
    scenarios = {
        "good": {
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
            "PAY_AMT6": 5000,
        },
        "risky": {
            "LIMIT_BAL": 50000,
            "SEX": 1,
            "EDUCATION": 3,
            "MARRIAGE": 2,
            "AGE": 25,
            "PAY_0": 2,
            "PAY_2": 2,
            "PAY_3": 1,
            "PAY_4": 1,
            "PAY_5": 0,
            "PAY_6": -1,
            "BILL_AMT1": 45000,
            "BILL_AMT2": 44000,
            "BILL_AMT3": 43000,
            "BILL_AMT4": 42000,
            "BILL_AMT5": 41000,
            "BILL_AMT6": 40000,
            "PAY_AMT1": 1000,
            "PAY_AMT2": 1000,
            "PAY_AMT3": 1000,
            "PAY_AMT4": 1000,
            "PAY_AMT5": 1000,
            "PAY_AMT6": 1000,
        },
        "default": {
            "LIMIT_BAL": 10000,
            "SEX": 1,
            "EDUCATION": 4,
            "MARRIAGE": 3,
            "AGE": 45,
            "PAY_0": 4,
            "PAY_2": 3,
            "PAY_3": 3,
            "PAY_4": 2,
            "PAY_5": 2,
            "PAY_6": 1,
            "BILL_AMT1": 9500,
            "BILL_AMT2": 9000,
            "BILL_AMT3": 8500,
            "BILL_AMT4": 8000,
            "BILL_AMT5": 7500,
            "BILL_AMT6": 7000,
            "PAY_AMT1": 0,
            "PAY_AMT2": 0,
            "PAY_AMT3": 0,
            "PAY_AMT4": 0,
            "PAY_AMT5": 0,
            "PAY_AMT6": 0,
        },
    }

    if scenario not in scenarios:
        available = list(scenarios.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Сценарий '{scenario}' не найден. Доступные: {available}",
        )

    descriptions = {
        "good": "Хороший заемщик - низкий риск",
        "risky": "Рискованный заемщик - средний риск",
        "default": "Проблемный заемщик - высокий риск дефолта",
    }

    return {
        "scenario": scenario,
        "description": descriptions[scenario],
        "data": scenarios[scenario],
    }


@app.get("/monitoring/drift", summary="Отчет о дрифте данных", tags=["Monitoring"])
async def get_drift_report():
    """Возвращает последний отчет о дрифте данных"""
    try:
        # Ищем последний отчет о дрифте
        reports_dir = Path("reports")
        drift_reports = list(reports_dir.glob("drift_report_*.json"))

        if not drift_reports:
            return {
                "status": "no_reports",
                "message": "Отчеты о дрифте не найдены",
                "suggestion": "Запустите скрипт мониторинга дрифта",
            }

        latest_report = max(drift_reports, key=os.path.getctime)

        with open(latest_report, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        return {
            "status": "success",
            "report": report_data,
            "report_file": latest_report.name,
            "generated_at": report_data.get("timestamp", "Неизвестно"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения отчета: {str(e)}")


@app.get("/testing/status", summary="Статус тестирования", tags=["Testing"])
async def get_testing_status():
    """Возвращает статус тестовой системы"""
    try:
        # Проверяем наличие тестов
        test_files = list(Path("tests").glob("test_*.py"))

        # Проверяем доступность pytest
        try:
            subprocess.run(["pytest", "--version"], capture_output=True, timeout=10)
            pytest_available = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest_available = False

        return {
            "status": "available",
            "test_files_count": len(test_files),
            "test_files": [f.name for f in test_files],
            "pytest_available": pytest_available,
            "last_run": "N/A",  
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")


@app.post("/testing/run", summary="Запуск тестов", tags=["Testing"])
async def run_tests():
    """Запускает выполнение тестов"""
    try:
        result = subprocess.run(
            ["pytest", "tests/", "-v"],
            capture_output=True,
            text=True,
            timeout=300,
        )

        return {
            "status": "completed",
            "return_code": result.returncode,
            "summary": f"Тесты завершены с кодом: {result.returncode}",
            "output": result.stdout[-1000:],  # Последние 1000 символов вывода
            "error": result.stderr[-500:] if result.stderr else None,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Тесты превысили лимит времени")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка запуска тестов: {str(e)}")


@app.post("/generate_feature_importance", summary="Создать график важности признаков", tags=["Reports"])
async def generate_feature_importance():
    """Запускает создание графика важности признаков"""
    try:
        # Проверяем существование скрипта
        script_path = Path("scripts/create_feature_importance.py")
        if not script_path.exists():
            return {
                "success": False,
                "message": "Скрипт создания графиков не найден",
            }

        result = subprocess.run(
            ["python", "scripts/create_feature_importance.py"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        success = result.returncode == 0

        return {
            "success": success,
            "message": (
                "График важности признаков создан"
                if success
                else f"Не удалось создать график: {result.stderr}"
            ),
            "output": result.stdout[-500:] if result.stdout else "Нет вывода",
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Создание графиков превысило лимит времени")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка создания графиков: {str(e)}")


@app.post("/reload_model", summary="Перезагрузить модель", tags=["Model"])
async def reload_model():
    """Перезагружает модель в памяти"""
    try:
        global model_predictor
        load_model()  # Перезагружаем модель

        model_loaded = model_predictor is not None and model_predictor.model_loaded

        return {
            "success": model_loaded,
            "message": (
                "Модель успешно перезагружена"
                if model_loaded
                else "Не удалось загрузить модель"
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка перезагрузки модели: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)