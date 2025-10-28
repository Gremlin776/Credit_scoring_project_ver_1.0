import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
import sys

# Добавляем путь для импортов
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedDriftDetector:
    def __init__(self):
        self.reference_stats = None
        
    def load_reference_data(self):
        """Загрузка референсных данных"""
        try:
            data_path = Path('data/processed/data_with_features.csv')
            if not data_path.exists():
                logger.error("Референсные данные не найдены")
                return None
                
            df = pd.read_csv(data_path)
            logger.info(f"Загружены референсные данные: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки референсных данных: {e}")
            return None
    
    def calculate_psi(self, expected, actual, buckets=10):
        """Расчет Population Stability Index (PSI)"""
        try:
            expected = expected[~np.isnan(expected)]
            actual = actual[~np.isnan(actual)]
            
            if len(expected) == 0 or len(actual) == 0:
                return float('inf')
                
            breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
            breakpoints = np.unique(breakpoints)
            
            expected_hist, _ = np.histogram(expected, bins=breakpoints)
            actual_hist, _ = np.histogram(actual, bins=breakpoints)
            
            expected_hist = expected_hist.astype(float) + 0.0001
            actual_hist = actual_hist.astype(float) + 0.0001
            
            expected_probs = expected_hist / np.sum(expected_hist)
            actual_probs = actual_hist / np.sum(actual_hist)
            
            psi = np.sum((actual_probs - expected_probs) * np.log(actual_probs / expected_probs))
            return max(0, psi)
            
        except Exception as e:
            logger.error(f"Ошибка расчета PSI: {e}")
            return float('inf')
    
    def detect_drift(self):
        """Основная функция детектирования дрифта"""
        reference_df = self.load_reference_data()
        if reference_df is None:
            return {"status": "error", "message": "Не удалось загрузить референсные данные"}
        
        current_df = reference_df.copy()
        
        # Добавляем небольшой дрифт
        np.random.seed(42)
        drift_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1']
        for feature in drift_features:
            if feature in current_df.columns:
                noise = np.random.normal(0, 0.1, len(current_df))
                current_df[feature] = current_df[feature] * (1 + noise)
        
        # Анализ дрифта
        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "status": "analyzed",
            "features_analyzed": [],
            "summary": {
                "total_features": len(reference_df.columns),
                "features_with_drift": 0,
                "max_psi": 0.0  # Гарантируем float
            }
        }
        
        numeric_columns = reference_df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns[:10]:
            try:
                psi_value = self.calculate_psi(
                    reference_df[column].values,
                    current_df[column].values
                )
                
                # ИСПРАВЛЕНО: преобразуем bool в int для JSON
                drift_detected = bool(psi_value > 0.1)
                
                feature_report = {
                    "feature": column,
                    "psi": float(psi_value),
                    "drift_detected": int(drift_detected),  # bool -> int
                    "reference_mean": float(reference_df[column].mean()),
                    "current_mean": float(current_df[column].mean())
                }
                
                drift_report["features_analyzed"].append(feature_report)
                
                if drift_detected:
                    drift_report["summary"]["features_with_drift"] += 1
                    drift_report["summary"]["max_psi"] = max(
                        drift_report["summary"]["max_psi"], 
                        float(psi_value)
                    )
                    
            except Exception as e:
                logger.warning(f"Не удалось проанализировать признак {column}: {e}")
        
        # Сохраняем отчет с правильной кодировкой
        reports_dir = Path('reports')
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # ИСПРАВЛЕНО: правильная кодировка для Windows
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(drift_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Отчет дрифта сохранен: {report_path}")
        
        if drift_report["summary"]["features_with_drift"] > 0:
            logger.warning(f"Обнаружен дрифт в {drift_report['summary']['features_with_drift']} признаках")
        else:
            logger.info("Дрифт не обнаружен")
            
        return drift_report

def main():
    """Основная функция"""
    logger.info("Запуск мониторинга дрифта...")
    
    detector = AdvancedDriftDetector()
    report = detector.detect_drift()
    
    print(f"РЕЗУЛЬТАТЫ МОНИТОРИНГА ДРИФТА:")
    print(f"Статус: {report['status']}")
    print(f"Проанализировано признаков: {len(report['features_analyzed'])}")
    print(f"Признаков с дрифтом: {report['summary']['features_with_drift']}")

if __name__ == "__main__":
    main()