import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Dict, Any, Tuple
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Детектор дрифта данных и моделей
    """
    
    def __init__(self, reference_data: pd.DataFrame, psi_threshold: float = 0.1):
        self.reference_data = reference_data
        self.psi_threshold = psi_threshold
        self.categorical_columns = self._identify_categorical_columns()
        
    def _identify_categorical_columns(self) -> list:
        """Идентифицирует категориальные колонки"""
        categorical_cols = []
        for col in self.reference_data.columns:
            if self.reference_data[col].dtype == 'object':
                categorical_cols.append(col)
            elif self.reference_data[col].nunique() < 20:  # Малое количество уникальных значений
                categorical_cols.append(col)
        return categorical_cols
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Parameters:
        -----------
        expected : reference distribution
        actual : current distribution  
        buckets : number of bins for histogram
        
        Returns:
        --------
        psi_value : float
        """
        # Handle infinite values
        expected = expected[np.isfinite(expected)]
        actual = actual[np.isfinite(actual)]
        
        if len(expected) == 0 or len(actual) == 0:
            return float('inf')
            
        # Create buckets based on reference data
        breakpoints = np.nanpercentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates
        
        # Handle case with too few unique breakpoints
        if len(breakpoints) < 2:
            breakpoints = np.linspace(np.min(expected), np.max(expected), buckets + 1)
        
        # Calculate histograms
        expected_hist, _ = np.histogram(expected, bins=breakpoints)
        actual_hist, _ = np.histogram(actual, bins=breakpoints)
        
        # Add small value to avoid division by zero
        expected_hist = expected_hist.astype(float) + 0.0001
        actual_hist = actual_hist.astype(float) + 0.0001
        
        # Normalize to probabilities
        expected_probs = expected_hist / np.sum(expected_hist)
        actual_probs = actual_hist / np.sum(actual_hist)
        
        # Calculate PSI
        psi_value = 0
        for i in range(len(expected_probs)):
            if expected_probs[i] > 0 and actual_probs[i] > 0:
                psi_value += (actual_probs[i] - expected_probs[i]) * np.log(actual_probs[i] / expected_probs[i])
        
        return max(0, psi_value)  # PSI should be non-negative
    
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Kullback-Leibler divergence"""
        p = p[np.isfinite(p)]
        q = q[np.isfinite(q)]
        
        if len(p) == 0 or len(q) == 0:
            return float('inf')
            
        # Create bins based on combined data
        combined = np.concatenate([p, q])
        bins = np.linspace(np.min(combined), np.max(combined), 20)
        
        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)
        
        # Add epsilon to avoid zeros
        p_hist = p_hist + 1e-10
        q_hist = q_hist + 1e-10
        
        # Normalize
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)
        
        return np.sum(p_hist * np.log(p_hist / q_hist))
    
    def detect_feature_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect feature drift between reference and current data
        
        Returns:
        --------
        drift_report : dict with drift metrics per feature
        """
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'features': {},
            'summary': {
                'total_features': len(self.reference_data.columns),
                'drifted_features': 0,
                'max_psi': 0,
                'drift_detected': False
            }
        }
        
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                logger.warning(f"Column {column} not found in current data")
                continue
                
            ref_col = self.reference_data[column].dropna()
            curr_col = current_data[column].dropna()
            
            if len(ref_col) == 0 or len(curr_col) == 0:
                logger.warning(f"Not enough data for column {column}")
                continue
            
            # Calculate PSI
            psi_value = self.calculate_psi(ref_col.values, curr_col.values)
            
            # For categorical data, also calculate chi-square
            chi2_stat, p_value = None, None
            if column in self.categorical_columns:
                try:
                    # Create contingency table
                    ref_counts = ref_col.value_counts()
                    curr_counts = curr_col.value_counts()
                    
                    # Align indices
                    all_categories = list(set(ref_counts.index) | set(curr_counts.index))
                    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                    curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
                    
                    if sum(ref_aligned) > 0 and sum(curr_aligned) > 0:
                        chi2_stat, p_value = stats.chisquare(curr_aligned, ref_aligned)
                except Exception as e:
                    logger.warning(f"Chi-square calculation failed for {column}: {e}")
            
            drift_detected = psi_value > self.psi_threshold
            
            drift_report['features'][column] = {
                'psi': float(psi_value),
                'drift_detected': drift_detected,
                'chi2_statistic': float(chi2_stat) if chi2_stat else None,
                'p_value': float(p_value) if p_value else None,
                'reference_mean': float(ref_col.mean()),
                'current_mean': float(curr_col.mean()),
                'reference_std': float(ref_col.std()),
                'current_std': float(curr_col.std())
            }
            
            if drift_detected:
                drift_report['summary']['drifted_features'] += 1
                drift_report['summary']['max_psi'] = max(drift_report['summary']['max_psi'], psi_value)
        
        drift_report['summary']['drift_detected'] = drift_report['summary']['drifted_features'] > 0
        
        return drift_report
    
    def save_drift_report(self, report: Dict[str, Any], filepath: str):
        """Save drift report to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Drift report saved to {filepath}")