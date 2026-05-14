import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def detect_drift(reference_data, current_data, features):
    """
    Detect drift in features using the Kolmogorov-Smirnov test.
    Returns features that have significant drift.
    """
    drift_results = {}
    
    for feature in features:
        if feature in reference_data.columns and feature in current_data.columns:
            stat, p_value = ks_2samp(reference_data[feature], current_data[feature])
            drift_results[feature] = {
                "ks_stat": round(stat, 4),
                "p_value": round(p_value, 4),
                "drift_detected": p_value < 0.05
            }
            
    return drift_results

def monitor_performance(y_true, y_pred):
    """
    Track precision/recall drift over time (simplified).
    """
    from sklearn.metrics import precision_score, recall_score
    
    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }
    return metrics

def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI) calculation.
    """
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.percentile(expected, breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # To avoid division by zero
    expected_percents = np.clip(expected_percents, a_min=0.0001, a_max=None)
    actual_percents = np.clip(actual_percents, a_min=0.0001, a_max=None)

    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    
    return psi_value
