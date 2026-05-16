import pandas as pd
import numpy as np
from scipy.stats import ks_2samp # Kolmogorov-Smirnov test for statistical distributions

def detect_drift(reference_data, current_data, features): # Monitor changes in feature distributions
    """
    Detects data drift in specific features using the Kolmogorov-Smirnov (KS) test.
    Compares the current production data distribution against a baseline (reference) dataset.
    Returns a dictionary of drift statistics and a boolean flag (p < 0.05).
    """
    drift_results = {}
    
    for feature in features:
        # Only compare features present in both datasets
        if feature in reference_data.columns and feature in current_data.columns:
            stat, p_value = ks_2samp(reference_data[feature], current_data[feature])
            drift_results[feature] = {
                "ks_stat": round(stat, 4),
                "p_value": round(p_value, 4),
                "drift_detected": p_value < 0.05 # True if distribution has significantly shifted
            }
            
    return drift_results

def monitor_performance(y_true, y_pred): # Track model accuracy degradation
    """
    Calculates key performance metrics (Precision and Recall) to monitor 
    model quality over time. Significant drops may indicate model drift.
    """
    from sklearn.metrics import precision_score, recall_score
    
    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred)
    }
    return metrics

def calculate_psi(expected, actual, buckets=10): # Quantify population shift
    """
    Calculates the Population Stability Index (PSI) to measure how much the 
    population distribution has changed between two points in time.
    PSI > 0.1 indicates slight change, PSI > 0.2 indicates significant shift.
    """
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    # Define bin breakpoints based on the expected (baseline) distribution
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.percentile(expected, breakpoints)
    breakpoints[0] = -np.inf # Ensure full range coverage
    breakpoints[-1] = np.inf

    # Calculate percentage of population in each bucket
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Apply clipping to avoid division by zero or log(0) errors
    expected_percents = np.clip(expected_percents, a_min=0.0001, a_max=None)
    actual_percents = np.clip(actual_percents, a_min=0.0001, a_max=None)

    # PSI Formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    
    return psi_value
