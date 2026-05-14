from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import shap
import numpy as np
import pandas as pd

def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    
    return model, X_test, y_test, X, explainer


def get_risk_segment(probability):
    if probability > 0.7:
        return "High Risk"
    elif probability > 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


def add_predictions(df, model, X):
    df['churn_probability'] = model.predict_proba(X)[:, 1]
    df['risk_segment'] = df['churn_probability'].apply(get_risk_segment)
    return df


def calculate_fairness(df, protected_col='Gender_M'):
    """
    Simple Demographic Parity check.
    Compares churn probability averages across groups.
    """
    if protected_col not in df.columns:
        return "Protected column not found"
        
    group_metrics = df.groupby(protected_col)['churn_probability'].mean().to_dict()
    
    # Calculate diff
    values = list(group_metrics.values())
    parity_diff = abs(values[0] - values[1]) if len(values) > 1 else 0
    
    return {
        "group_means": group_metrics,
        "parity_difference": parity_diff,
        "is_fair": parity_diff < 0.1  # Threshold from rules.json
    }


def get_shap_values(explainer, X_row):
    shap_values = explainer.shap_values(X_row)
    # For RandomForest in SHAP, shap_values is a list for each class. 
    # Class 1 is index 1.
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values

def save_model(model):
    joblib.dump(model, "../models/churn_model.pkl")
