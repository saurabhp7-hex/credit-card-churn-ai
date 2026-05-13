# AI-Powered Credit Card Churn Prediction & Retention Intelligence Platform

## 📌 Overview

This project builds an end-to-end AI system to predict credit card customer churn and optimize retention strategies. It combines machine learning, business intelligence, and customer analytics to help banks reduce churn and maximize revenue.

---

## 🎯 Objectives

- Predict customer churn proactively
- Calculate Customer Lifetime Value (CLV)
- Prioritize high-value customers
- Recommend retention strategies
- Optimize marketing ROI
- Provide explainable AI insights
- Build a real-time dashboard

---

## 🏗️ Project Structure

```
credit_card_churn_project/
│
├── data/                # Input dataset
├── models/              # Saved ML models
├── output/              # Generated results
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── business.py
│   ├── retention.py
│   ├── monitoring.py
│   ├── dashboard.py
│   └── main.py
└── README.md
```

---

## 🔧 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn / XGBoost
- SHAP (Explainability)
- Streamlit (Dashboard)

---

## ⚙️ How to Run

### 1. Train Model (One-time)

```
cd src
python main.py
```

### 2. Run Dashboard

```
streamlit run dashboard.py
```

---

## 📊 Features Implemented

### ✅ Churn Prediction

- ML model predicts churn probability

### ✅ Feature Engineering

- Behavioral features like spend trends, utilization, engagement score

### ✅ Explainability

- Feature importance and churn drivers

### ✅ Financial Intelligence

- CLV calculation
- Revenue at risk estimation

### ✅ Retention Engine

- Strategy recommendation (cashback, fee waiver, RM call)
- ROI optimization

### ✅ Monitoring

- Model performance tracking
- Basic drift detection

### ✅ Dashboard

- Interactive Streamlit dashboard
- KPIs, segmentation, insights

---

## 📈 Sample Output Metrics

- Churn Probability
- CLV
- Revenue at Risk
- Retention Strategy
- ROI

---

## 🧠 Business Impact

- Reduce customer churn
- Improve retention ROI
- Target high-value customers effectively
- Enable data-driven decision making

---

## 🚀 Future Enhancements

- Azure ML deployment
- CRM integration
- Real-time prediction API
- Advanced monitoring pipelines

---
