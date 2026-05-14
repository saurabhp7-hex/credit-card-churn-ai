# Deployment Architecture

This document outlines the enterprise-grade deployment architecture for the AI-Powered Credit Card Churn Prediction & Retention Intelligence Platform.

## 🏗️ System Overview

The system is designed for a hybrid-cloud environment, leveraging containerization for portability and scalability.

### 1. Data Layer
- **Source Systems**: Core Banking Systems (SQL/NoSQL) and CRM databases.
- **Data Lake/Warehouse**: Azure Data Lake Storage (ADLS) or AWS S3 for raw data landing.
- **Feature Store**: Redis or specialized feature stores for low-latency feature retrieval during inference.

### 2. Processing Layer (Batch/Real-time)
- **Batch Processing**: Apache Spark or Azure Data Factory pipelines for daily/weekly training and bulk predictions.
- **Real-time Inference**: FastAPI wrapper around the ML model, deployed on Azure Kubernetes Service (AKS) or AWS EKS.

### 3. Model Management (MLOps)
- **Model Registry**: MLflow or Azure ML Model Registry to track versions (e.g., `churn_model_v1.pkl`).
- **Explainability Agent**: Integrated SHAP computation as part of the inference pipeline.
- **Monitoring Agent**: Prometheus and Grafana for system health; Custom `monitoring.py` logic for detecting data and model drift.

### 4. Presentation Layer
- **Dashboard**: Streamlit application deployed via Docker containers.
- **BI Integration**: PowerBI or Tableau dashboards consuming results from the output database.

## 🚀 Deployment Workflow

1. **CI/CD Pipeline**: GitHub Actions or Azure DevOps handles testing, linting, and building Docker images.
2. **Containerization**:
   - `docker build -t churn-intelligence:latest .`
   - `docker push <registry>/churn-intelligence:latest`
3. **Orchestration**: Kubernetes manifest or Helm charts to deploy the dashboard and prediction API.
4. **Governance**: Integration with enterprise Active Directory for Role-Based Access Control (RBAC).

## 📊 Scalability & Security
- **Horizontal Pod Autoscaling (HPA)** for the inference API.
- **Encryption at rest** for the rule book and customer datasets.
- **VNET Isolation** to ensure data stays within the corporate perimeter.
