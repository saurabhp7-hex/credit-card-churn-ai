# Rule Book Documentation

This document explains the business logic and governance rules used in the Credit Card Churn Prediction & Retention Intelligence Platform.

## 📋 Churn Definition
- **Logic**: `Attrition_Flag == 'Attrited Customer'`
- **Description**: Customers who have explicitly closed their accounts are labeled as "Attrited". The model learns to predict this state proactively.

## 📊 Risk Segmentation
- **High Risk**: Churn probability > 70%. Requires immediate intervention.
- **Medium Risk**: Churn probability between 30% and 70%. Requires proactive monitoring.
- **Low Risk**: Churn probability < 30%. Stable customer segment.

## 💰 Customer Lifetime Value (CLV)
- **Calculation**: `Total_Trans_Amt * 0.2` (Approximation based on interchange fees and interest income).
- **High Value**: CLV > ₹10,000.
- **Medium Value**: CLV > ₹5,000.

## 🎯 Retention Strategies
1. **Fee Waiver**: Offsetting annual fees for high-value customers at high risk.
2. **Cashback**: Small financial incentives for standard customers at high risk.
3. **RM Call**: Personalized outreach for very high-value customers showing early churn signals.

## ⚖️ Governance & Compliance
- **Fairness Threshold**: Maximum allowable difference in churn prediction rates across demographic groups (e.g., Gender) is 10%.
- **Protected Attributes**: Gender, Marital Status.
- **Transparency**: Every prediction must be accompanied by SHAP-based feature importance.
