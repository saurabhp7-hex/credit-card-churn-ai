import streamlit as st
import pandas as pd
import joblib

# Import your modules (NO training here)
from preprocessing import load_data, clean_data, encode_features
from business import calculate_clv, revenue_at_risk
from retention import assign_strategy, simulate_campaign

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")

st.title("💳 Credit Card Churn & Retention Intelligence Platform")

# ========================
# LOAD MODEL (CACHED)
# ========================
@st.cache_resource
def load_model():
    return joblib.load("../models/churn_model.pkl")

model = load_model()

# ========================
# PIPELINE FUNCTION (FAST)
# ========================
@st.cache_data
def run_pipeline():
    # Step 1: Load data
    df = load_data("../data/bank_churners.csv")

    # Step 2: Clean + preprocess
    df = clean_data(df)
    df = encode_features(df)

    # Step 3: Predict (NO TRAINING HERE)
    X = df.drop('target', axis=1)
    df['churn_probability'] = model.predict_proba(X)[:, 1]

    # Step 4: Business logic
    df = calculate_clv(df)
    df = revenue_at_risk(df)

    # Step 5: Retention engine
    df = assign_strategy(df)
    df = simulate_campaign(df)

    return df


# ========================
# RUN BUTTON
# ========================
if st.button("🚀 Run Churn Analysis"):

    with st.spinner("Running AI pipeline..."):

        df = run_pipeline()

    st.success("✅ Analysis Completed!")

    # ========================
    # KPIs
    # ========================
    st.subheader("📊 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", len(df))
    col2.metric("Avg Churn Probability", round(df['churn_probability'].mean(), 2))
    col3.metric("Revenue at Risk", int(df['revenue_at_risk'].sum()))
    col4.metric("Total ROI", int(df['roi'].sum()))

    st.divider()

    # ========================
    # CHURN DISTRIBUTION
    # ========================
    st.subheader("📉 Churn Probability Distribution")
    st.bar_chart(df['churn_probability'])

    # ========================
    # FILTER
    # ========================
    st.subheader("🔍 Filter Customers")

    risk_range = st.slider("Select Churn Probability Range", 0.0, 1.0, (0.5, 1.0))

    filtered_df = df[
        (df['churn_probability'] >= risk_range[0]) &
        (df['churn_probability'] <= risk_range[1])
    ]

    st.write(filtered_df.head(20))

    # ========================
    # STRATEGY ANALYSIS
    # ========================
    st.subheader("🎯 Retention Strategy Distribution")
    st.bar_chart(df['retention_strategy'].value_counts())

    # ========================
    # ROI DISTRIBUTION
    # ========================
    st.subheader("💰 ROI Analysis")
    st.bar_chart(df['roi'])

    # ========================
    # HIGH RISK CUSTOMERS
    # ========================
    st.subheader("⚠️ High Risk Customers")

    high_risk = df[df['churn_probability'] > 0.7]

    st.write(high_risk[['churn_probability', 'CLV', 'retention_strategy']].head(10))

    # ========================
    # TOP REVENUE RISK
    # ========================
    st.subheader("🏆 Top Revenue-at-Risk Customers")

    top_customers = df.sort_values(by="revenue_at_risk", ascending=False).head(10)

    st.write(top_customers[['churn_probability', 'CLV', 'revenue_at_risk', 'retention_strategy']])

    # ========================
    # BUSINESS INSIGHT
    # ========================
    st.subheader("🧠 Key Business Insight")

    st.info(
        f"High-risk customers (>0.7 churn probability) contribute ₹{int(high_risk['revenue_at_risk'].sum())} "
        "in potential revenue loss. Targeted retention strategies can significantly improve ROI."
    )