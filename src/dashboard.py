import streamlit as st
import pandas as pd
import joblib

from preprocessing import load_data, clean_data, encode_features
from business import calculate_clv, revenue_at_risk
from retention import assign_strategy, simulate_campaign


# ========================
# CONFIG
# ========================
st.set_page_config(page_title="Churn Intelligence", layout="wide")

st.title("💳 Credit Card Churn & Retention Intelligence Platform")
st.markdown("Simple decision dashboard for customer churn analysis.")


# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model():
    return joblib.load("../models/churn_model.pkl")

model = load_model()


# ========================
# PIPELINE
# ========================
def run_pipeline():
    df = load_data("../data/bank_churners.csv")

    df = clean_data(df)
    df = encode_features(df)

    X = df.drop("target", axis=1)

    # Prediction
    df["churn_probability"] = model.predict_proba(X)[:, 1]

    df["customer_id"] = df.index

    # Business logic
    df = calculate_clv(df)
    df = revenue_at_risk(df)
    df = assign_strategy(df)
    df = simulate_campaign(df)

    return df


# ========================
# RUN BUTTON
# ========================
if st.button("🚀 Run Analysis"):
    with st.spinner("Running analysis..."):
        st.session_state["data"] = run_pipeline()

    st.success("✅ Analysis Completed")


# ========================
# SHOW DATA IF EXISTS
# ========================
if "data" in st.session_state:

    df = st.session_state["data"]

    # ========================
    # KPIs
    # ========================
    st.subheader("📊 Key Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("High Risk Customers", len(df[df["churn_probability"] > 0.7]))
    col3.metric("Total Revenue at Risk", int(df["revenue_at_risk"].sum()))

    st.divider()

    # ========================
    # CUSTOMER LIST
    # ========================
    st.subheader("📋 Customer List")

    # High risk customers
    high_risk = df[df["churn_probability"] > 0.7].head(10)

    # Low risk customers
    low_risk = df[df["churn_probability"] < 0.3].head(10)

    # Combine
    preview_df = pd.concat([high_risk, low_risk])

    # Select required columns
    preview_df = preview_df[[
        "customer_id",
        "churn_probability",
        "CLV",
        "retention_strategy"
    ]]

    st.dataframe(preview_df, use_container_width=True)

    # ========================
    # SELECT CUSTOMER
    # ========================
    st.subheader("👆 Select Customer")

    selected_customer = st.radio(
        "Choose a customer",
        preview_df["customer_id"]
    )

    # ========================
    # CUSTOMER DETAILS
    # ========================
    customer = df[df["customer_id"] == selected_customer]

    st.subheader("👤 Customer Details")

    st.write({
    "Churn Probability": round(float(customer["churn_probability"].values[0]), 2),
    "Customer Value (CLV)": round(float(customer["CLV"].values[0]), 2),
    "Revenue at Risk": round(float(customer["revenue_at_risk"].values[0]), 2),
    "Recommended Strategy": customer["retention_strategy"].values[0],
    "Expected ROI": round(float(customer["roi"].values[0]), 2),
    })

    # ========================
    # EXPLANATION
    # ========================
    st.subheader("🧠 Customer Behavior Explanation")

    churn_prob = float(customer["churn_probability"].values[0])

    explanation_points = []

    # ========================
    # HIGH RISK
    # ========================
    if churn_prob > 0.7:

        if customer["Months_Inactive_12_mon"].values[0] > 3:
            explanation_points.append("Customer has been inactive for multiple months")

        if customer["Total_Trans_Ct"].values[0] < 40:
            explanation_points.append("Low transaction activity observed")

        if customer["Total_Amt_Chng_Q4_Q1"].values[0] < 1:
            explanation_points.append("Spending trend is declining")

        st.markdown("### ⚠️ High Churn Risk Drivers")

    # ========================
    # LOW RISK
    # ========================
    elif churn_prob < 0.3:

        if customer["Total_Trans_Ct"].values[0] > 60:
            explanation_points.append("Customer shows strong transaction activity")

        if customer["Avg_Utilization_Ratio"].values[0] > 0.5:
            explanation_points.append("Healthy credit utilization")

        if customer["Months_Inactive_12_mon"].values[0] <= 2:
            explanation_points.append("Customer is actively engaged")

        st.markdown("### ✅ Customer Retention Indicators")

    # ========================
    # MEDIUM RISK
    # ========================
    else:

        explanation_points.append("Customer shows moderate engagement and usage patterns")
        st.markdown("### ⚠️ Moderate Risk Indicators")

# ========================
# DISPLAY OUTPUT
# ========================
    for point in explanation_points:
        st.write(f"• {point}")

    st.info("Explanation is aligned with predicted churn behavior.")
        # ========================
        # INSIGHT
        # ========================
    st.subheader("📌 Key Insight")

    high_risk = df[df["churn_probability"] > 0.7]

    st.write(
        f"{len(high_risk)} high-risk customers identified contributing ₹{int(high_risk['revenue_at_risk'].sum())} revenue at risk."
    )