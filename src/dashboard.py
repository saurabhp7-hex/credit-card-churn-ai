import streamlit as st
import pandas as pd
import joblib

from preprocessing import load_data, clean_data, encode_features
from business import calculate_clv, revenue_at_risk
from retention import assign_strategy, simulate_campaign
from llm_explainer import generate_explanation


# ========================
# CONFIG
# ========================
st.set_page_config(page_title="Churn Intelligence", layout="wide")

st.title("💳 Credit Card Churn & Retention Intelligence Platform")
st.markdown("AI-powered system for identifying at-risk customers and optimizing retention strategies.")


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

    df["churn_probability"] = model.predict_proba(X)[:, 1]
    df["customer_id"] = df.index

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
# MAIN UI (ONLY IF DATA EXISTS)
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
    # CUSTOMER LIST (BALANCED)
    # ========================
    st.subheader("📋 Customer List")

    high_risk = df[df["churn_probability"] > 0.7].head(10)
    low_risk = df[df["churn_probability"] < 0.3].head(10)

    preview_df = pd.concat([high_risk, low_risk])

    preview_df = preview_df[[
        "customer_id",
        "churn_probability",
        "CLV",
        "retention_strategy"
    ]]

    st.dataframe(preview_df, use_container_width=True)

    # ========================
    # CUSTOMER SELECTION
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
    # AI EXPLANATION (ONLY ONE)
    # ========================
    st.subheader("🧠 AI-Based Explanation")

    explanation_container = st.empty()

    with explanation_container.container():
        with st.spinner("Generating explanation..."):
            try:
                explanation = generate_explanation(customer.iloc[0])
            except:
                explanation = "Unable to generate AI explanation. Showing fallback interpretation."

        st.markdown(explanation)

    # ========================
    # KEY BUSINESS INSIGHT
    # ========================
    st.subheader("📌 Key Insight")

    high_risk_full = df[df["churn_probability"] > 0.7]

    st.write(
        f"{len(high_risk_full)} high-risk customers identified contributing "
        f"₹{int(high_risk_full['revenue_at_risk'].sum())} revenue at risk. "
        "Focused retention strategies can significantly reduce potential loss."
    )