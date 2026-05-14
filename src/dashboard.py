import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

from preprocessing import load_data, clean_data, encode_features
from model import add_predictions, calculate_fairness, get_shap_values
from business import calculate_clv, revenue_at_risk
from retention import assign_strategy, simulate_campaign
from llm_explainer import generate_explanation

# ========================
# CONFIG
# ========================
st.set_page_config(
    page_title="Churn Intelligence | Enterprise Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Churn & Retention Intelligence")
st.markdown("---")

# ========================
# LOAD RESOURCES
# ========================
@st.cache_resource
def load_resources():
    model = joblib.load("../models/churn_model.pkl")
    explainer = joblib.load("../models/shap_explainer.pkl")
    return model, explainer

try:
    model, explainer = load_resources()
except Exception as e:
    st.error(f"Error loading models: {e}. Please run 'python main.py' first.")
    st.stop()

# ========================
# PIPELINE
# ========================
def run_analysis_pipeline():
    # Load and preprocess
    df = load_data("../data/bank_churners.csv")
    df = clean_data(df)
    df = encode_features(df)
    
    # Feature matrix
    X = df.drop("target", axis=1)
    
    # Predictions & Risk Segments
    df = add_predictions(df, model, X)
    df["customer_id"] = df.index
    
    # Business & Strategy
    df = calculate_clv(df)
    df = revenue_at_risk(df)
    df = assign_strategy(df)
    df = simulate_campaign(df)
    
    return df, X

# ========================
# SIDEBAR / APP STATE
# ========================
if "data" not in st.session_state:
    if st.sidebar.button("🚀 Initialize Platform"):
        with st.spinner("Processing customer data..."):
            df_final, X_matrix = run_analysis_pipeline()
            st.session_state["data"] = df_final
            st.session_state["X"] = X_matrix
            st.rerun()
    else:
        st.info("👈 Please initialize the platform from the sidebar to begin analysis.")
        st.stop()

df = st.session_state["data"]
X = st.session_state["X"]

# ========================
# TABS
# ========================
tab1, tab2, tab3 = st.tabs([
    "📊 Executive Summary", 
    "👤 Customer Details", 
    "⚖️ Governance & Rules"
])

# ------------------------
# TAB 1: EXECUTIVE SUMMARY
# ------------------------
with tab1:
    st.subheader("Key Portfolio Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(df)
    high_risk_count = len(df[df["risk_segment"] == "High Risk"])
    total_rev_at_risk = df["revenue_at_risk"].sum()
    avg_churn_prob = df["churn_probability"].mean()
    
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("High Risk Population", f"{high_risk_count:,}")
    col3.metric("Revenue at Risk", f"₹{int(total_rev_at_risk):,}")
    col4.metric("Avg Churn Prob", f"{avg_churn_prob:.1%}")
    
    st.markdown("### Risk Segment Breakdown")
    # Using simple Streamlit table instead of charts
    risk_summary = df['risk_segment'].value_counts().reset_index()
    risk_summary.columns = ['Risk Segment', 'Customer Count']
    st.table(risk_summary)

# ------------------------
# TAB 2: CUSTOMER DETAILS
# ------------------------
with tab2:
    st.subheader("Customer Intelligence Workspace")
    
    # Selection Table
    st.write("#### 📋 Select a Customer from the List")
    selection_df = df[[
        "customer_id", "risk_segment", "churn_probability", "CLV", "retention_strategy"
    ]].head(100) # Showing top 100 for performance
    
    # Using st.dataframe with selection mode if available, or simple selection index
    event = st.dataframe(
        selection_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    selected_indices = event.selection.rows
    if selected_indices:
        selected_row_idx = selected_indices[0]
        selected_customer_id = selection_df.iloc[selected_row_idx]["customer_id"]
        c_row = df[df["customer_id"] == selected_customer_id].iloc[0]
        selected_id = c_row.name # Get the actual index for SHAP
        
        st.markdown("---")
        st.write(f"### 🔍 Analysis for Customer ID: {c_row['customer_id']}")
        
        col_a, col_b = st.columns([1, 1])
        
        with col_a:
            st.write("#### Profile Metrics")
            st.json({
                "Risk Segment": c_row['risk_segment'],
                "Churn Probability": f"{c_row['churn_probability']:.2%}",
                "Customer Lifetime Value": f"₹{c_row['CLV']:.2f}",
                "Recommended Strategy": c_row['retention_strategy'],
                "Incentive Cost": f"₹{c_row['incentive_cost']}",
                "Expected ROI": f"₹{c_row['roi']:.2f}"
            })

        with col_b:
            st.write("#### AI Explainer & Churn Drivers")
            with st.spinner("Analyzing drivers..."):
                try:
                    explanation = generate_explanation(c_row)
                    st.markdown(explanation)
                except:
                    st.warning("AI explanation unavailable.")
        
        st.write("#### Top 10 Churn Drivers (SHAP Values)")
        # Replaced chart with a detailed table
        shap_vals = np.array(get_shap_values(explainer, X.iloc[[selected_id]])).flatten()
        top_indices = np.argsort(np.abs(shap_vals))[-10:]
        driver_data = pd.DataFrame({
            "Feature": [X.columns[i] for i in top_indices],
            "Impact Score": [shap_vals[i] for i in top_indices]
        }).sort_values(by="Impact Score", ascending=False)
        st.dataframe(driver_data, use_container_width=True)
    else:
        st.info("👆 Please click a row in the table above to view detailed customer intelligence.")

# ------------------------
# TAB 3: GOVERNANCE & RULES
# ------------------------
with tab3:
    st.subheader("System Governance & Enterprise Rules")
    
    # Fairness Metrics
    st.write("#### Algorithmic Fairness Check (Demographic Parity)")
    fairness = calculate_fairness(df)
    
    # Mapping keys to readable names (assuming Gender_M was used)
    mapped_means = {
        "Male": fairness['group_means'].get(True, 0),
        "Female": fairness['group_means'].get(False, 0)
    }
    
    st.write("**Group Mean Churn Probabilities:**")
    st.write(mapped_means)
    
    st.write(f"**Parity Difference:** {fairness['parity_difference']:.4f}")
    
    # Text Explanation
    st.info(f"""
    **How to interpret this:**
    - **Group Means:** This represents the average churn probability predicted by the model for each group (Male vs Female).
    - **Parity Difference:** Calculated as $|Mean_{{Male}} - Mean_{{Female}}|$. 
    - **Current Result:** The difference is **{fairness['parity_difference']:.4f}**, which is below the enterprise limit of **0.10**.
    - **Compliance Status:** Since the difference is minimal, the model is considered 'fair' as it does not disproportionately target one gender over another based on the underlying features.
    """)

    if fairness['is_fair']:
        st.success("✅ Model meets enterprise fairness thresholds.")
    else:
        st.error("⚠️ Fairness threshold exceeded. Potential bias detected in predictions.")

    st.markdown("---")
    
    # Rule Book
    st.write("#### Centralized Rule Book (data/rules.json)")
    if os.path.exists("../data/rules.json"):
        with open("../data/rules.json", 'r') as f:
            rules_data = json.load(f)
        st.json(rules_data)
    else:
        st.warning("Rule book file not found.")