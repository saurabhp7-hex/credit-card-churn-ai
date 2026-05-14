import numpy as np

import json
import os

def load_rules():
    rules_path = os.path.join(os.path.dirname(__file__), "../data/rules.json")
    with open(rules_path, 'r') as f:
        return json.load(f)

def assign_strategy(df):
    rules = load_rules()
    strat_rules = rules['retention_strategies']
    
    def strategy(row):
        # 1. High Value + High Risk -> Fee Waiver or RM Call
        if row['risk_segment'] == "High Risk":
            if row['CLV'] >= strat_rules['relationship_manager_call']['min_clv']:
                return "Relationship Manager Call"
            elif row['CLV'] >= strat_rules['fee_waiver']['min_clv']:
                return "Fee Waiver"
            else:
                return "Cashback"
        
        # 2. Medium Risk + High Value -> Proactive Outreach
        elif row['risk_segment'] == "Medium Risk":
            if row['CLV'] >= strat_rules['relationship_manager_call']['min_clv']:
                return "Relationship Manager Call"
            else:
                return "Standard Engagement"
        
        return "No Action"

    df['retention_strategy'] = df.apply(strategy, axis=1)
    return df


def simulate_campaign(df):
    # Assign cost
    cost_map = {
        "Fee Waiver": 1000,
        "Cashback": 500,
        "Relationship Manager Call": 200,
        "Standard Engagement": 50,
        "No Action": 0
    }

    df['incentive_cost'] = df['retention_strategy'].map(cost_map)

    # Simulate response
    df['response'] = np.where(
        df['churn_probability'] > 0.7,
        np.random.binomial(1, 0.7, len(df)),
        np.random.binomial(1, 0.3, len(df))
    )

    # Calculate ROI
    df['roi'] = df['revenue_at_risk'] * df['response'] - df['incentive_cost']

    return df