import numpy as np
import json
import os

def load_rules(): # Load business governance from centralized Rule Book
    """
    Reads the 'rules.json' file to retrieve enterprise-defined retention thresholds.
    Ensures that logic remains configurable without code changes.
    """
    rules_path = os.path.join(os.path.dirname(__file__), "../data/rules.json")
    with open(rules_path, 'r') as f:
        return json.load(f)

def assign_strategy(df): # Apply rule-based logic to determine retention actions
    """
    Assigns a specific retention strategy to each customer based on their 
    risk segment and Customer Lifetime Value (CLV).
    Prioritizes high-touch interventions (RM Call) for high-value customers.
    """
    rules = load_rules()
    strat_rules = rules['retention_strategies']
    
    def strategy(row):
        # 1. High Risk Segment: Urgent intervention required
        if row['risk_segment'] == "High Risk":
            if row['CLV'] >= strat_rules['relationship_manager_call']['min_clv']:
                return "Relationship Manager Call" # White-glove service for top-tier clients
            elif row['CLV'] >= strat_rules['fee_waiver']['min_clv']:
                return "Fee Waiver" # Financial incentive for mid-tier clients
            else:
                return "Cashback" # Automated incentive for lower-tier clients
        
        # 2. Medium Risk Segment: Proactive engagement to prevent escalation
        elif row['risk_segment'] == "Medium Risk":
            if row['CLV'] >= strat_rules['relationship_manager_call']['min_clv']:
                return "Relationship Manager Call"
            else:
                return "Standard Engagement" # Educational or marketing outreach
        
        return "No Action" # Low-risk customers follow standard lifecycle

    # Apply the strategy logic row-wise
    df['retention_strategy'] = df.apply(strategy, axis=1)
    return df


def simulate_campaign(df): # Estimate financial impact and ROI of retention efforts
    """
    Calculates the cost of incentives, simulates customer response rates, 
    and determines the expected Return on Investment (ROI).
    """
    # Unit costs associated with each retention action
    cost_map = {
        "Fee Waiver": 1000,
        "Cashback": 500,
        "Relationship Manager Call": 200,
        "Standard Engagement": 50,
        "No Action": 0
    }

    df['incentive_cost'] = df['retention_strategy'].map(cost_map)

    # Simulation Logic: High-risk customers have a 70% chance of accepting an offer
    # whereas lower-risk customers respond at a 30% baseline.
    df['response'] = np.where(
        df['churn_probability'] > 0.7,
        np.random.binomial(1, 0.7, len(df)),
        np.random.binomial(1, 0.3, len(df))
    )

    # ROI Calculation: Potential Revenue Saved (Revenue at Risk * Response) - Campaign Cost
    df['roi'] = df['revenue_at_risk'] * df['response'] - df['incentive_cost']

    return df