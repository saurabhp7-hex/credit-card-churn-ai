import numpy as np

def assign_strategy(df):
    def strategy(row):
        if row['churn_probability'] > 0.7 and row['CLV'] > 5000:
            return "Fee Waiver"
        elif row['churn_probability'] > 0.7:
            return "Cashback"
        else:
            return "No Action"

    df['retention_strategy'] = df.apply(strategy, axis=1)
    return df


def simulate_campaign(df):
    # Assign cost
    cost_map = {
        "Fee Waiver": 1000,
        "Cashback": 500,
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