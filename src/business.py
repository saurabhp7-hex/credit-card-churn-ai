def calculate_clv(df):
    # Simple CLV approximation
    df['CLV'] = df['Total_Trans_Amt'] * 0.2
    return df


def revenue_at_risk(df):
    df['revenue_at_risk'] = df['CLV'] * df['churn_probability']
    return df