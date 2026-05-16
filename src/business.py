def calculate_clv(df): # Calculate Customer Lifetime Value (CLV)
    """
    Computes a simplified Customer Lifetime Value (CLV).
    Assumption: CLV is approximately 20% of the total transaction amount.
    """
    # Simple CLV approximation for banking portfolio
    df['CLV'] = df['Total_Trans_Amt'] * 0.2
    return df


def revenue_at_risk(df): # Identify potential financial loss from high-risk customers
    """
    Calculates the 'Revenue at Risk' metric.
    Only applies to customers with a churn probability > 70% (High Risk).
    Formula: CLV * Churn Probability
    """
    def calculate_risk(row):
        # Business Rule: Only quantify risk for the high-probability segment
        if row["churn_probability"] > 0.7:
            return row["CLV"] * row["churn_probability"]
        else:
            return 0

    # Apply the risk calculation row-wise to the dataframe
    df['revenue_at_risk'] = df.apply(calculate_risk, axis=1)

    return df