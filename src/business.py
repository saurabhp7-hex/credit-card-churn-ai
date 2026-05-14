def calculate_clv(df):
    # Simple CLV approximation
    df['CLV'] = df['Total_Trans_Amt'] * 0.2
    return df


def revenue_at_risk(df):

    def calculate_risk(row):
        if row["churn_probability"] > 0.7:
            return row["CLV"] * row["churn_probability"]
        else:
            return 0

    df['revenue_at_risk'] = df.apply(calculate_risk, axis=1)

    return df