import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Engineer behavioral features to improve churn prediction.
    """
    # 1. Transaction Velocity (Transactions per month on book)
    df['Transaction_Velocity'] = df['Total_Trans_Ct'] / df['Months_on_book']

    # 2. Average Transaction Amount
    df['Avg_Transaction_Amt'] = df['Total_Trans_Amt'] / df['Total_Trans_Ct']

    # 3. Engagement Score (Positive for relationships, negative for inactivity/complaints)
    # Higher score = More engaged
    df['Engagement_Score'] = (
        df['Total_Relationship_Count'] * 2 - 
        df['Months_Inactive_12_mon'] * 3 - 
        df['Contacts_Count_12_mon'] * 2
    )

    # 4. Financial Change Velocity (Interaction between amount and count changes)
    df['Financial_Change_Velocity'] = df['Total_Amt_Chng_Q4_Q1'] * df['Total_Ct_Chng_Q4_Q1']

    # 5. Credit Utilization Ratio (Explicitly calculated)
    # Total_Revolving_Bal / Credit_Limit
    df['Credit_Utilization_Ratio'] = df['Total_Revolving_Bal'] / df['Credit_Limit']

    # 6. Utilization Trend (Hypothetical proxy for change in usage)
    # Since we don't have historical utilization, we use Avg_Utilization_Ratio 
    # and compare it to Transaction Velocity
    df['Utilization_per_Trans'] = df['Avg_Utilization_Ratio'] / (df['Total_Trans_Ct'] + 1)

    return df
