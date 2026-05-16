import pandas as pd
import numpy as np

def engineer_features(df): # Create behavioral features from raw banking data
    """
    Synthesizes complex features to capture customer behavior trends, 
    engagement levels, and financial stability for better churn prediction.
    """
    # 1. Transaction Velocity: How frequently the customer uses the card relative to their tenure
    df['Transaction_Velocity'] = df['Total_Trans_Ct'] / df['Months_on_book']

    # 2. Average Transaction Amount: Measure of spending intensity per transaction
    df['Avg_Transaction_Amt'] = df['Total_Trans_Amt'] / df['Total_Trans_Ct']

    # 3. Engagement Score: A weighted composite of bank relationships (+) vs inactivity and support load (-)
    # Higher score indicates a "sticky" or well-integrated customer.
    df['Engagement_Score'] = (
        df['Total_Relationship_Count'] * 2 - 
        df['Months_Inactive_12_mon'] * 3 - 
        df['Contacts_Count_12_mon'] * 2
    )

    # 4. Financial Change Velocity: Captures the momentum of spending changes (amount and count)
    # A decline in both amount and count results in a high velocity of potential churn signal.
    df['Financial_Change_Velocity'] = df['Total_Amt_Chng_Q4_Q1'] * df['Total_Ct_Chng_Q4_Q1']

    # 5. Credit Utilization Ratio: Percentage of the assigned credit limit actually being used
    # High utilization can signal financial stress; extremely low utilization can signal card de-prioritization.
    df['Credit_Utilization_Ratio'] = df['Total_Revolving_Bal'] / df['Credit_Limit']

    # 6. Utilization Efficiency: Proxy for how much credit is 'consumed' per transaction
    # Uses Avg_Utilization_Ratio relative to transaction frequency.
    df['Utilization_per_Trans'] = df['Avg_Utilization_Ratio'] / (df['Total_Trans_Ct'] + 1)

    return df
