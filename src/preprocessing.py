import pandas as pd # Data manipulation library
from feature_engineering import engineer_features # Custom feature engineering module

def load_data(path): # Function to load raw data from a given file path
    """
    Loads raw CSV data from the specified path.
    """
    df = pd.read_csv(path) # Read CSV file into a pandas DataFrame
    return df


def clean_data(df): # Function to handle data cleaning and initial labeling
    """
    Cleans the raw dataframe:
    - Creates a binary 'target' column from 'Attrition_Flag'.
    - Removes unnecessary columns (Naive Bayes indicators and IDs).
    - Triggers the feature engineering pipeline.
    """
    # Create target: 1 for churners, 0 for existing customers
    df['target'] = df['Attrition_Flag'].apply(
        lambda x: 1 if x == 'Attrited Customer' else 0
    )

    # Drop Naive Bayes columns to prevent data leakage and redundancy
    nb_cols = [c for c in df.columns if 'Naive_Bayes' in c]
    df = df.drop(nb_cols, axis=1)

    # Drop non-predictive columns: ID and the original target string
    df = df.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1)

    # Trigger behavioral feature synthesis
    df = engineer_features(df)

    return df


def encode_features(df): # Function to transform categorical features into numerical format
    """
    Performs One-Hot Encoding on all categorical columns.
    Uses drop_first=True to avoid the dummy variable trap.
    """
    df = pd.get_dummies(df, drop_first=True) # Convert categorical to binary dummy variables
    return df