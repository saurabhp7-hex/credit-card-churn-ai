import pandas as pd
from feature_engineering import engineer_features

def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # Create target
    df['target'] = df['Attrition_Flag'].apply(
        lambda x: 1 if x == 'Attrited Customer' else 0
    )

    # Drop Naive Bayes columns if they exist
    nb_cols = [c for c in df.columns if 'Naive_Bayes' in c]
    df = df.drop(nb_cols, axis=1)

    # Drop unnecessary columns
    df = df.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1)

    # Engineer behavioral features
    df = engineer_features(df)

    return df


def encode_features(df):
    df = pd.get_dummies(df, drop_first=True)
    return df