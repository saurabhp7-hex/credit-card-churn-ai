import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # Create target
    df['target'] = df['Attrition_Flag'].apply(
        lambda x: 1 if x == 'Attrited Customer' else 0
    )

    # Drop unnecessary columns
    df = df.drop(['CLIENTNUM', 'Attrition_Flag'], axis=1)

    return df


def encode_features(df):
    df = pd.get_dummies(df, drop_first=True)
    return df