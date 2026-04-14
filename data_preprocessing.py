import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()
    df.fillna(method='ffill', inplace=True)
    return df
