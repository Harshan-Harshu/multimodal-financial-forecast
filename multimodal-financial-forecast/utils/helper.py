import pandas as pd

def rolling_window(df, column, window=5):
    return df[column].rolling(window=window).mean()

def normalize_column(df, column):
    return (df[column] - df[column].mean()) / df[column].std()
