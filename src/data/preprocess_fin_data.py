import pandas as pd

def add_time_features(df, prefix=''):
    df[f'{prefix}hour'] = df.index.hour
    df[f'{prefix}day'] = df.index.day
    df[f'{prefix}month'] = df.index.month
    df[f'{prefix}weekday'] = df.index.weekday
    df[f'{prefix}is_weekend'] = df[f'{prefix}weekday'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def calculate_technical_indicators(df, ticker):
    df[f'{ticker}_RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df[f'{ticker}_EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df[f'{ticker}_EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df[f'{ticker}_MACD'] = df[f'{ticker}_EMA_12'] - df[f'{ticker}_EMA_26']
    return df

def flatten_cols(df):
    df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns]
    return df
