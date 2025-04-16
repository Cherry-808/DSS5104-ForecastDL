import pandas as pd

def enrich_target_stock(df, ticker):
    # --- Time Features ---
    df[f'{ticker}_hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
    df[f'{ticker}_day'] = df.index.day
    df[f'{ticker}_month'] = df.index.month
    df[f'{ticker}_weekday'] = df.index.weekday
    df[f'{ticker}_is_weekend'] = df[f'{ticker}_weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # --- RSI ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df[f'{ticker}_RSI'] = 100 - (100 / (1 + rs))

    # --- EMAs ---
    df[f'{ticker}_EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df[f'{ticker}_EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # --- MACD ---
    df[f'{ticker}_MACD'] = df[f'{ticker}_EMA_12'] - df[f'{ticker}_EMA_26']
    df[f'{ticker}_MACD_Signal'] = df[f'{ticker}_MACD'].ewm(span=9, adjust=False).mean()

    # --- SMAs ---
    df[f'{ticker}_SMA_20'] = df['Close'].rolling(window=20).mean()
    df[f'{ticker}_SMA_50'] = df['Close'].rolling(window=50).mean()
    df[f'{ticker}_SMA_200'] = df['Close'].rolling(window=200).mean()

    # --- Bollinger Bands ---
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df[f'{ticker}_Bollinger_Upper'] = sma_20 + (2 * std_20)
    df[f'{ticker}_Bollinger_Lower'] = sma_20 - (2 * std_20)

    # --- Lagged Prices ---
    for lag in [1, 2, 3, 5, 10]:
        df[f'{ticker}_Lag_{lag}'] = df['Close'].shift(lag)

    # --- Daily Return ---
    df[f'{ticker}_Daily_Return'] = df['Close'].pct_change()

    # --- Rolling (Total) Returns ---
    for window in [5, 10, 20]:
        df[f'{ticker}_Total_Return_{window}d'] = df['Close'].pct_change(periods=window)

    return df

def flatten_cols(df):
    df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns]
    return df
