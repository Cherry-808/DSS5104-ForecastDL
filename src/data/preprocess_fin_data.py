import pandas as pd
import numpy as np
import pandas_ta as pta
from ta.trend import ADXIndicator

def enrich_target_stock(df, ticker):
    # --- Time Features ---
    # df[f'{ticker}_hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
    df[f'{ticker}_day'] = df.index.day
    df[f'{ticker}_month'] = df.index.month
    df[f'{ticker}_weekday'] = df.index.weekday
    # df[f'{ticker}_is_weekend'] = df[f'{ticker}_weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # Cyclical encoding
    df['day_sin'] = np.sin(2 * np.pi * df.index.day / 31)
    df['day_cos'] = np.cos(2 * np.pi * df.index.day / 31)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df.index.weekday / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df.index.weekday / 7)

    # --- Lagged Prices ---
    for lag in [1, 2, 3, 4, 5, 10, 20, 30, 60, 90, 120]:
        df[f'Close_{ticker}_Lag_{lag}'] = df['Close'].shift(lag)

    # --- Daily Return ---
    df[f'{ticker}_Daily_Return'] = df['Close'].pct_change()

    # --- Rolling (Total) Returns ---
    for window in [5, 10, 20, 60, 90, 120, 252]:
        df[f'{ticker}_Total_Return_{window}d'] = df['Close'].pct_change(periods=window)

    # --- RSI ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df[f'{ticker}_RSI'] = 100 - (100 / (1 + rs))

    # --- EMAs ---
    df[f'{ticker}_EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df[f'{ticker}_EMA_40'] = df['Close'].ewm(span=40, adjust=False).mean()

    # --- MACD ---
    df[f'{ticker}_MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df[f'{ticker}_MACD_Signal'] = df[f'{ticker}_MACD'].ewm(span=9, adjust=False).mean()

    # --- SMAs ---
    df[f'{ticker}_SMA_50'] = df['Close'].rolling(window=50).mean()
    df[f'{ticker}_SMA_100'] = df['Close'].rolling(window=100).mean()
    df[f'{ticker}_SMA_150'] = df['Close'].rolling(window=150).mean() # Corrected window size
    df[f'{ticker}_SMA_200'] = df['Close'].rolling(window=200).mean()

    # --- Bollinger Bands (20-period SMA, 2 std) ---
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df[f'{ticker}_BB_Mid'] = sma_20
    df[f'{ticker}_BB_High'] = sma_20 + (2 * std_20)
    df[f'{ticker}_BB_Low'] = sma_20 - (2 * std_20)

    # --- Ichimoku Cloud (Manual Calculation) ---
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    df[f'{ticker}_Ichimoku_Conversion_Line'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    df[f'{ticker}_Ichimoku_Base_Line'] = (high_26 + low_26) / 2

    df[f'{ticker}_Ichimoku_Lagging_Span'] = df['Close'].shift(-26)

    span_a = ((df[f'{ticker}_Ichimoku_Conversion_Line'] + df[f'{ticker}_Ichimoku_Base_Line']) / 2).shift(26)
    df[f'{ticker}_Ichimoku_Leading_Span_A'] = span_a

    high_52 = df['High'].rolling(window=52).max()
    low_52 = df['Low'].rolling(window=52).min()
    span_b = ((high_52 + low_52) / 2).shift(26)
    df[f'{ticker}_Ichimoku_Leading_Span_B'] = span_b

    # --- Parabolic SAR (using pandas_ta) ---
    if (
        {'High', 'Low', 'Close'}.issubset(df.columns) and
        not df[['High', 'Low', 'Close']].isnull().all(axis=1).any() and
        len(df) >= 2 # Ensure at least 2 rows for iloc in pandas_ta
    ):
        psar_df = pta.psar(high=df['High'], low=df['Low'], close=df['Close'], step=0.02, max_step=0.2)
        if isinstance(psar_df, pd.DataFrame) and not psar_df.empty:
            df[f'{ticker}_Parabolic_SAR'] = psar_df.iloc[:, 0]

    # --- ADX (Average Directional Index) ---
    if {'High', 'Low', 'Close'}.issubset(df.columns):
        adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df[f'{ticker}_ADX'] = adx_indicator.adx()
        df[f'{ticker}_DI_Pos'] = adx_indicator.adx_pos()
        df[f'{ticker}_DI_Neg'] = adx_indicator.adx_neg()

    # Open Gaps
    df[f'{ticker}_Gap_Up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
    df[f'{ticker}_Gap_Down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
    df[f'{ticker}_Gap_Size'] = df['Open'] - df['Close'].shift(1)

    # Open Returns
    df[f'{ticker}_Open_Return'] = df['Close'] / df['Open'] - 1

    # Open-based Moving Averages
    if 'Open' in df.columns:
        df[f'{ticker}_Open_SMA_10'] = df['Open'].rolling(window=10).mean()
        df[f'{ticker}_Open_EMA_10'] = df['Open'].ewm(span=10, adjust=False).mean()

        # Open-based VWAP approximation (if Volume is available)
        if 'Volume' in df.columns:
            df[f'{ticker}_VWAP_Open'] = (df['Open'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    return df

def flatten_cols(df):
    df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns]
    return df