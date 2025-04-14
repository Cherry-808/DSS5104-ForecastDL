import yfinance as yf
from fredapi import Fred
import pandas as pd

def load_finance_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2024-12-31")
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    data[f'{ticker}_Returns'] = data['Close'].pct_change().fillna(0)
    return data

def load_fred_data(series_id, name, api_key):
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id, observation_start='2010-01-01', observation_end='2024-12-31')
    data = pd.DataFrame(data, columns=[name]).dropna()
    data.index = pd.to_datetime(data.index)
    return data.resample('D').ffill()
