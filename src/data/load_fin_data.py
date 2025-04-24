import yfinance as yf
from fredapi import Fred
import pandas as pd

def load_finance_data(ticker, start_date, end_date, intra_day_price):
    data = yf.download(ticker, start=start_date, end=end_date)
    if intra_day_price:
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    else:
        data = data[['Close', 'Volume']].dropna()

    data[f'{ticker}_Returns'] = data['Close'].pct_change().fillna(0)
    return data

def load_fred_data(series_id, name, api_key, start_date, end_date):
    fred = Fred(api_key)
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    data = pd.DataFrame(data, columns=[name]).dropna()
    data.index = pd.to_datetime(data.index)
    return data.resample('D').ffill()
