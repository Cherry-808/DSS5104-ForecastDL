import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from memory_profiler import memory_usage
import time

def build_prophet_model(df, train_ratio=0.7, verbose=False):
    df = df.sort_values('ds').dropna().reset_index(drop=True)
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # record time and memory usage
    def train_model():
        model = Prophet()
        model.fit(train_df)
        return model

    start_time = time.time()
    mem_usage, model = memory_usage(train_model, retval=True, max_usage=True)
    end_time = time.time()

    print(f"time usage {end_time - start_time:.2f}s | memory usage {mem_usage:.2f} MB")

    # Predict
    future = pd.DataFrame({'ds': test_df['ds']})
    forecast = model.predict(future)
    predictions = forecast['yhat'].values.tolist()

    return predictions, train_df, test_df
