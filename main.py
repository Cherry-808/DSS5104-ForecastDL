
import os
import time
import psutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

def run_lstm_on_dataset(df, dataset_name, sequence_length=10, epochs=5, batch_size=32):
    results = {}
    start_time = time.time()

    log_system_usage(f"{dataset_name} - Before Training")

    data = df.values
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    X, y = np.array(X), np.array(y)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    log_system_usage(f"{dataset_name} - After Training")
    duration = time.time() - start_time

    gpus = tf.config.list_physical_devices('GPU')
    gpu_info = gpus[0].name if gpus else 'CPU'

    results['Dataset'] = dataset_name
    results['Rows'] = df.shape[0]
    results['Columns'] = df.shape[1]
    results['RMSE'] = rmse
    results['MAE'] = mae
    results['Training Time (s)'] = duration
    results['Device'] = gpu_info

    return results

def log_system_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    cpu = process.cpu_percent(interval=1)
    print(f"[{tag}] Memory Usage: {mem:.2f} MB | CPU Usage: {cpu:.2f}%")
    return mem, cpu

if __name__ == "__main__":
    simulated_datasets = {
        "Climate (Raw)": pd.DataFrame(np.random.rand(200, 3), columns=['temp', 'humidity', 'wind']),
        "Climate (Lagged)": pd.DataFrame(np.random.rand(200, 4), columns=['temp', 'humidity', 'wind', 'temp_lag']),
        "Energy (Raw)": pd.DataFrame(np.random.rand(300, 2), columns=['demand', 'price']),
        "Energy (Lagged)": pd.DataFrame(np.random.rand(300, 3), columns=['demand', 'price', 'demand_lag']),
        "Finance (Raw)": pd.DataFrame(np.random.rand(250, 2), columns=['close', 'volume']),
        "Finance (Lagged)": pd.DataFrame(np.random.rand(250, 3), columns=['close', 'volume', 'close_lag']),
        "Retail (Raw)": pd.DataFrame(np.random.rand(180, 2), columns=['sales', 'footfall']),
        "Retail (Lagged)": pd.DataFrame(np.random.rand(180, 3), columns=['sales', 'footfall', 'sales_lag']),
        "Transport": pd.DataFrame(np.random.rand(220, 2), columns=['volume', 'month_num'])
    }

    results = []
    for name, df in simulated_datasets.items():
        if df.shape[0] < 50:
            continue

        print(f"\n🔧 Running model for dataset: {name}")
        target = df.columns[0]
        start_time = time.time()
        mem_before, cpu_before = log_system_usage(f"{name} - Start")

        X_train, X_test, y_train, y_test, scaler, date_index = prepare_lstm_data(df, target, 30, 0.3)
        n_features = X_train.shape[2]
        model = build_lstm_model(30, n_features)
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        y_pred = model.predict(X_test)
        n_total_features = scaler.n_features_in_

        y_pred_full = np.zeros((len(y_pred), n_total_features))
        y_pred_full[:, -1] = y_pred.flatten()
        y_test_full = np.zeros((len(y_test), n_total_features))
        y_test_full[:, -1] = y_test.flatten()

        y_pred_inv = scaler.inverse_transform(y_pred_full)[:, -1]
        y_test_inv = scaler.inverse_transform(y_test_full)[:, -1]

        rmse, mae, r2 = evaluate_predictions(y_test_inv, y_pred_inv)
        duration = time.time() - start_time
        mem_after, cpu_after = log_system_usage(f"{name} - End")

        results.append({
            "Dataset": name,
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2,
            "Training Time (s)": duration,
            "Device": "GPU" if tf.config.list_physical_devices('GPU') else "CPU",
            "Memory Before (MB)": mem_before,
            "Memory After (MB)": mem_after,
            "CPU Before (%)": cpu_before,
            "CPU After (%)": cpu_after
        })

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("outputs/results/output_LSTM/lstm_evaluation_log.csv", index=False)
    print(metrics_df)
