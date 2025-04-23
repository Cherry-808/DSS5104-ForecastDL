from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, time, psutil, tensorflow as tf
from datetime import datetime
import gc
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Function to log system usage
def log_system_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    peak_mem = process.memory_info().peak_wset / (1024 ** 2) if hasattr(process.memory_info(), 'peak_wset') else mem
    
    cpu = process.cpu_percent(interval=1)
    print(f"[{tag}] Memory Usage: {mem:.2f} MB | Peak Memory Usage: {peak_mem:.2f} MB | CPU Usage: {cpu:.2f}%")
    return mem, peak_mem, cpu

def run_arima_on_dataset(data, target, date_col, test_ratio, engine="statsmodels", order=(5, 1, 0)):
    results = {}
    df = data.dropna().copy()

    # Attempt fast datetime parsing, fallback to individual parsing
    try:
        df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d")
    except Exception:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    df = df.sort_values(by=date_col)
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()
    if df.index.inferred_type == "datetime64" and df.index.freq is None:
        df.index = pd.DatetimeIndex(df.index)

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    inferred_freq = pd.infer_freq(df.index)
    if inferred_freq:
        try:
            df.index.freq = inferred_freq
        except ValueError:
            df.index.freq = None

    split_idx = int(len(df) * (1 - test_ratio))
    date_index = df.index[split_idx:].to_series().reset_index(drop=True)

    y = df[target]
    X = df.drop(columns=[target]) if df.shape[1] > 1 else pd.DataFrame(index=df.index)

    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train, X_test = X[:split_idx], X[split_idx:]

    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train = pd.to_numeric(y_train, errors='coerce').ffill()
    y_test = pd.to_numeric(y_test, errors='coerce').ffill()
    date_index = date_index.ffill()

    X_train.index = y_train.index
    X_test.index = pd.RangeIndex(start=0, stop=len(X_test), step=1)

    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
        raise ValueError("Training or testing data is empty after cleaning. Check for NaNs or formatting issues.")

    log_system_usage("Before Model Training")
    start_time = time.time()
    mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    if engine == "statsmodels":
        if X_train.empty:
            model = ARIMA(endog=y_train, order=order).fit()
            y_pred = model.forecast(steps=len(y_test))
        else:
            model = ARIMA(endog=y_train, exog=X_train, order=order).fit()
            y_pred = model.forecast(steps=len(y_test), exog=X_test)
    else:
        if X_train.empty:
            model = auto_arima(y_train, seasonal=False, stepwise=True, suppress_warnings=True)
            y_pred = model.predict(n_periods=len(y_test))
        else:
            model = auto_arima(y_train, exogenous=X_train, seasonal=False, stepwise=True, suppress_warnings=True)
            y_pred = model.predict(n_periods=len(y_test), exogenous=X_test)

    total_time = time.time() - start_time
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    mem_used = mem_after - mem_before
    log_system_usage("After Model Training")
    
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Memory Used (MB): {mem_used:.2f}")    

    # mse = mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(
    np.exp(y_test.values) if data.name == 'Energy' else y_test.values,
    np.exp(y_pred) if data.name == 'Energy' else y_pred
    )
    rmse = np.sqrt(mse)
    # mae = mean_absolute_error(y_test, y_pred)
    mae = mean_absolute_error(
    np.exp(y_test.values) if data.name == 'Energy' else y_test.values,
    np.exp(y_pred) if data.name == 'Energy' else y_pred
    )
    try:
        # mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        if data.name == 'Energy':
            mape = np.mean(np.abs((np.exp(y_test.values) - np.exp(y_pred)) / (np.exp(y_test.values) + 1e-10))) * 100
        else:
            mape = np.mean(np.abs((y_test.values - y_pred) / (y_test.values + 1e-10))) * 100
        
    except Exception:
        mape = np.nan
    # r2 = r2_score(y_test, y_pred)
    r2 = r2_score(
        np.exp(y_test.values) if data.name == 'Energy' else y_test.values,
        np.exp(y_pred) if data.name == 'Energy' else y_pred
    )    
    n = len(y_test)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    device = tf.config.list_physical_devices('GPU')[0].name if tf.config.list_physical_devices('GPU') else 'CPU'
    model_label = "Statsmodels ARIMA" if engine == "statsmodels" else "PMDARIMA"

    print(f"\n[ARIMA - {data.name}] Evaluation Summary ({model_label})")
    print("-" * 75)
    print(f"{'RMSE':<20}: {rmse:.3f}")
    print(f"{'MAE':<20}: {mae:.3f}")
    print(f"{'MAPE (%)':<20}: {mape:.3f}")
    print(f"{'R²':<20}: {r2:.3f}")
    print(f"{'Adjusted R²':<20}: {adj_r2:.3f}")
    print(f"{'Training Time (s)':<20}: {total_time:.2f}")
    print(f"{'Memory Used (MB)':<20}: {mem_used:.2f}")
    print(f"{'Device Used':<20}: {device}")
    print("-" * 75)

    plot_dir = os.path.join("outputs/results/output_ARIMA", data.name)
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    # plt.plot(date_index, y_test.values, label='Actual')
    # plt.plot(date_index, y_pred, label='Predicted', linestyle='--')
    plt.plot(date_index, np.exp(y_test.values) if data.name == 'Energy' else y_test.values, label='Actual')
    plt.plot(date_index, np.exp(y_pred) if data.name == 'Energy' else y_pred, label='Predicted', linestyle='--')    
    plt.title(f"{data.name} {model_label} Forecast")
    plt.xlabel("Date")
    plt.ylabel("Target")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{data.name}_prediction_plot.png"))
    plt.show()

    log_path = os.path.join(plot_dir, f"{data.name}_metrics_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {data.name}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Model: {model_label}\n")
        f.write(f"Target: {target}\n")
        f.write(f"Test Ratio: {test_ratio}\n")
        f.write(f"Training Time: {total_time:.2f} s\n")
        f.write(f"Memory Used: {mem_used:.2f} MB\n")
        f.write(f"Device: {device}\n\n")
        f.write("Evaluation Metrics:\n")
        f.write(f" - RMSE: {rmse:.3f}\n")
        f.write(f" - MSE: {mse:.3f}\n")
        f.write(f" - MAE: {mae:.3f}\n")
        f.write(f" - MAPE: {mape:.3f}\n")
        f.write(f" - R²: {r2:.3f}\n")
        f.write(f" - Adjusted R²: {adj_r2:.3f}\n")

    csv_metrics_path = os.path.join("outputs/results/output_ARIMA", "all_model_metrics.csv")
    metrics_entry = pd.DataFrame([{
        "Dataset": data.name,
        "Model": model_label,
        "Target": target,
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "MAPE": mape,
        "R²": r2,
        "Adj_R²": adj_r2,
        "TrainingTime_s": total_time,
        "TestRatio": test_ratio,
        "Device": device,
        "FinalMemoryMB": mem_after
    }])

    if os.path.exists(csv_metrics_path):
        metrics_entry.to_csv(csv_metrics_path, mode='a', index=False, header=False)
    else:
        metrics_entry.to_csv(csv_metrics_path, index=False)

    results.update(metrics_entry.iloc[0].to_dict())
    return results
