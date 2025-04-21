from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, time, psutil, tensorflow as tf
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

def run_arima_on_dataset(data, target, date_col, test_ratio, engine, order=None):
    results = {}
    df = data.dropna().copy()
    df = df.sort_values(by=date_col)

    # if date_col in df.columns:
    #     # date_index = df[date_col].iloc[-int(len(df) * test_ratio):].reset_index(drop=True)
    #     date_index = data[date_col].iloc[split_idx:].reset_index(drop=True)
    #     df = df.drop(columns=[date_col])
    # else:
    #     date_index = df.index[-int(len(df) * test_ratio):]
    
    split_idx = int(len(df) * (1 - test_ratio))
    date_index = data[date_col].iloc[split_idx:].reset_index(drop=True)
    df = df.drop(columns=[date_col])    

    y = df[target]
    X = df.drop(columns=[target])
    split_idx = int(len(df) * (1 - test_ratio))
    y_train, y_test = y[:split_idx], y[split_idx:]
    X_train, X_test = X[:split_idx], X[split_idx:]

    # Log start
    start_time = time.time()
    mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    if engine == "statsmodels":
        model = ARIMA(endog=y_train, exog=X_train, order=order).fit()
        y_pred = model.forecast(steps=len(y_test), exog=X_test)
    else:  # pmdarima
        model = auto_arima(y_train, exogenous=X_train, seasonal=False, stepwise=True, suppress_warnings=True)
        y_pred = model.predict(n_periods=len(y_test), exogenous=X_test)

    total_time = time.time() - start_time
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    mem_used = mem_after - mem_before

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    device = tf.config.list_physical_devices('GPU')[0].name if tf.config.list_physical_devices('GPU') else 'CPU'

    model_label = "Statsmodels ARIMA" if engine == "statsmodels" else "PMDARIMA"

    # Print summary
    print(f"\n[ARIMA - {data.name}] Evaluation Summary ({model_label})")
    print(f"{'-'*75}")
    print(f"{'RMSE':<20}: {rmse:.3f}")
    print(f"{'MAE':<20}: {mae:.3f}")
    print(f"{'MAPE (%)':<20}: {mape:.3f}")
    print(f"{'R²':<20}: {r2:.3f}")
    print(f"{'Adjusted R²':<20}: {adj_r2:.3f}")
    print(f"{'Training Time (s)':<20}: {total_time:.2f}")
    print(f"{'Memory Used (MB)':<20}: {mem_used:.2f}")
    print(f"{'Device Used':<20}: {device}")
    print(f"{'-'*75}")

    # Save plots
    plot_dir = os.path.join("outputs/results/output_ARIMA", data.name)
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(date_index, y_test.values, label='Actual')
    plt.plot(date_index, y_pred, label='Predicted', linestyle='--')
    plt.title(f"{data.name} {model_label} Forecast")
    plt.xlabel("Date")
    plt.ylabel("Target")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, f"{data.name}_prediction_plot.png"))
    plt.show()

    # Save metrics log
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

    # Append to master CSV
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
