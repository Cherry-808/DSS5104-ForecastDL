from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, psutil, tensorflow as tf
from datetime import datetime
import gc
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Build and compile the LSTM model
def build_lstm_model(seq_length, n_features):
    model = Sequential([
        Input(shape=(seq_length, n_features)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Create sequences from multivariate dataframe
def create_sequences(data, target_col, feature_cols, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq_x = data.iloc[i:i+seq_length][feature_cols].values
        seq_y = data.iloc[i+seq_length][target_col]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Scale and split data (scaling before log if needed)
def prepare_lstm_data(df, target_col, seq_length, test_ratio, date_col=None, log_transform=False):
    df = df.dropna()

    # Extract and store date index if specified
    if date_col and date_col in df.columns:
        date_index = df[date_col].iloc[-(len(df) - seq_length):].reset_index(drop=True)
        df = df.drop(columns=[date_col])  # Drop datetime from feature set
    else:
        date_index = df.index[-(len(df) - seq_length):]

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    feature_cols = [col for col in df_numeric.columns if col != target_col] # multivariate
    # feature_cols = [target_col]  # univariate - past values of itself only

    # Apply MinMax scaling
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns, index=df_numeric.index)
    
    gc.collect()       

    # Create sequences
    X, y = create_sequences(df_scaled, target_col, feature_cols, seq_length)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1), scaler, date_index[-len(y_test):]

# Plot predictions vs actuals
def plot_predictions(dates, actual, predicted, title, target, date_col):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, predicted, label='Predicted', linestyle='--')
    plt.title(title)
    plt.xlabel(date_col)
    plt.ylabel(target)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.close()

# Evaluate model performance
def evaluate_predictions(y_true, y_pred, n_features=1):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = n_features
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return rmse, mse, mae, mape, r2, adj_r2

# Function to log system usage
def log_system_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2) # in MB
    peak_memory_mb = process.memory_info().peak_wset / (1024 ** 2)  # in MB
    cpu = process.cpu_percent(interval=1)
    print(f"[{tag}] Memory Usage: {mem:.2f} MB | Peak Memory Usage: {peak_memory_mb:.2f} MB | CPU Usage: {cpu:.2f}%")
    return mem, peak_memory_mb, cpu

# Main function to run the LSTM model
def run_lstm_on_dataset(data, target, date_col, seq_length, test_ratio, epochs, batch_size, apply_log_transform=False):
    results = {}

    # Prepare your dataset
    X_train, X_test, y_train, y_test, scaler, dates = prepare_lstm_data(data, target, seq_length, test_ratio, date_col, log_transform=False) # Log handled after inverse

    # Build model
    n_features = X_train.shape[2]
    model = build_lstm_model(seq_length, n_features=n_features)

    # Log performance before training
    start_time = time.time()
    mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    log_system_usage("Before Training")

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    # Train model
    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_test, y_test),
              callbacks=[early_stop],
              verbose=1)

    # # Convert training and validation sets to tf.data.Dataset
    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
    #     .batch(batch_size) \
    #     .prefetch(tf.data.AUTOTUNE)

    # val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
    #     .batch(batch_size) \
    #     .prefetch(tf.data.AUTOTUNE)

    # # Train using Dataset pipeline
    # model.fit(train_dataset,
    #         validation_data=val_dataset,
    #         epochs=epochs,
    #         callbacks=[early_stop],
    #         verbose=1)

    gc.collect()

    log_system_usage("After Training")
    total_time = time.time() - start_time
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    mem_used = mem_after - mem_before

    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Memory Used (MB): {mem_used:.2f}")

    # Predict
    y_pred_scaled = model.predict(X_test)

    # Determine the correct number of columns from the scaler
    n_total_features = scaler.n_features_in_

    # Pad predictions with zeros for inverse transform
    y_pred_full = np.zeros((len(y_pred_scaled), n_total_features))
    y_pred_full[:, -1] = y_pred_scaled.flatten()

    y_test_full = np.zeros((len(y_test), n_total_features))
    y_test_full[:, -1] = y_test.flatten()

    # Inverse transform to the original scale
    y_pred_inv_scaled = scaler.inverse_transform(y_pred_full)[:, -1]
    y_test_inv_scaled = scaler.inverse_transform(y_test_full)[:, -1]

    # Apply exponential to reverse log if original data was logged
    y_pred_final = y_pred_inv_scaled
    y_test_final = y_test_inv_scaled

    # Plot results
    plot_predictions(dates, y_test_final, y_pred_final, "Model Predictions vs Actual", target, date_col)

    # Evaluate metrics
    rmse, mse, mae, mape, r2, adj_r2 = evaluate_predictions(
        y_test_final,
        y_pred_final,
        n_features=X_test.shape[2]
    )

    results.update({
        'Dataset': data.name if hasattr(data, 'name') else 'Unnamed',
        'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'MAPE': mape,
        'R²': r2, 'Adj_R²': adj_r2,
        'TrainingTime': total_time,
        'Device': tf.config.list_physical_devices('GPU')[0].name if tf.config.list_physical_devices('GPU') else 'CPU',
        'FinalMemoryMB': psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2),
        'PeakMemoryMB': psutil.Process(os.getpid()).memory_info().peak_wset / (1024 ** 2),
        'CPUUsage': psutil.cpu_percent(interval=1),
        'TrainingLoss': model.history.history['loss'][-1] if hasattr(model, 'history') else None,
        'Model': 'LSTM'
    })

    # After model training
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    peak_memory_mb = psutil.Process(os.getpid()).memory_info().peak_wset / (1024 ** 2)  # in MB
    device = tf.config.list_physical_devices('GPU')[0].name if tf.config.list_physical_devices('GPU') else 'CPU'

    print("\nEvaluation Summary")
    print("-" * 75)
    print(f"{'Metric':<15}{'Value':>15}")
    print("-" * 75)
    print(f"{'RMSE':<15}{rmse:>15.3f}")
    print(f"{'MSE':<15}{mse:>15.3f}")
    print(f"{'MAE':<15}{mae:>15.3f}")
    print(f"{'MAPE (%)':<15}{mape:>15.3f}")
    print(f"{'R²':<15}{r2:>15.3f}")
    print(f"{'Adj R²':<15}{adj_r2:>15.3f}")
    print(f"{'Training Time (s)':<15}{total_time:>15.2f}")
    print(f"{'Memory Usage (MB)':<15}{psutil.Process(os.getpid()).memory_info().rss / (1024**2):>15.2f}")
    print(f"{'Peak Memory (MB)':<15}{psutil.Process(os.getpid()).memory_info().peak_wset / (1024 ** 2) :>15.2f}")
    print(f"{'CPU Usage (%)':<15}{psutil.cpu_percent(interval=1):>15.2f}")
    print(f"{'Device Used':<15}{device:>15}")
    print("-" * 75)

    # --- Plot and Save Prediction Plot ---
    plot_dir = os.path.join("outputs/results/output_LSTM", data.name)
    os.makedirs(plot_dir, exist_ok=True)

    plot_path = os.path.join(plot_dir, f"{data.name}_prediction_plot.png")
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_test_final, label='Actual')
    plt.plot(dates, y_pred_final, label='Predicted', linestyle='--')
    plt.title(f"{data.name} LSTM Forecast")
    plt.xlabel("Date")
    plt.ylabel("Target")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    # plt.show()
    plt.close()

    # --- Save Evaluation Metrics Log ---
    log_path = os.path.join(plot_dir, f"{data.name}_metrics_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(log_path, "w") as f:
        f.write(f"Dataset: {data.name}\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Target Column: {target}\n")
        f.write(f"Model: LSTM\n")
        f.write(f"Sequence Length: {seq_length}\n")
        f.write(f"Test Ratio: {test_ratio}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Final Memory Usage: {mem:.2f} MB\n")
        f.write(f"Peak Memory Usage: {peak_memory_mb:.2f} MB\n")
        f.write(f"CPU Usage: {psutil.cpu_percent(interval=1):.2f}%\n")
        f.write(f"Training Time: {total_time:.2f} seconds\n\n")
        f.write(f"Training Loss: {model.history.history['loss'][-1] if hasattr(model, 'history') else None}\n")

        f.write("Evaluation Metrics:\n")
        f.write(f" - RMSE: {rmse:.3f}\n")
        f.write(f" - MSE: {mse:.3f}\n")
        f.write(f" - MAE: {mae:.3f}\n")
        f.write(f" - MAPE: {mape:.3f}\n")
        f.write(f" - R²: {r2:.3f}\n")
        f.write(f" - Adjusted R²: {adj_r2:.3f}\n")

    # --- Save model weights ---
    model_path = os.path.join(plot_dir, f"{data.name}_model.h5")
    model.save(model_path)

    # --- Plot training loss ---
    history_path = os.path.join(plot_dir, f"{data.name}_loss_plot.png")
    plt.figure(figsize=(8, 4))
    plt.plot(model.history.history['loss'], label='Training Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.title(f"{data.name} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(history_path)
    plt.show()
    plt.close()

    # --- CSV Log (append to master metrics CSV) ---
    csv_metrics_path = os.path.join("outputs/results/output_LSTM", "all_model_metrics.csv")
    metrics_entry = pd.DataFrame([{
        "Dataset": data.name,
        "Target": target,
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "MAPE": mape,
        "R²": r2,
        "Adj_R²": adj_r2,
        "TrainingTime_s": total_time,
        "BatchSize": batch_size,
        "Epochs": epochs,
        "SequenceLength": seq_length,
        "TestRatio": test_ratio,
        "Device": device,
        "FinalMemoryMB": mem,
        "PeakMemoryMB": peak_memory_mb,
        "CPUUsage": psutil.cpu_percent(interval=1),
        "Model": "LSTM",
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "TrainingLoss": model.history.history['loss'][-1] if hasattr(model, 'history') else None,
    }])

    # Append or create
    if os.path.exists(csv_metrics_path):
        metrics_entry.to_csv(csv_metrics_path, mode='a', index=False, header=False)
    else:
        metrics_entry.to_csv(csv_metrics_path, index=False)
        
    gc.collect()
    tf.keras.backend.clear_session()

    return results

