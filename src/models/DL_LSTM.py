from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, psutil, tensorflow as tf
from datetime import datetime

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

# Scale and split data (without scaling)
def prepare_lstm_data(df, target_col, seq_length, test_ratio, date_col=None):
    df = df.dropna()

    # Extract and store date index if specified
    if date_col and date_col in df.columns:
        date_index = df[date_col].iloc[-(len(df) - seq_length):].reset_index(drop=True)
        df = df.drop(columns=[date_col])  # Drop datetime from feature set
    else:
        date_index = df.index[-(len(df) - seq_length):]

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    feature_cols = [col for col in df_numeric.columns if col != target_col]

    # Apply MinMax scaling
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns, index=df_numeric.index)

    # Create sequences
    X, y = create_sequences(df_scaled, target_col, feature_cols, seq_length)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1), scaler, date_index[-len(y_test):]

# Plot predictions vs actuals
def plot_predictions(dates, actual, predicted, title="Model Predictions vs Actual"):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, predicted, label='Predicted', linestyle='--')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# Evaluate model performance
def evaluate_predictions(y_true, y_pred, n_features=1):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    p = n_features
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    return rmse, mse, mae, mape, r2, adj_r2

# Function to log system usage
def log_system_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    cpu = process.cpu_percent(interval=1)
    print(f"[{tag}] Memory Usage: {mem:.2f} MB | CPU Usage: {cpu:.2f}%")
    return mem, cpu

# Main function to run the LSTM model
def run_lstm_on_dataset(data, target, date_col, seq_length, test_ratio, epochs, batch_size):
    results = {}

    # Prepare your dataset
    X_train, X_test, y_train, y_test, scaler, date_index = prepare_lstm_data(data, target, seq_length, test_ratio, date_col)

    # Build model
    n_features = X_train.shape[2]
    model = build_lstm_model(seq_length, n_features=n_features)
    
    # Log performance before training
    start_time = time.time()
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

    log_system_usage("After Training")
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")
    
    # Predict
    y_pred = model.predict(X_test)

    # Determine the correct number of columns from the scaler
    n_total_features = scaler.n_features_in_

    # Pad predictions with zeros for inverse transform
    y_pred_full = np.zeros((len(y_pred), n_total_features))
    y_pred_full[:, -1] = y_pred.flatten()

    y_test_full = np.zeros((len(y_test), n_total_features))
    y_test_full[:, -1] = y_test.flatten()

    # Inverse transform and extract only the target column
    y_pred_inv = scaler.inverse_transform(y_pred_full)[:, -1]
    y_test_inv = scaler.inverse_transform(y_test_full)[:, -1]

    # Plot results
    plot_predictions(date_index, y_test_inv, y_pred_inv)

    # Print metrics
    rmse, mse, mae, mape, r2, adj_r2 = evaluate_predictions(y_test_inv, y_pred_inv, n_features=X_test.shape[2])
    print(f"RMSE: {rmse:.3f}, MSE: {mse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.3f}, R²: {r2:.3f}, Adj_R²: {adj_r2:.3f}")
 
    results['Dataset'] = data.name   
    results['RMSE'] = rmse
    results['MSE'] = mse
    results['MAE'] = mae
    results['MAPE'] = mape        
    results['R²'] = r2
    results['Adj_R²'] = adj_r2    
    
    return results

