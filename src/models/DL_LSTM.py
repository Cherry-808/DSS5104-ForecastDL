from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
def prepare_lstm_data(df, target_col, seq_length, test_ratio):
    df = df.dropna()

    # Drop datetime columns if any
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
    df = df.drop(columns=datetime_cols)

    feature_cols = [col for col in df.columns if col != target_col]

    X, y = create_sequences(df, target_col, feature_cols, seq_length)
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train.reshape(-1, 1), y_test.reshape(-1, 1), df.index[-len(y_test):]

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
def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2
