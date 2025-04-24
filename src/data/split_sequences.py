import numpy as np

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def prepare_train_val_test(combined_data, seq_length):
    combined_data.sort_values(by='Date', inplace=True)
    close_scaled_col = [col for col in combined_data.columns if col.lower().endswith('close_scaled')][0]
    close_prices = combined_data[close_scaled_col].values
    X, y = create_sequences(close_prices, seq_length)

    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size].reshape((-1, seq_length, 1))
    X_val = X[train_size:train_size+val_size].reshape((-1, seq_length, 1))
    X_test = X[train_size+val_size:].reshape((-1, seq_length, 1))
    y_train = y[:train_size]
    y_val = y[train_size:train_size+val_size]
    y_test = y[train_size+val_size:]

    return X_train, X_val, X_test, y_train, y_val, y_test
