import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_raw(raw_dir):
    """
    Load the raw M5 CSV files from raw_dir.
    Returns: calendar, sell_prices, sales_train_validation DataFrames
    """
    calendar = pd.read_csv(os.path.join(raw_dir, 'calendar.csv'))
    prices = pd.read_csv(os.path.join(raw_dir, 'sell_prices.csv'))
    sales = pd.read_csv(os.path.join(raw_dir, 'sales_train_validation.csv'))
    return calendar, prices, sales


def process_sales(sales_df):
    """
    Extracts the time series values from the sales DataFrame.
    Drops identifier columns and returns a numpy array of shape (n_series, n_days).
    """
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    ts = sales_df.drop(columns=id_cols)
    return ts.values


def create_dataset(ts_values, history, horizon):
    """
    Constructs sliding windows over the time series.
    - history: number of past days to use as input
    - horizon: number of future days to predict

    Returns:
      X: array of shape (num_samples, history)
      y: array of shape (num_samples, horizon)
    """
    X, y = [], []
    n_series, n_days = ts_values.shape
    # last starting point is at n_days - horizon
    max_start = n_days - horizon
    for i in tqdm(range(n_series), desc='Series'):
        series = ts_values[i]
        for end in range(history, max_start + 1):
            start = end - history
            X.append(series[start:end])
            y.append(series[end:end + horizon])
    return np.array(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description='Preprocess M5 data for forecasting tasks')
    parser.add_argument('--raw_dir', type=str, default='data/dataset_retail/m5/raw/M5-data',
                        help='Path to raw M5 CSV files')
    parser.add_argument('--processed_dir', type=str, default='data/dataset_retail/m5/processed',
                        help='Directory to save processed numpy arrays')
    parser.add_argument('--history', type=int, default=365,
                        help='Number of past days to use as input')
    parser.add_argument('--horizon', type=int, default=28,
                        help='Number of future days to predict')
    args = parser.parse_args()

    # Load raw data
    calendar, prices, sales = load_raw(args.raw_dir)

    # Process only the sales time series
    ts_values = process_sales(sales)

    # Save the raw time-series matrix (n_series, n_days)
    os.makedirs(args.processed_dir, exist_ok=True)
    np.save(os.path.join(args.processed_dir, 'ts_values.npy'), ts_values)
    print(f"Saved raw time series to {args.processed_dir}/ts_values.npy")


if __name__ == '__main__':
    main()
