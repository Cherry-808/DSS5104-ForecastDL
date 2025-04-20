# data_processing.py
"""
Purpose:
1. Load raw training data (train.csv).
2. Filter data for a specific store and item.
3. Perform feature engineering (time features, lags, rolling windows).
4. Split data into training and validation sets based on a cutoff date.
5. Handle NaNs resulting from feature engineering.
6. Export the processed training and validation dataframes to Parquet files.

How to run:
python data_processing.py --store 1 --item 1 --cutoff 2017-09-30
"""

import pandas as pd
import numpy as np
#import argparse
import os
from pathlib import Path
import logging # 导入 logging 模块
import datetime # 导入 datetime 用于生成时间戳

# --- 定义文件和目录路径 (可以放在脚本顶部) ---
RAW_DATA_DIR = 'data/dataset_retail/raw'  
PROCESSED_DATA_DIR = 'data/dataset_retail/processed'
LOG_DIR = 'logs' # 定义日志文件存放目录

def load_and_filter_data(input_file_path, store_id, item_id):
    """Loads data and filters for a specific store and item."""
    logging.info(f"Loading data from {input_file_path} for Store {store_id}, Item {item_id}...") # 使用 logging
    try:
        df = pd.read_csv(input_file_path, parse_dates=['date'])
    except FileNotFoundError:
        logging.error(f"Error: File not found at {input_file_path}") # 使用 logging
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred during data loading from {input_file_path}") # 记录异常
        return None

    my_series_df = df[(df['store'] == store_id) & (df['item'] == item_id)].copy()

    if my_series_df.empty:
        logging.warning(f"Warning: No data found for Store {store_id}, Item {item_id}.") # 使用 logging
        return None # 或者根据情况决定是否返回空 DataFrame

    my_series_df.sort_values('date', inplace=True)
    my_series_df = my_series_df[['date', 'sales']].reset_index(drop=True)
    logging.info(f"Data loaded and filtered. Shape: {my_series_df.shape}")
    logging.info(f"Date range: {my_series_df['date'].min()} to {my_series_df['date'].max()}")
    return my_series_df

def create_features(df):
    """Creates time series features from a datetime index."""
    logging.info("Creating features...")
    if df is None: # 增加对 None 的检查
        logging.error("Input dataframe is None in create_features. Skipping.")
        return None
    df = df.copy()
    TARGET = 'sales'
    logging.info("  Creating time features...")

    # --- Time features ---
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek # Monday=0, Sunday=6
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    # Add sine/cosine features for cyclicality (optional but good)
    # df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    # df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    # df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    # df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # --- Lag features ---
    logging.info("  Creating lag features...")
    lags = [1, 7, 14, 21, 28, 30, 60, 90, 180, 364, 365]
    for lag in lags:
         df[f'{TARGET}_lag_{lag}'] = df[TARGET].shift(lag)

    logging.info("  Creating rolling window features...")
    windows = [7, 14, 28, 60, 90]
    aggregations = ['mean', 'std', 'median', 'min', 'max']
    for window in windows:
        roll_series = df[TARGET].rolling(window=window, min_periods=max(1, window // 2))
        for agg in aggregations:
            df[f'{TARGET}_roll_{agg}_{window}d'] = getattr(roll_series, agg)()

    logging.info(f"Shape after feature engineering: {df.shape}")
    return df

def split_data(df, cutoff_date):
    """Splits data into training and validation sets."""
    if df is None:
        logging.error("Input dataframe is None in split_data. Skipping.")
        return None, None
    logging.info(f"Splitting data at {cutoff_date}...")
    cutoff_date_ts = pd.to_datetime(cutoff_date)
    train_data = df[df['date'] <= cutoff_date_ts].copy()
    validation_data = df[df['date'] > cutoff_date_ts].copy()
    logging.info(f"Train shape: {train_data.shape}, Validation shape: {validation_data.shape}")
    if train_data.empty or validation_data.empty:
        logging.warning("Warning: Train or Validation set is empty after split. Check cutoff date and data range.")
    return train_data, validation_data

def export_data(train_df, val_df, train_path, val_path):
    """Exports processed dataframes to specified paths."""
    if train_df is None or val_df is None:
        logging.error("Input dataframes are None in export_data. Skipping export.")
        return
    logging.info(f"Exporting data to {train_path} and {val_path}...")
    try:
        train_dir = os.path.dirname(train_path)
        val_dir = os.path.dirname(val_path)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        logging.info("Data exported successfully as Parquet.")
    except ImportError:
        logging.error("Error: 'pyarrow' library not found. Cannot save as Parquet.")
    except Exception as e:
        logging.exception(f"An error occurred during export to {train_path}/{val_path}") # Use exception


def main(store_id, item_id, cutoff_date, raw_dir, processed_dir):
    """Main execution pipeline for data processing."""
    logging.info("--- Starting Data Processing Pipeline ---")
    try: # 包裹主流程以捕捉意外错误
        input_csv_path = os.path.join(raw_dir, 'train.csv')

        # Step 1: Load and Filter
        my_series_df = load_and_filter_data(input_csv_path, store_id, item_id)
        if my_series_df is None:
            logging.error("Exiting pipeline due to data loading/filtering error.")
            return # 确保在出错时退出

        # Step 2: Feature Engineering
        my_series_featured = create_features(my_series_df)
        if my_series_featured is None:
             logging.error("Exiting pipeline due to feature engineering error.")
             return

        # Step 3: Split Data
        train_final, validation_final = split_data(my_series_featured, cutoff_date)
        if train_final is None or validation_final is None or train_final.empty or validation_final.empty:
             logging.error("Exiting pipeline due to empty train/validation set after split.")
             return

        # Step 4: Handle NaNs
        initial_train_rows = train_final.shape[0]
        train_final.dropna(inplace=True)
        rows_dropped = initial_train_rows - train_final.shape[0]
        logging.info(f"Handling NaNs: Dropped {rows_dropped} rows from train_final.")
        if train_final.empty:
            logging.error("Error: Training set is empty after dropping NaNs.")
            return

        # Step 5: Export Data
        train_filename = f'train_featured_s{store_id}_i{item_id}.parquet'
        validation_filename = f'validation_featured_s{store_id}_i{item_id}.parquet'
        train_output_path = os.path.join(processed_dir, train_filename)
        validation_output_path = os.path.join(processed_dir, validation_filename)
        export_data(train_final, validation_final, train_output_path, validation_output_path)

        logging.info("--- Data Processing Pipeline Finished Successfully ---")

    except Exception as e:
        logging.exception("An unexpected error occurred in the main pipeline") # 捕捉主流程中的其他错误


# --- Script Execution Guard ---
if __name__ == "__main__":

    # --- 直接在这里定义参数 ---
    STORE_ID_TO_RUN = 1
    ITEM_ID_TO_RUN = 1
    CUTOFF_DATE_TO_RUN = '2017-09-30'
    # 定义输入输出目录 (相对于脚本运行的位置)
    RAW_DIR = 'data/dataset_retail/raw'
    PROCESSED_DIR = 'data/dataset_retail/processed'
    LOG_DIR = 'logs' # 日志目录
    
    # --- 配置 Logging ---
    # 1. 创建日志目录 (如果不存在)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 2. 创建当前运行的日志文件名 (包含时间戳和参数)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'data_processing_s{STORE_ID_TO_RUN}_i{ITEM_ID_TO_RUN}_{current_time}.log'
    log_filepath = os.path.join(LOG_DIR, log_filename)

    # 3. 配置 basicConfig
    logging.basicConfig(
        level=logging.INFO, # 记录 INFO 及以上级别
        format='%(asctime)s - %(levelname)s - %(message)s', # 日志格式
        filename=log_filepath, # 输出到文件
        filemode='w' # 每次运行覆盖文件
    )

    # 添加一个 Handler 将日志同时输出到控制台 (可选)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) # 控制台也输出 INFO 及以上
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler) # 将 handler 添加到 root logger

    logging.info(f"Starting script execution for Store={STORE_ID_TO_RUN}, Item={ITEM_ID_TO_RUN}")
    logging.info(f"Log file for this run: {log_filepath}")

    # 调用主函数，传入硬编码的参数和目录路径
    main(
        store_id=STORE_ID_TO_RUN,
        item_id=ITEM_ID_TO_RUN,
        cutoff_date=CUTOFF_DATE_TO_RUN,
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR
    )
    
    logging.info("Script execution finished.")