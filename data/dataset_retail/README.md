# Processed Demand Forecasting Data

## Overview

This directory contains processed data derived from the Kaggle "Store Item Demand Forecasting Challenge". The data has been preprocessed and feature-engineered to facilitate model training for predicting item sales demand. Two main versions of the processed data are provided in Parquet format, catering to different modeling approaches.

The processing workflow prioritizes splitting the data into training and validation sets *before* applying time-dependent feature engineering (lags, rolling windows) to prevent data leakage.

## Source Data

The raw data originates from the Kaggle competition:
[https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)

Specifically, the `train.csv` file was used as input.

## Processing Script

The processed files were generated using the `data_processing_modified.py` script. The key steps performed by the script include:

1.  Loading the raw `train.csv` data.
2.  Filtering the data for a specific `store_id` and `item_id`.
3.  Splitting the filtered time series into training and validation sets based on a defined `cutoff_date`.
4.  Performing feature engineering separately on the training and validation sets:
    * Creating basic time features (year, month, day, dayofweek, etc.).
    * Creating lag features based on past sales values.
    * Creating rolling window statistical features based on past sales values.
5.  Combining the processed training and validation data into single files, adding a `split` column ('train'/'validation').
6.  Handling NaN values generated during feature engineering (specifically dropping initial rows from the *training* portion where lags/rolling windows couldn't be computed).
7.  Exporting the final datasets as Parquet files.

## Generated Files

The script generates two Parquet files per store-item combination, following this naming convention:

1.  `data_no_lags_s{store_id}_i{item_id}.parquet`: Contains base features and time features only.
2.  `data_with_lags_s{store_id}_i{item_id}.parquet`: Contains all features from the "no lags" version, plus lag and rolling window features.

Replace `{store_id}` and `{item_id}` with the specific store and item numbers used during processing (e.g., `data_with_lags_s1_i1.parquet`).

## File Content Descriptions

### 1. `data_no_lags_s{store_id}_i{item_id}.parquet`

This file is suitable for models that do not require or benefit from explicit lag or rolling window features (e.g., basic Prophet, simpler statistical models). It contains:

* The original date and sales data.
* Derived time-based features.
* A `split` column indicating whether the row belongs to the training or validation set.

### 2. `data_with_lags_s{store_id}_i{item_id}.parquet`

This file includes a richer feature set suitable for more complex models that can leverage historical patterns (e.g., XGBoost, LightGBM, LSTM, ARIMA with regressors). It contains:

* **All columns** from the `data_no_lags.parquet` file.
* **Lag features**: Past sales values from specific prior days (e.g., 1 day ago, 7 days ago, 365 days ago).
* **Rolling window features**: Statistical aggregations (mean, std dev, median, min, max) of sales over various preceding time windows (e.g., last 7 days, last 28 days).
* A `split` column indicating whether the row belongs to the training or validation set.

## Column Definitions

### Common Columns (Present in both files)

* `date` (datetime64[ns]): The timestamp for the record.
* `sales` (int64 or float64): The target variable; the actual number of items sold on that date.
* **Time Features:**
    * `year` (int): Year extracted from the date (e.g., 2013).
    * `month` (int): Month extracted from the date (1-12).
    * `day` (int): Day of the month (1-31).
    * `dayofweek` (int): Day of the week (Monday=0, Sunday=6).
    * `dayofyear` (int): Day of the year (1-366).
    * `weekofyear` (int): ISO week of the year (1-53).
    * `quarter` (int): Quarter of the year (1-4).
    * `is_month_start` (int): Indicator (1 if first day of month, 0 otherwise).
    * `is_month_end` (int): Indicator (1 if last day of month, 0 otherwise).
* `split` (string): Indicates if the row belongs to the 'train' or 'validation' set.

### Additional Columns (Present *only* in `data_with_lags.parquet`)

* **Lag Features:** (`sales_lag_N` format, float64 type) - Represent sales from N days prior. Examples based on the script:
    * `sales_lag_1`: Sales from 1 day ago.
    * `sales_lag_7`: Sales from 7 days ago.
    * `sales_lag_14`: Sales from 14 days ago.
    * ... (Generated for lags: `[1, 7, 14, 21, 28, 30, 60, 90, 180, 364, 365]`)
    * `sales_lag_365`: Sales from 365 days ago.
* **Rolling Window Features:** (`sales_roll_AGG_Wd` format, float64 type) - Represent statistical aggregation `AGG` (e.g., `mean`, `std`, `median`, `min`, `max`) over a rolling window of the past `W` days. Examples based on the script (for windows `[7, 14, 28, 60, 90]`):
    * `sales_roll_mean_7d`: Rolling mean sales over the past 7 days.
    * `sales_roll_std_7d`: Rolling standard deviation sales over the past 7 days.
    * ... (All combinations of aggregations and windows are generated)
    * `sales_roll_max_90d`: Rolling maximum sales over the past 90 days.

## How to Use

You can load these Parquet files using pandas in Python. Remember to install `pyarrow` or `fastparquet`.

```python
import pandas as pd

# Define the path to the desired file
file_path_with_lags = 'data/dataset_retail/processed/data_with_lags_s1_i1.parquet'
# or
# file_path_no_lags = 'data/dataset_retail/processed/data_no_lags_s1_i1.parquet'

# Load the data
df = pd.read_parquet(file_path_with_lags) # Choose the file you need

# Separate train and validation sets using the 'split' column
train_df = df[df['split'] == 'train'].copy()
validation_df = df[df['split'] == 'validation'].copy()

# Now you can use train_df and validation_df for modeling
print("Training data shape:", train_df.shape)
print("Validation data shape:", validation_df.shape)

# Example: Prepare features (X) and target (y) for a model
# Ensure to drop non-feature columns like 'date', 'sales', 'split' from X
# target_column = 'sales'
# feature_columns = [col for col in train_df.columns if col not in ['date', 'sales', 'split']]

# X_train = train_df[feature_columns]
# y_train = train_df[target_column]
# X_val = validation_df[feature_columns]
# y_val = validation_df[target_column]
