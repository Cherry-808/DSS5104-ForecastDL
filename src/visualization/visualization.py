import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# --- Heatmap ---
def plot_heatmap(metrics_df, index_col='Dataset', metric_cols=None, title_prefix='Metric Heatmap'):
    if metric_cols is None:
        metric_cols = ['rmse', 'mse', 'mae', 'mape', 'r2', 'adj_r2']

    if index_col not in metrics_df.columns:
        metrics_df = metrics_df.reset_index()

    for metric in metric_cols:
        if metric in metrics_df.columns:
            pivot_df = metrics_df.pivot(index=index_col, columns='model_name', values=metric)
            plt.figure(figsize=(10, 6))
            sns.heatmap(
                pivot_df,
                annot=True,
                cmap=sns.color_palette("Blues", as_cmap=True),  # Light to dark blue gradient
                fmt='.3f',
                linewidths=0.5,
                linecolor='gray'
            )
            plt.title(f"{title_prefix}: {metric.upper()}")
            plt.ylabel(index_col)
            plt.xlabel("Model")
            plt.tight_layout()
            plt.show()
            
# --- Basic Time-series Plot ---
def plot_time_series(data, title='Time Series Data'):
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Original Data')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# def evaluate_forecast(test, predictions):
#     error = mean_squared_error(test, predictions)
#     print(f'Test Mean Squared Error: {error:.3f}')
#     plt.figure(figsize=(10, 6))
#     plt.plot(test.index, test, label='Actual Data')
#     plt.plot(test.index, predictions, color='red', label='Predicted Data')
#     plt.title('Actual vs Predicted')
#     plt.xlabel('Date')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.show()
