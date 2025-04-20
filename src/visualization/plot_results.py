# src/visualization/plot_results.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging

def plot_forecast(dates: pd.Series | np.ndarray,
                  y_true: np.ndarray | pd.Series,
                  y_pred: np.ndarray | pd.Series,
                  model_name: str,
                  store_id: int,
                  item_id: int,
                  title_suffix: str = "(Validation Set)",
                  output_dir: str | None = None):
    """
    Plots actual vs. predicted values and optionally saves the figure.

    Args:
        dates (pd.Series | np.ndarray): Corresponding dates for the data points.
        y_true (np.ndarray | pd.Series): Array of true values.
        y_pred (np.ndarray | pd.Series): Array of predicted values.
        model_name (str): Name of the model (e.g., 'Prophet', 'Transformer').
        store_id (int): Store ID.
        item_id (int): Item ID.
        title_suffix (str): Suffix for the plot title (default: "(Validation Set)").
        output_dir (str | None): Directory to save the plot image. If None, plot is only shown.
    """
    if len(dates) != len(y_true) or len(dates) != len(y_pred):
         logging.error(f"Length mismatch in plot_forecast: dates={len(dates)}, y_true={len(y_true)}, y_pred={len(y_pred)}")
         return

    logging.info(f"Plotting results for {model_name} - Store {store_id}, Item {item_id}...")
    try:
        plt.figure(figsize=(15, 6))
        plt.plot(dates, y_true, label='Actual Sales', marker='.', markersize=4, alpha=0.8)
        plt.plot(dates, y_pred, label=f'{model_name} Forecast', marker='.', markersize=4, alpha=0.7)
        plt.title(f'{model_name} Forecast vs Actual Sales {title_suffix} - Store {store_id}, Item {item_id}')
        plt.xlabel('Date')
        plt.ylabel('Sales') # Consider adding if log transformed
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Adjust layout

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_filename = f'{model_name.lower()}_forecast_s{store_id}_i{item_id}.png'
            filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(filepath)
            logging.info(f"Plot saved to {filepath}")

        plt.show() # Display the plot
        plt.close() # Close the figure to free memory

    except Exception as e:
        logging.exception("An error occurred during plotting.")