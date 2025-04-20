# src/evaluation/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import os
import logging

def calculate_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict:
    """
    Calculates standard regression metrics.

    Args:
        y_true (np.ndarray | pd.Series): Array of true values.
        y_pred (np.ndarray | pd.Series): Array of predicted values.

    Returns:
        dict: Dictionary containing calculated metrics (RMSE, MAE, MAPE).
              Returns empty dict if inputs are invalid.
    """
    metrics = {}
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        logging.error(f"Invalid input for metrics calculation: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")
        return metrics

    try:
        # Ensure numpy arrays for calculations
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)

        # Calculate MAPE carefully, avoiding division by zero
        mask = y_true != 0
        if np.sum(mask) > 0:
            metrics['MAPE'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            metrics['MAPE'] = np.nan # Or indicate calculation wasn't possible
            logging.warning("MAPE calculation skipped as all true values are zero.")

        logging.info(f"Metrics calculated: {metrics}")

    except Exception as e:
        logging.exception(f"Error calculating metrics: {e}")
        return {} # Return empty dict on error

    return metrics

def save_metrics(metrics_dict: dict, filepath: str):
    """
    Saves a metrics dictionary to a JSON file.

    Args:
        metrics_dict (dict): Dictionary containing metric names and values.
        filepath (str): Path to save the JSON file.
    """
    if not metrics_dict:
        logging.warning(f"Metrics dictionary is empty. Skipping save to {filepath}")
        return

    logging.info(f"Saving metrics to {filepath}...")
    try:
        # Ensure directory exists
        output_dir = os.path.dirname(filepath)
        os.makedirs(output_dir, exist_ok=True)

        # Save as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_metrics = {k: (float(v) if isinstance(v, (np.number, np.bool_)) else v) for k, v in metrics_dict.items()}
            json.dump(serializable_metrics, f, indent=4)
        logging.info("Metrics saved successfully.")
    except TypeError as e:
        logging.exception(f"Error serializing metrics to JSON: {e}. Metrics: {metrics_dict}")
    except Exception as e:
        logging.exception(f"An error occurred saving metrics to {filepath}: {e}")
        
def save_predictions(dates_df, predictions, true_values, filename):
    """Saves predictions alongside actual values to a specific path."""
    if predictions is None or true_values is None:
        logging.warning("Predictions or true values are None, skipping saving.")
        return
    logging.info(f"Saving validation predictions to {filename}...")
    results_df = dates_df[['ds']].copy()
    # Ensure indices align if using pandas Series
    results_df['actual'] = true_values.values if isinstance(true_values, pd.Series) else true_values
    results_df['predicted'] = predictions.values if isinstance(predictions, pd.Series) else predictions

    try:
        pred_dir = os.path.dirname(filename)
        os.makedirs(pred_dir, exist_ok=True)
        results_df.to_csv(filename, index=False)
        logging.info(f"Predictions saved to {filename}")
    except Exception as e:
        logging.exception(f"Error saving predictions to {filename}: {e}")