# src/training/train_arima_retail.py
"""
Purpose: Train, evaluate, and save an ARIMA model based on config.
"""
import pandas as pd
import numpy as np
import pickle
import argparse
import os
import logging
import datetime
import sys
from statsmodels.tsa.arima.model import ARIMA
# Or use pmdarima for auto-selection:
# import pmdarima as pm

# Add project root for src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.utils.logging_setup import setup_logging
    from src.utils.config_loader import load_config
    from src.data.data_loaders import load_processed_data # Load featured data (maybe not needed if no exog?)
    # Need raw data loader or just use target from featured data
    from src.evaluation.metrics import calculate_metrics, save_metrics
    from src.visualization.plot_results import plot_forecast
except ModuleNotFoundError as e:
    print(f"Error importing utility modules: {e}")
    sys.exit(1)

# --- Default ARIMA Parameters ---
DEFAULT_ARIMA_ORDER = (1, 1, 1) # Example non-seasonal order (p, d, q)
DEFAULT_SEASONAL_ORDER = (0, 0, 0, 0) # Example seasonal order (P, D, Q, m) - m=0 means non-seasonal

def train_arima_model(train_series, arima_params):
    """Fits the ARIMA model."""
    logging.info("Initializing and fitting ARIMA model...")
    order = tuple(arima_params.get('order', DEFAULT_ARIMA_ORDER))
    seasonal_order = tuple(arima_params.get('seasonal_order', DEFAULT_SEASONAL_ORDER))
    # Note: Exog variables not handled in this basic version
    # Note: Stationarity checks and differencing (d, D) determination should ideally happen before this based on data analysis
    logging.info(f"Using ARIMA order={order}, seasonal_order={seasonal_order}")

    try:
        model = ARIMA(train_series, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit()
        logging.info("Model fitting complete.")
        logging.info(model_fit.summary()) # Log model summary
        return model_fit
    # except LinAlgError, ValueError etc. from statsmodels
    except Exception as e:
        logging.exception("An error occurred during ARIMA model fitting.")
        return None

def predict_and_evaluate(model_fit, validation_series, config):
    """Makes predictions and evaluates the ARIMA model."""
    logging.info("Predicting on validation set using ARIMA...")
    target_col = config['dataset']['target_col']
    use_log_transform = config['dataset'].get('use_log_transform', False)

    try:
        # Get predictions - predict requires start and end indices/timestamps
        start_index = validation_series.index[0]
        end_index = validation_series.index[-1]
        # Using get_prediction for confidence intervals, forecast for just points
        # forecast_result = model_fit.get_forecast(steps=len(validation_series))
        # y_pred = forecast_result.predicted_mean
        # conf_int = forecast_result.conf_int(alpha=0.05) # Optional
        y_pred = model_fit.predict(start=start_index, end=end_index)
        logging.info(f"Prediction complete. Predicted steps: {len(y_pred)}")

        # Align with validation series (predict might return slightly different index/length)
        y_true = validation_series.loc[y_pred.index] # Align true values

        # Inverse transform if necessary
        if use_log_transform:
            logging.info("Applying inverse log transform (expm1)...")
            y_pred = np.expm1(y_pred)
            y_true = np.expm1(y_true)

        y_pred = np.maximum(0, y_pred) # Ensure non-negative

        # Evaluate
        metrics = calculate_metrics(y_true, y_pred)
        if metrics:
             log_msg = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
             logging.info(f"Validation Metrics: {log_msg}")
        else:
             logging.error("Failed to calculate metrics for ARIMA.")

        return y_pred, y_true, metrics

    except Exception as e:
        logging.exception("An error occurred during ARIMA prediction or evaluation.")
        return None, None, None

# --- Saving/Loading ARIMA (uses pickle) ---
def save_model(model_fit, filename):
    logging.info(f"Saving the trained ARIMA model to {filename}...")
    try:
        model_dir = os.path.dirname(filename)
        os.makedirs(model_dir, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(model_fit, f)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.exception(f"Error saving ARIMA model to {filename}: {e}")

# --- Main Function ---
def main(config_path: str):
    config = load_config(config_path)
    if config is None: exit()

    try:
        experiment_name = config['experiment_name']
        dataset_config = config['dataset']
        store_id = dataset_config['store_id']
        item_id = dataset_config['item_id']
        processed_dir = dataset_config['processed_dir'] # May load from here or raw
        output_base_dir = config['output_base_dir']
        model_type = 'arima'
        target_col = dataset_config['target_col']
        use_log_transform = dataset_config.get('use_log_transform', False)

        output_dir = os.path.join(output_base_dir, experiment_name, model_type, f"s{store_id}_i{item_id}")
        models_dir = os.path.join(output_dir, "saved_model")
        predictions_dir = os.path.join(output_dir, "predictions")
        metrics_dir = os.path.join(output_dir, "metrics")
        figures_dir = os.path.join(output_dir, "figures")

        arima_params = config.get('models', {}).get(model_type, {}).get('params', {})

    except KeyError as e:
        logging.error(f"Error: Missing key in configuration file: {e}. Check config structure.")
        exit()

    log_prefix = f"{experiment_name}_{model_type}_s{store_id}_i{item_id}"
    log_filepath = setup_logging('logs', log_prefix) # Setup logging early

    logging.info(f"--- Starting ARIMA Model Training: {log_prefix} ---")
    logging.info(f"Config File: {config_path}")
    logging.info(f"Target Column: {target_col}, Use Log Transform: {use_log_transform}")
    logging.info(f"ARIMA Params: {arima_params}")

    try:
        # 1. Load Data - ARIMA needs the time series, usually without extra features unless using exog
        # We load the *processed* data to easily get the train/validation split based on date
        train_featured, validation_featured = load_processed_data(processed_dir, store_id, item_id)
        if train_featured is None: raise ValueError("Failed to load data.")

        # Prepare time series data (set date as index)
        train_series = train_featured.set_index('date')[target_col]
        validation_series = validation_featured.set_index('date')[target_col]

        # Apply log transform if specified (BEFORE fitting)
        if use_log_transform:
            logging.info("Applying log transform (log1p) to target series...")
            train_series = np.log1p(train_series)
            validation_series = np.log1p(validation_series) # Need this for comparison if inverse transform happens later

        # TODO: Add stationarity checks (e.g., ADF test) and differencing based on 'd'/'D' in params if needed.
        # This typically happens *before* calling train_arima_model if 'd'/'D' > 0.
        # Example: if arima_params.get('order', DEFAULT_ARIMA_ORDER)[1] > 0: train_series = train_series.diff().dropna() ...

        # 2. Train Model
        model_fit = train_arima_model(train_series, arima_params)
        if model_fit is None: raise ValueError("Failed to train ARIMA model.")

        # 3. Save Model
        model_filename = f'{model_type}_s{store_id}_i{item_id}.pkl'
        model_path = os.path.join(models_dir, model_filename)
        save_model(model_fit, model_path)

        # 4. Predict and Evaluate
        # Pass the validation series (possibly log-transformed)
        y_pred, y_true, metrics = predict_and_evaluate(model_fit, validation_series, config)
        if metrics is None: raise ValueError("Failed to evaluate predictions.")

        # 5. Save Metrics
        metrics_filename = f'{model_type}_metrics_s{store_id}_i{item_id}.json'
        metrics_path = os.path.join(metrics_dir, metrics_filename)
        save_metrics(metrics, metrics_path)

        # 6. Save Predictions
        pred_filename = f'validation_{model_type}_s{store_id}_i{item_id}.csv'
        pred_path = os.path.join(predictions_dir, pred_filename)
        # Create dates DataFrame for saving - use index from validation_series
        dates_df = validation_series.reset_index()[['date']].rename(columns={'date': 'ds'})
        # Align predictions and true values (which might have different index after predict/log transform)
        results_df = pd.DataFrame({'ds': dates_df['ds'], 'actual': y_true, 'predicted': y_pred})
        results_df.to_csv(pred_path, index=False) # Simple save for now
        logging.info(f"Predictions saved to {pred_path}")


        # 7. Plot Results
        plot_forecast(
            dates=results_df['ds'], # Use dates from results df
            y_true=results_df['actual'],
            y_pred=results_df['predicted'],
            model_name=model_type.capitalize(),
            store_id=store_id,
            item_id=item_id,
            output_dir=figures_dir
        )

        logging.info(f"--- ARIMA Model Training Pipeline Finished Successfully ---")

    except Exception as e:
        logging.exception(f"An critical error occurred in the main ARIMA pipeline for Store={store_id}, Item={item_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ARIMA model using a configuration file.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file.')
    args = parser.parse_args()
    main(config_path=args.config)