# src/training/train_prophet_retail.py
"""
Purpose:
Train, evaluate, and save a Prophet model based on parameters
specified in a configuration file. Loads preprocessed data.

How to run (usually called by run_experiment.py):
python src/training/train_prophet_retail.py --config outputs/configs/your_experiment_config.yaml
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
import argparse
import os
import logging # Use logging configured by setup function

# Import shared utility functions from the src package
# Assumes you are running from the project root (comparison_project/)
# and src has an __init__.py file
from src.utils.logging_setup import setup_logging
from src.utils.config_loader import load_config
from src.data.data_loaders import load_processed_data
from src.evaluation.metrics import calculate_metrics, save_metrics, save_predictions
from src.visualization.plot_results import plot_forecast

def prepare_prophet_data(train_df, val_df, config):
    """Prepares dataframes specifically for Prophet model based on config."""
    logging.info("Preparing data specifically for Prophet...")
    target_col = config['dataset']['target_col']

    prophet_train = train_df[['date', target_col]].rename(columns={'date': 'ds', target_col: 'y'})
    prophet_val = val_df[['date', target_col]].rename(columns={'date': 'ds', target_col: 'y'})

    # Identify potential regressors (all columns except date and target)
    all_feature_cols = [col for col in train_df.columns if col not in ['date', target_col]]

    # Decide which regressors to use - potentially controlled by config in the future
    # For now, using a similar logic as before, but referencing config could be added
    # Example: config['models']['prophet'].get('regressors', 'all_features_except_date_target')
    regressors = [col for col in all_feature_cols if 'lag' in col or 'roll' in col or 'dayofweek' in col or 'month' in col or 'year' in col] # Example selection
    logging.info(f"Identified {len(regressors)} potential regressors based on naming convention.")

    # Add selected regressors to prophet dataframes
    final_regressors = []
    for col in regressors:
        if col in train_df.columns and col in val_df.columns:
            prophet_train[col] = train_df[col].values
            prophet_val[col] = val_df[col].values
            final_regressors.append(col)
        else:
            logging.warning(f"Regressor '{col}' missing in loaded data. Skipping.")

    logging.info(f"Using {len(final_regressors)} final regressors.")
    logging.info("Prophet data formatting complete.")
    return prophet_train, prophet_val, final_regressors

def train_prophet_model(train_data, regressors, prophet_params):
    """Initializes, adds regressors, and fits the Prophet model using config params."""
    logging.info("Initializing and training Prophet model...")
    # Initialize Prophet with parameters from config (provide defaults if missing)
    model = Prophet(
        daily_seasonality=prophet_params.get('daily_seasonality', False),
        weekly_seasonality=prophet_params.get('weekly_seasonality', True),
        yearly_seasonality=prophet_params.get('yearly_seasonality', True),
        seasonality_mode=prophet_params.get('seasonality_mode', 'additive'),
        # Add other prophet parameters here from prophet_params dict if needed
        # growth=prophet_params.get('growth', 'linear'),
        # changepoint_prior_scale=prophet_params.get('changepoint_prior_scale', 0.05),
        # holidays=holidays_df # Can load holidays if specified in config
    )

    if regressors:
        logging.info(f"Adding {len(regressors)} regressors to the model.")
        for regressor in regressors:
            # Add regressor mode (additive/multiplicative) if specified in config
            regressor_params = prophet_params.get('regressor_params', {}).get(regressor, {})
            model.add_regressor(
                regressor,
                prior_scale=regressor_params.get('prior_scale', 10.0),
                standardize=regressor_params.get('standardize', 'auto'),
                mode=regressor_params.get('mode', 'additive') # Default to additive
            )

    try:
        model.fit(train_data)
        logging.info("Model fitting complete.")
        return model
    except Exception as e:
        logging.exception("An error occurred during Prophet model fitting.")
        return None


def save_model(model, filename):
    """Saves the trained model using pickle."""
    logging.info(f"Saving the trained model to {filename}...")
    try:
        model_dir = os.path.dirname(filename)
        os.makedirs(model_dir, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.exception(f"Error saving model to {filename}: {e}")

def predict_and_evaluate(model, validation_data, regressors, config):
    """Makes predictions and evaluates the model based on config."""
    logging.info("Predicting on validation set...")
    target_col = config['dataset']['target_col'] # Original target name
    use_log_transform = config['dataset'].get('use_log_transform', False) # Get from config

    # 1. Create future dataframe including regressors
    future_df = validation_data[['ds']].copy()
    if regressors:
        for regressor in regressors:
             if regressor in validation_data.columns:
                 future_df[regressor] = validation_data[regressor].values
             else:
                 logging.error(f"Regressor '{regressor}' missing in validation data for prediction! Filling with 0.")
                 future_df[regressor] = 0 # Or handle differently

    # 2. Predict
    try:
        forecast = model.predict(future_df)
        logging.info("Prediction complete.")
    except Exception as e:
        logging.exception("An error occurred during Prophet prediction.")
        return None, None

    # 3. Process predictions
    y_pred = forecast['yhat']
    y_true = validation_data['y'] # This column was renamed from target_col

    # 4. Inverse transform if necessary
    if use_log_transform:
        logging.info("Applying inverse log transform (expm1) to predictions and true values.")
        try:
            y_pred = np.expm1(y_pred)
            # Ensure y_true is also inverse transformed if it was log(y)
            y_true = np.expm1(y_true)
        except Exception as e:
            logging.exception("Error during inverse log transform.")
            return None, None

    # Ensure non-negative sales
    y_pred = np.maximum(0, y_pred)

    # 5. Evaluate using shared function
    metrics = calculate_metrics(y_true, y_pred)

    return y_pred, y_true, metrics




def main(config_path: str):
    """Main execution pipeline for Prophet model training using config."""

    # --- Load Configuration ---
    config = load_config(config_path)
    if config is None:
        # load_config logs the error, just exit
        exit()

    # --- Extract Basic Info ---
    try:
        experiment_name = config['experiment_name']
        dataset_config = config['dataset']
        store_id = dataset_config['store_id']
        item_id = dataset_config['item_id']
        processed_dir = dataset_config['processed_dir']
        models_base_dir = config['output_base_dir'] # Base dir for results/models
        model_type = 'prophet' # Explicitly set model type
        target_col = dataset_config['target_col']
        use_log_transform = dataset_config.get('use_log_transform', False)

        # Construct specific output paths
        output_dir = os.path.join(models_base_dir, experiment_name, model_type, f"s{store_id}_i{item_id}")
        models_dir = os.path.join(output_dir, "saved_model") # Subdir for model file
        predictions_dir = os.path.join(output_dir, "predictions")
        metrics_dir = os.path.join(output_dir, "metrics")
        figures_dir = os.path.join(output_dir, "figures") # For saving plots

        # Prophet specific parameters from config
        prophet_params = config.get('models', {}).get(model_type, {}).get('params', {})

    except KeyError as e:
        # Use basic logging until setup_logging is called
        print(f"Error: Missing key in configuration file: {e}")
        logging.error(f"Error: Missing key in configuration file: {e}")
        exit()

    # --- Setup Logging ---
    # Use experiment name, model type, store, item in log prefix for clarity
    log_prefix = f"{experiment_name}_{model_type}_s{store_id}_i{item_id}"
    log_filepath = setup_logging('logs', log_prefix) # Assuming global logs dir

    logging.info(f"--- Starting Prophet Model Training for Experiment: {experiment_name} ---")
    logging.info(f"Store ID: {store_id}, Item ID: {item_id}")
    logging.info(f"Using Config File: {config_path}")
    logging.info(f"Target Column: {target_col}, Use Log Transform: {use_log_transform}")
    logging.info(f"Processed Data Dir: {processed_dir}")
    logging.info(f"Base Output Dir: {models_base_dir}")
    logging.info(f"Prophet Params: {prophet_params}")

    try:
        # Step 1: Load Data
        train_featured, validation_featured = load_processed_data(processed_dir, store_id, item_id)
        if train_featured is None: raise ValueError("Failed to load data.")

        # Step 2: Prepare Prophet Data
        prophet_train, prophet_val, regressors = prepare_prophet_data(train_featured, validation_featured, config)
        if prophet_train is None: raise ValueError("Failed to prepare Prophet data.")

        # Step 3: Train Model
        model = train_prophet_model(prophet_train, regressors, prophet_params)
        if model is None: raise ValueError("Failed to train Prophet model.")

        # Step 4: Save Model
        model_filename = f'{model_type}_s{store_id}_i{item_id}.pkl'
        model_path = os.path.join(models_dir, model_filename)
        save_model(model, model_path)

        # Step 5 & 6: Predict and Evaluate
        y_pred, y_true, metrics = predict_and_evaluate(model, prophet_val, regressors, config)
        if metrics is None: raise ValueError("Failed to evaluate predictions.")

        # Step 6b: Save Metrics
        metrics_filename = f'{model_type}_metrics_s{store_id}_i{item_id}.json'
        metrics_path = os.path.join(metrics_dir, metrics_filename)
        save_metrics(metrics, metrics_path)

        # Step 7: Save Predictions
        pred_filename = f'validation_{model_type}_s{store_id}_i{item_id}.csv'
        pred_path = os.path.join(predictions_dir, pred_filename)
        save_predictions(prophet_val, y_pred, y_true, pred_path)

        # Step 8: Plot Results (and save)
        plot_forecast(
            dates=prophet_val['ds'],
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_type.capitalize(),
            store_id=store_id,
            item_id=item_id,
            output_dir=figures_dir # Pass directory to save figure
        )

        logging.info(f"--- Prophet Model Training Pipeline Finished Successfully ---")

    except Exception as e:
        logging.exception("An critical error occurred in the main Prophet training pipeline")

# --- Script Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Prophet model using a configuration file.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file.')
    args = parser.parse_args()

    main(config_path=args.config)