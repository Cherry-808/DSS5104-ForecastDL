# src/training/train_xgboost_retail.py
"""
Purpose: Train, evaluate, and save an XGBoost model based on config.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import argparse
import os
import logging
import datetime
import sys

# Add project root for src imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.utils.logging_setup import setup_logging
    from src.utils.config_loader import load_config
    from src.data.data_loaders import load_processed_data
    from src.evaluation.metrics import calculate_metrics, save_metrics
    from src.visualization.plot_results import plot_forecast
except ModuleNotFoundError as e:
    print(f"Error importing utility modules: {e}")
    sys.exit(1)

# --- Default XGBoost Parameters ---
DEFAULT_XGB_PARAMS = {
    'objective': 'reg:squarederror', # Regression task
    'eval_metric': 'rmse',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

def prepare_xgboost_data(train_df, val_df, feature_cols, target_col):
    """Prepares X and y matrices for XGBoost."""
    logging.info("Preparing data for XGBoost...")
    try:
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        logging.info(f"Data prepared: X_train shape={X_train.shape}, X_val shape={X_val.shape}")
        return X_train, y_train, X_val, y_val
    except KeyError as e:
        logging.error(f"Error: Feature column mismatch during XGBoost data prep: {e}. Check feature engineering.")
        return None, None, None, None
    except Exception as e:
        logging.exception("An error occurred during XGBoost data preparation.")
        return None, None, None, None


def train_xgboost_model(X_train, y_train, X_val, y_val, xgb_params):
    """Initializes and fits the XGBoost model."""
    logging.info("Initializing and fitting XGBoost model...")
    # Use early stopping
    early_stopping_rounds = xgb_params.pop('early_stopping_rounds', 20) # Get from params or use default

    try:
        model = xgb.XGBRegressor(**xgb_params) # Unpack params from config
        logging.info(f"Using XGBoost parameters: {model.get_params()}") # Log effective params
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  early_stopping_rounds=early_stopping_rounds,
                  verbose=False) # Set verbose=True or a number to see training progress
        logging.info("Model fitting complete.")
        logging.info(f"Best iteration: {model.best_iteration}, Best score (RMSE on validation): {model.best_score:.4f}")
        return model
    except Exception as e:
        logging.exception("An error occurred during XGBoost model fitting.")
        return None

def save_model(model, filename):
    """Saves the trained XGBoost model."""
    logging.info(f"Saving the trained XGBoost model to {filename}...")
    try:
        model_dir = os.path.dirname(filename)
        os.makedirs(model_dir, exist_ok=True)
        # Use XGBoost's saving method (more robust than pickle for XGB)
        model.save_model(filename)
        # Alternatively, use pickle:
        # with open(filename, 'wb') as f:
        #     pickle.dump(model, f)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.exception(f"Error saving XGBoost model to {filename}: {e}")

def predict_and_evaluate(model, X_val, y_val_orig, config):
    """Makes predictions and evaluates the XGBoost model."""
    logging.info("Predicting on validation set using XGBoost...")
    target_col = config['dataset']['target_col']
    use_log_transform = config['dataset'].get('use_log_transform', False)

    try:
        y_pred = model.predict(X_val)
        logging.info("Prediction complete.")

        y_true = y_val_orig # Use original validation targets

        # Inverse transform if necessary
        if use_log_transform:
            logging.info("Applying inverse log transform (expm1)...")
            y_pred = np.expm1(y_pred)
            y_true = np.expm1(y_true) # Ensure original y_val was log-transformed

        y_pred = np.maximum(0, y_pred) # Ensure non-negative

        # Evaluate
        # For Adjusted R2, n_features is the number of columns in X_val
        n_samples = len(y_true)
        n_features = X_val.shape[1]
        metrics = calculate_metrics(y_true, y_pred, n_samples, n_features)

        if metrics:
             log_msg = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
             logging.info(f"Validation Metrics: {log_msg}")
        else:
             logging.error("Failed to calculate metrics for XGBoost.")

        return y_pred, y_true, metrics

    except Exception as e:
        logging.exception("An error occurred during XGBoost prediction or evaluation.")
        return None, None, None

# --- Main Function ---
def main(config_path: str):
    config = load_config(config_path)
    if config is None: exit()

    try:
        experiment_name = config['experiment_name']
        dataset_config = config['dataset']
        store_id = dataset_config['store_id']
        item_id = dataset_config['item_id']
        processed_dir = dataset_config['processed_dir']
        output_base_dir = config['output_base_dir']
        model_type = 'xgboost'
        target_col = dataset_config['target_col']
        use_log_transform = dataset_config.get('use_log_transform', False)

        output_dir = os.path.join(output_base_dir, experiment_name, model_type, f"s{store_id}_i{item_id}")
        models_dir = os.path.join(output_dir, "saved_model")
        predictions_dir = os.path.join(output_dir, "predictions")
        metrics_dir = os.path.join(output_dir, "metrics")
        figures_dir = os.path.join(output_dir, "figures")

        # Combine default params with config params, config overrides default
        xgb_params_config = config.get('models', {}).get(model_type, {}).get('params', {})
        xgb_params = {**DEFAULT_XGB_PARAMS, **xgb_params_config}


    except KeyError as e:
        logging.error(f"Error: Missing key in configuration file: {e}.")
        exit()

    log_prefix = f"{experiment_name}_{model_type}_s{store_id}_i{item_id}"
    log_filepath = setup_logging('logs', log_prefix)

    logging.info(f"--- Starting XGBoost Model Training: {log_prefix} ---")
    logging.info(f"Config File: {config_path}")
    logging.info(f"Target Column: {target_col}, Use Log Transform: {use_log_transform}")
    logging.info(f"XGBoost Resolved Params: {xgb_params}")

    try:
        # 1. Load Data
        train_featured, validation_featured = load_processed_data(processed_dir, store_id, item_id)
        if train_featured is None: raise ValueError("Failed to load data.")

        # 2. Prepare XGBoost Data
        #    Identify features (all cols except date and target)
        feature_cols = [col for col in train_featured.columns if col not in ['date', target_col]]
        X_train, y_train, X_val, y_val = prepare_xgboost_data(train_featured, validation_featured, feature_cols, target_col)
        if X_train is None: raise ValueError("Failed to prepare XGBoost data.")

        # Apply log transform if specified (BEFORE fitting)
        y_train_final = y_train
        y_val_orig_final = y_val.copy() # Keep original for evaluation comparison
        if use_log_transform:
            logging.info("Applying log transform (log1p) to target variable for training...")
            y_train_final = np.log1p(y_train)
            # Do NOT transform y_val here, it's the original target. predict_and_evaluate handles inverse transform.

        # 3. Train Model
        model = train_xgboost_model(X_train, y_train_final, X_val, np.log1p(y_val) if use_log_transform else y_val, xgb_params)
        if model is None: raise ValueError("Failed to train XGBoost model.")

        # 4. Save Model
        model_filename = f'{model_type}_s{store_id}_i{item_id}.json' # Use json orubj format for XGB
        model_path = os.path.join(models_dir, model_filename)
        save_model(model, model_path)

        # 5. Predict and Evaluate
        #    Pass original (untransformed) y_val for final comparison
        y_pred, y_true, metrics = predict_and_evaluate(model, X_val, y_val_orig_final, config)
        if metrics is None: raise ValueError("Failed to evaluate predictions.")

        # 6. Save Metrics
        metrics_filename = f'{model_type}_metrics_s{store_id}_i{item_id}.json'
        metrics_path = os.path.join(metrics_dir, metrics_filename)
        save_metrics(metrics, metrics_path)

        # 7. Save Predictions
        pred_filename = f'validation_{model_type}_s{store_id}_i{item_id}.csv'
        pred_path = os.path.join(predictions_dir, pred_filename)
        # Use validation_featured for dates
        dates_df = validation_featured[['date']].rename(columns={'date': 'ds'})
        # Reuse the save_predictions function (assuming it's moved to utils or evaluation)
        # Need to import it if moved
        from src.evaluation.metrics import save_predictions # Example if moved to metrics.py
        save_predictions(dates_df, pd.Series(y_pred, index=y_true.index), y_true, pred_path)


        # 8. Plot Results
        plot_forecast(
            dates=dates_df['ds'],
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_type.capitalize(),
            store_id=store_id,
            item_id=item_id,
            output_dir=figures_dir
        )

        logging.info(f"--- XGBoost Model Training Pipeline Finished Successfully ---")

    except Exception as e:
        logging.exception(f"An critical error occurred in the main XGBoost pipeline for Store={store_id}, Item={item_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost model using a configuration file.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file.')
    args = parser.parse_args()
    main(config_path=args.config)