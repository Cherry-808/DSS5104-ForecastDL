# src/training/train_tcn_retail.py
"""
Purpose: Train, evaluate, and save a TCN model based on config.
Requires 'keras-tcn' library: pip install keras-tcn
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Try importing TCN, handle if not installed
try:
    from tcn import TCN # From keras-tcn library
except ImportError:
    print("Error: 'keras-tcn' library not found. Please install using 'pip install keras-tcn'")
    TCN = None # Set to None if import fails

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
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
    # Reuse scaler and sequence functions
    from src.training.train_transformer_retail import scale_features, create_sequences, predict_and_evaluate as predict_evaluate_base
    # Reuse train_model function
    from src.training.train_lstm_retail import train_model
except ModuleNotFoundError as e:
    print(f"Error importing utility modules or functions from other scripts: {e}")
    sys.exit(1)
except ImportError as e:
     print(f"Error importing utility modules or functions from other scripts: {e}")
     sys.exit(1)


# --- Default TCN Parameters ---
DEFAULT_TCN_FILTERS = 64
DEFAULT_TCN_KERNEL_SIZE = 3
DEFAULT_TCN_DILATIONS = [1, 2, 4, 8] # Example dilations
DEFAULT_TCN_DROPOUT = 0.1
DEFAULT_SEQUENCE_LENGTH = 60
DEFAULT_LR = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 10

def build_tcn_model(input_shape, tcn_filters, kernel_size, dilations, dropout_rate):
    """Builds a simple TCN model using keras-tcn."""
    if TCN is None:
         logging.error("keras-tcn library is not installed. Cannot build TCN model.")
         return None

    logging.info(f"Building TCN model with filters={tcn_filters}, kernel_size={kernel_size}, dilations={dilations}")
    inputs = keras.Input(shape=input_shape)

    # TCN Layer from keras-tcn
    # nb_stacks=1 means one block of TCN layers with increasing dilations
    x = TCN(nb_filters=tcn_filters,
            kernel_size=kernel_size,
            nb_stacks=1,
            dilations=dilations,
            padding='causal', # Important for time series
            use_skip_connections=True,
            dropout_rate=dropout_rate,
            activation='relu',
            return_sequences=False # Get output only from the last time step
           )(inputs)

    # Optional: Add Dense layers after TCN
    # x = layers.Dense(tcn_filters // 2, activation='relu')(x)
    # x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation='linear')(x)
    model = keras.Model(inputs, outputs)
    logging.info("TCN model built.")
    return model

# --- Main Function ---
def main(config_path: str):
    config = load_config(config_path)
    if config is None: exit()

    try:
        # --- Extract Config Parameters ---
        experiment_name = config['experiment_name']
        dataset_config = config['dataset']
        store_id = dataset_config['store_id']
        item_id = dataset_config['item_id']
        processed_dir = dataset_config['processed_dir']
        output_base_dir = config['output_base_dir']
        model_type = 'tcn' # Set model type
        target_col = dataset_config['target_col']
        use_log_transform = dataset_config.get('use_log_transform', False)

        # Output paths
        output_dir = os.path.join(output_base_dir, experiment_name, model_type, f"s{store_id}_i{item_id}")
        models_dir = os.path.join(output_dir, "saved_model")
        predictions_dir = os.path.join(output_dir, "predictions")
        metrics_dir = os.path.join(output_dir, "metrics")
        figures_dir = os.path.join(output_dir, "figures")
        scaler_filename = f'scaler_s{store_id}_i{item_id}.pkl'
        scaler_path = os.path.join(models_dir, scaler_filename)
        model_checkpoint_filename = f'{model_type}_best_s{store_id}_i{item_id}.keras'
        model_checkpoint_path = os.path.join(models_dir, model_checkpoint_filename)


        # Model & Training Params
        model_params = config.get('models', {}).get(model_type, {}).get('params', {})
        seq_len = model_params.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
        tcn_filters = model_params.get('nb_filters', DEFAULT_TCN_FILTERS)
        kernel_size = model_params.get('kernel_size', DEFAULT_TCN_KERNEL_SIZE)
        dilations = model_params.get('dilations', DEFAULT_TCN_DILATIONS)
        dropout = model_params.get('dropout_rate', DEFAULT_TCN_DROPOUT)


    except KeyError as e:
        logging.error(f"Error: Missing key in configuration file: {e}.")
        exit()

    # --- Setup Logging ---
    log_prefix = f"{experiment_name}_{model_type}_s{store_id}_i{item_id}"
    log_filepath = setup_logging('logs', log_prefix)
    logging.info(f"--- Starting TCN Model Training: {log_prefix} ---")
    logging.info(f"Config File: {config_path}")
    logging.info(f"Sequence Length: {seq_len}, TCN Filters: {tcn_filters}, Kernel Size: {kernel_size}, Dilations: {dilations}, Dropout: {dropout}")

    try:
        # 1. Load Data
        train_featured, validation_featured = load_processed_data(processed_dir, store_id, item_id)
        if train_featured is None: raise ValueError("Failed to load data.")

        # 2. Scale Features
        numerical_features = [col for col in train_featured.columns if col not in ['date'] and train_featured[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        logging.info(f"Identified {len(numerical_features)} numerical features for scaling.")
        target_in_num_features = target_col if target_col in numerical_features else None
        train_scaled, validation_scaled, scaler = scale_features(train_featured, validation_featured, numerical_features, scaler_path)
        if train_scaled is None: raise ValueError("Failed to scale features.")

        # 3. Create Sequences
        feature_cols_for_dl = [col for col in train_scaled.columns if col not in ['date', target_col]]
        X_train_seq, y_train_seq = create_sequences(train_scaled, seq_len, feature_cols_for_dl, target_col)
        X_val_seq, y_val_seq = create_sequences(validation_scaled, seq_len, feature_cols_for_dl, target_col)
        y_val_seq_orig = validation_featured[target_col].values[seq_len:]

        if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0:
             raise ValueError("Not enough data to create sequences.")

        # 4. Build Model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = build_tcn_model(input_shape, tcn_filters, kernel_size, dilations, dropout)
        if model is None: raise ValueError("Failed to build TCN model. Is keras-tcn installed?")
        model.summary(print_fn=logging.info)

        # 5. Train Model (reuse generic trainer)
        history, best_model_path = train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, config, model_type)
        if history is None: raise ValueError("Model training failed.")

        # 6. Predict and Evaluate (reuse generic evaluator)
        y_pred, y_true, metrics = predict_evaluate_base(
            best_model_path, scaler, X_val_seq, y_val_seq_orig,
            numerical_features, target_in_num_features, use_log_transform
        )
        if metrics is None: raise ValueError("Failed to evaluate predictions.")

        # 7. Save Metrics
        metrics_filename = f'{model_type}_metrics_s{store_id}_i{item_id}.json'
        metrics_path = os.path.join(metrics_dir, metrics_filename)
        save_metrics(metrics, metrics_path)

        # 8. Save Predictions
        pred_filename = f'validation_{model_type}_s{store_id}_i{item_id}.csv'
        pred_path = os.path.join(predictions_dir, pred_filename)
        validation_dates = validation_featured['date'].iloc[seq_len:]
        from src.evaluation.metrics import save_predictions # Assuming moved
        save_predictions(validation_dates.to_frame(name='ds'), y_pred, y_true, pred_path)

        # 9. Plot Results
        plot_forecast(
            dates=validation_dates, y_true=y_true, y_pred=y_pred,
            model_name=model_type.capitalize(), store_id=store_id, item_id=item_id,
            output_dir=figures_dir
        )

        logging.info(f"--- TCN Model Training Pipeline Finished Successfully ---")

    except Exception as e:
        logging.exception(f"An critical error occurred in the main TCN pipeline for Store={store_id}, Item={item_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TCN model using a configuration file.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file.')
    args = parser.parse_args()
    # Check if TCN was imported successfully before running main
    if TCN is None:
        logging.error("Exiting because TCN library is not available.")
        sys.exit(1)
    main(config_path=args.config)