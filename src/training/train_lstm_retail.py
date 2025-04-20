# src/training/train_lstm_retail.py
"""
Purpose: Train, evaluate, and save an LSTM model based on config.
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
    # Reuse scaler and sequence functions if they are generic enough
    from src.training.train_transformer_retail import scale_features, create_sequences, predict_and_evaluate as predict_evaluate_base # Reuse from transformer for now
except ModuleNotFoundError as e:
    print(f"Error importing utility modules or functions from transformer script: {e}")
    sys.exit(1)
except ImportError as e:
     print(f"Error importing utility modules or functions from transformer script: {e}")
     sys.exit(1)


# --- Default LSTM Parameters ---
DEFAULT_LSTM_UNITS = [64, 32] # Example: List for multiple layers
DEFAULT_LSTM_DROPOUT = 0.1
DEFAULT_LSTM_RECURRENT_DROPOUT = 0.1
DEFAULT_SEQUENCE_LENGTH = 60 # Default sequence length
DEFAULT_LR = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 10

def build_lstm_model(input_shape, lstm_units, dropout, recurrent_dropout):
    """Builds a simple LSTM model."""
    logging.info(f"Building LSTM model with units: {lstm_units}")
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape)) # Input shape: (seq_length, num_features)

    # Add LSTM layers dynamically based on the list provided
    for i, units in enumerate(lstm_units):
        is_last_lstm = (i == len(lstm_units) - 1)
        model.add(layers.LSTM(
            units=units,
            activation='relu', # or 'tanh'
            return_sequences=not is_last_lstm, # Only last LSTM layer returns single output
            dropout=dropout,
            recurrent_dropout=recurrent_dropout
        ))
        # Consider adding BatchNormalization after LSTM layers
        # model.add(layers.BatchNormalization())

    # Add a Dense output layer
    model.add(layers.Dense(1, activation='linear')) # Linear activation for regression
    logging.info("LSTM model built.")
    return model

# --- train_model function (can likely reuse from transformer or make generic) ---
def train_model(model, X_train, y_train, X_val, y_val, config, model_type):
    """Compiles and trains the Keras model with callbacks."""
    logging.info(f"Compiling and training the {model_type} model...")
    try:
        # Extract relevant training params from config
        params = config.get('models', {}).get(model_type, {}).get('params', {})
        lr = params.get('learning_rate', DEFAULT_LR)
        epochs = params.get('epochs', DEFAULT_EPOCHS)
        batch_size = params.get('batch_size', DEFAULT_BATCH_SIZE)
        patience = params.get('early_stopping_patience', DEFAULT_PATIENCE)

        # Define output paths from config
        experiment_name = config['experiment_name']
        store_id = config['dataset']['store_id']
        item_id = config['dataset']['item_id']
        output_dir = os.path.join(config['output_base_dir'], experiment_name, model_type, f"s{store_id}_i{item_id}")
        models_dir = os.path.join(output_dir, "saved_model")
        model_checkpoint_filename = f'{model_type}_best_s{store_id}_i{item_id}.keras'
        model_checkpoint_path = os.path.join(models_dir, model_checkpoint_filename)
        os.makedirs(models_dir, exist_ok=True)

        model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=["mean_absolute_error"]
        )
        logging.info(f"Model compiled with learning rate: {lr}")

        # Callbacks
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path, save_weights_only=False,
            monitor='val_loss', mode='min', save_best_only=True, verbose=1
        )
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1
        )

        # Training
        logging.info(f"Starting training for {epochs} epochs with batch size {batch_size}...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint_callback, early_stopping_callback],
            verbose=2
        )
        logging.info("Model training complete.")
        return history, model_checkpoint_path

    except Exception as e:
        logging.exception(f"An error occurred during {model_type} model compilation or training.")
        return None, None


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
        model_type = 'lstm' # Set model type
        target_col = dataset_config['target_col']
        use_log_transform = dataset_config.get('use_log_transform', False)

        # Output paths
        output_dir = os.path.join(output_base_dir, experiment_name, model_type, f"s{store_id}_i{item_id}")
        models_dir = os.path.join(output_dir, "saved_model")
        predictions_dir = os.path.join(output_dir, "predictions")
        metrics_dir = os.path.join(output_dir, "metrics")
        figures_dir = os.path.join(output_dir, "figures")
        scaler_filename = f'scaler_s{store_id}_i{item_id}.pkl'
        scaler_path = os.path.join(models_dir, scaler_filename) # Reuse scaler if exists, else create

        # Model & Training Params
        model_params = config.get('models', {}).get(model_type, {}).get('params', {})
        seq_len = model_params.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
        lstm_units = model_params.get('lstm_units', DEFAULT_LSTM_UNITS)
        dropout = model_params.get('dropout', DEFAULT_LSTM_DROPOUT)
        rec_dropout = model_params.get('recurrent_dropout', DEFAULT_LSTM_RECURRENT_DROPOUT)

    except KeyError as e:
        logging.error(f"Error: Missing key in configuration file: {e}.")
        exit() # Exit here before setting up logging if config fails

    # --- Setup Logging ---
    log_prefix = f"{experiment_name}_{model_type}_s{store_id}_i{item_id}"
    log_filepath = setup_logging('logs', log_prefix)
    logging.info(f"--- Starting LSTM Model Training: {log_prefix} ---")
    logging.info(f"Config File: {config_path}")
    # Log key parameters
    logging.info(f"Sequence Length: {seq_len}, LSTM Units: {lstm_units}, Dropout: {dropout}, Recurrent Dropout: {rec_dropout}")

    try:
        # 1. Load Data
        train_featured, validation_featured = load_processed_data(processed_dir, store_id, item_id)
        if train_featured is None: raise ValueError("Failed to load data.")

        # 2. Scale Features
        numerical_features = [col for col in train_featured.columns if col not in ['date'] and train_featured[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        logging.info(f"Identified {len(numerical_features)} numerical features for scaling: {numerical_features}")
        target_in_num_features = target_col if target_col in numerical_features else None # Track if target is scaled
        if target_in_num_features is None:
            logging.warning(f"Target '{target_col}' not in numerical features for scaling.")
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
        model = build_lstm_model(input_shape, lstm_units, dropout, rec_dropout)
        model.summary(print_fn=logging.info)

        # 5. Train Model (reuse generic trainer)
        history, best_model_path = train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, config, model_type)
        if history is None: raise ValueError("Model training failed.")

        # 6. Predict and Evaluate (reuse generic evaluator)
        y_pred, y_true, metrics = predict_evaluate_base(
            best_model_path, scaler, scaler_path, X_val_seq, y_val_seq_orig,
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
        # Need the shared save_predictions function
        from src.evaluation.metrics import save_predictions # Assuming it's moved here
        save_predictions(validation_dates.to_frame(name='ds'), y_pred, y_true, pred_path)


        # 9. Plot Results
        plot_forecast(
            dates=validation_dates, y_true=y_true, y_pred=y_pred,
            model_name=model_type.capitalize(), store_id=store_id, item_id=item_id,
            output_dir=figures_dir
        )

        logging.info(f"--- LSTM Model Training Pipeline Finished Successfully ---")

    except Exception as e:
        logging.exception(f"An critical error occurred in the main LSTM pipeline for Store={store_id}, Item={item_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM model using a configuration file.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file.')
    args = parser.parse_args()
    main(config_path=args.config)