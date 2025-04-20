# src/training/train_transformer_retail.py
"""
Purpose:
Train, evaluate, and save a Transformer model based on parameters
specified in a configuration file. Loads preprocessed data.

How to run (usually called by run_experiment.py):
python -m src.training.train_transformer_retail --config outputs/configs/your_experiment_config.yaml
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
import sys # To potentially add project root to path if needed

# --- Import shared utility functions ---
# Add project root to sys.path to allow absolute imports from src
# Assumes this script is in src/training/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.utils.logging_setup import setup_logging
    from src.utils.config_loader import load_config
    from src.data.data_loaders import load_processed_data
    from src.evaluation.metrics import calculate_metrics, save_metrics, save_predictions
    from src.visualization.plot_results import plot_forecast
except ModuleNotFoundError as e:
    print(f"Error importing utility modules: {e}")
    print(f"Ensure you are running from the project root directory ('{os.path.basename(project_root)}') "
          f"and all necessary __init__.py files exist.")
    sys.exit(1)

# --- Default Configuration (can be overridden by YAML) ---
# These are fallback values if not specified in the config, but it's better
# to have them fully specified in the YAML.
DEFAULT_SEQUENCE_LENGTH = 60
DEFAULT_HEAD_SIZE = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_FF_DIM = 128
DEFAULT_NUM_BLOCKS = 2
DEFAULT_MLP_UNITS = 64
DEFAULT_DROPOUT = 0.1
DEFAULT_MLP_DROPOUT = 0.1
DEFAULT_LR = 0.001
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 10

# --- Feature Scaling ---
def scale_features(train_df, val_df, numerical_features, scaler_path):
    """Scales numerical features using StandardScaler and saves the scaler."""
    logging.info("Scaling numerical features...")
    scaler = StandardScaler()
    scaler_loaded = False
    try:
        # Check if scaler exists to load it (useful for inference later, maybe not needed here)
        # if os.path.exists(scaler_path):
        #     with open(scaler_path, 'rb') as f:
        #         scaler = pickle.load(f)
        #     logging.info(f"Loaded existing scaler from {scaler_path}")
        #     scaler_loaded = True
        # else:
        # Fit ONLY on training data if not loaded
        scaler.fit(train_df[numerical_features])
        logging.info("Scaler fitted on training data.")
        # Save the scaler
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"Scaler saved to {scaler_path}")

        # Transform data
        train_scaled = train_df.copy()
        val_scaled = val_df.copy()
        train_scaled[numerical_features] = scaler.transform(train_df[numerical_features])
        val_scaled[numerical_features] = scaler.transform(val_df[numerical_features])
        logging.info("Training and validation data scaled.")
        return train_scaled, val_scaled, scaler
    except FileNotFoundError:
         logging.error(f"Scaler file specified but not found at {scaler_path} during load attempt.")
         return None, None, None
    except Exception as e:
        logging.exception("Error during feature scaling or scaler saving/loading.")
        return None, None, None

# --- Sequence Creation ---
def create_sequences(data, sequence_length, feature_cols, target_col):
    """Creates sequences from DataFrame."""
    logging.info(f"Creating sequences with length {sequence_length}...")
    X, y = [], []

    # Ensure target_col is in the DataFrame for extracting targets
    if target_col not in data.columns:
        logging.error(f"Target column '{target_col}' not found in data for sequencing.")
        return np.array(X), np.array(y)
    # Ensure all feature_cols are present
    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        logging.error(f"Feature columns missing in data for sequencing: {missing_features}")
        return np.array(X), np.array(y)

    feature_data = data[feature_cols].values
    target_data = data[target_col].values

    if len(data) <= sequence_length:
        logging.warning(f"Data length ({len(data)}) is not sufficient for sequence length ({sequence_length}). Returning empty arrays.")
        return np.array(X), np.array(y)

    for i in range(len(data) - sequence_length):
        X.append(feature_data[i:(i + sequence_length)])
        y.append(target_data[i + sequence_length]) # Target is the value AFTER the sequence

    X = np.array(X)
    y = np.array(y)
    logging.info(f"Sequences created: X shape={X.shape}, y shape={y.shape}")
    return X, y

# --- Transformer Model Definition ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Single Transformer block."""
    # Attention and Normalization
    x = layers.LayerNormalization(epsilon=1e-6)(inputs) # Pre-Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs # Add & Norm (Residual 1)

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res) # Pre-Normalization
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    outputs = x + res # Add & Norm (Residual 2)
    return outputs

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout):
    """Builds the complete Transformer model."""
    logging.info("Building Transformer model...")
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    if isinstance(mlp_units, list): # Allow list of MLP units
         for units in mlp_units:
              x = layers.Dense(units, activation="relu")(x)
              x = layers.Dropout(mlp_dropout)(x)
    elif mlp_units > 0: # Allow single MLP layer
         x = layers.Dense(mlp_units, activation="relu")(x)
         x = layers.Dropout(mlp_dropout)(x)
    # else: No MLP head if mlp_units is 0 or not specified properly

    outputs = layers.Dense(1, activation="linear")(x) # Final output layer
    model = keras.Model(inputs, outputs)
    logging.info("Transformer model built.")
    return model

# --- Model Training ---
def train_model(model, X_train, y_train, X_val, y_val, config):
    """Compiles and trains the Keras model with callbacks."""
    logging.info("Compiling and training the model...")
    model_type = 'transformer' # Assuming this script is only for transformer
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
        os.makedirs(models_dir, exist_ok=True) # Ensure checkpoint dir exists

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
            verbose=2 # Show less output per epoch
        )
        logging.info("Model training complete.")
        # No need to explicitly return model if using restore_best_weights=True
        # The model object itself will have the best weights loaded after training
        return history, model_checkpoint_path # Return path to best saved model

    except Exception as e:
        logging.exception("An error occurred during model compilation or training.")
        return None, None

# --- Prediction & Evaluation ---
def predict_and_evaluate(model_path, scaler, X_val_seq, y_val_seq_orig, numerical_features, target_col_name, use_log_transform=False):
    """Loads best model, predicts, inverse transforms, and evaluates."""
    logging.info(f"Loading best model from {model_path} for evaluation...")
    if not os.path.exists(model_path):
         logging.error(f"Model file not found at {model_path}. Cannot evaluate.")
         return None, None
    if scaler is None:
        logging.error("Scaler object is None. Cannot perform inverse transform.")
        return None, None

    try:
        # Load the best model saved by ModelCheckpoint
        model = keras.models.load_model(model_path)

        # Predict (output is scaled)
        logging.info("Making predictions on validation sequences...")
        y_pred_scaled = model.predict(X_val_seq).flatten()

        # Inverse Transform Predictions
        logging.info("Inverse transforming predictions...")
        y_pred = None
        # Check if the target column itself was scaled
        if target_col_name in numerical_features:
            try:
                target_index = numerical_features.index(target_col_name)
                dummy_features = np.zeros((len(y_pred_scaled), len(numerical_features)))
                dummy_features[:, target_index] = y_pred_scaled
                y_pred_inversed = scaler.inverse_transform(dummy_features)
                y_pred = y_pred_inversed[:, target_index]
                logging.info("Inverse transform using scaler complete.")
            except ValueError:
                logging.error(f"Target column '{target_col_name}' stated as numerical but not found in scaler's feature list during fit. Cannot inverse scale.")
                # Fallback: Assume prediction is log-transformed only or already correct scale
                y_pred = y_pred_scaled # Keep scaled value, handle log transform next
            except Exception as e:
                 logging.exception("Error during scaler inverse_transform.")
                 return None, None
        else:
            # Target was not scaled with the main features
            logging.info(f"Target column '{target_col_name}' was not in the numerical features list for scaling. Assuming predictions are already in the potentially log-transformed scale.")
            y_pred = y_pred_scaled # Keep the value as is (but might be log-transformed)

        # Inverse Log Transform (if applied)
        if use_log_transform:
            logging.info("Applying inverse log transform (expm1)...")
            y_pred = np.expm1(y_pred)
            # Ensure y_true is also inverse transformed if it represents log(sales)
            # y_val_seq_orig should hold the *original* sales values before any scaling/log
            # If y_val_seq_orig was log-transformed, it needs expm1 here too.
            # Let's assume y_val_seq_orig is the final true value needed. Check data prep.

        # Ensure non-negative predictions
        y_pred = np.maximum(0, y_pred)
        y_true = y_val_seq_orig # Use the original, unscaled validation targets

        logging.info("Prediction processing complete.")

        # Evaluate using shared function
        metrics = calculate_metrics(y_true, y_pred)
        if metrics:
             log_msg = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
             logging.info(f"Validation Metrics: {log_msg}")
        else:
             logging.error("Failed to calculate metrics.")

        return y_pred, y_true, metrics

    except Exception as e:
        logging.exception("An error occurred during prediction or evaluation.")
        return None, None, None

# --- Main Function ---
def main(config_path: str):
    """Main execution pipeline for Transformer model training using config."""

    # --- Load Configuration ---
    config = load_config(config_path)
    if config is None: exit()

    # --- Extract Basic Info & Setup Logging ---
    try:
        experiment_name = config['experiment_name']
        dataset_config = config['dataset']
        store_id = dataset_config['store_id']
        item_id = dataset_config['item_id']
        processed_dir = dataset_config['processed_dir']
        output_base_dir = config['output_base_dir']
        target_col = dataset_config['target_col']
        use_log_transform = dataset_config.get('use_log_transform', False)
        model_type = 'transformer'

        # Setup Logging
        log_prefix = f"{experiment_name}_{model_type}_s{store_id}_i{item_id}"
        log_filepath = setup_logging('logs', log_prefix)

        logging.info(f"--- Starting Transformer Model Training: {log_prefix} ---")
        logging.info(f"Using Config File: {config_path}")

        # --- Define Output Paths ---
        output_dir = os.path.join(output_base_dir, experiment_name, model_type, f"s{store_id}_i{item_id}")
        models_dir = os.path.join(output_dir, "saved_model")
        predictions_dir = os.path.join(output_dir, "predictions")
        metrics_dir = os.path.join(output_dir, "metrics")
        figures_dir = os.path.join(output_dir, "figures")
        scaler_filename = f'scaler_s{store_id}_i{item_id}.pkl' # Reuse scaler name convention
        scaler_path = os.path.join(models_dir, scaler_filename) # Save scaler with model
        model_checkpoint_filename = f'{model_type}_best_s{store_id}_i{item_id}.keras'
        model_checkpoint_path = os.path.join(models_dir, model_checkpoint_filename)

        # Extract model parameters with defaults
        model_params = config.get('models', {}).get(model_type, {}).get('params', {})
        seq_len = model_params.get('sequence_length', DEFAULT_SEQUENCE_LENGTH)
        head_size = model_params.get('head_size', DEFAULT_HEAD_SIZE)
        num_heads = model_params.get('num_heads', DEFAULT_NUM_HEADS)
        ff_dim = model_params.get('ff_dim', DEFAULT_FF_DIM)
        num_blocks = model_params.get('num_transformer_blocks', DEFAULT_NUM_BLOCKS)
        mlp_units = model_params.get('mlp_units', DEFAULT_MLP_UNITS)
        dropout = model_params.get('dropout', DEFAULT_DROPOUT)
        mlp_dropout = model_params.get('mlp_dropout', DEFAULT_MLP_DROPOUT)

        logging.info(f"Target Column: {target_col}, Use Log Transform: {use_log_transform}")
        logging.info(f"Sequence Length: {seq_len}")
        logging.info(f"Model Params: HeadSize={head_size}, NumHeads={num_heads}, FF_Dim={ff_dim}, Blocks={num_blocks}")
        logging.info(f"MLP Units: {mlp_units}, Dropout: {dropout}, MLP Dropout: {mlp_dropout}")

    except KeyError as e:
        logging.error(f"Error: Missing key in configuration file: {e}. Check config structure.")
        exit()
    except Exception as e:
         logging.exception("Error during initial setup.")
         exit()

    try:
        # 1. Load Data
        train_featured, validation_featured = load_processed_data(processed_dir, store_id, item_id)
        if train_featured is None: raise ValueError("Failed to load data.")
        
        # --- 添加下面几行进行小规模测试 ---
        NUM_TRAIN_SAMPLES = 1000 # 示例值，确保 > SEQUENCE_LENGTH
        NUM_VAL_SAMPLES = 300   # 示例值，确保 > SEQUENCE_LENGTH
        logging.warning(f"---!!! SMALL SCALE TEST: Using only {NUM_TRAIN_SAMPLES} train and {NUM_VAL_SAMPLES} validation samples !!!---")
        train_featured = train_featured.head(NUM_TRAIN_SAMPLES)
        validation_featured = validation_featured.head(NUM_VAL_SAMPLES)
        # --- 测试代码结束 ---


        # 2. Scale Features
        #    Identify numerical columns (ensure consistency with data processing)
        numerical_features = [col for col in train_featured.columns if col not in ['date'] and train_featured[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        logging.info(f"Identified {len(numerical_features)} numerical features for scaling: {numerical_features}")
        if target_col not in numerical_features:
             logging.warning(f"Target '{target_col}' not in numerical features for scaling. Inverse transform might only involve expm1 if log transform was used.")
        train_scaled, validation_scaled, scaler = scale_features(train_featured, validation_featured, numerical_features, scaler_path)
        if train_scaled is None: raise ValueError("Failed to scale features.")

        # 3. Create Sequences
        #    Features for DL sequences (usually all scaled numerical + encoded categoricals if any)
        feature_cols_for_dl = [col for col in train_scaled.columns if col not in ['date', target_col]] # Exclude original target
        # Check if target column itself is needed as an input feature (e.g., lagged target was scaled)
        # This depends on how features were engineered. Assuming needed features are in feature_cols_for_dl.
        X_train_seq, y_train_seq = create_sequences(train_scaled, seq_len, feature_cols_for_dl, target_col)
        X_val_seq, y_val_seq = create_sequences(validation_scaled, seq_len, feature_cols_for_dl, target_col)
        # Get original (unscaled, but maybe log-transformed) validation targets for evaluation
        y_val_seq_orig = validation_featured[target_col].values[seq_len:]

        if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0:
             raise ValueError("Not enough data to create sequences for training or validation.")

        # 4. Build Model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = build_transformer_model(
            input_shape, head_size, num_heads, ff_dim, num_blocks,
            mlp_units, dropout, mlp_dropout
        )
        model.summary(print_fn=logging.info)

        # 5. Train Model
        history, best_model_path = train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, config)
        if history is None: raise ValueError("Model training failed.")

        # 6. Predict and Evaluate (using the best model saved by checkpoint)
        #   Determine if target needs inverse scaling based on whether it was in numerical_features
        target_in_num_features = target_col if target_col in numerical_features else None
        y_pred, y_true, metrics = predict_and_evaluate(
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
        # Get dates corresponding to validation predictions
        validation_dates = validation_featured['date'].iloc[seq_len:] # Dates aligned with y_val_seq_orig
        save_predictions(validation_dates.to_frame(name='ds'), y_pred, y_true, pred_path)

        # 9. Plot Results
        plot_forecast(
            dates=validation_dates,
            y_true=y_true,
            y_pred=y_pred,
            model_name=model_type.capitalize(),
            store_id=store_id,
            item_id=item_id,
            output_dir=figures_dir # Pass directory to save figure
        )

        logging.info(f"--- Transformer Model Training Pipeline Finished Successfully ---")

    except Exception as e:
        logging.exception(f"An critical error occurred in the main Transformer training pipeline for Store={store_id}, Item={item_id}")

# --- Script Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Transformer model using a configuration file.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file.')
    args = parser.parse_args()

    main(config_path=args.config)