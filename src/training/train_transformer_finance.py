# src/training/train_transformer_finance.py
"""
Purpose:
Train, evaluate, and save a Transformer model for the finance dataset,
based on parameters specified in a configuration file.
Loads processed finance data (e.g., process_finance_data.csv).

How to run (usually called by run_experiment.py):
python -m src.training.train_transformer_finance --config outputs/configs/your_finance_config.yaml
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error # calculate_metrics handles these
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import logging
import datetime
import sys # To potentially add project root to path if needed

# --- Import shared utility functions ---
# Add project root to sys.path to allow absolute imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.utils.logging_setup import setup_logging
    from src.utils.config_loader import load_config
    # !!! IMPORTANT: Adapt data loading based on your data_loaders.py !!!
    # Option 1: Assume a generic loader or a specific finance loader exists
    # from src.data.data_loaders import load_finance_processed_data, split_data_by_date
    # Option 2: Keep the retail loader name IF it was made generic (unlikely based on errors)
    # from src.data.data_loaders import load_processed_data # This expects store/item IDs
    from src.evaluation.metrics import calculate_metrics, save_metrics, save_predictions
    from src.visualization.plot_results import plot_forecast
except ModuleNotFoundError as e:
    print(f"Error importing utility modules: {e}")
    print(f"Ensure you are running from the project root directory ('{os.path.basename(project_root)}') "
          f"and all necessary __init__.py files exist.")
    sys.exit(1)

# --- Default Configuration (can be overridden by YAML) ---
# (Defaults remain the same as retail example, adjust if needed)
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

# --- Feature Scaling (Function remains the same) ---
def scale_features(train_df, val_df, numerical_features, scaler_path):
    """Scales numerical features using StandardScaler and saves the scaler."""
    logging.info("Scaling numerical features...")
    scaler = StandardScaler()
    try:
        # Fit ONLY on training data
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
    except Exception as e:
        logging.exception("Error during feature scaling or scaler saving.")
        return None, None, None

# --- Sequence Creation (Function remains the same) ---
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
    target_data = data[target_col].values # Use scaled target for training y

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

# --- Transformer Model Definition (Functions remain the same) ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Single Transformer block."""
    x = layers.LayerNormalization(epsilon=1e-6)(inputs) # Pre-Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs # Add & Norm (Residual 1)
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

    outputs = layers.Dense(1, activation="linear")(x) # Final output layer
    model = keras.Model(inputs, outputs)
    logging.info("Transformer model built.")
    return model

# --- Model Training (Adapted paths) ---
def train_model(model, X_train, y_train, X_val, y_val, config, dataset_name):
    """Compiles and trains the Keras model with callbacks."""
    logging.info("Compiling and training the model...")
    model_type = 'transformer'
    try:
        # Extract relevant training params from config
        params = config.get('models', {}).get(model_type, {}).get('params', {})
        lr = params.get('learning_rate', DEFAULT_LR)
        epochs = params.get('epochs', DEFAULT_EPOCHS)
        batch_size = params.get('batch_size', DEFAULT_BATCH_SIZE)
        patience = params.get('early_stopping_patience', DEFAULT_PATIENCE)

        # --- Define output paths using dataset_name ---
        experiment_name = config['experiment_name']
        output_dir = os.path.join(config['output_base_dir'], experiment_name, model_type, dataset_name) # Use dataset_name
        models_dir = os.path.join(output_dir, "saved_model")
        # Update filename convention
        model_checkpoint_filename = f'{model_type}_best_{dataset_name}.keras'
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
        return history, model_checkpoint_path # Return path to best saved model

    except Exception as e:
        logging.exception("An error occurred during model compilation or training.")
        return None, None

# --- Prediction & Evaluation (Function mostly the same, call needs adjustment) ---
def predict_and_evaluate(model_path, scaler, X_val_seq, y_val_seq_orig, numerical_features, target_col_name, use_log_transform=False):
    """Loads best model, predicts, inverse transforms, and evaluates."""
    logging.info(f"Loading best model from {model_path} for evaluation...")
    if not os.path.exists(model_path):
         logging.error(f"Model file not found at {model_path}. Cannot evaluate.")
         return None, None, None # Return None for metrics too
    if scaler is None:
        logging.error("Scaler object is None. Cannot perform inverse transform.")
        return None, None, None # Return None for metrics too

    try:
        # Load the best model saved by ModelCheckpoint
        model = keras.models.load_model(model_path)

        # Predict (output is scaled if target was scaled)
        logging.info("Making predictions on validation sequences...")
        y_pred_scaled = model.predict(X_val_seq).flatten()

        # Inverse Transform Predictions
        logging.info("Inverse transforming predictions...")
        y_pred = None
        # Check if the target column itself was scaled
        if target_col_name is not None and target_col_name in numerical_features:
            try:
                target_index = numerical_features.index(target_col_name)
                # Create a dummy array matching the scaler's expected input shape
                dummy_features = np.zeros((len(y_pred_scaled), len(numerical_features)))
                dummy_features[:, target_index] = y_pred_scaled
                # Perform inverse transform
                y_pred_inversed = scaler.inverse_transform(dummy_features)
                # Extract the target column's inverse-scaled values
                y_pred = y_pred_inversed[:, target_index]
                logging.info("Inverse transform using scaler complete.")
            except ValueError:
                logging.error(f"Target column '{target_col_name}' stated as numerical but not found in scaler's fitted feature list. Cannot inverse scale.")
                y_pred = y_pred_scaled # Keep scaled value, handle log transform next
            except Exception as e:
                 logging.exception("Error during scaler inverse_transform.")
                 return None, None, None
        else:
            # Target was not scaled with the main features
            logging.info(f"Target column '{target_col_name}' was not in the numerical features list for scaling or not provided. Assuming predictions are in the potentially log-transformed scale.")
            y_pred = y_pred_scaled # Keep the value as is (but might be log-transformed)

        # Inverse Log Transform (if applied)
        if use_log_transform:
            logging.info("Applying inverse log transform (expm1)...")
            y_pred = np.expm1(y_pred)
            # We assume y_val_seq_orig ALREADY holds the original, non-log values.
            # If y_val_seq_orig itself was log(target), it should be expm1'd too *before* evaluation.
            # This depends heavily on how y_val_seq_orig was created.

        # Ensure non-negative predictions (common for sales, prices)
        y_pred = np.maximum(0, y_pred)
        y_true = y_val_seq_orig # Use the original, unscaled validation targets passed in

        logging.info("Prediction processing complete.")

        # Evaluate using shared function
        # Determine number of predictors 'p' for Adj R2.
        # This is tricky. Using number of input features to the model.
        num_predictors = X_val_seq.shape[2] if X_val_seq.ndim == 3 else 1
        metrics = calculate_metrics(y_true, y_pred, num_predictors=num_predictors) # Pass p for Adj R2
        if metrics:
             log_msg = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if not np.isnan(v)]) # Handle potential NaNs
             logging.info(f"Validation Metrics: {log_msg}")
        else:
             logging.error("Failed to calculate metrics.")

        return y_pred, y_true, metrics

    except Exception as e:
        logging.exception("An error occurred during prediction or evaluation.")
        return None, None, None

# --- Main Function (Adapted for Finance Data) ---
def main(config_path: str):
    """Main execution pipeline for Transformer model training for finance data."""

    # --- Load Configuration ---
    config = load_config(config_path)
    if config is None: exit()

    # --- Extract Basic Info & Setup Logging ---
    try:
        experiment_name = config['experiment_name']
        dataset_config = config['dataset']
        dataset_name = dataset_config['name'] # Use dataset name
        processed_dir = dataset_config['processed_dir']
        output_base_dir = config['output_base_dir']
        target_col = dataset_config['target_col']
        validation_cutoff_date = dataset_config['validation_cutoff_date']
        date_col = 'Date' # Assuming 'Date' from inspection, make configurable if needed
        use_log_transform = dataset_config.get('use_log_transform', False)
        model_type = 'transformer' # Hardcoded for this script

        # Setup Logging - use dataset name in prefix
        log_prefix = f"{experiment_name}_{model_type}_{dataset_name}"
        log_filepath = setup_logging('logs', log_prefix)

        logging.info(f"--- Starting Transformer Model Training: {log_prefix} ---")
        logging.info(f"Using Config File: {config_path}")

        # --- Define Output Paths (using dataset_name) ---
        output_dir = os.path.join(output_base_dir, experiment_name, model_type, dataset_name)
        models_dir = os.path.join(output_dir, "saved_model")
        predictions_dir = os.path.join(output_dir, "predictions")
        metrics_dir = os.path.join(output_dir, "metrics")
        figures_dir = os.path.join(output_dir, "figures")
        scaler_filename = f'scaler_{dataset_name}.pkl' # Use dataset name in scaler filename
        scaler_path = os.path.join(models_dir, scaler_filename) # Save scaler with model
        model_checkpoint_filename = f'{model_type}_best_{dataset_name}.keras'
        model_checkpoint_path = os.path.join(models_dir, model_checkpoint_filename)

        # Create directories if they don't exist
        for dir_path in [models_dir, predictions_dir, metrics_dir, figures_dir]:
             os.makedirs(dir_path, exist_ok=True)

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
        # --- 1. Load Data ---
        # !!! Simplification: Load directly and split here. Should be in data_loaders.py ideally. !!!
        logging.info(f"Loading data for dataset '{dataset_name}' from {processed_dir}")
        # Assume the relevant file is named consistently or identifiable
        data_file_path = os.path.join(processed_dir, 'process_finance_data.csv') # Adjust filename if needed
        if not os.path.exists(data_file_path):
             raise FileNotFoundError(f"Processed data file not found at {data_file_path}")
        df_full = pd.read_csv(data_file_path)
        logging.info(f"Loaded data shape: {df_full.shape}")

        # Convert date column and sort
        if date_col not in df_full.columns:
             raise ValueError(f"Date column '{date_col}' not found in loaded data.")
        df_full[date_col] = pd.to_datetime(df_full[date_col])
        df_full = df_full.sort_values(by=date_col).reset_index(drop=True)

        # Apply Log Transform if specified (before splitting for consistency, usually done after split but ok if careful)
        if use_log_transform:
            logging.info(f"Applying log1p transform to target column: {target_col}")
            # Check if target column exists
            if target_col not in df_full.columns:
                 raise ValueError(f"Target column '{target_col}' for log transform not found in DataFrame.")
            # Apply log1p (log(1+x)) to handle potential zeros
            df_full[target_col] = np.log1p(df_full[target_col])


        # Split data based on cutoff date
        logging.info(f"Splitting data using cutoff date: {validation_cutoff_date}")
        cutoff_dt = pd.to_datetime(validation_cutoff_date)
        train_df = df_full[df_full[date_col] <= cutoff_dt].copy()
        validation_df = df_full[df_full[date_col] > cutoff_dt].copy()

        if train_df.empty or validation_df.empty:
             raise ValueError(f"Data split resulted in empty train or validation set. Check cutoff date '{validation_cutoff_date}' and data range.")
        logging.info(f"Train set shape: {train_df.shape}, Validation set shape: {validation_df.shape}")
        # --- End Simplified Data Loading & Splitting ---


        # 2. Scale Features
        #    Identify numerical columns dynamically from the loaded training data
        #    Exclude date column, potentially exclude target if it shouldn't be scaled with others
        numerical_features = [col for col in train_df.columns if col not in [date_col] and train_df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        logging.info(f"Identified {len(numerical_features)} numerical features for scaling: {numerical_features}")
        if target_col not in numerical_features:
             logging.warning(f"Target '{target_col}' not in numerical features list for scaling. Inverse transform might be simpler.")
        # Scale features using the training set distribution
        train_scaled, validation_scaled, scaler = scale_features(train_df, validation_df, numerical_features, scaler_path)
        if train_scaled is None: raise ValueError("Failed to scale features.")


        # 3. Create Sequences
        #    Define features for the Transformer input (usually all scaled numerical)
        #    Ensure target_col is NOT included here unless it's an engineered feature (like lagged target)
        feature_cols_for_dl = [col for col in numerical_features if col != target_col] # Exclude the target itself
        # If your engineered features include scaled lags of the target, ensure they are in `numerical_features`
        logging.info(f"Using {len(feature_cols_for_dl)} features for sequence input: {feature_cols_for_dl}")

        X_train_seq, y_train_seq = create_sequences(train_scaled, seq_len, feature_cols_for_dl, target_col) # y_train_seq uses scaled target
        X_val_seq, y_val_seq = create_sequences(validation_scaled, seq_len, feature_cols_for_dl, target_col) # y_val_seq uses scaled target

        # Get original (non-scaled, non-log) validation targets for final evaluation
        # We need the original df *before* scaling and log transform
        if use_log_transform:
             # If log was applied, y_val_seq_orig should be expm1'd from the validation_df
             y_val_seq_orig = np.expm1(validation_df[target_col].values[seq_len:])
        else:
             # If no log transform, just grab the original values
             y_val_seq_orig = validation_df[target_col].values[seq_len:]
        y_val_seq_orig = np.maximum(0, y_val_seq_orig) # Ensure non-negative true values

        if X_train_seq.shape[0] == 0 or X_val_seq.shape[0] == 0:
             raise ValueError(f"Not enough data after split to create sequences of length {seq_len} for training or validation.")


        # 4. Build Model
        if X_train_seq.ndim != 3 or X_train_seq.shape[2] == 0:
             raise ValueError(f"Input sequences have unexpected shape: {X_train_seq.shape}. Expected 3 dimensions with non-zero features.")
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2]) # (sequence_length, num_features)
        model = build_transformer_model(
            input_shape, head_size, num_heads, ff_dim, num_blocks,
            mlp_units, dropout, mlp_dropout
        )
        model.summary(print_fn=logging.info)


        # 5. Train Model
        #    Pass dataset_name for correct path construction inside train_model
        history, best_model_path = train_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, config, dataset_name)
        if history is None: raise ValueError("Model training failed.")


        # 6. Predict and Evaluate (using the best model saved by checkpoint)
        #    Determine if target needs inverse scaling based on whether it was in numerical_features
        target_in_num_features = target_col if target_col in numerical_features else None
        y_pred, y_true, metrics = predict_and_evaluate(
            best_model_path, scaler, X_val_seq, y_val_seq_orig,
            numerical_features, target_in_num_features, use_log_transform
        )
        if metrics is None: raise ValueError("Failed to evaluate predictions.")


        # 7. Save Metrics
        metrics_filename = f'{model_type}_metrics_{dataset_name}.json' # Use dataset name
        metrics_path = os.path.join(metrics_dir, metrics_filename)
        save_metrics(metrics, metrics_path)


        # 8. Save Predictions
        pred_filename = f'validation_{model_type}_{dataset_name}.csv' # Use dataset name
        pred_path = os.path.join(predictions_dir, pred_filename)
        # Get dates corresponding to validation predictions
        # Need the dates from the original validation_df, aligned with sequences
        validation_dates = validation_df[date_col].iloc[seq_len:].reset_index(drop=True) # Dates aligned with y_val_seq_orig
        # Ensure lengths match before saving
        if len(validation_dates) == len(y_pred) == len(y_true):
             save_predictions(validation_dates.to_frame(name='ds'), y_pred, y_true, pred_path)
        else:
             logging.error(f"Length mismatch when saving predictions: dates={len(validation_dates)}, pred={len(y_pred)}, true={len(y_true)}")


        # 9. Plot Results (remove store/item id)
        plot_filename = f'forecast_{model_type}_{dataset_name}.png'
        plot_path = os.path.join(figures_dir, plot_filename)
        plot_forecast(
            dates=validation_dates,
            y_true=y_true,
            y_pred=y_pred,
            model_name=f"{model_type.capitalize()} ({dataset_name})", # Add dataset to title
            output_filepath=plot_path # Pass full path to save figure
        )

        logging.info(f"--- Transformer Model Training Pipeline Finished Successfully for {dataset_name} ---")

    except FileNotFoundError as e:
         logging.error(f"Data file not found: {e}")
    except ValueError as e:
         logging.error(f"Data processing or validation error: {e}")
    except Exception as e:
        logging.exception(f"An critical error occurred in the main Transformer training pipeline for dataset={dataset_name}")

# --- Script Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Transformer model for finance data using a configuration file.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file.')
    args = parser.parse_args()

    main(config_path=args.config)