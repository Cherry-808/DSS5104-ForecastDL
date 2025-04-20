# src/data/data_loaders.py
import pandas as pd
import os
import logging

def load_processed_data(processed_dir: str, store_id: int, item_id: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Loads preprocessed train and validation data from Parquet files.

    Args:
        processed_dir (str): Directory containing the processed Parquet files.
        store_id (int): Store ID.
        item_id (int): Item ID.

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]: Tuple containing training dataframe
                                                        and validation dataframe, or (None, None) on error.
    """
    train_filename = f'train_featured_s{store_id}_i{item_id}.parquet'
    validation_filename = f'validation_featured_s{store_id}_i{item_id}.parquet'
    train_path = os.path.join(processed_dir, train_filename)
    val_path = os.path.join(processed_dir, validation_filename)

    logging.info(f"Attempting to load data from {train_path} and {val_path}...")
    train_df, val_df = None, None
    try:
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        # Ensure date column is datetime
        train_df['date'] = pd.to_datetime(train_df['date'])
        val_df['date'] = pd.to_datetime(val_df['date'])
        logging.info(f"Data loaded successfully: Train shape={train_df.shape}, Validation shape={val_df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Input files not found. Looked for {train_path} and {val_path}. "
                      "Ensure data_processing script was run for store={store_id}, item={item_id}.")
    except ImportError:
        logging.error("Error: 'pyarrow' library not found. Cannot read Parquet files. Install with 'pip install pyarrow'.")
    except Exception as e:
        logging.exception(f"An error occurred loading processed data: {e}")

    return train_df, val_df