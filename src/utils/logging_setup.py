# src/utils/logging_setup.py
import logging
import os
import datetime

def setup_logging(log_dir: str, log_prefix: str, level=logging.INFO) -> str:
    """
    Configures logging to file and console.

    Args:
        log_dir (str): Directory to save log files.
        log_prefix (str): Prefix for the log file name (e.g., 'data_processing', 'train_prophet_s1_i1').
        level (int): Minimum logging level (default: logging.INFO).

    Returns:
        str: The full path to the created log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'{log_prefix}_{current_time}.log'
    log_filepath = os.path.join(log_dir, log_filename)

    # Remove existing handlers from the root logger to avoid duplicate logs
    # if multiple calls to setup_logging happen in the same process (less likely when using subprocess)
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', # Added filename/lineno
        filename=log_filepath,
        filemode='w' # Overwrite for each distinct run
    )

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Logging setup complete. Log file: {log_filepath}")
    return log_filepath