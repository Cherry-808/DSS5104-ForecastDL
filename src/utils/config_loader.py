# src/utils/config_loader.py
import yaml
import logging

def load_config(config_path: str) -> dict | None:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict | None: Loaded configuration dictionary, or None if an error occurs.
    """
    logging.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Error: Configuration file not found at {config_path}")
        return None
    except yaml.YAMLError as e:
        logging.exception(f"Error parsing YAML file {config_path}: {e}")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred loading config {config_path}: {e}")
        return None