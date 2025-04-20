# run_experiment.py (in project root)
import subprocess
import yaml
import argparse
import os
import logging
from src.utils.logging_setup import setup_logging # Import from src
from src.utils.config_loader import load_config   # Import from src

# --- Setup Logging for the orchestrator itself ---
# Note: Logging setup within subprocesses (train_*.py) will be separate
LOG_DIR_RUNNER = 'logs/runner' # Separate log dir for the runner
log_filepath_runner = setup_logging(LOG_DIR_RUNNER, 'run_experiment')

def run_training_script(model_type: str, config_path: str, src_dir: str = 'src'):
    """Constructs command and runs a specific training script AS A MODULE."""
    logging.info(f"Attempting to run training for model: {model_type}")

    # 1. 确定模块名和脚本文件名
    module_base_name = f'train_{model_type.lower()}_retail'
    script_filename = f'{module_base_name}.py'
    # 2. 构建 Python 模块路径 (用点分隔，无 .py)
    module_path = f'src.training.{module_base_name}'

    # (可选但推荐) 检查对应的 .py 文件是否存在，以防模块路径写错
    script_filepath = os.path.join(src_dir, 'training', script_filename)
    if not os.path.exists(script_filepath):
        logging.error(f"Training script file not found at {script_filepath}. Cannot run module {module_path}. Skipping model {model_type}.")
        return False # Indicate failure

    # 3. 构建命令，使用 -m 和 模块路径
    cmd = ['python', '-m', module_path, '--config', config_path] # <--- 修改在这里
    logging.info(f"Executing command: {' '.join(cmd)}")
    
    try:
        # 使用 subprocess.run 执行子脚本
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info(f"--- Start Output for {model_type} ---")
        logging.info(result.stdout)
        if result.stderr:
             logging.warning(f"--- Stderr for {model_type} ---")
             logging.warning(result.stderr)
        logging.info(f">>> Finished training successfully for model: {model_type} <<<")
        return True # Indicate success

    except subprocess.CalledProcessError as e:
        logging.error(f"!!! Error running {model_type} training script ({module_path}) !!!")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"Output (stdout):\n{e.stdout}")
        logging.error(f"Error Output (stderr):\n{e.stderr}")
        return False # Indicate failure
    except FileNotFoundError:
         # 这个错误理论上不应该发生，因为 python 命令本身应该在路径中
         logging.error(f"Error: 'python' command not found or script path resolution incorrect.")
         return False
    except Exception as e:
        logging.exception(f"An unexpected error occurred while trying to run module {module_path} for model {model_type}")
        return False



def main(config_path: str):
    """Loads config and orchestrates model training runs."""
    logging.info(f"--- Starting Experiment Runner ---")
    logging.info(f"Loading configuration from: {config_path}")

    config = load_config(config_path)
    if config is None:
        logging.error("Failed to load configuration. Exiting.")
        return

    experiment_name = config.get('experiment_name', 'default_experiment')
    models_to_run = config.get('models', []) # List of model names (e.g., ['prophet', 'transformer'])

    logging.info(f"Experiment Name: {experiment_name}")
    logging.info(f"Models to run based on config: {models_to_run}")

    if not models_to_run:
        logging.warning("No models specified in the configuration file.")
        return

    # --- Run training for each specified model ---
    successful_models = []
    failed_models = []
    for model_type in models_to_run:
        success = run_training_script(model_type, config_path)
        if success:
            successful_models.append(model_type)
        else:
            failed_models.append(model_type)
            # Decide if you want to stop the whole experiment on first failure
            # break

    # --- Experiment Summary ---
    logging.info(f"\n--- Experiment {experiment_name} Finished ---")
    logging.info(f"Successfully trained models: {successful_models}")
    if failed_models:
        logging.warning(f"Failed to train models: {failed_models}")
    else:
        logging.info("All specified models trained without critical errors reported by runner.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ML experiments based on a config file.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file (e.g., outputs/configs/experiment1.yaml)')
    args = parser.parse_args()

    main(config_path=args.config)