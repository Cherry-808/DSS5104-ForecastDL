# run_experiment.py (in project root)
import subprocess
import yaml
import argparse
import os
import logging
from src.utils.logging_setup import setup_logging # Import from src
from src.utils.config_loader import load_config   # Import from src

# --- Setup Logging for the orchestrator itself ---
LOG_DIR_RUNNER = 'logs/runner' # Separate log dir for the runner
log_filepath_runner = setup_logging(LOG_DIR_RUNNER, 'run_experiment')

# --- 修改后的 run_training_script 函数 ---
def run_training_script(model_type: str, dataset_name: str, config_path: str, src_dir: str = 'src'):
    """
    根据模型类型和数据集名称，构造命令并作为模块运行特定的训练脚本。
    (Constructs command and runs a specific training script AS A MODULE based on model type and dataset name.)
    """
    logging.info(f"Attempting to run training for model: {model_type} on dataset: {dataset_name}")

    # 1. 动态确定模块名和脚本文件名 (Dynamically determine module and script names)
    #    假设命名约定为 train_{模型类型}_{数据集名称}.py (Assumes naming convention train_{model_type}_{dataset_name}.py)
    module_base_name = f'train_{model_type.lower()}_{dataset_name.lower()}'
    script_filename = f'{module_base_name}.py'

    # 2. 构建 Python 模块路径 (点分隔，无 .py) (Construct Python module path)
    module_path = f'src.training.{module_base_name}'

    # 3. (可选但推荐) 检查对应的 .py 文件是否存在 (Check if the corresponding .py file exists)
    script_filepath = os.path.join(src_dir, 'training', script_filename)
    if not os.path.exists(script_filepath):
        logging.error(f"Training script file not found at {script_filepath}. "
                      f"Cannot run module {module_path}. Skipping model {model_type} for dataset {dataset_name}.")
        # 尝试查找通用的训练脚本，例如 train_{model_type}.py
        # (Attempt to find a generic training script, e.g., train_{model_type}.py)
        generic_module_base_name = f'train_{model_type.lower()}'
        generic_script_filename = f'{generic_module_base_name}.py'
        generic_module_path = f'src.training.{generic_module_base_name}'
        generic_script_filepath = os.path.join(src_dir, 'training', generic_script_filename)

        if os.path.exists(generic_script_filepath):
             logging.warning(f"Specific script {script_filename} not found, attempting to use generic script {generic_script_filename}...")
             module_path = generic_module_path # 使用通用模块路径 (Use generic module path)
             script_filepath = generic_script_filepath # 更新脚本路径用于后续检查 (Update script path for checks)
        else:
             logging.error(f"Neither specific script {script_filename} nor generic script {generic_script_filename} found. Skipping.")
             return False # Indicate failure


    # 4. 构建命令，使用 -m 和 模块路径 (Construct command using -m and module path)
    cmd = ['python', '-m', module_path, '--config', config_path]
    logging.info(f"Executing command: {' '.join(cmd)}")

    try:
        # 使用 subprocess.run 执行子脚本 (Execute the sub-script using subprocess.run)
        # 注意：如果子脚本有大量输出，可能需要调整 buffer 或使用 Popen
        # (Note: If the sub-script has large output, might need buffer adjustments or use Popen)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info(f"--- Start Output for {model_type} on {dataset_name} ({module_path}) ---")
        # 打印标准输出，避免日志过长，可以只打印部分或在出错时打印
        # (Print stdout, maybe only parts or on error to avoid long logs)
        # logging.info(result.stdout)
        stdout_lines = result.stdout.splitlines()
        num_lines_to_log = 50 # Example: log first/last lines or sample
        if len(stdout_lines) > num_lines_to_log * 2:
            logging.info("...(stdout truncated)...")
            for line in stdout_lines[:num_lines_to_log]: logging.info(line)
            logging.info("...")
            for line in stdout_lines[-num_lines_to_log:]: logging.info(line)
        else:
             logging.info(result.stdout)


        if result.stderr:
             logging.warning(f"--- Stderr for {model_type} on {dataset_name} ({module_path}) ---")
             logging.warning(result.stderr) # Standard error usually indicates warnings or non-fatal errors
        logging.info(f">>> Finished training run call for model: {model_type} on {dataset_name} <<<")
        return True # Indicate success

    except subprocess.CalledProcessError as e:
        logging.error(f"!!! Error running training script via module {module_path} !!!")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"--- Stdout ---:\n{e.stdout}") # Print stdout on error
        logging.error(f"--- Stderr ---:\n{e.stderr}") # Print stderr on error
        return False # Indicate failure
    except FileNotFoundError:
         # This error shouldn't happen if 'python' is in PATH
         logging.error(f"Error: 'python' command not found or script path resolution incorrect.")
         return False
    except Exception as e:
        # Catch other potential exceptions during subprocess execution
        logging.exception(f"An unexpected error occurred while trying to run module {module_path} for model {model_type}")
        return False


# --- 修改后的 main 函数 ---
def main(config_path: str):
    """Loads config and orchestrates model training runs for the specified dataset."""
    logging.info(f"--- Starting Experiment Runner ---")
    logging.info(f"Loading configuration from: {config_path}")

    config = load_config(config_path)
    if config is None:
        logging.error("Failed to load configuration. Exiting.")
        return

    experiment_name = config.get('experiment_name', 'default_experiment')

    # --- 获取数据集名称 (Get dataset name) ---
    dataset_config = config.get('dataset')
    if not dataset_config or 'name' not in dataset_config:
        logging.error("Configuration is missing 'dataset' section or 'dataset.name' key. Exiting.")
        return
    dataset_name = dataset_config['name']
    logging.info(f"Target Dataset: {dataset_name}") # Log the target dataset

    models_to_run_config = config.get('models', {}) # Expecting dict now { 'transformer': {...}, 'xgboost': {...} }
    if not isinstance(models_to_run_config, dict):
        logging.error("Configuration 'models' section should be a dictionary where keys are model types (e.g., 'transformer'). Exiting.")
        return
    models_to_run = list(models_to_run_config.keys()) # Get model names (keys) from the config dict

    logging.info(f"Experiment Name: {experiment_name}")
    logging.info(f"Models to run based on config keys: {models_to_run}")

    if not models_to_run:
        logging.warning("No models specified under the 'models' key in the configuration file.")
        return

    # --- 为配置中指定的每个模型运行训练 (Run training for each specified model) ---
    successful_models = []
    failed_models = []
    for model_type in models_to_run:
        # 将数据集名称传递给训练脚本运行函数
        # (Pass the dataset name to the training script runner function)
        success = run_training_script(model_type, dataset_name, config_path)
        if success:
            successful_models.append(f"{model_type}_on_{dataset_name}")
        else:
            failed_models.append(f"{model_type}_on_{dataset_name}")
            # 决定是否在第一个失败时停止整个实验
            # (Decide if you want to stop the whole experiment on first failure)
            # logging.warning("Stopping experiment due to failure in model training.")
            # break

    # --- 实验总结 (Experiment Summary) ---
    logging.info(f"\n--- Experiment {experiment_name} (Dataset: {dataset_name}) Finished ---")
    logging.info(f"Successfully completed run calls for models: {successful_models}")
    if failed_models:
        logging.warning(f"Failed run calls for models: {failed_models}")
    else:
        logging.info("All specified models attempted without critical errors reported by runner.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ML experiments based on a config file.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the experiment configuration YAML file (e.g., outputs/configs/experiment1.yaml)')
    args = parser.parse_args()

    main(config_path=args.config)