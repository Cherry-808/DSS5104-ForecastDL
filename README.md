# Retail Demand Forecasting: Model Comparison

## Description

This project aims to compare the performance, scalability, robustness, and practicality of various classical and deep learning time series forecasting models. The primary focus is on predicting retail demand using the "Store Item Demand Forecasting Challenge" dataset from Kaggle.

The comparison evaluates models based on standard regression metrics, including:
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)
* R-squared (R²)
* Adjusted R-squared (Adj R²)

## Project Structure
```
The project follows a modular structure:
comparison_project/
│
├── data/                       # Datasets
│   └── dataset_retail/         # Specific dataset directory
│       ├── raw/                # Original downloaded data (e.g., train.csv - likely gitignored)
│       └── processed/          # Processed data files (e.g., *.parquet - likely gitignored)
│
├── models/                     # Saved model files & scalers (likely gitignored)
│
├── src/                        # Source code
│   ├── data/                   # Data loading and preprocessing scripts/modules
│   │   └── data_loaders.py
│   ├── models/                 # Model architecture definitions
│   │   ├── prophet_wrapper.py  # (Example if you wrap it)
│   │   └── transformer.py      # (Example Keras/TF Transformer definition)
│   ├── training/               # Model training scripts
│   │   ├── train_prophet_retail.py
│   │   └── train_transformer_retail.py
│   │   └── ...                 # Add scripts for other models
│   ├── evaluation/             # Evaluation metrics and saving logic
│   │   ├── metrics.py
│   │   └── saving.py           # (Example for saving predictions)
│   ├── visualization/          # Plotting functions
│   │   └── plot_results.py
│   └── utils/                  # Shared utility functions
│       ├── config_loader.py
│       └── logging_setup.py
│
├── outputs/                    # Experiment outputs
│   ├── configs/                # Experiment configuration files (YAML)
│   │   └── retail_experiment_1.yaml # Example config
│   └── results/                # Experiment results (metrics, figures, predictions - likely gitignored)
│       └── retail/             # Results specific to the retail dataset
│           └── ...             # Subdirs for experiments, models, etc.
│
├── logs/                       # Log files for runs (likely gitignored)
│   └── runner/                 # Logs for the main runner script
│
├── notebooks/                  # Jupyter notebooks for exploration, analysis, visualization
│
├── main.py                     # (Optional) Main entry point if needed
├── run_experiment.py           # Script to orchestrate experiment runs
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Setup / Installation

1.  **Prerequisites:**
    * Python 3.9 or higher recommended.
    * `pip` and `venv` (or Conda).

2.  **Clone the Repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd comparison_project
    ```

3.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # You might need specific versions or builds depending on your OS/hardware
    # e.g., for TensorFlow GPU support on Linux/Windows or Metal support on macOS
    # Ensure 'pyarrow' is installed for Parquet support
    ```

## Dataset

* **Source:** [Kaggle Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/store-item-demand-forecasting-challenge/data)
* **Setup:**
    1.  Download `train.csv` and `test.csv` from the Kaggle competition page.
    2.  Create the directory structure `data/dataset_retail/raw/`.
    3.  Place the downloaded `train.csv` and `test.csv` files into the `data/dataset_retail/raw/` directory.
* **Note:** The `data/` directory (especially `raw/` and `processed/`) is typically included in `.gitignore` due to potential large file sizes. Users are expected to download the raw data themselves.

## Usage / Workflow

The project uses configuration files to define and run experiments comparing different models on specific data subsets.

1.  **Data Processing:**
    * Run the data processing script to generate features and split data for a specific store-item combination. This creates `.parquet` files in the `processed` data directory.
    * **Command:**
        ```bash
        python src/data/data_processing.py --store <store_id> --item <item_id> --cutoff <validation_cutoff_date> --input data/dataset_retail/raw/train.csv
        ```
    * *(Note: The script provided earlier hardcoded parameters. If using that version, ensure the hardcoded values are correct before running `python src/data/data_processing.py`)*

2.  **Configuration:**
    * Experiment parameters are defined in YAML files located in `outputs/configs/`.
    * See `outputs/configs/retail_experiment_1.yaml` for an example structure.
    * Key sections in the config file:
        * `experiment_name`: A unique name for the experiment run.
        * `dataset`: Specifies dataset paths, store/item IDs, target column, validation cutoff, etc.
        * `models`: A dictionary where keys are model names (e.g., `prophet`, `transformer`) and values contain model-specific configurations, including hyperparameters under a `params:` key.
        * `output_base_dir`, `models_dir`: Define base paths for saving results and models.
        * `metrics_to_calculate`: List of metrics to compute.

3.  **Running Experiments:**
    * Use the `run_experiment.py` script to execute the training and evaluation pipeline based on a configuration file.
    * **Command:**
        ```bash
        python run_experiment.py --config outputs/configs/your_experiment_config.yaml
        ```
    * This script reads the config, identifies the models to run, and calls the corresponding `src/training/train_MODEL_retail.py` script for each model, passing the configuration path.

4.  **Outputs:**
    * **Logs:** Detailed logs for the main runner and each model training run are saved in the `logs/` directory (organized by runner vs. model runs).
    * **Processed Data:** Feature-engineered training/validation sets are saved as Parquet files in `data/dataset_retail/processed/`.
    * **Saved Models:** Trained model files (e.g., `.pkl`, `.keras`) and scalers are saved in the `models/` directory (potentially within subdirectories defined in training scripts).
    * **Results:** Experiment results (metrics, predictions, figures) are saved under the `output_base_dir` specified in the config, typically structured like: `outputs/results/retail/<experiment_name>/<model_name>/s<store_id>_i<item_id>/`.

## Implemented Models

*(As of YYYY-MM-DD - Update this)*

* **Prophet:** Using the `prophet` library.
* **Transformer:** Implemented using Keras/TensorFlow.
* *(Planned/In Progress): XGBoost, LSTM, TCN, ARIMA*

## Contributing

*(Optional: Add guidelines if applicable)*

## License

*(Optional: Specify project license, e.g., MIT, Apache 2.0)*
Before finalizing:

Replace <your-repo-url> if you clone from Git.
Fill in the date in the "Implemented Models" section.
Review the paths specified (e.g., data/dataset_retail/raw/) and make sure they exactly match your project's structure.
Add any other specific setup notes relevant to your environment or chosen libraries.