# DSS5104-ForecastDL

## Project Overview
This project aims to compare the predictive performance of various deep learning models against baseline models across multiple datasets. Through systematic experimental design and evaluation methodologies, we can comprehensively analyze the strengths and limitations of different models.

## Project Structure
```
comparison_project/
│
├── data/                           # 存放所有数据集 / All datasets
│   ├── dataset_1/                  # 第一个数据集 / First dataset
│   │   ├── raw/                    # 原始数据 / Raw data
│   │   └── processed/              # 预处理后的数据 / Processed data
│   ├── dataset_2/                  # 第二个数据集 / Second dataset
│   │   ├── raw/
│   │   └── processed/
│   └── ...
│
├── models/                         # 存放模型文件 / Model files
│   ├── deep_model_1/               # 第一个深度学习模型 / First deep learning model
│   │   └── saved_checkpoints/      # 保存的检查点 / Saved checkpoints
│   ├── deep_model_2/               # 第二个深度学习模型 / Second deep learning model
│   │   └── saved_checkpoints/
│   ├── baseline_models/            # 基准模型 / Baseline models
│   │   └── saved_states/           # 保存的状态 / Saved states
│   └── ...
│
├── src/                            # 源代码 / Source code
│   ├── data/                       # 数据处理 / Data processing
│   │   ├── preprocessing.py        # 通用预处理函数 / Common preprocessing functions
│   │   └── data_loaders.py         # 数据加载器 / Data loaders
│   │
│   ├── models/                     # 模型定义 / Model definitions
│   │   ├── base_model.py           # 基础模型接口 / Base model interface
│   │   ├── deep_model_1.py         # 第一个深度学习模型实现 / First deep learning model implementation
│   │   ├── deep_model_2.py         # 第二个深度学习模型实现 / Second deep learning model implementation
│   │   └── baseline_models.py      # 基准模型实现 / Baseline models implementation
│   │
│   ├── training/                   # 训练相关 / Training related
│   │   ├── train.py                # 训练函数 / Training functions
│   │   └── hyperparameter_tuning.py # 超参数调优 / Hyperparameter tuning
│   │
│   ├── evaluation/                 # 评估相关 / Evaluation related
│   │   ├── metrics.py              # 评估指标 / Evaluation metrics
│   │   ├── compare_models.py       # 模型比较函数 / Model comparison functions
│   │   └── statistical_tests.py    # 统计显著性测试 / Statistical significance tests
│   │
│   └── visualization/              # 可视化 / Visualization
│       └── plot_results.py         # 结果可视化 / Results visualization
│
├── outputs/                    # 实验配置和结果 / Experiment configurations and results
│   ├── configs/                    # 实验配置 / Experiment configurations
│   │   ├── experiment_1.yaml       # 第一个实验配置 / First experiment configuration
│   │   └── experiment_2.yaml       # 第二个实验配置 / Second experiment configuration
│   │
│   └── results/                    # 实验结果 / Experiment results
│       ├── experiment_1/           # 第一个实验结果 / First experiment results
│       │   ├── metrics/            # 性能指标 / Performance metrics
│       │   ├── figures/            # 图表 / Figures
│       │   └── model_comparisons/  # 模型比较 / Model comparisons
│       └── experiment_2/
│
├── notebooks/                      # Jupyter notebooks
│   ├── exploratory/                # 探索性分析 / Exploratory analysis
│   ├── model_comparisons/          # 模型比较可视化 / Model comparison visualization
│   └── result_analysis/            # 结果分析 / Result analysis
│
├── Reference/                      # 参考文献 / Reference
├── main.py                         # 主执行脚本 / Main execution script
├── run_experiment.py               # 实验运行脚本 / Experiment running script
├── requirements.txt                # 项目依赖 / Project dependencies
└── README.md                       # 项目说明 / Project description
```
## Installation and Setup

### Environment Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU training)

### Installation Steps

1. Clone the repository
   ```bash
   git clone https://github.com/Cherry-808/DSS5104-ForecastDL.git
   cd DSS5104-ForecastDL
   ```

2. Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # 在Windows上: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

1. Place raw data in the corresponding `data/<dataset_name>/raw/` directory

2. Run preprocessing script
   ```bash
   python src/data/preprocessing.py --dataset dataset_1
   ```

### Training Models

Run training script, specifying model and dataset
```bash
python run_experiment.py --config experiments/configs/experiment_1.yaml
```

### Model Evaluation

Evaluate model performance and generate comparison report
```bash
python src/evaluation/compare_models.py --results_dir experiments/results/experiment_1
```

### Visualizing Results

Generate result visualizations
```bash
python src/visualization/plot_results.py --results_dir experiments/results/experiment_1
```

## Adding New Models

1. Create a new model implementation file in the `src/models/` directory

2. Ensure the new model inherits from the base class defined in `base_model.py`

3. Add parameters for the new model in the configuration file

## Adding New Datasets

1. Create a new dataset directory under the `data/` directory

2. Implement corresponding data loading functions

3. Update configuration files to include the new dataset

## Experiment Configuration

Example configuration file (`experiments/configs/experiment_1.yaml`):

```yaml
experiment_name: "experiment_1"
random_seed: 42

datasets:
  - name: "dataset_1"
    train_split: 0.7
    val_split: 0.15
    test_split: 0.15
    
  - name: "dataset_2"
    train_split: 0.8
    val_split: 0.1
    test_split: 0.1

models:
  - name: "deep_model_1"
    type: "deep"
    params:
      hidden_layers: [128, 64]
      dropout: 0.2
      
  - name: "baseline_model_1"
    type: "baseline"
    params:
      max_depth: 10

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  early_stopping: 10
  optimizer: "adam"

evaluation:
  metrics: ["accuracy", "f1_score", "precision", "recall", "roc_auc"]
  cross_validation: 5
```

## 贡献指南 / Contribution Guidelines

1. Fork仓库 / Fork the repository
2. 创建功能分支 / Create a feature branch
3. 提交更改 / Commit your changes
4. 推送到分支 / Push to the branch
5. 创建Pull Request / Create a Pull Request
