import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.data_loaders import get_dataset
from src.models.model_factory import model_factory
from src.evaluation.metrics import MAE, RMSE
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Train forecasting model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to merged YAML config')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device setup: auto downgrade to CPU if CUDA unavailable
    cfg_dev = config['training'].get('device', 'cpu')
    device = torch.device('cuda' if cfg_dev=='cuda' and torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader (Train)
    print(">> before get_dataset (train)")
    train_ds = get_dataset(
        name=config['dataset']['name'],
        ts_path=f"{config['dataset']['processed_dir']}/ts_values.npy",
        history=config['dataset']['history'],
        horizon=config['dataset']['horizon'],
        train=True  # 显式指定加载训练集
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    print(">> after get_dataset (train), before trainer.fit")

    # Model
    model = model_factory(
        name=config['model']['name'],
        **config['model']['params']
    )
    model = model.to(device)

    # Optimizer and Loss
    lr_cfg = config['training']['lr']
    lr = float(lr_cfg) if isinstance(lr_cfg, str) else lr_cfg
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    print(">> after trainer.fit")

    # Training loop
    epochs = config['training']['epochs']
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        model.train() # Set model to training mode
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} - avg loss: {avg_loss:.4f}")

    # Save final model
    ckpt_dir = config['paths'].get('checkpoint_dir', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/final_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")

    # Load test dataset
    print(">> before get_dataset (test)")
    test_ds = get_dataset(
        name=config['dataset']['name'],
        ts_path=f"{config['dataset']['processed_dir']}/ts_values.npy",
        history=config['dataset']['history'],
        horizon=config['dataset']['horizon'],
        train=False  # 显式指定加载测试集
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print(">> after get_dataset (test), before evaluation")

    # Evaluation
    print(">> Starting evaluation...")
    model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            predictions.append(pred.cpu().numpy())
            ground_truth.append(y.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)

    metrics = {}
    if 'MAE' in config['evaluation']['metrics']:
        mae = MAE(predictions, ground_truth)
        metrics['MAE'] = mae
        print(f"MAE: {mae:.4f}")
    if 'RMSE' in config['evaluation']['metrics']:
        rmse = RMSE(predictions, ground_truth)
        metrics['RMSE'] = rmse
        print(f"RMSE: {rmse:.4f}")

    # Save results
    output_dir = config['paths'].get('output_dir', 'outputs/results')
    os.makedirs(output_dir, exist_ok=True)

    prediction_path = f"{output_dir}/predictions.npy"
    ground_truth_path = f"{output_dir}/ground_truth.npy"
    metrics_path = f"{output_dir}/metrics.yaml"

    np.save(prediction_path, predictions)
    np.save(ground_truth_path, ground_truth)

    with open(metrics_path, 'w') as f:
        yaml.safe_dump(metrics, f)

    print(f"Predictions saved to {prediction_path}")
    print(f"Ground truth saved to {ground_truth_path}")
    print(f"Evaluation metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()