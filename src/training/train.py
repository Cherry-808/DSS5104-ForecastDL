import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.data_loaders import get_dataset
from src.models.model_factory import model_factory


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

    # Dataset and DataLoader
    print(">> before get_dataset")
    ds = get_dataset(
        name=config['dataset']['name'],
        ts_path=f"{config['dataset']['processed_dir']}/ts_values.npy",
        history=config['dataset']['history'],
        horizon=config['dataset']['horizon']
    )
    loader = DataLoader(
        ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    print(">> after get_dataset, before trainer.fit")
    # Model
    model = model_factory(
        name=config['model']['name'],
        **config['model']['params']
    )
    model = model.to(device)

    # Optimizer and Loss
    # Ensure learning rate is float
    lr_cfg = config['training']['lr']
    lr = float(lr_cfg) if isinstance(lr_cfg, str) else lr_cfg
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    print(">> after trainer.fit")

    # Training loop
    epochs = config['training']['epochs']
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} - avg loss: {avg_loss:.4f}")

    # Save final model
    ckpt_dir = config['paths'].get('checkpoint_dir', 'checkpoints')
    ckpt_path = f"{ckpt_dir}/final_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == '__main__':
    main()
