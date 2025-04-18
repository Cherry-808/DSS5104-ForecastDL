# src/evaluation/compare_models.py
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data.data_loaders import get_dataset
from src.models.model_factory import model_factory
import src.evaluation.metrics as metrics

def main():
    # 1) 加载配置
    cfg_path = 'path/to/merged.yaml'
    cfg = yaml.safe_load(open(cfg_path))

    # 2) 准备数据（用相同的 history/horizon）
    ds = get_dataset(
        name=cfg['dataset']['name'],
        ts_path=f"{cfg['dataset']['processed_dir']}/ts_values.npy",
        history=cfg['dataset']['history'],
        horizon=cfg['dataset']['horizon'],
    )
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=False)

    # 3) 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_factory(name=cfg['model']['name'], **cfg['model']['params'])
    model.load_state_dict(torch.load(cfg['paths']['checkpoint_dir'] + '/final_model.pt', map_location=device))
    model.to(device).eval()

    # 4) 推断并收集
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y.numpy())
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    # 5) 计算配置中指定的指标
    results = {}
    for m in cfg['evaluation']['metrics']:
        fn = getattr(metrics, m.lower())
        results[m] = fn(y_true, y_pred)

    # 6) 打印／保存
    print("Evaluation Results:", results)

if __name__ == '__main__':
    main()