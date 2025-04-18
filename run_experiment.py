import argparse
import yaml
import subprocess
import tempfile

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_config',   required=True, help='如 outputs/configs/data_m5.yaml')
    p.add_argument('--model_config',  required=True, help='如 outputs/configs/model_transformer.yaml')
    args = p.parse_args()

    # 1) 载入两个配置
    data_cfg  = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)

    # 2) 合并成一个 dict
    #    注意：key 不冲突的情况下，可以简单 update
    config = {}
    config.update(data_cfg)
    config.update(model_cfg)

    # 3) 把合并后的配置写到一个临时 YAML
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tmp:
        yaml.safe_dump(config, tmp)
        merged_path = tmp.name

    # 4) 调用 train.py
    subprocess.run([
    'python', '-m', 'src.training.train',
    '--config', merged_path
    ], check=True)

if __name__ == '__main__':
    main()