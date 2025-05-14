import argparse
import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Machine Learning Force Field Training with SchNetPack")
    parser.add_argument("--config", type=str, default="input.yaml", help="Path to the configuration YAML file")
    args = parser.parse_args()
    return args

def get_optimizer_class(name):
    optimizers_map = {
        'Adam':  optim.Adam,
        'AdamW': optim.AdamW,
        'SGD':   optim.SGD
    }
    if name not in optimizers_map:
        raise ValueError(f"Unsupported optimizer '{name}'. Available: {list(optimizers_map.keys())}")
    return optimizers_map[name]

def get_scheduler_class(name):
    schedulers_map = {
        'ReduceLROnPlateau': lr_sched.ReduceLROnPlateau,
        'StepLR':            lr_sched.StepLR
    }
    if name not in schedulers_map:
        raise ValueError(f"Unsupported scheduler '{name}'. Available: {list(schedulers_map.keys())}")
    return schedulers_map[name]