import argparse
import os
import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

def load_config_preproc(config_file=None):
    # If no config file is specified, use a default path relative to this script.
    if config_file is None:
        # Adjust parents[...] as necessary based on your directory structure
        default_path = "preprocess_config.yaml"
        config_file = str(default_path)  # convert Path object to string if needed

    try:
        # Load user-defined configuration
        with open(config_file, "r") as file:
            user_config = yaml.safe_load(file)
            if user_config is None:
                user_config = {}
    except FileNotFoundError:
        print(f"Configuration file '{config_file}' not found. Using only default settings.")
        user_config = {}

    return user_config

def load_config(config_file="config.yaml"):
    """
    Loads configuration from a YAML file.

    Parameters:
        config_file (str): Path to the YAML configuration file (default: "config.yaml").

    Returns:
        dict or None: Configuration dictionary if loaded successfully, or None if there was an error.
    """
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        return None

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_file}.")
        return config
    except Exception as e:
        print(f"Error loading configuration from '{config_file}': {e}")
        return None

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

