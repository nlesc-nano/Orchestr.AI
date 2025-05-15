"""
config_loader.py

Module to load configuration settings from a YAML file.
"""

import os
import yaml


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

