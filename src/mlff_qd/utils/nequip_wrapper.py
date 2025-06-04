import yaml
import logging
from nequip.scripts.train import main as nequip_train_main

def run_nequip_training(config_path):
    """
    Run NequIP training with the specified config file.
    
    Args:
        config_path (str): Path to the NequIP YAML config file.
    """
    try:
        # Load config to check for specific settings
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        
        # Convert optimizer_betas to tuple if it exists
        if "optimizer_betas" in config and isinstance(config["optimizer_betas"], list):
            config["optimizer_betas"] = tuple(config["optimizer_betas"])
        
        # Write updated config to a temporary file
        import tempfile
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w")
        yaml.dump(config, temp)
        temp.close()
        config_path = temp.name

        # Default flags, configurable via YAML
        flags = []
        if config.get("equivariance_test", True):
            flags.append("--equivariance-test")
        if config.get("warn_unused", True):
            flags.append("--warn-unused")
        
        # Construct argv
        argv = ["nequip-train", config_path] + flags
        
        # Preserve original sys.argv
        import sys
        original_argv = sys.argv
        sys.argv = argv
        try:
            nequip_train_main()
        finally:
            sys.argv = original_argv
            import os
            os.unlink(config_path)  # Clean up temporary file
    except Exception as e:
        logging.error(f"NequIP training failed with config {config_path}: {str(e)}")
        raise