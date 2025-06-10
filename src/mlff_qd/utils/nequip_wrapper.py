import yaml
import logging
import tempfile
import os
import sys
from nequip.scripts.train import main as nequip_train_main

def run_nequip_training(config_path):
    """
    Run NequIP training with the specified config file using the latest NequIP version with Hydra.
    
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
        
        # Write updated config to a temporary file in the scratch directory
        temp_dir = os.path.dirname(config_path)  # Use the existing scratch directory
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", dir=temp_dir)
        yaml.dump(config, temp_file)
        temp_file.close()
        updated_config_path = temp_file.name
        config_name = os.path.basename(updated_config_path)  # e.g., tmpm9p3l_h9.yaml

        # Construct argv with -cp and -cn
        argv = ["nequip-train", "-cp", temp_dir, "-cn", config_name]
        
        # Preserve original sys.argv
        original_argv = sys.argv
        sys.argv = argv
        try:
            nequip_train_main()
        finally:
            sys.argv = original_argv
            os.unlink(updated_config_path)  # Clean up temporary file
    except Exception as e:
        logging.error(f"NequIP training failed with config {config_path}: {str(e)}")
        raise