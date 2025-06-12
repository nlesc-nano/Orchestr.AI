import yaml
import logging
import tempfile
import os
import sys
from nequip.scripts.train import main as nequip_train_main

def run_nequip_training(config_path):
    """
    Run NequIP or Allegro training with the specified config file using the latest NequIP version with Hydra.
    
    Args:
        config_path (str): Path to the NequIP or Allegro YAML config file.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        if "optimizer_betas" in config and isinstance(config["optimizer_betas"], list):
            config["optimizer_betas"] = tuple(config["optimizer_betas"])
        
        if "data" in config and "split_dataset" in config["data"] and "input_xyz_file" in config:
            from mlff_qd.utils.data_conversion import preprocess_data_for_platform
            converted_file = preprocess_data_for_platform(config["input_xyz_file"], "nequip" if config.get("model", {}).get("_target_") != "allegro.model.AllegroModel" else "allegro")
            config["data"]["split_dataset"]["file_path"] = converted_file
        
        temp_dir = os.path.dirname(config_path)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8", dir=temp_dir)
        yaml.dump(config, temp_file, allow_unicode=True)
        temp_file.flush()
        temp_file.close()
        updated_config_path = temp_file.name
        config_name = os.path.basename(updated_config_path)

        argv = ["nequip-train", "-cp", temp_dir, "-cn", config_name]
        
        original_argv = sys.argv
        sys.argv = argv
        try:
            nequip_train_main()
        finally:
            sys.argv = original_argv
            os.unlink(updated_config_path)
    except Exception as e:
        logging.error(f"NequIP/Allegro training failed with config {config_path}: {str(e)}")
        raise