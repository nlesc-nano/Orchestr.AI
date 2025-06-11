import argparse
import logging
import yaml
import os
import tempfile

from mlff_qd.utils.helpers import load_config
from mlff_qd.utils.yaml_utils import extract_engine_yaml
from mlff_qd.utils.nequip_wrapper import run_nequip_training
from mlff_qd.utils.mace_wrapper import run_mace_training
from mlff_qd.training.training import run_schnet_training
from mlff_qd.training.inference import run_schnet_inference

def parse_args():
    parser = argparse.ArgumentParser(description="MLFF-QD CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--engine", required=False, help="Engine override (allegro, mace, nequip, schnet, painn, fusion)")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    platform = (args.engine or config.get("platform", "")).lower()

    # List of all supported platforms
    all_platforms = ["nequip", "allegro", "mace", "schnet", "painn", "fusion"]

    # Validate platform
    if platform not in all_platforms:
        raise ValueError(f"Unknown platform/engine: {platform}. Supported platforms are {all_platforms}")

    # Generate engine-specific YAML with updated config
    engine_yaml = extract_engine_yaml(args.config, platform)

    try:
        # Load the engine-specific config to adjust for training.py
        with open(engine_yaml, "r", encoding="utf-8") as f:
            engine_config = yaml.safe_load(f) or {}
        
        # For SchNet-based platforms, wrap flat or existing settings into a single settings structure
        if platform in ["schnet", "painn", "fusion"]:
            adjusted_config = {"settings": {}}
            # Merge settings from the original config (unified) or engine_config (individual)
            settings_source = {}
            if "common" in config:  # Unified YAML
                settings_source.update(config["common"])
            if platform in config and isinstance(config[platform], dict):  # Platform-specific settings
                settings_source.update(config[platform])
            settings_source.update(engine_config)  # Merge with engine_config (converted settings)
            # Explicitly include general if present
            if "general" in settings_source or "general" in engine_config:
                adjusted_config["settings"]["general"] = settings_source.get("general", engine_config.get("general", {}))
            # Copy other relevant keys
            for key in ["data", "model", "outputs", "training", "logging", "testing", "resume_training", "fine_tuning"]:
                if key in settings_source or key in engine_config:
                    adjusted_config["settings"][key] = settings_source.get(key, engine_config.get(key, {}))
            # Write adjusted config to a new temporary file
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{platform}_adjusted.yaml", mode="w", encoding="utf-8")
            yaml.dump(adjusted_config, temp, allow_unicode=True)
            temp.flush()
            temp.close()
            adjusted_engine_yaml = temp.name
            try:
                run_schnet_training(adjusted_engine_yaml)
                run_schnet_inference(adjusted_engine_yaml)
            finally:
                os.unlink(adjusted_engine_yaml)
        else:
            if platform == "nequip":
                run_nequip_training(engine_yaml)
            elif platform == "mace":
                run_mace_training(engine_yaml)
            elif platform == "allegro":
                run_nequip_training(engine_yaml)
    except Exception as e:
        logging.error(f"Training or inference failed for platform {platform}: {str(e)}")
        raise
    finally:
        os.unlink(engine_yaml)  # Clean up the original engine YAML

if __name__ == "__main__":
    main()