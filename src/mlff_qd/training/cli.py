import argparse
import logging
import yaml

from mlff_qd.utils.helpers import load_config
from mlff_qd.utils.yaml_utils import extract_engine_yaml
from mlff_qd.utils.nequip_wrapper import run_nequip_training
from mlff_qd.utils.mace_wrapper import run_mace_training
from mlff_qd.training.training import run_schnet_training
from mlff_qd.training.inference import run_schnet_inference

def parse_args():
    parser = argparse.ArgumentParser(description="MLFF-QD CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--engine", required=True, help="Engine override (allegro, mace, nequip, schnet, painn, fusion)")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    platform = args.engine.lower() 

    # List of all supported platforms
    all_platforms = ["nequip", "allegro", "mace", "schnet", "painn", "fusion"]

    if platform not in all_platforms:
        raise ValueError(f"Unknown platform/engine: {platform}. Supported platforms are {all_platforms}")

    engine_yaml = extract_engine_yaml(args.config, platform)
    try:
        if platform == "nequip":
            run_nequip_training(engine_yaml)
        elif platform == "mace":
            run_mace_training(engine_yaml)
        elif platform == "allegro":
            run_nequip_training(engine_yaml) 
        elif platform in ["schnet", "painn", "fusion"]:
            run_schnet_training(engine_yaml)
            run_schnet_inference(engine_yaml)
    except Exception as e:
        logging.error(f"Training or inference failed for platform {platform}: {str(e)}")
        raise

if __name__ == "__main__":
    main()