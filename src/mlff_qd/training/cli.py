import argparse
import logging
import yaml

from mlff_qd.utils.helpers import load_config
from mlff_qd.utils.yaml_utils import extract_engine_yaml
from mlff_qd.utils.nequip_wrapper import run_nequip_training
from mlff_qd.utils.mace_wrapper import run_mace_training
from mlff_qd.utils.training import run_schnet_training
from mlff_qd.training.inference import run_schnet_inference


def parse_args():
    parser = argparse.ArgumentParser(description="MLFF-QD CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--engine", required=False, help="Engine override (allegro, mace, nequip, schnet, painn, fusion, nequip_painn_fusion)")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    platform = (args.engine or config.get("platform", "")).lower()

    if platform not in ["allegro", "mace", "nequip", "schnet", "painn", "fusion"]:
        raise ValueError(f"Unknown or missing platform/engine: {platform}")

    engine_yaml = extract_engine_yaml(args.config, platform)
    
    try:
        if platform == "nequip":
            run_nequip_training(engine_yaml)
        elif platform == "mace":
            run_mace_training(engine_yaml)
        elif platform == "allegro":
            run_nequip_training(engine_yaml)
        elif platform in ["schnet", "painn", "fusion"]:  # All use the same training function
            run_schnet_training(engine_yaml)
            run_schnet_inference(engine_yaml)
    except Exception as e:
        logging.error(f"Training failed for platform {platform}: {str(e)}")
        raise

if __name__ == "__main__":
    main()