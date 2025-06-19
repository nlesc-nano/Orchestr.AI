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
from mlff_qd.utils.standardize_output import standardize_output

def parse_args():
    parser = argparse.ArgumentParser(description="MLFF-QD CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--engine", required=False, help="Engine override (allegro, mace, nequip, schnet, painn, fusion)")
    parser.add_argument("--input", help="Path to input XYZ file (overrides input_xyz_file in YAML)")
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

    # Use SCRATCH_DIR for temporary files if set, otherwise use default temp directory
    scratch_dir = os.environ.get("SCRATCH_DIR", tempfile.gettempdir())
    os.makedirs(scratch_dir, exist_ok=True)

    # Generate engine-specific YAML with updated config
    engine_yaml = os.path.join(scratch_dir, f"engine_{platform}.yaml")
    engine_cfg = extract_engine_yaml(args.config, platform, input_xyz=args.input)
    with open(engine_yaml, "w", encoding="utf-8") as f:
        yaml.dump(engine_cfg, f)
        logging.debug(f"Written engine_cfg for {platform}: {engine_cfg}")
        with open(engine_yaml, "r", encoding="utf-8") as vf:
            written_content = yaml.safe_load(vf)
            logging.debug(f"Verified content of {engine_yaml}: {written_content}")

    try:
        if platform in ["schnet", "painn", "fusion"]:
            run_schnet_training(engine_yaml)
            run_schnet_inference(engine_yaml)
        elif platform == "nequip":
            run_nequip_training(engine_yaml)
        elif platform == "mace":
            run_mace_training(engine_yaml)
        elif platform == "allegro":
            run_nequip_training(engine_yaml)
            
            
        
        #Standardize output after training/inference
        #standardized_dir = os.path.join(scratch_dir, "standardized")
        #standardize_output(platform, scratch_dir, standardized_dir)
        #logging.info(f"Output standardized to {standardized_dir}")


    except Exception as e:
        logging.error(f"Training or inference failed for platform {platform}: {str(e)}")
        raise

if __name__ == "__main__":
    main()