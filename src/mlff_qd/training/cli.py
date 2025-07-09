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
from mlff_qd.utils.data_conversion import preprocess_data_for_platform

def parse_args():
    parser = argparse.ArgumentParser(description="MLFF-QD CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--engine", required=False, help="Engine override (allegro, mace, nequip, schnet, painn, fusion)")
    parser.add_argument("--input", help="Path to input XYZ file (overrides input_xyz_file in YAML)")
    parser.add_argument("--only-generate", action="store_true", help="Only generate engine YAML, do not run training")
    parser.add_argument("--train-after-generate", action="store_true", help="Generate engine YAML and immediately start training")
    return parser.parse_args()

def get_output_dir(engine_cfg, platform):
    key_paths = {
        "nequip":   [["trainer", "logger", 0, "save_dir"], ["output_dir"]],
        "allegro":  [["trainer", "logger", 0, "save_dir"], ["output_dir"]],
        "schnet":   [["logging", "folder"], ["output_dir"]],
        "painn":    [["logging", "folder"], ["output_dir"]],
        "fusion":   [["logging", "folder"], ["output_dir"]],
        "mace":     [["output_dir"]],
    }
    for keys in key_paths.get(platform, []):
        val = engine_cfg
        try:
            for k in keys:
                if isinstance(val, dict) and isinstance(k, str):
                    val = val[k]
                elif isinstance(val, list) and isinstance(k, int):
                    val = val[k]
                else:
                    break
            else:
                if isinstance(val, str) and val.strip():
                    return val
        except Exception:
            continue
    return "./results"

def patch_and_convert_yaml(yaml_path, platform, xyz_path=None, scratch_dir=None, write_temp=True):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_path = xyz_path
    if not data_path:
        if platform in ["schnet", "painn", "fusion"]:
            data_path = config.get("data", {}).get("dataset_path", None)
        elif platform in ["nequip", "allegro"]:
            data_path = config.get("data", {}).get("split_dataset", {}).get("file_path", None)
        elif platform == "mace":
            data_path = config.get("train_file", None)
    if not data_path or not os.path.exists(data_path):
        raise ValueError(
            f"YAML file {yaml_path} is missing a valid data path for platform '{platform}'.\n"
            "Either add the correct dataset path to your YAML or provide --input."
        )
    converted_file = preprocess_data_for_platform(
        data_path, platform, output_dir=os.path.join(os.path.dirname(data_path), "converted_data")
    )
    if platform in ["schnet", "painn", "fusion"]:
        config.setdefault("data", {})["dataset_path"] = converted_file
    elif platform in ["nequip", "allegro"]:
        config.setdefault("data", {}).setdefault("split_dataset", {})["file_path"] = converted_file
    elif platform == "mace":
        config["train_file"] = converted_file

    if write_temp:
        if not scratch_dir:
            scratch_dir = tempfile.gettempdir()
        tmp_yaml = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", dir=scratch_dir).name
        with open(tmp_yaml, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        return tmp_yaml
    else:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f)
        return yaml_path

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    platform = (args.engine or config.get("platform", "")).lower()
    all_platforms = ["nequip", "allegro", "mace", "schnet", "painn", "fusion"]
    if platform not in all_platforms:
        raise ValueError(f"Unknown platform/engine: {platform}. Supported platforms are {all_platforms}")

    scratch_dir = os.environ.get("SCRATCH_DIR", tempfile.gettempdir())
    os.makedirs(scratch_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        user_yaml_dict = yaml.safe_load(f)
    is_unified = "common" in user_yaml_dict

    # Always generate engine YAML for unified YAMLs, and optionally for legacy (with --only-generate)
    if is_unified or args.only_generate:
        engine_yaml = os.path.join(scratch_dir, f"engine_{platform}.yaml")
        if is_unified:
            engine_cfg = extract_engine_yaml(args.config, platform, input_xyz=args.input)
        else:
            # For legacy, patch/convert but don't run training if only_generate
            engine_cfg = None  # Will be loaded in next step
            engine_yaml = patch_and_convert_yaml(args.config, platform, xyz_path=args.input, scratch_dir=scratch_dir, write_temp=True)
        if is_unified:
            with open(engine_yaml, "w", encoding="utf-8") as f:
                yaml.dump(engine_cfg, f)
                logging.debug(f"Written engine_cfg for {platform}: {engine_cfg}")

        print(f"\n[INFO] Engine YAML generated at: {engine_yaml}\n")
        print("[INFO] Edit the generated engine YAML if you want to tweak advanced options.")
        print("[INFO] To launch training, run:")
        print(f"    python cli.py --config {engine_yaml} --engine {platform}\n")
        if args.only_generate or (is_unified and not args.train_after_generate):
            return  # Stop here

    # At this point, we know we need to train, so prepare engine_yaml path and engine_cfg
    if is_unified or args.only_generate:
        # Already set above
        with open(engine_yaml, "r", encoding="utf-8") as f:
            engine_cfg = yaml.safe_load(f)
    else:
        # Legacy mode, always patch/convert and run
        engine_yaml = patch_and_convert_yaml(args.config, platform, xyz_path=args.input, scratch_dir=scratch_dir, write_temp=True)
        with open(engine_yaml, "r", encoding="utf-8") as f:
            engine_cfg = yaml.safe_load(f)

    # After YAML is written and loaded, but before starting training
    if not (args.only_generate or (is_unified and not args.train_after_generate)):
        print(f"[INFO] Engine YAML generated at: {engine_yaml}")
        print(f"[INFO] Now starting training for {platform}...\n")
        
    try:
        if platform in ["schnet", "painn", "fusion"]:
            run_schnet_training(engine_yaml)
            run_schnet_inference(engine_yaml)
        elif platform == "nequip":
            run_nequip_training(os.path.abspath(engine_yaml))
        elif platform == "mace":
            run_mace_training(engine_yaml)
        elif platform == "allegro":
            run_nequip_training(os.path.abspath(engine_yaml))

        results_dir = get_output_dir(engine_cfg, platform)
        best_model_dir = engine_cfg.get("logging", {}).get("checkpoint_dir", None)
        standardized_dir = os.path.join(scratch_dir, "standardized")
        standardize_output(
            platform,
            scratch_dir,
            standardized_dir,
            results_dir=results_dir,
            config_yaml_path=engine_yaml,
            best_model_dir=best_model_dir,
        )
        logging.info(f"Output standardized to {standardized_dir}")

    except Exception as e:
        logging.error(f"Training or inference failed for platform {platform}: {str(e)}")
        raise

if __name__ == "__main__":
    main()
