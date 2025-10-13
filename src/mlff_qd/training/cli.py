import argparse
import logging
import yaml
import os
import tempfile

import shutil
import pandas as pd

from mlff_qd.utils.helpers import load_config
from mlff_qd.utils.yaml_utils import extract_engine_yaml, validate_input_file
from mlff_qd.utils.nequip_wrapper import run_nequip_training
from mlff_qd.utils.mace_wrapper import run_mace_training
from mlff_qd.training.training import run_schnet_training
from mlff_qd.training.inference import run_schnet_inference
from mlff_qd.utils.standardize_output import standardize_output
from mlff_qd.utils.yaml_utils import get_dataset_paths_from_yaml

from mlff_qd.benchmarks.benchmark_mlff import extract_metrics, post_process_benchmark


def run_benchmark(args, scratch_dir):
    engines = ['schnet', 'painn', 'fusion', 'nequip', 'allegro', 'mace']
    benchmark_results_dir = './benchmark_results'
    os.makedirs(benchmark_results_dir, exist_ok=True)
    
    results = []
    for engine in engines:
        print(f"Benchmarking {engine}...")
        
        # Generate engine YAML
        engine_yaml_path = os.path.join(scratch_dir, f'engine_{engine}.yaml')
        engine_cfg = extract_engine_yaml(args.config, engine, input_xyz=args.input)
        
        # Patch unique output_dir and DB for schnet/painn/fusion to fix CSV/PKL issues
        if engine in ['schnet', 'painn', 'fusion']:
            unique_dir = f"./results_{engine}"
            if 'logging' in engine_cfg:
                engine_cfg['logging']['folder'] = unique_dir
            if 'testing' in engine_cfg:
                engine_cfg['testing']['trained_model_path'] = unique_dir
            if 'general' in engine_cfg and 'database_name' in engine_cfg['general']:
                engine_cfg['general']['database_name'] = f"{engine.capitalize()}.db"
            print(f"Patched dir/DB for {engine}: {unique_dir}/{engine_cfg.get('general', {}).get('database_name', 'CdSe.db')}")

        with open(engine_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(engine_cfg, f)
        
        # Run training + inference
        if engine in ['schnet', 'painn', 'fusion']:
            run_schnet_training(engine_yaml_path)
            run_schnet_inference(engine_yaml_path)
        elif engine in ['nequip', 'allegro']:
            import omegaconf
            omegaconf.OmegaConf.clear_resolvers()
            run_nequip_training(os.path.abspath(engine_yaml_path))
        elif engine == 'mace':
            run_mace_training(engine_yaml_path)
        
        # Standardize
        results_dir = get_output_dir(engine_cfg, engine)
        standardized_src = os.path.join(scratch_dir, 'standardized')
        explicit_paths = get_dataset_paths_from_yaml(engine, engine_yaml_path)
        standardize_output(
            engine,
            scratch_dir,
            standardized_src,
            results_dir=results_dir,
            config_yaml_path=engine_yaml_path,
            explicit_data_paths=explicit_paths
        )
        
        # Move to persistent dir
        engine_results_dir = os.path.join(benchmark_results_dir, engine)
        shutil.copytree(standardized_src, engine_results_dir, dirs_exist_ok=True)
        shutil.rmtree(standardized_src)
        
        # Extract metrics (use your extract logic here or from benchmark_mlff.py if separate)
        engine_df = extract_metrics(engine_results_dir, engine, engine_cfg)  # Assume you have this function
        results.append(engine_df)
    
    if results:
        combined_df = pd.concat(results, ignore_index=True)
        print("\nBenchmark Summary:\n")
        print(combined_df.to_markdown(index=False))
        combined_df.to_csv('benchmark_summary.csv', index_label='Engine')

def parse_args():
    parser = argparse.ArgumentParser(description="MLFF-QD CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--engine", required=False, help="Engine override (allegro, mace, nequip, schnet, painn, fusion)")
    parser.add_argument("--input", help="Path to input XYZ file (overrides input_xyz_file in YAML)")
    parser.add_argument("--only-generate", action="store_true", help="Only generate engine YAML, do not run training")
    parser.add_argument("--train-after-generate", action="store_true", help="Generate engine YAML and immediately start training")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarking across all engines")  # New flag
    parser.add_argument("--post-process", action="store_true", help="Post-process benchmark results and generate summary")
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

def patch_and_validate_yaml(yaml_path, platform, xyz_path=None, scratch_dir=None, write_temp=True):
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
           
     # --- NEW: extension validation only, no conversion ---
    data_path = validate_input_file(data_path, platform)
    
    # Patch original path back into config (no conversion)
    if platform in ["schnet", "painn", "fusion"]:
        config.setdefault("data", {})["dataset_path"] = data_path
    elif platform in ["nequip", "allegro"]:
        config.setdefault("data", {}).setdefault("split_dataset", {})["file_path"] = data_path
    elif platform == "mace":
        config["train_file"] = data_path
    

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


    # if args.benchmark:
        # run_benchmark(args, scratch_dir)
        # return
        
    if args.benchmark:
        run_benchmark(args, scratch_dir)
        return
    
    if args.post_process:
        post_process_benchmark()
        return
        
        
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
            engine_yaml = patch_and_validate_yaml(args.config, platform, xyz_path=args.input, scratch_dir=scratch_dir, write_temp=True)
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
        engine_yaml = patch_and_validate_yaml(args.config, platform, xyz_path=args.input, scratch_dir=scratch_dir, write_temp=True)
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
        explicit_paths = get_dataset_paths_from_yaml(platform, engine_yaml)
        standardize_output(
            platform,
            scratch_dir,
            standardized_dir,
            results_dir=results_dir,
            config_yaml_path=engine_yaml,
            best_model_dir=best_model_dir,
            explicit_data_paths=explicit_paths
        )
        logging.info(f"Output standardized to {standardized_dir}")

    except Exception as e:
        logging.error(f"Training or inference failed for platform {platform}: {str(e)}")
        raise

if __name__ == "__main__":
    main()
