import argparse
import logging
import yaml
import os
import tempfile

import shutil
import pandas as pd

from mlff_qd.utils.helpers import load_config
from mlff_qd.utils.yaml_utils import extract_engine_yaml, validate_input_file, apply_autoddp, _env_world_size
from mlff_qd.utils.nequip_wrapper import run_nequip_training
from mlff_qd.utils.mace_wrapper import run_mace_training
from mlff_qd.training.training import run_schnet_training
from mlff_qd.training.inference import run_schnet_inference
from mlff_qd.utils.schnetpack_wrapper import run_schnetpack_training
from mlff_qd.utils.standardize_output import standardize_output
from mlff_qd.utils.yaml_utils import get_dataset_paths_from_yaml
from mlff_qd.benchmarks.benchmark_mlff import extract_metrics, post_process_benchmark
try:
    from mlff_qd.fine_tuning.fine_tune import main as run_schnet_fine_tuning
except ImportError:
    run_schnet_fine_tuning = None

def run_benchmark(args, scratch_dir):
    engines = ['schnet', 'painn', 'fusion', 'nequip', 'allegro', 'mace', 'so3net', 'field_schnet']
    benchmark_results_dir = './benchmark_results'
    os.makedirs(benchmark_results_dir, exist_ok=True)
    
    results = []
    
    # Pre-resolve the input file for conversion if needed
    from mlff_qd.utils.helpers import load_config
    tmp_cfg = load_config(args.config)
    resolved_input = args.input or tmp_cfg.get("common", {}).get("data", {}).get("input_xyz_file")
    
    db_path = None
    if resolved_input and os.path.exists(resolved_input):
        from mlff_qd.utils.schnetpack_wrapper import convert_to_schnetpack_db
        atomrefs_dict = tmp_cfg.get("common", {}).get("data", {}).get("atomrefs", None)
        db_path = convert_to_schnetpack_db(resolved_input, scratch_dir, atomrefs_dict=atomrefs_dict)

    for engine in engines:
        print(f"Benchmarking {engine}...")
        
        # Generate engine YAML
        engine_yaml_path = os.path.join(scratch_dir, f'engine_{engine}.yaml')
        
        engine_input = db_path if engine in ['schnet', 'painn', 'so3net', 'field_schnet'] and db_path else args.input
        engine_cfg = extract_engine_yaml(args.config, engine, input_xyz=engine_input)

        # Patch unique output_dir and DB for schnet/painn/fusion to fix CSV/PKL issues
        if engine in ['schnet', 'painn', 'fusion', 'so3net', 'field_schnet']:
            unique_dir = f"./results_{engine}"
            if 'logging' in engine_cfg:
                engine_cfg['logging']['folder'] = unique_dir
            if 'run' in engine_cfg:
                engine_cfg['run']['work_dir'] = unique_dir
            if 'testing' in engine_cfg:
                engine_cfg['testing']['trained_model_path'] = unique_dir
            if 'general' in engine_cfg and 'database_name' in engine_cfg['general']:
                engine_cfg['general']['database_name'] = f"{engine.capitalize()}.db"
            print(f"Patched dir/DB for {engine}: {unique_dir}/{engine_cfg.get('general', {}).get('database_name', 'CdSe.db')}")

        with open(engine_yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(engine_cfg, f)


        # Run training + inference
        if engine in ['schnet', 'painn', 'so3net', 'field_schnet']:
            run_schnetpack_training(engine_yaml_path)
        elif engine == 'fusion':
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
        
        # Extract metrics
        engine_df = extract_metrics(engine_results_dir, engine, engine_cfg)
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
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarking across all engines") 
    parser.add_argument("--post-process", action="store_true", help="Post-process benchmark results and generate summary")
    return parser.parse_args()

def get_output_dir(engine_cfg, platform):
    key_paths = {
        "nequip":   [["trainer", "logger", 0, "save_dir"], ["output_dir"]],
        "allegro":  [["trainer", "logger", 0, "save_dir"], ["output_dir"]],
        "schnet":   [["run", "work_dir"], ["logging", "folder"], ["output_dir"]],
        "painn":    [["run", "work_dir"], ["logging", "folder"], ["output_dir"]],
        "so3net":   [["run", "work_dir"], ["logging", "folder"], ["output_dir"]],
        "field_schnet": [["run", "work_dir"], ["logging", "folder"], ["output_dir"]],
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

EXPLICIT_KEYS = ("train_file_path", "val_file_path", "test_file_path")

def _key_present(dct, key: str) -> bool:
    # key exists in YAML even if value is null/empty
    return isinstance(dct, dict) and (key in dct)

def _missing_value(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")

def patch_and_validate_yaml(yaml_path, platform, xyz_path=None, scratch_dir=None, write_temp=True):
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_path = xyz_path

    if not data_path:
        if platform in ["schnet", "painn", "fusion", "so3net", "field_schnet"]:
            data_path = config.get("data", {}).get("dataset_path", None) or config.get("data", {}).get("datapath", None)
        
        elif platform in ["nequip", "allegro"]:
            d = config.get("data", {}) or {}

            # MODE A intent: if user provided ANY explicit key (even null/empty), treat as explicit-mode attempt
            explicit_intent = any(_key_present(d, k) for k in EXPLICIT_KEYS)

            if explicit_intent:
                logging.info(f"[{platform}] Running in MODE A: Explicit files (train_file_path, val_file_path, test_file_path)")

                # Require all three keys
                missing_keys = [k for k in EXPLICIT_KEYS if not _key_present(d, k)]
                if missing_keys:
                    raise ValueError(
                        f"[{platform}] Explicit file mode detected, but missing keys: {', '.join(missing_keys)}. "
                        f"Either provide all of {', '.join(EXPLICIT_KEYS)} OR remove them and use data.split_dataset."
                    )

                # Require non-empty values
                bad_vals = [k for k in EXPLICIT_KEYS if _missing_value(d.get(k))]
                if bad_vals:
                    raise ValueError(
                        f"[{platform}] Explicit file mode detected, but these are null/empty: {', '.join(bad_vals)}. "
                        f"Either set valid paths for all three OR remove those keys and use data.split_dataset instead."
                    )

                # Validate files exist + extension
                for k in EXPLICIT_KEYS:
                    p = d[k]
                    if not os.path.exists(p):
                        raise ValueError(f"[{platform}] Missing file: {p}")
                    validate_input_file(p, platform)

                # Remove split_dataset to avoid ambiguity
                d.pop("split_dataset", None)
                config["data"] = d

                # Placeholder for the legacy flow checks below
                data_path = d["train_file_path"]

            else:
                logging.info(f"[{platform}] Running in MODE B: Using split_dataset for data split.")
                data_path = (d.get("split_dataset") or {}).get("file_path", None)

                if not data_path:
                    raise ValueError(
                        f"[{platform}] Missing split_dataset.file_path (and no explicit train/val/test keys provided)."
                    )
                if not os.path.exists(data_path):
                    raise ValueError(f"[{platform}] Missing file for split_dataset: {data_path}")

        
        elif platform == "mace":
            data_path = config.get("train_file", None)

    # If user passed --input (xyz_path), it should override ONLY in split mode.
    # In explicit mode, we do NOT override train/val/test with --input.
    if platform in ["nequip", "allegro"] and xyz_path:
        d = config.get("data", {}) or {}
        explicit_mode = any(k in d for k in ("train_file_path", "val_file_path", "test_file_path"))

        if not explicit_mode:
            data_path = xyz_path

    # Fallback: If the YAML specifies a .db that doesn't exist, try to find an .xyz or .npz to convert
    if data_path and data_path.endswith(".db") and not os.path.exists(data_path):
        for ext in [".xyz", ".npz"]:
            fb = data_path[:-3] + ext
            if os.path.exists(fb):
                data_path = fb
                break

    # Generic missing-path check
    if not data_path or not os.path.exists(data_path):
        raise ValueError(
            f"YAML file {yaml_path} is missing a valid data path for platform '{platform}'.\n"
            "Either add the correct dataset path to your YAML or provide --input."
        )

    # Input validation only, no conversion
    data_path = validate_input_file(data_path, platform)

    # Patch path back into config
    if platform in ["schnet", "painn", "fusion", "so3net", "field_schnet"]:
        if "datapath" in config.get("data", {}):
            config["data"]["datapath"] = data_path
        else:
            config.setdefault("data", {})["dataset_path"] = data_path

    elif platform in ["nequip", "allegro"]:
        d = config.setdefault("data", {})

        # If explicit mode exists, keep it and ensure split_dataset removed
        if d.get("train_file_path") and d.get("val_file_path"):
            d.pop("split_dataset", None)

            # re-validate again after patching, just to be safe but its optional:
            for k in ("train_file_path", "val_file_path", "test_file_path"):
                p = d.get(k)
                if p is None:
                    continue
                validate_input_file(p, platform)

        # Otherwise split mode
        else:
            d.setdefault("split_dataset", {})["file_path"] = data_path

    elif platform == "mace":
        config["train_file"] = data_path

    # Normalize engine-specific flags before writing YAML
    if platform in ["nequip", "allegro"]:
        apply_autoddp(config)

    # Write YAML out
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
    all_platforms = ["nequip", "allegro", "mace", "schnet", "painn", "fusion", "so3net", "field_schnet"]
    if platform not in all_platforms:
        raise ValueError(f"Unknown platform/engine: {platform}. Supported platforms are {all_platforms}")

    scratch_dir = os.environ.get("SCRATCH_DIR", tempfile.gettempdir())
    os.makedirs(scratch_dir, exist_ok=True)
        
    if args.benchmark:
        run_benchmark(args, scratch_dir)
        return
    
    if args.post_process:
        post_process_benchmark()
        return
        
        
    with open(args.config, "r", encoding="utf-8") as f:
        user_yaml_dict = yaml.safe_load(f)
    is_unified = "common" in user_yaml_dict

    # Convert dataset to SchNetPack .db format if needed
    if platform in ["schnet", "painn", "so3net", "field_schnet"]:
        input_path = args.input
        if not input_path:
            if is_unified:
                input_path = user_yaml_dict.get("common", {}).get("data", {}).get("input_xyz_file")
            else:
                input_path = user_yaml_dict.get("data", {}).get("dataset_path") or user_yaml_dict.get("data", {}).get("datapath")
        
        # Fallback: If the YAML specifies a .db that doesn't exist, try to find an .xyz or .npz to convert
        if input_path and input_path.endswith(".db") and not os.path.exists(input_path):
            for ext in [".xyz", ".npz"]:
                fb = input_path[:-3] + ext
                if os.path.exists(fb):
                    logging.info(f"[{platform}] {input_path} not found, falling back to {fb}")
                    input_path = fb
                    break

        if input_path and not input_path.endswith(".db") and os.path.exists(input_path):
            logging.info(f"[{platform}] Converting {input_path} to SchNetPack DB...")
            from mlff_qd.utils.schnetpack_wrapper import convert_to_schnetpack_db

            # Extract atomrefs for conversion
            if is_unified:
                atomrefs_dict = user_yaml_dict.get("common", {}).get("data", {}).get("atomrefs", None)
            else:
                atomrefs_dict = user_yaml_dict.get("data", {}).get("atomrefs", None)
                
            args.input = convert_to_schnetpack_db(input_path, scratch_dir, atomrefs_dict=atomrefs_dict)

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
    
    # Fine-tuning 
    try:
        is_finetuning = user_yaml_dict.get("common", {}).get("fine_tuning", {}).get("enabled", False)
        if platform in ["schnet", "painn", "fusion", "so3net", "field_schnet"]:
        
            if is_finetuning:
                print(f"[CLI] Fine-tuning mode detected for {platform}.")
                
                if run_schnet_fine_tuning is None:
                    raise ImportError("Could not import 'main' from mlff_qd.fine_tuning.fine_tune.")

                # Create mock args because fine_tune.py expects args.config
                class MockArgs:
                    config = engine_yaml
                
                run_schnet_fine_tuning(MockArgs())
                print(f"[CLI] Fine-tuning completed.")
            else:
                if platform in ["schnet", "painn", "so3net", "field_schnet"]:
                    run_schnetpack_training(engine_yaml)
                else:
                    run_schnet_training(engine_yaml, engine=platform)
                    run_schnet_inference(engine_yaml, engine=platform)

        elif platform == "nequip":
            run_nequip_training(os.path.abspath(engine_yaml))
        elif platform == "mace":
            run_mace_training(engine_yaml)
        elif platform == "allegro":
            run_nequip_training(os.path.abspath(engine_yaml))

        results_dir = get_output_dir(engine_cfg, platform)
        if platform in ["schnet", "painn", "so3net", "field_schnet"]:
            results_dir = scratch_dir 
            
            # Extract raw model path string
            best_model_name = engine_cfg.get("callbacks", {}).get("model_checkpoint", {}).get("model_path", "best_model")
            
            # Manually resolve common Hydra interpolations
            if best_model_name == "${globals.model_path}":
                best_model_name = engine_cfg.get("globals", {}).get("model_path", "best_model")
            
            # The best model is saved directly in the scratch directory
            best_model_dir = os.path.normpath(os.path.join(scratch_dir, os.path.basename(best_model_name)))
                
            logging.info(f"[paths] Resolved SchNetPack best_model_dir: {best_model_dir}")
        else:
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
        logging.info(f"[paths] Result dir (engine output): {results_dir}")
        logging.info(f"[paths] Standardized output dir: {standardized_dir}")

    except Exception as e:
        logging.error(f"Training or inference failed for platform {platform}: {str(e)}")
        raise

if __name__ == "__main__":
    main()
