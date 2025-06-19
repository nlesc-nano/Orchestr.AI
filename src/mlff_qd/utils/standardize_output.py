import os
import shutil
import logging
import argparse

def standardize_output(platform, source_dir, dest_dir):
    """Standardize the output folder structure for a given platform."""
    logging.basicConfig(level=logging.INFO)
    os.makedirs(dest_dir, exist_ok=True)

    # Define standardized subdirectories
    standardized_dirs = {
        "engine_yaml": os.path.join(dest_dir, "engine_yaml"),
        "converted_data": os.path.join(dest_dir, "converted_data"),
        "best_model": os.path.join(dest_dir, "best_model"),
        "checkpoints": os.path.join(dest_dir, "checkpoints"),
        "logs": os.path.join(dest_dir, "logs"),
        "lightning_logs": os.path.join(dest_dir, "lightning_logs")
    }

    for dir_path in standardized_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Move engine_yaml
    engine_yaml = [f for f in os.listdir(source_dir) if f.startswith("engine_") and f.endswith(".yaml")]
    if engine_yaml:
        shutil.move(os.path.join(source_dir, engine_yaml[0]), standardized_dirs["engine_yaml"])

    # Move converted_data
    converted_data = os.path.join(source_dir, "converted_data")
    if os.path.exists(converted_data):
        shutil.move(converted_data, standardized_dirs["converted_data"])

    # Move best model (platform-specific logic)
    best_model_moved = False
    if platform in ["schnet", "painn", "fusion"]:
        best_model_path = os.path.join(source_dir, "results", "best_inference_model")
        if os.path.exists(best_model_path):
            shutil.move(best_model_path, standardized_dirs["best_model"])
            best_model_moved = True
    elif platform in ["nequip", "allegro"]:
        best_model_path = os.path.join(source_dir, "results", "best.ckpt")
        if os.path.exists(best_model_path):
            shutil.move(best_model_path, standardized_dirs["best_model"])
            best_model_moved = True
    elif platform == "mace":
        best_model_paths = [f for f in os.listdir(source_dir) if f.startswith("mace_") and ("epoch" in f or "compiled" in f)]
        if best_model_paths:
            for path in best_model_paths:
                shutil.move(os.path.join(source_dir, path), standardized_dirs["best_model"])
            best_model_moved = True

    if not best_model_moved and os.path.exists(os.path.join(source_dir, "best_model")):
        shutil.move(os.path.join(source_dir, "best_model"), standardized_dirs["best_model"])

    # Move checkpoints
    checkpoints_dirs = [
        os.path.join(source_dir, "checkpoints"),
    ]
    for check_dir in checkpoints_dirs:
        if os.path.exists(check_dir):
            shutil.move(check_dir, standardized_dirs["checkpoints"])
    checkpoint_files = [f for f in os.listdir(source_dir) if "epoch=" in f or ".ckpt" in f]
    for file in checkpoint_files:
        shutil.move(os.path.join(source_dir, file), standardized_dirs["checkpoints"])

    # Move lightning_logs if it exists anywhere in the source directory
    lightning_logs_found = None
    for root, dirs, files in os.walk(source_dir):
        if "lightning_logs" in dirs:
            lightning_logs_found = os.path.join(root, "lightning_logs")
            break
    if lightning_logs_found:
        shutil.move(lightning_logs_found, standardized_dirs["lightning_logs"])
        logging.info(f"Moved lightning_logs from {lightning_logs_found} to {standardized_dirs['lightning_logs']}")

    # Move other logs, excluding lightning_logs content
    log_dirs = [
        os.path.join(source_dir, "logs"),
        os.path.join(source_dir, "results"),
        os.path.join(source_dir, "outputs")
    ]
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            for item in os.listdir(log_dir):
                source_path = os.path.join(log_dir, item)
                # Skip if inside lightning_logs to avoid duplication
                if not any(source_path.startswith(os.path.join(log_dir, subdir)) for subdir in ["lightning_logs"]):
                    if item.endswith(".log") or "hparams.yaml" in item:
                        shutil.move(source_path, standardized_dirs["logs"])
    root_logs = [f for f in os.listdir(source_dir) if f.endswith(".log")]
    for log in root_logs:
        shutil.move(os.path.join(source_dir, log), standardized_dirs["logs"])

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Standardize output folder structure for MLFF-QD.")
    parser.add_argument("--platform", type=str, required=True, help="Platform name (e.g., schnet, mace)")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory to standardize")
    parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory for standardized structure")
    return parser.parse_args()

def main():
    args = parse_args()
    standardize_output(args.platform, args.source_dir, args.dest_dir)

if __name__ == "__main__":
    main()