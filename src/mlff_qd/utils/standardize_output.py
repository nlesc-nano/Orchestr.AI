import os
import shutil
import logging
import argparse

def move_if_exists(src, dst_dir, rename=None):
    if os.path.exists(src):
        dst = os.path.join(dst_dir, rename if rename else os.path.basename(src))
        try:
            if os.path.isdir(src):
                # Move directory (including contents)
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
            else:
                shutil.move(src, dst)
            logging.info(f"Moved {src} → {dst}")
        except Exception as e:
            logging.warning(f"Could not move {src}: {e}")

def standardize_output(platform, source_dir, dest_dir, results_dir=None, config_yaml_path=None):
    """Standardize the output folder structure for a given platform."""
    logging.basicConfig(level=logging.INFO)
    os.makedirs(dest_dir, exist_ok=True)

    standardized_dirs = {
        "engine_yaml": os.path.join(dest_dir, "engine_yaml"),
        "converted_data": os.path.join(dest_dir, "converted_data"),
        "best_model": os.path.join(dest_dir, "best_model"),
        "checkpoints": os.path.join(dest_dir, "checkpoints"),
        "logs": os.path.join(dest_dir, "logs"),
        "lightning_logs": os.path.join(dest_dir, "lightning_logs"),
    }
    for d in standardized_dirs.values():
        os.makedirs(d, exist_ok=True)

    # ---- Always copy the YAML file actually used for this run ----
    if config_yaml_path and os.path.exists(config_yaml_path):
        dst = os.path.join(standardized_dirs["engine_yaml"], os.path.basename(config_yaml_path))
        shutil.copy(config_yaml_path, dst)
        logging.info(f"Copied config YAML: {config_yaml_path} → {dst}")
    else:
        logging.warning(f"No config YAML found at {config_yaml_path}; skipping YAML copy.")

    # Always move converted_data if it exists
    move_if_exists(os.path.join(source_dir, "converted_data"), standardized_dirs["converted_data"])

    if not results_dir:
        results_dir = os.path.join(source_dir, "results")

    if platform in ("schnet", "painn", "fusion"):
        # Best model: results/best_inference_model/
        move_if_exists(os.path.join(results_dir, "best_inference_model"), standardized_dirs["best_model"])
        # Checkpoints: results/lightning_logs/version_*/checkpoints/
        lightning_logs = os.path.join(results_dir, "lightning_logs")
        if os.path.exists(lightning_logs):
            for version_folder in os.listdir(lightning_logs):
                vpath = os.path.join(lightning_logs, version_folder, "checkpoints")
                if os.path.exists(vpath):
                    move_if_exists(vpath, standardized_dirs["checkpoints"])
            move_if_exists(lightning_logs, standardized_dirs["lightning_logs"])
        # Logs: move .log files and hparams.yaml from results/
        if os.path.exists(results_dir):
            for f in os.listdir(results_dir):
                if f.endswith(".log") or "hparams.yaml" in f:
                    move_if_exists(os.path.join(results_dir, f), standardized_dirs["logs"])
        for f in os.listdir(source_dir):
            if f.endswith(".log"):
                move_if_exists(os.path.join(source_dir, f), standardized_dirs["logs"])

    elif platform in ("nequip", "allegro"):
        # Move checkpoints (.ckpt) from results/
        move_if_exists(os.path.join(results_dir, "best.ckpt"), standardized_dirs["best_model"])
        move_if_exists(os.path.join(results_dir, "last.ckpt"), standardized_dirs["checkpoints"])
        # Move lightning logs (tutorial_log/version_0) from results/
        tutorial_log = os.path.join(results_dir, "tutorial_log")
        if os.path.exists(tutorial_log):
            for version_folder in os.listdir(tutorial_log):
                vpath = os.path.join(tutorial_log, version_folder)
                move_if_exists(vpath, standardized_dirs["lightning_logs"])
        # Logs: outputs/<date>/<time>/train.log
        outputs_dir = os.path.join(source_dir, "outputs")
        if os.path.exists(outputs_dir):
            for date_folder in os.listdir(outputs_dir):
                date_path = os.path.join(outputs_dir, date_folder)
                if os.path.isdir(date_path):
                    for time_folder in os.listdir(date_path):
                        time_path = os.path.join(date_path, time_folder)
                        train_log = os.path.join(time_path, "train.log")
                        if os.path.exists(train_log):
                            move_if_exists(train_log, standardized_dirs["logs"])

    elif platform == "mace":
        checkpoints_dir = os.path.join(source_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            move_if_exists(checkpoints_dir, standardized_dirs["best_model"])
        logs_dir = os.path.join(source_dir, "logs")
        if os.path.exists(logs_dir):
            for f in os.listdir(logs_dir):
                move_if_exists(os.path.join(logs_dir, f), standardized_dirs["logs"])
        if os.path.exists(results_dir):
            for f in os.listdir(results_dir):
                move_if_exists(os.path.join(results_dir, f), standardized_dirs["logs"])
        valid_indices = os.path.join(source_dir, "valid_indices_42.txt")
        if os.path.exists(valid_indices):
            move_if_exists(valid_indices, standardized_dirs["logs"])

def parse_args():
    parser = argparse.ArgumentParser(description="Standardize output folder structure for MLFF-QD.")
    parser.add_argument("--platform", type=str, required=True, help="Platform name (e.g., schnet, mace)")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory to standardize")
    parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory for standardized structure")
    parser.add_argument("--results_dir", type=str, default=None, help="(Optional) Results/logs root for the engine")
    parser.add_argument("--config_yaml_path", type=str, default=None, help="Path to the config YAML actually used for this run")
    return parser.parse_args()

def main():
    args = parse_args()
    standardize_output(
        args.platform,
        args.source_dir,
        args.dest_dir,
        results_dir=args.results_dir,
        config_yaml_path=args.config_yaml_path
    )

if __name__ == "__main__":
    main()
