import os
import shutil
import logging
import argparse
import glob
from mlff_qd.utils.yaml_utils import NPZ_ENGINES, XYZ_ENGINES

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

def move_best_model(results_dir, dest_dir, pattern="*model*"):
    best_models = glob.glob(os.path.join(results_dir, pattern))
    logging.info(f"Best model candidates found by pattern '{pattern}': {best_models}")
    if not best_models:
        # Fallback: try to guess model file if only 3 things exist
        special_items = {"lightning_logs"}
        candidates = [
            f for f in os.listdir(results_dir)
            if f not in special_items and not f.endswith('.db')
        ]
        logging.info(f"Fallback best model candidates: {candidates}")
        if len(candidates) == 1:
            bm = os.path.join(results_dir, candidates[0])
            move_if_exists(bm, dest_dir)
            logging.warning(f"Used fallback: moved model candidate {bm}")
        else:
            logging.warning(f"No best model found by pattern or fallback in {results_dir}")
    else:
        if len(best_models) > 1:
            logging.warning(f"Multiple best model candidates found: {best_models}")
        for bm in best_models:
            move_if_exists(bm, dest_dir)
 
def move_prediction_files(source_dir, results_dir, dest_dir):
    """
    Moves all .csv and .pkl files from source_dir and results_dir to dest_dir/Prediction.
    """
    prediction_dir = os.path.join(dest_dir, "Prediction")
    os.makedirs(prediction_dir, exist_ok=True)
    n_found = 0
    for base in [source_dir, results_dir]:
        if base and os.path.exists(base):
            for ext in ("*.csv", "*.pkl"):  # Non-recursive
                files = glob.glob(os.path.join(base, ext), recursive=False)
                for f in files:
                    if os.path.isfile(f):
                        print(f"Debug: Moving prediction file {f} to {prediction_dir}")  # Temporary debug
                        move_if_exists(f, prediction_dir)
                        n_found += 1
    if n_found == 0:
        logging.warning(f"[Prediction] No .csv or .pkl files found in {source_dir} or {results_dir}.")
    else:
        logging.info(f"[Prediction] {n_found} .csv/.pkl prediction files moved to {prediction_dir}.")

def move_prepared_data(platform, source_dir, results_dir, dest_dir):
    """
    Move the pre-prepared dataset files (*.npz or *.xyz) into dest_dir/Data depending on engine type.
    Uses engine-to-extension mapping defined in yaml_utils.py.
    """
    data_dir = os.path.join(dest_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)

    # Determine expected extension based on platform
    if platform in NPZ_ENGINES:
        patterns = ("**/*.npz",)
    elif platform in XYZ_ENGINES:
        patterns = ("**/*.xyz",)
    else:
        logging.warning(f"[Data] Unknown platform '{platform}', scanning both .npz and .xyz.")
        patterns = ("**/*.npz", "**/*.xyz")

    seen = set()
    moved = 0

    for base in [source_dir, results_dir]:
        if not base or not os.path.exists(base):
            continue
        for pat in patterns:
            for f in glob.glob(os.path.join(base, pat), recursive=True):
                if not os.path.isfile(f) or f in seen:
                    continue
                seen.add(f)
                try:
                    move_if_exists(f, data_dir)
                    moved += 1
                except Exception as e:
                    logging.warning(f"[Data] Could not move {f}: {e}")

    if moved == 0:
        logging.info(f"[Data] No {patterns} files found for platform {platform}.")
    else:
        logging.info(f"[Data] Moved {moved} file(s) ({patterns}) into {data_dir}.")
        
def standardize_output(platform, source_dir, dest_dir, results_dir=None, config_yaml_path=None, best_model_dir=None):
    """Standardize the output folder structure for a given platform."""
    logging.basicConfig(level=logging.INFO)
    os.makedirs(dest_dir, exist_ok=True)

    standardized_dirs = {
        "engine_yaml": os.path.join(dest_dir, "engine_yaml"),
        "Data": os.path.join(dest_dir, "Data"),
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

    # Always move data if it exists
    move_prepared_data(platform, source_dir, results_dir, dest_dir)

    if not results_dir:
        results_dir = os.path.join(source_dir, "results")

    if platform in ("schnet", "painn", "fusion"):
        # Generalized: Move best model(s) by user-supplied or default pattern
        
        if best_model_dir:
            logging.info(f"Using best model dir from YAML: {best_model_dir}")
            move_if_exists(os.path.join(results_dir, best_model_dir), standardized_dirs["best_model"])
        else:
            move_best_model(results_dir, standardized_dirs["best_model"])  # fallback
        
        
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
        
        # Always move .csv/.pkl predictions from source/results
        move_prediction_files(source_dir, results_dir, dest_dir)
    
    elif platform in ("nequip", "allegro"):
        # Move all .ckpt files from results/ to checkpoints/
        for ckpt_file in glob.glob(os.path.join(results_dir, "*.ckpt")):
            move_if_exists(ckpt_file, standardized_dirs["checkpoints"])

        # Move checkpoints (.ckpt) from results/
        #move_if_exists(os.path.join(results_dir, "best.ckpt"), standardized_dirs["checkpoints"])
        #move_if_exists(os.path.join(results_dir, "last.ckpt"), standardized_dirs["checkpoints"])
        
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
