import sys
from mace.cli.run_train import main as mace_train_main
import yaml
import logging
import tempfile
import os

def run_mace_training(config_path):
    """
    Run MACE training with the specified config file.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        
        # NEW: fail-fast validation (stop on mismatch)
        from mlff_qd.utils.yaml_utils import validate_mace_launch_policy
        validate_mace_launch_policy(config)
        
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8")
        yaml.dump(config, temp, allow_unicode=True)
        temp.flush()
        temp.close()
        temp_path = temp.name

        old_argv = list(sys.argv)
        sys.argv = ["mace_run_train", "--config", temp_path]

        try:
            mace_train_main()
        finally:
            sys.argv = old_argv
            os.unlink(temp_path)
    except Exception as e:
        logging.error(f"MACE training failed with config {config_path}: {str(e)}")
        raise