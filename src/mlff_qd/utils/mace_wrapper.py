import sys
from mace.cli.run_train import main as mace_train_main

def run_mace_training(config_path):
    sys.argv = [
        "mace_run_train",
        "--config", config_path,
        "--device", "cuda"
    ]
    mace_train_main()