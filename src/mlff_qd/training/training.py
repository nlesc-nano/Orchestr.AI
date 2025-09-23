import os
import yaml
import logging
import numpy as np
import torch
from torch.nn import DataParallel

# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_model_summary(model, model_name="Model"):
    print(f"\n{'=' * 50}")
    print(f"{model_name} Summary")
    print(f"{'=' * 50}")
    total_params = 0
    for name, module in model.named_children():
        print(f"\nComponent: {name}")
        print(module)
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += params
        print(f"Trainable parameters in {name}: {params:,}")
    print(f"\n{'-' * 50}")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"{'=' * 50}\n")

# ---------------------------------------------------------
# SchNet Training (keep existing)
# ---------------------------------------------------------
def run_schnet_training(config_path):
    # --- existing SchNet code ---
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        if not config_dict:
            raise ValueError("SchNet config is empty")
        config = config_dict
    except Exception as e:
        logging.error(f"SchNet training failed with config {config_path}: {str(e)}")
        raise

    set_seed(config['general']['seed'])

    from mlff_qd.data import load_data, preprocess_data, setup_logging_and_dataset, prepare_transformations, setup_data_module, show_dataset_info

    data = load_data(config)
    atoms_list, property_list = preprocess_data(data)
    new_dataset, property_units = setup_logging_and_dataset(config, atoms_list, property_list)
    show_dataset_info(new_dataset)

    transformations = prepare_transformations(config, "train")
    custom_data = setup_data_module(
        config,
        os.path.join(config['logging']['folder'], config['general']['database_name']),
        transformations,
        property_units
    )

    from mlff_qd.training.trainer_utils import setup_task_and_trainer
    nnpot, outputs = setup_model(config)
    print_model_summary(nnpot, model_name="NeuralNetworkPotential")
    task, trainer = setup_task_and_trainer(config, nnpot, outputs, config['logging']['folder'])

    checkpoint_path = config['resume_training'].get('resume_checkpoint_dir')
    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(task, datamodule=custom_data, ckpt_path=checkpoint_path)
    else:
        logging.info("Starting training from scratch")
        trainer.fit(task, datamodule=custom_data)

# ---------------------------------------------------------
# MACE Training (multi-GPU)
# ---------------------------------------------------------
def run_mace_training(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        if not config_dict:
            raise ValueError("MACE config is empty")
        config = config_dict
    except Exception as e:
        logging.error(f"MACE training failed with config {config_path}: {str(e)}")
        raise

    set_seed(config['seed'])

    from mlff_qd.data import load_data, preprocess_data, setup_logging_and_dataset, prepare_transformations, setup_data_module, show_dataset_info

    data = load_data(config)
    atoms_list, property_list = preprocess_data(data)
    new_dataset, property_units = setup_logging_and_dataset(config, atoms_list, property_list)
    show_dataset_info(new_dataset)

    transformations = prepare_transformations(config, "train")
    custom_data = setup_data_module(
        config,
        os.path.join(config['logging']['folder'], config['database_name']),
        transformations,
        property_units
    )

    from mace import MACE
    model = MACE(config)

    # -------------------------
    # Multi-GPU setup
    # -------------------------
    device = torch.device("cuda")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model)

    print_model_summary(model, model_name="MACE Model")

    # ---------------------------------------------------------
    # Optimizer & Trainer
    # ---------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.0))

    from mlff_qd.training.trainer_utils import setup_task_and_trainer
    task, trainer = setup_task_and_trainer(config, model, outputs=None, log_dir=config['logging']['folder'])

    checkpoint_path = config.get('resume_checkpoint_dir', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(task, datamodule=custom_data, ckpt_path=checkpoint_path)
    else:
        logging.info("Starting training from scratch")
        trainer.fit(task, datamodule=custom_data)

# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    args = parser.parse_args()

    # Choose which training function to call based on config or user input
    # For now, assume MACE
    run_mace_training(args.config)


