import numpy as np
import torch
import logging
import os
import yaml 

from mlff_qd.training.trainer_utils import setup_task_and_trainer
from mlff_qd.utils.data_processing import ( load_data, preprocess_data, setup_logging_and_dataset,
        prepare_transformations, setup_data_module, show_dataset_info )
from mlff_qd.utils.model import setup_model
from mlff_qd.utils.helpers import load_config

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
    
def run_schnet_training(config_path):
    #config = load_config(args.config)
    
    try:
        # Load config
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        
        # Validate config
        if not config_dict or "settings" not in config_dict:
            raise ValueError("SchNet config must contain a 'settings' key")
        
        config = config_dict # Extract the settings dictionary
            
    except Exception as e:
        logging.error(f"SchNet training failed with config {config_path}: {str(e)}")
        raise
        

    set_seed(config['settings']['general']['seed'])

    data = load_data(config)
    atoms_list, property_list = preprocess_data(data)
    new_dataset, property_units = setup_logging_and_dataset(config, atoms_list, property_list)
    
    # Show dataset information
    show_dataset_info(new_dataset)
    
    transformations = prepare_transformations(config,"train")
    custom_data = setup_data_module(config, os.path.join(config['settings']['logging']['folder'], config['settings']['general']['database_name']), transformations, property_units)
    
    nnpot, outputs = setup_model(config)
    print_model_summary(nnpot, model_name="NeuralNetworkPotential")
    
    task, trainer = setup_task_and_trainer(config, nnpot, outputs, config['settings']['logging']['folder'])
    
    # Check if we should resume from a checkpoint
    checkpoint_path = config['settings']['resume_training'].get('resume_checkpoint_dir')

    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(task, datamodule=custom_data, ckpt_path=checkpoint_path)
    else:
        logging.info("Starting training from scratch")
        trainer.fit(task, datamodule=custom_data)
