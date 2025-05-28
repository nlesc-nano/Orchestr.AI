import logging
import os
from utils.data_processing import load_data, preprocess_data, setup_logging_and_dataset, prepare_transformations, setup_data_module, show_dataset_info
from utils.model import setup_model
from utils.trainer_utils import setup_task_and_trainer
from utils.helpers import load_config
import numpy as np
import torch

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    config = load_config(args.config)
    set_seed(config['settings']['general']['seed'])

    data = load_data(config)
    atoms_list, property_list = preprocess_data(data)
    new_dataset, property_units = setup_logging_and_dataset(config, atoms_list, property_list)
    
    # Show dataset information
    show_dataset_info(new_dataset)
    
    transformations = prepare_transformations(config)
    custom_data = setup_data_module(config, os.path.join(config['settings']['logging']['folder'], config['settings']['general']['database_name']), transformations, property_units)
    
    nnpot, outputs = setup_model(config)
    task, trainer = setup_task_and_trainer(config, nnpot, outputs, config['settings']['logging']['folder'])
    
    # Check if we should resume from a checkpoint
    checkpoint_path = config['settings']['resume_training'].get('resume_checkpoint_dir')

    if checkpoint_path and os.path.exists(checkpoint_path):
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(task, datamodule=custom_data, ckpt_path=checkpoint_path)
    else:
        logging.info("Starting training from scratch")
        trainer.fit(task, datamodule=custom_data)