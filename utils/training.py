import logging
import os
from utils.data_processing import load_data, preprocess_data, setup_logging_and_dataset, prepare_transformations, setup_data_module
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

    data, df_eigenvalues, include_homo_lumo, include_bandgap, include_eigenvalues_vector = load_data(config)
    atoms_list, property_list = preprocess_data(data, df_eigenvalues, include_homo_lumo, include_bandgap, include_eigenvalues_vector)
    new_dataset, property_units = setup_logging_and_dataset(config, atoms_list, property_list, include_homo_lumo, include_bandgap, include_eigenvalues_vector)
    
    transformations = prepare_transformations(config, include_homo_lumo, include_bandgap)
    custom_data = setup_data_module(config, os.path.join(config['settings']['logging']['folder'], config['settings']['general']['database_name']), transformations, property_units)
    
    nnpot, outputs = setup_model(config, include_homo_lumo, include_bandgap, include_eigenvalues_vector)
    task, trainer = setup_task_and_trainer(config, nnpot, outputs, config['settings']['logging']['folder'])
    
    logging.info("Starting training")
    trainer.fit(task, datamodule=custom_data)