import os
import logging
import numpy as np
import pandas as pd
import h5py
from ase import Atoms
import schnetpack as spk
import schnetpack.transform as trn

def load_data(config):
    try:
        data = np.load(config['settings']['data']['dataset_path'])
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {config['settings']['data']['dataset_path']}")
        raise

    return data

def preprocess_data(data):
    numbers = data["z"]
    atoms_list = []
    property_list = []
    
    positions_array = np.array(data["R"])
    energies_array = np.array(data["E"])
    forces_array = np.array(data["F"])

    for idx in range(len(positions_array)):
        ats = Atoms(positions=positions_array[idx], numbers=numbers)
        properties = {
            'energy': np.array([energies_array[idx]], dtype=np.float32),
            'forces': forces_array[idx].astype(np.float32)
        }
        atoms_list.append(ats)
        property_list.append(properties)

    return atoms_list, property_list

def setup_logging_and_dataset(config, atoms_list, property_list):
    folder = config['settings']['logging']['folder']
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "lightning_logs"), exist_ok=True)

    db_path = os.path.join(folder, config['settings']['general']['database_name'])
    if os.path.exists(db_path):
        os.remove(db_path)

    property_units = {
        'energy': config['settings']['model']['property_unit_dict']['energy'],
        'forces': config['settings']['model']['property_unit_dict']['forces']
    }

    new_dataset = spk.data.ASEAtomsData.create(
        db_path,
        distance_unit=config['settings']['model']['distance_unit'],
        property_unit_dict=property_units
    )

    new_dataset.add_systems(property_list, atoms_list)
    logging.info(f"Dataset created at {db_path}")
    
    return new_dataset, property_units

def prepare_transformations(config, task_type):
    
    cutoff = config['settings']['model']['cutoff']
    transformations = [trn.ASENeighborList(cutoff=cutoff)]
    
    if task_type == "train":
        transformations.append(trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False))
    elif task_type != "infer":
        raise ValueError(f"Unsupported task type: {task_type}")
    
    transformations.append(trn.CastTo32())
    
    return transformations
    
    
def setup_data_module(config, db_path, transformations, property_units):
    custom_data = spk.data.AtomsDataModule(
        db_path,
        batch_size=config['settings']['training']['batch_size'],
        distance_unit=config['settings']['model']['distance_unit'],
        property_units=property_units,
        num_train=config['settings']['training']['num_train'],
        num_val=config['settings']['training']['num_val'],
        transforms=transformations,
        num_workers=config['settings']['training']['num_workers'],
        pin_memory=config['settings']['training']['pin_memory']
    )

    custom_data.prepare_data()
    custom_data.setup()
    logging.info("Data module prepared and set up")
    
    return custom_data
    
    

def show_dataset_info(dataset):
    """
    Display information about the dataset, including available properties and an example molecule.
    
    Args:
        dataset (spk.data.ASEAtomsData): The dataset to inspect.
    """
    print('Number of reference calculations:', len(dataset))

    print('Available properties:')
    for p in dataset.available_properties:
        print('-', p)
    print()

    example = dataset[0]
    print('Properties of molecule with id 0:')
    for k, v in example.items():
        print('-', k, ':', v.shape)