import os
import time
import functools
import logging
import numpy as np
import torch
from ase import Atoms
import schnetpack as spk
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.data_processing import read_homo_lumo_and_gaps_to_dataframe, read_eigenvalues_to_dataframe
from utils.model import setup_model
from utils.helpers import load_config, get_optimizer_class, get_scheduler_class
import schnetpack.transform as trn

from utils.logging_utils import  timer

@timer
def main(args):
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    config = load_config(args.config)

    data = np.load(config['settings']['data']['dataset_path'])
    output_type = config['settings']['data']['output_type']
    include_homo_lumo = output_type == 2
    include_bandgap = output_type == 3
    include_eigenvalues_vector = output_type == 4

    if include_homo_lumo or include_bandgap or include_eigenvalues_vector:
        data_hdf = config['settings']['data']['hdf5_file']

    if include_homo_lumo or include_bandgap:
        df_eigenvalues = read_homo_lumo_and_gaps_to_dataframe(data_hdf)
        
    if include_eigenvalues_vector:
        eigenvalue_labels = config['settings']['data']['eigenvalue_labels']
        df_eigenvalues = read_eigenvalues_to_dataframe(data_hdf, eigenvalue_labels)

    # Prepare data for SchNetPack
    numbers = data["z"]
    atoms_list = []
    property_list = []
    
    logging.info("Done data loading")
    
    # Preprocess data to avoid in-loop data fetching
    positions_array = np.array(data["R"])
    energies_array = np.array(data["E"])
    forces_array = np.array(data["F"])
    
    if include_homo_lumo or include_bandgap or include_eigenvalues_vector:
        df_eigenvalues.set_index('index', inplace=True, drop=False)
        homo_values = df_eigenvalues["HOMO_Energy (eV)"].values if include_homo_lumo else None
        lumo_values = df_eigenvalues["LUMO_Energy (eV)"].values if include_homo_lumo else None
        bandgap_values = df_eigenvalues["HOMO_LUMO_Gap (eV)"].values if include_bandgap else None

    if include_eigenvalues_vector:
        eigenvalue_columns = [col for col in df_eigenvalues.columns if col not in ['index', 'HOMO_Energy (eV)', 'LUMO_Energy (eV)', 'HOMO_LUMO_Gap (eV)']]
        eigenvalues_vectors = df_eigenvalues[eigenvalue_columns].values.astype(np.float32)

    for idx in range(len(positions_array)):
        ats = Atoms(positions=positions_array[idx], numbers=numbers)
        properties = {
            'energy': np.array([energies_array[idx]], dtype=np.float32),
            'forces': forces_array[idx].astype(np.float32)
        }

        if include_homo_lumo:
            properties.update({
                'homo': np.array([homo_values[idx]], dtype=np.float32),
                'lumo': np.array([lumo_values[idx]], dtype=np.float32)
            })

        if include_bandgap:
            properties['bandgap'] = np.array([bandgap_values[idx]], dtype=np.float32)

        if include_eigenvalues_vector:
            properties['eigenvalues_vector'] = eigenvalues_vectors[idx].reshape(1, -1)  # Adjusted here

        atoms_list.append(ats)
        property_list.append(properties)

    logging.info("loop complete")
    
    folder = config['settings']['logging']['folder']
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "lightning_logs"), exist_ok=True)

    if os.path.exists(os.path.join(folder, 'cspbbr3.db')):
        os.remove(os.path.join(folder, 'cspbbr3.db'))

    property_units = {
        'energy': config['settings']['model']['property_unit_dict']['energy'],
        'forces': config['settings']['model']['property_unit_dict']['forces']
    }
    if include_homo_lumo:
        property_units.update({'homo': 'eV', 'lumo': 'eV'})
    if include_bandgap:
        property_units.update({'bandgap': 'eV'})
    if include_eigenvalues_vector:
        property_units.update({'eigenvalues_vector': 'eV'})

    new_dataset = spk.data.ASEAtomsData.create(
        os.path.join(folder, 'cspbbr3.db'),
        distance_unit=config['settings']['model']['distance_unit'],
        property_unit_dict=property_units
    )

    new_dataset.add_systems(property_list, atoms_list)
    
    #######################################################################

    # Now we can have a look at the data
    print('Number of reference calculations:', len(new_dataset))

    print('Available properties:')
    for p in new_dataset.available_properties:
        print('-', p)
    print()

    example = new_dataset[0]
    print('Properties of molecule with id 0:')
    for k, v in example.items():
        print('-', k, ':', v.shape)

    #######################################################################
    
    del data, atoms_list, property_list
    
    if include_homo_lumo or include_bandgap or include_eigenvalues_vector:
        del df_eigenvalues

    transformations = [
        trn.ASENeighborList(cutoff=config['settings']['model']['cutoff']),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ]
    if include_homo_lumo:
        transformations.append(trn.RemoveOffsets("homo", remove_mean=True, remove_atomrefs=False))
        transformations.append(trn.RemoveOffsets("lumo", remove_mean=True, remove_atomrefs=False))
    if include_bandgap:
        transformations.append(trn.RemoveOffsets("bandgap", remove_mean=True, remove_atomrefs=False))

    custom_data = spk.data.AtomsDataModule(
        os.path.join(folder, 'cspbbr3.db'),
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

    logging.info("Done data Preparation")
    
    nnpot, outputs = setup_model(config, include_homo_lumo, include_bandgap, include_eigenvalues_vector)

    optimizer_name = config['settings']['training']['optimizer']['type']
    scheduler_name = config['settings']['training']['scheduler']['type']

    optimizer_cls = get_optimizer_class(optimizer_name)
    scheduler_cls = get_scheduler_class(scheduler_name)
    
    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=outputs,
        optimizer_cls=optimizer_cls,
        optimizer_args={"lr": config['settings']['training']['optimizer']['lr']},
        scheduler_cls=scheduler_cls,
        scheduler_args={"mode": "min", "factor": config['settings']['training']['scheduler']['factor'],
                        "patience": config['settings']['training']['scheduler']['patience'],
                        "verbose": config['settings']['training']['scheduler']['verbose']},
        scheduler_monitor=config['settings']['logging']['monitor']
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=folder)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(folder,
                                    config['settings']['logging']['checkpoint_dir']),
            save_top_k=1,
            monitor=config['settings']['logging']['monitor']
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=folder,
        max_epochs=config['settings']['training']['max_epochs'],
        accelerator=config['settings']['training']['accelerator'],
        precision=config['settings']['training']['precision'],
        devices=config['settings']['training']['devices']
    )
    logging.info("Start training")
    
    trainer.fit(task, datamodule=custom_data)