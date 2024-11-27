import numpy as np
import pandas as pd
import os
import h5py
from ase import Atoms
import torch
import torchmetrics
import pytorch_lightning as pl
import schnetpack as spk
import schnetpack.transform as trn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import yaml
import argparse
import functools
import logging
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def setup_logging():
    script_directory = os.getcwd()
    log_file_path = os.path.join(script_directory, 'Output_Training_times.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def timer(func):
    """A decorator that records the execution time of the function it decorates and logs the time."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def find_middle_two_numbers(n):
    if n % 2 == 0:
        return [(n // 2) - 1, n // 2]
    else:
        return [n // 2]

def read_homo_lumo_and_gaps_to_dataframe(filename):
    with h5py.File(filename, 'r') as f:
        eigenvalues_group = f['eigenvalues']
        first_point_name = 'point_0'
        if first_point_name in eigenvalues_group:
            num_eigenvalues = len(eigenvalues_group[first_point_name][()])
            middle_indices = find_middle_two_numbers(num_eigenvalues)
            if len(middle_indices) == 1:
                homo_index = middle_indices[0]
                lumo_index = homo_index + 1
            else:
                homo_index = middle_indices[0]
                lumo_index = middle_indices[1]
        else:
            print(f'Warning: {first_point_name} not found in the file.')
            return None

        ntotal = len(eigenvalues_group)
        print(f'Calculated Total points: {ntotal}')
        print(f'Total eigenvalues: {num_eigenvalues}')
        print(f'Mid two points points: {middle_indices}')
        
        homo_lumo_energies = []
        hl_gaps = []
        valid_indices = []
        for i in range(ntotal):
            point_name = f'point_{i}'
            if point_name in eigenvalues_group:
                eigenvalues = eigenvalues_group[point_name][()] * 27.211  # Convert to eV
                homo_energy = eigenvalues[homo_index]
                lumo_energy = eigenvalues[lumo_index]
                gap = lumo_energy - homo_energy
                homo_lumo_energies.append((homo_energy, lumo_energy))
                hl_gaps.append(gap)
                valid_indices.append(i)
            else:
                print(f'Warning: {point_name} not found in the file. Skipping.')

        df = pd.DataFrame(homo_lumo_energies, columns=['HOMO_Energy (eV)', 'LUMO_Energy (eV)'])
        df['HOMO_LUMO_Gap (eV)'] = hl_gaps
        df['index'] = valid_indices  # Keep track of indices
        return df

def get_homo_lumo_indices(num_eigenvalues):
    """Calculate the indices for HOMO and LUMO based on the total number of eigenvalues."""
    if num_eigenvalues % 2 == 0:
        homo_index = (num_eigenvalues // 2) - 1
        lumo_index = homo_index + 1
    else:
        homo_index = num_eigenvalues // 2
        lumo_index = homo_index + 1
    return homo_index, lumo_index

def parse_eigenvalue_labels(eigenvalue_labels, num_eigenvalues):
    """Convert labels like 'HOMO', 'LUMO+1' to actual indices."""
    logging.info("In parse_eigenvalue_labels() *********: ")
    
    homo_index, lumo_index = get_homo_lumo_indices(num_eigenvalues)
    index_mapping = {}
    for label in eigenvalue_labels:
        if 'HOMO' in label:
            offset = label.replace('HOMO', '')
            if offset == '':
                idx = homo_index
            else:
                idx = homo_index + int(offset)
        elif 'LUMO' in label:
            offset = label.replace('LUMO', '')
            if offset == '':
                idx = lumo_index
            else:
                idx = lumo_index + int(offset)
        else:
            raise ValueError(f"Invalid eigenvalue label: {label}")
        if idx < 0 or idx >= num_eigenvalues:
            raise ValueError(f"Eigenvalue index {idx} out of bounds for label {label}")
        index_mapping[label] = idx
    return index_mapping

def read_eigenvalues_to_dataframe(filename, eigenvalue_labels):
    logging.info("In read_eigenvalues_to_dataframe() *********: ")
    
    with h5py.File(filename, 'r') as f:
        eigenvalues_group = f['eigenvalues']
        first_point_name = 'point_0'
        if first_point_name in eigenvalues_group:
            num_eigenvalues = len(eigenvalues_group[first_point_name][()])
            index_mapping = parse_eigenvalue_labels(eigenvalue_labels, num_eigenvalues)
        else:
            print(f'Warning: {first_point_name} not found in the file.')
            return None

        eigenvalues_list = []
        valid_indices = []
        
        logging.info("In read_eigenvalues_to_dataframe() first loop started *********: ")
        for i in range(len(eigenvalues_group)):
            point_name = f'point_{i}'
            if point_name in eigenvalues_group:
                eigenvalues = eigenvalues_group[point_name][()] * 27.211  # Convert to eV
                selected_eigenvalues = [eigenvalues[idx]
                                        for idx in index_mapping.values()]
                eigenvalues_list.append(selected_eigenvalues)
                valid_indices.append(i)
            else:
                print(f'Warning: {point_name} not found in the file. Skipping.')
        df = pd.DataFrame(eigenvalues_list, columns=eigenvalue_labels)
        df['index'] = valid_indices  # Keep track of indices
        return df

# Define a custom loss function for 'homo_lumo'
def homo_lumo_loss_fn(pred, target):
    # Reshape target to match the shape of pred
    target = target.view(pred.shape)
    return torch.nn.functional.mse_loss(pred, target)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Machine Learning Force Field Training with SchNetPack")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration YAML file")
    args = parser.parse_args()
    return args

@timer
def main():

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    args = parse_args()

    setup_logging()
    logging.info(f"{'*' * 30} Started {'*' * 30}")
    print("Started")

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

    cutoff = config['settings']['model']['cutoff']
    n_rbf = config['settings']['model']['n_rbf']
    n_atom_basis = config['settings']['model']['n_atom_basis']
    n_interactions = config['settings']['model']['n_interactions']
    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)

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
        trn.ASENeighborList(cutoff=cutoff),
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

    logging.info("Done data Prepeation")
    
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis, n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )

    output_modules = [
        spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='energy'),
        spk.atomistic.Forces(energy_key='energy', force_key='forces')
    ]

    if include_homo_lumo:
        output_modules.extend([
            spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='homo'),
            spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='lumo')
        ])

    if include_bandgap:
        output_modules.append(spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='bandgap'))

    if include_eigenvalues_vector:
        n_out = len(config['settings']['data']['eigenvalue_labels'])
        output_modules.append(spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='eigenvalues_vector', n_out=n_out))

    postprocessors = [trn.CastTo64(), trn.AddOffsets('energy', add_mean=True, add_atomrefs=False)]
    if include_homo_lumo:
        postprocessors.append(trn.AddOffsets('homo', add_mean=True, add_atomrefs=False))
        postprocessors.append(trn.AddOffsets('lumo', add_mean=True, add_atomrefs=False))
    if include_bandgap:
        postprocessors.append(trn.AddOffsets('bandgap', add_mean=True, add_atomrefs=False))

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=output_modules,
        postprocessors=postprocessors
    )

    output_energy = spk.task.ModelOutput(
        name='energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=config['settings']['outputs']['energy']['loss_weight'],
        metrics={"MAE": torchmetrics.MeanAbsoluteError()}
    )

    output_forces = spk.task.ModelOutput(
        name='forces',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=config['settings']['outputs']['forces']['loss_weight'],
        metrics={"MAE": torchmetrics.MeanAbsoluteError()}
    )

    outputs = [output_energy, output_forces]

    if include_homo_lumo:
        outputs.extend([
            spk.task.ModelOutput(name='homo', loss_fn=torch.nn.MSELoss(), loss_weight=config['settings']['outputs']['homo']['loss_weight'], metrics={"MAE": torchmetrics.MeanAbsoluteError()}),
            spk.task.ModelOutput(name='lumo', loss_fn=torch.nn.MSELoss(), loss_weight=config['settings']['outputs']['lumo']['loss_weight'], metrics={"MAE": torchmetrics.MeanAbsoluteError()})
        ])

    if include_bandgap:
        outputs.append(spk.task.ModelOutput(
            name='bandgap', loss_fn=torch.nn.MSELoss(),
            loss_weight=config['settings']['outputs']['gap']['loss_weight'],
            metrics={"MAE": torchmetrics.MeanAbsoluteError()}
        ))

    if include_eigenvalues_vector:
        outputs.append(spk.task.ModelOutput(
            name='eigenvalues_vector', loss_fn=homo_lumo_loss_fn,
            loss_weight=config['settings']['outputs']['eigenvalues_vector']['loss_weight'],
            metrics={"MAE": torchmetrics.MeanAbsoluteError()}
        ))

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=outputs,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": config['settings']['training']['optimizer']['lr']},
        scheduler_cls=ReduceLROnPlateau,
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


if __name__ == '__main__':
    main()