import os
import logging
import numpy as np
import pandas as pd
import h5py
from ase import Atoms
import schnetpack as spk
import schnetpack.transform as trn

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

def load_data(config):
    try:
        data = np.load(config['settings']['data']['dataset_path'])
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {config['settings']['data']['dataset_path']}")
        raise

    output_type = config['settings']['data']['output_type']
    include_homo_lumo = output_type == 2
    include_bandgap = output_type == 3
    include_eigenvalues_vector = output_type == 4

    df_eigenvalues = None
    if include_homo_lumo or include_bandgap or include_eigenvalues_vector:
        data_hdf = config['settings']['data']['hdf5_file']

    if include_homo_lumo or include_bandgap:
        df_eigenvalues = read_homo_lumo_and_gaps_to_dataframe(data_hdf)
        
    if include_eigenvalues_vector:
        eigenvalue_labels = config['settings']['data']['eigenvalue_labels']
        df_eigenvalues = read_eigenvalues_to_dataframe(data_hdf, eigenvalue_labels)

    return data, df_eigenvalues, include_homo_lumo, include_bandgap, include_eigenvalues_vector

def preprocess_data(data, df_eigenvalues, include_homo_lumo, include_bandgap, include_eigenvalues_vector):
    numbers = data["z"]
    atoms_list = []
    property_list = []
    
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
            properties['eigenvalues_vector'] = eigenvalues_vectors[idx].reshape(1, -1)

        atoms_list.append(ats)
        property_list.append(properties)

    return atoms_list, property_list

def setup_logging_and_dataset(config, atoms_list, property_list, include_homo_lumo, include_bandgap, include_eigenvalues_vector):
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
    if include_homo_lumo:
        property_units.update({'homo': 'eV', 'lumo': 'eV'})
    if include_bandgap:
        property_units.update({'bandgap': 'eV'})
    if include_eigenvalues_vector:
        property_units.update({'eigenvalues_vector': 'eV'})

    new_dataset = spk.data.ASEAtomsData.create(
        db_path,
        distance_unit=config['settings']['model']['distance_unit'],
        property_unit_dict=property_units
    )

    new_dataset.add_systems(property_list, atoms_list)
    logging.info(f"Dataset created at {db_path}")
    
    return new_dataset, property_units

def prepare_transformations(config, include_homo_lumo, include_bandgap):
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