import numpy as np
import pandas as pd
import h5py
import logging
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