
import numpy as np
import pandas as pd
import h5py
import logging

def find_middle_two_numbers(n):
    return [(n // 2) - 1, n // 2] if n % 2 == 0 else [n // 2]

def read_homo_lumo_and_gaps_to_dataframe(filename):
    with h5py.File(filename, 'r') as f:
        eigenvalues_group = f['eigenvalues']
        first_point_name = 'point_0'
        if first_point_name in eigenvalues_group:
            num_eigenvalues = len(eigenvalues_group[first_point_name][()])
            middle_indices = find_middle_two_numbers(num_eigenvalues)
            homo_index, lumo_index = middle_indices[0], middle_indices[1]
        else:
            print(f"Warning: {first_point_name} not found in the file.")
            return None

        ntotal = len(eigenvalues_group)
        homo_lumo_energies, hl_gaps, valid_indices = [], [], []
        for i in range(ntotal):
            point_name = f"point_{i}"
            if point_name in eigenvalues_group:
                eigenvalues = eigenvalues_group[point_name][()] * 27.211  # Convert to eV
                homo_energy, lumo_energy = eigenvalues[homo_index], eigenvalues[lumo_index]
                gap = lumo_energy - homo_energy
                homo_lumo_energies.append((homo_energy, lumo_energy))
                hl_gaps.append(gap)
                valid_indices.append(i)
            else:
                print(f"Warning: {point_name} not found in the file. Skipping.")

        df = pd.DataFrame(homo_lumo_energies, columns=['HOMO_Energy (eV)', 'LUMO_Energy (eV)'])
        df['HOMO_LUMO_Gap (eV)'] = hl_gaps
        df['index'] = valid_indices
        return df

def prepare_data(config):
    data = np.load(config['settings']['data']['dataset_path'])
    include_homo_lumo = config['settings']['data']['output_type'] == 2
    include_bandgap = config['settings']['data']['output_type'] == 3
    include_eigenvalues_vector = config['settings']['data']['output_type'] == 4

    df_eigenvalues = None
    if include_homo_lumo or include_bandgap or include_eigenvalues_vector:
        data_hdf = config['settings']['data']['hdf5_file']
        df_eigenvalues = read_homo_lumo_and_gaps_to_dataframe(data_hdf)

    return data, df_eigenvalues
