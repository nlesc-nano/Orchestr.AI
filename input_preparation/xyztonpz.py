#!/usr/bin/python

from __future__ import print_function

import argparse
import os
import sys
import numpy as np
from ase.io import read

_z_str_to_z_dict = { 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 
                    'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
                      'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 
                      'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 
                      'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 
                      'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108,
                        'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uuq': 114, 'Uuh': 116, }

# Assumes that the atoms in each molecule are in the same order.
def read_nonstd_ext_xyz(f):
    n_atoms = None

    R, z, E, F = [], [], [], []
    for i, line in enumerate(f):
        line = line.strip()
        if not n_atoms:
            n_atoms = int(line)
            print('Number of atoms per geometry: {:,}'.format(n_atoms))

        file_i, line_i = divmod(i, n_atoms + 2)

        if line_i == 1:
            try:
                e = float(line)
            except ValueError:
                pass
            else:
                E.append(e)

        cols = line.split()
        if line_i >= 2:
            R.append(list(map(float, cols[1:4])))
            if file_i == 0:  # first molecule
                # z.append(cols[0])  # Store atomic symbols
                z.append(_z_str_to_z_dict[cols[0]])
            F.append(list(map(float, cols[4:7])))

        if file_i % 1000 == 0:
            sys.stdout.write('\rNumber of geometries found so far: {:,}'.format(file_i))
            sys.stdout.flush()
    sys.stdout.write('\rNumber of geometries found so far: {:,}'.format(file_i))
    sys.stdout.flush()
    print()

    R = np.array(R).reshape(-1, n_atoms, 3)
    z = np.array(z)
    E = None if not E else np.array(E)
    F = np.array(F).reshape(-1, n_atoms, 3)

    if F.shape[0] != R.shape[0]:
        print('[FAIL] Force labels are missing from the dataset or are incomplete!', file=sys.stderr)
        sys.exit(1)

    f.close()
    return (R, z, E, F)


# Parse arguments
parser = argparse.ArgumentParser(description='Creates a dataset from extended XYZ format.')
parser.add_argument('dataset', metavar='<dataset>', type=argparse.FileType('r'),
                    help='Path to extended xyz dataset file')
parser.add_argument('-o', '--overwrite', dest='overwrite', action='store_true',
                    help='Overwrite existing dataset file')
args = parser.parse_args()
dataset = args.dataset

name = os.path.splitext(os.path.basename(dataset.name))[0]
dataset_file_name = name + '.npz'

# Check for existing file
dataset_exists = os.path.isfile(dataset_file_name)
if dataset_exists and args.overwrite:
    print('[INFO] Overwriting existing dataset file.')
if not dataset_exists or args.overwrite:
    print(f'Writing dataset to \'{dataset_file_name}\'...')
else:
    print(f'[FAIL] Dataset \'{dataset_file_name}\' already exists.', file=sys.stderr)
    sys.exit(1)

lattice, R, z, E, F = None, None, None, None, None

# Read dataset
try:
    mols = read(dataset.name, index=':')
    calc = mols[0].get_calculator()
    is_extxyz = calc is not None
except Exception:
    is_extxyz = False

if is_extxyz:
    print("\rNumber of geometries found: {:,}\n".format(len(mols)))

    if 'forces' not in calc.results:
        print('[FAIL] Forces are missing in the input file!', file=sys.stderr)
        sys.exit(1)

    lattice = np.array(mols[0].get_cell().T)
    if not np.any(lattice):  # all zeros
        print('[INFO] No lattice vectors specified in extended XYZ file.')
        lattice = None

    Z = np.array([mol.get_atomic_numbers() for mol in mols])
    all_z_the_same = (Z == Z[0]).all()
    if not all_z_the_same:
        print('[FAIL] Order of atoms changes across the dataset!', file=sys.stderr)
        sys.exit(1)

    R = np.array([mol.get_positions() for mol in mols])
    z = Z[0]

    if 'Energy' in mols[0].info:
        E = np.array([mol.info['Energy'] for mol in mols])
    if 'energy' in mols[0].info:
        E = np.array([mol.info['energy'] for mol in mols])
    F = np.array([mol.get_forces() for mol in mols])

else:  # legacy non-standard XYZ format
    with open(dataset.name) as f:
        R, z, E, F = read_nonstd_ext_xyz(f)

# Prepare dataset variables
base_vars = {
    'type': 'dataset',
    'name': name,
    'R': R,
    'z': z,
    'F': F,
    'F_min': np.min(F.ravel()),
    'F_max': np.max(F.ravel()),
    'F_mean': np.mean(F.ravel()),
    'F_var': np.var(F.ravel()),
}

# Ask user for units
print('Please provide a description of the length unit used in your input file, e.g., "Ang" or "au": ')
r_unit = input('> ').strip()
if r_unit != '':
    base_vars['r_unit'] = r_unit

print('Please provide a description of the energy unit used in your input file, e.g., "kcal/mol" or "eV": ')
e_unit = input('> ').strip()
if e_unit != '':
    base_vars['e_unit'] = e_unit

if E is not None:
    base_vars['E'] = E
    base_vars['E_min'], base_vars['E_max'] = np.min(E), np.max(E)
    base_vars['E_mean'], base_vars['E_var'] = np.mean(E), np.var(E)
else:
    print('[INFO] No energy labels found in the dataset.')

if lattice is not None:
    base_vars['lattice'] = lattice

# Save dataset
np.savez_compressed(dataset_file_name, **base_vars)
print('[DONE] Dataset saved to:', dataset_file_name)

