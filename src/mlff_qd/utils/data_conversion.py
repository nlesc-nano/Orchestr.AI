import os
import numpy as np
from ase.io import read, write
import logging

# Mapping of models to output formats
MODEL_OUTPUTS = {
    'schnet': {'ext': '.npz', 'desc': 'schnetpack'},
    'painn': {'ext': '.npz', 'desc': 'painn'},
    'fusion': {'ext': '.npz', 'desc': 'fusion'},
    'nequip': {'ext': '.xyz', 'desc': 'nequip'},
    'allegro': {'ext': '.xyz', 'desc': 'allegro'},
    'mace': {'ext': '.xyz', 'desc': 'mace'},
}

# Atomic number mapping
_z_str_to_z_dict = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
    'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
    'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
    'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
    'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
    'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73,
    'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
    'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108,
    'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Uuq': 114, 'Uuh': 116,
}

def parse_custom_xyz(xyz_file):
    """Parse a basic custom XYZ file into positions, forces, energies, and elements."""
    positions, forces, elements, energies = [], [], [], []
    with open(xyz_file, 'r') as f:
        while True:
            n_line = f.readline()
            if not n_line:
                break
            n_atoms = int(n_line.strip())
            energy_line = f.readline()
            if not energy_line:
                break
            energy = float(energy_line.strip())
            pos_block, frc_block, elem_block = [], [], []
            for _ in range(n_atoms):
                line = f.readline().strip()
                if not line:
                    break
                tokens = line.split()
                elem_block.append(tokens[0])
                pos_block.append([float(x) for x in tokens[1:4]])
                frc_block.append([float(x) for x in tokens[4:7]])
            positions.append(pos_block)
            forces.append(frc_block)
            energies.append(energy)
            elements = elem_block  # Assume same elements for all frames
    return np.array(positions), np.array(forces), np.array(energies), np.array(elements)

def convert_to_npz(xyz_file, out_file):
    """Convert custom XYZ to NPZ format for SchNetPack-based models."""
    positions, forces, energies, elements = parse_custom_xyz(xyz_file)
    z = np.array([_z_str_to_z_dict[s] for s in elements])
    base_vars = {
        'type': 'dataset',
        'name': os.path.splitext(os.path.basename(xyz_file))[0],
        'R': positions,
        'z': z,
        'F': forces,
        'F_min': np.min(forces.ravel()),
        'F_max': np.max(forces.ravel()),
        'F_mean': np.mean(forces.ravel()),
        'F_var': np.var(forces.ravel()),
        'E': energies,
        'E_min': np.min(energies),
        'E_max': np.max(energies),
        'E_mean': np.mean(energies),
        'E_var': np.var(energies),
        'r_unit': 'Ang',
        'e_unit': 'eV'
    }
    np.savez_compressed(out_file, **base_vars)
    logging.info(f"[DONE] Dataset saved to: {out_file}")
    return out_file

def convert_to_mace_xyz(xyz_file, out_file):
    """Convert custom XYZ to MACE-compatible XYZ format."""
    positions, forces, energies, elements = parse_custom_xyz(xyz_file)
    n_frames = len(energies)
    n_atoms = len(elements)
    with open(out_file, 'w', encoding='utf-8') as outfile:
        for i in range(n_frames):
            outfile.write(f"{n_atoms}\n")
            outfile.write(
                'Lattice="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0" '
                'Properties=species:S:1:pos:R:3:forces:R:3 config_type=Default '
                'pbc="F F F" '
                f'energy={energies[i]}\n'
            )
            for j in range(n_atoms):
                s = elements[j]
                p = positions[i][j]
                fr = forces[i][j]
                outfile.write(f"{s:2} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f} {fr[0]:12.6f} {fr[1]:12.6f} {fr[2]:12.6f}\n")
    logging.info(f"[DONE] MACE-format .xyz saved to: {out_file}")
    return out_file

def preprocess_data_for_platform(input_xyz, platform, output_dir="./converted_data"):
    """Preprocess the basic XYZ file based on the platform and return the converted file path."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_xyz))[0]
    ext = MODEL_OUTPUTS[platform]['ext']
    out_file = os.path.join(output_dir, f"{base_name}_{platform}{ext}")

    if platform in ['schnet', 'painn', 'fusion']:
        return convert_to_npz(input_xyz, out_file)
    elif platform in ['nequip', 'allegro', 'mace']:
        return convert_to_mace_xyz(input_xyz, out_file)
    else:
        raise ValueError(f"Unsupported platform: {platform}")
        