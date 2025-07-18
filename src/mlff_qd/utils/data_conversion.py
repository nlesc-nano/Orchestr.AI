import os
import numpy as np
import logging
import shutil
import re

MODEL_OUTPUTS = {
    'schnet': {'ext': '.npz', 'desc': 'schnetpack'},
    'painn': {'ext': '.npz', 'desc': 'painn'},
    'fusion': {'ext': '.npz', 'desc': 'fusion'},
    'nequip': {'ext': '.xyz', 'desc': 'nequip'},
    'allegro': {'ext': '.xyz', 'desc': 'allegro'},
    'mace': {'ext': '.xyz', 'desc': 'mace'},
}

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

def parse_xyz_header(header_line):
    """Extract metadata (Lattice, pbc, Properties, energy, stress, etc) from header line."""
    meta = {}
    for field in re.findall(r'(\w+)=("[^"]+"|[^\s]+)', header_line):
        k, v = field
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        meta[k] = v
    return meta

def is_prepared_xyz(header_line):
    """Check if header has all expected XYZ keys (strict)."""
    keys = ["Lattice=", "Properties=", "pbc="]
    return all(k in header_line for k in keys)

def copy_to_converted_data(src_file, out_file):
    if os.path.abspath(src_file) != os.path.abspath(out_file):
        shutil.copy(src_file, out_file)
        logging.info(f"[INFO] Copied {src_file} → {out_file}")
    else:
        logging.info(f"[INFO] Input already at destination: {src_file}")
    return out_file

def format_stress_for_xyz(stress_val):
    if stress_val is None:
        return None
    if isinstance(stress_val, str):
        # If already quoted, return as is
        if stress_val.startswith('"') and stress_val.endswith('"'):
            return stress_val
        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", stress_val)
        if len(nums) == 9:
            return '"' + " ".join(nums) + '"'
    if isinstance(stress_val, (list, tuple, np.ndarray)):
        arr = np.asarray(stress_val).flatten()
        if arr.size == 9:
            return '"' + " ".join(f"{float(x):.8f}" for x in arr) + '"'
    return None

def parse_xyz_any(xyz_file):
    """Parse both raw and prepared XYZ. Returns positions, forces, energies, elements, and extra metadata per frame."""
    positions, forces, energies, elements = [], [], [], []
    lattice_list, pbc_list, stress_list = [], [], []
    with open(xyz_file, 'r') as f:
        while True:
            n_line = f.readline()
            if not n_line:
                break
            n_atoms = int(n_line.strip())
            header_line = f.readline()
            if not header_line:
                break
            meta = parse_xyz_header(header_line)
            # Determine if prepared or raw:
            if is_prepared_xyz(header_line):
                energy = float(meta.get("energy", "0.0"))
                lattice = meta.get("Lattice", None)
                pbc = meta.get("pbc", None)
                stress = meta.get("stress", None)
            else:
                # Raw: second line is just energy
                energy = float(header_line.strip())
                lattice = None
                pbc = None
                stress = None
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
            elements = elem_block  # Assume same for all frames
            lattice_list.append(lattice)
            pbc_list.append(pbc)
            stress_list.append(stress)
    return (np.array(positions), np.array(forces), np.array(energies), np.array(elements),
            lattice_list, pbc_list, stress_list)

def convert_to_npz(xyz_file, out_file):
    (positions, forces, energies, elements,
     lattice_list, pbc_list, stress_list) = parse_xyz_any(xyz_file)
    z = np.array([_z_str_to_z_dict[s] for s in elements])
    base_vars = {
        'type': 'dataset',
        'name': os.path.splitext(os.path.basename(xyz_file))[0],
        'R': positions,
        'z': z,
        'F': forces,
        'E': energies,
        'F_min': np.min(forces.ravel()),
        'F_max': np.max(forces.ravel()),
        'F_mean': np.mean(forces.ravel()),
        'F_var': np.var(forces.ravel()),
        'E_min': np.min(energies),
        'E_max': np.max(energies),
        'E_mean': np.mean(energies),
        'E_var': np.var(energies),
        'r_unit': 'Ang',
        'e_unit': 'eV',
    }
    if any(lattice_list):
        base_vars['lattice'] = np.array([l if l is not None else '' for l in lattice_list])
    if any(pbc_list):
        base_vars['pbc'] = np.array([p if p is not None else '' for p in pbc_list])
    if any(stress_list):
        base_vars['stress'] = np.array([s if s is not None else '' for s in stress_list])
    np.savez_compressed(out_file, **base_vars)
    logging.info(f"[DONE] Dataset saved to: {out_file}")
    return out_file

def convert_npz_to_xyz(npz_file, out_file):
    data = np.load(npz_file)
    R = data['R']
    F = data['F']
    E = data['E']
    z = data['z']
    z_rev = {v: k for k, v in _z_str_to_z_dict.items()}
    elements = [z_rev[zi] for zi in z]
    lattice_arr = data['lattice'] if 'lattice' in data else None
    pbc_arr = data['pbc'] if 'pbc' in data else None
    stress_arr = data['stress'] if 'stress' in data else None

    n_frames = R.shape[0]
    n_atoms = R.shape[1]
    with open(out_file, 'w', encoding='utf-8') as f:
        for i in range(n_frames):
            f.write(f"{n_atoms}\n")
            lattice = lattice_arr[i] if lattice_arr is not None else '1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0'
            pbc = pbc_arr[i] if pbc_arr is not None else 'F F F'
            stress = format_stress_for_xyz(stress_arr[i]) if stress_arr is not None else None
            header = (
                f'Lattice="{lattice}" '
                'Properties=species:S:1:pos:R:3:forces:R:3 config_type=Default '
                f'pbc="{pbc}" energy={E[i]}'
            )
            if stress:
                header += f' stress={stress}'
            f.write(header + '\n')
            for j in range(n_atoms):
                s = elements[j]
                p = R[i][j]
                fr = F[i][j]
                f.write(f"{s:2} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f} {fr[0]:12.6f} {fr[1]:12.6f} {fr[2]:12.6f}\n")
    logging.info(f"[DONE] npz→xyz conversion complete: {out_file}")
    return out_file

def convert_to_mace_xyz(xyz_file, out_file):
    (positions, forces, energies, elements,
     lattice_list, pbc_list, stress_list) = parse_xyz_any(xyz_file)
    n_frames = len(energies)
    n_atoms = len(elements)
    with open(out_file, 'w', encoding='utf-8') as outfile:
        for i in range(n_frames):
            outfile.write(f"{n_atoms}\n")
            lattice = lattice_list[i] if lattice_list[i] else '1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0'
            pbc = pbc_list[i] if pbc_list[i] else 'F F F'
            stress = format_stress_for_xyz(stress_list[i]) if stress_list[i] else None
            header = (
                f'Lattice="{lattice}" '
                'Properties=species:S:1:pos:R:3:forces:R:3 config_type=Default '
                f'pbc="{pbc}" energy={energies[i]}'
            )
            if stress:
                header += f' stress={stress}'
            outfile.write(header + '\n')
            for j in range(n_atoms):
                s = elements[j]
                p = positions[i][j]
                fr = forces[i][j]
                outfile.write(f"{s:2} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f} {fr[0]:12.6f} {fr[1]:12.6f} {fr[2]:12.6f}\n")
    logging.info(f"[DONE] Special/prepared .xyz saved to: {out_file}")
    return out_file

def preprocess_data_for_platform(input_file, platform, output_dir="./converted_data"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    ext = MODEL_OUTPUTS[platform]['ext']
    out_file = os.path.join(output_dir, f"{base_name}_{platform}{ext}")

    if input_file.lower().endswith('.npz'):
        if platform in ['schnet', 'painn', 'fusion']:
            return copy_to_converted_data(input_file, out_file)
        elif platform in ['nequip', 'allegro', 'mace']:
            xyz_out = out_file.replace('.xyz', '_from_npz.xyz')
            return convert_npz_to_xyz(input_file, xyz_out)

    elif input_file.lower().endswith('.xyz'):
        with open(input_file) as f:
            f.readline()  # skip atom count
            header = f.readline()
        if platform in ['nequip', 'allegro', 'mace']:
            if is_prepared_xyz(header):
                return copy_to_converted_data(input_file, out_file)
            else:
                return convert_to_mace_xyz(input_file, out_file)
        elif platform in ['schnet', 'painn', 'fusion']:
            return convert_to_npz(input_file, out_file)
    else:
        raise ValueError(f"Unsupported input file extension: {input_file}")

    return out_file
