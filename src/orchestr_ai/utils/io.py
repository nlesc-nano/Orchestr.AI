import numpy as np
from pathlib import Path
import os
import re

import logging
logger = logging.getLogger(__name__)

def save_xyz(filename, positions, atom_types, energies=None, comment="Frame"):
    """
    Save atomic positions to an XYZ file in 'data/processed'.

    Parameters
    ----------
    filename : str
        The output filename (e.g. 'aligned_positions.xyz').
    frames : (num_frames, num_atoms, 3) array-like
        Atomic positions or forces for each frame.
    atom_types : list of str
        The atomic symbols corresponding to each atom (e.g., ["Cs", "Br", ...]).
    energies : list[float] or None, optional
        If provided, each frame's energy is appended to the comment line.
        Must match the number of frames if given.
    comment : str, optional
        A custom label for the comment line. Defaults to "Frame".

    Notes
    -----
    - This function always writes to 'processed_data/filename'.
    - If 'energies' is provided, each frame's comment line includes that frame's energy.
    - You can use 'comment' to clarify if the frames are "Aligned positions", "Aligned forces", etc.
    """

    # Determine processed output directory
    processed_dir = Path.cwd() / "processed_data"

    # Build the full path
    output_path = processed_dir / filename
        
    # Convert frames to a NumPy array if needed
    frames = np.asarray(positions)
    num_frames = len(frames)
    num_atoms = len(atom_types)
    has_energies = (energies is not None) and (len(energies) == num_frames)

    logger.info(f"Saving XYZ data to: {output_path}")
    with open(output_path, "w") as f:
        for i, frame in enumerate(frames):
            f.write(f"{num_atoms}\n")

            # Construct the comment line
            comment_line = f"{comment} {i+1}"
            if has_energies and energies[i] is not None:
                comment_line += f", Energy = {energies[i]:.6f} eV"

            f.write(comment_line + "\n")

            # Write each atom line
            for atom, (x, y, z) in zip(atom_types, frame):
                f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")

    logger.info(f"Done. Wrote {num_frames} frames to '{output_path}'.")

def reorder_xyz_trajectory(input_file, output_file, num_atoms):
    """Reorder atoms in the XYZ trajectory."""
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / output_file

    logger.info(f"Reordering atoms in trajectory file: {input_file}")

    with open(input_file, "r") as infile, open(output_path, "w") as outfile:
        lines = infile.readlines()
        num_lines_per_frame = num_atoms + 2
        for i in range(0, len(lines), num_lines_per_frame):
            header = lines[i:i + 2]
            atom_lines = lines[i + 2:i + 2 + num_atoms]
            sorted_atoms = sorted(atom_lines, key=lambda x: x.split()[0])
            outfile.writelines(header)
            outfile.writelines(sorted_atoms)
    
    logger.info(f"Reordered trajectory saved to: {output_path}")

def parse_positions_xyz(filename, num_atoms):
    """
    Parse positions from an XYZ file.

    Parameters:
        filename (str): Path to the XYZ file.
        num_atoms (int): Number of atoms in each frame.

    Returns:
        np.ndarray: Atomic positions (num_frames, num_atoms, 3).
        list: Atomic types.
        list: Total energies for each frame (if available; otherwise, empty list).
    """
    logger.info(f"Parsing positions XYZ file: {filename}")
    positions = []
    atom_types = []
    total_energies = []

    with open(filename, "r") as f:
        lines = f.readlines()
        num_lines_per_frame = num_atoms + 2  # 2 lines for header and comment

        for i in range(0, len(lines), num_lines_per_frame):
            atom_lines = lines[i + 2:i + 2 + num_atoms]
            comment_line = lines[i + 1]

            # Try parsing the energy; otherwise, skip
            try:
                total_energy = float(comment_line.split("=")[-1].strip())
                total_energies.append(total_energy)
            except ValueError:
                total_energies.append(None)  # Placeholder for missing energy

            frame_positions = []
            for line in atom_lines:
                parts = line.split()
                atom_types.append(parts[0])
                frame_positions.append([float(x) for x in parts[1:4]])

            positions.append(frame_positions)

    return np.array(positions), atom_types[:num_atoms], total_energies

def parse_forces_xyz(filename, num_atoms):
    """Parse forces from an XYZ file."""
    logger.info(f"Parsing forces XYZ file: {filename}")
    forces = []
    with open(filename, "r") as f:
        lines = f.readlines()
        num_lines_per_frame = num_atoms + 2
        for i in range(0, len(lines), num_lines_per_frame):
            frame_forces = []
            for j in range(2, 2 + num_atoms):
                parts = lines[i + j].split()
                frame_forces.append(list(map(float, parts[1:4])))
            forces.append(frame_forces)
    
    return np.array(forces)

def get_num_atoms(filename):
    """
    Retrieve the number of atoms from the first line of an XYZ file.

    Parameters:
        filename (str): Path to the XYZ file.

    Returns:
        int: Number of atoms in the structure.
    """
    with open(filename, "r") as f:
        num_atoms = int(f.readline().strip())
    
    logger.info(f"Number of atoms: {num_atoms}")
    
    return num_atoms
    
    

# Newer consolidated I/O helpers (moved from consolidate_ter.py)

def save_to_npz(
    filename: str,
    atomic_numbers: np.ndarray,          # (n_atoms,)  or (n_frames,n_atoms)
    positions:      np.ndarray,          # (N, n_atoms, 3)
    energies:       np.ndarray,          # (N,)         or list-like
    forces:         np.ndarray,          # (N, n_atoms, 3)
    cells:  np.ndarray = None,
    pbc:    np.ndarray = None,
):
    """
    Save a dataset exactly like your legacy exporter, but guarantee that
    every E[i] is a 1-element float-64 array so torch can infer dtype.
    """
    N, A, _ = positions.shape

    # numeric arrays
    R = np.asarray(positions, dtype=np.float32)          # (N,A,3)
    F = np.asarray(forces,    dtype=np.float32)          # (N,A,3)

    # Energies: (N,1) float64  →  row.data['E'] is 1-D, not scalar
    E = np.asarray(energies, dtype=np.float64)

    # Atomic numbers: 1-D (A,)
    z = np.asarray(atomic_numbers, dtype=np.int32)
    if z.ndim == 2:
        z = z[0]                 # order is identical, keep first row
    if z.ndim != 1 or z.size != A:
        raise ValueError(f"atomic_numbers must be 1-D of length {A}, got {z.shape}")

    # assemble dict 
    base = {
        "type": "dataset",
        "name": os.path.splitext(os.path.basename(filename))[0],
        "R":    R,
        "z":    z,
        "E":    E,      # (N,1) float64  ← key point
        "F":    F,
        "F_min":  float(F.min()),  "F_max":  float(F.max()),
        "F_mean": float(F.mean()), "F_var":  float(F.var()),
        "E_min":  float(E.min()),  "E_max":  float(E.max()),
        "E_mean": float(E.mean()), "E_var":  float(E.var()),
    }
    if cells is not None: base["lattice"] = np.asarray(cells, dtype=np.float32)
    if pbc   is not None: base["pbc"]     = np.asarray(pbc,   dtype=bool)

    np.savez_compressed(filename, **base)

    logger.info(f"[I/O] Saved {filename}")
    logger.info(f"      R {R.shape}, z {z.shape}, E {E.shape}, F {F.shape}")


def parse_stacked_xyz(filename):
    """
    Parse stacked XYZ returning (energies, positions, forces, atom_types).
    """
    energies, positions, forces, atom_types = [], [], [], []
    with open(filename,'r') as f:
        lines = f.readlines()
    idx = 0
    while idx < len(lines):
        n = int(lines[idx].strip()); idx+=1
        e = float(lines[idx].split()[0]); idx+=1
        fr_pos, fr_for = [], []
        if not atom_types:
            for i in range(n):
                parts = lines[idx].split()
                atom_types.append(parts[0])
                fr_pos.append([float(x) for x in parts[1:4]])
                fr_for.append([float(x) for x in parts[4:7]])
                idx+=1
        else:
            for i in range(n):
                parts = lines[idx].split()
                fr_pos.append([float(x) for x in parts[1:4]])
                fr_for.append([float(x) for x in parts[4:7]])
                idx+=1
        energies.append(e)
        positions.append(fr_pos)
        forces.append(fr_for)
    return (np.array(energies),
            np.array(positions),
            np.array(forces),
            atom_types)
            
# def save_stacked_xyz(filename, energies, positions, forces, atom_types):
#     num_frames, num_atoms, _ = positions.shape
#     with open(filename,'w') as f:
#         for i in range(num_frames):
#             f.write(f"{num_atoms}\n")
#             f.write(f"{energies[i]:.6f}\n")
#             for atom,(x,y,z),(fx,fy,fz) in zip(atom_types, positions[i], forces[i]):
#                 f.write(f"{atom:<2} {x:12.6f} {y:12.6f} {z:12.6f}"
#                         f" {fx:12.6f} {fy:12.6f} {fz:12.6f}\n")


def save_stacked_xyz(filename, E, P, F, atoms, spin_state="single", E_s=None, E_t=None, dE=None, F_s=None, F_t=None):
    """
    Saves geometries, energies, and forces to a stacked XYZ file.
    Supports single state or dual spin state (MACE extxyz format).
    """
    with open(filename, "w") as f:
        for i in range(len(E)):
            f.write(f"{len(atoms)}\n")
            
            if spin_state == "dual":
                header = (f'Lattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" '
                          f'Properties=species:S:1:pos:R:3:f_singlet:R:3:f_triplet:R:3 '
                          f'config_type=Default pbc="F F F" '
                          f'E_singlet={E_s[i]:.12f} E_triplet={E_t[i]:.12f} Delta_E={dE[i]:.12f}\n')
                f.write(header)
                for j, atom in enumerate(atoms):
                    pos = P[i, j]
                    fs, ft = F_s[i, j], F_t[i, j]
                    f.write(f"{atom} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                            f"{fs[0]:.6f} {fs[1]:.6f} {fs[2]:.6f} "
                            f"{ft[0]:.6f} {ft[1]:.6f} {ft[2]:.6f}\n")
            else:
                f.write(f"{E[i]:.6f}\n")
                for atom, (x, y, z), (fx, fy, fz) in zip(atoms, P[i], F[i]):
                    f.write(f"{atom:<2} {x:12.6f} {y:12.6f} {z:12.6f}"
                            f" {fx:12.6f} {fy:12.6f} {fz:12.6f}\n")


def parse_dual_spin_xyz(filepath):
    """Parses 10-column dual-state XYZ into numpy arrays."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    E_s_list, E_t_list, dE_list, P_list, F_s_list, F_t_list = [], [], [], [], [], []
    atoms = None
    idx = 0
    # Pre-compile regexes to safely parse energies supporting both '=' and ':' anywhere in comment line
    e_s_re = re.compile(r"E_singlet[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    e_t_re = re.compile(r"E_triplet[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    de_re  = re.compile(r"Delta_E[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    
    while idx < len(lines):
        if not lines[idx].strip():
            idx += 1
            continue
            
        natoms = int(lines[idx].strip())
        comment = lines[idx+1].strip()
        
        e_s_match = e_s_re.search(comment)
        e_t_match = e_t_re.search(comment)
        de_match  = de_re.search(comment)
        
        if not e_s_match or not e_t_match:
            raise ValueError(f"Could not parse singlet/triplet energies from comment line: {comment}")
            
        e_s = float(e_s_match.group(1))
        e_t = float(e_t_match.group(1))
        de = float(de_match.group(1)) if de_match else (e_s - e_t)
        
        E_s_list.append(e_s); E_t_list.append(e_t); dE_list.append(de)
        
        current_P, current_Fs, current_Ft, current_atoms = [], [], [], []
        for i in range(natoms):
            parts = lines[idx + 2 + i].split()
            current_atoms.append(parts[0])
            current_P.append([float(x) for x in parts[1:4]])
            current_Fs.append([float(x) for x in parts[4:7]])
            current_Ft.append([float(x) for x in parts[7:10]])
            
        P_list.append(current_P); F_s_list.append(current_Fs); F_t_list.append(current_Ft)
        if atoms is None: atoms = current_atoms 
        idx += natoms + 2
        
    return (np.array(E_s_list), np.array(E_t_list), np.array(dE_list), 
            np.array(P_list), np.array(F_s_list), np.array(F_t_list), atoms)