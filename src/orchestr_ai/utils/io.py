import numpy as np
from pathlib import Path
import os

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
            
def save_stacked_xyz(filename, energies, positions, forces, atom_types):
    num_frames, num_atoms, _ = positions.shape
    with open(filename,'w') as f:
        for i in range(num_frames):
            f.write(f"{num_atoms}\n")
            f.write(f"{energies[i]:.6f}\n")
            for atom,(x,y,z),(fx,fy,fz) in zip(atom_types, positions[i], forces[i]):
                f.write(f"{atom:<2} {x:12.6f} {y:12.6f} {z:12.6f}"
                        f" {fx:12.6f} {fy:12.6f} {fz:12.6f}\n")