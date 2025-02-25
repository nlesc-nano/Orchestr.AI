import numpy as np
import pickle 
import random
import matplotlib.pyplot as plt
import yaml 
import pprint
import argparse
from pathlib import Path
from periodictable import elements
from scipy.spatial.transform import Rotation as R
from scm.plams import Molecule
from CAT.recipes import replace_surface
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

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
    - This function always writes to 'data/processed/filename'.
    - If 'energies' is provided, each frame's comment line includes that frame's energy.
    - You can use 'comment' to clarify if the frames are "Aligned positions", "Aligned forces", etc.
    """

    # Determine processed output directory
    processed_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Build the full path
    output_path = processed_dir / filename

    # Convert frames to a NumPy array if needed
    frames = np.asarray(positions)
    num_frames = len(frames)
    num_atoms = len(atom_types)
    has_energies = (energies is not None) and (len(energies) == num_frames)

    print(f"Saving XYZ data to: {output_path}")
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

    print(f"Done. Wrote {num_frames} frames to '{output_path}'.")
