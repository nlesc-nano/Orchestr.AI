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

from mlff_qd.utils.io import save_xyz

import logging
logger = logging.getLogger(__name__)

def center_positions(positions, masses):
    """
    Center atomic positions by translating the center of mass (COM) to the origin.

    Parameters:
        positions (np.ndarray): Atomic positions (num_frames, num_atoms, 3).
        masses (np.ndarray): Atomic masses (num_atoms).

    Returns:
        np.ndarray: Centered atomic positions (num_frames, num_atoms, 3).
    """
    num_frames, num_atoms, _ = positions.shape
    com = np.zeros((num_frames, 3))
    
    for i in range(num_frames):
        com[i] = np.sum(positions[i] * masses[:, None], axis=0) / masses.sum()
    
    return positions - com[:, None, :]

def align_to_reference(positions, reference):
    """Align each frame to the reference using SVD."""
    num_frames = positions.shape[0]
    aligned = np.zeros_like(positions)
    rotations = np.zeros((num_frames, 3, 3))
    
    for i, frame in enumerate(positions):
        H = frame.T @ reference
        U, _, Vt = np.linalg.svd(H)
        Rmat = U @ Vt
        rotations[i] = Rmat
        aligned[i] = frame @ Rmat.T
    
    return aligned, rotations

def rotate_forces(forces, rotation_matrices):
    """Rotate forces using the corresponding rotation matrices."""
    rotated = np.zeros_like(forces)
    for i, frame in enumerate(forces):
        rotated[i] = frame @ rotation_matrices[i].T
    
    return rotated

def create_mass_dict(atom_types):
    """
    Create a dictionary mapping atom types to their atomic masses.

    Parameters:
        atom_types (list): List of atomic types as strings.

    Returns:
        dict: Dictionary where keys are atom types and values are atomic masses.
    """
    mass_dict = {atom: elements.symbol(atom).mass for atom in set(atom_types)}
    logger.info(f"Generated mass dictionary: {mass_dict}")
    return mass_dict

def generate_randomized_samples(
    md_positions,
    atom_types,
    num_samples=100,
    base_scale=0.1
):
    """
    Generate random configurations by applying Gaussian perturbations to atomic positions.
    The displacements are fixed in scale and not adjusted to match any RMSD target.

    Parameters:
        md_positions (list of np.ndarray): List of MD frames (num_frames, num_atoms, 3).
        atom_types (list of str): Atom types for each atom in the structure.
        num_samples (int): Number of randomized configurations to generate.
        base_scale (float): Standard deviation of the Gaussian noise applied to displace each atom.

    Returns:
        np.ndarray: Array of randomized configurations (num_samples, num_atoms, 3).
    """
    randomized_structures = []

    for i in range(num_samples):
        # Select a random reference frame
        start_idx = np.random.choice(len(md_positions))
        reference_positions = md_positions[start_idx]

        # Generate random displacements
        displacement = np.random.normal(loc=0.0, scale=base_scale, size=reference_positions.shape)

        # Remove net translation
        mean_disp = np.mean(displacement, axis=0)  # Average over all atoms
        displacement -= mean_disp  # Now no net translation

        # Apply displacements to the reference positions
        randomized_structure = reference_positions + displacement
        randomized_structures.append(randomized_structure)

        # Print progress every 100 samples or for the first sample
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Generating sample {i + 1}/{num_samples}...")

    print(f"Generated {len(randomized_structures)} randomized samples.")

    # Save the randomized samples
    save_xyz("randomized_samples.xyz", randomized_structures, atom_types)
    print(f"Saved {len(randomized_structures)} randomized samples to 'randomized_samples.xyz'.")

    return np.array(randomized_structures)

def iterative_alignment_fixed(centered_positions, tol=1e-6, max_iter=10):
    """Iteratively align positions to a converged reference."""
    ref = centered_positions[0]
    prev_ref = None
    
    for _ in range(max_iter):
        aligned, rotations = align_to_reference(centered_positions, ref)
        new_ref = np.mean(aligned, axis=0)
        if prev_ref is not None and np.linalg.norm(new_ref - prev_ref) < tol:
            break
        prev_ref = new_ref
        ref = new_ref
    
    return aligned, rotations, new_ref
