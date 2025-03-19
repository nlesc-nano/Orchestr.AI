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
    com_frames = np.zeros((num_frames, 3))

    # Calculate COM for each frame
    for frame_idx in range(num_frames):
        com_frames[frame_idx] = np.sum(
            positions[frame_idx] * masses[:, np.newaxis], axis=0
        ) / masses.sum()

    # Subtract COM from all positions
    centered_positions = positions - com_frames[:, np.newaxis, :]
    return centered_positions

def align_to_reference(positions, reference_positions):
    """
    Align atomic positions of each frame to a reference structure using least-squares fitting.

    Parameters:
        positions (np.ndarray): Atomic positions (num_frames, num_atoms, 3).
        reference_positions (np.ndarray): Reference positions (num_atoms, 3).

    Returns:
        np.ndarray: Aligned atomic positions (num_frames, num_atoms, 3).
    """
    aligned_positions = np.zeros_like(positions)
    for frame_idx, frame in enumerate(positions):
        # Calculate optimal rotation matrix
        H = frame.T @ reference_positions
        U, _, Vt = np.linalg.svd(H)
        rotation_matrix = U @ Vt

        # Apply rotation to align the frame
        aligned_positions[frame_idx] = frame @ rotation_matrix.T
    
    return aligned_positions

def rotate_forces(forces, rotation_matrices):
    """
    Rotate forces for each frame to match the aligned orientation.

    Parameters:
        forces (np.ndarray): Atomic forces (num_frames, num_atoms, 3).
        rotation_matrices (np.ndarray): Rotation matrices (num_frames, 3, 3).

    Returns:
        np.ndarray: Rotated forces (num_frames, num_atoms, 3).
    """
    rotated_forces = np.zeros_like(forces)
    for frame_idx, frame_forces in enumerate(forces):
        rotated_forces[frame_idx] = frame_forces @ rotation_matrices[frame_idx]
    
    return rotated_forces

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
