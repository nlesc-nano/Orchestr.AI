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

from mlff_qd.utils.analysis import compute_rmsd_matrix
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

def generate_randomized_samples(md_positions, atom_types, num_samples=100, base_scale=0.1):
    """Generate random structures by Gaussian perturbation."""
    randomized = []
    
    for i in range(num_samples):
        ref = random.choice(md_positions)
        disp = np.random.normal(0, base_scale, size=ref.shape)
        disp -= np.mean(disp, axis=0)
        randomized.append(ref + disp)
    
        if (i+1) % 100 == 0 or i==0:
            logger.info(f"Generated {i+1}/{num_samples} randomized samples...")
    
    save_xyz("randomized_samples.xyz", randomized, atom_types)
    
    logger.info("Saved randomized samples to 'randomized_samples.xyz'")
    
    return np.array(randomized)

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

def find_medoid_structure(aligned_positions):
    """Find the medoid structure from aligned positions."""
    rmsd_mat = compute_rmsd_matrix(aligned_positions)
    mean_rmsd = np.mean(rmsd_mat, axis=1)
    idx = np.argmin(mean_rmsd)
    return aligned_positions[idx], idx
