import numpy as np
import random

from periodictable import elements

from orchestr_ai.utils.analysis import compute_rmsd_matrix
from orchestr_ai.utils.io import save_xyz

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


def compute_and_scale_delta_properties(cfg, target_file):
    """
    Computes atomic baselines for E_singlet and Delta_E, determines standard
    deviations and scaling factors k_E and k_F, scales Delta_E and f_delta targets,
    saves the scaled dataset to an XYZ file, and saves scale metadata to JSON.
    """
    import os
    import json
    import ase.io

    ds = cfg.get("dataset", {})
    prefix = ds.get("output_prefix", "dataset_scaled")
    spin_delta_cfg = ds.get("spin_delta_scaling", {})

    base_energy_key = spin_delta_cfg.get("base_energy_key", "E_singlet")
    base_forces_key = spin_delta_cfg.get("base_forces_key", "f_singlet")
    target_energy_key = spin_delta_cfg.get("target_energy_key", "E_triplet")
    target_forces_key = spin_delta_cfg.get("target_forces_key", "f_triplet")
    output_meta_json = spin_delta_cfg.get("output_meta_json", "mace_scale_metadata.json")

    output_file = f"{prefix}_scaled.xyz"

    logger.info(f"Reading target XYZ dataset from: {target_file}")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Target dataset file not found: {target_file}")

    frames = ase.io.read(target_file, index=":")
    if not frames:
        raise ValueError(f"No frames found or read from: {target_file}")

    num_frames = len(frames)
    logger.info(f"Loaded {num_frames} frames for delta-scaling.")

    # Extract unique atomic numbers to build baseline regression matrix
    frames_atomic_numbers = [atoms.get_atomic_numbers() for atoms in frames]
    all_elements = set()
    for list_z in frames_atomic_numbers:
        all_elements.update(list_z)
    unique_elements = sorted(list(int(x) for x in all_elements))
    num_elements = len(unique_elements)

    logger.info(f"Unique elements identified for baseline fitting: {unique_elements}")

    # 1. Regression Matrix A: counts of elements in each frame
    A = np.zeros((num_frames, num_elements), dtype=np.float64)
    for i, list_z in enumerate(frames_atomic_numbers):
        for z in list_z:
            idx = unique_elements.index(z)
            A[i, idx] += 1.0

    # 2. Extract singlet and triplet energies, compute Delta_E
    E_singlet = []
    E_triplet = []
    for idx, atoms in enumerate(frames):
        if base_energy_key not in atoms.info:
            raise KeyError(f"Frame {idx} is missing base_energy_key '{base_energy_key}' in info.")
        if target_energy_key not in atoms.info:
            raise KeyError(f"Frame {idx} is missing target_energy_key '{target_energy_key}' in info.")
        E_singlet.append(atoms.info[base_energy_key])
        E_triplet.append(atoms.info[target_energy_key])

    E_singlet = np.array(E_singlet, dtype=np.float64)
    E_triplet = np.array(E_triplet, dtype=np.float64)
    Delta_E = E_singlet - E_triplet

    # 3. Fit linear atomic baseline regression for E_singlet
    # E_singlet ~ A * E0_singlet
    E0_singlet_coeffs, _, _, _ = np.linalg.lstsq(A, E_singlet, rcond=None)
    E_singlet_baseline = A @ E0_singlet_coeffs
    E_singlet_residual = E_singlet - E_singlet_baseline

    # 4. Fit linear atomic baseline regression for Delta_E
    # Delta_E ~ A * E0_delta
    E0_delta_coeffs, _, _, _ = np.linalg.lstsq(A, Delta_E, rcond=None)
    Delta_E_baseline = A @ E0_delta_coeffs
    Delta_E_residual = Delta_E - Delta_E_baseline

    # 5. Compute standard deviations of baseline-subtracted energy components
    std_singlet = np.std(E_singlet_residual, ddof=0)
    std_delta = np.std(Delta_E_residual, ddof=0)

    logger.info(f"Energy STD (singlet residual): {std_singlet:.6f} eV")
    logger.info(f"Energy STD (delta residual): {std_delta:.6f} eV")

    if std_delta < 1e-12:
        logger.warning("Warning: Standard deviation of delta energy residual is near-zero. k_E set to 1.0.")
        k_E = 1.0
    else:
        k_E = std_singlet / std_delta

    # 6. Extract singlet and triplet forces, compute F_delta
    all_F_singlet = []
    all_F_triplet = []
    for idx, atoms in enumerate(frames):
        if base_forces_key not in atoms.arrays:
            raise KeyError(f"Frame {idx} is missing base_forces_key '{base_forces_key}' in arrays.")
        if target_forces_key not in atoms.arrays:
            raise KeyError(f"Frame {idx} is missing target_forces_key '{target_forces_key}' in arrays.")
        all_F_singlet.append(atoms.arrays[base_forces_key])
        all_F_triplet.append(atoms.arrays[target_forces_key])

    F_singlet_concat = np.concatenate(all_F_singlet, axis=0).astype(np.float64)
    F_triplet_concat = np.concatenate(all_F_triplet, axis=0).astype(np.float64)
    F_delta_concat = F_singlet_concat - F_triplet_concat

    # 7. Compute standard deviations of force components
    std_F_singlet = np.std(F_singlet_concat, ddof=0)
    std_F_delta = np.std(F_delta_concat, ddof=0)

    logger.info(f"Forces STD (singlet): {std_F_singlet:.6f} eV/A")
    logger.info(f"Forces STD (delta): {std_F_delta:.6f} eV/A")

    if std_F_delta < 1e-12:
        logger.warning("Warning: Standard deviation of F_delta is near-zero. k_F set to 1.0.")
        k_F = 1.0
    else:
        k_F = std_F_singlet / std_F_delta

    logger.info(f"Computed scaling factors: k_E = {k_E:.12f}, k_F = {k_F:.12f}")

    # 8. Scale and update properties for each frame
    for idx, atoms in enumerate(frames):
        # Scale energy
        e_singlet_val = atoms.info[base_energy_key]
        e_triplet_val = atoms.info[target_energy_key]
        delta_e_true = e_singlet_val - e_triplet_val
        delta_e_scaled = k_E * delta_e_true

        # Save to Delta_E key
        atoms.info["Delta_E"] = delta_e_scaled

        # Scale forces
        f_singlet_arr = atoms.arrays[base_forces_key].astype(np.float64)
        f_triplet_arr = atoms.arrays[target_forces_key].astype(np.float64)
        f_delta_true = f_singlet_arr - f_triplet_arr
        f_delta_scaled = k_F * f_delta_true

        # Save to f_delta key (appended to arrays)
        atoms.arrays["f_delta"] = f_delta_scaled

    # 9. Save the scaled XYZ dataset
    logger.info(f"Saving scaled XYZ dataset to: {output_file}")
    ase.io.write(output_file, frames, format="extxyz")

    # 10. Save scaling metadata to JSON
    metadata = {
        "k_E": float(k_E),
        "k_F": float(k_F),
        "base_head": "singlet",
        "delta_head": "delta",
        "base_energy_key": base_energy_key,
        "target_energy_key": target_energy_key,
        "base_forces_key": base_forces_key,
        "target_forces_key": target_forces_key,
        "delta_energy_key": "Delta_E",
        "delta_forces_key": "f_delta",
        "unique_elements": unique_elements,
        "E0_singlet": E0_singlet_coeffs.tolist(),
        "E0_delta": E0_delta_coeffs.tolist(),
        "input_file": target_file,
        "output_file": output_file
    }

    logger.info(f"Saving scaling metadata to: {output_meta_json}")
    with open(output_meta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Delta properties computation and scaling successful.")
