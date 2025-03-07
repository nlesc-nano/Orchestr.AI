#!/usr/bin/env python3
"""
Reworked MLFF dataset generation script.

Key improvements:
• Redundant functions removed.
• All generated plots are saved to files.
"""

import os
import pickle
import random
import yaml
import pprint
import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from scipy.spatial.distance import cdist
from dscribe.descriptors import SOAP
from ase import Atoms
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from periodictable import elements
from scm.plams import Molecule
from CAT.recipes import replace_surface

from mlff_qd.utils.config import load_config
from mlff_qd.utils.io import save_xyz, reorder_xyz_trajectory

# --- Set up logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("mlff_dataset.log")])
logger = logging.getLogger(__name__)

# --- Constants ---
hartree_bohr_to_eV_angstrom = 51.42208619083232
hartree_to_eV = 27.211386245988
bohr_to_angstrom = 0.529177210903
amu_to_kg = 1.66053906660e-27
c = 2.99792458e10

# --- Utility Functions ---
def compute_global_distance_fluctuation_cdist(md_positions):
    """
    Compute the standard deviation of interatomic distances for each frame using scipy's cdist.
    Returns an array of shape (num_frames,) where each entry is the std. dev. for that frame.
    
    Parameters:
        md_positions (np.ndarray): shape (num_frames, num_atoms, 3)
                                   Each frame is (num_atoms, 3)

    Returns:
        np.ndarray: shape (num_frames,) with distance fluctuation for each frame
    """
    fluctuations = []
    for frame in md_positions:
        # cdist computes pairwise distances among points
        dists = cdist(frame, frame)  # shape => (num_atoms, num_atoms)
        
        # Extract upper triangle to skip diagonal (0) and duplicates
        tri_idx = np.triu_indices(dists.shape[0], k=1)
        upper_dists = dists[tri_idx]
        
        # Compute standard deviation
        std_dev = np.std(upper_dists)
        fluctuations.append(std_dev)
    
    return np.array(fluctuations)


def compute_surface_indices_with_replace_surface_dynamic(
    input_file, surface_atom_types, f=1.0, surface_replaced_file="surface_replaced.xyz"
):
    """
    Dynamically replace surface atoms with random elements from the periodic table that are not in the molecule.

    Parameters:
        input_file (str): Path to the input .xyz file.
        surface_atom_types (list): List of atom types considered as surface atoms to be replaced.
        f (float): Fraction of surface atoms to replace (default: 1.0).
        surface_replaced_file (str): Path to the output .xyz file with surface atoms replaced.

    Returns:
        surface_indices (list): Indices of the replaced surface atoms in the structure.
        replaced_atom_types (list): Updated atom types after replacement.
    """
    import random
    from periodictable import elements

    print(f"Reading molecule from {input_file}...")
    mol_original = Molecule(input_file)  # Load the original molecule
    mol_updated = mol_original.copy()  # Create a copy to track cumulative changes

    # Identify atom types in the molecule
    molecule_atom_types = {atom.symbol for atom in mol_original}

    # Generate a list of replacement elements from the periodic table
    available_elements = [el.symbol for el in elements if el.symbol not in molecule_atom_types]

    # Prepare replacements dynamically
    replacements = [
        (atom_type, random.choice(available_elements)) for atom_type in surface_atom_types
    ]
    print(f"Dynamic replacements: {replacements}")

    surface_indices = []  # Collect indices of replaced surface atoms
    for i, (original_symbol, replacement_symbol) in enumerate(replacements):
        print(f"Replacing surface atoms: {original_symbol} -> {replacement_symbol} (f={f})...")

        # Create a new molecule for this replacement
        mol_new = replace_surface(mol_updated, symbol=original_symbol, symbol_new=replacement_symbol, f=f)

        # Update `mol_updated` to incorporate the changes
        mol_updated = mol_new.copy()

        # Identify the replaced atoms in the molecule
        for idx, atom in enumerate(mol_new):
            if atom.symbol == replacement_symbol:
                surface_indices.append(idx)

        print(f"Replacement {i+1}: {len(surface_indices)} surface atoms replaced so far.")

    # Save the final updated molecule to the output file
    print(f"Writing modified molecule with replacements to {surface_replaced_file}...")
    mol_updated.write(surface_replaced_file)

    # Extract updated atom types
    replaced_atom_types = [atom.symbol for atom in mol_updated]

    print(f"Surface replacements completed. {len(surface_indices)} surface atoms identified and replaced.")
    return surface_indices, replaced_atom_types

def generate_pca_samples_in_pca_space(ref_descriptor, pca, n_samples, scaling_factor):
    """
    Given a descriptor in the *normalized* space (the same dimension PCA was fit on),
    return random samples in PCA space that we can transform back outside this function.

    Steps:
    1. transform => PCA coords
    2. random perturbation => new PCA coords
    3. Return the new PCA coords directly (no inverse_transform here)
    """
    # Convert descriptor to PCA space
    ref_coeffs = pca.transform(ref_descriptor.reshape(1, -1))[0]
    new_samples_pca = []
    for _ in range(n_samples):
        # Add random noise in PCA space
        perturbation = np.random.normal(scale=scaling_factor, size=ref_coeffs.shape)
        new_coeffs = ref_coeffs + perturbation
        new_samples_pca.append(new_coeffs)
    return np.array(new_samples_pca)  # shape: (n_samples, pca_dim)


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
    print(f"Parsing positions XYZ file: {filename}")
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
    logger.info(f"Parsing forces XYZ file: {filename}")
    forces = []
    with open(filename, "r") as f:
        lines = f.readlines()
        step = num_atoms + 2
        for i in range(0, len(lines), step):
            frame = []
            for line in lines[i+2:i+2+num_atoms]:
                parts = line.split()
                frame.append(list(map(float, parts[1:4])))
            forces.append(frame)
    return np.array(forces)

def get_num_atoms(filename):
    """Return number of atoms from the first line of an XYZ file."""
    with open(filename, "r") as f:
        num = int(f.readline().strip())
    logger.info(f"Number of atoms: {num}")
    return num

def create_mass_dict(atom_types):
    """Return a dict mapping atom types to their masses."""
    mass_dict = {atom: elements.symbol(atom).mass for atom in set(atom_types)}
    logger.info(f"Mass dictionary: {mass_dict}")
    return mass_dict

def center_positions(positions, masses):
    """Center positions by subtracting the center-of-mass."""
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

def rotate_forces(forces, rotation_matrices):
    """Rotate forces using the corresponding rotation matrices."""
    rotated = np.zeros_like(forces)
    for i, frame in enumerate(forces):
        rotated[i] = frame @ rotation_matrices[i].T
    return rotated

def find_medoid_structure(aligned_positions):
    """Find the medoid structure from aligned positions."""
    rmsd_mat = compute_rmsd_matrix(aligned_positions)
    mean_rmsd = np.mean(rmsd_mat, axis=1)
    idx = np.argmin(mean_rmsd)
    return aligned_positions[idx], idx

def compute_rmsd_matrix(structures1, structures2=None):
    """Compute RMSD matrix between two sets of structures."""
    structures1 = np.array(structures1)
    if structures2 is None:
        structures2 = structures1
        symmetric = True
    else:
        structures2 = np.array(structures2)
        symmetric = False
    N1, A, _ = structures1.shape
    N2 = structures2.shape[0]
    if symmetric:
        rmsd_mat = np.zeros((N1, N2))
        for i in range(N1):
            diff = structures1[i:] - structures1[i]
            rmsd_vals = np.sqrt(np.sum(diff**2, axis=(1,2)) / A)
            rmsd_mat[i, i:] = rmsd_vals
            rmsd_mat[i:, i] = rmsd_vals
    else:
        diff = structures1[:, None, :, :] - structures2[None, :, :, :]
        rmsd_mat = np.sqrt(np.sum(diff**2, axis=(2,3)) / A)
    return rmsd_mat

def plot_rmsd_histogram(rmsd_matrix, bins=50, title="RMSD Histogram", xlabel="RMSD (Å)", ylabel="Frequency", savefile="rmsd_histogram.png"):
    """Plot and save RMSD histogram."""
    rmsd_vals = rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)]
    plt.figure(figsize=(8, 6))
    plt.hist(rmsd_vals, bins=bins, edgecolor="black", alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(savefile)
    logger.info(f"Saved RMSD histogram plot to {savefile}")
    plt.close()

def compute_soap_descriptors(md_positions, atom_types, soap):
    """Compute a global (averaged) SOAP descriptor for each MD frame."""
    descriptors = []
    for pos in md_positions:
        atoms = Atoms(symbols=atom_types, positions=pos)
        soap_desc = soap.create(atoms)
        descriptors.append(np.mean(soap_desc, axis=0))
    return np.array(descriptors)

def cluster_trajectory(descriptor_matrix, method, num_clusters, md_positions, atom_types):
    """Cluster the MD trajectory using the provided descriptor matrix and return representative structures."""
    logger.info("Clustering MD trajectory based on SOAP descriptors...")
    if method == "KMeans":
        model = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = model.fit_predict(descriptor_matrix)
    elif method == "DBSCAN":
        model = DBSCAN(eps=0.2, min_samples=10)
        cluster_labels = model.fit_predict(descriptor_matrix)
    elif method == "GMM":
        model = GaussianMixture(n_components=num_clusters, random_state=42)
        cluster_labels = model.fit_predict(descriptor_matrix)
    else:
        raise ValueError(f"Invalid clustering method: {method}")
    rep_indices = []
    unique = np.unique(cluster_labels)
    for cid in unique:
        if cid == -1:
            continue
        members = np.where(cluster_labels == cid)[0]
        centroid = np.mean(descriptor_matrix[members], axis=0)
        distances = np.linalg.norm(descriptor_matrix[members] - centroid, axis=1)
        rep_idx = members[np.argmin(distances)]
        rep_indices.append(rep_idx)
    logger.info(f"Selected {len(rep_indices)} representatives from {len(unique)} clusters.")
    rep_structs = [md_positions[i] for i in rep_indices]
    save_xyz("clustered_md_sample.xyz", rep_structs, atom_types)
    return rep_structs

def generate_structures_from_pca(
    md_positions,
    md_forces,
    representative_md,
    atom_types,
    num_samples,
    scaling_factor=0.4,
    pca_variance_threshold=0.90
):
    """
    Generate PCA-based structures using combined descriptors of positions and forces ONLY.
    Skips SOAP entirely.

    Parameters:
        md_positions (np.ndarray): MD frames, shape (num_frames, num_atoms, 3)
        md_forces (np.ndarray): MD forces, shape (num_frames, num_atoms, 3)
        representative_md (list of np.ndarray): Subset of frames used as "reference" seeds
        atom_types (list): Atom types
        num_samples (int): How many new structures to generate
        pca_variance_threshold (float): Fraction of variance to keep (e.g. 0.90)

    Returns:
        np.ndarray: Array of shape (num_samples, num_atoms, 3) with the new generated structures.
    """
    logger.info("Generating PCA-based structures from positions+forces only (no SOAP).")

    # 1) Build descriptors (positions and forces flattened)
    #    shape => (num_frames, 6*num_atoms)
    descriptors = []
    for pos, frc in zip(md_positions, md_forces):
        pf = np.concatenate([pos.flatten(), frc.flatten()])  # shape => (6*num_atoms,)
        descriptors.append(pf)
    descriptors = np.array(descriptors)
    logger.info(f"Built descriptor matrix of shape: {descriptors.shape}")

    # 2) Normalize (standardize) each feature
    scaler = StandardScaler()
    descriptors_norm = scaler.fit_transform(descriptors)

    # 3) PCA on normalized descriptors
    pca = PCA()
    pca.fit(descriptors_norm)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    opt_comp = np.argmax(cumvar >= pca_variance_threshold) + 1
    logger.info(f"Optimal PCA components: {opt_comp} (captures {cumvar[opt_comp-1]*100:.2f}% variance)")
    pca = PCA(n_components=opt_comp)
    pca.fit(descriptors_norm)

    # 4) We'll just set scaling_factor = max_displacement directly
    #    If you prefer a distance-based approach, do that here.
    distance_flucts = compute_global_distance_fluctuation_cdist(md_positions)
    mean_dist_fluct = np.mean(distance_flucts)
    max_dist_fluct  = np.max(distance_flucts)
    print(f"Mean distance fluctuation = {mean_dist_fluct:.3f}")
    print(f"Max distance fluctuation  = {max_dist_fluct:.3f}")
    logger.info(f"Sampling amplitude (scaling_factor) set to {scaling_factor:.2f}")

    # 5) Loop over representative frames, generate new structures
    new_structures = []
    for i in range(num_samples):
        idx = random.choice(range(len(representative_md)))
        ref_struct = representative_md[idx]
        ref_index = np.where(np.all(md_positions == ref_struct, axis=(1, 2)))[0][0]

        # Build the reference descriptor (pos + frc), skipping SOAP
        ref_pos = md_positions[ref_index].flatten()
        ref_frc = md_forces[ref_index].flatten()
        ref_desc = np.concatenate([ref_pos, ref_frc])  # shape => (6*num_atoms,)

        # Transform to normalized, PCA space
        ref_desc_norm = scaler.transform(ref_desc.reshape(1, -1))[0]          # shape => (6*N,)
        ref_desc_pca = pca.transform(ref_desc_norm.reshape(1, -1))[0]         # shape => (opt_comp,)

        # Perturb in PCA space
        perturbation = np.random.normal(scale=scaling_factor, size=ref_desc_pca.shape)
        new_desc_pca = ref_desc_pca + perturbation

        # Inverse transform: PCA -> normalized descriptor space
        new_desc_norm = pca.inverse_transform(new_desc_pca.reshape(1, -1))[0] # shape => (6*N,)

        # Un-normalize => original scale
        new_desc = scaler.inverse_transform(new_desc_norm.reshape(1, -1))[0]  # shape => (6*N,)

        # Extract new positions from the first 3*N
        n_atoms = ref_struct.shape[0]
        new_pos = new_desc[:n_atoms * 3].reshape(n_atoms, 3)
        new_structures.append(new_pos)

        # Log progress
        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f"Generated {i+1}/{num_samples} PCA samples...")

    # 6) Convert to NumPy array and save
    new_structures = np.array(new_structures)
    save_xyz("pca_samples_no_soap.xyz", new_structures, atom_types)
    logger.info("Saved PCA-based samples (no SOAP) to 'pca_samples_no_soap.xyz'")

    return new_structures

def generate_surface_core_pca_samples(
    md_positions,
    md_forces,
    atom_types,
    surface_atom_types,
    representative_md,
    num_samples, 
    scaling_surf=0.6, 
    scaling_core=0.4,
    medoid_structure_file="medoid_structure.xyz"
):
    """
    Generate PCA-based samples for surface and core atoms separately,
    but WITHOUT SOAP. Only positions + forces are used.
    
    Parameters:
        md_positions (np.ndarray): MD frames, shape (num_frames, num_atoms, 3).
        md_forces (np.ndarray): MD forces, shape (num_frames, num_atoms, 3).
        atom_types (list): List of atom types (for saving structures).
        surface_atom_types (list): Atom types considered as surface.
        representative_md (list): Subset of frames used as seeds.
        num_samples (int): Number of PCA-based samples to generate.
        medoid_structure_file (str): Path to the structure used to identify surface atoms (default "medoid_structure.xyz").

    Returns:
        np.ndarray: Array of shape (num_samples, num_atoms, 3) with new perturbed structures.
    """
    logger.info("Generating PCA-based surface-core samples (positions+forces only, no SOAP)...")

    # 1) Identify surface and core indices
    logger.info("Identifying surface vs. core atoms...")
    surface_repl_file = "surface_replaced.xyz"
    surface_indices, _ = compute_surface_indices_with_replace_surface_dynamic(
        medoid_structure_file, surface_atom_types, f=1.0, surface_replaced_file=surface_repl_file
    )
    num_atoms = md_positions.shape[1]  # simpler: shape => (num_frames, num_atoms, 3)
    core_indices = np.setdiff1d(np.arange(num_atoms), surface_indices)
    surface_indices = np.array(surface_indices)
    core_indices = np.array(core_indices)
    logger.info(f"surface_indices length={len(surface_indices)}, core_indices length={len(core_indices)}")

    # 2) Build surface and core descriptors using POS+FORCES ONLY
    #    Each descriptor = [positions for those atoms, forces for those atoms]
    surf_desc = []
    core_desc = []
    for pos, frc in zip(md_positions, md_forces):
        surf_d = np.concatenate([pos[surface_indices].flatten(), frc[surface_indices].flatten()])
        core_d = np.concatenate([pos[core_indices].flatten(), frc[core_indices].flatten()])
        surf_desc.append(surf_d)
        core_desc.append(core_d)
    surf_desc = np.array(surf_desc)  # shape => (num_frames, (surface_atoms*3)*2)
    core_desc = np.array(core_desc)  # shape => (num_frames, (core_atoms*3)*2)
    logger.info(f"Surface descriptor shape={surf_desc.shape}, Core descriptor shape={core_desc.shape}")

    # 3) Normalize each descriptor set
    scaler_surf = StandardScaler()
    scaler_core = StandardScaler()
    surf_desc_norm = scaler_surf.fit_transform(surf_desc)
    core_desc_norm = scaler_core.fit_transform(core_desc)

    # 4) PCA on each
    pca_surf_final = PCA()
    pca_surf_final.fit(surf_desc_norm)
    cum_var_s = np.cumsum(pca_surf_final.explained_variance_ratio_)
    opt_s = np.argmax(cum_var_s >= 0.90) + 1
    logger.info(f"Optimal surface PCA components: {opt_s} (captures {cum_var_s[opt_s-1]*100:.2f}% variance)")
    pca_surf_final = PCA(n_components=opt_s)
    pca_surf_final.fit(surf_desc_norm)

    pca_core_final = PCA()
    pca_core_final.fit(core_desc_norm)
    cum_var_c = np.cumsum(pca_core_final.explained_variance_ratio_)
    opt_c = np.argmax(cum_var_c >= 0.90) + 1
    logger.info(f"Optimal core PCA components: {opt_c} (captures {cum_var_c[opt_c-1]*100:.2f}% variance)")
    pca_core_final = PCA(n_components=opt_c)
    pca_core_final.fit(core_desc_norm)

    # 5) Decide on scaling factors for surface/core
    #    We'll just do something simple like:
    #    scaling_surf = min(1.0 * sqrt(Tsurf/T), max_disp)
    #    scaling_core = min(1.0 * sqrt(Tcore/T), max_disp)
    distance_flucts = compute_global_distance_fluctuation_cdist(md_positions)
    mean_dist_fluct = np.mean(distance_flucts)
    max_dist_fluct  = np.max(distance_flucts)
    scaling_surf = 0.6
    scaling_core = 0.4 
    logger.info(f"Surface scaling: {scaling_surf:.2f}, Core scaling: {scaling_core:.2f}")

    new_structures = []
    for i in range(num_samples):
        idx = random.choice(range(len(representative_md)))
        ref_struct = representative_md[idx]
        # find the corresponding index in the full array
        ref_index = np.where(np.all(md_positions == ref_struct, axis=(1,2)))[0][0]
        ref_pos = md_positions[ref_index]
        ref_frc = md_forces[ref_index]

        # Build surface & core descriptors (pos+forces only)
        ref_s_desc = np.concatenate([ref_pos[surface_indices].flatten(), ref_frc[surface_indices].flatten()])
        ref_c_desc = np.concatenate([ref_pos[core_indices].flatten(), ref_frc[core_indices].flatten()])

        # Normalize, transform -> PCA space, add random noise
        ref_s_norm = scaler_surf.transform(ref_s_desc.reshape(1, -1))[0]
        ref_c_norm = scaler_core.transform(ref_c_desc.reshape(1, -1))[0]

        # convert to PCA space
        ref_s_pca = pca_surf_final.transform(ref_s_norm.reshape(1, -1))[0]
        ref_c_pca = pca_core_final.transform(ref_c_norm.reshape(1, -1))[0]

        # add random noise
        perturb_s = np.random.normal(scale=scaling_surf, size=ref_s_pca.shape)
        perturb_c = np.random.normal(scale=scaling_core, size=ref_c_pca.shape)
        new_s_pca = ref_s_pca + perturb_s
        new_c_pca = ref_c_pca + perturb_c

        # inverse PCA
        new_s_norm = pca_surf_final.inverse_transform(new_s_pca.reshape(1, -1))[0]
        new_c_norm = pca_core_final.inverse_transform(new_c_pca.reshape(1, -1))[0]

        # un-normalize
        new_s = scaler_surf.inverse_transform(new_s_norm.reshape(1, -1))[0]
        new_c = scaler_core.inverse_transform(new_c_norm.reshape(1, -1))[0]

        # extract new positions
        num_s_atoms = len(surface_indices)
        new_s_pos = new_s[:num_s_atoms*3].reshape(-1, 3)
        new_c_pos = new_c[:(len(core_indices))*3].reshape(-1, 3)

        # assemble
        combined = ref_struct.copy()
        combined[surface_indices] = new_s_pos
        combined[core_indices] = new_c_pos

        new_structures.append(combined)
        if (i+1) % 100 == 0 or i == 0:
            logger.info(f"Generated {i+1}/{num_samples} surface-core samples...")

    new_structures = np.array(new_structures)
    save_xyz("pca_surface_core_combined_samples_no_soap.xyz", new_structures, atom_types)
    logger.info("Saved PCA-based surface-core samples (no SOAP) to 'pca_surface_core_combined_samples_no_soap.xyz'")
    return new_structures


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

def plot_generated_samples_extended(combined_samples, atom_types, soap):
    """
    Plot PCA visualizations using:
      - Positions (flattened)
      - SOAP descriptors (averaged over atoms)
      - Global distance fluctuation (rotationally invariant metric)
    
    Saves the figure to a file.
    
    Parameters:
        combined_samples (dict): Dictionary with dataset names as keys and
          position arrays as values (shape: (num_samples, num_atoms, 3)).
        atom_types (list): List of atom type labels.
        soap: DScribe SOAP descriptor object.
    """
    from ase import Atoms
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    
    logger.info("Preparing extended PCA plots for positions, SOAP, and global distance fluctuation...")
    
    # --- Panel 1: Positions PCA ---
    flattened_pos = {}
    for name, samples in combined_samples.items():
        samples = np.array(samples)
        if samples.ndim == 3:
            flattened_pos[name] = samples.reshape(samples.shape[0], -1)
        elif samples.ndim == 2:
            flattened_pos[name] = samples
        else:
            raise ValueError(f"Unexpected shape for {name}: {samples.shape}")
    all_pos = np.concatenate(list(flattened_pos.values()), axis=0)
    pos_labels = sum([[name]*len(data) for name, data in flattened_pos.items()], [])
    pca_positions = PCA(n_components=2).fit_transform(all_pos)
    
    # --- Panel 2: SOAP PCA ---
    soap_desc = {}
    for name, samples in combined_samples.items():
        samples = np.array(samples)
        if samples.ndim == 3:
            desc_list = []
            for pos in samples:
                atoms = Atoms(symbols=atom_types, positions=pos)
                d = soap.create(atoms)
                desc_list.append(np.mean(d, axis=0))
            soap_desc[name] = np.array(desc_list)
        else:
            soap_desc[name] = samples
    all_soap = np.concatenate(list(soap_desc.values()), axis=0)
    soap_labels = sum([[name]*len(data) for name, data in soap_desc.items()], [])
    pca_soap = PCA(n_components=2).fit_transform(all_soap)
    
    # --- Panel 3: Global Distance Fluctuation ---
    fluct_dict = {}
    for name, samples in combined_samples.items():
        samples = np.array(samples)
        fluct_dict[name] = compute_global_distance_fluctuation_cdist(samples)
    
    # --- Plotting ---
    unique_labels = sorted(set(pos_labels))
    cmap = {label: plt.cm.tab10(i/len(unique_labels)) for i, label in enumerate(unique_labels)}
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Positions PCA
    for label in unique_labels:
        idxs = [i for i, l in enumerate(pos_labels) if l == label]
        axes[0].scatter(pca_positions[idxs, 0], pca_positions[idxs, 1],
                        label=label, color=cmap[label], alpha=0.7)
    axes[0].set_title("PCA: Positions")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend()
    axes[0].grid(True)
    
    # SOAP PCA
    for label in unique_labels:
        idxs = [i for i, l in enumerate(soap_labels) if l == label]
        axes[1].scatter(pca_soap[idxs, 0], pca_soap[idxs, 1],
                        label=label, color=cmap[label], alpha=0.7)
    axes[1].set_title("PCA: SOAP Descriptor")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()
    axes[1].grid(True)
    
    # Global Distance Fluctuation: boxplot per dataset.
    data_to_plot = [fluct_dict[label] for label in unique_labels]
    axes[2].boxplot(data_to_plot, labels=unique_labels)
    axes[2].set_title("Global Distance Fluctuation")
    axes[2].set_ylabel("Std. deviation of interatomic distances")
    
    plt.tight_layout()
    plt.savefig("extended_pca_plots.png")
    logger.info("Saved extended PCA plots to 'extended_pca_plots.png'")
    plt.close()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLFF Data Generation")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the YAML config file.")
    args = parser.parse_args()
    
    # Load yaml config
    config = load_config(config_file=args.config)

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    # Extract config values
    pos_file_name = config.get("pos_file", "trajectory_pos.xyz")
    frc_file_name = config.get("frc_file", "trajectory_frc.xyz")
    pos_file_path = PROJECT_ROOT / pos_file_name
    frc_file_path = PROJECT_ROOT / frc_file_name

    max_random_displacement = config.get("max_random_displacement", 0.25)
    scaling_factor = config.get("scaling_factor", 0.40)
    scaling_surf = config.get("scaling_surf", 0.60)
    scaling_core = config.get("scaling_core", 0.40) 
    surface_atom_types = config.get("surface_atom_types", ["Cs", "Br"])
    clustering_method = config.get("clustering_method", "KMeans")
    num_clusters = config.get("num_clusters", 100)
    num_samples_pca = config.get("num_samples_pca", 600)
    num_samples_pca_surface = config.get("num_samples_pca_surface", 300)
    num_samples_randomization = config.get("num_samples_randomization", 100)
    soap_params = config["SOAP"]
    logger.info("Configuration:")
    pprint.pprint(config, indent=4, width=80)

    num_atoms = get_num_atoms(pos_file_path)
    positions, atom_types, energies_hartree = parse_positions_xyz(pos_file_path, num_atoms)
    mass_dict = create_mass_dict(atom_types)
    masses = np.array([mass_dict[atom] for atom in atom_types])
    reorder_xyz_trajectory(pos_file_path, "reordered_positions.xyz", num_atoms)
    reorder_xyz_trajectory(frc_file_path, "reordered_forces.xyz", num_atoms)
    positions, atom_types, energies_hartree = parse_positions_xyz("reordered_positions.xyz", num_atoms)
    forces = parse_forces_xyz("reordered_forces.xyz", num_atoms)  # using global parse_forces_xyz if defined

    # Preprocessing
    centered_positions = center_positions(positions, masses)
    aligned_positions, rotations, _ = iterative_alignment_fixed(centered_positions)
    aligned_forces = rotate_forces(forces, rotations)
    medoid_structure, rep_idx = find_medoid_structure(aligned_positions)
    
    save_xyz("medoid_structure.xyz", medoid_structure[np.newaxis,:,:], atom_types)
    save_xyz("aligned_positions.xyz", aligned_positions, atom_types, energies_hartree, comment="Aligned positions")
    save_xyz("aligned_forces.xyz", aligned_forces, atom_types, comment="Aligned forces")

    energies_ev = [e * hartree_to_eV if e is not None else None for e in energies_hartree]
    aligned_forces_ev = aligned_forces * hartree_bohr_to_eV_angstrom
    
    save_xyz("aligned_positions_ev.xyz", aligned_positions, atom_types, energies_ev, comment="Aligned positions")
    save_xyz("aligned_forces_eV.xyz", aligned_forces_ev, atom_types, comment="Aligned forces")

    aligned_positions_path = PROJECT_ROOT / "data" / "processed" / "aligned_positions.xyz"
    md_positions, atom_types, _ = parse_positions_xyz(aligned_positions_path, num_atoms)
    md_forces = parse_forces_xyz("aligned_forces.xyz", num_atoms)
    species = sorted(list(set(atom_types)))
    soap = SOAP(
        species=species,
        r_cut=soap_params["r_cut"],
        n_max=soap_params["n_max"],
        l_max=soap_params["l_max"],
        sigma=soap_params["sigma"],
        periodic=soap_params["periodic"],
        sparse=soap_params["sparse"]
    )
    logger.info("Computing SOAP descriptors for MD frames...")
    precomputed_soap = compute_soap_descriptors(md_positions, atom_types, soap)
    logger.info("Performing clustering based on SOAP descriptors...")
    representative_md = cluster_trajectory(precomputed_soap, clustering_method, num_clusters, md_positions, atom_types)
    pca_samples = generate_structures_from_pca(md_positions, md_forces, representative_md, atom_types, 
                                               num_samples_pca, scaling_factor, pca_variance_threshold=0.90)
    
    medoid_structure_path = PROJECT_ROOT / "data" / "processed" / "medoid_structure.xyz"
    pca_surface_samples = generate_surface_core_pca_samples(md_positions, md_forces, atom_types, surface_atom_types,
            representative_md, num_samples_pca_surface, scaling_surf, scaling_core,
            medoid_structure_path) 
    
    randomized_samples = generate_randomized_samples(representative_md, atom_types, num_samples_randomization,
                                                     max_random_displacement)

    combined_samples = {
        "MD": md_positions,
        "PCA": pca_samples,
        "PCA_Surface": pca_surface_samples,
        "Randomized": randomized_samples,
    }
    for name, samples in combined_samples.items():
        logger.info(f"{name}: {np.array(samples).shape}")

    plot_generated_samples_extended(combined_samples, atom_types, soap)

    # Combine non-MD samples for training dataset.
    combined_list = []
    frame_titles = []
    for name, samples in combined_samples.items():
        if name != "MD":
            samples_arr = np.array(samples)
            combined_list.append(samples_arr)
            for i in range(len(samples)):
                frame_titles.append(f"Frame {i+1} from {name}")
    training_dataset = np.vstack(combined_list).reshape(-1, num_atoms, 3)
    with open("training_dataset.xyz", "w") as f:
        for i, (struct, title) in enumerate(zip(training_dataset, frame_titles)):
            f.write(f"{len(struct)}\n{title}\n")
            for atom, coords in zip(atom_types, struct):
                f.write(f"{atom} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")
    logger.info(f"Saved training dataset with {training_dataset.shape[0]} structures to 'training_dataset.xyz'")

