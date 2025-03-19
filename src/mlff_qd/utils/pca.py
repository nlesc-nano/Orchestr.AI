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
from sklearn.preprocessing import StandardScaler

from mlff_qd.utils.analysis import ( compute_rmsd_matrix, plot_rmsd_histogram,
        compute_global_distance_fluctuation_cdist )
from mlff_qd.utils.io import save_xyz
from mlff_qd.utils.surface import compute_surface_indices_with_replace_surface_dynamic

import logging
logger = logging.getLogger(__name__)

def generate_pca_samples(reference_structure, pca_model, num_samples, scaling_factor=1.0):
    """
    Generate new structures by perturbing along PCA components.

    Parameters:
        reference_structure (np.ndarray): Atomic positions of the reference structure (num_atoms, 3).
        pca_model (PCA): Precomputed PCA model.
        num_samples (int): Number of new structures to generate.
        scaling_factor (float): Scaling factor for perturbations.

    Returns:
        np.ndarray: Array of generated structures (num_samples, num_atoms, 3).
    """
    flattened_ref = reference_structure.flatten()  # Flatten the reference structure
    num_features = pca_model.components_.shape[1]  # Number of features used in PCA

    # Validate that the reference structure matches PCA components
    if flattened_ref.size != num_features:
        raise ValueError(
            f"Mismatch in size: flattened_ref ({flattened_ref.size}) does not match PCA components ({num_features})"
        )

    generated_structures = []

    for _ in range(num_samples):
        perturbation = np.zeros_like(flattened_ref)
        for pc_idx in range(pca_model.n_components_):
            perturbation += (
                scaling_factor
                * np.random.uniform(-1, 1)
                * pca_model.components_[pc_idx]
            )
        new_structure = flattened_ref + perturbation
        generated_structures.append(new_structure.reshape(-1, 3))

    return np.array(generated_structures)

def perform_pca_and_plot(datasets, num_components=2, labels=None, ax=None):
    """
    Perform PCA and plot the results.

    Parameters:
        datasets (dict): A dictionary of dataset names and their corresponding samples.
        num_components (int): Number of PCA components.
        labels (list): List of sample labels (optional).
        ax (matplotlib.axes.Axes): Pre-created axis for plotting (optional).
    """

    print(f"Performing PCA for {len(datasets)} datasets...")
    combined_data = np.concatenate(list(datasets.values()))
    combined_labels = labels if labels else sum([[name] * len(data) for name, data in datasets.items()], [])

    # Perform PCA
    pca = PCA(n_components=num_components)
    pca_transformed = pca.fit_transform(combined_data)

    # Plot results
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(set(combined_labels))
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    for idx, label in enumerate(unique_labels):
        label_indices = [i for i, lbl in enumerate(combined_labels) if lbl == label]
        ax.scatter(
            pca_transformed[label_indices, 0],
            pca_transformed[label_indices, 1],
            label=label,
            alpha=0.7,
            color=colors(idx),
        )

    explained_variance = pca.explained_variance_ratio_ * 100
    ax.set_title(
        f"PCA Analysis of Molecular Configurations\n"
        f"Principal Component 1 ({explained_variance[0]:.2f}%) vs "
        f"Principal Component 2 ({explained_variance[1]:.2f}%)"
    )
    ax.set_xlabel(f"Principal Component 1 ({explained_variance[0]:.2f}%)")
    ax.set_ylabel(f"Principal Component 2 ({explained_variance[1]:.2f}%)")
    ax.legend()

    if ax is None:
        plt.tight_layout()
        plt.show()

def generate_structures_from_pca(
    rmsd_md_internal,
    md_positions,
    representative_md,
    atom_types,
    num_samples,
    max_displacement,
    temperature,
    temperature_target
):
    """
    Generate PCA-based structures from a given MD trajectory and save them to "pca_generated_samples.xyz".

    Parameters:
        rmsd_md_internal (np.ndarray): RMSD matrix of the MD trajectory (N x N).
        md_positions (list of np.ndarray): Aligned MD frames, each frame is (num_atoms, 3).
        representative_md (list of np.ndarray): A subset of MD frames used as reference frames.
        atom_types (list of str): Atom types for each atom in the structure.
        num_samples (int): Number of PCA-based samples to generate.
        generate_pca_samples (function): Function to generate samples given a reference structure and PCA model.
        max_displacement (float): Maximum displacement cap to apply to the scaling factors.
        temperature (float): Temperature at which the MD was performed.
        temperature_target (float): Target temperature for scaling.

    Returns:
        np.ndarray: Array of PCA-generated structures with shape (num_samples, num_atoms, 3).
    """
    print("Computing RMSD-based scaling factor...")
    rmsd_values = np.mean(rmsd_md_internal, axis=1)  # Mean RMSD of each frame to all others
    rmsd_scaling_factor = np.mean(rmsd_values)
    print(f"RMSD-based scaling factor: {rmsd_scaling_factor:.2f}")

    # Perform PCA on the full aligned MD trajectory
    flattened_md_positions = np.array([frame.flatten() for frame in md_positions])  # (N_frames, num_atoms*3)
    pca = PCA()
    pca.fit(flattened_md_positions)

    # Determine the number of components to capture at least 90% variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = np.argmax(explained_variance >= 0.90) + 1
    print(
        f"Optimal number of PCA components: {optimal_components} "
        f"(captures {explained_variance[optimal_components - 1] * 100:.2f}% variance)"
    )

    # Refit PCA with the optimal number of components
    pca = PCA(n_components=optimal_components)
    pca.fit(flattened_md_positions)

    # Compute adjusted scaling factor based on target temperature
    scaling_factors_temp_adjusted = rmsd_scaling_factor * np.sqrt(temperature_target / temperature)
    print(f"Scaling factors for PCA components (adjusted for T={temperature_target}K): {scaling_factors_temp_adjusted}")

    # Apply a cap to the scaling factors for maximum displacement
    scaling_factors_temp_adjusted = min(scaling_factors_temp_adjusted, max_displacement)
    print(f"Scaled and capped scaling factors: {scaling_factors_temp_adjusted}")

    # Generate PCA-based samples by applying displacements to representative frames
    print(f"Generating {num_samples} PCA-based samples...")
    pca_samples = []
    for i in range(num_samples):
        start_idx = np.random.choice(len(representative_md))  # Choose random frame from representative_md
        reference_structure = representative_md[start_idx]

        # Generate a new structure by applying PCA perturbations
        pca_sample = generate_pca_samples(
            reference_structure,
            pca,
            1,
            scaling_factor=scaling_factors_temp_adjusted
        )[0]
        pca_samples.append(pca_sample)

        # Print progress every 100 samples or for the first sample
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Generating sample {i + 1}/{num_samples}...")

    pca_samples = np.array(pca_samples)
    print(f"Generated {len(pca_samples)} PCA-based samples with shape {pca_samples.shape}.")

    pca_samples_rmsd = compute_rmsd_matrix(pca_samples)
    plot_rmsd_histogram(pca_samples_rmsd, bins=50, title="RMSD Histogram PCA", xlabel="RMSD (Å)", ylabel="Frequency")

    # Write the PCA-based samples directly to "pca_generated_samples.xyz"
    save_xyz("pca_samples.xyz", pca_samples, atom_types)
    print(f"Saved {len(pca_samples)} PCA-based samples to 'pca_samples.xyz'.")

    return pca_samples

def generate_surface_core_pca_samples(
    md_positions,
    md_forces,
    atom_types,
    surface_atom_types,
    representative_md,
    num_samples, 
    scaling_surf=0.6, 
    scaling_core=0.4,
    medoid_structure_file="medoid_structure.xyz",
    surface_replaced_file="surface_replaced.xyz"
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
        surface_replaced_file (str): Path to the replaced-surface structure (default is "surface_replaced.xyz").

    Returns:
        np.ndarray: Array of shape (num_samples, num_atoms, 3) with new perturbed structures.
    """
    logger.info("Generating PCA-based surface-core samples (positions+forces only, no SOAP)...")

    # 1) Identify surface and core indices
    logger.info("Identifying surface vs. core atoms...")
    surface_repl_file = "surface_replaced.xyz"
    surface_indices, _ = compute_surface_indices_with_replace_surface_dynamic(
        medoid_structure_file, surface_atom_types, f=1.0, surface_replaced_file=surface_replaced_file
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
