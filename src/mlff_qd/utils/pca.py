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

from ase import Atoms
from scm.plams import Molecule
from CAT.recipes import replace_surface

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mlff_qd.utils.analysis import compute_global_distance_fluctuation_cdist
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

def plot_generated_samples(combined_samples, atom_types, soap):
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
