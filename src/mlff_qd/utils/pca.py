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

from mlff_qd.utils.analysis import compute_rmsd_matrix, plot_rmsd_histogram
from mlff_qd.utils.io import save_xyz

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
