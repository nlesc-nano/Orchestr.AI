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
from scipy.spatial.distance import cdist

from scm.plams import Molecule
from CAT.recipes import replace_surface

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def compute_rmsd_matrix(structures1, structures2=None):
    """
    Compute the RMSD matrix for pairs of structures efficiently.

    Parameters:
        structures1 (list or np.ndarray): Array of shape (N1, num_atoms, 3)
                                          representing the first set of structures.
        structures2 (list or np.ndarray): Array of shape (N2, num_atoms, 3)
                                          representing the second set of structures.
                                          If None, computes the RMSD matrix within structures1.

    Returns:
        np.ndarray: RMSD matrix of shape (N1, N2).
    """
    # Ensure we have NumPy arrays
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
        # Compute only upper triangle and mirror it to avoid redundant computation
        rmsd_matrix = np.zeros((N1, N2), dtype=float)
        for i in range(N1):
            # Only compute from j = i to the end
            diff = structures2[i:] - structures1[i]  # shape: (N2 - i, A, 3)
            # sum of squared differences per structure
            sq_diff_sum = np.sum(diff**2, axis=(1, 2)) / A
            rmsd_vals = np.sqrt(sq_diff_sum)
            rmsd_matrix[i, i:] = rmsd_vals
            rmsd_matrix[i:, i] = rmsd_vals  # mirror to maintain symmetry
    else:
        # When structures2 is different, compute fully
        # Vectorized approach
        # Broadcast to create a differences array of shape (N1, N2, A, 3)
        diff = structures1[:, None, :, :] - structures2[None, :, :, :]
        # sum over atoms and coordinates
        sq_diff_sum = np.sum(diff**2, axis=(2, 3)) / A
        rmsd_matrix = np.sqrt(sq_diff_sum)

    return rmsd_matrix

def plot_generated_samples(combined_samples):
    """
    Plot PCA visualizations of the combined samples dataset for both "positions" and "RMSD" representations,
    without outlier detection.

    Parameters:
        combined_samples (dict): Dictionary with dataset names as keys and samples as values. 
                                 Each dataset should have shape (num_samples, num_atoms, 3) or (num_samples, flattened_dim).
    """
    # Flatten datasets for PCA
    print("Flattening datasets for PCA...")
    flattened_samples = {}
    for name, samples in combined_samples.items():
        samples = np.array(samples)  # Ensure data is a NumPy array
        if samples.ndim == 3:  # Shape: (num_samples, num_atoms, 3)
            flattened_samples[name] = samples.reshape(samples.shape[0], -1)  # Flatten to (num_samples, 3*num_atoms)
        elif samples.ndim == 2:  # Already flattened
            flattened_samples[name] = samples
        else:
            raise ValueError(f"Unexpected shape for dataset '{name}': {samples.shape}")

    # Concatenate all datasets
    print("Concatenating datasets for PCA...")
    all_samples_positions = np.concatenate(list(flattened_samples.values()), axis=0)  # Combine datasets row-wise
    sample_labels = sum([[name] * len(data) for name, data in flattened_samples.items()], [])

    # Prepare RMSD matrix for RMSD-based PCA
    print("Computing RMSD matrix for all samples...")
    all_samples_reshaped = all_samples_positions.reshape(all_samples_positions.shape[0], -1, 3)  # Reshape to 3D
    rmsd_matrix = compute_rmsd_matrix(all_samples_reshaped)

    # Perform PCA for "positions"
    print("Performing PCA for 'positions'...")
    pca_positions = PCA(n_components=2)
    pca_transformed_positions = pca_positions.fit_transform(all_samples_positions)
    explained_variance_positions = np.sum(pca_positions.explained_variance_ratio_) * 100
    print(f"Explained variance by first two components (positions): {explained_variance_positions:.2f}%")

    # Perform PCA for "RMSD"
    print("Performing PCA for 'RMSD'...")
    pca_rmsd = PCA(n_components=2)
    pca_transformed_rmsd = pca_rmsd.fit_transform(rmsd_matrix)
    explained_variance_rmsd = np.sum(pca_rmsd.explained_variance_ratio_) * 100
    print(f"Explained variance by first two components (RMSD): {explained_variance_rmsd:.2f}%")

    # Create color map for datasets
    unique_labels = sorted(set(sample_labels))
    color_map = {label: plt.cm.tab10(idx / len(unique_labels)) for idx, label in enumerate(unique_labels)}

    # Plotting
    print("Plotting PCA results...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot for "positions"
    for label in unique_labels:
        label_indices = [i for i, lbl in enumerate(sample_labels) if lbl == label]
        axes[0].scatter(
            pca_transformed_positions[label_indices, 0],
            pca_transformed_positions[label_indices, 1],
            label=label,
            color=color_map[label],
            alpha=0.7
        )
    axes[0].set_title("PCA: Positions Representation")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    axes[0].legend()
    axes[0].grid(True)

    # Plot for "RMSD"
    for label in unique_labels:
        label_indices = [i for i, lbl in enumerate(sample_labels) if lbl == label]
        axes[1].scatter(
            pca_transformed_rmsd[label_indices, 0],
            pca_transformed_rmsd[label_indices, 1],
            label=label,
            color=color_map[label],
            alpha=0.7
        )
    axes[1].set_title("PCA: RMSD Representation")
    axes[1].set_xlabel("Principal Component 1")
    axes[1].set_ylabel("Principal Component 2")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

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
