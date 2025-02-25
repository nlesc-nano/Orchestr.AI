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
