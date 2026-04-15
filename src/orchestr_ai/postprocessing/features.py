"""
features.py

This module computes descriptors and features for atomic configurations.
It uses SOAP from the dscribe library along with bond length and inter-feature
distances, and also performs scaling and PCA transformations. Additionally, it
can generate diagnostic plots for training mode.
"""

import os
import numpy as np
from ase.io import read

try:
    from dscribe.descriptors import SOAP
except ImportError:
    print("Warning: dscribe library not found. SOAP features cannot be computed.")
    SOAP = None  # Set to None if import fails

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import joblib


def compute_avg_bond_length(frames, cutoff=3.5):
    """
    Computes the average bond length for each frame.

    Parameters:
        frames (list of ase.Atoms): List of atomic configurations.
        cutoff (float): Maximum distance for bonds to be considered (default: 3.5 Å).
    
    Returns:
        np.ndarray: Array of average bond lengths with shape (n_frames, 1).
    """
    avg_bond_lengths = []
    for frame in frames:
        if len(frame) < 2:
            avg_bond_lengths.append(0.0)
            continue
        
        distances = frame.get_all_distances(mic=False)
        # Select only unique bond pairs (upper triangle without the diagonal)
        bond_indices = np.triu_indices_from(distances, k=1)
        bond_lengths = distances[bond_indices]
        valid_bonds = bond_lengths[bond_lengths < cutoff]
        if len(valid_bonds) > 0:
            avg_bond_lengths.append(np.mean(valid_bonds))
        else:
            avg_bond_lengths.append(0.0)
    
    return np.array(avg_bond_lengths).reshape(-1, 1)


def compute_features(all_frames, config, training_data_path=None, train_mask=None, eval_mask=None,
                     scaler_load_path=None, pca_load_path=None):
    """
    Computes features for a list of frames by combining SOAP descriptors, average bond lengths,
    and the minimum distances in PCA space.

    Parameters:
        all_frames (list of ase.Atoms): List of atomic configurations.
        config (dict): Configuration dictionary containing settings for SOAP, scaling, and PCA.
        training_data_path (str, optional): Path to training data (unused here).
        train_mask (np.ndarray, optional): Boolean array indicating training frames.
        eval_mask (np.ndarray, optional): Boolean array indicating evaluation frames.
        scaler_load_path (str, optional): File path for loading a pre-trained scaler.
        pca_load_path (str, optional): File path for loading a pre-trained PCA.
    
    Returns:
        tuple: A tuple containing:
            - features_all (np.ndarray): Combined features array.
            - min_distances_all (np.ndarray): Array of minimum distances in PCA space.
            - soap_pca_all (np.ndarray): PCA-transformed SOAP descriptors.
            - pca (PCA object): Fitted PCA object.
            - scaler (StandardScaler object): Fitted scaler object.
    """
    # Ensure SOAP library is available
    if SOAP is None:
        print("Error: dscribe not installed. Cannot compute SOAP features.")
        return None, None, None, None, None

    # --- SOAP Setup ---
    soap_config = config.get("eval", {}).get("SOAP", {})
    species = soap_config.get("species", [])
    if not species:
        print("Warning: SOAP 'species' not defined in config['eval']['SOAP']. Attempting to infer from first frame.")
        if all_frames:
            species = sorted(list(set(all_frames[0].get_chemical_symbols())))
        else:
            raise ValueError("Cannot infer SOAP species: No frames provided and not defined in config.")
        print(f"Inferred species: {species}")

    r_cut = soap_config.get("r_cut", 12.0)
    n_max = soap_config.get("n_max", 9)
    l_max = soap_config.get("l_max", 6)
    sigma = soap_config.get("sigma", 0.05)
    periodic = soap_config.get("periodic", False)
    sparse = soap_config.get("sparse", False)
    average = soap_config.get("average", "off")

    print(f"Initializing SOAP: species={species}, r_cut={r_cut}, n_max={n_max}, l_max={l_max}, average={average}")
    soap = SOAP(
        species=species, r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma,
        periodic=periodic, sparse=sparse, average=average
    )
    n_feat_soap = soap.get_number_of_features()

    # --- Compute SOAP Descriptors ---
    print(f"Computing SOAP descriptors for {len(all_frames)} frames...")
    n_jobs_soap = config.get("eval", {}).get("soap_n_jobs", 1)
    soap_raw_list = soap.create(all_frames, n_jobs=n_jobs_soap)

    soap_avg_all = []
    for i, s in enumerate(soap_raw_list):
        if s is None or s.shape[0] == 0:
            print(f"Warning: Frame {i} resulted in empty SOAP descriptor. Using zeros.")
            soap_avg_all.append(np.zeros(n_feat_soap))
        else:
            # If the SOAP descriptor is an array for all atoms, average over the atoms.
            if s.ndim == 2 and s.shape[0] > 0:
                soap_avg_all.append(np.mean(s, axis=0))
            # If already averaged (1D array) and matches expected feature length:
            elif s.ndim == 1 and s.shape[0] == n_feat_soap:
                soap_avg_all.append(s)
            else:
                print(f"Warning: Frame {i} SOAP descriptor has unexpected shape {s.shape}. Using zeros.")
                soap_avg_all.append(np.zeros(n_feat_soap))
    soap_avg_all = np.array(soap_avg_all)
    print(f"Averaged SOAP shape: {soap_avg_all.shape}")

    # --- Scale and PCA ---
    pca, scaler = None, None
    mode = "unknown"  # Track if we are in training or prediction mode.
    if scaler_load_path and pca_load_path and \
       os.path.exists(scaler_load_path) and os.path.exists(pca_load_path):
        mode = "prediction"
        print(f"Loading pre-trained Scaler from {scaler_load_path}")
        scaler = joblib.load(scaler_load_path)
        print(f"Loading pre-trained PCA from {pca_load_path}")
        pca = joblib.load(pca_load_path)
        soap_scaled_all = scaler.transform(soap_avg_all)
        soap_pca_all = pca.transform(soap_scaled_all)
    elif train_mask is not None:
        mode = "training"
        print("Fitting Scaler and PCA on training data...")
        scaler = StandardScaler()
        soap_avg_train = soap_avg_all[train_mask]
        if len(soap_avg_train) == 0:
            raise ValueError("Cannot fit Scaler/PCA: No training data.")
        soap_scaled_train = scaler.fit_transform(soap_avg_train)
        soap_scaled_all = scaler.transform(soap_avg_all)

        n_components_pca = config.get("eval", {}).get("pca_n_components", 0.99)
        pca = PCA(n_components=n_components_pca, svd_solver='auto')
        pca.fit(soap_scaled_train)
        print(f"PCA fitted: {pca.n_components_} components explain {sum(pca.explained_variance_ratio_):.4f} variance.")
        soap_pca_all = pca.transform(soap_scaled_all)

        # Save fitted models for future predictions.
        scaler_save_path = "soap_scaler.joblib"
        pca_save_path = "soap_pca.joblib"
        joblib.dump(scaler, scaler_save_path)
        joblib.dump(pca, pca_save_path)
        print(f"Saved Scaler to {scaler_save_path}, PCA to {pca_save_path}")
    else:
        raise ValueError("compute_features needs either train_mask or load paths for scaler/pca.")

    print(f"PCA features shape: {soap_pca_all.shape}")

    # --- Compute Minimum Distances in PCA Space ---
    print("Computing minimum distances in PCA space...")
    distances_pca = cdist(soap_pca_all, soap_pca_all, metric='euclidean')
    np.fill_diagonal(distances_pca, np.inf)
    min_distances_all = np.min(distances_pca, axis=1).reshape(-1, 1)
    print(f"Min distances shape: {min_distances_all.shape}")

    # --- Compute Average Bond Lengths ---
    print("Computing average bond lengths...")
    bond_len_cutoff = config.get("eval", {}).get("bond_length_cutoff", 3.5)
    bond_len_all = compute_avg_bond_length(all_frames, cutoff=bond_len_cutoff)
    print(f"Avg bond lengths shape: {bond_len_all.shape}")

    # --- Combine Features ---
    features_all = np.hstack([soap_pca_all, bond_len_all, min_distances_all])
    print(f"Combined features shape: {features_all.shape}")

    # --- Generate Diagnostic Plots (Training Mode Only) ---
    if mode == "training" and train_mask is not None and eval_mask is not None:
        print("Generating diagnostic plots...")
        diag_dir = "diagnostics"
        os.makedirs(diag_dir, exist_ok=True)
        try:
            plt.figure(figsize=(8, 6))
            plt.scatter(soap_pca_all[train_mask, 0], soap_pca_all[train_mask, 1],
                        alpha=0.5, label='Train', s=10)
            plt.scatter(soap_pca_all[eval_mask, 0], soap_pca_all[eval_mask, 1],
                        alpha=0.5, label='Eval', s=10)
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title('PCA of Averaged SOAP Descriptors')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(diag_dir, "soap_pca_scatter.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.hist(min_distances_all[train_mask].flatten(), bins=50, alpha=0.5,
                     label='Train', density=True)
            plt.hist(min_distances_all[eval_mask].flatten(), bins=50, alpha=0.5,
                     label='Eval', density=True)
            plt.xlabel('Minimum SOAP Distance in PCA Space')
            plt.ylabel('Density')
            plt.title('Histogram of Minimum SOAP Distances')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(diag_dir, "hist_min_distances.png"))
            plt.close()
            print(f"Diagnostic plots saved to '{diag_dir}/'")
        except Exception as e_plot:
            print(f"Warning: Failed to generate diagnostic plots: {e_plot}")

    return features_all, min_distances_all, soap_pca_all, pca, scaler



