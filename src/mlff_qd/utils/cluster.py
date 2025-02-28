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

from mlff_qd.utils.io import save_xyz

def cluster_trajectory(rmsd_md_internal, clustering_method, num_clusters, md_positions, atom_types):
    """
    Perform clustering on MD trajectory based on RMSD and return a subset of representative structures.
    The representative structure for each cluster is chosen as the centroid in RMSD space.

    Parameters:
        rmsd_md_internal (np.ndarray): RMSD matrix of shape (N, N), where N is the number of frames.
        clustering_method (str): Clustering method to use ("KMeans", "DBSCAN", or "GMM").
        num_clusters (int): Number of clusters to form (if applicable).
        md_positions (list of np.ndarray): List of MD frames, each frame is an array of shape (num_atoms, 3).
        atom_types (list of str): Atom type labels corresponding to each atom in the frames.

    Returns:
        list of np.ndarray: The clustered_md, a list of representative structures selected from each cluster.
    """
    print("Performing clustering on MD trajectory...")

    # Select clustering method
    if clustering_method == "KMeans":
        print("Using KMeans clustering...")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(rmsd_md_internal)
    elif clustering_method == "DBSCAN":
        print("Using DBSCAN clustering...")
        eps = 0.2  # Distance threshold for forming clusters
        min_samples = 10  # Minimum number of points to form a dense region
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        cluster_labels = dbscan.fit_predict(rmsd_md_internal)
    elif clustering_method == "GMM":
        print("Using Gaussian Mixture Model clustering...")
        gmm = GaussianMixture(n_components=num_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(rmsd_md_internal)
    else:
        raise ValueError(f"Invalid clustering method: {clustering_method}")

    # Select one representative structure (centroid) from each cluster
    clustered_indices = []
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            # Skip noise points (only applicable to DBSCAN)
            continue
        cluster_members = np.where(cluster_labels == cluster_id)[0]

        # Compute the centroid of the cluster in RMSD space
        # The centroid is defined as the structure that has the minimum average RMSD to all others in the cluster.
        cluster_rmsd_matrix = rmsd_md_internal[np.ix_(cluster_members, cluster_members)]
        centroid_idx_within_cluster = np.argmin(cluster_rmsd_matrix.mean(axis=1))
        representative_idx = cluster_members[centroid_idx_within_cluster]
        clustered_indices.append(representative_idx)

    clustered_md = [md_positions[i] for i in clustered_indices]
    print(f"Clustered MD trajectory: {len(clustered_md)} structures from {len(unique_clusters)} clusters.")

    # Save the clustered MD sample
    save_xyz("clustered_md_sample.xyz", clustered_md, atom_types)

    return clustered_md
