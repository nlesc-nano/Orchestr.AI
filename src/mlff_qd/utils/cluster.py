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
from ase import Atoms

from mlff_qd.utils.io import save_xyz

import logging
logger = logging.getLogger(__name__)

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

def compute_soap_descriptors(md_positions, atom_types, soap):
    """Compute a global (averaged) SOAP descriptor for each MD frame."""
    descriptors = []
    
    for pos in md_positions:
        atoms = Atoms(symbols=atom_types, positions=pos)
        soap_desc = soap.create(atoms)
        descriptors.append(np.mean(soap_desc, axis=0))
    
    return np.array(descriptors)
