import numpy as np
from typing import Sequence, Dict
from sklearn.cluster import KMeans
from ase.data import atomic_numbers as _ase_atomic_numbers
from mlff_qd.utils.io import parse_stacked_xyz

import logging
logger = logging.getLogger(__name__)

def select_kmeans_medoids(features, n_clusters: int, random_state: int = 0):
    """
    KMeans clustering + medoid selection: pick, for each cluster, the member closest to its centroid.
    Returns an array of selected indices (length = n_clusters).
    """
    kmed = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(features)
    centers = kmed.cluster_centers_
    cluster_lbls = kmed.labels_

    sel_idxs = []
    for lbl in range(n_clusters):
        members = np.where(cluster_lbls == lbl)[0]
        dists   = np.linalg.norm(features[members] - centers[lbl], axis=1)
        sel_idxs.append(members[np.argmin(dists)])
    return np.asarray(sel_idxs, dtype=int)

def compute_kmeans_elbow(features, k_values, random_state: int = 0):
    """
    Compute WCSS (inertia) for a sequence of k values.
    Returns:
        ks   : np.ndarray of valid cluster counts
        wcss : np.ndarray of inertia values
    """
    X = np.asarray(features)
    n_samples = len(X)

    ks = []
    wcss = []

    for k in k_values:
        k = int(k)
        if k < 1:
            logger.warning(f"[compute_kmeans_elbow] Skipping invalid k={k}")
            continue
        if k > n_samples:
            logger.warning(f"[compute_kmeans_elbow] Skipping k={k} because k > n_samples={n_samples}")
            continue

        logger.info(f"[compute_kmeans_elbow] Fitting KMeans for k={k}")
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init="auto",
        ).fit(X)

        ks.append(k)
        wcss.append(km.inertia_)

    return np.asarray(ks, dtype=int), np.asarray(wcss, dtype=float)
    
def sample_indices(n_total: int,
                   n_target: int,
                   mode: str="subsample",
                   bootstrap_factor: int=1,
                   rng: np.random.Generator=None) -> np.ndarray:
    """
    Return indices for subsample (unique) or bootstrap (with replacement).
    If bootstrap, concatenates `bootstrap_factor` replicates.
    """
    rng = rng or np.random.default_rng()
    if mode not in ("subsample","bootstrap"):
        raise ValueError(f"Invalid mode {mode}")
    if mode=="subsample":
        if n_target>n_total:
            raise ValueError(f"Subsample {n_target}>{n_total}")
        return rng.choice(n_total, n_target, replace=False)
    # bootstrap
    reps=[]
    for _ in range(max(1,bootstrap_factor)):
        reps.append(rng.choice(n_total, n_target, replace=True))
    return np.concatenate(reps)
