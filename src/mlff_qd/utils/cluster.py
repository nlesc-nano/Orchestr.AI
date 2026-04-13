import numpy as np
from typing import Sequence, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_chunked
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


def select_farthest_point_sampling(features, n_select, random_state=0):
    """
    Farthest Point Sampling (FPS).

    Iteratively select points that maximize the minimum distance
    to the already selected set.

    Parameters
    ----------
    features : np.ndarray, shape (n_samples, n_features)
    n_select : int
        Number of points to select
    random_state : int

    Returns
    -------
    sel_idxs : np.ndarray of shape (n_select,)
    """
    X = np.asarray(features)
    n_samples = X.shape[0]

    if n_select >= n_samples:
        return np.arange(n_samples)

    rng = np.random.default_rng(random_state)

    # Start from a random point
    first_idx = rng.integers(0, n_samples)
    selected = [first_idx]

    # Initialize distances to selected set
    # min_dists = np.linalg.norm(X - X[first_idx], axis=1)
    diff = X - X[first_idx]
    min_dists = np.einsum("ij,ij->i", diff, diff)

    for _ in range(1, n_select):
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)

        # Update minimum distances
        # dists = np.linalg.norm(X - X[next_idx], axis=1)
        diff = X - X[next_idx]
        dists = np.einsum("ij,ij->i", diff, diff)
        min_dists = np.minimum(min_dists, dists)

    return np.array(selected, dtype=int)

def assign_kmeans_labels(features, n_clusters: int, random_state: int = 0):
    """
    Fit KMeans and return cluster labels and fitted model.
    """ 
    X = np.asarray(features)
    n_samples = len(X)

    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if n_clusters > n_samples:
        raise ValueError(f"n_clusters={n_clusters} cannot exceed n_samples={n_samples}")

    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    ).fit(X)

    return km.labels_, km


def compute_selection_entropy(cluster_labels_full, selected_idx):
    """
    Compute Shannon entropy of a selected subset over full-data cluster labels.

    Parameters
    ----------
    cluster_labels_full : array-like, shape (n_samples,)
        Cluster labels for the full dataset.
    selected_idx : array-like, shape (n_selected,)
        Indices of selected subset.

    Returns
    -------
    metrics : dict
        Dictionary with:
          - entropy
          - normalized_entropy
          - occupied_clusters
          - total_clusters
    """
    labels = np.asarray(cluster_labels_full, dtype=int)
    sel = np.asarray(selected_idx, dtype=int)

    if labels.ndim != 1:
        raise ValueError("cluster_labels_full must be 1D")
    if sel.ndim != 1:
        raise ValueError("selected_idx must be 1D")
    if len(sel) == 0:
        raise ValueError("selected_idx is empty")
    if sel.min() < 0 or sel.max() >= len(labels):
        raise ValueError("selected_idx contains invalid indices")

    sel_labels = labels[sel]
    counts = np.bincount(sel_labels)
    counts = counts[counts > 0]

    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs))

    total_clusters = int(labels.max()) + 1
    occupied_clusters = int(len(counts))

    if total_clusters > 1:
        normalized_entropy = float(entropy / np.log(total_clusters))
    else:
        normalized_entropy = 0.0

    return {
        "entropy": float(entropy),
        "normalized_entropy": normalized_entropy,
        "occupied_clusters": occupied_clusters,
        "total_clusters": total_clusters,
    }
    
def compute_subset_coverage_metrics(features, selected_idx):
    """
    Compute nearest-selected-point coverage metrics.

    For each point in the full dataset, compute distance to the nearest
    selected point. Returns summary metrics and the full distance array.

    Parameters
    ----------
    features : array-like, shape (n_samples, n_features)
        Full feature matrix.
    selected_idx : array-like, shape (n_selected,)
        Indices of selected subset.

    Returns
    -------
    metrics : dict
        Dictionary with:
          - mean_min_dist
          - max_min_dist
          - p95_min_dist
          - n_selected
          - n_total
    min_dists : np.ndarray
        Distance from each full point to its nearest selected point.
    """
    X = np.asarray(features)
    sel = np.asarray(selected_idx, dtype=int)

    if X.ndim != 2:
        raise ValueError("features must be a 2D array")
    if sel.ndim != 1:
        raise ValueError("selected_idx must be a 1D array")
    if len(sel) == 0:
        raise ValueError("selected_idx is empty")
    if sel.min() < 0 or sel.max() >= len(X):
        raise ValueError("selected_idx contains invalid indices")

    X_sel = X[sel]

    min_dists_chunks = []
    for chunk in pairwise_distances_chunked(X, X_sel, metric="euclidean"):
        min_d = np.min(chunk, axis=1)
        min_dists_chunks.append(min_d)

    min_dists = np.concatenate(min_dists_chunks)

    metrics = {
        "mean_min_dist": float(np.mean(min_dists)),
        "max_min_dist": float(np.max(min_dists)),
        "p95_min_dist": float(np.percentile(min_dists, 95)),
        "n_selected": int(len(sel)),
        "n_total": int(len(X)),
    }

    return metrics, min_dists

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


def recommend_elbow_k(ks, wcss):
    """
    Recommend an elbow k from (k, wcss) using the maximum-distance-to-line method.

    Parameters
    ----------
    ks : array-like
        Cluster counts.
    wcss : array-like
        Inertia / WCSS values.

    Returns
    -------
    best_k : int or None
        Recommended elbow cluster count, or None if not enough points.
    """
    ks = np.asarray(ks, dtype=float)
    wcss = np.asarray(wcss, dtype=float)

    if len(ks) < 3 or len(wcss) < 3:
        logger.warning("[recommend_elbow_k] Need at least 3 elbow points. Skipping recommendation.")
        return None

    if len(ks) != len(wcss):
        raise ValueError("ks and wcss must have the same length")

    # Line from first to last point
    p1 = np.array([ks[0], wcss[0]], dtype=float)
    p2 = np.array([ks[-1], wcss[-1]], dtype=float)

    line_vec = p2 - p1
    line_norm = np.linalg.norm(line_vec)
    if line_norm == 0:
        logger.warning("[recommend_elbow_k] Degenerate elbow line. Skipping recommendation.")
        return None

    # Perpendicular distance from each point to the line
    distances = []
    for k, val in zip(ks, wcss):
        p = np.array([k, val], dtype=float)
        dist = np.abs(np.cross(line_vec, p - p1)) / line_norm
        distances.append(dist)

    distances = np.asarray(distances, dtype=float)
    best_idx = int(np.argmax(distances))
    best_k = int(round(ks[best_idx]))

    logger.info(f"[recommend_elbow_k] Recommended elbow k = {best_k}")
    return best_k

def suggest_elbow_k_values(
    n_samples: int,
    requested_sizes=None,
    max_k: int = 1000,
):
    """
    Suggest a compact, dataset-size-aware list of k values for elbow analysis.

    Strategy:
      - dense at small k
      - moderate sampling at medium k
      - always include requested subset sizes if provided
      - never exceed max_k (or n_samples - 1)
    """
    if n_samples < 3:
        return []

    requested_sizes = requested_sizes or []
    upper = min(int(max_k), n_samples - 1)

    base_small = [50, 150, 300, 450]
    base_medium = [600, 800, 1000, 1200, 1500, 2000, 2500, 3500, 4500, 5000]

    candidates = set()

    for k in base_small + base_medium:
        if 2 <= k <= upper:
            candidates.add(k)

    for s in requested_sizes:
        try:
            s = int(s)
            if 2 <= s <= upper:
                candidates.add(s)
        except Exception:
            pass

    ks = sorted(candidates)
    return ks

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
