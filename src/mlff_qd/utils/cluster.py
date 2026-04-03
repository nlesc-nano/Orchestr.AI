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
    
def suggest_elbow_k_values(n_samples: int, requested_sizes=None):
    """
    Suggest a compact, dataset-size-aware list of k values for elbow analysis.

    Strategy:
      - dense at small k
      - moderate sampling at medium k
      - a few larger values based on fractions of n_samples
      - always include requested subset sizes if provided

    Returns a sorted Python list of valid k values.
    """
    if n_samples < 3:
        return []

    requested_sizes = requested_sizes or []

    base_small = [50, 100, 200, 300, 500]
    base_medium = [600, 800, 1000, 1200, 1500]

    frac_vals = [
        int(round(0.25 * n_samples)),
        int(round(0.35 * n_samples)),
        int(round(0.50 * n_samples)),
        int(round(0.75 * n_samples)),
    ]

    candidates = set(base_small + base_medium + frac_vals)

    for s in requested_sizes:
        try:
            candidates.add(int(s))
        except Exception:
            pass

    # valid k must satisfy 2 <= k < n_samples
    ks = sorted(k for k in candidates if 2 <= k < n_samples)

    # avoid overly long expensive lists for huge datasets
    if len(ks) > 15:
        # keep first 10 and last 5 as a simple compact strategy
        ks = ks[:10] + ks[-5:]
        ks = sorted(set(ks))

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
