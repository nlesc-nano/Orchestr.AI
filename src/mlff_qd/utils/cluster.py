import numpy as np
from sklearn.cluster import KMeans

import logging
logger = logging.getLogger(__name__)


def select_kmeans_medoids(features, n_clusters: int, random_state: int = 0):
    """
    KMeans clustering + medoid selection: pick, for each cluster, the member closest to its centroid.
    Returns an array of selected indices (length = n_clusters).
    """
    kmed = KMeans(n_clusters=n_clusters, random_state=random_state).fit(features)
    centers = kmed.cluster_centers_
    cluster_lbls = kmed.labels_

    sel_idxs = []
    for lbl in range(n_clusters):
        members = np.where(cluster_lbls == lbl)[0]
        dists   = np.linalg.norm(features[members] - centers[lbl], axis=1)
        sel_idxs.append(members[np.argmin(dists)])
    return np.asarray(sel_idxs, dtype=int)