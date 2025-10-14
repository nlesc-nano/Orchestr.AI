from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

import logging
logger = logging.getLogger(__name__)

def project_pca2(features):
    """Return (X2d, fitted_pca) where X2d = PCA(n_components=2).fit_transform(features)."""
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(features)
    return X2d, pca
    
def detect_outliers(features, contamination: float, labels, title: str, filename: str, random_state: int = 0):
    """
    IsolationForest-based outlier detection. Returns a boolean mask of inliers.
    Also renders the outlier plot using existing plot_outliers(...).
    """
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    y_pred = clf.fit_predict(features)          # -1 outlier, +1 inlier
    
    # Local import avoids circular dependency at module import time
    try:
        from mlff_qd.utils.plots import plot_outliers  # noqa: WPS433 (local import by design)
        plot_outliers(features, labels, y_pred, title, filename)
    except Exception as e:
        logger.warning(f"plot_outliers unavailable during detect_outliers plotting: {e}")
        
    return (y_pred == 1)