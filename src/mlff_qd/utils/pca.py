from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

import logging
logger = logging.getLogger(__name__)

def project_pca2(features):
    """Return (X2d, fitted_pca) where X2d = PCA(n_components=2).fit_transform(features)."""
    pca = PCA(n_components=2,svd_solver="randomized",random_state=0)
    X2d = pca.fit_transform(features)
    return X2d, pca
    
def detect_outliers(features, contamination: float, labels, title: str, filename: str, random_state: int = 0):
    """
    IsolationForest-based outlier detection. Returns a boolean mask of inliers.
    Also renders the outlier plot using existing plot_outliers(...).
    """
<<<<<<< HEAD
    if contamination == 0:
        logger.info("[Filter] contamination=0, keeping all frames as inliers")
        return slice(None)
    
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    y_pred = clf.fit_predict(features)          # -1 outlier, +1 inlier
    
||||||| a1c69fc
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    y_pred = clf.fit_predict(features)          # -1 outlier, +1 inlier
    
=======
    logger.info("[detect_outliers] Starting....")

    clf = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    y_pred = clf.fit_predict(features)   # -1 outlier, +1 inlier

>>>>>>> dev_z
    try:
        from mlff_qd.utils.plots import plot_outliers

        plot_outliers(
            features,
            labels,
            y_pred,
            title=title,
            filename=filename,
            method="pca",
            random_state=random_state,
        )

        base = filename.rsplit(".", 1)[0]

        plot_outliers(
            features,
            labels,
            y_pred,
            title=f"{title} [t-SNE]",
            filename=f"{base}_tsne.png",
            method="tsne",
            random_state=random_state,
        )

        plot_outliers(
            features,
            labels,
            y_pred,
            title=f"{title} [UMAP]",
            filename=f"{base}_umap.png",
            method="umap",
            random_state=random_state,
        )

        logger.info("[detect_outliers] Outlier plots generated.")
    except Exception as e:
        logger.warning(f"[detect_outliers] Plotting failed: {e}")

    logger.info(
        f"[detect_outliers] Done. Inliers: {(y_pred == 1).sum()}, "
        f"Outliers: {(y_pred == -1).sum()}"
    )
    return (y_pred == 1)