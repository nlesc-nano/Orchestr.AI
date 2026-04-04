import numpy as np
import matplotlib.pyplot as plt

from mlff_qd.utils.pca import project_pca2
from sklearn.manifold import TSNE

import logging
logger = logging.getLogger(__name__)


def _save_close(filename, dpi=200):
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info(f"[plot] Saved: {filename}")


def _validate_selected(selected_idx, n):
    if selected_idx is None:
        return None
    selected_idx = np.asarray(selected_idx)
    if selected_idx.ndim != 1:
        raise ValueError("selected_idx must be a 1D array of indices")
    if len(selected_idx) == 0:
        return selected_idx
    if selected_idx.min() < 0 or selected_idx.max() >= n:
        raise ValueError("selected_idx contains invalid indices")
    return selected_idx


def _subsample_indices(n, max_points=None, random_state=0):
    if max_points is None or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return rng.choice(n, size=max_points, replace=False)

def plot_kmeans_elbow(
    ks,
    wcss,
    title="KMeans Elbow",
    filename="kmeans_elbow.png",
    dpi=200,
):
    """
    Plot WCSS vs number of clusters (k) for elbow analysis.
    """
    ks = np.asarray(ks)
    wcss = np.asarray(wcss)

    if len(ks) == 0 or len(wcss) == 0:
        logger.warning("[plot_kmeans_elbow] Empty ks/wcss. Skipping elbow plot.")
        return

    if len(ks) != len(wcss):
        raise ValueError("ks and wcss must have the same length")

    logger.info("[plot_kmeans_elbow] Starting....")

    plt.figure(figsize=(20, 6))
    plt.plot(ks, wcss, marker="o")
    plt.xticks(ks)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS / Inertia")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    _save_close(filename, dpi=dpi)

    
def plot_outliers(
    features,
    labels,
    outliers,
    title,
    filename,
    method="pca",
    max_plot=20000,
    random_state=0,
    dpi=200,
):
    """
    Plot inliers vs outliers in 2D.
    labels is kept only for compatibility; it is not used for grouping.
    """
    logger.info("[plot_outliers] Starting....")

    X = np.asarray(features)
    op = np.asarray(outliers)

    if X.shape[0] != len(op):
        raise ValueError("features and outliers must have same number of rows")

    idx = _subsample_indices(len(X), max_points=max_plot, random_state=random_state)
    Xp = X[idx]
    op = op[idx]

    if len(X) > len(idx):
        logger.info(f"[plot_outliers] Subsampling for plot: {len(idx)}/{len(X)}")

    if method == "pca":
        red, _ = project_pca2(Xp)

    elif method == "tsne":
        perplexity = min(30, max(5, len(Xp) - 1))
        red = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        ).fit_transform(Xp)

    elif method == "umap":
        try:
            import umap
            red = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                random_state=random_state,
            ).fit_transform(Xp)
        except ImportError:
            logger.warning("[plot_outliers] UMAP not installed. Skipping UMAP outlier plot.")
            return
        except Exception as e:
            logger.warning(f"[plot_outliers] UMAP failed. Skipping plot. Error: {e}")
            return

    else:
        raise ValueError("method must be one of: 'pca', 'tsne', 'umap'")

    m_in = (op == 1)
    m_out = (op == -1)

    plt.figure(figsize=(8, 6))
    plt.scatter(red[m_in, 0], red[m_in, 1], s=8, alpha=0.45, label="inliers")
    plt.scatter(red[m_out, 0], red[m_out, 1], s=28, alpha=0.9, marker="x", label="outliers")

    plt.title(title)
    plt.legend()
    _save_close(filename, dpi=dpi)

def plot_cluster_map(
    features,
    cluster_labels,
    title="Cluster Map",
    filename="cluster_map.png",
    method="pca",
    max_plot=30000,
    random_state=0,
    dpi=200,
):
    """
    Fast cluster visualization for full dataset.

    Uses one scatter call with c=cluster_labels to avoid slow per-cluster loops.
    """
    logger.info(f"[plot_cluster_map] Starting.... method={method}")

    X = np.asarray(features)
    y = np.asarray(cluster_labels)

    if len(X) != len(y):
        raise ValueError("features and cluster_labels must have same length")

    idx = _subsample_indices(len(X), max_points=max_plot, random_state=random_state)
    Xp = X[idx]
    yp = y[idx]

    if len(X) > len(idx):
        logger.info(f"[plot_cluster_map] Subsampling for plot: {len(idx)}/{len(X)}")

    if method == "pca":
        red, _ = project_pca2(Xp)

    elif method == "tsne":
        perplexity = min(30, max(5, len(Xp) - 1))
        red = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        ).fit_transform(Xp)

    elif method == "umap":
        try:
            import umap
            red = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                random_state=random_state,
            ).fit_transform(Xp)
        except ImportError:
            logger.warning("[plot_cluster_map] UMAP not installed. Skipping UMAP cluster plot.")
            return
        except Exception as e:
            logger.warning(f"[plot_cluster_map] UMAP failed. Skipping plot. Error: {e}")
            return

    else:
        raise ValueError("method must be one of: 'pca', 'tsne', 'umap'")

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        red[:, 0],
        red[:, 1],
        c=yp,
        s=10,
        alpha=0.75,
        cmap="tab20",
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # optional compact colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label("Cluster ID")

    _save_close(filename, dpi=dpi)
    
def plot_pca(
    features,
    labels=None,
    title="PCA Projection",
    filename="pca.png",
    selected_idx=None,
    max_plot=30000,
    random_state=0,
    dpi=200,
):
    """
    PCA plot for full dataset, optionally overlaying selected subset.
    Uses project_pca2(...) from mlff_qd.utils.pca.
    """
    logger.info("[plot_pca] Starting....")

    X = np.asarray(features)
    n = len(X)
    selected_idx = _validate_selected(selected_idx, n)

    bg_idx = _subsample_indices(n, max_points=max_plot, random_state=random_state)
    if selected_idx is not None and len(selected_idx) > 0:
        bg_idx = np.unique(np.concatenate([bg_idx, selected_idx]))

    X_plot = X[bg_idx]
    red, pca = project_pca2(X_plot)

    plt.figure(figsize=(8, 6))

    if selected_idx is None:
        plt.scatter(red[:, 0], red[:, 1], s=8, alpha=0.6, label="frames")
    else:
        sel_set = set(selected_idx.tolist())
        sel_mask = np.array([i in sel_set for i in bg_idx])
        bg_mask = ~sel_mask

        plt.scatter(red[bg_mask, 0], red[bg_mask, 1], s=8, alpha=0.25, label="all frames")
        plt.scatter(red[sel_mask, 0], red[sel_mask, 1], s=35, alpha=0.95, label=f"selected ({sel_mask.sum()})")

    evr = pca.explained_variance_ratio_
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    _save_close(filename, dpi=dpi)

def plot_umap(
    features,
    title="UMAP Projection",
    filename="umap.png",
    selected_idx=None,
    max_plot=20000,
    random_state=0,
    n_neighbors=15,
    min_dist=0.1,
    dpi=200,
):
    """
    UMAP plot for full dataset, optionally overlaying selected subset.
    """
    logger.info("[plot_umap] Starting....")

    try:
        import umap
    except ImportError:
        logger.warning("[plot_umap] UMAP not installed. Skipping UMAP plot.")
        return

    X = np.asarray(features)
    n = len(X)
    selected_idx = _validate_selected(selected_idx, n)

    bg_idx = _subsample_indices(n, max_points=max_plot, random_state=random_state)
    if selected_idx is not None and len(selected_idx) > 0:
        bg_idx = np.unique(np.concatenate([bg_idx, selected_idx]))

    X_plot = X[bg_idx]

    try:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        red = reducer.fit_transform(X_plot)
    except Exception as e:
        logger.warning(f"[plot_umap] UMAP failed. Skipping plot. Error: {e}")
        return

    plt.figure(figsize=(8, 6))

    if selected_idx is None:
        plt.scatter(red[:, 0], red[:, 1], s=8, alpha=0.6, label="frames")
    else:
        sel_set = set(selected_idx.tolist())
        sel_mask = np.array([i in sel_set for i in bg_idx])
        bg_mask = ~sel_mask

        plt.scatter(red[bg_mask, 0], red[bg_mask, 1], s=8, alpha=0.25, label="all frames")
        plt.scatter(red[sel_mask, 0], red[sel_mask, 1], s=35, alpha=0.95, label=f"selected ({sel_mask.sum()})")

    plt.title(title)
    plt.legend()
    _save_close(filename, dpi=dpi)

def plot_tsne(
    features,
    title="t-SNE Projection",
    filename="tsne.png",
    selected_idx=None,
    max_plot=5000,
    random_state=0,
    perplexity=30,
    dpi=200,
):
    """
    t-SNE plot for full dataset, optionally overlaying selected subset.
    """
    logger.info("[plot_tsne] Starting....")

    X = np.asarray(features)
    n = len(X)
    selected_idx = _validate_selected(selected_idx, n)

    bg_idx = _subsample_indices(n, max_points=max_plot, random_state=random_state)
    if selected_idx is not None and len(selected_idx) > 0:
        bg_idx = np.unique(np.concatenate([bg_idx, selected_idx]))

    X_plot = X[bg_idx]
    effective_perplexity = min(perplexity, max(5, len(X_plot) - 1))

    red = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=effective_perplexity,
        random_state=random_state,
    ).fit_transform(X_plot)

    plt.figure(figsize=(8, 6))

    if selected_idx is None:
        plt.scatter(red[:, 0], red[:, 1], s=8, alpha=0.6, label="frames")
    else:
        sel_set = set(selected_idx.tolist())
        sel_mask = np.array([i in sel_set for i in bg_idx])
        bg_mask = ~sel_mask

        plt.scatter(red[bg_mask, 0], red[bg_mask, 1], s=8, alpha=0.25, label="all frames")
        plt.scatter(red[sel_mask, 0], red[sel_mask, 1], s=35, alpha=0.95, label=f"selected ({sel_mask.sum()})")

    plt.title(title)
    plt.legend()
    _save_close(filename, dpi=dpi)


# def plot_pca(features, labels, title="PCA", filename="pca.png"):
#     red, _  = project_pca2(features)
#     plt.figure(figsize=(8,6))
#     cmap = ["blue","green","red","orange","purple","brown","pink","gray"]
#     for lbl in np.unique(labels):
#         m = (labels==lbl)
#         plt.scatter(red[m,0],red[m,1],label=f"grp{lbl}",c=cmap[lbl%len(cmap)],alpha=0.7)
#     plt.legend(); plt.title(title)
#     plt.savefig(filename, dpi=300); plt.close()



# def plot_outliers(features,labels,outliers,title,filename):
#     logger.info(f"[plot_outliers] Starting....")
#     red, _  = project_pca2(features)
#     logger.info(f"[project_pca2] PCA projection done.")
#     plt.figure(figsize=(8,6))
#     cmap = ["blue","green","red","orange"]
#     for lbl in np.unique(labels):
#         m_all = (labels==lbl)
#         m_in = m_all & (outliers==1)
#         m_out= m_all & (outliers==-1)
#         plt.scatter(red[m_in,0],red[m_in,1],c=cmap[lbl % len(cmap)],label=f"{lbl} in")
#         plt.scatter(red[m_out,0],red[m_out,1],c=cmap[lbl % len(cmap)],marker='x',s=50,label=f"{lbl} out")
#     plt.title(title); plt.legend()
#     plt.savefig(filename, dpi=300); plt.close()

  
def plot_energy_and_forces(energies, forces, filename='analysis.png'):
    """Plot energy-per-frame, energy-per-atom, max/avg force with thresholds."""
    num_frames = len(energies)
    frames     = np.arange(num_frames)
    num_atoms  = forces.shape[1]

    energy_per_atom = energies / num_atoms
    mean_epa = np.mean(energy_per_atom)
    std_epa  = np.std(energy_per_atom)
    epa_2p = mean_epa + 2*std_epa
    epa_3p = mean_epa + 3*std_epa
    epa_2m = mean_epa - 2*std_epa
    epa_3m = mean_epa - 3*std_epa
    chem_p = mean_epa + 0.05
    chem_m = mean_epa - 0.05

    fmagn = np.linalg.norm(forces, axis=2)
    maxF = np.max(fmagn, axis=1)
    avgF = np.mean(fmagn, axis=1)
    mean_avgF = np.mean(avgF)
    std_avgF  = np.std(avgF)

    fig, axes = plt.subplots(4,1, figsize=(10,20))
    # Total Energy
    axes[0].plot(frames, energies, 'o-', label='Total E')
    axes[0].set(title='Total Energy per Frame', xlabel='Frame', ylabel='E (eV)')
    axes[0].legend()
    # Energy/atom
    axes[1].plot(frames, energy_per_atom, 'o-', color='purple', label='E/atom')
    for y, lbl in [(mean_epa,'Mean'), (epa_2p,'Mean+2σ'), (epa_3p,'Mean+3σ'),
                   (epa_2m,''), (epa_3m,''), (chem_p,'±0.05 eV/atom'), (chem_m,'')]:
        axes[1].axhline(y, linestyle='--' if 'σ' in lbl else ':', color='gray', label=lbl)
    axes[1].set(title='Energy per Atom', xlabel='Frame', ylabel='E/N (eV)')
    axes[1].legend()
    # Max force
    axes[2].plot(frames, maxF, 'o-', color='red', label='Max F')
    axes[2].set(title='Max Force per Frame', xlabel='Frame', ylabel='|F| (eV/Å)')
    axes[2].legend()
    # Avg force
    axes[3].plot(frames, avgF, 'o-', color='green', label='Avg F')
    axes[3].axhline(mean_avgF, linestyle='--', color='gray', label='Mean')
    axes[3].axhline(mean_avgF+2*std_avgF, linestyle='--', color='orange', label='Mean+2σ')
    axes[3].axhline(mean_avgF+3*std_avgF, linestyle='--', color='red', label='Mean+3σ')
    axes[3].set(title='Average Force per Frame', xlabel='Frame', ylabel='|F| (eV/Å)')
    axes[3].legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    logger.info(f"[Plot] Energy/force plots saved to {filename}")
