#!/usr/bin/env python

import numpy as np
import csv
from typing import Dict
from sklearn.preprocessing import StandardScaler
from ase.data import atomic_numbers as _ase_atomic_numbers
from orchestr_ai.utils.io import ( parse_stacked_xyz, save_stacked_xyz,
                              save_to_npz )
from orchestr_ai.utils.plots import (
    plot_energy_and_forces,
    plot_pca,
    plot_umap,
    plot_tsne,
    plot_kmeans_elbow,
    plot_cluster_map,
    plot_coverage_histogram,
    plot_pca_density_contour,
)
from orchestr_ai.utils.helpers import ( analyze_reference_forces,
                                   suggest_thresholds )
from orchestr_ai.utils.pca import detect_outliers
from orchestr_ai.utils.cluster import (
    select_kmeans_medoids,
    compute_kmeans_elbow,
    suggest_elbow_k_values,
    recommend_elbow_k,
    assign_kmeans_labels,
    sample_indices,
    compute_subset_coverage_metrics,
    select_farthest_point_sampling,
    compute_selection_entropy,
)
from orchestr_ai.utils.descriptors import compute_local_descriptors
from orchestr_ai.utils.centering import process_xyz
from orchestr_ai.utils.data_conversion import preprocess_data_for_platform
from orchestr_ai.utils.coverage_report import plot_coverage_summary_report


import logging
logger = logging.getLogger(__name__)

def export_subset_bundle(
    feats,
    E,
    P,
    F,
    atoms,
    atomic_numbers_1d,
    sel_idxs,
    prefix,
    set_id,
    tgt,
    method_name,
    coverage_rows,
    random_state=0,
    entropy_cluster_labels=None,
    entropy_k=None,
):
    """
    Export one selected subset and all associated plots/files.

    This handles:
      - coverage metrics + histogram
      - PCA/UMAP/t-SNE coverage plots
      - XYZ save
      - energy/force plot
      - MACE preprocessing
      - centering
      - NPZ save

    Parameters
    ----------
    feats : np.ndarray
        Full inlier feature matrix.
    E, P, F : np.ndarray
        Inlier energies, positions, forces.
    atoms : list
        Atomic symbols.
    atomic_numbers_1d : np.ndarray
        1D atomic numbers for NPZ export.
    sel_idxs : np.ndarray
        Selected indices for this subset.
    prefix : str
        Output prefix.
    set_id : int
        Repetition / split index.
    tgt : int
        Target subset size.
    method_name : str
        Selection method name, e.g. "kmeans", "random", "fps".
    coverage_rows : list
        List to append summary rows into.
    random_state : int
        Random seed for visualization.
    """
    nsel = len(sel_idxs)
    n_total = len(feats)

    tag = "" if method_name == "kmeans" else f"_{method_name}"
    method_label = method_name.upper()

    # 1) Coverage metrics
    cov_metrics, cov_min_dists = compute_subset_coverage_metrics(feats, sel_idxs)
    logger.info(
        f"[Coverage-{method_label}] set={set_id}, size={tgt}, "
        f"mean={cov_metrics['mean_min_dist']:.6f}, "
        f"p95={cov_metrics['p95_min_dist']:.6f}, "
        f"max={cov_metrics['max_min_dist']:.6f}"
    )

    entropy_metrics = {
        "entropy": None,
        "normalized_entropy": None,
        "occupied_clusters": None,
        "total_clusters": entropy_k,
    }

    if entropy_cluster_labels is not None:
        try:
            entropy_metrics = compute_selection_entropy(
                entropy_cluster_labels,
                sel_idxs,
            )
            logger.info(
                f"[Entropy-{method_label}] set={set_id}, size={tgt}, "
                f"H={entropy_metrics['entropy']:.6f}, "
                f"Hnorm={entropy_metrics['normalized_entropy']:.6f}, "
                f"occupied={entropy_metrics['occupied_clusters']}/{entropy_metrics['total_clusters']}"
            )
        except Exception as e:
            logger.warning(f"[Entropy-{method_label}] Failed: {e}")

    coverage_rows.append({
        "set_id": int(set_id),
        "size": int(nsel),
        "method": method_name,
        "mean_min_dist": float(cov_metrics["mean_min_dist"]),
        "p95_min_dist": float(cov_metrics["p95_min_dist"]),
        "max_min_dist": float(cov_metrics["max_min_dist"]),
        "entropy": entropy_metrics["entropy"],
        "normalized_entropy": entropy_metrics["normalized_entropy"],
        "occupied_clusters": entropy_metrics["occupied_clusters"],
        "total_clusters": entropy_metrics["total_clusters"],
    })

    plot_coverage_histogram(
        cov_min_dists,
        title=f"Coverage Histogram ({method_label}): selected {nsel} from {n_total} inliers",
        filename=f"{prefix}_set{set_id}_{tgt}_coverage_hist{tag}.png",
    )

    # 2) Coverage plots
    plot_pca(
        feats,
        title=f"PCA Coverage ({method_label}): selected {nsel} from {n_total} inliers",
        filename=f"{prefix}_set{set_id}_{tgt}_coverage_pca{tag}.png",
        selected_idx=sel_idxs,
        random_state=random_state,
    )

    try:
        plot_umap(
            feats,
            title=f"UMAP Coverage ({method_label}): selected {nsel} from {n_total} inliers",
            filename=f"{prefix}_set{set_id}_{tgt}_coverage_umap{tag}.png",
            selected_idx=sel_idxs,
            random_state=random_state,
        )
    except Exception as e:
        logger.warning(f"[UMAP-{method_name}] Skipped due to error: {e}")

    plot_tsne(
        feats,
        title=f"t-SNE Coverage ({method_label}): selected {nsel} from {n_total} inliers",
        filename=f"{prefix}_set{set_id}_{tgt}_coverage_tsne{tag}.png",
        selected_idx=sel_idxs,
        random_state=random_state,
    )

    # 3) Save XYZ
    xyz_fn = f"{prefix}_set{set_id}_{tgt}{tag}.xyz"
    save_stacked_xyz(xyz_fn, E[sel_idxs], P[sel_idxs], F[sel_idxs], atoms)

    # 4) Energy/force plots
    plot_energy_and_forces(
        E[sel_idxs],
        F[sel_idxs],
        filename=f"{prefix}_set{set_id}_EF_{method_name}_{tgt}.png"
    )

    # 5) Platform preprocessing
    preprocess_data_for_platform(xyz_fn, "mace")

    # 6) Centering
    centered_xyz = f"{prefix}_set{set_id}_{tgt}{tag}_centered.xyz"
    centered_png = f"{prefix}_set{set_id}_{tgt}{tag}_centered.png"
    process_xyz(xyz_fn, centered_xyz, centered_png)

    # 7) Save NPZ
    npz_fn = f"{prefix}_set{set_id}_{tgt}{tag}.npz"
    save_to_npz(
        filename=npz_fn,
        atomic_numbers=atomic_numbers_1d,
        positions=P[sel_idxs],
        energies=E[sel_idxs],
        forces=F[sel_idxs],
    )

    if False:  # diagnostic only
        plot_pca_density_contour(
            embeddings=feats,
            selected_indices=sel_idxs,
            title=f"PCA Density Contour ({method_label}): selected {nsel} from {n_total} inliers",
            output_path=f"{prefix}_set{set_id}_{tgt}_pca_density_contour{tag}.png",
            random_state=random_state,
        )

def save_coverage_summary(prefix, coverage_rows):
    """
    Save full coverage summary to CSV and print compact grouped summary in logs.
    """
    if not coverage_rows:
        logger.info("[CoverageSummary] No coverage rows collected. Skipping summary export.")
        return

    coverage_rows = sorted(
        coverage_rows,
        key=lambda r: (r["size"], r["method"], r["set_id"])
    )

    coverage_csv = f"{prefix}_coverage_summary.csv"
    fieldnames = [
        "set_id",
        "size",
        "method",
        "mean_min_dist",
        "p95_min_dist",
        "max_min_dist",
        "entropy",
        "normalized_entropy",
        "occupied_clusters",
        "total_clusters",
    ]

    with open(coverage_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(coverage_rows)

    compact_rows = []

    grouped = {}
    for row in coverage_rows:
        key = (row["size"], row["method"])
        grouped.setdefault(key, {
            "mean_min_dist": [],
            "p95_min_dist": [],
            "max_min_dist": [],
            "entropy": [],
            "normalized_entropy": [],
            "occupied_clusters": [],
        })
        grouped[key]["mean_min_dist"].append(row["mean_min_dist"])
        grouped[key]["p95_min_dist"].append(row["p95_min_dist"])
        grouped[key]["max_min_dist"].append(row["max_min_dist"])
        grouped[key]["entropy"].append(row["entropy"])
        grouped[key]["normalized_entropy"].append(row["normalized_entropy"])
        grouped[key]["occupied_clusters"].append(row["occupied_clusters"])

    for (size, method), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        compact_rows.append({
            "size": size,
            "method": method,
            "mean_min_dist_avg": sum(vals["mean_min_dist"]) / len(vals["mean_min_dist"]),
            "p95_min_dist_avg": sum(vals["p95_min_dist"]) / len(vals["p95_min_dist"]),
            "max_min_dist_avg": sum(vals["max_min_dist"]) / len(vals["max_min_dist"]),
            "entropy_avg": sum(vals["entropy"]) / len(vals["entropy"]),
            "normalized_entropy_avg": sum(vals["normalized_entropy"]) / len(vals["normalized_entropy"]),
            "occupied_clusters_avg": sum(vals["occupied_clusters"]) / len(vals["occupied_clusters"]),
            "n_repeats": len(vals["mean_min_dist"]),
        })

    compact_csv = f"{prefix}_coverage_summary_compact.csv"
    compact_fieldnames = [
        "size",
        "method",
        "mean_min_dist_avg",
        "p95_min_dist_avg",
        "max_min_dist_avg",
        "entropy_avg",
        "normalized_entropy_avg",
        "occupied_clusters_avg",
        "n_repeats",
    ]

    with open(compact_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compact_fieldnames)
        writer.writeheader()
        writer.writerows(compact_rows)

    logger.info(f"[CoverageSummary] Saved compact coverage summary to {compact_csv}")


def run_elbow_analysis(
    feats,
    sizes,
    prefix,
    seed,
    elbow_enabled=False,
    elbow_k_values=None,
    elbow_max_k=1000,
    auto_add_elbow_size=False,
    max_auto_elbow_size=2000,
    elbow_selection_method="knee",
):
    """
    Run elbow analysis and optionally auto-add a recommended subset size.

    Returns
    -------
    sizes : list[int]
        Possibly updated subset sizes.
    elbow_best_k : int or None
        Recommended elbow cluster count, if available.
    """
    elbow_best_k = None

    if not elbow_enabled:
        return sizes, elbow_best_k

    n_inliers = len(feats)

    if elbow_k_values is None:
        elbow_k_values = suggest_elbow_k_values(
            n_samples=n_inliers,
            requested_sizes=sizes,
            max_k=elbow_max_k,
        )
        logger.info(f"[Elbow] Auto-generated k values from n_inliers={n_inliers}: {elbow_k_values}")
    else:
        logger.info(f"[Elbow] Using user-provided k values: {elbow_k_values}")

    ks, wcss = compute_kmeans_elbow(
        feats,
        k_values=elbow_k_values,
        random_state=seed,
    )

    plot_kmeans_elbow(
        ks,
        wcss,
        title="KMeans Elbow on Inlier Feature Space",
        filename=f"{prefix}_kmeans_elbow.png",
    )

    if len(ks) > 0:
        logger.info("[Elbow] Finished. Inspect the elbow plot for a good cluster count.")

    if elbow_selection_method == "knee":
        elbow_best_k = recommend_elbow_k(ks, wcss)
    else:
        logger.warning(
            f"[Elbow] Unsupported method '{elbow_selection_method}'. Using 'knee'."
        )
        elbow_best_k = recommend_elbow_k(ks, wcss)

    if elbow_best_k is not None:
        elbow_best_k = min(elbow_best_k, len(feats))
        logger.info(f"[Elbow] Recommended elbow size: {elbow_best_k}")

        if auto_add_elbow_size:
            if elbow_best_k <= max_auto_elbow_size:
                logger.info(f"[Elbow] Auto-selected additional subset size: {elbow_best_k}")
                sizes = sorted(set(list(sizes) + [int(elbow_best_k)]))
                logger.info(f"[Elbow] Final subset sizes after merge: {sizes}")
            else:
                logger.warning(
                    f"[Elbow] Recommended k={elbow_best_k} exceeds "
                    f"max_auto_elbow_size={max_auto_elbow_size}. Skipping auto-add."
                )
    else:
        logger.warning("[Elbow] Could not determine a recommended elbow size.")

    return sizes, elbow_best_k

def maybe_plot_cluster_map(
    feats,
    elbow_best_k,
    prefix,
    seed,
    max_cluster_map_k=500,
):
    """
    Plot cluster map only when elbow_best_k is available and not too large.
    """
    if elbow_best_k is None:
        return

    if elbow_best_k > max_cluster_map_k:
        logger.warning(
            f"[ClusterMap] Skipping cluster map because elbow_best_k={elbow_best_k} "
            f"> max_cluster_map_k={max_cluster_map_k}"
        )
        return

    try:
        cluster_labels, _ = assign_kmeans_labels(
            feats,
            n_clusters=elbow_best_k,
            random_state=seed,
        )

        plot_cluster_map(
            feats,
            cluster_labels,
            title=f"PCA Cluster Map (k={elbow_best_k})",
            filename=f"{prefix}_cluster_map_k{elbow_best_k}.png",
            method="pca",
            random_state=seed,
        )
    except Exception as e:
        logger.warning(f"[ClusterMap] Failed to generate cluster map: {e}")

def consolidate_dataset(cfg: Dict):
    """
    Main pipeline: parse, outlier‐filter, SOAP, features, clustering,
    then generate sampled subsets per config.
    """
    # 1) Load config
    ds       = cfg["dataset"]
    infile   = ds["input_file"]
    prefix   = ds["output_prefix"]
    sizes    = ds["sizes"]
    sampling = ds.get("sampling", "subsample")
    n_sets   = ds.get("n_sets", 1)
    bf       = ds.get("bootstrap_factor", 1)
    cont     = ds.get("contamination", 0.05)
    seed     = ds.get("seed", 0)

    # Optional elbow plot config
    create_random_baseline = ds.get("create_random_baseline", False)
    create_fps_baseline = ds.get("create_fps_baseline", False)
    max_fps_size = ds.get("max_fps_size", 300)

    elbow_enabled = ds.get("plot_elbow", False)
    elbow_k_values = ds.get("elbow_k_values", None)
    elbow_max_k = ds.get("elbow_max_k", 1000)
    auto_add_elbow_size = ds.get("auto_add_elbow_size", False)
    max_auto_elbow_size = ds.get("max_auto_elbow_size", 2000)
    max_cluster_map_k = ds.get("max_cluster_map_k", 500)
    elbow_selection_method = ds.get("elbow_selection_method", "knee")

    logger.info(f"[Consolidate] parsing {infile}…")
    # 2) Parse stacked XYZ
    E, P, F, atoms = parse_stacked_xyz(infile)
    labels_full = np.arange(len(E))  

    n_frames = len(E)
    n_atoms  = len(atoms)
    logger.info(f" frames={n_frames}, atoms/frame={n_atoms}")

    # 3) Quick diagnostics on energy/forces
    plot_energy_and_forces(E, F, "initial_energy_forces.png")

    # 4) Global feature: [ E, avg |F|, var(F) ]
    avg_force = np.linalg.norm(F, axis=2).mean(axis=1)
    var_force = F.var(axis=(1,2))
    global_feats = np.vstack((E, avg_force, var_force)).T

    # 5) Outlier statistics and suggested thresholds
    force_stats = analyze_reference_forces(F, atoms)
    suggest_thresholds(force_stats)

    # 6) Local SOAP descriptors
    soap_params = cfg.get("SOAP")
    local_feats = compute_local_descriptors(P, atoms, soap_params)

    # 7) Combine + scale features
    raw_feats = np.hstack((global_feats, local_feats))
    feats     = StandardScaler().fit_transform(raw_feats)

    logger.info(f"[StandardScaler] Done, feature shape: {feats.shape}")
    
    # 8) Outlier detection via IsolationForest
    inliers_mask = detect_outliers(
        feats,
        contamination=cont,
        labels=labels_full,
        title=f"Outlier Detection (cont={cont})",
        filename=f"{prefix}_outliers_if.png",
        random_state=seed,
    )

    # 9) Keep only inliers
    labels = labels_full[inliers_mask]
    feats = feats[inliers_mask]
    E = E[inliers_mask]
    P = P[inliers_mask]
    F = F[inliers_mask]
    logger.info(f"[Filter] kept {len(E)} frames after outlier removal")

    sizes, elbow_best_k = run_elbow_analysis(
        feats=feats,
        sizes=sizes,
        prefix=prefix,
        seed=seed,
        elbow_enabled=elbow_enabled,
        elbow_k_values=elbow_k_values,
        elbow_max_k=elbow_max_k,
        auto_add_elbow_size=auto_add_elbow_size,
        max_auto_elbow_size=max_auto_elbow_size,
        elbow_selection_method=elbow_selection_method,
    )

    maybe_plot_cluster_map(
        feats=feats,
        elbow_best_k=elbow_best_k,
        prefix=prefix,
        seed=seed,
        max_cluster_map_k=max_cluster_map_k,
    )

    # Reference clustering for entropy calculation
    entropy_cluster_labels = None
    entropy_k = None

    if len(feats) > 1:
        if elbow_best_k is not None:
            entropy_k = min(int(elbow_best_k), len(feats) - 1)
        else:
            entropy_k = min(100, len(feats) - 1)

        if entropy_k >= 2:
            try:
                entropy_cluster_labels, _ = assign_kmeans_labels(
                    feats,
                    n_clusters=entropy_k,
                    random_state=seed,
                )
                logger.info(f"[Entropy] Using reference clustering with k={entropy_k}")
            except Exception as e:
                logger.warning(f"[Entropy] Failed to compute reference clustering: {e}")
                entropy_cluster_labels = None

    plot_energy_and_forces(E, F, "postfilter_EF.png")
    plot_pca(
        feats,
        title="Inliers PCA",
        filename=f"{prefix}_inliers_pca.png",
        random_state=seed,
    )

    # 10) Save full inliers file
    save_stacked_xyz(f"{prefix}_inliers_full_dataset.xyz", E, P, F, atoms)

    # Precompute 1D atomic_numbers for NPZ
    atomic_numbers_1d = np.array([_ase_atomic_numbers[sym] for sym in atoms],
                                 dtype=np.int32)

    # Collect coverage metrics for CSV + log summary
    coverage_rows = []

    # 11) K-means medoid selection for each target size, repeated n_sets times with different seeds
    for set_id in range(n_sets):
        set_seed = seed + set_id
        logger.info(f"[Select] set_id={set_id}/{n_sets-1}, seed={set_seed}")

        for tgt in sizes:
            if elbow_best_k is not None and int(tgt) == int(elbow_best_k):
                logger.info(f"[Select] size={tgt} was auto-added from elbow recommendation")

            nsel = min(len(feats), tgt)

            # ==========================================================
            # 1) Diverse subset via KMeans + medoid
            # ==========================================================
            sel_idxs = select_kmeans_medoids(feats, nsel, random_state=set_seed)
            logger.info(f"[KMeans] set={set_id} selected {len(sel_idxs)} reps for size {tgt}")

            export_subset_bundle(
                feats=feats,
                E=E,
                P=P,
                F=F,
                atoms=atoms,
                atomic_numbers_1d=atomic_numbers_1d,
                sel_idxs=sel_idxs,
                prefix=prefix,
                set_id=set_id,
                tgt=tgt,
                method_name="kmeans",
                coverage_rows=coverage_rows,
                random_state=set_seed,
                entropy_cluster_labels=entropy_cluster_labels,
                entropy_k=entropy_k,
            )

            # ==========================================================
            # 2) Optional random baseline subset
            # ==========================================================
            if create_random_baseline:
                rng = np.random.default_rng(set_seed)
                rnd_idxs = sample_indices(
                    n_total=len(feats),
                    n_target=nsel,
                    mode="subsample",
                    bootstrap_factor=1,
                    rng=rng,
                )
                logger.info(f"[Random] set={set_id} selected {len(rnd_idxs)} random frames for size {tgt}")

                export_subset_bundle(
                    feats=feats,
                    E=E,
                    P=P,
                    F=F,
                    atoms=atoms,
                    atomic_numbers_1d=atomic_numbers_1d,
                    sel_idxs=rnd_idxs,
                    prefix=prefix,
                    set_id=set_id,
                    tgt=tgt,
                    method_name="random",
                    coverage_rows=coverage_rows,
                    random_state=set_seed,
                    entropy_cluster_labels=entropy_cluster_labels,
                    entropy_k=entropy_k,
                )

            # ==========================================================
            # 3) Optional FPS subset
            # ==========================================================
            if create_fps_baseline:
                if nsel <= max_fps_size:
                    fps_idxs = select_farthest_point_sampling(
                        feats,
                        nsel,
                        random_state=set_seed,
                    )

                    logger.info(
                        f"[FPS] set={set_id} selected {len(fps_idxs)} frames for size {tgt}"
                    )

                    export_subset_bundle(
                        feats=feats,
                        E=E,
                        P=P,
                        F=F,
                        atoms=atoms,
                        atomic_numbers_1d=atomic_numbers_1d,
                        sel_idxs=fps_idxs,
                        prefix=prefix,
                        set_id=set_id,
                        tgt=tgt,
                        method_name="fps",
                        coverage_rows=coverage_rows,
                        random_state=set_seed,                
                        entropy_cluster_labels=entropy_cluster_labels,
                        entropy_k=entropy_k,
                    )
                else:
                    logger.warning(
                        f"[FPS] Skipping FPS for size={nsel} because "
                        f"nsel > max_fps_size={max_fps_size}"
                    )

    # 12) Save coverage summary CSV + print compact summary
    save_coverage_summary(prefix, coverage_rows)
    plot_coverage_summary_report(f"{prefix}_coverage_summary_compact.csv")