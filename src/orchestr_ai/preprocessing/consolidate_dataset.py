#!/usr/bin/env python

import numpy as np
import csv
from typing import Dict
from sklearn.preprocessing import StandardScaler
from ase.data import atomic_numbers as _ase_atomic_numbers
from mlff_qd.utils.io import ( parse_stacked_xyz, save_stacked_xyz,
                              save_to_npz )
from mlff_qd.utils.plots import (
    plot_energy_and_forces,
    plot_pca,
    plot_umap,
    plot_tsne,
    plot_kmeans_elbow,
    plot_cluster_map,
    plot_coverage_histogram,
)
from mlff_qd.utils.helpers import ( analyze_reference_forces,
                                   suggest_thresholds )
from mlff_qd.utils.pca import detect_outliers
from mlff_qd.utils.cluster import (
    select_kmeans_medoids,
    compute_kmeans_elbow,
    suggest_elbow_k_values,
    recommend_elbow_k,
    assign_kmeans_labels,
    sample_indices,
    compute_subset_coverage_metrics,
)
from mlff_qd.utils.descriptors import compute_local_descriptors
from mlff_qd.utils.centering import process_xyz
from mlff_qd.utils.data_conversion import preprocess_data_for_platform

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

    coverage_rows.append({
        "set_id": int(set_id),
        "size": int(nsel),
        "method": method_name,
        "mean_min_dist": float(cov_metrics["mean_min_dist"]),
        "p95_min_dist": float(cov_metrics["p95_min_dist"]),
        "max_min_dist": float(cov_metrics["max_min_dist"]),
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

    elbow_best_k = None

    if elbow_enabled:
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

        # Determine recommended elbow k independently of auto-add
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

    if elbow_best_k is not None:
        if elbow_best_k <= max_cluster_map_k:
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
        else:
            logger.warning(
                f"[ClusterMap] Skipping cluster map because elbow_best_k={elbow_best_k} "
                f"> max_cluster_map_k={max_cluster_map_k}"
            )

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
                )
    # 12) Save coverage summary CSV + print compact summary
    if coverage_rows:
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
        ]

        with open(coverage_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(coverage_rows)

        logger.info(f"[CoverageSummary] Saved full coverage summary to {coverage_csv}")
    
        grouped = {}
        for row in coverage_rows:
            key = (row["size"], row["method"])
            grouped.setdefault(key, {
                "mean_min_dist": [],
                "p95_min_dist": [],
                "max_min_dist": [],
            })
            grouped[key]["mean_min_dist"].append(row["mean_min_dist"])
            grouped[key]["p95_min_dist"].append(row["p95_min_dist"])
            grouped[key]["max_min_dist"].append(row["max_min_dist"])

        for (size, method), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            mean_avg = sum(vals["mean_min_dist"]) / len(vals["mean_min_dist"])
            p95_avg = sum(vals["p95_min_dist"]) / len(vals["p95_min_dist"])
            max_avg = sum(vals["max_min_dist"]) / len(vals["max_min_dist"])

            logger.info(
                f"[CoverageSummary] size={size}, method={method}, "
                f"mean={mean_avg:.6f}, p95={p95_avg:.6f}, max={max_avg:.6f}"
            )
