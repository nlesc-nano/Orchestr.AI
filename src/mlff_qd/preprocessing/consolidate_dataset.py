#!/usr/bin/env python

import numpy as np
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
)
from mlff_qd.utils.descriptors import compute_local_descriptors
from mlff_qd.utils.centering import process_xyz
from mlff_qd.utils.data_conversion import preprocess_data_for_platform

import logging
logger = logging.getLogger(__name__)

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

        if auto_add_elbow_size:

            if elbow_selection_method == "knee":
                elbow_best_k = recommend_elbow_k(ks, wcss)
            else:
                logger.warning(
                    f"[Elbow] Unsupported method '{elbow_selection_method}'. Using 'knee'."
                )
                elbow_best_k = recommend_elbow_k(ks, wcss)

            if elbow_best_k is not None:
                elbow_best_k = min(elbow_best_k, len(feats))

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

            # Coverage plots: full inlier space + selected subset overlay
            plot_pca(
                feats,
                title=f"PCA Coverage: selected {nsel} from {len(feats)} inliers",
                filename=f"{prefix}_set{set_id}_{tgt}_coverage_pca.png",
                selected_idx=sel_idxs,
                random_state=set_seed,
            )

            try:
                plot_umap(
                    feats,
                    title=f"UMAP Coverage: selected {nsel} from {len(feats)} inliers",
                    filename=f"{prefix}_set{set_id}_{tgt}_coverage_umap.png",
                    selected_idx=sel_idxs,
                    random_state=set_seed,
                )
            except Exception as e:
                logger.warning(f"[UMAP] Skipped due to error: {e}")

            plot_tsne(
                feats,
                title=f"t-SNE Coverage: selected {nsel} from {len(feats)} inliers",
                filename=f"{prefix}_set{set_id}_{tgt}_coverage_tsne.png",
                selected_idx=sel_idxs,
                random_state=set_seed,
            )

            # ---- output names include set_id ----
            xyz_fn = f"{prefix}_set{set_id}_{tgt}.xyz"
            save_stacked_xyz(xyz_fn, E[sel_idxs], P[sel_idxs], F[sel_idxs], atoms)

            plot_energy_and_forces(
                E[sel_idxs], F[sel_idxs],
                filename=f"{prefix}_set{set_id}_EF_sel_{tgt}.png"
            )

            preprocess_data_for_platform(xyz_fn, 'mace')

            centered_xyz = f"{prefix}_set{set_id}_{tgt}_centered.xyz"
            centered_png = f"{prefix}_set{set_id}_{tgt}_centered.png"
            process_xyz(xyz_fn, centered_xyz, centered_png)

            npz_fn = f"{prefix}_set{set_id}_{tgt}.npz"
            save_to_npz(
                filename=npz_fn,
                atomic_numbers=atomic_numbers_1d,
                positions=P[sel_idxs],
                energies=E[sel_idxs],
                forces=F[sel_idxs],
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

                plot_pca(
                    feats,
                    title=f"PCA Coverage (random): selected {nsel} from {len(feats)} inliers",
                    filename=f"{prefix}_set{set_id}_{tgt}_random_coverage_pca.png",
                    selected_idx=rnd_idxs,
                    random_state=set_seed,
                )

                try:
                    plot_umap(
                        feats,
                        title=f"UMAP Coverage (random): selected {nsel} from {len(feats)} inliers",
                        filename=f"{prefix}_set{set_id}_{tgt}_random_coverage_umap.png",
                        selected_idx=rnd_idxs,
                        random_state=set_seed,
                    )
                except Exception as e:
                    logger.warning(f"[UMAP-random] Skipped due to error: {e}")

                plot_tsne(
                    feats,
                    title=f"t-SNE Coverage (random): selected {nsel} from {len(feats)} inliers",
                    filename=f"{prefix}_set{set_id}_{tgt}_random_coverage_tsne.png",
                    selected_idx=rnd_idxs,
                    random_state=set_seed,
                )

                rnd_xyz_fn = f"{prefix}_set{set_id}_{tgt}_random.xyz"
                save_stacked_xyz(rnd_xyz_fn, E[rnd_idxs], P[rnd_idxs], F[rnd_idxs], atoms)

                plot_energy_and_forces(
                    E[rnd_idxs], F[rnd_idxs],
                    filename=f"{prefix}_set{set_id}_EF_random_{tgt}.png"
                )

                preprocess_data_for_platform(rnd_xyz_fn, "mace")

                rnd_centered_xyz = f"{prefix}_set{set_id}_{tgt}_random_centered.xyz"
                rnd_centered_png = f"{prefix}_set{set_id}_{tgt}_random_centered.png"
                process_xyz(rnd_xyz_fn, rnd_centered_xyz, rnd_centered_png)

                rnd_npz_fn = f"{prefix}_set{set_id}_{tgt}_random.npz"
                save_to_npz(
                    filename=rnd_npz_fn,
                    atomic_numbers=atomic_numbers_1d,
                    positions=P[rnd_idxs],
                    energies=E[rnd_idxs],
                    forces=F[rnd_idxs],
                )
