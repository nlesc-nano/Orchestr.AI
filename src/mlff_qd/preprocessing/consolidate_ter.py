#!/usr/bin/env python

import os
import yaml
import time
import traceback
import numpy as np
from typing import Sequence, Dict
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ase import Atoms
from dscribe.descriptors import SOAP


def load_config(config_file: str) -> Dict:
    """Load input parameters from a YAML configuration file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

import os
import numpy as np

from mlff_qd.utils.io import save_to_npz

from mlff_qd.utils.plots import plot_energy_and_forces

from mlff_qd.utils.helpers import analyze_fluctuations




from mlff_qd.utils.io import parse_stacked_xyz


def create_labels_from_counts(counts):
    """Turn [n1,n2,…] into [0×n1,1×n2,…]."""
    total = sum(counts)
    labels = np.empty(total,dtype=int)
    s=0
    for i,c in enumerate(counts):
        labels[s:s+c]=i
        s+=c
    return labels


from mlff_qd.utils.plots import plot_pca, plot_outliers, plot_final_selection

from mlff_qd.utils.io import save_stacked_xyz

from mlff_qd.utils.descriptors import compute_local_descriptors

from mlff_qd.utils.helpers import analyze_reference_forces, suggest_thresholds

from mlff_qd.utils.cluster import (
    generate_random_subsets,
    generate_md_random_subsets,
)

# ───────────────────────────────────────────────────────────────
# UPDATED: CONSOLIDATION PIPELINE
# ───────────────────────────────────────────────────────────────
from typing import Dict, Sequence
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ase.data import atomic_numbers as _ase_atomic_numbers
from dscribe.descriptors import SOAP


from mlff_qd.utils.pca import detect_outliers

from mlff_qd.utils.cluster import select_kmeans_medoids


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

    print(f"[Consolidate] parsing {infile}…")
    # 2) Parse stacked XYZ
    E, P, F, atoms = parse_stacked_xyz(infile)
    labels_full = np.arange(len(E))  # Or your group/label logic here

    n_frames = len(E)
    n_atoms  = len(atoms)
    print(f" frames={n_frames}, atoms/frame={n_atoms}")

    # 3) Quick diagnostics on energy/forces
    plot_energy_and_forces(E, F, "initial_energy_forces.png")

    # 4) Global feature: [ E, avg |F|, var(F) ]
    avg_force = np.linalg.norm(F, axis=2).mean(axis=1)
    var_force = F.var(axis=(1,2))
    global_feats = np.vstack((E, avg_force, var_force)).T

    # 5) Outlier stats
    force_stats = analyze_reference_forces(F, atoms)
    suggest_thresholds(force_stats)

    # 6) Local SOAP descriptors
    soap = SOAP(**cfg["SOAP"])
    local_feats = compute_local_descriptors(P, atoms, soap)

    # 7) Combine + scale features
    raw_feats = np.hstack((global_feats, local_feats))
    feats     = StandardScaler().fit_transform(raw_feats)

    # 8) IsolationForest filtering
    inliers_mask = detect_outliers(
        feats,
        contamination=cont,
        labels=labels_full,
        title=f"Outlier Detection (cont={cont})",
        filename=f"{prefix}_outliers_if.png",
        random_state=0,
    )

    # 9) Keep only inliers
    labels = labels_full[inliers_mask]
    feats = feats[inliers_mask]
    E = E[inliers_mask]
    P = P[inliers_mask]
    F = F[inliers_mask]
    print(f"[Filter] kept {len(E)} frames after outlier removal")

    plot_energy_and_forces(E, F, "postfilter_EF.png")
    plot_pca(
        feats,
        labels,
        title="Inliers PCA",
        filename=f"{prefix}_inliers_pca.png"
    )

    # 10) Save full inliers file
    save_stacked_xyz("inliers_full_dataset.xyz", E, P, F, atoms)

    # Precompute 1D atomic_numbers for NPZ
    atomic_numbers_1d = np.array([_ase_atomic_numbers[sym] for sym in atoms],
                                 dtype=np.int32)

    # 11) K-means medoid selection for each target size
    for tgt in sizes:
        nsel = min(len(feats), tgt)
        sel_idxs = select_kmeans_medoids(feats, nsel, random_state=0)

        print(f"[KMeans] selected {len(sel_idxs)} representatives for size {tgt}")

        # Save XYZ and energy/force plots
        xyz_fn = f"{prefix}_{tgt}.xyz"
        save_stacked_xyz(xyz_fn, E[sel_idxs], P[sel_idxs], F[sel_idxs], atoms)
        plot_energy_and_forces(E[sel_idxs], F[sel_idxs],
                               filename=f"{prefix}_EF_sel_{tgt}.png")

        # Save NPZ with all four arrays: z, R, E, Fs
        npz_fn = f"{prefix}_{tgt}.npz"
        save_to_npz(
            filename=      npz_fn,
            atomic_numbers=atomic_numbers_1d,
            positions=     P[sel_idxs],
            energies=      E[sel_idxs],
            forces=        F[sel_idxs]
        )

    # 12) Finally, optional random subsets on the inliers pool
    generate_random_subsets(
        "inliers_full_dataset.xyz",
        prefix,
        sizes,
        sampling=sampling,
        n_sets=n_sets,
        bootstrap_factor=bf
    )


# ───────────────────────────────────────────────────────────────
if __name__=="__main__":
    cfg = load_config("input.yaml")
    # MD-only quick first
    ds = cfg["dataset"]
    generate_md_random_subsets(
        ds["input_file"],
        ds["subset_counts"].get("MD",0),
        "md_random_dataset",
        ds["sizes"],
        sampling=ds.get("sampling","subsample"),
        n_sets=ds.get("n_sets",1),
        bootstrap_factor=ds.get("bootstrap_factor",1)
    )
    consolidate_dataset(cfg)

