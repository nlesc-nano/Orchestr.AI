import numpy as np
from typing import Sequence, Dict
from sklearn.cluster import KMeans
from ase.data import atomic_numbers as _ase_atomic_numbers
from mlff_qd.utils.io import (
    parse_stacked_xyz,
    save_stacked_xyz,
    save_to_npz,
)
from mlff_qd.utils.plots import plot_energy_and_forces

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
    
def generate_random_subsets(input_file: str,
                            output_prefix: str,
                            sizes: Sequence[int],
                            sampling: str="subsample",
                            n_sets: int=1,
                            bootstrap_factor: int=1):
    """
    Create `n_sets` subsets per size, via sampling mode, and report
    how many frames are duplicated (i.e. picked more than once).
    """
    energies, positions, forces, atom_types = parse_stacked_xyz(input_file)
    n_tot = len(energies)
    rng   = np.random.default_rng(42)

    # build once, outside the loop
    atomic_numbers = np.array(
        [_ase_atomic_numbers[s] for s in atom_types],
        dtype=np.int32
    )

    for n in sizes:
        for i in range(n_sets):
            idx = sample_indices(n_tot, n, sampling, bootstrap_factor, rng)

            # Diagnostics
            total_picks = len(idx)
            unique_picks = len(np.unique(idx))
            dup_picks   = total_picks - unique_picks
            mode_info = (f"{sampling.upper()}, bootstrap_factor={bootstrap_factor}"
                         if sampling=="bootstrap"
                         else "SUBSAMPLE")
            print(f"[INFO] Set {i+1}/{n_sets} for size={n}: mode={mode_info} → "
                  f"total picks={total_picks}, unique={unique_picks}, duplicates={dup_picks}")

            # Filenames
            tag = f"{output_prefix}_{n}_{sampling}_set{i+1}"
            if sampling=="bootstrap":
                tag += f"_bf{bootstrap_factor}"

            # save XYZ
            xyz_fn = f"{tag}.xyz"
            save_stacked_xyz(xyz_fn,
                             energies[idx], positions[idx], forces[idx], atom_types)

            # save NPZ with the new function
            npz_fn = f"{tag}.npz"
            save_to_npz(
                filename=       npz_fn,
                atomic_numbers= atomic_numbers,
                positions=      positions[idx],
                energies=       energies[idx],
                forces=         forces[idx]
            )

            # plot
            plot_energy_and_forces(energies[idx], forces[idx],
                                   f"plot_EF_{tag}.png")
            print(f"[OK]  Saved subset → {xyz_fn}\n")



def generate_md_random_subsets(input_file: str,
                               md_subset_size: int,
                               output_prefix: str,
                               sizes: Sequence[int],
                               sampling: str="subsample",
                               n_sets: int=1,
                               bootstrap_factor: int=1):
    """
    Subset only first md_subset_size frames, then delegate to generate_random_subsets.
    """
    energies, positions, forces, atom_types = parse_stacked_xyz(input_file)
    mdn = min(md_subset_size, len(energies))
    # write a temp truncated file or slice arrays directly
    # Here we slice arrays and loop
    tmp_file = None
    for n in sizes:
        generate_random_subsets(input_file, output_prefix, [n],
                                sampling=sampling, n_sets=n_sets,
                                bootstrap_factor=bootstrap_factor)

