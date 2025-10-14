#!/usr/bin/env python
# dataset_tools.py -----------------------------------------------------------

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

# ───────────────────────────────────────────────────────────────
# CONFIG I/O
# ───────────────────────────────────────────────────────────────
def load_config(config_file: str) -> Dict:
    """Load input parameters from a YAML configuration file."""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

import os
import numpy as np

def save_to_npz(
    filename: str,
    atomic_numbers: np.ndarray,          # (n_atoms,)  or (n_frames,n_atoms)
    positions:      np.ndarray,          # (N, n_atoms, 3)
    energies:       np.ndarray,          # (N,)         or list-like
    forces:         np.ndarray,          # (N, n_atoms, 3)
    cells:  np.ndarray = None,
    pbc:    np.ndarray = None,
):
    """
    Save a dataset exactly like your legacy exporter, but guarantee that
    every E[i] is a 1-element float-64 array so torch can infer dtype.
    """
    N, A, _ = positions.shape

    # ---- numeric arrays -------------------------------------------------
    R = np.asarray(positions, dtype=np.float32)          # (N,A,3)
    F = np.asarray(forces,    dtype=np.float32)          # (N,A,3)

    # Energies: (N,1) float64  →  row.data['E'] is 1-D, not scalar
    E = np.asarray(energies, dtype=np.float64)

    # Atomic numbers: 1-D (A,)
    z = np.asarray(atomic_numbers, dtype=np.int32)
    if z.ndim == 2:
        z = z[0]                 # order is identical, keep first row
    if z.ndim != 1 or z.size != A:
        raise ValueError(f"atomic_numbers must be 1-D of length {A}, got {z.shape}")

    # ---- assemble dict --------------------------------------------------
    base = {
        "type": "dataset",
        "name": os.path.splitext(os.path.basename(filename))[0],
        "R":    R,
        "z":    z,
        "E":    E,      # (N,1) float64  ← key point
        "F":    F,
        "F_min":  float(F.min()),  "F_max":  float(F.max()),
        "F_mean": float(F.mean()), "F_var":  float(F.var()),
        "E_min":  float(E.min()),  "E_max":  float(E.max()),
        "E_mean": float(E.mean()), "E_var":  float(E.var()),
    }
    if cells is not None: base["lattice"] = np.asarray(cells, dtype=np.float32)
    if pbc   is not None: base["pbc"]     = np.asarray(pbc,   dtype=bool)

    np.savez_compressed(filename, **base)

    print(f"[I/O] Saved {filename}")
    print(f"      R {R.shape}, z {z.shape}, E {E.shape}, F {F.shape}")


# ───────────────────────────────────────────────────────────────
# PLOTTING & ANALYSIS
# ───────────────────────────────────────────────────────────────
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
    print(f"[Plot] Energy/force plots saved to {filename}")


def analyze_fluctuations(energies, forces):
    """
    Analyze fluctuation to suggest loss weights.
    """
    energy_fluct = np.std(energies)
    force_fluct  = np.std(forces)
    wE_raw = force_fluct / energy_fluct
    wF_raw = 1.0
    S = wE_raw + wF_raw
    wE_norm = wE_raw/S
    wF_norm = wF_raw/S

    print(f"[Stats] σ(E)={energy_fluct:.4f}, σ(F)={force_fluct:.4f}")
    print(f"[Weights] raw: E={wE_raw:.4f}, F={wF_raw:.1f}; norm: E={wE_norm:.4f}, F={wF_norm:.4f}")
    return {'raw':{'energy':wE_raw,'forces':wF_raw},
            'normalized':{'energy':wE_norm,'forces':wF_norm}}

# ───────────────────────────────────────────────────────────────
# PARSING / FORMATTING
# ───────────────────────────────────────────────────────────────
def parse_stacked_xyz(filename):
    """
    Parse stacked XYZ returning (energies, positions, forces, atom_types).
    """
    energies, positions, forces, atom_types = [], [], [], []
    with open(filename,'r') as f:
        lines = f.readlines()
    idx = 0
    while idx < len(lines):
        n = int(lines[idx].strip()); idx+=1
        e = float(lines[idx].split()[0]); idx+=1
        fr_pos, fr_for = [], []
        if not atom_types:
            for i in range(n):
                parts = lines[idx].split()
                atom_types.append(parts[0])
                fr_pos.append([float(x) for x in parts[1:4]])
                fr_for.append([float(x) for x in parts[4:7]])
                idx+=1
        else:
            for i in range(n):
                parts = lines[idx].split()
                fr_pos.append([float(x) for x in parts[1:4]])
                fr_for.append([float(x) for x in parts[4:7]])
                idx+=1
        energies.append(e)
        positions.append(fr_pos)
        forces.append(fr_for)
    return (np.array(energies),
            np.array(positions),
            np.array(forces),
            atom_types)

def create_labels_from_counts(counts):
    """Turn [n1,n2,…] into [0×n1,1×n2,…]."""
    total = sum(counts)
    labels = np.empty(total,dtype=int)
    s=0
    for i,c in enumerate(counts):
        labels[s:s+c]=i
        s+=c
    return labels

# ───────────────────────────────────────────────────────────────
# PCA / OUTLIER / SOAP / SAVE XYZ
# ───────────────────────────────────────────────────────────────
def plot_pca(features, labels, title="PCA", filename="pca.png"):
    pca = PCA(2); red = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    cmap = ["blue","green","red","orange","purple","brown","pink","gray"]
    for lbl in np.unique(labels):
        m = (labels==lbl)
        plt.scatter(red[m,0],red[m,1],label=f"grp{lbl}",c=cmap[lbl%len(cmap)],alpha=0.7)
    plt.legend(); plt.title(title)
    plt.savefig(filename, dpi=300); plt.close()

def plot_outliers(features,labels,outliers,title,filename):
    pca = PCA(2); red = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    cmap = ["blue","green","red","orange"]
    for lbl in np.unique(labels):
        m_all = (labels==lbl)
        m_in = m_all & (outliers==1)
        m_out= m_all & (outliers==-1)
        plt.scatter(red[m_in,0],red[m_in,1],c=cmap[lbl % len(cmap)],label=f"{lbl} in")
        plt.scatter(red[m_out,0],red[m_out,1],c=cmap[lbl % len(cmap)],marker='x',s=50,label=f"{lbl} out")
    plt.title(title); plt.legend()
    plt.savefig(filename, dpi=300); plt.close()

def plot_final_selection(features,labels,sel,title,filename):
    pca = PCA(2); red = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    cmap = ["blue","green","red","orange"]
    for lbl in np.unique(labels):
        m = labels==lbl
        plt.scatter(red[m,0],red[m,1],c=cmap[lbl % len(cmap)],alpha=0.5,label=f"{lbl}")
    plt.scatter(red[sel,0],red[sel,1],facecolors='none',edgecolors='k',s=100,label='selected')
    plt.title(title); plt.legend()
    plt.savefig(filename, dpi=300); plt.close()

def save_stacked_xyz(filename, energies, positions, forces, atom_types):
    num_frames, num_atoms, _ = positions.shape
    with open(filename,'w') as f:
        for i in range(num_frames):
            f.write(f"{num_atoms}\n")
            f.write(f"{energies[i]:.6f}\n")
            for atom,(x,y,z),(fx,fy,fz) in zip(atom_types, positions[i], forces[i]):
                f.write(f"{atom:<2} {x:12.6f} {y:12.6f} {z:12.6f}"
                        f" {fx:12.6f} {fy:12.6f} {fz:12.6f}\n")

def compute_local_descriptors(positions, atom_types, soap):
    print("[SOAP] Computing descriptors...")
    desc=[]
    for i in range(positions.shape[0]):
        if i%100==0: print(f" SOAP frame {i+1}/{positions.shape[0]}")
        A = Atoms(symbols=atom_types, positions=positions[i], pbc=False)
        S = soap.create(A).mean(axis=0)
        desc.append(S)
    desc=np.array(desc)
    print(f"[SOAP] Done, shape={desc.shape}")
    return desc

def analyze_reference_forces(forces, atom_types):
    """
    Return dict of per-atom, per-frame, overall force stats.
    """
    fm = np.linalg.norm(forces, axis=2)

    per_atom_mean  = fm.mean(axis=0)
    per_atom_std   = fm.std(axis=0)
    per_atom_rng   = np.ptp(fm, axis=0)
    per_frame_mean = fm.mean(axis=1)
    per_frame_std  = fm.std(axis=1)
    per_frame_rng  = np.ptp(fm, axis=1)
    summary = {
        'per_atom_means':  per_atom_mean,
        'per_atom_stds':   per_atom_std,
        'per_atom_ranges': per_atom_rng,
        'per_frame_means': per_frame_mean,
        'per_frame_stds':  per_frame_std,
        'per_frame_ranges': per_frame_rng,
        'overall_mean':   fm.mean(),
        'overall_std':    fm.std(),
        'overall_range':  np.ptp(fm)      
    }
    # per-type
    for t in set(atom_types):
        idxs = [i for i, a in enumerate(atom_types) if a == t]
        arr  = fm[:, idxs]
        summary.setdefault('atom_type_means', {})[t]   = arr.mean()
        summary.setdefault('atom_type_stds',  {})[t]   = arr.std()
        summary.setdefault('atom_type_ranges', {})[t]  = np.ptp(arr)  # ← changed
    return summary

def suggest_thresholds(force_stats, std_fraction=0.1, range_fraction=0.1):
    overall_std   = force_stats['overall_std']
    overall_rng   = force_stats['overall_range']
    thr_std   = std_fraction  * overall_std
    thr_range = range_fraction* overall_rng
    print(f"[THR] Std thr={thr_std:.4f}, Range thr={thr_range:.4f}")
    per_type={}
    for t in force_stats['atom_type_stds']:
        ts = force_stats['atom_type_stds'][t]*std_fraction
        tr = force_stats['atom_type_ranges'][t]*range_fraction
        per_type[t]={'std_thr':ts,'range_thr':tr}
        print(f" {t}: std_thr={ts:.4f}, range_thr={tr:.4f}")
    return {'overall':{'std_thr':thr_std,'range_thr':thr_range},
            'per_type':per_type}

# ───────────────────────────────────────────────────────────────
# NEW: SAMPLING HELPERS
# ───────────────────────────────────────────────────────────────
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

# ───────────────────────────────────────────────────────────────
# UPDATED: RANDOM SUBSET GENERATORS
# ───────────────────────────────────────────────────────────────
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


def detect_outliers(features, contamination: float, labels, title: str, filename: str, random_state: int = 0):
    """
    IsolationForest-based outlier detection. Returns a boolean mask of inliers.
    Also renders the outlier plot using existing plot_outliers(...).
    """
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    y_pred = clf.fit_predict(features)          # -1 outlier, +1 inlier
    plot_outliers(features, labels, y_pred, title, filename)
    return (y_pred == 1)

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

