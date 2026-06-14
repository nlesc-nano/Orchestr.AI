"""
active_learning.py

Refactored in 2025: All legacy functions, PyTorch/HDBSCAN dependencies, 
and unused RDF calculations have been purged.

This module implements:
  1. Influence-based AL for the Validation Set.
  2. A highly modular Class-based Pool Active Learner for OOD sampling.
"""

import csv
import os
import time
import numpy as np
import scipy.optimize
import scipy.linalg
import scipy.spatial.distance
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from typing import Tuple, List, Optional, Dict, Any
from sklearn.isotonic import IsotonicRegression
from orchestr_ai.postprocessing.rdf import compute_rdf_thresholds_from_reference, fast_filter_by_rdf_kdtree, fast_filter_connectivity_and_arms

_GAUSSIAN_SIGMA_TO_ABS = float(np.sqrt(2.0 / np.pi))

def compute_soap_features(frames, train_frames=None, species=None, r_cut=4.0, n_max=4, l_max=4):
    """
    Computes averaged SOAP descriptors for a list of ASE Atoms objects.
    
    Parameters:
        frames (list): List of ASE Atoms objects.
        train_frames (list, optional): List of training ASE Atoms objects to gather chemical symbols.
        species (list, optional): Predefined list of species (chemical symbols).
        r_cut (float): Cutoff radius in Angstrom. Default 4.0.
        n_max (int): Number of radial basis functions. Default 4.
        l_max (int): Maximum degree of spherical harmonics. Default 4.
        
    Returns:
        tuple: (features_array, species_list) or (None, None) on failure.
    """
    try:
        from dscribe.descriptors import SOAP
        
        # 1. Determine chemical species if not provided
        if species is None:
            species_set = set()
            for fr in frames:
                species_set.update(fr.get_chemical_symbols())
            if train_frames is not None:
                for fr in train_frames:
                    species_set.update(fr.get_chemical_symbols())
            species = sorted(list(species_set))
            
        print(f"[SOAP] Computing descriptors for species: {species} (rcut={r_cut}, nmax={n_max}, lmax={l_max})")
        
        # 2. Construct SOAP descriptor
        soap = SOAP(
            species=species,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            periodic=False,     # Quantum dots in vacuum
            average="outer",    # Average SOAP over all atoms in the frame to get a per-frame descriptor
            sparse=False
        )
        
        # 3. Create SOAP vectors frame-by-frame to avoid multiprocessing hangs and show progress
        features_list = []
        n_frames = len(frames)
        print(f"[SOAP] Computing features sequentially for {n_frames} frames...")
        for idx, fr in enumerate(frames):
            feat = soap.create(fr)
            # Ensure it is at least a 1D/2D array
            feat_arr = np.asarray(feat)
            features_list.append(feat_arr)
            if (idx + 1) % max(1, n_frames // 10) == 0 or idx == n_frames - 1:
                print(f"  -> SOAP progress: {idx + 1}/{n_frames} frames completed...")
        
        features = np.vstack(features_list)
        
        # Make sure it's 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        return features, species
    except Exception as e:
        print(f"[SOAP] Warning: Failed to compute SOAP descriptors. Falling back to default latents. Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =============================================================================
# 1. MATH & LATENT SPACE UTILITIES
# =============================================================================

def _as_frame_arrays(values, *, frame_counts=None, name="values") -> List[np.ndarray]:
    """Return per-frame arrays, accepting either a list or a concatenated array."""
    if values is None:
        raise ValueError(f"{name} is required")

    if isinstance(values, (list, tuple)):
        return [np.asarray(v, dtype=float) for v in values]

    arr = np.asarray(values, dtype=float)
    if frame_counts is None:
        if arr.ndim < 3:
            raise ValueError(f"{name} needs frame_counts when passed as a concatenated array")
        return [np.asarray(v, dtype=float) for v in arr]

    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"{name} component array must be divisible by 3")
        arr = arr.reshape(-1, 3)
    if arr.shape[0] != int(np.sum(frame_counts)):
        raise ValueError(f"{name} length mismatch: got {arr.shape[0]}, expected {int(np.sum(frame_counts))}")

    splits = np.cumsum(frame_counts)[:-1]
    return [np.asarray(v, dtype=float) for v in np.split(arr, splits)]


def _frame_counts_from_arrays(arrays: List[np.ndarray]) -> np.ndarray:
    return np.array([np.asarray(a).shape[0] for a in arrays], dtype=float)


def _frame_force_max(arrays: List[np.ndarray]) -> np.ndarray:
    return np.array([np.nanmax(np.linalg.norm(a, axis=1)) if np.asarray(a).size else np.nan for a in arrays], dtype=float)


def _frame_sigma_max(arrays: List[np.ndarray]) -> np.ndarray:
    return np.array([np.nanmax(np.linalg.norm(a, axis=1)) if np.asarray(a).size else np.nan for a in arrays], dtype=float)


def _frame_sigma_mean_norm(arrays: List[np.ndarray]) -> np.ndarray:
    return np.array([np.nanmean(np.linalg.norm(a, axis=1)) if np.asarray(a).size else np.nan for a in arrays], dtype=float)

def calibrate_alpha_reg_gcv(
    F_eval: np.ndarray,
    y: np.ndarray,
    lambda_bounds: Tuple[float, float] = (1e-6, 1e4)
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate a ridge (GP) model via GCV and compute predictive variances."""
    n, d = F_eval.shape
    print(f"\n[GCV] Starting calibration. Matrix shape: {n} samples x {d} features")

    try:
        print("[GCV] Running SVD (Divide-and-Conquer)...")
        # Removed memory-heavy lapack_driver='gesvd', letting SciPy use efficient defaults
        U, s, _ = scipy.linalg.svd(F_eval, full_matrices=False)
        print("[GCV] SVD complete.")
    except MemoryError:
        print("[GCV] ERROR: Ran out of memory during SVD!")
        raise
        
    UTy = U.T @ y
    s2 = s**2

    def gcv_obj(log_lam):
        lam = np.exp(log_lam)
        a = s2 / (s2 + lam)
        df = a.sum()
        y_hat = (a * UTy) @ U.T
        resid = y - y_hat
        return np.log((resid @ resid) / (n - df)**2)

    print("[GCV] Optimizing ridge parameter...")
    res = scipy.optimize.minimize_scalar(gcv_obj, bounds=np.log(lambda_bounds), method='bounded')
    lam_opt = np.exp(res.x)

    print("[GCV] Building covariance matrix and running Cholesky...")
    A = F_eval.T @ F_eval + lam_opt * np.eye(d)

    base_jitter = 1e-8 * np.trace(A) / d
    jitter = 0.0
    for i in range(6): 
        try:
            L = np.linalg.cholesky(A + jitter * np.eye(d))
            break
        except np.linalg.LinAlgError:
            jitter = base_jitter * (10 ** i)
    else:
        raise np.linalg.LinAlgError(f"A not PD even after jitter up to {jitter:.1e}")

    print("[GCV] Computing latent variance terms...")
    G_eval = scipy.linalg.solve_triangular(L, F_eval.T, lower=True).T
    terms_lat = np.sum(G_eval**2, axis=1)

    a = s2 / (s2 + lam_opt)
    y_hat = (a * UTy) @ U.T
    resid_mean = np.mean((y - y_hat)**2)
    alpha_sq = resid_mean / np.mean(terms_lat)
    
    print("[GCV] Calibration successful.")
    return alpha_sq, lam_opt, terms_lat, G_eval, L

def d_optimal_full_order(X_cand: np.ndarray, X_train: np.ndarray, *, reg: float = 1e-6, verbose: bool = False):
    """Return *all* candidate indices in greedy D‑optimal order + γ for all."""
    m, d = X_cand.shape
    M_inv = np.linalg.inv(X_train.T @ X_train + reg * np.eye(d))

    quad0  = np.einsum("id,dk,ik->i", X_cand, M_inv, X_cand)
    gamma0 = np.sqrt(quad0)

    quad = quad0.copy()
    order, gains = [], []

    for _ in range(m):
        i_best = int(np.argmax(quad))
        gain   = np.log1p(quad[i_best])
        order.append(i_best)
        gains.append(gain)

        x = X_cand[i_best]
        v = M_inv @ x
        denom = 1.0 + x @ v
        M_inv -= np.outer(v, v) / denom

        alpha = X_cand @ v
        quad -= (alpha ** 2) / denom
        quad[i_best] = -np.inf  

    return np.asarray(order, int), np.asarray(gains, float), gamma0

# =============================================================================
# 2. IN-DISTRIBUTION ACTIVE LEARNING (Validation Set)
# =============================================================================

def adaptive_learning_ensemble_calibrated(
        all_frames: List, eval_mask: np.ndarray, delta_E_frame: np.ndarray, mean_l_al: np.ndarray, *,
        force_rmse_per_comp: Optional[np.ndarray] = None, denom_all: Optional[np.ndarray] = None,
        reference_frames: Optional[List] = None, beta: float = 0.5, drop_init: float = 1.0,
        min_k: int = 5, max_k: Optional[int] = None, score_floor: Optional[float] = None,
        base: str = "al_ens_v1", **kwargs) -> Tuple[List, np.ndarray]:
    
    eps = 1e-9
    train_idx, eval_idx = np.where(~eval_mask)[0], np.where(eval_mask)[0]
    atom_counts = np.array([len(fr) for fr in all_frames], dtype=float)
    if np.any(atom_counts <= 0):
        raise ValueError("All frames must contain at least one atom for active learning")
    delta_E_atom = np.asarray(delta_E_frame, dtype=float) / atom_counts
    
    # 1. RDF Filter
    if reference_frames:
        rdf_thresholds = compute_rdf_thresholds_from_reference(reference_frames)
        eval_frames_list = [all_frames[i] for i in eval_idx]
        realistic_mask = fast_filter_by_rdf_kdtree(eval_frames_list, rdf_thresholds)
    else:
        realistic_mask = np.ones(len(eval_idx), dtype=bool)

    # 2. Latent space calculations
    alpha_sq, lam_opt, terms_lat, G_all, L_E = calibrate_alpha_reg_gcv(mean_l_al, delta_E_atom)
    G_train, G_eval_E = G_all[train_idx], G_all[eval_idx]

    # 3. Setup force RMSEs
    if force_rmse_per_comp is not None:
        comps_pf = np.array([3*len(fr) for fr in all_frames], int)
        starts   = np.concatenate(([0], np.cumsum(comps_pf[:-1])))
        rmse_F_pf_max = np.maximum.reduceat(force_rmse_per_comp, starts)[:len(all_frames)]
        rmse_F_pf_mean = np.array([force_rmse_per_comp[starts[i]:starts[i] + comps_pf[i]].mean() for i in range(len(all_frames))])
    else:
        rmse_F_pf_max = rmse_F_pf_mean = np.zeros(len(all_frames))

    rmse_F_eval = rmse_F_pf_max[eval_idx]
    rmse_Fmean_eval = rmse_F_pf_mean[eval_idx]
    delta_E_eval = np.abs(delta_E_atom[eval_idx])

    # 4. Normalization and Ranking
    z_sigma = (delta_E_eval - delta_E_eval.mean()) / (delta_E_eval.std() + eps)
    z_rmse = (rmse_F_eval - rmse_F_eval.mean()) / (rmse_F_eval.std() + eps)
    u_frame_z = 0.5 * z_sigma + 0.5 * z_rmse 
    U_norm = (u_frame_z - u_frame_z.mean()) / (u_frame_z.std() + eps)
    
    order, gains_full, gamma0 = d_optimal_full_order(X_cand=G_eval_E, X_train=G_train)
    
    diversity_score = np.empty_like(gamma0)
    diversity_score[order] = gains_full
    D_norm = (diversity_score - diversity_score.mean()) / (diversity_score.std() + eps)

    # 5. Hybrid Score selection
    hybrid = 0.5 * U_norm + 0.5 * D_norm
    hybrid_floor = 0.5 * U_norm.mean() + 0.5 * D_norm.mean()

    keep_mask = (hybrid > hybrid_floor) & realistic_mask
    idx_keep = np.where(keep_mask)[0]
    
    k_budget = 250
    if len(idx_keep) > k_budget:
        top_rel = idx_keep[np.argsort(hybrid[idx_keep])[-k_budget:]]
    else:
        top_rel = idx_keep
        
    sel_idx = eval_idx[top_rel]
    return [all_frames[i] for i in sel_idx], sel_idx

# =============================================================================
# 3. POOL ACTIVE LEARNER (OOD Sampling)
# =============================================================================

class _PoolActiveLearner:
    """
    Object-Oriented orchestrator for Pool-Based Active Learning.
    Cleanly segments Thresholding, Window Evaluation, and Diagnostics.
    """
    def __init__(self, **kwargs):
        self.base = kwargs.get("base", "al_pool")
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.all_frame_records = {}
        self.records_tmp = []
        self.final_pool_indices = []
        self.sel_frames = []
        
        self.n_uncertain_total = 0
        self.n_gamma_gate_total = 0
        self.n_ood_risk_total = 0

    def run(self):
        print(f"\n[AL] --- PoolActiveLearner Orchestrator ---")
        self._setup_latent_space()
        self._setup_thresholds()

        # Convergence Check
        if self.n_hi_total < 10:
            print("[AL] Convergence heuristic: fewer than 10 frames exceed any lower bound.")
            print("[AL] Nothing significant left to label.")
            self._evaluate_windows()
            self.final_pool_indices = []
            self.sel_frames = []
        else:
            self._evaluate_windows()
            self._finalize_selection()

        self._write_diagnostics()
        self._collect_csv_diagnostics()
        self._print_summary()
        return self.sel_frames, self.final_pool_indices

    def _threshold_metadata(self):
        return {
            "thr_sigma_E_low": getattr(self, "thr_sigma_E_low", np.nan),
            "thr_sigma_E_hi_eff": getattr(self, "thr_sigma_E_hi_eff", np.nan),
            "thr_sigma_F": getattr(self, "thr_sigma_F", np.nan),
            "thr_sigma_F_hi_eff": getattr(self, "thr_sigma_F_hi_eff", np.nan),
            "thr_sigma_Fmean": getattr(self, "thr_sigma_Fmean", np.nan),
            "thr_sigma_Fmean_hi_eff": getattr(self, "thr_sigma_Fmean_hi_eff", np.nan),
            "thr_Fmag": getattr(self, "thr_Fmag", np.nan),
            "thr_Fmag_hi_eff": getattr(self, "thr_Fmag_hi_eff", np.nan),
            "hard_sigma_E_atom_min": getattr(self, "hard_sigma_E_atom_min", np.nan),
            "hard_sigma_F_mean_min": getattr(self, "hard_sigma_F_mean_min", np.nan),
            "hard_sigma_F_max_min": getattr(self, "hard_sigma_F_max_min", np.nan),
            "train_Fmax_hard_cap": getattr(self, "train_Fmax_hard_cap", np.nan),
            "calibration_support_fraction": float(np.mean(self.calibration_in_support)) if hasattr(self, "calibration_in_support") else np.nan,
        }

    def _csv_diagnostic_rows(self):
        state = str(getattr(self, "state", getattr(self, "base", "unknown")))
        pool_indices = getattr(self, "pool_indices", None)
        if pool_indices is None:
            pool_indices = np.arange(len(self.pool_frames), dtype=int)
        pool_indices = np.asarray(pool_indices, dtype=int)
        shortlist_set = set(self.final_pool_indices)
        rows = []
        for pidx in sorted(self.all_frame_records.keys()):
            R = self.all_frame_records[pidx]
            n_atoms = float(self.pool_atom_counts[pidx]) if hasattr(self, "pool_atom_counts") else float(len(self.pool_frames[pidx]))
            rows.append({
                "state": state,
                "idx": int(pool_indices[pidx]) if pidx < len(pool_indices) else int(pidx),
                "pool_row": int(pidx),
                "window": R["window"],
                "n_atoms": int(n_atoms),
                "geom_ok": int(R["rdf_ok"]),
                "caps_ok": int(R["pass_caps"]),
                "force_inf": int(R["force_inf"]),
                "gamma_gate": int(R["gamma_gate"]),
                "gamma0": float(R["gamma0"]),
                "dM": float(R["dM"]),
                "Dgain": float(R["dgain_train"]),
                "raw_score": float(R["raw_score_window"]),
                "E_pred": float(R["mu_E"]),
                "E_pred_atom": float(R["mu_E_atom"]),
                "sigma_E": float(R["sigma_E"]),
                "sigma_E_atom": float(R["sigma_E_atom"]),
                "sigma_F_max": float(R["sigma_F_max"]),
                "sigma_F_mean": float(R["sigma_F_mean"]),
                "Eabs_exp": float(R["exp_abs_E_atom"]),
                "Fabs_mean": float(R["exp_abs_F_mean"]),
                "Fabs_max": float(R["exp_abs_F_max"]),
                "cal_ok": int(R["cal_support"]),
                "ood": int(R["ood_risk"]),
                "Fmax": float(R["Fmax"]),
                "Fmean": float(R["Fmean"]),
                "selected": int(R["selected"]),
                "shortlist": int(pidx in shortlist_set),
            })
        return rows

    def _collect_csv_diagnostics(self):
        collector = getattr(self, "diagnostics_collector", None)
        if collector is None:
            return
        collector.append({
            "state": str(getattr(self, "state", getattr(self, "base", "unknown"))),
            "thresholds": self._threshold_metadata(),
            "rows": self._csv_diagnostic_rows(),
        })

    def _setup_latent_space(self):
        self.G_train = scipy.linalg.solve_triangular(self.L, self.F_train.T, lower=True).T
        self.G_pool  = scipy.linalg.solve_triangular(self.L, self.F_pool.T,  lower=True).T

        self.M_inv_global = np.linalg.inv(self.G_train.T @ self.G_train + 1e-6 * np.eye(self.G_train.shape[1]))
        
        gamma_train = np.sqrt(np.einsum('id,dk,ik->i', self.G_train, self.M_inv_global, self.G_train))
        self.gamma_thr = np.quantile(gamma_train, self.percentile_gamma / 100.0)

        self.mu_Gtrain = self.G_train.mean(axis=0)
        Cov_Gtrain = np.cov(self.G_train, rowvar=False) + 1e-6 * np.eye(self.G_train.shape[1])
        self.Cov_inv = np.linalg.inv(Cov_Gtrain)
        
        diff_train = self.G_train - self.mu_Gtrain
        dM_train = np.sqrt(np.einsum("id,dk,ik->i", diff_train, self.Cov_inv, diff_train))
        self.dM_thr = np.quantile(dM_train, 99.0 / 100.0)

    def _setup_thresholds(self):
        self.forces_train_frames = _as_frame_arrays(self.forces_train, name="forces_train")
        self.train_atom_counts = _frame_counts_from_arrays(self.forces_train_frames)
        self.pool_atom_counts = np.array([len(fr) for fr in self.pool_frames], dtype=float)
        self.calibration_in_support = np.asarray(
            getattr(self, "calibration_in_support", np.ones(len(self.pool_frames), dtype=bool)),
            dtype=bool,
        )
        self.ood_risk_mask = np.asarray(
            getattr(self, "ood_risk_mask", ~self.calibration_in_support),
            dtype=bool,
        )
        self.expected_abs_E_atom = np.asarray(
            getattr(self, "expected_abs_E_atom", np.full(len(self.pool_frames), np.nan)),
            dtype=float,
        )
        self.expected_abs_F_mean = np.asarray(
            getattr(self, "expected_abs_F_mean", np.full(len(self.pool_frames), np.nan)),
            dtype=float,
        )
        self.expected_abs_F_max = np.asarray(
            getattr(self, "expected_abs_F_max", np.full(len(self.pool_frames), np.nan)),
            dtype=float,
        )
        if not (
            len(self.calibration_in_support)
            == len(self.ood_risk_mask)
            == len(self.expected_abs_E_atom)
            == len(self.expected_abs_F_mean)
            == len(self.expected_abs_F_max)
            == len(self.pool_frames)
        ):
            raise ValueError("Calibration diagnostic arrays must contain one value per pool frame")
        self.n_atoms_pool = int(np.median(self.pool_atom_counts))
        self.large_cluster_threshold = int(getattr(self, "large_cluster_threshold", 300))
        if getattr(self, "surface_relax_factor", None) is None:
            self.surface_relax_factor = (
                1.5 if self.n_atoms_pool > self.large_cluster_threshold else 1.0
            )
        print(
            f"[AL] Pool size: median n_atoms={self.n_atoms_pool}, "
            f"surface_relax_factor={self.surface_relax_factor} "
            f"(threshold>{self.large_cluster_threshold})"
        )

        self.sigma_force_frames = _as_frame_arrays(
            self.sigma_force,
            frame_counts=self.train_atom_counts.astype(int),
            name="sigma_force",
        )
        has_pool_force_summaries = all(
            hasattr(self, name)
            for name in ("sigma_F_pool_mean", "sigma_F_pool_max", "frame_max_force_pool")
        )

        if has_pool_force_summaries:
            self.sigma_F_pool_mean = np.asarray(self.sigma_F_pool_mean, dtype=float)
            self.sigma_F_pool_max = np.asarray(self.sigma_F_pool_max, dtype=float)
            self.frame_max_force_pool = np.asarray(self.frame_max_force_pool, dtype=float)
            self.frame_mean_force_pool = np.asarray(
                getattr(self, "frame_mean_force_pool", np.full(len(self.pool_frames), np.nan)),
                dtype=float,
            )

            if not (
                len(self.sigma_F_pool_mean)
                == len(self.sigma_F_pool_max)
                == len(self.frame_max_force_pool)
                == len(self.frame_mean_force_pool)
                == len(self.pool_frames)
            ):
                raise ValueError("Pool force summary arrays must contain one value per pool frame")
        else:
            self.mu_F_pool_frames = _as_frame_arrays(
                self.mu_F_pool,
                frame_counts=self.pool_atom_counts.astype(int),
                name="mu_F_pool",
            )
            self.sigma_F_pool_frames = _as_frame_arrays(
                self.sigma_F_pool,
                frame_counts=self.pool_atom_counts.astype(int),
                name="sigma_F_pool",
            )

        if len(self.sigma_energy) != len(self.train_atom_counts):
            raise ValueError(
                f"sigma_energy must contain one value per training frame; got {len(self.sigma_energy)} "
                f"for {len(self.train_atom_counts)} training frames"
            )

        # Energies
        self.mu_E_atom_train = np.asarray(self.mu_E_frame_train, dtype=float) / self.train_atom_counts
        self.sigma_E_atom_train = np.asarray(self.sigma_energy, dtype=float) / self.train_atom_counts
        self.mu_E_atom_pool = np.asarray(self.mu_E_pool, dtype=float) / self.pool_atom_counts
        self.sigma_E_atom_pool = np.asarray(self.sigma_E_pool, dtype=float) / self.pool_atom_counts

        self.thr_E_hi_atom = self.mu_E_atom_train.max() + 0.5
        self.E_lo_pool_atom = (self.mu_E_pool - 3.0 * self.sigma_E_pool) / self.pool_atom_counts
        self.E_hi_pool_atom = (self.mu_E_pool + 3.0 * self.sigma_E_pool) / self.pool_atom_counts

        # Forces
        self.sigma_F_train_max = _frame_sigma_max(self.sigma_force_frames)
        self.sigma_F_train_mean = _frame_sigma_mean_norm(self.sigma_force_frames)
        self.frame_max_force_train = _frame_force_max(self.forces_train_frames)

        if not has_pool_force_summaries:
            self.sigma_F_pool_max = _frame_sigma_max(self.sigma_F_pool_frames)
            self.sigma_F_pool_mean = _frame_sigma_mean_norm(self.sigma_F_pool_frames)
            self.frame_max_force_pool = _frame_force_max(self.mu_F_pool_frames)
            self.frame_mean_force_pool = _frame_sigma_mean_norm(self.mu_F_pool_frames)

        # Baseline Thresholds (optional stratification by cluster size for mixed train sets)
        stratify = bool(getattr(self, "stratify_train_by_size", False))
        size_split = int(getattr(self, "size_split_atoms", self.large_cluster_threshold))
        train_ref = np.ones(len(self.train_atom_counts), dtype=bool)
        if stratify:
            n_large = int((self.train_atom_counts >= size_split).sum())
            n_small = int((self.train_atom_counts < size_split).sum())
            if self.n_atoms_pool > size_split and n_large >= 20:
                train_ref = self.train_atom_counts >= size_split
                print(
                    f"[AL] Stratified train thresholds: large subset "
                    f"({train_ref.sum()} frames, n>={size_split})"
                )
            elif self.n_atoms_pool <= size_split and n_small >= 20:
                train_ref = self.train_atom_counts < size_split
                print(
                    f"[AL] Stratified train thresholds: small subset "
                    f"({train_ref.sum()} frames, n<{size_split})"
                )
            else:
                print(
                    f"[AL] Stratify requested but insufficient frames "
                    f"(large={n_large}, small={n_small}); using all train frames."
                )

        self._train_ref_mask = train_ref
        e_ref = self.sigma_E_atom_train[train_ref]
        fmax_ref = self.sigma_F_train_max[train_ref]
        fmean_ref = self.sigma_F_train_mean[train_ref]
        fm_ref = self.frame_max_force_train[train_ref]

        self.thr_sigma_E_low = np.percentile(e_ref, self.percentile_F_low)
        self.thr_sigma_F = np.percentile(fmax_ref, self.percentile_F_low)
        self.thr_sigma_Fmean = np.percentile(fmean_ref, self.percentile_F_low)
        self.thr_Fmag = np.percentile(fm_ref, self.percentile_F_low)
        self.train_Fmax_hard_cap = float(fm_ref.max()) * float(self.hard_Fmax_train_mult)

        self.user_hard_sigma_E_atom_min = float(getattr(self, "hard_sigma_E_atom_min", 0.001))
        self.user_hard_sigma_F_mean_min = float(getattr(self, "hard_sigma_F_mean_min", 0.075))
        self.user_hard_sigma_F_max_min = float(getattr(self, "hard_sigma_F_max_min", 0.15))

        if bool(getattr(self, "hard_floors_from_calibrated_train", False)):
            pct = float(self.percentile_F_low)
            self.hard_sigma_E_atom_min = float(
                max(self.hard_sigma_E_atom_min, np.percentile(e_ref, pct))
            )
            self.hard_sigma_F_mean_min = float(
                max(self.hard_sigma_F_mean_min, np.percentile(fmean_ref, pct))
            )
            self.hard_sigma_F_max_min = float(
                max(self.hard_sigma_F_max_min, np.percentile(fmax_ref, pct))
            )
            print(
                "[AL] Hard floors raised from calibrated train percentiles "
                f"(p{pct}): σE>={self.hard_sigma_E_atom_min:.4g}, "
                f"σF_mean>={self.hard_sigma_F_mean_min:.4g}, "
                f"σF_max>={self.hard_sigma_F_max_min:.4g}"
            )

        # Apply RDF Filter (Catches overlaps)
        self.rdf_ok_mask = fast_filter_by_rdf_kdtree(self.pool_frames, self.rdf_thresholds)
    
        # ---> NEW: Apply Fully Automated Connectivity & Arm Filter <---
        # Fetch configurations (with safe fallbacks)
        margin = getattr(self, 'detachment_margin', 0.8)
        arm_tol = float(getattr(self, 'arm_tolerance', 0.5))

        # Update the mask using our dedicated geometric function
        self.rdf_ok_mask = fast_filter_connectivity_and_arms(
            frames=self.pool_frames, 
            ok_mask=self.rdf_ok_mask, 
            margin=margin, 
            arm_tol=arm_tol,
            verbose=True
        )

        # ---------------------------------------------------------
        ok_idx = np.where(self.rdf_ok_mask)[0]

        # Calculate Adaptive Upper Caps from OK frames
        if len(ok_idx) > 0:
            # ---> NEW: Define a "Calibration Subset" of strictly healthy frames <---
            # Filter to only include frames whose predicted energy per atom is roughly 
            # within the bounds of the training data. Distorting systems spike in energy.
            E_train_min = self.mu_E_atom_train.min()
            E_train_max = self.mu_E_atom_train.max()
            allowed_buffer = 0.1  # Allow up to 100 meV/atom drift for calibration
            
            calib_mask = self.rdf_ok_mask & \
                         (self.mu_E_atom_pool >= E_train_min - allowed_buffer) & \
                         (self.mu_E_atom_pool <= E_train_max + allowed_buffer)
            
            calib_idx = np.where(calib_mask)[0]
            
            # If the trajectory blew up instantly, fallback to the first 10% of the trajectory
            if len(calib_idx) < 20:
                print(f"[AL] Warning: Few frames match training energy. Calibrating caps using early stable frames.")
                calib_idx = ok_idx[:max(20, len(ok_idx)//10)]
            else:
                print(f"[AL] Calibrating adaptive caps using {len(calib_idx)} energy-stable inlier frames.")

            # 1. Calculate the adaptive percentile from the CALIBRATION pool only
            pool_E_hi = np.percentile(self.sigma_E_atom_pool[calib_idx], self.percentile_F_hi)
            pool_F_hi = np.percentile(self.sigma_F_pool_max[calib_idx], self.percentile_F_hi)
            pool_Fmean_hi = np.percentile(self.sigma_F_pool_mean[calib_idx], self.percentile_F_hi)
            pool_Fmag_hi = np.percentile(self.frame_max_force_pool[calib_idx], self.percentile_F_hi)

            # 2. Hard caps relative to the training data max uncertainty.
            # Force uncertainty ceilings are anchored to the hard AL floors so
            # broken trajectory frames cannot inflate the physically useful cap.
            floor_E = getattr(self, 'hard_sigma_E_atom_min', 0.0)
            floor_Fmax = getattr(self, 'hard_sigma_F_max_min', 0.0)
            floor_Fmean = getattr(self, 'hard_sigma_F_mean_min', 0.0)
            floor_E_user = getattr(self, 'user_hard_sigma_E_atom_min', 0.001)
            floor_Fmax_user = getattr(self, 'user_hard_sigma_F_max_min', 0.15)
            floor_Fmean_user = getattr(self, 'user_hard_sigma_F_mean_min', 0.075)

            unc_mult = 15.0
            max_allowed_E_hi = max(self.sigma_E_atom_train.max() * unc_mult, floor_E * 5.0)
            max_allowed_Fmag_hi = max(self.frame_max_force_train.max() * 5.0, 20.0)
            force_hi_mult = 3.0

            # 3. Final effective thresholds (bounded by the hard caps)
            # Decoupled from pool percentiles to prevent outlier contamination
            self.thr_sigma_E_hi_eff = floor_E_user * 2.5
            self.thr_sigma_F_hi_eff = floor_Fmax_user * 2.5
            self.thr_sigma_Fmean_hi_eff = floor_Fmean_user * 2.5
            self.thr_Fmag_hi_eff = min(max(pool_Fmag_hi, self.thr_Fmag * 2.0), max_allowed_Fmag_hi)

            # Ensure it never drops below the absolute low percentiles either
            self.thr_sigma_E_hi_eff = max(self.thr_sigma_E_low, self.thr_sigma_E_hi_eff)
            self.thr_sigma_F_hi_eff = max(self.thr_sigma_F, self.thr_sigma_F_hi_eff)
            self.thr_sigma_Fmean_hi_eff = max(self.thr_sigma_Fmean, self.thr_sigma_Fmean_hi_eff)
            self.thr_Fmag_hi_eff = max(self.thr_Fmag, self.thr_Fmag_hi_eff)

            self.allowed_offset_eff = max(0.0, np.percentile(self.mu_E_atom_pool[calib_idx], 5) - np.percentile(self.mu_E_atom_train, 95)) + 0.05

        else:
            floor_E_user = getattr(self, 'user_hard_sigma_E_atom_min', 0.001)
            floor_Fmax_user = getattr(self, 'user_hard_sigma_F_max_min', 0.15)
            floor_Fmean_user = getattr(self, 'user_hard_sigma_F_mean_min', 0.075)

            self.thr_sigma_E_hi_eff = max(self.thr_sigma_E_low, floor_E_user * 2.5)
            self.thr_sigma_F_hi_eff = max(self.thr_sigma_F, floor_Fmax_user * 2.5)
            self.thr_sigma_Fmean_hi_eff = max(self.thr_sigma_Fmean, floor_Fmean_user * 2.5)
            self.thr_Fmag_hi_eff = max(self.thr_Fmag, 2.0 * self.frame_max_force_train.max())
            self.allowed_offset_eff = 2.0 / float(np.nanmedian(self.train_atom_counts))

        # Count coverage
        n_hi_E = (self.sigma_E_atom_pool > self.thr_sigma_E_low).sum()
        n_hi_Fmax = (self.sigma_F_pool_max > self.thr_sigma_F).sum()
        n_hi_Fmean = (self.sigma_F_pool_mean > self.thr_sigma_Fmean).sum()
        n_hi_Fmag = (self.frame_max_force_pool > self.thr_Fmag).sum()
        n_hi_ood = (self.ood_risk_mask & self.rdf_ok_mask).sum()
        self.n_hi_total = n_hi_E + n_hi_Fmax + n_hi_Fmean + n_hi_Fmag + n_hi_ood

    def _evaluate_windows(self):
        n_pool = len(self.pool_frames)
        print(f"[AL] Evaluating {n_pool} pool frames in windows of size {self.window_size}...")

        for w0 in range(0, n_pool, self.window_size):
            win = list(range(w0, min(w0 + self.window_size, n_pool)))
            win_end = win[-1] + 1

            # --- Explicit Filtering with Counters ---
            win_good = [i for i in win if self.rdf_ok_mask[i]]
            
            win_E, win_CI, win_phys = [], [], []
            drop_E_hi = drop_CI = drop_sE_hi = drop_sFmax_hi = drop_sFmean_hi = drop_Fmag_hi = 0

            for i in win_good:
                # 1. Energy Mean check
                #if self.mu_E_atom_pool[i] >= self.thr_E_hi_atom:
                #    drop_E_hi += 1
                #    continue
                #win_E.append(i)
                
                # 2. Confidence Interval overlap check
                #if not ((self.E_hi_pool_atom[i] >= (self.mu_E_atom_train.min() - self.allowed_offset_eff)) and
                #        (self.E_lo_pool_atom[i] <= (self.mu_E_atom_train.max() + self.allowed_offset_eff))):
                #    drop_CI += 1
                #    continue
                #win_CI.append(i)
                
                # 3. Physics / Uncertainty upper bounds (The "Ceiling")
                if self.sigma_E_atom_pool[i] >= self.thr_sigma_E_hi_eff:
                    drop_sE_hi += 1
                    continue
                if self.sigma_F_pool_max[i] >= self.thr_sigma_F_hi_eff:
                    drop_sFmax_hi += 1
                    continue
                if self.sigma_F_pool_mean[i] >= self.thr_sigma_Fmean_hi_eff:
                    drop_sFmean_hi += 1
                    continue
                if self.frame_max_force_pool[i] >= self.thr_Fmag_hi_eff:
                    drop_Fmag_hi += 1
                    continue
                    
                win_phys.append(i)
            # ----------------------------------------

            # Hard Triggers (Uncertainty minimums - The "Floor")
            # Relax per-atom max-force sigma on large clusters so surface spikes do not flood triage.
            _large_thr = int(getattr(self, "large_cluster_threshold", 300))
            _srf = float(
                getattr(
                    self,
                    "surface_relax_factor",
                    1.5 if self.n_atoms_pool > _large_thr else 1.0,
                )
            )
            _thr_fmax = self.hard_sigma_F_max_min * _srf
            high = [
                i for i in win_phys
                if self.frame_max_force_pool[i] <= self.train_Fmax_hard_cap
                and (
                    self.sigma_E_atom_pool[i] >= self.hard_sigma_E_atom_min
                    or self.sigma_F_pool_mean[i] >= self.hard_sigma_F_mean_min
                    or self.sigma_F_pool_max[i] >= _thr_fmax
                )
            ]
            ood_high = [
                i for i in win_phys
                if self.ood_risk_mask[i]
                and self.frame_max_force_pool[i] <= self.train_Fmax_hard_cap
                and (
                    self.sigma_E_atom_pool[i] >= self.hard_sigma_E_atom_min
                    or self.sigma_F_pool_mean[i] >= self.hard_sigma_F_mean_min
                    or self.sigma_F_pool_max[i] >= _thr_fmax
                )
            ]
            if ood_high:
                high = sorted(set(high).union(ood_high))

            # Process Window Metrics
            win_all = np.array(win, dtype=int)
            sub_G_all = self.G_pool[win_all]
            quad_all = np.einsum("id,dk,ik->i", sub_G_all, self.M_inv_global, sub_G_all)
            gamma_all = np.sqrt(quad_all)
            diff_all = sub_G_all - self.mu_Gtrain
            dM_all = np.sqrt(np.einsum("id,dk,ik->i", diff_all, self.Cov_inv, diff_all))

            keep_gamma_mask = (gamma_all > self.gamma_thr)
            cand_mask = np.isin(win_all, high) & keep_gamma_mask

            self.n_uncertain_total += len(high)
            self.n_ood_risk_total += len(ood_high)
            self.n_gamma_gate_total += cand_mask.sum()

            cand_idx_local = np.where(cand_mask)[0]
            selected_local = []

            # D-Optimal Selection
            if cand_idx_local.size > 0:
                X_cand = sub_G_all[cand_idx_local]
                order, gains, _ = d_optimal_full_order(X_cand, self.G_train)

                # Intra-batch diversity (column-standardized distances + decay kernel)
                X_cand_scaled = X_cand.astype(float, copy=True)
                col_std = X_cand_scaled.std(axis=0)
                col_std[col_std < 1e-9] = 1.0
                X_cand_scaled = (X_cand_scaled - X_cand_scaled.mean(axis=0)) / col_std
                dist_mat = np.linalg.norm(
                    X_cand_scaled[:, None, :] - X_cand_scaled[None, :, :], axis=2
                )
                pos_dists = dist_mat[dist_mat > 0]
                tau = max(float(np.median(pos_dists)), 1e-9) if pos_dists.size else 1.0
                remaining = list(range(X_cand.shape[0]))

                while len(selected_local) < self.min_k and remaining:
                    best_score, best_r = -np.inf, None
                    for r in remaining:
                        min_dist = (
                            1.0
                            if not selected_local
                            else float(np.min(dist_mat[r, selected_local]))
                        )
                        div_weight = 1.0 if not selected_local else float(1.0 - np.exp(-min_dist / tau))
                        div_weight = max(div_weight, 1e-6)
                        score = gains[r] * div_weight
                        if score > best_score:
                            best_score, best_r = score, r
                    selected_local.append(best_r)
                    remaining.remove(best_r)

            picks_abs = win_all[cand_idx_local[selected_local]].tolist() if selected_local else []

            # --- NEW INFORMATIVE PRINT LOGGING WITH REJECTION REASONS ---
            print(f"  -> Window [{w0:4d} - {win_end:4d}] | "
                  f"GeomOK: {len(win_good):3d} | PhysOK: {len(win_phys):3d} | "
                  f"Uncertain: {len(high):3d} | OOD: {len(ood_high):3d} | "
                  f"Novel: {cand_mask.sum():3d} | Sel: {len(selected_local):2d}")
            if len(win_good) > len(win_phys):
                reasons = []
                if drop_E_hi: reasons.append(f"Energy Ceiling={drop_E_hi}")
                if drop_CI: reasons.append(f"CI Overlap={drop_CI}")
                if drop_sE_hi: reasons.append(f"σE Ceiling={drop_sE_hi}")
                if drop_sFmax_hi: reasons.append(f"σFmax Ceiling={drop_sFmax_hi}")
                if drop_sFmean_hi: reasons.append(f"σFmean Ceiling={drop_sFmean_hi}")
                if drop_Fmag_hi: reasons.append(f"Force Ceiling={drop_Fmag_hi}")
                print(f"       Ceiling Drops: {', '.join(reasons)}")
            if len(high) > 0:
                trig_sE = sum(1 for i in win_phys if self.frame_max_force_pool[i] <= self.train_Fmax_hard_cap and self.sigma_E_atom_pool[i] >= self.hard_sigma_E_atom_min)
                trig_sFmean = sum(1 for i in win_phys if self.frame_max_force_pool[i] <= self.train_Fmax_hard_cap and self.sigma_F_pool_mean[i] >= self.hard_sigma_F_mean_min)
                trig_sFmax = sum(1 for i in win_phys if self.frame_max_force_pool[i] <= self.train_Fmax_hard_cap and self.sigma_F_pool_max[i] >= _thr_fmax)
                trig_ood = len(ood_high)
                
                reasons_trig = []
                if trig_sE: reasons_trig.append(f"σE Floor={trig_sE}")
                if trig_sFmax: reasons_trig.append(f"σFmax Floor={trig_sFmax}")
                if trig_sFmean: reasons_trig.append(f"σFmean Floor={trig_sFmean}")
                if trig_ood: reasons_trig.append(f"OOD={trig_ood}")
                print(f"       Floor Triggers: {', '.join(reasons_trig)}")
            # ------------------------------------------------------------

            # Save Records
            ood_high_set = set(ood_high)
            for j_local, pidx in enumerate(win_all):
                self.all_frame_records[pidx] = {
                    "pool_idx": pidx, "window": f"{w0}-{win[-1]+1}",
                    "rdf_ok": pidx in win_good, "pass_caps": pidx in win_phys,
                    "force_inf": pidx in high, "gamma_gate": keep_gamma_mask[j_local],
                    "gamma0": gamma_all[j_local], "dM": dM_all[j_local],
                    "dgain_train": np.log1p(quad_all[j_local]),
                    "raw_score_window": gamma_all[j_local] * np.log1p(quad_all[j_local]),
                    "mu_E": self.mu_E_pool[pidx],
                    "sigma_E": self.sigma_E_pool[pidx],
                    "sigma_E_atom": self.sigma_E_atom_pool[pidx],
                    "sigma_F_max": self.sigma_F_pool_max[pidx],
                    "sigma_F_mean": self.sigma_F_pool_mean[pidx],
                    "exp_abs_E_atom": self.expected_abs_E_atom[pidx],
                    "exp_abs_F_mean": self.expected_abs_F_mean[pidx],
                    "exp_abs_F_max": self.expected_abs_F_max[pidx],
                    "cal_support": self.calibration_in_support[pidx],
                    "ood_risk": pidx in ood_high_set,
                    "Fmax": self.frame_max_force_pool[pidx],
                    "Fmean": self.frame_mean_force_pool[pidx],
                    "mu_E_atom": self.mu_E_atom_pool[pidx],
                    "selected": pidx in picks_abs
                }

            for r_sel in selected_local:
                j_local = cand_idx_local[r_sel]
                pidx = win_all[j_local]
                self.records_tmp.append({
                    "pool_idx": pidx, "raw_score": gamma_all[j_local] * np.log1p(quad_all[j_local])
                })

    def _finalize_selection(self):
        unique_map = {}
        for rec in self.records_tmp:
            idx = rec["pool_idx"]
            if idx not in unique_map or rec["raw_score"] > unique_map[idx]["raw_score"]:
                unique_map[idx] = rec

        unique_records = sorted(unique_map.values(), key=lambda x: x["raw_score"], reverse=True)
        if self.budget_max and len(unique_records) > self.budget_max:
            unique_records = unique_records[:self.budget_max]

        self.final_pool_indices = [r["pool_idx"] for r in unique_records]
        self.sel_frames = [self.pool_frames[i] for i in self.final_pool_indices]

    def _write_diagnostics(self):
        with open(f"{self.base}_per_frame_diagnostics.txt", "w") as fh:
            fh.write("# Pool Active Learning Diagnostics\n")
            
            # --- 1. Write Thresholds for the Plotting Script ---
            fh.write(f"# thr_sigma_E_low        = {self.thr_sigma_E_low:.6f}\n")
            fh.write(f"# thr_sigma_E_hi_eff     = {self.thr_sigma_E_hi_eff:.6f}\n")
            fh.write(f"# thr_sigma_F            = {self.thr_sigma_F:.6f}\n")
            fh.write(f"# thr_sigma_F_hi_eff     = {self.thr_sigma_F_hi_eff:.6f}\n")
            fh.write(f"# thr_sigma_Fmean        = {self.thr_sigma_Fmean:.6f}\n")
            fh.write(f"# thr_sigma_Fmean_hi_eff = {self.thr_sigma_Fmean_hi_eff:.6f}\n")
            fh.write(f"# thr_Fmag               = {self.thr_Fmag:.6f}\n")
            fh.write(f"# thr_Fmag_hi_eff        = {self.thr_Fmag_hi_eff:.6f}\n")
            fh.write(f"# hard_sigma_E_atom_min  = {getattr(self, 'hard_sigma_E_atom_min', np.nan):.6f}\n")
            fh.write(f"# hard_sigma_F_mean_min  = {getattr(self, 'hard_sigma_F_mean_min', np.nan):.6f}\n")
            fh.write(f"# hard_sigma_F_max_min   = {getattr(self, 'hard_sigma_F_max_min', np.nan):.6f}\n")
            fh.write(f"# train_Fmax_hard_cap    = {self.train_Fmax_hard_cap:.6f}\n")
            fh.write(f"# calibration_support_fraction = {float(np.mean(self.calibration_in_support)):.6f}\n")
            fh.write("# ----------------------------------------\n\n")

            # --- 2. Write Pool Diagnostics ---
            fh.write(f"{'idx':<8} {'window':>12} {'geom_ok':>8} {'caps_ok':>8} {'force_inf':>10} {'γ_gate':>8} "
                     f"{'gamma0':>12} {'dM':>12} {'Dgain':>14} {'raw_score':>12} {'σE_atom':>12} "
                     f"{'σF_max':>12} {'σF_mean':>12} {'Eabs_exp':>12} {'Fabs_mean':>12} "
                     f"{'cal_ok':>8} {'ood':>6} {'Fmax':>12} {'selected':>10} {'shortlist':>11}\n")
            
            shortlist_set = set(self.final_pool_indices)
            for pidx in sorted(self.all_frame_records.keys()):
                R = self.all_frame_records[pidx]
                fh.write(f"{pidx:<8d} {R['window']:>12} {int(R['rdf_ok']):>8d} {int(R['pass_caps']):>8d} "
                         f"{int(R['force_inf']):>10d} {int(R['gamma_gate']):>8d} {R['gamma0']:>12.6f} {R['dM']:>12.6f} "
                         f"{R['dgain_train']:>14.6f} {R['raw_score_window']:>12.6f} {R['sigma_E_atom']:>12.6f} "
                         f"{R['sigma_F_max']:>12.6f} {R['sigma_F_mean']:>12.6f} {R['exp_abs_E_atom']:>12.6f} "
                         f"{R['exp_abs_F_mean']:>12.6f} {int(R['cal_support']):>8d} {int(R['ood_risk']):>6d} "
                         f"{R['Fmax']:>12.6f} "
                         f"{int(R['selected']):>10d} {int(pidx in shortlist_set):>11d}\n")

            # --- 3. Write Train Data Block ---
            fh.write("\n# TRAIN DATASET UNCERTAINTIES\n")
            fh.write("# idx  sigma_E_atom  sigma_F_max  sigma_F_mean  Fmax\n")
            for t_idx in range(len(self.forces_train_frames)):
                fh.write(f"{t_idx:<6d} {self.sigma_E_atom_train[t_idx]:.6f} {self.sigma_F_train_max[t_idx]:.6f} "
                         f"{self.sigma_F_train_mean[t_idx]:.6f} {self.frame_max_force_train[t_idx]:.6f}\n")

    def _print_summary(self):
        n_total = len(self.rdf_ok_mask)
        n_ok = self.rdf_ok_mask.sum()
        frac_geom = n_ok / max(1, n_total)
        
        print(f"\n[AL] Summary:")
        print(f"    Geom OK fraction: {frac_geom:.3f} ({n_ok}/{n_total} frames)")
        print(f"    Hard triggers passed: {self.n_uncertain_total}")
        print(f"    Calibration OOD triggers: {self.n_ood_risk_total}")
        print(f"    Calibration in-support fraction: {np.mean(self.calibration_in_support):.3f}")
        print(f"    Shortlisted frames: {len(self.final_pool_indices)}")

        # --- NEW: ACTIVE LEARNING STRATEGY WARNING ---
        print("\n" + "="*60)
        print("AL STRATEGY ADVISORY:")
        if frac_geom < 0.40:
            print(f"WARNING: Only {frac_geom*100:.1f}% of your trajectory is physically stable.")
            print("To maximize Active Learning value, it is highly recommended to:")
            print(f" 1. Identify the frame 'n' where the simulation first fractures.")
            print(f" 2. Re-run a shorter MD trajectory up to the time of frame 'n'.")
            print(f" 3. Increase the sampling frequency to still collect 5000 frames.")
            print("This creates a denser grid of 'on-the-edge' frames for DFT labeling.")
        else:
            print(f"Trajectory Stability: {frac_geom*100:.1f}%. The pool is healthy for selection.")
        print("="*60 + "\n")

# --- Wrapper to maintain evaluate.py compatibility ---
def adaptive_learning_mig_pool_windowed(*args, **kwargs):
    learner = _PoolActiveLearner(**dict(zip([
        "pool_frames", "F_pool", "F_train", "alpha_sq", "L", "forces_train", "sigma_energy", 
        "sigma_force", "mu_E_frame_train", "mu_E_pool", "sigma_E_pool", "mu_F_pool", 
        "sigma_F_pool", "rdf_thresholds"], args)), **kwargs)
    return learner.run()


def write_pool_al_diagnostics_csv(runs, path="al_pool_diagnostics.csv"):
    """Write stacked per-frame pool AL diagnostics for one or more electronic states."""
    fieldnames = [
        "state", "idx", "pool_row", "window", "n_atoms",
        "geom_ok", "caps_ok", "force_inf", "gamma_gate",
        "gamma0", "dM", "Dgain", "raw_score",
        "E_pred", "E_pred_atom", "sigma_E", "sigma_E_atom",
        "sigma_F_max", "sigma_F_mean", "Eabs_exp", "Fabs_mean", "Fabs_max",
        "cal_ok", "ood", "Fmax", "Fmean", "selected", "shortlist",
    ]
    states = [str(run.get("state", "unknown")) for run in runs]
    rows = []
    for run in runs:
        rows.extend(run.get("rows", []))

    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write("# Pool Active Learning Diagnostics\n")
        fh.write("# format = al_diagnostics_v2\n")
        fh.write("# generated_by = Orchestr.AI\n")
        fh.write(f"# states = {','.join(states)}\n")
        if rows:
            fh.write(f"# n_rows = {len(rows)}\n")
            fh.write(f"# n_pool_frames = {len(set(int(r['idx']) for r in rows))}\n")
        for run in runs:
            state = str(run.get("state", "unknown"))
            for key, value in sorted(run.get("thresholds", {}).items()):
                try:
                    value = float(value)
                    fh.write(f"# threshold[state={state}].{key} = {value:.10g}\n")
                except (TypeError, ValueError):
                    fh.write(f"# threshold[state={state}].{key} = {value}\n")
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[AL] Wrote stacked diagnostics CSV to '{path}' ({len(rows)} rows).")

class UQCalibrator:
    """Handles Isotonic Regression mapping for Bias and Uncertainty Calibration."""
    def __init__(self):
        self.iso_unc = IsotonicRegression(y_min=0.0, out_of_bounds='clip')
        self.iso_bias = IsotonicRegression(y_min=None, y_max=None, out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, mu_E_train, sigma_E_train, delta_E_train):
        print("\n[UQCalibrator] Fitting BIAS and UNCERTAINTY calibrators...")
        self.iso_bias.fit(mu_E_train, delta_E_train)
        self.iso_unc.fit(sigma_E_train, np.abs(delta_E_train))
        self.is_fitted = True
        print("[UQCalibrator] Fitting complete.")

    def calibrate(self, mu_E_raw, sigma_E_raw):
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before calling calibrate().")
        bias_correction = self.iso_bias.predict(mu_E_raw)
        sigma_calibrated = self.iso_unc.predict(sigma_E_raw)
        mu_E_calibrated = mu_E_raw - bias_correction
        return mu_E_calibrated, sigma_calibrated, bias_correction

    def plot_diagnostics(self, mu_E_train, sigma_E_train, delta_E_train, out_dir="uq_plots"):
        if not self.is_fitted: return
        os.makedirs(out_dir, exist_ok=True)
        delta_train_abs = np.abs(delta_E_train)
        
        # Uncertainty Scatter
        plt.figure(figsize=(5,4))
        plt.scatter(sigma_E_train, delta_train_abs, s=8, alpha=0.6, label="train")
        s_sorted = np.sort(sigma_E_train)
        plt.plot(s_sorted, self.iso_unc.predict(s_sorted), color="C1", lw=2, label="isotonic f(σ)")
        plt.plot([s_sorted.min(), s_sorted.max()], [s_sorted.min(), s_sorted.max()], 'k--', lw=1, label="identity")
        plt.xlabel("ensemble σ (training)")
        plt.ylabel("|ΔE| (training)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/calibration_scatter.png", dpi=200)
        plt.close()

        # Bias Scatter
        plt.figure(figsize=(5,4))
        plt.scatter(mu_E_train, delta_E_train, s=8, alpha=0.6, label="train (signed error)")
        e_sorted = np.sort(mu_E_train)
        plt.plot(e_sorted, self.iso_bias.predict(e_sorted), color="C1", lw=2, label="isotonic bias f(E)")
        plt.plot([e_sorted.min(), e_sorted.max()], [0, 0], 'k--', lw=1, label="zero bias")
        plt.xlabel("Predicted Energy μE (training)")
        plt.ylabel("Signed Error ΔE (training)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/bias_calibration_scatter.png", dpi=200)
        plt.close()
        print(f"[UQCalibrator] Diagnostic plots saved to {out_dir}/")


# =============================================================================
# 4. UQ CALIBRATION FOR POOL SELECTION & REFERENCE SLOPE VALIDATION
# =============================================================================

def apply_sigma_comp_calibration(
    sigma_comp: np.ndarray,
    calibrators: Optional[Dict[str, Any]],
    mode: str = "var",
) -> np.ndarray:
    """Apply train-fitted force calibrators to flat component uncertainties."""
    if calibrators is None or sigma_comp is None:
        return sigma_comp
    sigma = np.asarray(sigma_comp, dtype=float)
    cal_var = calibrators.get("cal_var_F")
    if cal_var is None:
        return sigma
    out = cal_var.transform(sigma)
    if str(mode).lower() == "iso":
        cal_iso = calibrators.get("cal_iso_F")
        if cal_iso is not None:
            out = cal_iso.transform(out)
    return out


def apply_sigma_energy_calibration(
    sigma_energy: np.ndarray,
    calibrators: Optional[Dict[str, Any]],
    mode: str = "var",
) -> np.ndarray:
    """Apply train-fitted energy calibrators to per-frame ensemble σ(E)."""
    if calibrators is None or sigma_energy is None:
        return sigma_energy
    sigma = np.asarray(sigma_energy, dtype=float)
    cal_var = calibrators.get("cal_var_E")
    if cal_var is None:
        return sigma
    out = cal_var.transform(sigma)
    if str(mode).lower() == "iso":
        cal_iso = calibrators.get("cal_iso_E")
        if cal_iso is not None:
            out = cal_iso.transform(out)
    return out


def calibrate_sigma_force_frames(
    sigma_force_frames: List[np.ndarray],
    calibrators: Optional[Dict[str, Any]],
    mode: str = "var",
) -> List[np.ndarray]:
    """Calibrate per-frame force σ arrays (list of (n_atoms, 3))."""
    if calibrators is None or not sigma_force_frames:
        return sigma_force_frames
    flat_parts = [np.asarray(f, dtype=float).reshape(-1) for f in sigma_force_frames]
    flat = np.concatenate(flat_parts)
    flat_cal = apply_sigma_comp_calibration(flat, calibrators, mode)
    splits = np.cumsum([p.size for p in flat_parts])[:-1]
    chunks = np.split(flat_cal, splits)
    return [
        c.reshape(np.asarray(f).shape)
        for c, f in zip(chunks, sigma_force_frames)
    ]


def scale_pool_force_summaries(
    sigma_F_mean: np.ndarray,
    sigma_F_max: np.ndarray,
    calibrators: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale pool light-mode σ summaries by train variance-scaling factor."""
    if calibrators is None:
        return sigma_F_mean, sigma_F_max
    cal_var = calibrators.get("cal_var_F")
    if cal_var is None:
        return sigma_F_mean, sigma_F_max
    s = float(getattr(cal_var, "s", 1.0))
    return np.asarray(sigma_F_mean, float) * s, np.asarray(sigma_F_max, float) * s


def calibration_safe_for_selection(
    metrics_result: Optional[Dict[str, Any]],
    min_spearman: float = 0.4,
    max_ence_raw: float = 0.20,
) -> bool:
    """Return True if train UQ metrics support using calibrated σ in pool AL."""
    if not metrics_result:
        return False
    m = metrics_result.get("metrics", {})
    for key in ("Spearman_raw", "Spearman_calVAR", "Spearman_calISO"):
        sp = m.get(key)
        if sp is not None and np.isfinite(sp) and sp >= min_spearman:
            return True
    ence = m.get("ENCE_raw")
    if ence is not None and np.isfinite(ence) and ence <= max_ence_raw:
        return True
    return False


def validate_consecutive_reference_deltas(
    frames,
    predicted_energies: np.ndarray,
    true_energies: np.ndarray,
    predicted_forces: List[np.ndarray],
    true_forces: List[np.ndarray],
    system_tag: str = "cluster",
):
    """
    Validate consecutive-frame intensive energy slopes and force transitions.

    Frames must be time-ordered within a single trajectory; shuffled reference
    XYZ files produce meaningless consecutive differences.
    """
    n_frames = len(frames)
    if n_frames < 2:
        print(f"[{system_tag.upper()}] Insufficient frames for consecutive delta validation.")
        return

    atom_counts = np.array([len(f) for f in frames], dtype=float)
    avg_atoms = np.mean(atom_counts)

    e_pred_atom = np.asarray(predicted_energies, dtype=float) / atom_counts
    e_true_atom = np.asarray(true_energies, dtype=float) / atom_counts

    delta_e_true_step = np.diff(e_true_atom)
    delta_e_pred_step = np.diff(e_pred_atom)

    slope_errors = np.abs(delta_e_true_step - delta_e_pred_step) * 1000.0
    mae_slope = float(np.mean(slope_errors))
    max_slope = float(np.max(slope_errors))

    force_mae_list = []
    for i in range(n_frames - 1):
        f_true_diff = np.asarray(true_forces[i + 1], dtype=float) - np.asarray(true_forces[i], dtype=float)
        f_pred_diff = np.asarray(predicted_forces[i + 1], dtype=float) - np.asarray(predicted_forces[i], dtype=float)
        force_mae_list.append(float(np.mean(np.abs(f_true_diff - f_pred_diff))))
    mean_force_delta_error = float(np.mean(force_mae_list)) if force_mae_list else float("nan")

    print("\n" + "=" * 70)
    print(f"REFERENCE SLOPE METRICS VALIDATION: {system_tag.upper()}")
    print(f"    Total Frames Analyzed : {n_frames} | Average Cluster Size: {avg_atoms:.1f} atoms")
    print(f"    Consecutive ΔE Slope MAE: {mae_slope:.4f} meV/atom")
    print(f"    Consecutive ΔE Slope MAX: {max_slope:.4f} meV/atom")
    print(f"    Force Transition MAE    : {mean_force_delta_error:.4f} eV/Å")
    print("    STATUS ASSESSMENT       : ", end="")
    if mae_slope <= 3.0 and mean_force_delta_error <= 0.05:
        print(
            "EXCELLENT. Gradient slopes are size-consistent. "
            "Total energy offsets are benign rigid shifts."
        )
    elif mae_slope <= 8.0:
        print(
            "ACCEPTABLE. Suitable for structural dynamics, "
            "but watch out for localized surface drift."
        )
    else:
        print(
            "POOR GENERALIZATION. The model is misinterpreting structural transitions "
            "at this scale. SWA or dataset balancing required."
        )
    print("=" * 70 + "\n")

    out_path = f"reference_slope_validation_{system_tag}.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"# System: {system_tag}\n")
        fh.write("atoms_mean,mae_slope_meVA,max_slope_meVA,mae_force_eVAng\n")
        fh.write(f"{avg_atoms},{mae_slope},{max_slope},{mean_force_delta_error}\n")
    print(f"[{system_tag.upper()}] Wrote slope summary to {out_path}")
