"""
active_learning.py

Refactored in 2025: All legacy functions, PyTorch/HDBSCAN dependencies, 
and unused RDF calculations have been purged.

This module implements:
  1. Influence-based AL for the Validation Set.
  2. A highly modular Class-based Pool Active Learner for OOD sampling.
"""

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
from typing import Tuple, List, Optional
from mlff_qd.postprocessing.rdf import compute_rdf_thresholds_from_reference, fast_filter_by_rdf_kdtree

# =============================================================================
# 1. MATH & LATENT SPACE UTILITIES
# =============================================================================

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
    
    # 1. RDF Filter
    if reference_frames:
        rdf_thresholds = compute_rdf_thresholds_from_reference(reference_frames)
        eval_frames_list = [all_frames[i] for i in eval_idx]
        realistic_mask = fast_filter_by_rdf_kdtree(eval_frames_list, rdf_thresholds)
    else:
        realistic_mask = np.ones(len(eval_idx), dtype=bool)

    # 2. Latent space calculations
    alpha_sq, lam_opt, terms_lat, G_all, L_E = calibrate_alpha_reg_gcv(mean_l_al, delta_E_frame)
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
    delta_E_eval = np.abs(delta_E_frame[eval_idx])

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

    def run(self):
        print(f"\n[AL] --- PoolActiveLearner Orchestrator ---")
        self._setup_latent_space()
        self._setup_thresholds()

        # Convergence Check
        if self.n_hi_total < 10:
            print("[AL] Convergence heuristic: fewer than 10 frames exceed any lower bound.")
            print("[AL] Nothing significant left to label.")
        else:
            self._evaluate_windows()
            self._finalize_selection()

        self._write_diagnostics()
        self._print_summary()
        return self.sel_frames, self.final_pool_indices

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
        self.n_atoms_train = self.forces_train.shape[1]
        self.n_atoms_pool = self.pool_frames[0].get_positions().shape[0]

        # Energies
        self.mu_E_atom_train = self.mu_E_frame_train / self.n_atoms_train
        self.sigma_E_atom_train = self.sigma_energy / self.n_atoms_train
        self.mu_E_atom_pool = self.mu_E_pool / self.n_atoms_pool
        self.sigma_E_atom_pool = self.sigma_E_pool / self.n_atoms_pool

        self.thr_E_hi_atom = self.mu_E_atom_train.max() + 0.5
        self.E_lo_pool_atom = (self.mu_E_pool - 3.0 * self.sigma_E_pool) / self.n_atoms_pool
        self.E_hi_pool_atom = (self.mu_E_pool + 3.0 * self.sigma_E_pool) / self.n_atoms_pool

        # Forces
        sigma_F_train = self.sigma_force.reshape(self.forces_train.shape[0], self.n_atoms_train, 3)
        self.sigma_F_train_max = sigma_F_train.max(axis=(1, 2))
        self.sigma_F_train_mean = np.linalg.norm(sigma_F_train, axis=2).mean(axis=1)
        self.frame_max_force_train = np.linalg.norm(self.forces_train, axis=2).max(axis=1)

        sigma_F_pool_3d = self.sigma_F_pool.reshape(len(self.pool_frames), self.n_atoms_pool, 3)
        self.sigma_F_pool_max = sigma_F_pool_3d.max(axis=(1, 2))
        self.sigma_F_pool_mean = np.linalg.norm(sigma_F_pool_3d, axis=2).mean(axis=1)
        self.frame_max_force_pool = np.linalg.norm(self.mu_F_pool.reshape(len(self.pool_frames), -1, 3), axis=2).max(axis=1)

        # Baseline Thresholds
        self.thr_sigma_E_low = np.percentile(self.sigma_E_atom_train, self.percentile_F_low)
        self.thr_sigma_F = np.percentile(self.sigma_F_train_max, self.percentile_F_low)
        self.thr_sigma_Fmean = np.percentile(self.sigma_F_train_mean, self.percentile_F_low)
        self.thr_Fmag = np.percentile(self.frame_max_force_train, self.percentile_F_low)
        self.train_Fmax_hard_cap = float(self.frame_max_force_train.max()) * float(self.hard_Fmax_train_mult)

        # Apply RDF Filter
        self.rdf_ok_mask = fast_filter_by_rdf_kdtree(self.pool_frames, self.rdf_thresholds)
        ok_idx = np.where(self.rdf_ok_mask)[0]

        # Calculate Adaptive Upper Caps from OK frames
        if len(ok_idx) > 0:
            self.thr_sigma_E_hi_eff = max(self.thr_sigma_E_low, np.percentile(self.sigma_E_atom_pool[ok_idx], self.percentile_F_hi))
            self.thr_sigma_F_hi_eff = max(self.thr_sigma_F, np.percentile(self.sigma_F_pool_max[ok_idx], self.percentile_F_hi))
            self.thr_sigma_Fmean_hi_eff = max(self.thr_sigma_Fmean, np.percentile(self.sigma_F_pool_mean[ok_idx], self.percentile_F_hi))
            self.thr_Fmag_hi_eff = max(self.thr_Fmag, np.percentile(self.frame_max_force_pool[ok_idx], self.percentile_F_hi))
            self.allowed_offset_eff = max(0.0, np.percentile(self.mu_E_atom_pool[ok_idx], 5) - np.percentile(self.mu_E_atom_train, 95)) + 0.05
        else:
            self.thr_sigma_E_hi_eff = max(self.thr_sigma_E_low, 0.01)
            self.thr_sigma_F_hi_eff = max(self.thr_sigma_F, 2.0 * self.sigma_F_train_max.max())
            self.thr_sigma_Fmean_hi_eff = max(self.thr_sigma_Fmean, 2.0 * self.sigma_F_train_mean.max())
            self.thr_Fmag_hi_eff = max(self.thr_Fmag, 2.0 * self.frame_max_force_train.max())
            self.allowed_offset_eff = 2.0 / self.n_atoms_train

        # Count coverage
        n_hi_E = (self.sigma_E_atom_pool > self.thr_sigma_E_low).sum()
        n_hi_Fmax = (self.sigma_F_pool_max > self.thr_sigma_F).sum()
        n_hi_Fmean = (self.sigma_F_pool_mean > self.thr_sigma_Fmean).sum()
        n_hi_Fmag = (self.frame_max_force_pool > self.thr_Fmag).sum()
        self.n_hi_total = n_hi_E + n_hi_Fmax + n_hi_Fmean + n_hi_Fmag

    def _evaluate_windows(self):
        n_pool = len(self.pool_frames)
        for w0 in range(0, n_pool, self.window_size):
            win = list(range(w0, min(w0 + self.window_size, n_pool)))
            
            # Apply Filters
            win_good = [i for i in win if self.rdf_ok_mask[i]]
            win_E = [i for i in win_good if self.mu_E_atom_pool[i] < self.thr_E_hi_atom]
            win_CI = [i for i in win_E if (self.E_hi_pool_atom[i] >= (self.mu_E_atom_train.min() - self.allowed_offset_eff)) and 
                                          (self.E_lo_pool_atom[i] <= (self.mu_E_atom_train.max() + self.allowed_offset_eff))]
            win_phys = [i for i in win_CI if self.sigma_E_atom_pool[i] < self.thr_sigma_E_hi_eff and 
                                             self.sigma_F_pool_max[i] < self.thr_sigma_F_hi_eff and 
                                             self.sigma_F_pool_mean[i] < self.thr_sigma_Fmean_hi_eff and 
                                             self.frame_max_force_pool[i] < self.thr_Fmag_hi_eff]
            
            # Hard Triggers
            high = [i for i in win_phys if self.frame_max_force_pool[i] <= self.train_Fmax_hard_cap and 
                                          (self.sigma_E_atom_pool[i] >= self.hard_sigma_E_atom_min or 
                                           self.sigma_F_pool_mean[i] >= self.hard_sigma_F_mean_min or 
                                           self.sigma_F_pool_max[i] >= self.hard_sigma_F_max_min)]
            
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
            self.n_gamma_gate_total += cand_mask.sum()

            cand_idx_local = np.where(cand_mask)[0]
            selected_local = []
            
            # D-Optimal Selection
            if cand_idx_local.size > 0:
                X_cand = sub_G_all[cand_idx_local]
                order, gains, _ = d_optimal_full_order(X_cand, self.G_train)
                
                # Intra-batch diversity
                dist_mat = np.linalg.norm(X_cand[:, None, :] - X_cand[None, :, :], axis=2)
                remaining = list(range(X_cand.shape[0]))
                
                while len(selected_local) < self.min_k and remaining:
                    best_score, best_r = -np.inf, None
                    for r in remaining:
                        score = gains[r] * (1.0 if not selected_local else float(np.min(dist_mat[r, selected_local])))
                        if score > best_score:
                            best_score, best_r = score, r
                    selected_local.append(best_r)
                    remaining.remove(best_r)
                    
            picks_abs = win_all[cand_idx_local[selected_local]].tolist() if selected_local else []

            # Save Records
            for j_local, pidx in enumerate(win_all):
                self.all_frame_records[pidx] = {
                    "pool_idx": pidx, "window": f"{w0}-{win[-1]+1}",
                    "rdf_ok": pidx in win_good, "pass_caps": pidx in win_phys,
                    "force_inf": pidx in high, "gamma_gate": keep_gamma_mask[j_local],
                    "gamma0": gamma_all[j_local], "dM": dM_all[j_local],
                    "dgain_train": np.log1p(quad_all[j_local]),
                    "raw_score_window": gamma_all[j_local] * np.log1p(quad_all[j_local]),
                    "sigma_E_atom": self.sigma_E_atom_pool[pidx],
                    "sigma_F_max": self.sigma_F_pool_max[pidx],
                    "sigma_F_mean": self.sigma_F_pool_mean[pidx],
                    "Fmax": self.frame_max_force_pool[pidx],
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
            fh.write(f"{'idx':<8} {'window':>12} {'rdf_ok':>8} {'caps_ok':>8} {'force_inf':>10} {'γ_gate':>8} "
                     f"{'gamma0':>12} {'dM':>12} {'Dgain':>14} {'raw_score':>12} {'σE_atom':>12} "
                     f"{'σF_max':>12} {'σF_mean':>12} {'Fmax':>12} {'selected':>10} {'shortlist':>11}\n")
            
            shortlist_set = set(self.final_pool_indices)
            for pidx in sorted(self.all_frame_records.keys()):
                R = self.all_frame_records[pidx]
                fh.write(f"{pidx:<8d} {R['window']:>12} {int(R['rdf_ok']):>8d} {int(R['pass_caps']):>8d} "
                         f"{int(R['force_inf']):>10d} {int(R['gamma_gate']):>8d} {R['gamma0']:>12.6f} {R['dM']:>12.6f} "
                         f"{R['dgain_train']:>14.6f} {R['raw_score_window']:>12.6f} {R['sigma_E_atom']:>12.6f} "
                         f"{R['sigma_F_max']:>12.6f} {R['sigma_F_mean']:>12.6f} {R['Fmax']:>12.6f} "
                         f"{int(R['selected']):>10d} {int(pidx in shortlist_set):>11d}\n")

    def _print_summary(self):
        frac_geom = self.rdf_ok_mask.sum() / max(1, len(self.rdf_ok_mask))
        print(f"\n[AL] Summary:\n    Geom OK fraction: {frac_geom:.3f}\n    Hard triggers passed: {self.n_uncertain_total}\n"
              f"    Shortlisted frames: {len(self.final_pool_indices)}\n")

# --- Wrapper to maintain evaluate.py compatibility ---
def adaptive_learning_mig_pool_windowed(*args, **kwargs):
    learner = _PoolActiveLearner(**dict(zip([
        "pool_frames", "F_pool", "F_train", "alpha_sq", "L", "forces_train", "sigma_energy", 
        "sigma_force", "mu_E_frame_train", "mu_E_pool", "sigma_E_pool", "mu_F_pool", 
        "sigma_F_pool", "rdf_thresholds"], args)), **kwargs)
    return learner.run()

