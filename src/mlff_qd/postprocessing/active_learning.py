"""
active_learning.py

This module implements two active learning strategies for ML force-fields:
  1. An influence-based strategy (adaptive_learning_influence_calibrated)
  2. A traditional strategy using HDBSCAN clustering (adaptive_learning)

It also includes helper functions to normalize data, compute similarity matrices,
perform greedy selection balancing score and diversity, and compute average pairwise
similarity.
"""
import numpy as np
import os
import shutil
import heapq
import traceback
import matplotlib.pyplot as plt
import scipy.optimize
import torch
from typing import List, Optional, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from scipy.special import erfinv
from scipy.stats import spearmanr
from scipy.spatial.distance import cdist
from ase.data import chemical_symbols

import scipy.linalg
import scipy.linalg, scipy.spatial.distance
from mlff_qd.postprocessing.plotting import plot_swapped_final_tight
from typing import Tuple, List, Optional 
from scipy.ndimage import gaussian_filter1d
from kneed import KneeLocator

# Check for hdbscan availability
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not found. Traditional active_learning function will not work.")

# Check for PyTorch availability
try:
    import torch  # Needed for pairwise distance calculations
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Traditional active_learning's distance feature calculation will fail.")

# =============================================================================
# MIG-Based Active Learning Strategy
# =============================================================================
# -----------------------------------------------------------------------------
# 1. Hyper‑parameter calibration (reg, α²) via GCV + moment‑matching
# -----------------------------------------------------------------------------

import numpy as np
import scipy.optimize
import scipy.linalg
from typing import Tuple

def calibrate_alpha_reg_gcv(
    F_eval: np.ndarray,
    y: np.ndarray,
    lambda_bounds: Tuple[float, float] = (1e-6, 1e4)
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibrate a ridge (GP) model via GCV and compute predictive variances.
    Adds adaptive jitter to ensure the covariance matrix is PD for Cholesky.

    Args:
      F_eval       (n_eval x d): features for validation frames.
      y            (n_eval,)    : target residuals for validation.
      lambda_bounds: (lam_min, lam_max) for GCV search.

    Returns:
      alpha_sq     : estimated noise scale (α²).
      lam_opt      : chosen ridge parameter.
      terms_lat    : predictive variance terms per evaluation row.
      G_eval       : whitened latent features for evaluation (n_eval x d).
      L            : Cholesky factor (d x d) of (F^T F + lam_opt I + jitter).
    """
    # Dimensions
    n, d = F_eval.shape

    # SVD for GCV
    U, s, _ = np.linalg.svd(F_eval, full_matrices=False)
    UTy = U.T @ y
    s2 = s**2

    # GCV objective: log((||y - y_hat||^2)/(n - df)^2)
    def gcv_obj(log_lam):
        lam = np.exp(log_lam)
        a = s2 / (s2 + lam)
        df = a.sum()
        y_hat = (a * UTy) @ U.T
        resid = y - y_hat
        return np.log((resid @ resid) / (n - df)**2)

    # Find optimal lambda
    res = scipy.optimize.minimize_scalar(
        gcv_obj,
        bounds=np.log(lambda_bounds),
        method='bounded'
    )
    lam_opt = np.exp(res.x)

    # Build covariance matrix A = F^T F + lam_opt * I
    A = F_eval.T @ F_eval + lam_opt * np.eye(d)

    # Adaptive jitter for Cholesky stability
    base_jitter = 1e-8 * np.trace(A) / d
    jitter = 0.0
    for i in range(6):  # jitter from base*10^0 up to base*10^5
        try:
            L = np.linalg.cholesky(A + jitter * np.eye(d))
            if jitter > 0:
                print(f"[GCV] added jitter={jitter:.1e} to make A PD")
            break
        except np.linalg.LinAlgError:
            jitter = base_jitter * (10 ** i)
    else:
        raise np.linalg.LinAlgError(
            f"A not PD even after jitter up to {jitter:.1e}" )

    # Whiten evaluation features: G_eval = (L^{-1} F^T)^T
    G_eval = scipy.linalg.solve_triangular(L, F_eval.T, lower=True).T

    # Predictive variance terms per evaluation row
    terms_lat = np.sum(G_eval**2, axis=1)

    # Compute residual mean square from SVD-based fit
    a = s2 / (s2 + lam_opt)
    y_hat = (a * UTy) @ U.T
    resid_mean = np.mean((y - y_hat)**2)

    # Noise scale alpha^2
    alpha_sq = resid_mean / np.mean(terms_lat)

    print(f"[GCV] λ = {lam_opt:.3e}, α² = {alpha_sq:.3e}")
    return alpha_sq, lam_opt, terms_lat, G_eval, L

# -----------------------------------------------------------------------------
# 2. FPS‑W sampler (uncertainty‑biased farthest‑point sampling)
# -----------------------------------------------------------------------------

def fps_uncertainty(
    G_eval: np.ndarray,
    G_ref: np.ndarray,
    sigma_eval: np.ndarray,
    *,
    beta: float = 0.5,
    drop_init: float = 0.5,
    min_k: int = 5,
    score_floor: Optional[float] = None,
    verbose: bool = True
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Farthest-point sampling with uncertainty weighting.

    Stopping criteria (OR logic):
      1) absolute floor: best_now < score_floor (after min_k)
      2) relative drop: best_now < best_initial * (1 - drop_init) (after min_k)
      3) exhaustion: no candidates left

    Returns:
        selected        : list of indices into eval pool
        sel_score       : scores at the moments of selection
        sel_dist        : distances at the moments of selection
        score_all_start : initial FPS-W score for all eval frames
        dist_all_start  : initial distance for all eval frames
    """
    # Prepare tensors
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GE = torch.as_tensor(G_eval, device=dev)
    GR = torch.as_tensor(G_ref, device=dev)
    sig = torch.as_tensor(sigma_eval, device=dev)

    # Initial distances and scores
    d = torch.cdist(GE, GR).min(dim=1).values
    s = d ** (1 - beta) * sig ** beta

    # Save full arrays
    score_all_start = s.cpu().numpy().copy()
    dist_all_start = d.cpu().numpy().copy()

    selected, sel_score, sel_dist = [], [], []
    best_initial = s.max().item()
    floor = score_floor if score_floor is not None else -np.inf

    if verbose:
        print(f"[FPS] pool={len(s)}, min_k={min_k}, drop_init={drop_init:.2f}, score_floor={floor:.5f}")

    # Main loop
    while True:
        # Remaining candidate indices
        avail = torch.nonzero(~torch.isinf(s), as_tuple=False).view(-1)
        if avail.numel() == 0:
            if verbose:
                print("[FPS] stop: no candidates left")
            break

        # Choose best-scoring available
        idx_local = torch.argmax(s[avail]).item()
        idx = avail[idx_local].item()
        best_now = float(s[idx].item())
        n_sel = len(selected) + 1

        # 1) Absolute floor
        if n_sel >= min_k and best_now < floor:
            if verbose:
                print(f"[FPS] stop: best_now {best_now:.5f} < floor {floor:.5f}")
            break

        # 2) Relative drop
#        if n_sel >= min_k and best_now < best_initial * (1 - drop_init):
#            if verbose:
#                print(f"[FPS] stop: best_now {best_now:.3e} < {best_initial * (1 - drop_init):.3e} (drop_init)")
#            break

        # Record selection
        selected.append(idx)
        sel_score.append(best_now)
        sel_dist.append(float(d[idx].item()))

        # Mark as used
        s[idx] = -np.inf

        # Update distances and scores
        new_d = torch.cdist(GE, GE[idx:idx+1]).squeeze()
        d = torch.minimum(d, new_d)
        s = d ** (1 - beta) * sig ** beta

        if verbose and n_sel % 50 == 1:
            print(f"[FPS] step {n_sel:4d}: idx={idx:4d}, "
                  f"score={best_now:.5f}, dist={sel_dist[-1]:.5f}, sigma={sig[idx].item():.5f}")

    return (
        selected,
        np.array(sel_score),
        np.array(sel_dist),
        score_all_start,
        dist_all_start
    )

# -----------------------------------------------------------------------------
#  PURE D‑OPTIMAL (no early stops)
# -----------------------------------------------------------------------------

def d_optimal_full_order(
    X_cand: np.ndarray,
    X_train: np.ndarray,
    *,
    reg: float = 1e-6,
    verbose: bool = False,
):
    """Return *all* candidate indices in greedy D‑optimal order + γ for all.

    No gain‑floor, no min_k/max_k, no rho — it simply orders the pool so that
    the first element gives the largest log‑det jump, the second the next, …
    γ (Mahalanobis distance) is computed **before any removals** and returned
    alongside.
    """
    m, d = X_cand.shape
    I_d  = np.eye(d)
    M_inv = np.linalg.inv(X_train.T @ X_train + reg * I_d)

    # initial Mahalanobis‑squared for every candidate (used later for γ)
    quad0  = np.einsum("id,dk,ik->i", X_cand, M_inv, X_cand)
    gamma0 = np.sqrt(quad0)

    # Working copy for greedy ordering
    quad = quad0.copy()
    order, gains = [], []

    for _ in range(m):
        i_best = int(np.argmax(quad))
        q      = quad[i_best]
        gain   = np.log1p(q)
        order.append(i_best)
        gains.append(gain)
        if verbose:
            print(f"pick {len(order):4d}/{m}: gain={gain:.3e}")

        # rank‑1 update
        x = X_cand[i_best]
        v = M_inv @ x
        denom = 1.0 + x @ v
        M_inv -= np.outer(v, v) / denom

        # fast downdate for remainder
        alpha = X_cand @ v
        quad -= (alpha ** 2) / denom
        quad[i_best] = -np.inf  # lock in picked point

    return np.asarray(order, int), np.asarray(gains, float), gamma0


# -----------------------------------------------------------------------------
# 3. Main adaptive‑learning routine (v4) with RMSE‑based threshold
# -----------------------------------------------------------------------------

def adaptive_learning_mig_calibrated(
        all_frames: List,
        eval_mask: np.ndarray,
        delta_E_frame: np.ndarray,
        mean_l_al: np.ndarray,
        *,
        force_rmse_per_comp: Optional[np.ndarray] = None,
        target_rmse_conv: float,
        beta: float = 0.5,
        drop_init: float = 0.5,
        min_k: int = 5,
        max_k: Optional[int] = None,
        score_floor: Optional[float] = None,
        base: str = "al_mig_v7"
) -> Tuple[List, np.ndarray]:
    """Active-learning: energy uncertainty *or* raw force RMSE trigger.

    Frames enter FPS-W if
        σ_E > σ_E,emp  OR  RMSE_F_raw > 0.03 eV Å⁻¹.

    No force calibration is performed; the raw per-frame RMSE array is used
    directly. FPS-W still works with latent distance and energy uncertainty only.
    """

    # ---------- 1) split ----------------------------------------------------
    train_idx = np.where(~eval_mask)[0]
    eval_idx  = np.where(eval_mask)[0]
    F_train   = mean_l_al[train_idx]
    F_eval    = mean_l_al[eval_idx]
    y_val_E   = delta_E_frame[eval_idx]

    # ---------- 2) energy calibration --------------------------------------
    alpha_sq_E, lam_E, terms_lat_E, G_eval_E, L_E = calibrate_alpha_reg_gcv(F_eval, y_val_E)
    G_train = scipy.linalg.solve_triangular(L_E, F_train.T, lower=True).T
    sigma_eval_E = np.sqrt(alpha_sq_E * terms_lat_E)

    # ---------- 3) latent distances ----------------------------------------
    d_lat_eval = scipy.spatial.distance.cdist(G_eval_E, G_train).min(axis=1)
    score_all_full = (d_lat_eval ** (1 - beta)) * (sigma_eval_E ** beta)
    dist_all_full  = d_lat_eval.copy()

    # ---------- 4) empirical energy floor ----------------------------------
    n_atoms     = len(all_frames[0])
    rho_eV      = 0.002
    sigma_emp_E = rho_eV * np.sqrt(n_atoms)
    q75_d       = float(np.quantile(d_lat_eval, 0.75))
    beta_floor  = beta
    auto_floor  = (q75_d ** (1 - beta_floor)) * (sigma_emp_E ** beta_floor)
    used_floor  = score_floor if score_floor is not None else auto_floor

    # Diagnostics
    print(f"[AL] rho_eV               : {rho_eV:.5f} eV/atom")
    print(f"[AL] sigma_emp_E (total)   : {sigma_emp_E:.5f} eV")
    print(f"[AL] q75_d                 : {q75_d:.5f}")
    print(f"[AL] beta (floor calc)     : {beta_floor:.5f}")
    print(f"[AL] auto score_floor      : (q75_d^(1-beta))*(sigma_emp_E^beta) = {auto_floor:.5f}")
    print(f"[AL] drop_init             : {drop_init:.5f}")
    if score_floor is not None:
        print(f"[AL] user-supplied floor   : {score_floor:.5f} (overriding)")
    print(f"[AL] *** FLOOR IN USE      : {used_floor:.5f}")

    # ---------- 5) raw force RMSE (per‑component → per‑frame) -----------------
    use_forces = force_rmse_per_comp is not None
    print(f"shape force_rmse_per_comp: {force_rmse_per_comp.shape}")
    if use_forces:
        # sanity check on length
        n_total_comp = sum(3 * len(fr) for fr in all_frames)
        if force_rmse_per_comp.shape[0] != n_total_comp:
            raise ValueError("force_rmse_per_comp length mismatch "
                             f"({force_rmse_per_comp.shape[0]} ≠ {n_total_comp})")
    
        # reduce to one scalar per frame: max component‑wise error in that frame
        rmse_F_per_frame = np.empty(len(all_frames))
        flat_ptr = 0
        for i, fr in enumerate(all_frames):
            n_comp = 3 * len(fr)
            rmse_F_per_frame[i] = np.max(force_rmse_per_comp[flat_ptr:flat_ptr + n_comp])
            flat_ptr += n_comp
    
        # keep only the eval split
        rmse_F_eval = rmse_F_per_frame[eval_idx]
    else:
        rmse_F_eval = np.full(eval_idx.shape, np.nan)
    
    rmse_thresh_F = 0.05      # eV / Å
    print(f"[AL] rmse threshold (per‑component→frame max): {rmse_thresh_F:.5f}")
    
    # DEBUG ---------------------------------------------------------------
    print(f"[Debug] σ_E    min / max : {np.nanmin(sigma_eval_E):.4f} / {np.nanmax(sigma_eval_E):.4f}")
    if use_forces:
        print(f"[Debug] RMSE_F per‑frame (max comp) min / max : "
              f"{np.nanmin(rmse_F_eval):.4f} / {np.nanmax(rmse_F_eval):.4f}")
        print(f"[Debug] RMSE_F sample : {rmse_F_eval[:10]} ...")
    
    # ---------- 6) candidate mask ---------------------------------------
    if use_forces:
        trig_E = sigma_eval_E > sigma_emp_E
        trig_F = rmse_F_eval   > rmse_thresh_F
        cand_pos = np.where(trig_E | trig_F)[0]
    
        # extra diagnostics
        print(f"[Debug] candidates from σ_E only   : {np.sum(trig_E & ~trig_F)}")
        print(f"[Debug] candidates from RMSE_F only: {np.sum(trig_F & ~trig_E)}")
        print(f"[Debug] candidates from both       : {np.sum(trig_E &  trig_F)}")
    else:
        cand_pos = np.where(sigma_eval_E > sigma_emp_E)[0]
    
    print(f"[Debug] total candidate positions   : {len(cand_pos)}")

    if len(cand_pos) == 0:
        print("[AL] No candidates selected; returning empty.")
        return [], np.array([], dtype=int)
    cand_idx = eval_idx[cand_pos]

    # ---------- 7) FPS-W sampling -----------------------------------------
    sel_rel_pos, fps_scores_cand, dists_cand, score_all_cand, dist_all_cand = fps_uncertainty(
        G_eval_E[cand_pos], G_train, sigma_eval_E[cand_pos],
        beta=beta, drop_init=drop_init, min_k=min_k, score_floor=used_floor, verbose=True)

    # ---------- 8) compute TI metric for latent --------------------------
    # information-based score, useful for logging
    delta_i_lat = 0.5 * np.log1p(alpha_sq_E * terms_lat_E)  # shape (n_eval,)

    # ---------- 9) assemble logs -------------------------------------------
    fps_full, dist_full = score_all_full.copy(), dist_all_full.copy()
    sel_pos = cand_pos[sel_rel_pos]
    fps_full[sel_pos]  = fps_scores_cand
    dist_full[sel_pos] = dists_cand
    sel_idx  = cand_idx[sel_rel_pos]

    print(f"[Select] picked {len(sel_idx)} / {len(cand_idx)} candidates (eval {len(eval_idx)})")

    # ---------- logging -----------------------------------------------------
    log_name = f"{base}_detailed_log.txt"
    with open(log_name, 'w') as f:
        f.write("Idx	σ_E	FPS_score	Dist	Δ_i_lat	RMSE_F	Selected")
        for i, idx in enumerate(eval_idx):
            f.write(f"{idx}	{sigma_eval_E[i]:.5f}	{fps_full[i]:.5f}	{dist_full[i]:.5f}	"
                    f"{delta_i_lat[i]:.5f}	{rmse_F_eval[i]:.5f}	{idx in sel_idx}")
    print(f"[Log] {log_name}")

    # ---------- plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6,5))
    sel_flag = np.isin(eval_idx, sel_idx)
    ax.scatter(sigma_eval_E[~sel_flag], fps_full[~sel_flag], alpha=0.3, label='not selected')
    ax.scatter(sigma_eval_E[sel_flag], fps_full[sel_flag], marker='x', label='selected')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('σ_E'); ax.set_ylabel('FPS score'); ax.legend()
    plt.tight_layout(); plt.savefig(f"{base}_sigma_vs_score.png", dpi=300); plt.close()

    # ---------- return ------------------------------------------------------
    sel_objs = [all_frames[i - min(eval_idx)] for i in sel_idx]
    return sel_objs, sel_idx

# =============================================================================
#  Ensemble‑based active learning  (n_models ≥ 2)
# =============================================================================

def adaptive_learning_ensemble_calibrated(
        all_frames: List,
        eval_mask: np.ndarray,
        delta_E_frame: np.ndarray,
        mean_l_al: np.ndarray,
        *,
        force_rmse_per_comp: Optional[np.ndarray] = None,
        denom_all: Optional[np.ndarray] = None,
        beta: Optional[float] = 0.5,
        drop_init: float = 1.0,
        min_k: int = 5,
        max_k: Optional[int] = None,
        score_floor: Optional[float] = None,
        base: str = "al_ens_v1",
        **kwargs) -> Tuple[List, np.ndarray]:
    """
    Two-stage AL (signature unchanged):
      1) hard gate on normalized uncertainties (σ_E + max|F|_err)
      2) diversity FPS on raw latent distances (β=0)

    Optional kwargs for tolerances:
      rho_eV     : eV per atom for σ_E tolerance   (default 0.002)
      rmse_tol_F : eV/Å tolerance for max|F|_err   (default 0.05)
    """
    tol = 1e-3 
    eps = 1e-9
    rho_eV     = kwargs.get("rho_eV", 0.002)
    rmse_tol_F = kwargs.get("rmse_tol_F", 0.10)
    n_atoms     = len(all_frames[0])
    delta_tol_E = rho_eV * np.sqrt(n_atoms)

    # 1) split train/eval
    train_idx = np.where(~eval_mask)[0]
    eval_idx  = np.where(eval_mask)[0]
    print(f"[1] split: train={len(train_idx)}, eval={len(eval_idx)}")

    # 2) latent whitening & raw distance
    alpha_sq, lam_opt, terms_lat, G_all, L_E = calibrate_alpha_reg_gcv(mean_l_al, delta_E_frame)

    # distance from eval 
    G_train    = G_all[train_idx]          # latent vectors for current training set
    G_eval_E   = G_all[eval_idx]           # latent vectors for eval/AL pool

    # nearest-neighbour distance distribution **within training**
    D_train = scipy.spatial.distance.cdist(G_train, G_train)
    np.fill_diagonal(D_train, np.inf)
    d_nn_train = D_train.min(axis=1)
    print(f"[2] d_nn train stats: mean={d_nn_train.mean():.3f}, "
          f"std={d_nn_train.std():.3f}, min={d_nn_train.min():.3f}, max={d_nn_train.max():.3f}")

    D_eval = scipy.spatial.distance.cdist(G_eval_E, G_eval_E) 
    np.fill_diagonal(D_eval, np.inf)
    d_nn_eval = D_eval.min(axis=1)
    print(f"[2] d_nn eval stats: mean={d_nn_eval.mean():.3f}, "
          f"std={d_nn_eval.std():.3f}, min={d_nn_eval.min():.3f}, max={d_nn_eval.max():.3f}")

    d_lat_eval = scipy.spatial.distance.cdist(G_eval_E, G_train).min(axis=1)
    print(f"[2] d_lat_eval stats: mean={d_lat_eval.mean():.3f}, "
          f"std={d_lat_eval.std():.3f}, min={d_lat_eval.min():.3f}, max={d_lat_eval.max():.3f}")

    plt.figure()
    plt.hist(d_nn_train, bins=50, alpha=0.6, label='Train–Train NN')
    plt.hist(d_nn_eval,   bins=50, alpha=0.6, label='Eval–Eval NN')
    plt.hist(d_lat_eval,   bins=50, alpha=0.6, label='Eval–Train NN')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distributions in Latent Space')
    plt.legend()
    plt.tight_layout()
    
    outpath = 'distance_distributions.png'
#    plt.savefig(outpath)
    print(f"Saved distance distributions plot to {outpath}")
    
    # 3) per-frame max-RMSE_F
    if force_rmse_per_comp is not None:
        comps_pf = np.array([3*len(fr) for fr in all_frames], int)
        starts   = np.concatenate(([0], np.cumsum(comps_pf[:-1])))
        rmse_F_pf_max = np.maximum.reduceat(force_rmse_per_comp, starts)[:len(all_frames)]
        rmse_F_pf_mean = np.array([force_rmse_per_comp[starts[i]:starts[i] + comps_pf[i]].mean() for i in range(len(all_frames))])
        print(f"[3] computed max RMSE_F and relative error per frame")
    else:
        rmse_F_pf_max = np.zeros(len(all_frames))
        rmse_F_pf_mean = np.zeros(len(all_frames))
        print("[3] no force_rmse_per_comp → using zeros")

    rmse_F_train = rmse_F_pf_max[train_idx]
    rmse_Fmean_train = rmse_F_pf_mean[train_idx]
    delta_E_train = np.abs(delta_E_frame[train_idx])

    rmse_F_eval  = rmse_F_pf_max[eval_idx]
    rmse_Fmean_eval = rmse_F_pf_mean[eval_idx]
    delta_E_eval = np.abs(delta_E_frame[eval_idx])

    print(f"[3] rmse_Fmax_train: mean={rmse_F_train.mean():.3f}, std={rmse_F_train.std():.3f}, min={rmse_F_train.min():.3f}, max={rmse_F_train.max():.3f}")
    print(f"[3] rmse_Fmean_train: mean={rmse_Fmean_train.mean():.3f}, std={rmse_Fmean_train.std():.3f}, min={rmse_Fmean_train.min():.3f}, max={rmse_Fmean_train.max():.3f}")
    print(f"[3] delta_E_train: mean={delta_E_train.mean():.3f}, std={delta_E_train.std():.3f}, min={delta_E_train.min():.3f}, max={delta_E_train.max():.3f}")

    print(f"[3] rmse_F_eval: mean={rmse_F_eval.mean():.3f}, std={rmse_F_eval.std():.3f}, min={rmse_F_eval.min():.3f}, max={rmse_F_eval.max():.3f}")
    print(f"[3] rmse_Fmean_eval: mean={rmse_Fmean_eval.mean():.3f}, std={rmse_Fmean_eval.std():.3f}, min={rmse_Fmean_eval.min():.3f}, max={rmse_Fmean_eval.max():.3f}")
    print(f"[3] delta_E_eval: mean={delta_E_eval.mean():.3f}, std={delta_E_eval.std():.3f}, min={delta_E_eval.min():.3f}, max={delta_E_eval.max():.3f}")

    # Assess tolerance from reference true values
    all_true_F = denom_all 
    train_F       = all_true_F[train_idx]   # same dims
    eval_F        = all_true_F[eval_idx]

    # — 1. Noise floor δ from DFT forces via MAD → σₙ
    noise      = 0.05 # eV/Ang 
    print(f"[3] noise-floor δ (eV/Å):{noise:.3f}")
    
    # — 2. Per-frame RMS of the true forces
    F_mag_train = np.linalg.norm(train_F, axis=-1)      # (n_train_frames, n_atoms)
    F_rms_train = np.sqrt((F_mag_train**2).mean(axis=1))  # RMS per train frame
    mean_F_rms   = F_rms_train.mean()
    print(f"[3] mean true–force RMS per frame (eV/Å): {mean_F_rms:.3f}")

    # — 3. Pick relative‐error budget ε and derive per‐frame RMSE tol
    epsilon          = 0.05   # 5% relative error
    rmse_tol_F_frame = epsilon * mean_F_rms
    print(f"[3] predicted RMSE/frame tolerance (eV/Å): {rmse_tol_F_frame:.3f}")

    # (optional) still print your percentile-based hard caps
    F_mag_train_flat = F_mag_train.ravel()
    F_mag_eval_flat  = np.linalg.norm(eval_F, axis=-1).ravel()
    p99_train        = np.percentile(F_mag_train_flat, 99)
    p99_eval        = np.percentile(F_mag_eval_flat, 99)
    print(f"[3] 99th-pctile train |F|: {p99_train:.3f}")
    print(f"[3] 99th-pctile eval  |F|: {p99_eval:.3f}")

    # hard component cap (absolute)
    rmse_tol_F = 2 * epsilon * p99_train
    print(f"[3] threshold on max force comp: {rmse_tol_F:.3f}")
    print(f"[3] threshold on force per frame: {rmse_tol_F_frame:.3f}")

    norm_E = delta_E_eval / delta_tol_E          # σ_E   in “tolerance units”
    norm_F = rmse_F_eval  / rmse_tol_F           # RMSE_F in “tolerance units”

    std_E = norm_E.std() + eps
    std_F = norm_F.std()  + eps
    w_E = std_E / (std_E + std_F)
    w_F = 1.0  - w_E            # keeps w_E + w_F = 1
    
    print(f"[3] dynamic weights: w_E={w_E:.3f}, w_F={w_F:.3f}")
    
    # 4) normalize uncertainties (z-score)
    z_sigma      = (delta_E_eval - delta_E_eval.mean()) / (delta_E_eval.std() + eps)
    z_rmse       = (rmse_F_eval  - rmse_F_eval.mean())   / (rmse_F_eval.std()   + eps)
    z_rmse_frame = (rmse_Fmean_eval  - rmse_Fmean_eval.mean())   / (rmse_Fmean_eval.std()   + eps) 
    print(f"[4] z_sigma stats: mean={z_sigma.mean():.3f}, std={z_sigma.std():.3f}, "
          f"min={z_sigma.min():.3f}, max={z_sigma.max():.3f}")
    print(f"[4] z_rmse  stats: mean={z_rmse.mean():.3f}, std={z_rmse.std():.3f}, "
          f"min={z_rmse.min():.3f}, max={z_rmse.max():.3f}")
    print(f"[4] z_rmse_frame  stats: mean={z_rmse_frame.mean():.3f}, std={z_rmse_frame.std():.3f}, "
          f"min={z_rmse_frame.min():.3f}, max={z_rmse_frame.max():.3f}")

    # 5) empirical anchor in same z-units
    z_sigma_emp      = (delta_tol_E - delta_E_eval.mean()) / (delta_E_eval.std() + eps)
    z_rmse_emp       = (rmse_tol_F  - rmse_F_eval.mean())   / (rmse_F_eval.std()   + eps)
    z_rmse_emp_frame = (rmse_tol_F_frame - rmse_Fmean_eval.mean())   / (rmse_Fmean_eval.std()   + eps)
    print(f"[5] σ_tol_E={delta_tol_E:.5f} (z={z_sigma_emp:+.3f}), "
          f"F_tol={rmse_tol_F:.5f} (z={z_rmse_emp:+.3f}), "
          f"F_tol_frame={rmse_tol_F_frame:.5f} (z={z_rmse_emp_frame:+.3f})")

    # 6) shift so empirical anchor → 0
    u_frame_z         = w_E*z_sigma + w_F*z_rmse
    u_frame_z_frame   = w_E*z_sigma + w_F*z_rmse_frame
    u_emp             = w_E*z_sigma_emp + w_F*z_rmse_emp
    u_emp_frame       = z_rmse_emp_frame 
    print(f"[6] u_emp anchor = {u_emp:+.3f}")
    print(f"[6] u_frame_z stats: mean={u_frame_z.mean():.3f}, std={u_frame_z.std():.3f}, "
          f"min={u_frame_z.min():.3f}, max={u_frame_z.max():.3f}")
    print(f"[6] u_emp_frame anchor = {u_emp_frame:+.3f}")
    print(f"[6] u_frame_z_frame stats: mean={u_frame_z_frame.mean():.3f}, std={u_frame_z_frame.std():.3f}, "
          f"min={u_frame_z_frame.min():.3f}, max={u_frame_z_frame.max():.3f}")
 
    # normalize uncertainty for hybrid
    U_norm             = (u_frame_z - u_frame_z.mean()) / (u_frame_z.std() + eps)
    U_norm_frame       = (u_frame_z_frame - u_frame_z_frame.mean()) / (u_frame_z_frame.std() + eps)
    U_norm_floor       = (u_emp - u_frame_z.mean()) / (u_frame_z.std() + eps)
    U_norm_floor_frame = (u_emp_frame - u_frame_z.mean()) / (u_frame_z.std() + eps)
    print(f"[7] U_norm_floor = {U_norm_floor:+.3f}")
    print(f"[7] U_norm stats: mean={U_norm.mean():.3f}, std={U_norm.std():.3f}, "
          f"min={U_norm.min():.3f}, max={U_norm.max():.3f}")
    print(f"[7] U_norm_floor_frame = {U_norm_floor_frame:+.3f}")
    print(f"[7] U_norm_frame stats: mean={U_norm_frame.mean():.3f}, std={U_norm_frame.std():.3f}, "
          f"min={U_norm_frame.min():.3f}, max={U_norm_frame.max():.3f}")
    
    # 7) full D-optimal ordering + initial Mahalanobis distance
    order, gains_full, gamma0 = d_optimal_full_order(
        X_cand  = G_eval_E.astype(np.float64),
        X_train = G_train.astype(np.float64),
        reg     = 1e-6)

    # diversity score = per-pick log-det gain
    diversity_score = np.empty_like(gamma0)
    diversity_score[order] = gains_full
    D_norm = (diversity_score - diversity_score.mean()) / (diversity_score.std() + eps)
    print(f"[8] D_norm stats: mean={D_norm.mean():.3f}, std={D_norm.std():.3f}, "
          f"min={D_norm.min():.3f}, max={D_norm.max():.3f}")
    
    # compute D_norm_floor at the Mahalanobis threshold γ=1
    idx_gamma1    = np.argmin(np.abs(gamma0 - 1.0))
    D_norm_floor  = D_norm[idx_gamma1]
    print(f"[8] D_norm_floor = {D_norm_floor:+.3f}")

    conv_U = False
    conv_U_frame = False
    conv_D = False
    
    if U_norm_floor >= U_norm.max() - tol:
        conv_U = True
        print("Error/Uncertainty… with max force comp converged")
    if U_norm_floor_frame >= U_norm_frame.max() - tol:
        conv_U_frame = True
        print("Error/Uncertainty… with force per frame converged")
    if D_norm_floor >= D_norm.max() - tol:
        conv_D = True
        print("Diversity has converged")
    if conv_U and conv_U_frame and conv_D:
        print("Active Learning has fully converged")
        return 
    if conv_U_frame and conv_D:
        print("Active Learning has fully converged on diversity and per frame. Out of distribution AL is suggested")
        return

    # 8) hybrid score & selection
    norm_U = u_frame_z / (u_emp + eps)                     # eps to avoid 0/0
    norm_D = diversity_score / (diversity_score[idx_gamma1] + eps)
    
    std_U = norm_U.std() + eps
    std_D = norm_D.std() + eps
    
    lam_hybrid = std_U / (std_U + std_D)          # dynamic 0–1
    print(f"[8] dynamic λ = {lam_hybrid:.3f}  (std_U={std_U:.3f}, std_D={std_D:.3f})")

    hybrid = lam_hybrid * U_norm + (1.0 - lam_hybrid) * D_norm

    # hybrid_floor mixes the two thresholds
    hybrid_floor = lam_hybrid*U_norm_floor + (1.0-lam_hybrid)*D_norm_floor

    print(f"[9] lam_hybrid = {lam_hybrid:.3f}")
    print(f"[9] hybrid stats: mean={hybrid.mean():.3f}, std={hybrid.std():.3f}, "
          f"min={hybrid.min():.3f}, max={hybrid.max():.3f}")
    print(f"[9] hybrid_floor = {hybrid_floor:+.3f}")
    
    # ---------------- 6) gate then budget ----------------------------------
    keep_mask = hybrid > hybrid_floor
    idx_keep  = np.where(keep_mask)[0]
    n_keep    = idx_keep.size
    k_budget  = 250 if lam_hybrid < 0.5 else 500
    if n_keep > k_budget:
        top_rel = idx_keep[np.argsort(hybrid[idx_keep])[-k_budget:]]
    else:
        top_rel = idx_keep
    sel_idx = eval_idx[top_rel]
    print(f"[8] kept {n_keep} frames after floor, selecting {len(sel_idx)}")

    # ---------------- 7) logging -------------------------------------------
    log_path = f"{base}_log.txt"
    with open(log_path, "w") as fh:
        fh.write("Idx\tU_norm\tD_norm\tlam_hybrid\thybrid\thybrid_floor\tpicked\n")
        for i_pool, idx in enumerate(eval_idx):
            picked = idx in sel_idx
            fh.write(f"{idx}\t{U_norm[i_pool]:.3f}\t{D_norm[i_pool]:.3f}\t"
                     f"{lam_hybrid:.3f}\t{hybrid[i_pool]:.3f}\t{hybrid_floor:.3f}\t{picked}\n")
    print(f"[9] log saved to {log_path}")

    # ---------------- 8) return --------------------------------------------
    sel_objs = [all_frames[i] for i in sel_idx]
    return sel_objs, sel_idx


def adaptive_learning_ensemble_calibrated_old(
        all_frames: List,
        eval_mask: np.ndarray,
        sigma_E_cal: np.ndarray,
        delta_E_frame: np.ndarray,
        mean_l_al: np.ndarray,
        *,
        force_rmse_per_comp: Optional[np.ndarray] = None,
        denom_all: Optional[np.ndarray] = None,
        beta: float = 0.5,
        drop_init: float = 1.0,
        min_k: int = 5,
        max_k: Optional[int] = None,
        score_floor: Optional[float] = None,
        base: str = "al_ens_v1") -> Tuple[List, np.ndarray]:
    """Active learning driven by **calibrated ensemble σ_E** plus latent distance.

    Args
    ----
    sigma_E_cal
        1‑D array of per‑frame calibrated ensemble uncertainties (already scaled).
    mean_l_al
        Ensemble‑averaged latent feature matrix (n_frames × d).
    beta
        Mixing weight: 0 ⇒ pure latent distance, 1 ⇒ pure σ_E.
    """

    # 1) split indices -------------------------------------------------------
    train_idx = np.where(~eval_mask)[0]
    eval_idx  = np.where(eval_mask)[0]
    F_train   = mean_l_al[train_idx]
    F_eval    = mean_l_al[eval_idx]
    y_val_E   = delta_E_frame[eval_idx]

    # 2) latent whitening on *averaged* features ----------------------------
    alpha_sq_E, lam_E, terms_lat_E, G_eval_E, L_E = calibrate_alpha_reg_gcv(F_eval, y_val_E)
    G_train = scipy.linalg.solve_triangular(L_E, F_train.T, lower=True).T
    d_lat_eval = scipy.spatial.distance.cdist(G_eval_E, G_train).min(axis=1)

    # 3) combined score -----------------------------------------------------
    score_all_full = (d_lat_eval ** (1 - beta)) * (sigma_E_cal[eval_idx] ** beta)

    # 4) empirical floor ----------------------------------------------------
    n_atoms     = len(all_frames[0])
    rho_eV      = 0.002
    sigma_emp_E = rho_eV * np.sqrt(n_atoms)
    q75_d       = float(np.quantile(d_lat_eval, 0.75))
    beta_floor  = beta 
    auto_floor  = (q75_d ** (1 - beta)) * (sigma_emp_E ** beta)
    used_floor  = score_floor if score_floor is not None else auto_floor

    print(f"[AL-ENS] rho_eV               : {rho_eV:.5f} eV/atom")
    print(f"[AL-ENS] sigma_emp_E (total)   : {sigma_emp_E:.5f} eV")
    print(f"[AL-ENS] q75_d                 : {q75_d:.5f}")
    print(f"[AL-ENS] beta (floor calc)     : {beta_floor:.5f}")
    print(f"[AL-ENS] auto score_floor      : (q75_d^(1-beta))*(sigma_emp_E^beta) = {auto_floor:.5f}")
    print(f"[AL-ENS] drop_init             : {drop_init:.5f}")
    if score_floor is not None:
        print(f"[AL-ENS] user-supplied floor   : {score_floor:.5f} (overriding)")
    print(f"[AL-ENS] *** FLOOR IN USE      : {used_floor:.5f}")

    # ---------- 5) raw force RMSE (per?~@~Qcomponent ?~F~R per?~@~Qframe) -----------------
    use_forces = force_rmse_per_comp is not None
    print(f"shape force_rmse_per_comp: {force_rmse_per_comp.shape}")
    if use_forces:
        # sanity check on length
        n_total_comp = sum(3 * len(fr) for fr in all_frames)
        if force_rmse_per_comp.shape[0] != n_total_comp:
            raise ValueError("force_rmse_per_comp length mismatch "
                             f"({force_rmse_per_comp.shape[0]} ?~I|  {n_total_comp})")

        # reduce to one scalar per frame: max component?~@~Qwise error in that frame
        rmse_F_per_frame = np.empty(len(all_frames))
        flat_ptr = 0
        for i, fr in enumerate(all_frames):
            n_comp = 3 * len(fr)
            rmse_F_per_frame[i] = np.max(force_rmse_per_comp[flat_ptr:flat_ptr + n_comp])
            flat_ptr += n_comp

        # keep only the eval split
        rmse_F_eval = rmse_F_per_frame[eval_idx]
    else:
        rmse_F_eval = np.full(eval_idx.shape, np.nan)

    rmse_thresh_F = 0.05      # eV / ?~E
    print(f"[AL] rmse threshold (per?~@~Qcomponent?~F~Rframe max): {rmse_thresh_F:.5f}")

    # DEBUG ---------------------------------------------------------------
    print(f"[Debug] ?~C_E    min / max : {np.nanmin(sigma_E_cal):.4f} / {np.nanmax(sigma_E_cal):.4f}")
    if use_forces:
        print(f"[Debug] RMSE_F per?~@~Qframe (max comp) min / max : "
              f"{np.nanmin(rmse_F_eval):.4f} / {np.nanmax(rmse_F_eval):.4f}")
        print(f"[Debug] RMSE_F sample : {rmse_F_eval[:10]} ...")

    # ---------- 6) candidate mask ---------------------------------------
    if use_forces:
        sigma_E_eval = sigma_E_cal[eval_idx]      # shape (n_val,)
        trig_E = sigma_E_eval > sigma_emp_E
        trig_F = rmse_F_eval   > rmse_thresh_F
        cand_pos = np.where(trig_E | trig_F)[0]

        # extra diagnostics
        print(f"[Debug] candidates from ?~C_E only   : {np.sum(trig_E & ~trig_F)}")
        print(f"[Debug] candidates from RMSE_F only: {np.sum(trig_F & ~trig_E)}")
        print(f"[Debug] candidates from both       : {np.sum(trig_E &  trig_F)}")
    else:
        cand_pos = np.where(sigma_E_eval > sigma_emp_E)[0]

    print(f"[Debug] total candidate positions   : {len(cand_pos)}")
    cand_idx = eval_idx[cand_pos]

    # 7) FPS‑W sampling -----------------------------------------------------
    sel_rel_pos, fps_scores_cand, dists_cand, score_all_cand, dist_all_cand = fps_uncertainty(
        G_eval_E[cand_pos], G_train, sigma_E_cal[eval_idx][cand_pos],
        beta=beta, drop_init=drop_init, min_k=min_k, score_floor=used_floor, verbose=True)

    # 8) logging ------------------------------------------------------------
    fps_full, dist_full = score_all_full.copy(), d_lat_eval.copy()
    sel_pos = cand_pos[sel_rel_pos]
    fps_full[sel_pos]  = fps_scores_cand
    dist_full[sel_pos] = dists_cand
    sel_idx  = cand_idx[sel_rel_pos]

    print(f"[AL‑ENS] picked {len(sel_idx)} / {len(cand_idx)} candidates")

    log_name = f"{base}_log.txt"
    with open(log_name, 'w') as f:
        f.write("Idx	σ_E_cal	d_lat	FPS_score	RMSE_F	Selected")
        for i, idx in enumerate(eval_idx):
            f.write(f"{idx}	{sigma_E_cal[idx]:.5f}	{d_lat_eval[i]:.5f}	{fps_full[i]:.5f}	"
                    f"{rmse_F_eval[i]:.5f}	{idx in sel_idx}")
    print(f"[Log] {log_name}")

    # 9) return -------------------------------------------------------------
    sel_objs = [all_frames[i - min(eval_idx)] for i in sel_idx]
    return sel_objs, sel_idx

# ======================
# MIG For Unlabelled
# ======================

import numpy as np
import torch
from scipy.linalg import solve_triangular

def predict_sigma_from_L(
        F_new: np.ndarray,
        L: np.ndarray,
        alpha_sq: float,
        batch: int | None = None
    ) -> np.ndarray:
    """
    Compute calibrated σ for frames not seen in GCV.
    L, alpha_sq from calibrate_alpha_reg_gcv().
    """
    if batch is None:
        G = solve_triangular(L, F_new.T, lower=True).T
        return np.sqrt(alpha_sq * np.sum(G**2, axis=1))

    sig = np.empty(len(F_new))
    for i in range(0, len(F_new), batch):
        blk = F_new[i:i+batch]
        G_blk = solve_triangular(L, blk.T, lower=True).T
        sig[i:i+batch] = np.sqrt(alpha_sq * np.sum(G_blk**2, axis=1))
    return sig

def adaptive_learning_mig_pool(
        pool_frames: list,
        F_pool: np.ndarray,
        F_train: np.ndarray,
        alpha_sq: float,
        L: np.ndarray,
        mu_E_pool: np.ndarray,            # <-- new arg
        beta: float = 0.5,
        drop_init: float = 0.5,
        min_k: int = 20,
        k_select: int = 50,
        base: str = "al_mig_pool"
    ) -> tuple[list, list]:
    # 1) compute force‐uncertainty for pool
    sigma_pool = predict_sigma_from_L(F_pool, L, alpha_sq)

    # 2) build G‐matrices for FPS
    G_train = solve_triangular(L, F_train.T, lower=True).T
    G_pool  = solve_triangular(L, F_pool.T, lower=True).T

    # 3) FPS‐Uncertainty selection
    sel_rel, fps_scores, dists = fps_uncertainty(
        G_pool, G_train, sigma_pool,
        beta=beta, drop_init=drop_init, min_k=min_k
    )

    # 4) pick top‐k by σ
    sel_rel = sorted(sel_rel, key=lambda i: sigma_pool[i], reverse=True)[:k_select]
    sel_frames = [pool_frames[i] for i in sel_rel]

    # 5) write indices + μE + σ
    idx_file = f"{base}_idxs.txt"
    with open(idx_file, "w") as f:
        f.write("idx\tpred_energy\tuncertainty\n")
        for i in sel_rel:
            f.write(f"{i}\t{mu_E_pool[i]:.6f}\t{sigma_pool[i]:.6f}\n")
    print(f"[Pool‑AL] saved selection indices + stats to {idx_file}")

    return sel_frames, sel_rel

def compute_bond_thresholds(
        frames,
        neighbor_list,
        first_shell_cutoff=3.0,
        device=None,
        cache_path="thresholds.npz"
    ):
    """
    Compute (or load) bonded‐pair thresholds (min, max, mean).
    Caches result in `cache_path`.
    """
    # 1) Try to load
    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=True) as data:
            thresholds = data["thresholds"].item()
        print(f"[compute_bond_thresholds] Loaded thresholds from {cache_path}")
        return thresholds

    # 2) Compute from scratch
    cache_file = os.path.join(neighbor_list.cache_path, "cache_0.pt")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file {cache_file} not found")
    cache_data = torch.load(cache_file)
    idx_i = cache_data["_idx_i"].numpy()
    idx_j = cache_data["_idx_j"].numpy()
    offsets = cache_data["_offsets"].numpy()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
          if device is None else device

    bond_lengths = {}
    n_frames = len(frames)
    for idx, frame in enumerate(frames):
        if idx % 50 == 0:
            print(f"compute_bond_thresholds: processing frame {idx+1}/{n_frames}")
        pos = frame.get_positions()
        Z   = frame.get_atomic_numbers()
        cell = frame.get_cell()
        for i, j, off in zip(idx_i, idx_j, offsets):
            if j <= i:
                continue
            p_i = pos[i] + np.dot(off, cell)
            p_j = pos[j]
            d = np.linalg.norm(p_i - p_j)
            if d > first_shell_cutoff:
                continue
            pair = tuple(sorted((Z[i], Z[j])))
            bond_lengths.setdefault(pair, []).append(d)

    thresholds = {
        pair: {
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
            "mean": float(np.mean(dists))
        }
        for pair, dists in bond_lengths.items()
    }
    print("Bond thresholds (pair: {min, max, mean}):")
    for pair, stats in thresholds.items():
        elems = tuple(chemical_symbols[z] for z in pair)
        print(f"  {elems}: {{min: {stats['min']:.3f}, max: {stats['max']:.3f}, mean: {stats['mean']:.3f}}}")

    # 3) Save cache
    np.savez_compressed(cache_path, thresholds=thresholds)
    print(f"[compute_bond_thresholds] Saved thresholds to {cache_path}")
    return thresholds


def filter_unrealistic_indices(
        frames,
        neighbor_list,
        thresholds,
        pct_tol=0.5,
        first_shell_cutoff=4.4,
        device=None,
        cache_path="bad_globals.npz"
    ):
    """
    Identify (or load) set of frame‐indices with impossible bonds.
    Caches result in `cache_path`.
    """
    # 1) Try to load
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        bad = set(data["bad_global"].tolist())
        print(f"[filter_unrealistic_indices] Loaded bad_global from {cache_path}")
        return bad

    # 2) Compute
    cache_file = os.path.join(neighbor_list.cache_path, "cache_0.pt")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file {cache_file} not found")
    cache_data = torch.load(cache_file)
    idx_i = cache_data["_idx_i"].numpy()
    idx_j = cache_data["_idx_j"].numpy()
    offsets = cache_data["_offsets"].numpy()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
          if device is None else device

    bad = []
    n_frames = len(frames)
    for idx, frame in enumerate(frames):
        if idx % 50 == 0:
            print(f"filter_unrealistic_indices: checking frame {idx+1}/{n_frames}")
        pos = frame.get_positions()
        Z   = frame.get_atomic_numbers()
        cell = frame.get_cell()
        flagged = False
        for i, j, off in zip(idx_i, idx_j, offsets):
            if j <= i:
                continue
            p_i = pos[i] + np.dot(off, cell)
            p_j = pos[j]
            d = np.linalg.norm(p_i - p_j)
            if d > first_shell_cutoff:
                continue
            pair = tuple(sorted((Z[i], Z[j])))
            if pair not in thresholds:
                continue
            d_min = thresholds[pair]["min"]
            d_max = thresholds[pair]["max"]
            d_mean = thresholds[pair]["mean"]
            lo = min((1 - pct_tol) * d_mean, d_min)
            hi = max((1 + pct_tol) * d_mean, d_max)
            if d < lo or d > hi:
                elems = tuple(chemical_symbols[z] for z in pair)
                print(
                    f"[Warning] Frame {idx+1}: bond {elems} distance {d:.3f} Å "
                    f"outside [{lo:.3f}, {hi:.3f}] (ref mean={d_mean:.3f}, "
                    f"min={d_min:.3f}, max={d_max:.3f} Å)"
                )
                bad.append(idx)
                flagged = True
                break
        if flagged:
            # stop checking further bonds in this frame
            continue

    bad_set = set(bad)

    # 3) Save cache
    np.savez_compressed(cache_path, bad_global=np.array(sorted(bad), dtype=int))
    print(f"[filter_unrealistic_indices] Saved bad_global to {cache_path}")
    return bad_set

def _compute_relative_thresholds(
        sigma_energy_train: np.ndarray,
        sigma_force_train:  np.ndarray,
        F_train:            np.ndarray,
        f_E: float = 1.30,          # 30 % above current worst train σ(E)
        f_F: float = 1.30,          # 30 % above current worst σ(F)
        f_Fmag: float = 1.20        # 20 % above current worst |F|max
):
    """Return absolute thresholds derived from training statistics."""
    thr_sigma_E = f_E    * float(sigma_energy_train.max())
    thr_sigma_F = f_F    * float(sigma_force_train.max())
    thr_Fmag    = f_Fmag * float(np.linalg.norm(F_train, axis=1).max())
    return thr_sigma_E, thr_sigma_F, thr_Fmag

def adaptive_learning_mig_pool_windowed(
        pool_frames: list,
        F_pool: np.ndarray,
        F_train: np.ndarray,
        alpha_sq: float,
        L: np.ndarray,
        forces_train: np.ndarray,
        sigma_energy: np.ndarray,
        sigma_force:  np.ndarray,
        mu_E_pool: np.ndarray,
        sigma_E_pool: np.ndarray,
        mu_F_pool: np.ndarray,
        sigma_F_pool: np.ndarray,
        bad_global: set,
        thr_E_hi: float=0.02, 
        rho_eV: float = 0.0025,
        min_k: int = 5,
        window_size: int = 100,
        base: str = "al_mig_pool_v6",
    ) -> tuple[list, list]:
    """
    Pool-based AL with windowed FPS and bond sanity filter via precomputed bad_global.
    """
    F_pool  = np.asarray(F_pool,  dtype=np.float64)
    F_train = np.asarray(F_train, dtype=np.float64)
    L       = np.asarray(L,       dtype=np.float64)

    G_train = solve_triangular(L, F_train.T, lower=True).T
    G_pool = solve_triangular(L, F_pool.T, lower=True).T
    reg = 1e-6                       # keep the same regulariser everywhere
    M_inv_global = np.linalg.inv(G_train.T @ G_train + reg * np.eye(G_train.shape[1]))

    n_pool_frames = len(pool_frames) 
    n_atoms       = pool_frames[0].get_positions().shape[0]

    # ------------------------------------------------------------------
    # 1 · fixed σ(E/atom) threshold  (5 meV atom⁻¹)
    # ------------------------------------------------------------------

    sigma_E_per_atom_train = sigma_energy / n_atoms
    sigma_E_per_atom_pool  = sigma_E_pool / n_atoms 
    thr_sigma_E_low = 0.002  # 2 meV/atom lower floor (as before)

    # ------------------------------------------------------------------
    # 2 · force–uncertainty thresholds (relative)
    # ------------------------------------------------------------------
    f_F      = 1.20    # 30% above max train uncertainty in force
    f_Fmean  = 1.15   # 15 % above train max σF mean
    f_Fmag   = 1.10    # 20% above max train force magnitude

    n_train_frames = sigma_force.shape[0] // (n_atoms * 3) 

    sigma_F_train       = sigma_force.reshape(n_train_frames, n_atoms, 3)
    sigma_F_train_max   = sigma_F_train.max(axis=(1, 2))
    sigma_F_train_mean  = np.linalg.norm(sigma_F_train, axis=2).mean(axis=1)

    thr_sigma_F     = f_F     * sigma_F_train_max .max()
    thr_sigma_Fmean = f_Fmean * sigma_F_train_mean.max()

    # ------------------------------------------------------------------
    # 3 · max-|F| threshold
    # ------------------------------------------------------------------
    force_magnitudes_train = np.linalg.norm(forces_train, axis=2)
    frame_max_force_train = force_magnitudes_train.max(axis=1)
    thr_Fmag = f_Fmag * frame_max_force_train.max()

    # Upper bounds (more conservative, 2x typical)
    sigma_E_per_atom_train_max = sigma_E_per_atom_train.max()
    sigma_F_train_max_max = sigma_F_train_max.max()
    sigma_F_train_mean_max = sigma_F_train_mean.max()
    frame_max_force_train_max = frame_max_force_train.max()

    thr_sigma_E_hi  = 0.05 # Hard cap. 3.0 * sigma_E_per_atom_train_max 
    thr_sigma_F_hi  = 2.0 * sigma_F_train_max_max
    thr_sigma_Fmean_hi = 2.0 * sigma_F_train_mean_max
    thr_Fmag_hi     = 2.0 * frame_max_force_train_max

    print(f"[Pool-AL] Lower/Upper bounds for metrics:")
    print(f"  sigma_E/atom:  > {thr_sigma_E_low:.4f}  < {thr_sigma_E_hi:.4f} eV")
    print(f"  sigma_F_max:   > {thr_sigma_F:.4f}  < {thr_sigma_F_hi:.4f} eV/Å")
    print(f"  sigma_F_mean:  > {thr_sigma_Fmean:.4f}  < {thr_sigma_Fmean_hi:.4f} eV/Å")
    print(f"  |F|_max:       > {thr_Fmag:.4f}  < {thr_Fmag_hi:.4f} eV/Å")
    
    # ------------------------------------------------------------------
    # 4 · pool-side per-frame scalars
    # ------------------------------------------------------------------
    mu_F_pool = mu_F_pool.reshape(n_pool_frames, n_atoms, 3)
    sigma_F_pool   = sigma_F_pool.reshape(n_pool_frames, n_atoms, 3)

    sigma_F_pool_max  = sigma_F_pool.max(axis=(1, 2))
    sigma_F_pool_mean = np.linalg.norm(sigma_F_pool, axis=2).mean(axis=1)
    frame_max_force_pool = np.linalg.norm(mu_F_pool, axis=2).max(axis=1)
    E_atom_pool = mu_E_pool / n_atoms

    # ------------------------------------------------------------------
    # 5 · counts
    # ------------------------------------------------------------------

    n_hi_E      = int((sigma_E_per_atom_pool  > thr_sigma_E_low).sum())
    n_hi_Fmax   = int((sigma_F_pool_max       > thr_sigma_F).sum())
    n_hi_Fmean  = int((sigma_F_pool_mean      > thr_sigma_Fmean).sum())
    n_hi_Fmag   = int((frame_max_force_pool   > thr_Fmag).sum())
    
    n_lo_E      = int((sigma_E_per_atom_pool  < thr_sigma_E_hi).sum())
    n_lo_Fmax   = int((sigma_F_pool_max       < thr_sigma_F_hi).sum())
    n_lo_Fmean  = int((sigma_F_pool_mean      < thr_sigma_Fmean_hi).sum())
    n_lo_Fmag   = int((frame_max_force_pool   < thr_Fmag_hi).sum())
    
    # Count frames in *both* bounds (eligible for selection, before OOD etc)
    n_in_E      = int(((sigma_E_per_atom_pool > thr_sigma_E_low) & (sigma_E_per_atom_pool < thr_sigma_E_hi)).sum())
    n_in_Fmax   = int(((sigma_F_pool_max > thr_sigma_F) & (sigma_F_pool_max < thr_sigma_F_hi)).sum())
    n_in_Fmean  = int(((sigma_F_pool_mean > thr_sigma_Fmean) & (sigma_F_pool_mean < thr_sigma_Fmean_hi)).sum())
    n_in_Fmag   = int(((frame_max_force_pool > thr_Fmag) & (frame_max_force_pool < thr_Fmag_hi)).sum())
    
    print(f"[AL] Frames above sigma_E/atom lower  : {n_hi_E}")
    print(f"[AL] Frames above sigma_F_max lower   : {n_hi_Fmax}")
    print(f"[AL] Frames above sigma_F_mean lower  : {n_hi_Fmean}")
    print(f"[AL] Frames above |F|_max lower       : {n_hi_Fmag}")
    print(f"[AL] Frames below sigma_E/atom upper  : {n_lo_E}")
    print(f"[AL] Frames below sigma_F_max upper   : {n_lo_Fmax}")
    print(f"[AL] Frames below sigma_F_mean upper  : {n_lo_Fmean}")
    print(f"[AL] Frames below |F|_max upper       : {n_lo_Fmag}")
    print(f"[AL] Frames in (lower, upper) for sigma_E/atom: {n_in_E}")
    print(f"[AL] Frames in (lower, upper) for sigma_F_max : {n_in_Fmax}")
    print(f"[AL] Frames in (lower, upper) for sigma_F_mean: {n_in_Fmean}")
    print(f"[AL] Frames in (lower, upper) for |F|_max     : {n_in_Fmag}")
    
    good_frames = len(pool_frames) - len(bad_global)
    print(f"[AL] analyzing {good_frames}/{len(pool_frames)} good frames (excluded {len(bad_global)} bad frames)")
    
    # ------------------------------------------------------------------
    # 6 · per-frame diagnostics file
    # ------------------------------------------------------------------
    diag_file = f"{base}_per_frame_diagnostics.txt"
    with open(diag_file, "w") as fh:
        fh.write("# idx\tσE_per_atom\tσF_max\tσF_mean\t|F|_max\n")
        for i in range(n_pool_frames):
            fh.write(f"{i}\t"
                     f"{sigma_E_per_atom_pool[i]:.6f}\t"
                     f"{sigma_F_pool_max[i]:.6f}\t"
                     f"{sigma_F_pool_mean[i]:.6f}\t"
                     f"{frame_max_force_pool[i]:.6f}\n")
    print(f"[AL] Wrote per-frame diagnostics to {diag_file}")

    # Save diagnostics as npz for plotting
    diag_npz = f"{base}_per_frame_diagnostics.npz"
    np.savez(
        diag_npz,
        sigma_E_per_atom=sigma_E_per_atom_pool,
        sigma_F_max=sigma_F_pool_max,
        sigma_F_mean=sigma_F_pool_mean,
        F_max=frame_max_force_pool,
    )
    print(f"[AL] Wrote per-frame diagnostics to {diag_npz}")

    if (n_hi_E + n_hi_Fmax + n_hi_Fmean + n_hi_Fmag) < 10:
        print("[AL] Convergence reached — nothing significant left to label.")
        return [], []
    # ------------------------------------------------------------------
    # Windowed D-optimal selection (γ₀ > 1 cutoff, no D_norm)
    # ------------------------------------------------------------------
    cand_global = []
    records     = []
    
    for w0 in range(0, n_pool_frames, window_size):
        w1  = min(w0 + window_size, n_pool_frames)
        win = list(range(w0, w1))
        print(f"\n[Pool-AL] Window {w0}-{w1}: {len(win)} total frames")

        # NEW FILTER: Only candidates exceeding any threshold and not in bad_global
        # 1. Not in bad_global (geometry)
        win_good = [i for i in win if i not in bad_global]
        print(f"[Pool-AL]   {len(win_good)} pass bad_global (loose geometry)")

        # 2. Energy cap
        win_E = [i for i in win_good if E_atom_pool[i] < thr_E_hi]
        print(f"[Pool-AL]   {len(win_E)} below E/atom cap ({thr_E_hi:.3f} eV)")

        # 3. Uncertainty/force/energy upper caps
        win_sigmaE = [i for i in win_E if sigma_E_per_atom_pool[i] < thr_sigma_E_hi]
        print(f"[Pool-AL]   {len(win_sigmaE)} below sigma_E/atom cap ({thr_sigma_E_hi:.3f} eV)")

        win_sigmaF = [i for i in win_sigmaE if sigma_F_pool_max[i] < thr_sigma_F_hi]
        print(f"[Pool-AL]   {len(win_sigmaF)} below sigma_F cap ({thr_sigma_F_hi:.3f} eV/Å)")

        win_sigmaFmean = [i for i in win_sigmaF if sigma_F_pool_mean[i] < thr_sigma_Fmean_hi]
        print(f"[Pool-AL]   {len(win_sigmaFmean)} below sigma_Fmean cap ({thr_sigma_Fmean_hi:.3f} eV/Å)")

        win_Fmag = [i for i in win_sigmaFmean if frame_max_force_pool[i] < thr_Fmag_hi]
        print(f"[Pool-AL]   {len(win_Fmag)} below |F| cap ({thr_Fmag_hi:.3f} eV/Å)")

        # 4. LOWER bounds: "interesting" (at least one metric above its lower bound)
        high = [
            i for i in win_Fmag if (
                (sigma_E_per_atom_pool[i]   > thr_sigma_E_low) or
                (sigma_F_pool_max[i]        > thr_sigma_F) or
                (sigma_F_pool_mean[i]       > thr_sigma_Fmean) or
                (frame_max_force_pool[i]    > thr_Fmag)
            )
        ]
        print(f"[Pool-AL]   {len(high)} above at least one 'interesting' threshold (lower bounds)")

        if not high:
            print(f"[Pool-AL]   No high-uncertainty frames in this window.")
            continue

        sub_G = G_pool[high].astype(np.float64)
        print(f"[Pool-AL]   sub_G shape = {sub_G.shape}")
   
        # raw Mahalanobis distances w.r.t. *current* training set
        quad0  = np.einsum("id,dk,ik->i", sub_G, M_inv_global, sub_G)
        gamma0 = np.sqrt(quad0)
        gamma_thr = 1.0  
        print(f"[Pool-AL]   Mahalanobis γ₀: mean={gamma0.mean():.3f}, "
              f"std={gamma0.std():.3f}, min={gamma0.min():.3f}, max={gamma0.max():.3f}")
        keep = np.where(gamma0 > gamma_thr)[0]
        print(f"[Pool-AL]   {len(keep)} frames are OOD (γ₀ > {gamma_thr:.2f})")
        if keep.size == 0:
           print(f"[Pool-AL]   Skipping window; no OOD candidates.")
           continue
        # ---- D-optimal ordering on OOD subset ----
        X_cand_OOD = sub_G[keep]
        order, gains_full, gamma0_sub = d_optimal_full_order(
            X_cand=X_cand_OOD,
            X_train=G_train,
            reg=reg,
        )
        print(
            f"[Pool-AL]   D-opt got {len(order)} OOD candidates, "
            f"gains ∈ [{gains_full.min():.3f}, {gains_full.max():.3f}]"
        )
    
        # Optional: print some of the top OOD γ₀/gain values
        for n in range(min(3, len(order))):
            idx = order[n]
            print(f"  [DEBUG][OOD]   Pick {n+1}: γ₀={gamma0[keep[idx]]:.3f}, Dgain={gains_full[n]:.3f}")

        # --- Hybrid gain threshold + min_k fallback ---
        gain_floor = np.percentile(gains_full, 80)  # top 20% only
        idx_keep = np.where(gains_full > gain_floor)[0]
        print(f"[Pool-AL]   {len(idx_keep)} frames above gain_floor={gain_floor:.3f}")
        
        if idx_keep.size == 0:
            # No gains above floor; fallback to first min_k picks
            idx_keep = order[:min_k]
            print(f"[Pool-AL]   Fallback: taking first {min_k} by D-opt order.")
        elif idx_keep.size < min_k:
            # If not enough, pad with next-best frames (by D-opt order)
            extras = [i for i in order if i not in idx_keep][: (min_k - idx_keep.size)]
            idx_keep = list(idx_keep) + extras
            print(f"[Pool-AL]   Padded to min_k={min_k} with D-opt order.")
        
        # Map back to global indices
        picks = [high[keep[i]] for i in idx_keep]
        print(f"[Pool-AL]   Selected {len(picks)} frames in this window (hybrid logic).")

        cand_global.extend(picks)

        for idx_local, pick in enumerate(picks):
            gain_val  = gains_full[idx_local]
            gamma_val = gamma0[keep[idx_keep[idx_local]]]
            records.append((
                pick,
                mu_E_pool[pick],
                sigma_E_pool[pick],
                gain_val,
                gamma_val,
                f"{w0}-{w1}"
            ))
        
    # ------------------------------------------------------------------
    # Write out the selection log
    # ------------------------------------------------------------------
    idx_file = f"{base}_idxs.txt"
    with open(idx_file, "w") as fh:
        fh.write("idx\tpred_E\tσ\tDgain\tgamma0\twindow\n")
        for idx, pred_E, sigma, dgain, g0, window in records:
            fh.write(f"{idx}\t{pred_E:.5f}\t{sigma:.5f}\t{dgain:.5f}\t{g0:.3f}\t{window}\n")
    print(f"[AL] wrote log with {len(records)} entries to {idx_file}")
    
    sel_frames = [pool_frames[i] for i in cand_global]
    return sel_frames, cand_global


# =============================================================================
# Traditional Active Learning Strategy (HDBSCAN-based)
# =============================================================================

def adaptive_learning(all_frames, eval_mask, sigma_atom_all_al, force_rmse_atom_array_al,
                        train_atom_mask, eval_atom_indices_arr_ignored, x=100,
                        base_filename="al_traditional"):
    """
    Performs Traditional Active Learning using HDBSCAN clustering and combined score selection.

    This function computes pairwise distance features for evaluation frames,
    clusters them with HDBSCAN, calculates scores based on aggregated uncertainty
    and error, and selects a set number of frames. It also saves the data needed
    for later plotting.

    Args:
        all_frames (list): List of ASE Atoms objects.
        eval_mask (np.ndarray): Boolean mask for evaluation frames.
        sigma_atom_all_al (np.ndarray): Flat array of per-atom uncertainties.
        force_rmse_atom_array_al (np.ndarray): Flat array of per-atom RMSE values.
        train_atom_mask (np.ndarray): Boolean mask for training atoms.
        eval_atom_indices_arr_ignored: (Ignored parameter.)
        x (int): Number of frames to select.
        base_filename (str): Base filename for saving plot data.

    Returns:
        tuple: (selected_frames_objects, selected_indices, plot_data_npz_path)
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan required for traditional AL.")
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for traditional AL.")

    print("\n--- Starting Traditional Active Learning (HDBSCAN) ---")
    n_frames = len(all_frames)
    atom_counts = [len(f) for f in all_frames]
    n_total_atoms = sum(atom_counts)
    cumulative_atoms = np.cumsum([0] + atom_counts)
    eps = 1e-9

    if len(eval_mask) != n_frames or len(sigma_atom_all_al) != n_total_atoms or \
       len(force_rmse_atom_array_al) != n_total_atoms or len(train_atom_mask) != n_total_atoms:
        raise ValueError("Input dimension mismatch in traditional AL.")

    print("Filtering per-atom data for evaluation set...")
    eval_frame_indices = np.where(eval_mask)[0]
    if len(eval_frame_indices) == 0:
        print("No eval frames.")
        return [], [], None

    atom_eval_mask = ~train_atom_mask
    filt_unc = sigma_atom_all_al[atom_eval_mask]
    filt_err = force_rmse_atom_array_al[atom_eval_mask]
    atom_to_frame_map = np.concatenate([np.full(atom_counts[i], i) for i in range(n_frames)])[atom_eval_mask]
    if len(filt_unc) == 0:
        print("No eval atoms found.")
        return [], [], None

    print("Step 1: Computing pairwise distance features...")
    distance_features_list = []
    import torch
    for i in eval_frame_indices:
        frame_pos = torch.tensor(all_frames[i].get_positions(), dtype=torch.float32)
        if len(frame_pos) < 2:
            continue
        distances = torch.cdist(frame_pos, frame_pos)
        triu_mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
        pairwise_distances = distances[triu_mask]
        sorted_distances = (torch.sort(pairwise_distances)[0].numpy()
                            if pairwise_distances.numel() > 0 else np.array([], dtype=np.float32))
        distance_features_list.append(sorted_distances)
    if not distance_features_list:
        print("No distances computed.")
        return [], [], None
    max_pairs = max(len(feat) for feat in distance_features_list) if distance_features_list else 0
    if max_pairs == 0:
        distance_features = np.zeros((len(eval_frame_indices), 1), dtype=np.float32)
    else:
        distance_features = np.array(
            [np.pad(f.astype(np.float32), (0, max_pairs - len(f)))
             if len(f) < max_pairs else f[:max_pairs].astype(np.float32)
             for f in distance_features_list], dtype=np.float32
        )

    print("Step 2: Clustering with HDBSCAN...")
    unique_eval_frames = eval_frame_indices  # Original indices
    num_unique_frames = len(unique_eval_frames)
    hdbscan_cluster_labels = np.full(num_unique_frames, -1, dtype=int)
    unique_clusters = []
    n_noise = num_unique_frames

    if distance_features.shape[0] < 5:
        print("Warning: Too few samples for reliable clustering. Skipping HDBSCAN.")
    else:
        if distance_features.shape[1] > 1 and np.any(np.var(distance_features, axis=0) > eps):
            n_pca_components = min(50, distance_features.shape[1], distance_features.shape[0] - 1)
            n_pca_components = max(1, n_pca_components)
            pca = PCA(n_components=n_pca_components, svd_solver='auto')
            try:
                distance_features_reduced = pca.fit_transform(distance_features)
            except Exception as e_pca:
                print(f"PCA failed: {e_pca}. Using original features.")
                distance_features_reduced = distance_features
        else:
            distance_features_reduced = distance_features

        print(f"Using feature shape for HDBSCAN: {distance_features_reduced.shape}")
        min_samples_hdbscan = min(5, max(2, distance_features_reduced.shape[0] // 10))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples_hdbscan,
                                    metric='euclidean',
                                    cluster_selection_method='eom',
                                    allow_single_cluster=True)
        try:
            hdbscan_cluster_labels = clusterer.fit_predict(distance_features_reduced)
        except Exception as e_hdb:
            print(f"HDBSCAN failed: {e_hdb}. Treating all as noise.")
        unique_clusters = np.unique(hdbscan_cluster_labels[hdbscan_cluster_labels >= 0])
        n_noise = np.sum(hdbscan_cluster_labels == -1)
        print(f"Clustered into {len(unique_clusters)} clusters (+ {n_noise} noise points).")

    print("Step 3: Computing frame scores...")
    atom_scores = filt_unc / (filt_err + eps)
    agg_error = np.zeros(num_unique_frames)
    agg_unc = np.zeros(num_unique_frames)
    mean_score = np.zeros(num_unique_frames)
    mean_top5_score = np.zeros(num_unique_frames)
    frame_idx_to_pos = {frame_idx: pos for pos, frame_idx in enumerate(unique_eval_frames)}
    for pos, f_idx in enumerate(unique_eval_frames):
        mask = (atom_to_frame_map == f_idx)
        if not np.any(mask):
            continue
        errors = filt_err[mask]
        uncs = filt_unc[mask]
        scores = atom_scores[mask]
        agg_error[pos] = np.nanmean(errors)
        agg_unc[pos] = np.nanmean(uncs)
        mean_score[pos] = np.nanmean(scores)
        if len(scores) > 0:
            top_perc_val = np.percentile(scores, 95)
            top_scores = scores[scores >= top_perc_val]
            if len(top_scores) == 0:
                top_scores = scores
            mean_top5_score[pos] = np.mean(top_scores)
        else:
            mean_top5_score[pos] = mean_score[pos]

    alpha = 0.50
    combined_score = alpha * mean_score + (1 - alpha) * mean_top5_score
    mean_rmse_eval = np.nanmean(agg_error)
    error_score = agg_error / max(eps, mean_rmse_eval)
    std_combined = np.nanstd(combined_score)
    std_cs = ((combined_score - np.nanmean(combined_score)) / max(eps, std_combined)
              if std_combined > eps else np.zeros_like(combined_score))
    std_error = np.nanstd(error_score)
    std_es = ((error_score - np.nanmean(error_score)) / max(eps, std_error)
              if std_error > eps else np.zeros_like(error_score))
    q_low, q_high = 0.05, 0.95
    z_threshold_low = np.sqrt(2) * erfinv(max(-1 + eps, min(1 - eps, 2 * q_low - 1)))
    z_threshold_high = np.sqrt(2) * erfinv(max(-1 + eps, min(1 - eps, 2 * q_high - 1)))
    N_over = np.sum(std_cs < z_threshold_low)
    N_under = np.sum(std_cs > z_threshold_high)
    N_conf = N_over + N_under
    N_high_error = np.sum(std_es > z_threshold_high)
    lambda_ = N_conf / max(eps, N_conf + N_high_error)
    overall_score = lambda_ * std_cs + (1 - lambda_) * std_es
    sigma_cs = np.nanstd(combined_score)
    av_cs = np.nanmean(combined_score)
    lower_thr_cs = av_cs + z_threshold_low * sigma_cs if sigma_cs > eps else av_cs
    upper_thr_cs = av_cs + z_threshold_high * sigma_cs if sigma_cs > eps else av_cs
    frame_conf_labels = ["over" if s < lower_thr_cs else "under" if s > upper_thr_cs else "within"
                         for s in combined_score]
    frame_error_labels = ["high" if se > z_threshold_high else "normal" for se in std_es]

    # --- Logging ---
    log_filename = "adaptive_learning_scores_traditional.log"
    try:
        with open(log_filename, "w") as flog:
            flog.write("Frame\tAggErr\tAggUnc\tCS\tTop5CS\tES\tOverall\tStdCS\tStdES\tConf\tError\n")
            for i, f_idx in enumerate(unique_eval_frames):
                flog.write(f"{f_idx}\t{agg_error[i]:.4f}\t{agg_unc[i]:.4f}\t{combined_score[i]:.4f}\t"
                           f"{mean_top5_score[i]:.4f}\t{error_score[i]:.4f}\t{overall_score[i]:.4f}\t"
                           f"{std_cs[i]:.4f}\t{std_es[i]:.4f}\t{frame_conf_labels[i]}\t{frame_error_labels[i]}\n")
            flog.write(f"\nSummary: Over={N_over}, Under={N_under}, HighErr={N_high_error}, Lambda={lambda_:.4f}\n")
            flog.write(f"CS Thr: Low={lower_thr_cs:.4f}, High={upper_thr_cs:.4f}\n")
    except Exception as e:
        print(f"Logging failed: {e}")

    # --- Step 4: Select Frames ---
    print("Step 4: Selecting frames...")
    selected_indices = []
    for cluster_id in unique_clusters:
        mask = (hdbscan_cluster_labels == cluster_id)
        cluster_orig_indices = unique_eval_frames[mask]
        cluster_scores = overall_score[mask]
        if len(cluster_orig_indices) > 0:
            selected_indices.append(cluster_orig_indices[np.argmax(cluster_scores)])
    noise_mask = (hdbscan_cluster_labels == -1)
    noise_orig_indices = unique_eval_frames[noise_mask]
    noise_scores = overall_score[noise_mask]
    if len(noise_scores) > 0:
        threshold = np.percentile(overall_score[~np.isnan(overall_score)], 95)
        selected_noise = noise_orig_indices[noise_scores > threshold]
        selected_indices.extend(list(selected_noise))
    selected_indices = list(np.unique(selected_indices))
    current_count = len(selected_indices)
    if current_count > x:
        scores_dict = {idx: overall_score[frame_idx_to_pos[idx]] for idx in selected_indices}
        selected_indices = sorted(selected_indices, key=lambda idx: scores_dict.get(idx, -np.inf), reverse=True)[:x]
    elif current_count < x:
        needed = x - current_count
        all_scores = [(idx, overall_score[frame_idx_to_pos[idx]]) for idx in unique_eval_frames
                      if not np.isnan(overall_score[frame_idx_to_pos[idx]])]
        candidates = [(idx, score) for idx, score in all_scores if idx not in selected_indices]
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected_indices.extend([idx for idx, _ in candidates[:needed]])
    print(f"Selected {len(selected_indices)} frames.")

    try:
        with open(log_filename, "a") as flog:
            flog.write("\n--- Selected Frames ---\n")
            flog.write("Frame\tOverall\tCS\tES\tStdCS\tStdES\tConf\tError\tCluster\n")
            for idx in selected_indices:
                pos = frame_idx_to_pos[idx]
                cluster_label = hdbscan_cluster_labels[pos]
                flog.write(f"{idx}\t{overall_score[pos]:.4f}\t{combined_score[pos]:.4f}\t"
                           f"{error_score[pos]:.4f}\t{std_cs[pos]:.4f}\t{std_es[pos]:.4f}\t"
                           f"{frame_conf_labels[pos]}\t{frame_error_labels[pos]}\t{cluster_label}\n")
    except Exception as e:
        print(f"Logging selected failed: {e}")

    # --- Save Plot Data ---
    npz_filename = f"{base_filename}_plot_data.npz"
    print(f"Saving plotting data to {npz_filename}...")
    try:
        save_dict = {
            "overall_score": overall_score,
            "agg_error": agg_error,
            "agg_unc": agg_unc,
            "combined_score": combined_score,
            "error_score": error_score,
            "frame_conf_labels": np.array(frame_conf_labels),
            "frame_error_labels": np.array(frame_error_labels),
            "hdbscan_cluster_labels": hdbscan_cluster_labels,
            "unique_eval_frames": unique_eval_frames,
            "selected_indices": np.array(selected_indices, dtype=int),
            "lower_thr_cs": lower_thr_cs,
            "upper_thr_cs": upper_thr_cs,
            "av_cs": av_cs,
            "z_threshold_high_es": z_threshold_high,
            "mean_es": np.nanmean(error_score),
            "std_es": np.nanstd(error_score)
        }
        np.savez_compressed(npz_filename, **save_dict)
    except Exception as e_save:
        print(f"Warning: Failed to save AL traditional plot data: {e_save}")
        npz_filename = None

    # --- Return ---
    selected_frames_objects = [all_frames[idx].copy() for idx in selected_indices]
    print(f"Traditional adaptive learning returning {len(selected_indices)} frames.")
    return selected_frames_objects, selected_indices, npz_filename

# End of active_learning.py

