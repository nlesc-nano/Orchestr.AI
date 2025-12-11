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
import time
from itertools import combinations
from typing import List, Optional, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from scipy.special import erfinv
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
import scipy.linalg
import scipy.linalg, scipy.spatial.distance
from ase.data import chemical_symbols
from ase import Atoms
from ase.geometry.analysis import Analysis
from collections import defaultdict
from typing import Tuple, List, Optional 

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

def debug_plot_rdfs(reference_frames,
                    rdf_thresholds,
                    r_max=6.0,
                    dr=0.02,
                    outprefix="rdf_DEBUG"):
    """
    For each species pair in rdf_thresholds:
    - build g_AB(r) from reference_frames
    - plot it
    - overlay vertical lines at r_soft and r_hard
    Writes: {outprefix}_{A}-{B}.png
    """
    print("[RDF] Debug plotting RDFs for each pair...")

    # 1. collect list of unique pairs from thresholds to control the loop order
    all_pairs = list(rdf_thresholds.keys())

    # precompute (positions, species) arrays for speed
    ref_pos  = [atoms.get_positions() for atoms in reference_frames]
    ref_syms = [atoms.get_chemical_symbols() for atoms in reference_frames]

    bins = np.arange(0.0, r_max + dr, dr)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    shell_vol = 4.0/3.0 * np.pi * (bins[1:]**3 - bins[:-1]**3)
    n_frames = len(reference_frames)

    for pair in all_pairs:
        A, B = pair
        # histogram accumulator for this pair
        hist = np.zeros(len(bins) - 1, dtype=np.float64)

        # loop frames
        for coords, syms in zip(ref_pos, ref_syms):
            idx_A = [ii for ii,s in enumerate(syms) if s == A]
            idx_B = [jj for jj,s in enumerate(syms) if s == B]
            if not idx_A or not idx_B:
                continue

            coords_A = np.asarray(coords)[idx_A]
            coords_B = np.asarray(coords)[idx_B]

            # all pairwise distances A-B
            dAB = np.linalg.norm(coords_A[:,None,:] - coords_B[None,:,:], axis=2)

            if A == B:
                # only upper triangle for same-species to avoid double count
                iu = np.triu_indices_from(dAB, k=1)
                d_flat = dAB[iu]
            else:
                d_flat = dAB.ravel()

            # keep only < r_max
            d_use = d_flat[d_flat < r_max]
            if d_use.size == 0:
                continue
            hist += np.histogram(d_use, bins=bins)[0]

        # normalize to something RDF-like
        # we're not going for perfect normalization here, we just want peak shape
        smooth = gaussian_filter1d(hist, sigma=2)

        # fetch thresholds
        r_soft, r_hard = rdf_thresholds[pair]

        # plot
        plt.figure(figsize=(5,4))
        plt.plot(r_centers, smooth, label=f"g_{A}-{B}(r) (smoothed)")
        plt.axvline(r_soft, color='red', linestyle='--', label=f"r_soft={r_soft:.2f} Å")
        plt.axvline(r_hard, color='orange', linestyle=':', label=f"r_hard={r_hard:.2f} Å")
        plt.xlabel("r (Å)")
        plt.ylabel("arb. RDF units")
        plt.title(f"{A}-{B} RDF (validation set)")
        plt.xlim(0.0, r_max)
        plt.ylim(bottom=0.0)
        plt.legend()
        plt.tight_layout()

        outname = f"{outprefix}_{A}-{B}.png"
        plt.savefig(outname, dpi=200)
        plt.close()
        print(f"[RDF] wrote {outname}")

def _collect_species_lists(frames, stride: int):
    """
    Return (positions_list, types_list, unique_pairs) using only every `stride`-th frame.
    stride=1 means use all frames.
    """
    positions_list = []
    types_list     = []
    species_sets   = set()

    # subsample frames deterministically
    for fr_i in range(0, len(frames), stride):
        fr   = frames[fr_i]
        pos  = fr.get_positions()
        syms = fr.get_chemical_symbols()

        positions_list.append(pos.copy())
        types_list.append(np.array(syms, dtype=object))

        uniq = np.unique(syms)
        # build all unordered element pairs present in this frame
        for a_i, a in enumerate(uniq):
            for b in uniq[a_i:]:
                pair = tuple(sorted((a, b)))
                species_sets.add(pair)

    unique_pairs = sorted(species_sets)
    return positions_list, types_list, unique_pairs

def _gather_pair_distances_vectorized(positions_list, types_list, A, B, cutoff):
    """
    Vectorized distance collection for one pair (A,B) across the *subsampled* frames.
    """
    pair_dists = []
    for pos, types in zip(positions_list, types_list):
        idx_A = np.where(types == A)[0]
        idx_B = np.where(types == B)[0]
        if idx_A.size == 0 or idx_B.size == 0:
            continue

        coords_A = pos[idx_A]  # (nA,3)
        coords_B = pos[idx_B]  # (nB,3)

        # all pairwise distances A-B
        d_ij = coords_A[:, None, :] - coords_B[None, :, :]
        d2   = np.einsum("ijk,ijk->ij", d_ij, d_ij)
        d    = np.sqrt(d2)

        if A == B:
            iu = np.triu_indices_from(d, k=1)
            d  = d[iu]

        d = d[d <= cutoff]
        if d.size > 0:
            pair_dists.append(d)

    if not pair_dists:
        return np.array([], dtype=float)
    return np.concatenate(pair_dists)

def compute_rdf_thresholds_from_reference(
    reference_frames,
    cutoff: float = 6.0,
    bins: int = 300,
    stride: int = 1,
    min_peak_height_frac: float = 0.05,
    min_peak_prominence_frac: float = 0.10,
    left_baseline_frac: float = 0.05,
    beta: float = 0.5,
    r_min_physical: float = 1.0,
):
    """
    Build RDFs for each element pair using ONLY trusted reference frames
    (validation set), then infer two cutoffs per pair:
        r_soft = start of 1st-neighbour shell
        r_hard = beta * r_soft   (beta ~ 0.5 per your request)

    Fix vs previous version:
    - we NO LONGER take global argmax.
    - we detect ALL local peaks in the smoothed RDF and choose
      the *leftmost physically meaningful* one (closest to 0 Å),
      provided it's tall/prominent enough.
    - this avoids picking the 2nd shell just because it's taller.

    Returns
    -------
    thresholds : dict { (A,B): (r_soft, r_hard) }
    Also caches per-pair r, smooth_rdf for debugging in
    rdf_thresholds_cache.npz (same filename you already save).
    """

    import numpy as np
    from collections import defaultdict

    t0_all = time.time()

    # -------- 0. subsample trusted frames for speed --------
    ref_use = reference_frames[::stride]
    print(f"[RDF] Using {len(ref_use)}/{len(reference_frames)} reference frames (stride={stride}).")

    # -------- 1. collect all pair distances per species pair --------
    pair_dists = defaultdict(list)
    for fr in ref_use:
        pos  = fr.get_positions()
        syms = fr.get_chemical_symbols()
        n    = len(syms)
        for i, j in combinations(range(n), 2):
            rij = np.linalg.norm(pos[i] - pos[j])
            if rij <= cutoff:
                pair = tuple(sorted((syms[i], syms[j])))
                pair_dists[pair].append(rij)

    print(f"[RDF] Found {len(pair_dists)} unique species pairs.")

    thresholds = {}
    debug_rdfs = {}

    # precompute bin edges once
    r_edges = np.linspace(0.0, cutoff, bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    for pair, dlist in pair_dists.items():
        t0 = time.time()
        darray = np.asarray(dlist, dtype=np.float64)

        if darray.size < 20:
            print(f"[RDF][{pair}] WARNING: only {darray.size} samples, skipping cutoff inference.")
            continue

        # histogram of raw distances
        hist, _ = np.histogram(darray, bins=r_edges, density=False)

        # smooth RDF proxy
        smooth = gaussian_filter1d(hist.astype(np.float64), sigma=2)

        # Keep for debug/plot
        debug_rdfs[pair] = {
            "r": r_centers.copy(),
            "g_smooth": smooth.copy(),
        }

        # -------- 2. peak detection (all local maxima) --------
        # local max: smooth[k] > smooth[k-1] and smooth[k] >= smooth[k+1]
        locmax_idx = []
        for k in range(1, len(smooth) - 1):
            if smooth[k] > smooth[k-1] and smooth[k] >= smooth[k+1]:
                locmax_idx.append(k)
        locmax_idx = np.array(locmax_idx, dtype=int)

        if locmax_idx.size == 0:
            print(f"[RDF][{pair}] WARNING: no local maxima, skipping.")
            continue

        peak_vals = smooth[locmax_idx]
        global_max = peak_vals.max()

        # require some minimal absolute height relative to tallest peak
        height_mask = peak_vals >= (min_peak_height_frac * global_max)

        # require some "prominence": peak must be higher than its local baseline
        # crude prominence = peak_val - min(neighboring valley in small window)
        prominence_mask = []
        for idx, pidx in enumerate(locmax_idx):
            left_win  = max(0, pidx - 5)
            right_win = min(len(smooth), pidx + 6)
            local_min = np.min(smooth[left_win:right_win])
            prom = peak_vals[idx] - local_min
            prominence_mask.append(prom >= (min_peak_prominence_frac * global_max))
        prominence_mask = np.array(prominence_mask, dtype=bool)

        good_mask = height_mask & prominence_mask

        # also enforce physical lower bound r > r_min_physical
        good_mask = good_mask & (r_centers[locmax_idx] > r_min_physical)

        good_peaks = locmax_idx[good_mask]

        if good_peaks.size == 0:
            print(f"[RDF][{pair}] WARNING: no 'good' peak after filtering, "
                  f"falling back to earliest local max above r_min_physical.")
            # fallback: first peak to the right of r_min_physical
            fallback_idx = np.argmin(
                np.where(r_centers[locmax_idx] > r_min_physical,
                         r_centers[locmax_idx],
                         np.inf)
            )
            chosen_peak_idx = locmax_idx[fallback_idx]
        else:
            # choose the LEFTMOST "good" peak (smallest radius)
            # -> THIS is the first coordination shell we care about
            chosen_peak_idx = good_peaks[np.argmin(r_centers[good_peaks])]

        peak_r   = r_centers[chosen_peak_idx]
        peak_val = smooth[chosen_peak_idx]

        # -------- 3. locate left edge of that chosen peak --------
        # we walk left from chosen_peak_idx until smooth falls below
        # left_baseline_frac * peak_val, then take that radius as r_soft.
        thresh_val = left_baseline_frac * peak_val
        left_region = np.where(
            (r_centers[:chosen_peak_idx] > r_min_physical) &
            (smooth[:chosen_peak_idx] < thresh_val)
        )[0]

        if left_region.size > 0:
            left_edge_idx = left_region[-1]
            r_soft = float(r_centers[left_edge_idx])
        else:
            # no clean baseline → just back off ~1 bin from the peak,
            # but never below r_min_physical
            approx_soft = max(r_min_physical,
                              r_centers[max(chosen_peak_idx - 1, 0)])
            # additional sanity: can't exceed peak_r
            r_soft = float(min(approx_soft, peak_r))

        # final hard cutoff
        r_hard = beta * r_soft  # beta now 0.5 per your request

        thresholds[pair] = (r_soft, r_hard)

        dt = time.time() - t0
        print(f"[RDF]   {pair}: "
              f"N={darray.size} samples, "
              f"peak@{peak_r:.3f} Å, "
              f"r_soft={r_soft:.3f} Å, "
              f"r_hard={r_hard:.3f} Å, "
              f"time={dt:.2f}s")

    total_dt = time.time() - t0_all
    print(f"[RDF] Done building thresholds for {len(thresholds)} pairs in "
          f"{total_dt:.2f}s total (stride={stride}).")

    # cache for debugging / plotting
    np.savez_compressed(
        "rdf_thresholds_cache.npz",
        thresholds=np.array(
            [(str(p[0]), str(p[1]), rs, rh)
             for p, (rs, rh) in thresholds.items()],
            dtype=object,
        ),
        rdfs={str(p): debug_rdfs[p] for p in debug_rdfs},
        meta=dict(
            cutoff=cutoff,
            bins=bins,
            stride=stride,
            beta=beta,
            min_peak_height_frac=min_peak_height_frac,
            min_peak_prominence_frac=min_peak_prominence_frac,
            left_baseline_frac=left_baseline_frac,
            r_min_physical=r_min_physical,
        )
    )

    return thresholds

def fast_filter_by_rdf_kdtree(
    frames,
    rdf_thresholds,
    verbose=True,
    debug_first_bad=5,
):
    """
    Reject unphysical geometries using KDTree neighbor search.

    We say a frame is "catastrophic" (reject) if it contains ANY atom pair
    whose distance is below that pair's r_hard.

    Parameters
    ----------
    frames : list[ase.Atoms]
        Frames to check.
    rdf_thresholds : dict
        {('In','P'): (r_soft, r_hard), ...}
        NOTE: we assume ('A','B') keys are sorted tuples, same convention
        as compute_rdf_thresholds_from_reference.
    verbose : bool
        If True, prints progress every ~100 frames and summary at end.
    debug_first_bad : int
        For the first N rejected frames, print which pair / distance broke it.

    Returns
    -------
    ok_mask : np.ndarray[bool]
        True  -> frame passes (all interatomic distances above r_hard for
                every known pair)
        False -> at least one catastrophic contact < r_hard
    """

    from scipy.spatial import cKDTree

    n_frames = len(frames)
    ok_mask  = np.ones(n_frames, dtype=bool)

    if not rdf_thresholds:
        print("[RDF] WARNING: No rdf_thresholds provided. Keeping all frames.")
        return ok_mask

    # largest hard cutoff across all species pairs
    r_hard_max = max(r_hard for (_, r_hard) in rdf_thresholds.values())
    if verbose:
        print(f"[RDF] Using r_hard_max = {r_hard_max:.3f} Å for KDTree neighbor search.")

    # We'll accumulate some diagnostics:
    # - shortest distance we ever saw per pair
    # - how many frames we killed
    global_min_dist_per_pair = {}
    n_rejected = 0
    printed_bad = 0

    t0g = time.perf_counter()

    for f_idx, atoms in enumerate(frames):
        if verbose and (f_idx % 100 == 0):
            dt = time.perf_counter() - t0g
            print(f"[RDF]   Checking frame {f_idx}/{n_frames} (elapsed {dt:.1f} s)")

        pos  = atoms.get_positions()
        syms = atoms.get_chemical_symbols()
        n_atoms = len(pos)
        if n_atoms < 2:
            continue  # trivially OK

        # KDTree for neighbors within r_hard_max
        tree = cKDTree(pos)
        close_pairs = tree.query_pairs(r_hard_max, output_type='ndarray')
        if close_pairs.size == 0:
            continue  # nobody closer than r_hard_max ⇒ frame passes trivially

        # Compute all those close distances
        diff  = pos[close_pairs[:, 0]] - pos[close_pairs[:, 1]]
        dists = np.linalg.norm(diff, axis=1)

        # Check for catastrophic contacts
        bad_frame = False
        for (i_atom, j_atom), rij in zip(close_pairs, dists):
            pair = tuple(sorted((syms[i_atom], syms[j_atom])))
            # track global min distance we ever see for this pair
            prev = global_min_dist_per_pair.get(pair, np.inf)
            if rij < prev:
                global_min_dist_per_pair[pair] = rij

            th = rdf_thresholds.get(pair, None)
            if th is None:
                # This pair type didn't get a cutoff from reference,
                # so we can't classify it as catastrophic. Ignore.
                continue

            _, r_hard = th
            if rij < r_hard:
                # Catastrophic overlap for this frame
                ok_mask[f_idx] = False
                bad_frame = True

                # Verbose debug for first few rejected frames
                if printed_bad < debug_first_bad:
                    print(f"[RDF][REJECT] frame {f_idx}: "
                          f"{pair} at {rij:.3f} Å < r_hard {r_hard:.3f} Å")
                    printed_bad += 1
                break  # no need to keep checking pairs in this frame

        if bad_frame:
            n_rejected += 1

    dt_tot = time.perf_counter() - t0g
    n_ok = int(np.sum(ok_mask))
    if verbose:
        print(f"[RDF] Finished in {dt_tot:.2f} s for {n_frames} frames "
              f"({dt_tot / max(n_frames,1):.4f} s/frame).")
        print(f"[RDF] Geometric sanity: {n_ok}/{n_frames} frames OK "
              f"({100.0 * n_ok / max(n_frames,1):.1f}%).")
        print(f"[RDF] Rejected {n_rejected} frames total.")

        # print summary of closest approaches per pair
        if global_min_dist_per_pair:
            print("[RDF] Closest distances observed per pair (Å):")
            for pair, dmin in sorted(global_min_dist_per_pair.items(),
                                     key=lambda x: x[0]):
                soft_hard = rdf_thresholds.get(pair, (None, None))
                print(f"        {pair}: "
                      f"min_seen={dmin:.3f} Å, "
                      f"r_hard={soft_hard[1]:.3f} Å, "
                      f"r_soft={soft_hard[0]:.3f} Å")

    return ok_mask

def collect_pair_distances(frames, cutoff: float = 6.0):
    """
    Build a dict of all pair distances per element pair across a list of frames.
    
    Returns
    -------
    pair_distances : dict
        { (A,B): [r_1, r_2, ...] } with A,B sorted alphabetically.
    """
    from collections import defaultdict
    import numpy as np

    pair_distances = defaultdict(list)

    for atoms in frames:
        pos = atoms.get_positions()
        syms = atoms.get_chemical_symbols()
        n = len(syms)
        for i, j in combinations(range(n), 2):
            rij = np.linalg.norm(pos[i] - pos[j])
            if rij <= cutoff:
                pair = tuple(sorted((syms[i], syms[j])))
                pair_distances[pair].append(rij)

    # convert lists to numpy arrays for convenience
    for pair, vals in pair_distances.items():
        pair_distances[pair] = np.asarray(vals, dtype=float)

    return pair_distances

def make_rdf_hist(pair_distance_dict,
                  cutoff: float = 6.0,
                  bins: int = 300,
                  smooth_sigma: float = 2.0):
    """
    Turn the pair-distance dict from collect_pair_distances(...) into
    smoothed histograms that look like RDFs (not normalized by shell volume,
    just a density-like diagnostic).

    Returns
    -------
    rdf_dict : dict
        {
          (A,B): {
             "r": bin_centers,
             "rdf": smoothed_hist
          },
          ...
        }
    """
    import numpy as np

    rdf_dict = {}

    for pair, dists in pair_distance_dict.items():
        if dists.size == 0:
            continue

        hist, edges = np.histogram(
            dists,
            bins=bins,
            range=(0.0, cutoff),
            density=True
        )
        r_centers = 0.5 * (edges[:-1] + edges[1:])
        smooth = gaussian_filter1d(hist, sigma=smooth_sigma)

        rdf_dict[pair] = {
            "r": r_centers,
            "rdf": smooth,
        }

    return rdf_dict

def plot_rdf_comparison(
        pair,
        rdf_ref,
        rdf_all,
        rdf_kept,
        rdf_thresholds,
        r_max: float = 4.0,
        outprefix: str = "rdf_poolcheck"
    ):
    """
    For a given element pair (A,B), compare:
      - reference RDF (trusted validation frames)
      - all thinned pool frames
      - only thinned pool frames that pass RDF hard cutoff

    Also draw vertical lines for r_hard and r_soft from rdf_thresholds.

    Saves to disk as f"{outprefix}_{A}{B}.png".
    """

    # If we don't have this pair in ref, just skip
    if pair not in rdf_ref:
        print(f"[RDF-plot] pair {pair} not in rdf_ref, skipping.")
        return

    r_ref   = rdf_ref[pair]["r"]
    g_ref   = rdf_ref[pair]["rdf"]

    r_all   = rdf_all.get(pair, {}).get("r",   None)
    g_all   = rdf_all.get(pair, {}).get("rdf", None)

    r_kept  = rdf_kept.get(pair, {}).get("r",   None)
    g_kept  = rdf_kept.get(pair, {}).get("rdf", None)

    r_soft, r_hard = rdf_thresholds.get(pair, (None, None))

    fig, ax = plt.subplots(figsize=(4,3), dpi=150)

    ax.plot(r_ref, g_ref, label="ref (val_mask)", linewidth=1.5)

    if r_all is not None and g_all is not None:
        ax.plot(r_all, g_all, label="pool_thin (all)", linestyle="--", linewidth=1.0)

    if r_kept is not None and g_kept is not None:
        ax.plot(r_kept, g_kept, label="pool_thin (RDF ok)", linestyle=":", linewidth=1.0)

    # draw verticals for hard/soft cutoffs
    if r_soft is not None:
        ax.axvline(r_soft, color="green", linestyle=":", linewidth=1.0, label="r_soft")
    if r_hard is not None:
        ax.axvline(r_hard, color="red", linestyle="--", linewidth=1.0, label="r_hard")

    ax.set_xlim(0.0, r_max)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r) ~ density")
    ax.set_title(f"RDF {pair[0]}-{pair[1]}")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()

    outname = f"{outprefix}_{pair[0]}{pair[1]}.png"
    fig.savefig(outname, dpi=200)
    plt.close(fig)

    print(f"[RDF-plot] Wrote {outname}")

def adaptive_learning_ensemble_calibrated(
        all_frames: List,
        eval_mask: np.ndarray,
        delta_E_frame: np.ndarray,
        mean_l_al: np.ndarray,
        *,
        force_rmse_per_comp: Optional[np.ndarray] = None,
        denom_all: Optional[np.ndarray] = None,
        reference_frames: Optional[List] = None,  
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
    # --- RDF-based physical sanity filter ---
    # reference_frames should come from eval_input_xyz (DFT baseline),
    # not from incrementally grown training set.
    if reference_frames is not None and len(reference_frames) > 0:
        rdf_thresholds = compute_rdf_thresholds_from_reference(reference_frames)
        eval_frames_list = [all_frames[i] for i in eval_idx]
        realistic_mask = filter_by_rdf(eval_frames_list, rdf_thresholds)
        print(f"[RDF] realistic_mask: {realistic_mask.sum()}/{len(realistic_mask)} eval frames pass hard cutoff")
    else:
        print("[RDF] WARNING: reference_frames not provided; assuming all eval frames are physically valid.")
        realistic_mask = np.ones(len(eval_idx), dtype=bool)

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
#    keep_mask = hybrid > hybrid_floor
    keep_mask = (hybrid > hybrid_floor) & realistic_mask
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
        mu_E_frame_train: np.ndarray,
        mu_E_pool: np.ndarray,
        sigma_E_pool: np.ndarray,
        mu_F_pool: np.ndarray,
        sigma_F_pool: np.ndarray,
        rdf_thresholds: dict,
        # ---- NEW: hard AL triggers (should come from config.yaml) ----
        hard_sigma_E_atom_min: float = 0.002,   # eV/atom  (2 meV/atom)
        hard_sigma_F_mean_min: float = 0.10,    # eV/Å
        hard_sigma_F_max_min:  float = 0.20,    # eV/Å
        hard_Fmax_train_mult:  float = 1.5,     # × max |F| in TRAIN
        # --------------------------------------------------------------
        rho_eV: float = 0.0025,
        min_k: int = 5,
        window_size: int = 100,
        base: str = "al_mig_pool_v7",
        budget_max: int = 50,
        percentile_gamma: float = 90.0,
        percentile_F_low: float = 95.0,
        percentile_F_hi: float = 99.0,
    ) -> tuple[list, list]:
    """
    Pool-based Active Learning (AL) with:
      - RDF geometric sanity filter
      - percentile-based lower and upper bounds
      - per-atom energy CI overlap sanity
      - novelty screening via Mahalanobis distance γ0 > γ_thr
      - D-opt diversity selection
      - global score ranking + budget
      - diagnostics + convergence report
      - **NEW**: hard literature-style AL triggers on σ(E), σF_mean, σF_max
      - **NEW**: training-dataset per-frame uncertainty dump at the bottom
    """

    import numpy as np
    from scipy.linalg import solve_triangular

    # ---------------- helper ----------------
    def clamp_threshold(name: str, lower: float, upper: float) -> float:
        """Ensure upper cap is never below lower bound."""
        if upper < lower:
            print(f"[AL][WARN] {name}: upper cap {upper:.6f} < lower {lower:.6f} — corrected to lower.")
            return lower
        return upper

    print(f"\n[AL] --- adaptive_learning_mig_pool_windowed (v7+hard) ---")
    print(f"[AL] percentile_gamma  = {percentile_gamma:.1f}% (γ novelty)")
    print(f"[AL] percentile_F_low  = {percentile_F_low:.1f}% (lower bounds)")
    print(f"[AL] percentile_F_hi   = {percentile_F_hi:.1f}% (adaptive caps)")
    print(f"[AL] budget_max        = {budget_max}")
    print(f"[AL] hard_sigma_E_atom_min = {hard_sigma_E_atom_min:.6f} eV/atom")
    print(f"[AL] hard_sigma_F_mean_min = {hard_sigma_F_mean_min:.6f} eV/Å")
    print(f"[AL] hard_sigma_F_max_min  = {hard_sigma_F_max_min:.6f} eV/Å")
    print(f"[AL] hard_Fmax_train_mult  = {hard_Fmax_train_mult:.3f} × train |F|_max")

    # ------------------------------------------------------------------
    # latent-space prep (whitening transform via L)
    # ------------------------------------------------------------------
    F_pool  = np.asarray(F_pool,  dtype=np.float64)
    F_train = np.asarray(F_train, dtype=np.float64)
    L       = np.asarray(L,       dtype=np.float64)

    G_train = solve_triangular(L, F_train.T, lower=True).T
    G_pool  = solve_triangular(L, F_pool.T,  lower=True).T

    reg = 1e-6
    M_inv_global = np.linalg.inv(G_train.T @ G_train + reg * np.eye(G_train.shape[1]))

    gamma_train = np.sqrt(np.einsum('id,dk,ik->i', G_train, M_inv_global, G_train))
    gamma_thr   = np.quantile(gamma_train, percentile_gamma / 100.0)

    print("\n[AL] --- γ (Mahalanobis novelty) ---")
    print(f"[AL] γ_train mean={gamma_train.mean():.4f} min={gamma_train.min():.4f} max={gamma_train.max():.4f}")
    print(f"[AL] γ_thr ({percentile_gamma:.1f}th pctl) = {gamma_thr:.6f}")

    # ------------------------------------------------------------------
    # Classical Mahalanobis distance (distributional / basin check)
    # ------------------------------------------------------------------
    mu_Gtrain = G_train.mean(axis=0)
    Cov_Gtrain = np.cov(G_train, rowvar=False)
    eps_cov = 1e-6
    Cov_Gtrain_reg = Cov_Gtrain + eps_cov * np.eye(Cov_Gtrain.shape[0])
    Cov_inv = np.linalg.inv(Cov_Gtrain_reg)

    diff_train = G_train - mu_Gtrain
    dM2_train = np.einsum("id,dk,ik->i", diff_train, Cov_inv, diff_train)
    dM_train  = np.sqrt(dM2_train)

    basin_percentile = 99.0
    dM_thr = np.quantile(dM_train, basin_percentile / 100.0)

    print("\n[AL] --- dM (classical Mahalanobis-to-mean) ---")
    print(f"[AL] dM_train mean={dM_train.mean():.4f} min={dM_train.min():.4f} max={dM_train.max():.4f}")
    print(f"[AL] dM_thr ({basin_percentile:.1f}th pctl) = {dM_thr:.6f}")

    # ------------------------------------------------------------------
    # system sizes and reshaping
    # ------------------------------------------------------------------
    n_pool_frames = len(pool_frames)
    n_atoms_pool  = pool_frames[0].get_positions().shape[0]

    n_train_frames, n_atoms_train, _ = forces_train.shape
    n_train_frames_check = sigma_force.shape[0] // (n_atoms_train * 3)
    if n_train_frames_check != n_train_frames:
        print("[WARN] n_train_frames inferred from sigma_force does not match forces_train.shape[0]. "
              f"({n_train_frames_check} vs {n_train_frames}) Using forces_train.shape[0].")

    print("\n[DEBUG] --- System sizes ---")
    print(f"[DEBUG]   n_train_frames = {n_train_frames}")
    print(f"[DEBUG]   n_pool_frames  = {n_pool_frames}")
    print(f"[DEBUG]   n_atoms_train  = {n_atoms_train}")
    print(f"[DEBUG]   n_atoms_pool   = {n_atoms_pool}")

    # reshape sigma_F for training
    sigma_F_train = sigma_force.reshape(n_train_frames, n_atoms_train, 3)

    # ------------------------------------------------------------------
    # ENERGY PER ATOM SETUP
    # ------------------------------------------------------------------
    mu_E_atom_train         = mu_E_frame_train / n_atoms_train
    sigma_E_per_atom_train  = sigma_energy    / n_atoms_train

    mu_E_atom_pool          = mu_E_pool       / n_atoms_pool
    sigma_E_per_atom_pool   = sigma_E_pool    / n_atoms_pool

    thr_sigma_E_low = np.percentile(sigma_E_per_atom_train, percentile_F_low)
    thr_E_hi_atom   = mu_E_atom_train.max() + 0.5  # eV/atom

    E_train_min_atom = mu_E_atom_train.min()
    E_train_max_atom = mu_E_atom_train.max()

    CI_K = 3.0
    E_lo_pool_atom = (mu_E_pool - CI_K * sigma_E_pool) / n_atoms_pool
    E_hi_pool_atom = (mu_E_pool + CI_K * sigma_E_pool) / n_atoms_pool

    margin_hi_total    = 2.0  # eV total, legacy slack
    margin_hi_per_atom_legacy = margin_hi_total / n_atoms_train

    print("\n[DEBUG] --- Energy per atom stats (TRAIN) ---")
    print(f"[DEBUG]   mu_E_atom_train mean={mu_E_atom_train.mean():.6f} min={mu_E_atom_train.min():.6f} max={mu_E_atom_train.max():.6f}")
    print(f"[DEBUG]   sigma_E_atom_train mean={sigma_E_per_atom_train.mean():.6f} min={sigma_E_per_atom_train.min():.6f} max={sigma_E_per_atom_train.max():.6f}")
    print(f"[DEBUG]   thr_sigma_E_low (pctl {percentile_F_low:.1f}%) = {thr_sigma_E_low:.6f} eV/atom")
    print(f"[DEBUG]   thr_E_hi_atom = {thr_E_hi_atom:.6f} eV/atom")

    # ------------------------------------------------------------------
    # FORCE / UNCERTAINTY SETUP  (percentile-based)
    # ------------------------------------------------------------------
    sigma_F_train_max   = sigma_F_train.max(axis=(1, 2))
    sigma_F_train_mean  = np.linalg.norm(sigma_F_train, axis=2).mean(axis=1)
    force_magnitudes_train = np.linalg.norm(forces_train, axis=2)
    frame_max_force_train  = force_magnitudes_train.max(axis=1)

    thr_sigma_F      = np.percentile(sigma_F_train_max,  percentile_F_low)
    thr_sigma_Fmean  = np.percentile(sigma_F_train_mean, percentile_F_low)
    thr_Fmag         = np.percentile(frame_max_force_train, percentile_F_low)

    # legacy fallback "upper caps"
    thr_sigma_E_hi_legacy     = 0.01  # eV/atom
    thr_sigma_F_hi_legacy     = 2.0 * sigma_F_train_max.max()
    thr_sigma_Fmean_hi_legacy = 2.0 * sigma_F_train_mean.max()
    thr_Fmag_hi_legacy        = 2.0 * frame_max_force_train.max()

    # ---- NEW: train |F| cap for pool (physicality) ----
    train_Fmax_hard_cap = float(frame_max_force_train.max()) * float(hard_Fmax_train_mult)
    print(f"[AL] Train max |F| = {frame_max_force_train.max():.6f} eV/Å → hard pool cap = {train_Fmax_hard_cap:.6f} eV/Å")

    # --- TRAIN MEDIANS for info (we no longer use RATIO_FLOOR for selection) ---
    avg_sigmaF_train_max  = float(np.median(sigma_F_train_max))
    avg_sigmaF_train_mean = float(np.median(sigma_F_train_mean))
    avg_Fmag_train        = float(np.median(frame_max_force_train))

    print("\n[TRAIN medians (informative)]")
    print(f"[TRAIN] median σF_max  = {avg_sigmaF_train_max:.6f} eV/Å")
    print(f"[TRAIN] median σF_mean = {avg_sigmaF_train_mean:.6f} eV/Å")
    print(f"[TRAIN] median |F|_max = {avg_Fmag_train:.6f} eV/Å")

    # reshape pool forces arrays
    mu_F_pool    = mu_F_pool.reshape(n_pool_frames, n_atoms_pool, 3)
    sigma_F_pool = sigma_F_pool.reshape(n_pool_frames, n_atoms_pool, 3)

    sigma_F_pool_max  = sigma_F_pool.max(axis=(1, 2))
    sigma_F_pool_mean = np.linalg.norm(sigma_F_pool, axis=2).mean(axis=1)
    frame_max_force_pool = np.linalg.norm(mu_F_pool, axis=2).max(axis=1)
    E_atom_pool = mu_E_atom_pool  # alias

    print("\n[DEBUG] --- Force stats (POOL) ---")
    print(f"[DEBUG]   frame_max_force_pool mean={frame_max_force_pool.mean():.6f} min={frame_max_force_pool.min():.6f} max={frame_max_force_pool.max():.6f}")
    print(f"[DEBUG]   sigma_F_pool_max mean={sigma_F_pool_max.mean():.6f} min={sigma_F_pool_max.min():.6f} max={sigma_F_pool_max.max():.6f}")
    print(f"[DEBUG]   sigma_F_pool_mean mean={sigma_F_pool_mean.mean():.6f} min={sigma_F_pool_mean.min():.6f} max={sigma_F_pool_mean.max():.6f}")

    # ------------------------------------------------------------------
    # 1. RDF hard cutoff – geometric sanity
    # ------------------------------------------------------------------
    rdf_ok_mask = fast_filter_by_rdf_kdtree(pool_frames, rdf_thresholds)
    ok_idx = np.where(rdf_ok_mask)[0]
    n_ok = ok_idx.size
    print(f"\n[Pool-AL] RDF filter: {n_ok}/{len(pool_frames)} frames pass hard-cutoff geometry.")

    # ------------------------------------------------------------------
    # 2. STATS ON RDF-OK FRAMES -> ADAPTIVE caps from percentile_F_hi
    # ------------------------------------------------------------------
    if n_ok > 0:
        mu_E_ok_atom     = mu_E_atom_pool[ok_idx]
        sigE_ok_atom     = sigma_E_per_atom_pool[ok_idx]
        sigFmax_ok       = sigma_F_pool_max[ok_idx]
        sigFmean_ok      = sigma_F_pool_mean[ok_idx]
        Fmax_ok          = frame_max_force_pool[ok_idx]

        print("\n[DEBUG] --- OK-FRAMES (RDF-filtered) STATS ---")
        print(f"[DEBUG]   mu_E_atom_pool[OK]: mean={mu_E_ok_atom.mean():.6f} "
              f"min={mu_E_ok_atom.min():.6f} median={np.median(mu_E_ok_atom):.6f} max={mu_E_ok_atom.max():.6f}")
        print(f"[DEBUG]   sigma_F_pool_max[OK]: mean={sigFmax_ok.mean():.6f} "
              f"min={sigFmax_ok.min():.6f} median={np.median(sigFmax_ok):.6f} max={sigFmax_ok.max():.6f}")

        thr_sigma_E_hi_adapt_raw     = np.percentile(sigE_ok_atom,    percentile_F_hi)
        thr_sigma_F_hi_adapt_raw     = np.percentile(sigFmax_ok,      percentile_F_hi)
        thr_sigma_Fmean_hi_adapt_raw = np.percentile(sigFmean_ok,     percentile_F_hi)
        thr_Fmag_hi_adapt_raw        = np.percentile(Fmax_ok,         percentile_F_hi)

        pool_ok_lowest   = np.percentile(mu_E_ok_atom, 5)
        train_highest    = np.percentile(mu_E_atom_train, 95)
        energy_offset    = pool_ok_lowest - train_highest
        slack            = 0.05
        allowed_offset_adapt_raw = max(0.0, energy_offset) + slack

        print("\n[ADAPT raw from OK frames]")
        print(f"[ADAPT] thr_sigma_E_hi_adapt_raw     = {thr_sigma_E_hi_adapt_raw:.6f} eV/atom")
        print(f"[ADAPT] thr_sigma_F_hi_adapt_raw     = {thr_sigma_F_hi_adapt_raw:.6f} eV/Å")
        print(f"[ADAPT] thr_sigma_Fmean_hi_adapt_raw = {thr_sigma_Fmean_hi_adapt_raw:.6f} eV/Å")
        print(f"[ADAPT] thr_Fmag_hi_adapt_raw        = {thr_Fmag_hi_adapt_raw:.6f} eV/Å")
        print(f"[ADAPT] allowed_offset_adapt_raw     = {allowed_offset_adapt_raw:.6f} eV/atom")

        thr_sigma_E_hi_eff     = thr_sigma_E_hi_adapt_raw
        thr_sigma_F_hi_eff     = thr_sigma_F_hi_adapt_raw
        thr_sigma_Fmean_hi_eff = thr_sigma_Fmean_hi_adapt_raw
        thr_Fmag_hi_eff        = thr_Fmag_hi_adapt_raw
        allowed_offset_eff     = allowed_offset_adapt_raw

    else:
        print("[DEBUG] No frames passed RDF sanity filter — using legacy fallback caps.")
        thr_sigma_E_hi_eff     = thr_sigma_E_hi_legacy
        thr_sigma_F_hi_eff     = thr_sigma_F_hi_legacy
        thr_sigma_Fmean_hi_eff = thr_sigma_Fmean_hi_legacy
        thr_Fmag_hi_eff        = thr_Fmag_hi_legacy
        allowed_offset_eff     = margin_hi_per_atom_legacy

    # clamp to avoid upper < lower
    thr_sigma_E_hi_eff     = clamp_threshold("σ_E/atom",  thr_sigma_E_low,    thr_sigma_E_hi_eff)
    thr_sigma_F_hi_eff     = clamp_threshold("σF_max",    thr_sigma_F,        thr_sigma_F_hi_eff)
    thr_sigma_Fmean_hi_eff = clamp_threshold("σF_mean",   thr_sigma_Fmean,    thr_sigma_Fmean_hi_eff)
    thr_Fmag_hi_eff        = clamp_threshold("|F|_max",   thr_Fmag,           thr_Fmag_hi_eff)

    print("\n[CAPS] Effective caps in use (after clamp):")
    print(f"[CAPS]   thr_sigma_E_low        = {thr_sigma_E_low:.6f} eV/atom (lower)")
    print(f"[CAPS]   thr_sigma_E_hi_eff     = {thr_sigma_E_hi_eff:.6f} eV/atom (upper/adapt)")
    print(f"[CAPS]   thr_sigma_F            = {thr_sigma_F:.6f} eV/Å (lower)")
    print(f"[CAPS]   thr_sigma_F_hi_eff     = {thr_sigma_F_hi_eff:.6f} eV/Å (upper/adapt)")
    print(f"[CAPS]   thr_sigma_Fmean        = {thr_sigma_Fmean:.6f} eV/Å (lower)")
    print(f"[CAPS]   thr_sigma_Fmean_hi_eff = {thr_sigma_Fmean_hi_eff:.6f} eV/Å (upper/adapt)")
    print(f"[CAPS]   thr_Fmag               = {thr_Fmag:.6f} eV/Å (lower)")
    print(f"[CAPS]   thr_Fmag_hi_eff        = {thr_Fmag_hi_eff:.6f} eV/Å (upper/adapt)")
    print(f"[CAPS]   allowed_offset_eff     = {allowed_offset_eff:.6f} eV/atom (CI slack)")
    print(f"[CAPS]   gamma_thr              = {gamma_thr:.6f} (Mahalanobis novelty cutoff)")
    print(f"[CAPS]   hard triggers          = σE≥{hard_sigma_E_atom_min:.4f} eV/atom OR "
          f"σF_mean≥{hard_sigma_F_mean_min:.3f} eV/Å OR σF_max≥{hard_sigma_F_max_min:.3f} eV/Å; "
          f"|F|_max≤{train_Fmax_hard_cap:.3f} eV/Å")
    print(f"[CAPS]   percentiles (γ/F_low/F_hi) = {percentile_gamma:.1f}/{percentile_F_low:.1f}/{percentile_F_hi:.1f}")

    # ------------------------------------------------------------------
    # 3. Coverage / convergence heuristic
    # ------------------------------------------------------------------
    n_hi_E      = int((sigma_E_per_atom_pool  > thr_sigma_E_low).sum())
    n_hi_Fmax   = int((sigma_F_pool_max       > thr_sigma_F).sum())
    n_hi_Fmean  = int((sigma_F_pool_mean      > thr_sigma_Fmean).sum())
    n_hi_Fmag   = int((frame_max_force_pool   > thr_Fmag).sum())

    print("\n[AL] --- Coverage stats ---")
    print(f"[AL] Total pool frames: {n_pool_frames}")
    print(f"[AL] RDF-ok frames:     {n_ok}")
    print(f"[AL] γ_thr (pctl {percentile_gamma:.1f}%): {gamma_thr:.4f}")
    print(f"[AL] Frames above lower bounds:")
    print(f"     σE/atom  > {thr_sigma_E_low:.4f} : {n_hi_E}")
    print(f"     σF_max   > {thr_sigma_F:.4f} : {n_hi_Fmax}")
    print(f"     σF_mean  > {thr_sigma_Fmean:.4f} : {n_hi_Fmean}")
    print(f"     |F|_max  > {thr_Fmag:.4f} : {n_hi_Fmag}")

    if (n_hi_E + n_hi_Fmax + n_hi_Fmean + n_hi_Fmag) < 10:
        print("[AL] Convergence heuristic: fewer than 10 frames exceed any lower bound.")
        print("[AL] Nothing significant left to label.")
        # still write a minimal diagnostics later
        final_pool_indices = []
        sel_frames = []
        # we'll still write the diagnostic file including TRAIN frames
        # (see below, outside main loop)
        # but we skip the big window loop
        n_uncertain_total = 0
        n_gamma_gate_total = 0
    else:
        n_uncertain_total   = 0
        n_gamma_gate_total  = 0

    # ------------------------------------------------------------------
    # 4. Windowed selection with OOD filter and D-opt
    # ------------------------------------------------------------------
    cand_global = []
    records_tmp = []
    all_frame_records = {}
    ood_total   = 0

    for w0 in range(0, n_pool_frames, window_size):
        w1  = min(w0 + window_size, n_pool_frames)
        win = list(range(w0, w1))
        print(f"\n[Pool-AL] Window {w0}-{w1}: {len(win)} total frames")

        # (1) RDF pass
        win_good = [i for i in win if rdf_ok_mask[i]]
        print(f"[Pool-AL]   {len(win_good)} pass RDF hard cutoff")

        # (2) per-atom energy sanity
        win_E = [i for i in win_good if mu_E_atom_pool[i] < thr_E_hi_atom]
        print(f"[Pool-AL]   {len(win_E)} below E/atom cap ({thr_E_hi_atom:.3f} eV/atom)")

        # (3) per-atom energy CI overlap
        train_min = E_train_min_atom
        train_max = E_train_max_atom
        win_CI = [
            i for i in win_E
            if (
                (E_hi_pool_atom[i] >= (train_min - allowed_offset_eff)) and
                (E_lo_pool_atom[i] <= (train_max + allowed_offset_eff))
            )
        ]
        print(f"[Pool-AL]   {len(win_CI)} frames pass adaptive CI overlap "
              f"(train=[{train_min:.3f},{train_max:.3f}] ± {allowed_offset_eff:.3f})")

        # (4) enforce adaptive upper caps on σE/σF/σFmean
        win_sigmaE = [i for i in win_CI if sigma_E_per_atom_pool[i] < thr_sigma_E_hi_eff]
        win_sigmaF = [i for i in win_sigmaE if sigma_F_pool_max[i] < thr_sigma_F_hi_eff]
        win_sigmaFmean = [i for i in win_sigmaF if sigma_F_pool_mean[i] < thr_sigma_Fmean_hi_eff]
        win_Fcap = [i for i in win_sigmaFmean if frame_max_force_pool[i] < thr_Fmag_hi_eff]
        win_phys = win_Fcap
        print(f"[Pool-AL]   {len(win_phys)} pass σE/σF/σFmean/|F|_max adaptive caps")

        # (5) NEW: hard AL trigger (literature-like)
        high = []
        for i in win_phys:
            cond_E    = (sigma_E_per_atom_pool[i]  >= hard_sigma_E_atom_min)
            cond_Favg = (sigma_F_pool_mean[i]      >= hard_sigma_F_mean_min)
            cond_Fmax = (sigma_F_pool_max[i]       >= hard_sigma_F_max_min)
            cond_Fmag = (frame_max_force_pool[i]   <= train_Fmax_hard_cap)
            if cond_Fmag and (cond_E or cond_Favg or cond_Fmax):
                high.append(i)
        print(f"[Pool-AL]   {len(high)} pass HARD AL triggers "
              f"(σE≥{hard_sigma_E_atom_min}, σF_mean≥{hard_sigma_F_mean_min}, "
              f"σF_max≥{hard_sigma_F_max_min}, |F|≤train×{hard_Fmax_train_mult})")

        if not high:
            print("[Pool-AL]   No HARD-trigger frames in this window.")
            # still record diagnostics for all frames in this window
            win_all = np.array(win, dtype=int)
            sub_G_all = G_pool[win_all].astype(np.float64)
            quad_all   = np.einsum("id,dk,ik->i", sub_G_all, M_inv_global, sub_G_all)
            gamma_all  = np.sqrt(quad_all)
            diff_all   = sub_G_all - mu_Gtrain
            dM2_all    = np.einsum("id,dk,ik->i", diff_all, Cov_inv, diff_all)
            dM_all     = np.sqrt(dM2_all)
            dgain_train_all = np.log1p(quad_all)
            raw_score_all   = gamma_all * dgain_train_all
            keep_gamma_mask_all = (gamma_all > gamma_thr)
            win_good_set = set(win_good)
            win_phys_set = set(win_phys)
            high_set     = set(high)
            for j_local, pidx in enumerate(win_all):
                rec = {
                    "pool_idx": int(pidx),
                    "window":   f"{w0}-{w1}",
                    "rdf_ok":     bool(pidx in win_good_set),
                    "pass_caps":  bool(pidx in win_phys_set),
                    "force_inf":  bool(pidx in high_set),      # here: hard-triggered
                    "gamma_gate": bool(keep_gamma_mask_all[j_local]),
                    "above_lower": bool(pidx in high_set),     # keep legacy name
                    "gamma0":           float(gamma_all[j_local]),
                    "dM":               float(dM_all[j_local]),
                    "dgain_train":      float(dgain_train_all[j_local]),
                    "dgain_greedy":     float("nan"),
                    "raw_score_window": float(raw_score_all[j_local]),
                    "sigma_E_atom": float(sigma_E_per_atom_pool[pidx]),
                    "sigma_F_max":  float(sigma_F_pool_max[pidx]),
                    "sigma_F_mean": float(sigma_F_pool_mean[pidx]),
                    "Fmax":         float(frame_max_force_pool[pidx]),
                    "mu_E_atom":    float(mu_E_atom_pool[pidx]),
                    # training-relative ratios just for plots
                    "r_sigmaF_max":  float(sigma_F_pool_max[pidx]  / (avg_sigmaF_train_max  + 1e-12)),
                    "r_sigmaF_mean": float(sigma_F_pool_mean[pidx] / (avg_sigmaF_train_mean + 1e-12)),
                    "r_Fmag":        float(frame_max_force_pool[pidx] / (avg_Fmag_train + 1e-12)),
                    "selected": False,
                }
                all_frame_records[int(pidx)] = rec
            continue

        # ----- A. Metrics for ALL frames in this window (no filtering yet) -----
        win_all = np.array(win, dtype=int)
        sub_G_all = G_pool[win_all].astype(np.float64)

        quad_all   = np.einsum("id,dk,ik->i", sub_G_all, M_inv_global, sub_G_all)
        gamma_all  = np.sqrt(quad_all)

        diff_all   = sub_G_all - mu_Gtrain
        dM2_all    = np.einsum("id,dk,ik->i", diff_all, Cov_inv, diff_all)
        dM_all     = np.sqrt(dM2_all)

        dgain_train_all = np.log1p(quad_all)
        raw_score_all   = gamma_all * dgain_train_all

        print(f"[Pool-AL]   Mahalanobis γ₀ (ALL in window): mean={gamma_all.mean():.3f}, "
              f"min={gamma_all.min():.3f}, max={gamma_all.max():.3f}")
        print(f"[Pool-AL]   dM to-train (ALL in window):   mean={dM_all.mean():.3f}, "
              f"min={dM_all.min():.3f}, max={dM_all.max():.3f}")
        print(f"[Pool-AL]   Raw γ×Dgain_train stats: mean={raw_score_all.mean():.3f}, "
              f"min={raw_score_all.min():.3f}, max={raw_score_all.max():.3f}")

        keep_gamma_mask_all = (gamma_all > gamma_thr)
        ood_total += int(keep_gamma_mask_all.sum())

        win_good_set = set(win_good)
        win_phys_set = set(win_phys)
        high_set     = set(high)

        is_high_all = np.array([(idx in high_set) for idx in win_all], dtype=bool)

        cand_mask = is_high_all & keep_gamma_mask_all

        n_uncertain_total  += int(is_high_all.sum())
        n_gamma_gate_total += int(cand_mask.sum())

        cand_idx_local  = np.where(cand_mask)[0]

        dgain_greedy_all = np.full(win_all.shape[0], np.nan, dtype=float)

        selected_local   = []
        gains_full_greedy = None

        if cand_idx_local.size > 0:
            X_cand = sub_G_all[cand_idx_local]
            order, gains_full_greedy, gamma0_sub = d_optimal_full_order(
                X_cand=X_cand,
                X_train=G_train,
                reg=reg,
            )

            diff_expanded = X_cand[:, None, :] - X_cand[None, :, :]
            dist_mat = np.linalg.norm(diff_expanded, axis=2)

            remaining = list(range(X_cand.shape[0]))
            while (len(selected_local) < min_k) and remaining:
                best_r = None
                best_score = -np.inf
                for r in remaining:
                    score_inter = gains_full_greedy[r]
                    if not selected_local:
                        diversity_penalty = 1.0
                    else:
                        diversity_penalty = float(np.min(dist_mat[r, selected_local]))
                    score_candidate = score_inter * diversity_penalty
                    if score_candidate > best_score:
                        best_score = score_candidate
                        best_r = r
                selected_local.append(best_r)
                remaining.remove(best_r)

            for loc_r in range(len(order)):
                j_local = cand_idx_local[loc_r]
                dgain_greedy_all[j_local] = float(gains_full_greedy[loc_r])

        if len(selected_local) > 0:
            picks_abs = win_all[cand_idx_local[selected_local]].tolist()
        else:
            picks_abs = []

        print(f"[Pool-AL]   Selected {len(picks_abs)} frames in this window "
              f"(γ gate + D-opt gain + intra-batch diversity).")

        # ----- D. Record EVERY frame’s metrics into all_frame_records (for .txt) -----
        for j_local, pidx in enumerate(win_all):
            r_sigmaF_max  = float(sigma_F_pool_max[pidx]  / (avg_sigmaF_train_max  + 1e-12))
            r_sigmaF_mean = float(sigma_F_pool_mean[pidx] / (avg_sigmaF_train_mean + 1e-12))
            r_Fmag        = float(frame_max_force_pool[pidx] / (avg_Fmag_train + 1e-12))

            rec = {
                "pool_idx": int(pidx),
                "window":   f"{w0}-{w1}",
                "rdf_ok":     bool(pidx in win_good_set),
                "pass_caps":  bool(pidx in win_phys_set),
                "force_inf":  bool(pidx in high_set),                  # <- NOW: means "passed hard AL trigger"
                "gamma_gate": bool(keep_gamma_mask_all[j_local]),
                "above_lower": bool(pidx in high_set),                  # legacy name, same meaning
                "gamma0":           float(gamma_all[j_local]),
                "dM":               float(dM_all[j_local]),
                "dgain_train":      float(dgain_train_all[j_local]),
                "dgain_greedy":     float(dgain_greedy_all[j_local]) if not np.isnan(dgain_greedy_all[j_local]) else float("nan"),
                "raw_score_window": float(raw_score_all[j_local]),
                "sigma_E_atom": float(sigma_E_per_atom_pool[pidx]),
                "sigma_F_max":  float(sigma_F_pool_max[pidx]),
                "sigma_F_mean": float(sigma_F_pool_mean[pidx]),
                "Fmax":         float(frame_max_force_pool[pidx]),
                "mu_E_atom":    float(mu_E_atom_pool[pidx]),
                "r_sigmaF_max":  r_sigmaF_max,
                "r_sigmaF_mean": r_sigmaF_mean,
                "r_Fmag":        r_Fmag,
            }

            rec["selected"] = bool(pidx in picks_abs)

            all_frame_records[int(pidx)] = rec

        # ----- E. Keep compact record for final shortlist -----
        if len(selected_local) > 0:
            for pick_rel_idx, r_sel in enumerate(selected_local):
                j_local = int(cand_idx_local[r_sel])
                pidx    = int(win_all[j_local])

                gamma_val = float(gamma_all[j_local])
                dM_val    = float(dM_all[j_local])
                dgain_val = float(gains_full_greedy[r_sel])
                raw_val   = float(gamma_val * dgain_train_all[j_local])

                cand_global.append(pidx)
                records_tmp.append({
                    "pool_idx":     pidx,
                    "E_atom_pred":  float(mu_E_atom_pool[pidx]),
                    "sigma_E_atom": float(sigma_E_per_atom_pool[pidx]),
                    "E_lo_atom":    float(E_lo_pool_atom[pidx]),
                    "E_hi_atom":    float(E_hi_pool_atom[pidx]),
                    "sigma_F_max":  float(sigma_F_pool_max[pidx]),
                    "sigma_F_mean": float(sigma_F_pool_mean[pidx]),
                    "Fmax":         float(frame_max_force_pool[pidx]),
                    "gamma0":       gamma_val,
                    "dM":           dM_val,
                    "dgain":        dgain_val,
                    "raw_score":    raw_val,
                    "window":       f"{w0}-{w1}",
                })

    # ------------------------------------------------------------------
    # 5. Normalize debug score across ALL frames
    # ------------------------------------------------------------------
    if len(all_frame_records) > 0:
        rs = np.array([rec["raw_score_window"] for rec in all_frame_records.values()], dtype=float)
        rs_max = float(rs.max()) if rs.size else 1.0
        if rs_max == 0.0:
            rs_max = 1.0
        for rec in all_frame_records.values():
            rec["score_norm"] = rec["raw_score_window"] / rs_max
    else:
        print("[AL][WARN] No frames recorded in all_frame_records.")

    # ------------------------------------------------------------------
    # 6. Deduplicate global picks and score them (final shortlist we return)
    # ------------------------------------------------------------------
    unique_map = {}
    for rec in records_tmp:
        idxp = rec["pool_idx"]
        if (idxp not in unique_map) or (rec["raw_score"] > unique_map[idxp]["raw_score"]):
            unique_map[idxp] = rec

    unique_records = list(unique_map.values())
    picked_total = len(unique_records)

    eps = 1e-12
    if picked_total > 0:
        all_raw = np.array([r["raw_score"] for r in unique_records], dtype=float)
        raw_norm = all_raw / (all_raw.max() + eps)
        for i, r in enumerate(unique_records):
            r["score"] = float(raw_norm[i])
    else:
        for r in unique_records:
            r["score"] = 0.0

    unique_records.sort(key=lambda x: x["score"], reverse=True)

    if budget_max and len(unique_records) > budget_max:
        print(f"[AL] Budget cap active: picked_total={picked_total} > budget_max={budget_max}")
        unique_records = unique_records[:budget_max]
        picked_total = budget_max

    final_pool_indices = [r["pool_idx"] for r in unique_records]

    # ------------------------------------------------------------------
    # 7. Diagnostics file
    # ------------------------------------------------------------------
    diag_file = f"{base}_per_frame_diagnostics.txt"
    with open(diag_file, "w") as fh:
        fh.write("# Active Learning diagnostics (full pool)\n")
        fh.write("#\n")
        fh.write("# For every frame in every window:\n")
        fh.write("#   rdf_ok        : passed RDF geometric sanity\n")
        fh.write("#   pass_caps     : passed σE/σF/σFmean adaptive *upper* caps\n")
        fh.write("#   force_inf     : passed HARD AL trigger (σE or σF_mean or σF_max) AND within |F| cap\n")
        fh.write("#   gamma_gate    : gamma0 > gamma_thr (novel vs training span)\n")
        fh.write("#   gamma0        : leverage-like Mahalanobis novelty vs training span (x^T M^{-1} x)^0.5\n")
        fh.write("#   dM            : classical Mahalanobis to train mean (drift monitor only)\n")
        fh.write("#   dgain_train   : log(1 + x^T M^{-1} x), 1-step D-opt gain vs TRAIN ALONE\n")
        fh.write("#   dgain_greedy  : D-opt greedy gain used in window selection (only defined for γ-high ∧ force_inf)\n")
        fh.write("#   raw_γxD       : gamma0 * dgain_train (defined for ALL frames)\n")
        fh.write("#   score_norm    : raw_γxD normalized to [0,1] across ALL frames (debug ranking proxy)\n")
        fh.write("#   selected      : 1 if frame made the greedy min_k batch in its own window\n")
        fh.write("#   shortlist     : 1 if frame survived global dedupe+budget (will actually be suggested to add)\n")
        fh.write("#\n")
        fh.write("# ------------------- GLOBAL THRESHOLDS / CUTS -------------------\n")
        fh.write(f"# percentile_gamma      = {percentile_gamma:.1f}\n")
        fh.write(f"# gamma_thr             = {gamma_thr:.6f}   # γ gate cutoff\n")
        fh.write(f"# basin_percentile      = {basin_percentile:.1f}\n")
        fh.write(f"# dM_thr                = {dM_thr:.6f}      # 99th pct Mahalanobis-to-mean radius of TRAIN\n")
        fh.write("#\n")
        fh.write(f"# percentile_F_low      = {percentile_F_low:.1f}\n")
        fh.write(f"# percentile_F_hi       = {percentile_F_hi:.1f}\n")
        fh.write(f"# thr_sigma_E_low       = {thr_sigma_E_low:.6f}  eV/atom   # lower floor on σ(E/atom)\n")
        fh.write(f"# thr_sigma_E_hi_eff    = {thr_sigma_E_hi_eff:.6f}  eV/atom # adaptive upper cap on σ(E/atom)\n")
        fh.write(f"# thr_sigma_F           = {thr_sigma_F:.6f}    eV/Å       # lower floor on σF_max\n")
        fh.write(f"# thr_sigma_F_hi_eff    = {thr_sigma_F_hi_eff:.6f}    eV/Å   # adaptive upper cap on σF_max\n")
        fh.write(f"# thr_sigma_Fmean       = {thr_sigma_Fmean:.6f}  eV/Å     # lower floor on σF_mean\n")
        fh.write(f"# thr_sigma_Fmean_hi_eff= {thr_sigma_Fmean_hi_eff:.6f}  eV/Å # adaptive upper cap on σF_mean\n")
        fh.write(f"# thr_Fmag              = {thr_Fmag:.6f}    eV/Å         # lower floor on |F|_max\n")
        fh.write(f"# thr_Fmag_hi_eff       = {thr_Fmag_hi_eff:.6f}    eV/Å   # adaptive upper cap on |F|_max\n")
        fh.write(f"# allowed_offset_eff    = {allowed_offset_eff:.6f}  eV/atom # CI slack for energy overlap\n")
        fh.write("#\n")
        fh.write(f"# HARD AL TRIGGERS (from config):\n")
        fh.write(f"#   hard_sigma_E_atom_min = {hard_sigma_E_atom_min:.6f}  eV/atom\n")
        fh.write(f"#   hard_sigma_F_mean_min = {hard_sigma_F_mean_min:.6f}  eV/Å\n")
        fh.write(f"#   hard_sigma_F_max_min  = {hard_sigma_F_max_min:.6f}  eV/Å\n")
        fh.write(f"#   train_Fmax_hard_cap   = {train_Fmax_hard_cap:.6f}  eV/Å  # = {hard_Fmax_train_mult:.3f} × max |F| in TRAIN\n")
        fh.write("#\n")
        fh.write(f"# min_k                 = {min_k}\n")
        fh.write(f"# window_size           = {window_size}\n")
        fh.write(f"# budget_max            = {budget_max}\n")
        fh.write("#\n")

        shortlist_set = set(final_pool_indices)

        # header row for pool table (single space between columns)
        fh.write(
            f"{'idx':<8} "
            f"{'window':>12} "
            f"{'rdf_ok':>8} "
            f"{'caps_ok':>8} "
            f"{'force_inf':>10} "
            f"{'γ_gate':>8} "
            f"{'gamma0':>12} "
            f"{'dM':>12} "
            f"{'Dgain_train':>14} "
            f"{'Dgain_greedy':>15} "
            f"{'raw_γxD':>12} "
            f"{'score_norm':>12} "
            f"{'σE_atom':>12} "
            f"{'σF_max':>12} "
            f"{'σF_mean':>12} "
            f"{'Fmax':>12} "
            f"{'μE_atom':>12} "
            f"{'r_σFmax':>10} "
            f"{'r_σFmean':>10} "
            f"{'r_|F|max':>10} "
            f"{'selected':>10} "
            f"{'shortlist':>11} "
            "\n"
        )
        fh.write("-" * 260 + "\n")  # increase line to cover extra spaces
        
        for pidx in sorted(all_frame_records.keys()):
            R = all_frame_records[pidx]
            fh.write(
                f"{pidx:<8d} "
                f"{R['window']:>12} "
                f"{int(R['rdf_ok']):>8d} "
                f"{int(R['pass_caps']):>8d} "
                f"{int(R['force_inf']):>10d} "
                f"{int(R['gamma_gate']):>8d} "
                f"{R['gamma0']:>12.6f} "
                f"{R['dM']:>12.6f} "
                f"{R['dgain_train']:>14.6f} "
                f"{(R['dgain_greedy'] if not np.isnan(R['dgain_greedy']) else float('nan')):>15.6f} "
                f"{R['raw_score_window']:>12.6f} "
                f"{R.get('score_norm', float('nan')):>12.6f} "
                f"{R['sigma_E_atom']:>20.12f} "
                f"{R['sigma_F_max']:>12.6f} "
                f"{R['sigma_F_mean']:>12.6f} "
                f"{R['Fmax']:>12.6f} "
                f"{R['mu_E_atom']:>12.6f} "
                f"{R['r_sigmaF_max']:>10.3f} "
                f"{R['r_sigmaF_mean']:>10.3f} "
                f"{R['r_Fmag']:>10.3f} "
                f"{int(R.get('selected', False)):>10d} "
                f"{int(pidx in shortlist_set):>11d} "
                "\n"
            )


        # ------------------------------------------------------------------
        # EXTRA: TRAIN-DATASET UNCERTAINTY DUMP (frame by frame)
        # ------------------------------------------------------------------
        fh.write("\n# ---------------------------------------------------------------\n")
        fh.write("# TRAIN DATASET UNCERTAINTIES (per frame)\n")
        fh.write("# Columns: train_idx  sigma_E_atom  sigma_F_max  sigma_F_mean  Fmax\n")
        fh.write("# Units  : eV/atom    eV/Å          eV/Å         eV/Å\n")
        fh.write("# ---------------------------------------------------------------\n")
        for t in range(n_train_frames):
            fh.write(
                f"{t:<8d}"
                f"{sigma_E_per_atom_train[t]:>20.12f}"
                f"{sigma_F_train_max[t]:>14.6f}"
                f"{sigma_F_train_mean[t]:>14.6f}"
                f"{frame_max_force_train[t]:>14.6f}"
                "\n"
            )

    # ------------------------------------------------------------------
    # 8. Summary / stop check
    # ------------------------------------------------------------------
    frac_geom_ok = float(rdf_ok_mask.sum()) / float(len(rdf_ok_mask)) if len(rdf_ok_mask) > 0 else 0.0
    picked_after_budget = len(final_pool_indices)

    print("\n[AL] Summary for stop check:")
    print(f"    geom_ok fraction                         : {frac_geom_ok:.3f}")
    print(f"    frames passing HARD triggers             : {n_uncertain_total}")
    print(f"    ... of which also pass γ gate            : {n_gamma_gate_total}")
    print(f"    shortlisted after greedy+diversity       : {picked_total}")
    print(f"    kept after global budget cap ({budget_max}) : {picked_after_budget}")

    converged_geometry   = (frac_geom_ok >= 0.9)
    converged_uncert_gam = (n_gamma_gate_total == 0)
    converged_budget     = (picked_after_budget == 0)

    if converged_geometry and converged_uncert_gam and converged_budget:
        print("[AL] >>> TRUE CONVERGENCE: model sufficiently covers this regime.")
    else:
        print("[AL] Active learning still in progress (not converged).")

    sel_frames = [pool_frames[i] for i in final_pool_indices]
    return sel_frames, final_pool_indices


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

