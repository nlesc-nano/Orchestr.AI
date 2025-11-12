"""
evaluate.py

This module orchestrates the evaluation process for ML force-field models.
It loads evaluation (and optionally training) data, sets up parameters,
loads a base model, computes predictions and statistics (via MLFFStats),
trains and evaluates UQ models (error estimation, ensemble, GMM-based, etc.),
and optionally triggers active learning procedures. All plotting is handled
via separate functions that work with saved NPZ files.
"""

import numpy as np
import os
import shutil
import torch
import time
import glob 
from datetime import datetime
import traceback
import contextlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.linalg

from ase.io import read, write  # Needed for reading training/eval data

# === Local Module Imports ===
from mlff_qd.postprocessing.parsing import parse_extxyz, save_stacked_xyz
from mlff_qd.postprocessing.calculator import setup_neighbor_list, assign_charges, evaluate_model
from mlff_qd.postprocessing.stats import MLFFStats
from mlff_qd.postprocessing.features import compute_features
from mlff_qd.postprocessing.uq_models import train_uq_models, predict_uncertainties, fit_gmm_and_compute_uncertainty
from mlff_qd.postprocessing.uq_metrics_calculator import VarianceScalingCalibrator, run_uq_metrics, calculate_uq_metrics
from mlff_qd.postprocessing.mlff_plotting import plot_mlff_stats
from mlff_qd.postprocessing.plotting import (
    generate_uq_plots, 
    generate_al_influence_plots, generate_al_traditional_plots
)
from mlff_qd.postprocessing.active_learning import adaptive_learning, adaptive_learning_mig_calibrated, adaptive_learning_mig_pool,calibrate_alpha_reg_gcv, predict_sigma_from_L, adaptive_learning_mig_pool_windowed, adaptive_learning_ensemble_calibrated 

from mlff_qd.postprocessing.active_learning import (
    compute_rdf_thresholds_from_reference,
    fast_filter_by_rdf_kdtree,
    collect_pair_distances,
    make_rdf_hist,
    plot_rdf_comparison,
    debug_plot_rdfs
)

def setup_evaluation(config):
    """
    Sets up parameters and loads initial data for evaluation.
    
    It loads the evaluation input file (extended XYZ), parses true energies and forces,
    and builds the list of frames. It also reads evaluation configuration parameters.
    
    Args:
        config (dict): Configuration dictionary.
    
    Returns:
        tuple: Contains device, initial true energies, initial true forces, evaluation frames,
               batch size, uncertainty method(s), n_mc, training_data_path, evaluation log file,
               unique MC log filename, and evaluation output file base.
    """
    print("--- Setting up Evaluation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eval_config = config.get("eval", {})
    assert eval_config, "Missing 'eval' in config."
    eval_input_xyz = eval_config.get("eval_input_xyz")
    assert eval_input_xyz and os.path.exists(eval_input_xyz), f"File not found: {eval_input_xyz}"

    print(f"Loading initial evaluation data from: {eval_input_xyz}")
    initial_true_energies, initial_true_forces, initial_eval_positions = parse_extxyz(eval_input_xyz, "eval")
    initial_frames = read(eval_input_xyz, index=":", format="extxyz")
    assert initial_frames, f"No frames read from {eval_input_xyz}"
    n_frames = len(initial_frames)
    print(f"Loaded {n_frames} initial evaluation frames.")

    batch_size = eval_config.get("batch_size", 32)
    uncertainty_methods = eval_config.get("uncertainty", ["none"])
    if not isinstance(uncertainty_methods, list):
        uncertainty_methods = [uncertainty_methods]
    n_mc = eval_config.get("n_mc", 10)
    training_data_path = eval_config.get("training_data")
    eval_log_file = eval_config.get("eval_log_file", "eval_log.txt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_mc_log_file = f"mc_energies_{timestamp}.txt" if "dropout" in uncertainty_methods else None
    eval_output_file_base = eval_config.get("eval_output_file", "evaluation_results")

    os.makedirs("diagnostics", exist_ok=True)
    os.makedirs("uq_plots", exist_ok=True)
    try:
        open(eval_log_file, "w").close()  # Clear log file
    except IOError:
        print(f"Warn: Could not clear {eval_log_file}")

    print("Evaluation setup complete.")
    return (device, initial_true_energies, initial_true_forces, initial_eval_positions, initial_frames,
            batch_size, uncertainty_methods, n_mc, training_data_path,
            eval_log_file, unique_mc_log_file, eval_output_file_base)
            
# --- Safe model loader to avoid map_location bug and standardize dtype/device ---
def _safe_load_model(model_path: str, device: torch.device, force_dtype: torch.dtype | None = torch.float32):
    """
    Load a torch model without using map_location (avoids PyTorch 2.4.x thread-local bug),
    then move it to the desired device and dtype.
    """
    try:
        mdl = torch.load(model_path, weights_only=False)  # don't pass map_location
    except AttributeError as e:
        # Rare fallback if the bug triggers in another codepath
        print(f"[safe_load] AttributeError on first load attempt: {e}. Retrying simplest path…")
        mdl = torch.load(model_path)

    # Move to device (and dtype if requested)
    if force_dtype is None:
        mdl = mdl.to(device=device)
    else:
        mdl = mdl.to(device=device, dtype=force_dtype)

    mdl.eval()
    return mdl


def generate_ensemble_sizes(max_size, min_size=2, step=1):
    """
    Generates a sorted list of ensemble sizes to test.
    
    Args:
        max_size (int): Maximum ensemble size.
        min_size (int): Minimum ensemble size (default: 2).
        step (int): Step between sizes (default: 1).
    
    Returns:
        list: Sorted list of unique ensemble sizes.
    """
    if max_size < min_size:
        return []
    sizes = list(range(min_size, max_size + 1, step))
    if max_size not in sizes and max_size >= min_size:
        sizes.append(max_size)
    return sorted(list(set(sizes)))


def analyze_ensemble_convergence(all_metrics_convergence, ensemble_sizes_tested):
    """
    Analyzes and logs metric convergence as a function of ensemble size.
    
    Args:
        all_metrics_convergence (dict): Convergence metrics keyed by ensemble size.
        ensemble_sizes_tested (list): List of ensemble sizes tested.
    
    Returns:
        None
    """
    print("\n--- Analyzing Ensemble Convergence ---")
    if not all_metrics_convergence:
        print("No metrics for convergence analysis.")
        return

    log_filename = "ensemble_convergence_metrics.log"
    convergence_threshold = 0.01
    try:
        with open(log_filename, "w") as log:
            log.write(f"Ensemble Convergence Analysis (Threshold={convergence_threshold:.1%})\n")
            log.write(f"Tested Sizes: {ensemble_sizes_tested}\n")
            largest_size = ensemble_sizes_tested[-1]
            ref_metrics = all_metrics_convergence.get(largest_size, {})
            metric_keys = set()
            if ref_metrics.get("train") and ref_metrics["train"]["metrics"]:
                metric_keys.update(ref_metrics["train"]["metrics"].keys())
            if ref_metrics.get("eval") and ref_metrics["eval"]["metrics"]:
                metric_keys.update(ref_metrics["eval"]["metrics"].keys())
            if not metric_keys:
                log.write("No metric keys found.")
                return
            sorted_metric_keys = sorted(list(metric_keys))
            for dataset in ["train", "eval"]:
                log.write(f"\n--- {dataset.capitalize()} Set Convergence ---\n")
                header = "Size\t" + "\t".join(sorted_metric_keys) + "\n"
                log.write(header)
                data_for_dataset = []
                for size in ensemble_sizes_tested:
                    result_dict = all_metrics_convergence.get(size, {}).get(dataset, {})
                    metrics = result_dict.get("metrics") if isinstance(result_dict, dict) else None
                    data_for_dataset.append((size, metrics))
                    vals = [metrics.get(k, np.nan) for k in sorted_metric_keys] if metrics else [np.nan] * len(sorted_metric_keys)
                    row = f"{size}\t" + "\t".join([f"{v:.6f}" if isinstance(v, (float, np.number)) and not np.isnan(v) else "NaN" for v in vals]) + "\n"
                    log.write(row)
                log.write("\nConvergence Check:\n")
                for key in sorted_metric_keys:
                    values = []
                    valid_indices = []
                    for i, (size, metrics) in enumerate(data_for_dataset):
                        val = metrics.get(key, np.nan) if metrics else np.nan
                        values.append(val)
                        if not np.isnan(val):
                            valid_indices.append(i)
                    if len(valid_indices) < 2:
                        log.write(f"  {key}: N/A (Insufficient data)\n")
                        continue
                    converged = False
                    for k in range(1, len(valid_indices)):
                        prev_val = values[valid_indices[k-1]]
                        curr_val = values[valid_indices[k]]
                        if abs(prev_val) > 1e-9:
                            rel_change = abs(curr_val - prev_val) / abs(prev_val)
                            if rel_change < convergence_threshold:
                                conv_size = ensemble_sizes_tested[valid_indices[k]]
                                log.write(f"  {key}: Converged at size {conv_size} (RelChange:{rel_change:.4f})\n")
                                converged = True
                                break
                        elif abs(curr_val) < 1e-9:
                            conv_size = ensemble_sizes_tested[valid_indices[k]]
                            log.write(f"  {key}: Converged at size {conv_size} (Near zero)\n")
                            converged = True
                            break
                    if not converged:
                        log.write(f"  {key}: Not converged.\n")
    except Exception as e:
        print(f"Error analyzing convergence: {e}")
        traceback.print_exc()
    print(f"Ensemble convergence analysis logged to {log_filename}")

def purge_training_structures(true_energies, eval_positions, train_energies, train_positions, num_atoms_to_check=3):
    """
    Returns a Boolean mask for eval structures that are NOT redundant with the training set.
    A structure is considered redundant if its energy (rounded to 5 decimals) is within tolerance of a training structure's energy,
    and the coordinates of its first num_atoms_to_check atoms are close (within pos_tol).
    """
    eval_mask = []
    energy_tol, pos_tol = 0.0001, 0.0001
    for i, (e_eval, p_eval) in enumerate(zip(true_energies, eval_positions)):
        is_redundant = False
        e_eval_rounded = round(e_eval, 5)
        for j, (e_train, p_train) in enumerate(zip(train_energies, train_positions)):
            e_train_rounded = round(e_train, 5)
            if abs(e_eval_rounded - e_train_rounded) < energy_tol:
                if p_eval.shape[0] >= num_atoms_to_check and p_train.shape[0] >= num_atoms_to_check:
                    if np.allclose(p_eval[:num_atoms_to_check], p_train[:num_atoms_to_check], atol=pos_tol):
                        print(f"Redundant structure found: Eval frame {i} is redundant with Training frame {j}.")
                        is_redundant = True
                        break
        eval_mask.append(not is_redundant)
    return eval_mask

def evaluate_and_cache_ensemble(frames,
                                model_folder,
                                device,
                                batch_size,
                                log_path,
                                config,
                                neighbor_list,
                                npz_path="ensemble.npz",
                                model_glob=("*",)):
    """Run every Torch model in *model_folder* on *frames* (via ``evaluate_model``)
    and cache the stacked predictions/latents to *npz_path*  **or** load them
    if the file already exists.

    Parameters
    ----------
    frames : list[ase.Atoms]
        Structures to evaluate.
    model_folder : str
        Directory containing the ensemble *.pt / *.pth files.
    device : torch.device or str
    batch_size : int
    log_path : str | None
        Passed straight through to ``evaluate_model``.
    config : dict
        Same config that ``run_eval`` is using (passed through).
    neighbor_list : Any
        Pre‑built neighbour list for the model (passed through).
    npz_path : str, default "ensemble.npz"
        Where to cache / load the stacked results.
    model_glob : tuple[str,...]
        File patterns inside *model_folder* that count as models.

    Returns
    -------
    tuple:
        - energy_preds : np.ndarray, shape (n_models, n_frames)
        - force_preds  : np.ndarray, shape (n_models, n_atoms_total, 3)
        - latent_preds : np.ndarray, shape (n_models, n_frames, latent_dim)
        - latent_atom_preds : np.ndarray, shape (n_models, n_atoms_total, latent_atom_dim)
        - model_list   : list of loaded models, or ``None`` if loaded from cache
    """
    # ------------------------------------------------------------------
    # 1. Fast‑path: load from cache
    # ------------------------------------------------------------------
    if os.path.isfile(npz_path):
        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                return (
                    npz["ensemble_energy_preds"],   # (n_models, n_frames)
                    npz["ensemble_force_preds"],    # (n_models, total_atoms, 3)
                    npz["ensemble_latent_preds"],   # (n_models, n_frames, latent_dim)
                    npz["mean_latent_atom"],  # (total_atoms, latent_atom_dim)
                    npz["std_latent_atom"],         # (total_atoms, latent_atom_dim)
                    None
                )
        except Exception as e:
            print(f"[Ensemble‑cache] Failed to load '{npz_path}': {e}. Recomputing…")

    # ------------------------------------------------------------------
    # 2. Gather model files
    # ------------------------------------------------------------------
    model_files = []
    for pat in model_glob:
        for f in glob.glob(os.path.join(model_folder, pat)):
            if os.path.isfile(f):
                model_files.append(f)
    if not model_files:
        for entry in os.listdir(model_folder):
            fp = os.path.join(model_folder, entry)
            if os.path.isfile(fp):
                model_files.append(fp)
    model_files = sorted(model_files)
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_folder}")

    n_frames = len(frames)
    total_atoms = sum(len(fr) for fr in frames)
    print(f"[Ensemble] evaluating {len(model_files)} models on {n_frames} frames…")

    # ------------------------------------------------------------------
    # 3. Run each model
    # ------------------------------------------------------------------
    energy_list        = []  # (n_frames,)
    force_list         = []  # (n_atoms_total,3)
    latent_list        = []  # (n_frames, latent_dim)
    latent_atom_list   = []  # (n_atoms_total, latent_atom_dim)
    loaded_models      = []

    dummy_E = [0.0] * n_frames
    dummy_F = [None]  * n_frames

    for idx, mfile in enumerate(model_files, 1):
        try:
            # mdl = torch.load(mfile, map_location=device).to(torch.float32)
            # mdl.eval()
            # loaded_models.append(mdl)
            
            mdl = _safe_load_model(mfile, device=device, force_dtype=torch.float32)
            loaded_models.append(mdl)   

            # evaluate_model now returns latent_per_atom
            pred_E, pred_F_per_frame, latent_per_frame, latent_per_atom = evaluate_model(
                frames,
                dummy_E,
                dummy_F,
                mdl,
                device,
                batch_size,
#                "ensemble",
#                1,
                log_path,
#                None,
                config,
                neighbor_list,
            )

            # ---- energy ------------------------------------------
            pred_E = np.asarray(pred_E, dtype=np.float64)
            if pred_E.shape[0] != n_frames:
                raise ValueError("Energy array length mismatch.")
            energy_list.append(pred_E)

            # ---- forces ------------------------------------------
            flat_F = np.concatenate(pred_F_per_frame, axis=0)
            if flat_F.shape != (total_atoms, 3):
                raise ValueError("Flattened force array has wrong shape.")
            force_list.append(flat_F)

            # ---- latent per frame -------------------------------
            lat_arr = np.stack(latent_per_frame)
            if lat_arr.shape[0] != n_frames:
                raise ValueError("Latent array length mismatch.")
            latent_list.append(lat_arr)

            # ---- latent per atom --------------------------------
            flat_La = np.concatenate(latent_per_atom, axis=0)
            if flat_La.shape[0] != total_atoms:
                raise ValueError("Flattened atom latent array has wrong shape.")
            latent_atom_list.append(flat_La)

            print(f"  [{idx:>3}/{len(model_files)}] {os.path.basename(mfile)} done")

        except Exception as e:
            print(f"  [Warning] model '{mfile}' failed: {e}")
            traceback.print_exc()

    if not energy_list:
        raise RuntimeError("All ensemble evaluations failed.")

    # ------------------------------------------------------------------
    # 4. Stack & save (cast to float32 to shrink file size)
    # ------------------------------------------------------------------
    arr_E      = np.stack(energy_list).astype(np.float64)          # (n_models, n_frames)
    arr_F      = np.stack(force_list).astype(np.float32)           # (n_models, total_atoms, 3)
    arr_L      = np.stack(latent_list).astype(np.float32)          # (n_models, n_frames, latent_dim)
    arr_L_atom = np.stack(latent_atom_list).astype(np.float32)     # (n_models, total_atoms, latent_atom_dim)
   
    mean_L_atom = arr_L_atom.mean(axis=0)                          # (total_atoms, latent_atom_dim)
    std_L_atom  = arr_L_atom.std(axis=0, ddof=1)                   # (total_atoms, latent_atom_dim)

    try:
        np.savez_compressed(
            npz_path,
            ensemble_energy_preds      = arr_E,
            ensemble_force_preds       = arr_F,
            ensemble_latent_preds      = arr_L,
            mean_latent_atom     = mean_L_atom,
            std_latent_atom      = std_L_atom
        )
        print(f"[Ensemble-cache] saved → {npz_path}")
    except Exception as e:
        print(f"[Warning] Could not save ensemble cache '{npz_path}': {e}")
    
    return arr_E, arr_F, arr_L, mean_L_atom, std_L_atom, loaded_models
    

def run_eval(config):
    """
    Full evaluation and active-learning driver.
    Supports base-model evaluation, UQ methods (none, ensemble, error_estimate, GMM),
    active learning on validation set if no unlabeled pool,
    and pool-based active learning if unlabeled pool is provided.
    """
    if config is None:
        print("Error: Config not loaded.")
        return

    # 1) Setup
    try:
        (device,
         init_E,
         init_F,
         init_pos,
         init_frames,
         batch_size,
         uq_methods,
         n_mc,
         train_xyz_path,
         eval_log,
         uniq_mc_log,
         out_base) = setup_evaluation(config)
    except Exception as e:
        print(f"Fatal error in setup: {e}")
        traceback.print_exc()
        return

    eval_cfg           = config.get("eval", {})
    error_estimate     = eval_cfg.get("error_estimate", False)
    al_flag            = eval_cfg.get("active_learning", None)
    ensemble_folder    = eval_cfg.get("ensemble_folder", None)
    pool_xyz_path      = eval_cfg.get("unlabeled_pool_path", None)
    do_plot            = eval_cfg.get("plot", False)
    budget_max         = eval_cfg.get("budget_max", 50)
    percentile_gamma   = eval_cfg.get("percentile_gamma", 90.0) 
    percentile_F_low   = eval_cfg.get("percentile_F_low", 95.0) 
    percentile_F_hi    = eval_cfg.get("percentile_F_hi", 99.0) 
    thr_sE_atom        = eval_cfg.get("thr_sE_atom", 0.002)  # eV/atom  (2 meV/atom)
    thr_sF_mean        = eval_cfg.get("thr_sF_mean", 0.10)   # eV/Å
    thr_sF_max         = eval_cfg.get("thr_sF_max", 0.20)    # eV/Å
    thr_Fmax_mult      = eval_cfg.get("thr_Fmax_mult", 1.5)  # × max |F| in TRAIN
    neighbour_list     = setup_neighbor_list(config)

    # disable validation AL if pool is present
    al_val_flag = None if pool_xyz_path else al_flag

    # 2) Load training and validation data
    train_frames = []
    train_E = []
    train_F = []
    train_pos = []
    if train_xyz_path and os.path.exists(train_xyz_path):
        print(f"Loading training data from {train_xyz_path}")
        train_E, train_F, train_pos = parse_extxyz(train_xyz_path, "training_data")
        train_frames = read(train_xyz_path, index=":", format="extxyz")
        print(f"Loaded {len(train_frames)} training frames.")

    val_frames = init_frames
    val_E      = init_E
    val_F      = init_F
    val_pos    = init_pos

    # purge overlaps
    if train_frames:
        try:
            keep_mask = purge_training_structures(val_E, val_pos, train_E, train_pos)
            val_frames = [f for f,k in zip(val_frames, keep_mask) if k]
            val_E      = [e for e,k in zip(val_E, keep_mask) if k]
            val_F      = [f for f,k in zip(val_F, keep_mask) if k]
            print(f"Validation frames after purge: {len(val_frames)}")
        except Exception as e:
            print(f"Purge error: {e}. Continuing with full validation set.")

    all_frames      = train_frames + val_frames
    all_true_E      = np.array(train_E + val_E)
    all_true_F      = train_F + val_F
    all_true_F_list = train_F + val_F

    n_train = len(train_frames)
    n_val   = len(val_frames)
    train_mask = np.array([True]*n_train + [False]*n_val, dtype=bool)
    val_mask   = np.array([False]*n_train + [True]*n_val, dtype=bool)
    print(f"Total frames: {len(all_frames)} (train={n_train}, val={n_val})")

    # --- convenience indices available downstream (avoid UnboundLocal issues) ---
    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]

    # --- Ensure forces_train is a NumPy array with shape (n_frames, n_atoms, 3) ---
    # all_true_F_list is a list of (n_atoms,3) arrays for every frame (train+val).
    # We'll try a safe stack but fail clearly if frames have different sizes.
    forces_train_arr = None
    try:
        forces_train_arr = np.asarray(all_true_F_list, dtype=float)
        if forces_train_arr.ndim != 3 or forces_train_arr.shape[2] != 3:
            # it's possible np.asarray returned a 1D array of objects -> fall back to stack
            raise ValueError("forces_train_arr not a (n_frames,n_atoms,3) array")
    except Exception:
        # defensive check: verify consistent atom counts
        atom_counts = [f.shape[0] for f in all_true_F_list]
        if len(set(atom_counts)) != 1:
            raise RuntimeError(
                f"Inconsistent atom counts across frames: {sorted(set(atom_counts))}. "
                "adaptive_learning_mig_pool_windowed requires fixed-size frames or needs refactor."
            )
        forces_train_arr = np.stack(all_true_F_list, axis=0).astype(float)

    # 3) Load base model if needed
    base_model = None
    need_base = ("none" in uq_methods) or error_estimate or ("GMM" in uq_methods)
    if need_base:
        base_path = config.get("model_path", "")
        if not base_path or not os.path.exists(base_path):
            print("Base model not found: disabling dependent UQ methods.")
            need_base      = False
            error_estimate = False
            uq_methods     = [m for m in uq_methods if m not in ("none","GMM")]
        else:
            # base_model = torch.load(base_path, map_location=device).to(torch.float32)
            # base_model.eval()
            # print(f"Loaded base model from {base_path}")
            
            base_model = _safe_load_model(base_path, device=device, force_dtype=torch.float32)
            print(f"Loaded base model from {base_path}")

    # 4) Base evaluation and feature computation
    stats_base = None
    features_all = None
    min_dists_all = None
    all_latent = None
    all_peratom_latent = None
    if need_base and base_model:
        pred_E, pred_F, all_latent, all_peratom_latent = evaluate_model(
            all_frames,
            list(all_true_E),
            all_true_F,
            base_model,
            device,
            batch_size,
#            tag="none",
#            n_mc=1,
            log_path=eval_log,
#            unique_log=None,
            config=config,
            neighbor_list=neighbour_list
        )
        if isinstance(pred_F, np.ndarray) and pred_F.ndim == 3:
            flat = pred_F
            pf_list = []
            idx = 0
            for fr in all_frames:
                n = len(fr)
                pf_list.append(flat[idx:idx+n])
                idx += n
            pred_F = pf_list
        stats_base = MLFFStats(all_true_E, pred_E, all_true_F, pred_F, train_mask, val_mask)
        if error_estimate or "none" in uq_methods:
            features_all, min_dists_all, _, pca, scaler = compute_features(
                all_frames,
                config,
                train_xyz_path,
                train_mask,
                val_mask
            )

    # 5) UQ: error_estimate
    if error_estimate:
        # existing error_estimate code
        pass

    # 6) UQ: "none"
    if "none" in uq_methods:
        print("\n--- Evaluating Base Model Performance ('none' UQ) ---")
        if stats_base is None:
            print("Skipping 'none': Base stats not available.")
        elif features_all is None or min_distances_all is None:
            print("Warning: Features/distances missing for 'none' plots.")
        else:
            print("Plotting base model statistics...")
            plot_mlff_stats(stats_base, min_dists_all, "validation_results" + "_base", True, train_mask, val_mask)

    # 7) Ensemble UQ on labeled data
    # --------------------------------------------------
    # evaluate.py  —  Section 7: Ensemble UQ + Active‑Learning (FULL)
    # --------------------------------------------------
    
    # 7) Ensemble UQ on labeled data
    ensemble_models = None
    mean_L_al       = None
    alpha_sq = L_chol = None
    
    if "ensemble" in uq_methods and ensemble_folder and os.path.isdir(ensemble_folder):
    
        ens_E, ens_F, ens_L_frame, mean_L_atom, std_L_atom, ensemble_models = evaluate_and_cache_ensemble(
            all_frames, ensemble_folder, device, batch_size, eval_log,
            config, neighbour_list, npz_path="ensemble.npz")

        available = ens_E.shape[0]
        # honor user-configured ensemble_size (fallback to available models)
        n_models = min(eval_cfg.get("ensemble_size", available), available)
        print(f"n_models = {n_models}") 
        # consider only the first `n_models` ensembles as per config
        ens_E_sel       = ens_E[:n_models]           # shape (n_models, n_frames)
        ens_F_sel       = ens_F[:n_models]           # shape depends on evaluate_and_cache_ensemble
        ens_L_frame_sel = ens_L_frame[:n_models]     # (n_models, n_frames, d_lat)

        # Precompute ensemble summary stats ONCE (frame-wise and component-wise)
        mu_E_frame  = np.mean(ens_E_sel, axis=0)            # (n_frames,)
        std_E_frame = np.std(ens_E_sel, axis=0, ddof=0)
        sigma_E_raw = np.std(ens_E_sel, axis=0, ddof=1)
        min_E_frame = np.min(ens_E_sel, axis=0)
        max_E_frame = np.max(ens_E_sel, axis=0)

        # Forces: ensemble mean and std across models (flattening semantics preserved)
        mu_F_flat = np.mean(ens_F_sel, axis=0)
        sigma_F_flat = np.std(ens_F_sel, axis=0, ddof=1)
        mu_F_comp = mu_F_flat
        std_F_comp = sigma_F_flat.flatten()
        min_F_comp = np.min(ens_F_sel, axis=0).flatten()
        max_F_comp = np.max(ens_F_sel, axis=0).flatten()

        # Latents averaged across ensemble (frame-level)
        mean_L_frame = np.mean(ens_L_frame_sel, axis=0)     # (n_frames, d_lat)

        # build MLFFStats ----------------------------------------------------
        mf_list, idx = [], 0
        for fr in all_frames:
            n = len(fr)
            mf_list.append(mu_F_flat[idx:idx+n])
            idx += n
        stats_ens = MLFFStats(all_true_E, mu_E_frame, all_true_F_list, mf_list, train_mask, val_mask)
       
        # Try to create stacked force array only when possible (we assume fixed-size frames)
        all_true_F_arr = None
        try:
            tmp = np.asarray(all_true_F_list, dtype=float)
            if tmp.ndim == 3:
                all_true_F_arr = tmp  # (n_frames, n_atoms, 3)
        except Exception:
            all_true_F_arr = None
 
        if n_models > 1:
            # Ensemble-based spreads
            sigma_comp = sigma_F_flat.flatten()
            sigma_E_used = sigma_E_raw
        else:
            
            # Energy uncertainty from GP calibration
            F_val_lat    = mean_L_frame[val_mask]
            y_val_E      = stats_ens.delta_E_frame[val_mask]
            print(f"Calibrating energy uncertainty on single model")
            alpha_sq_E, _, terms_lat_E, _, _ = calibrate_alpha_reg_gcv(F_val_lat, y_val_E)
            sigma_val_E   = np.sqrt(alpha_sq_E * terms_lat_E)
            sigma_E_used  = np.full_like(all_true_E, np.nan)
            sigma_E_used[val_mask] = sigma_val_E
    
            # -------- NEW: latent‑GP for forces -------------------------------
            if mean_L_atom is not None:
                try:
                    atom_counts = stats_ens.atom_counts
                    comp_mask = np.repeat(val_mask, atom_counts * 3)
                    latents_comp = np.repeat(mean_L_atom, 3, axis=0)
                    F_val_lat_F = latents_comp[comp_mask]
                    y_val_F     = np.abs(stats_ens.all_force_residuals.flatten()[comp_mask])
                    print("Calibrating force uncertainty on single model")
                    alpha_sq_F, _, _, _, L_F = calibrate_alpha_reg_gcv(F_val_lat_F, y_val_F)
                    G_all_F     = scipy.linalg.solve_triangular(L_F, latents_comp.T, lower=True).T
                    terms_lat_F_all = np.sum(G_all_F**2, axis=1)
                    sigma_comp_full = np.sqrt(alpha_sq_F * terms_lat_F_all)
                    sigma_comp = sigma_comp_full
                except Exception as e:
                    print(f"Force latent calibration failed: {e}")
                    sigma_comp = np.full((mean_L_atom.shape[0] * 3,), np.nan)
            else:
                sigma_comp = np.full(1, np.nan)

        # per-atom uncertainty
        if sigma_comp is not None and sigma_comp.size % 3 == 0:
            sigma_atom = np.linalg.norm(sigma_comp.reshape(-1, 3), axis=1)
        else:
            sigma_atom = np.array([])

        # Ensemble summary
        print("=== Ensemble summary ===")
        print(f"Energy: mean={mu_E_frame.mean():.4f}, std={mu_E_frame.std():.4f}, min={min_E_frame.min():.4f}, max={max_E_frame.max():.4f}")
        print(f"Force : mean={mu_F_comp.mean():.4f}, std={std_F_comp.std():.4f}, min={min_F_comp.min():.4f}, max={max_F_comp.max():.4f}")

        # plotting & metrics -------------------------------------------------
    
        # histograms if plotting
#        do_plot = False  
        if do_plot:
    
            # Energy histograms
            plt.figure()
            plt.hist(mu_E_frame, bins=50)
            plt.xlabel("Frame-wise Mean Energy")
            plt.ylabel("Count")
            plt.title("Ensemble: Mean Energy per Frame")
#            plt.savefig("ensemble_energy_mean_hist.png")
    
            plt.figure()
            plt.hist(std_E_frame, bins=50)
            plt.xlabel("Frame-wise Energy Std")
            plt.ylabel("Count")
            plt.title("Ensemble: Energy Std Dev per Frame")
 #           plt.savefig("ensemble_energy_std_hist.png")
    
            # Force histograms
            plt.figure()
            plt.hist(mu_F_comp, bins=100)
            plt.xlabel("Comp-wise Mean Force")
            plt.ylabel("Count")
            plt.title("Ensemble: Mean Force per Component")
  #          plt.savefig("ensemble_force_mean_hist.png")
    
            plt.figure()
            plt.hist(std_F_comp, bins=100)
            plt.xlabel("Comp-wise Force Std")
            plt.ylabel("Count")
            plt.title("Ensemble: Force Std Dev per Component")
   #         plt.savefig("ensemble_force_std_hist.png")
    
            plt.show()
        # ——— end ensemble summary ———
#        do_plot = False  
#        if do_plot:
#            features_all, min_dists_all, _, pca, scaler = compute_features(
#                all_frames, config, train_xyz_path, train_mask, val_mask)
#            plot_mlff_stats(stats_ens, min_dists_all,
#                            "validation_results_ensemble", True, train_mask, val_mask)

        metrics_train = calculate_uq_metrics(stats_ens, sigma_comp, sigma_atom,
                                             sigma_E_raw, "Train", "ensemble", eval_log)
        if do_plot:
            generate_uq_plots(metrics_train["npz_path"], "Train", "error_model", calibration="var")

        metrics_eval  = calculate_uq_metrics(stats_ens, sigma_comp, sigma_atom,
                                             sigma_E_raw, "Eval", "ensemble", eval_log)
        if do_plot:
            generate_uq_plots(metrics_eval["npz_path"], "Eval", "error_model", calibration="var")
    
        # averaged latents across ensemble ----------------------------------
    
        if al_val_flag and al_val_flag.lower() in ["influence", "traditional"]:
    
            sel_objs, sel_idx = adaptive_learning_ensemble_calibrated(
                all_frames            = all_frames,
                eval_mask             = val_mask,
                delta_E_frame         = stats_ens.delta_E_frame,
                mean_l_al             = mean_L_frame,
                force_rmse_per_comp   = rmse_force_comp,
                denom_all             = all_true_F, 
                reference_frames      = reference_frames,   # <<< NEW
                beta                  = 0,
                drop_init             = 1.0,
                min_k                 = 5,
                max_k                 = 500,
                score_floor           = None,
                base                  = "al_ens_val")
    
            # optional “traditional” AL -------------------------------------
            if al_val_flag.lower() == "traditional":
                train_atom_mask = stats_ens._get_atom_mask(train_mask)
                sel_objs, sel_idx, _ = adaptive_learning(
                    all_frames, val_mask,
                    sigma_atom, stats_ens.force_rmse_per_atom,
                    train_atom_mask, None,
                    eval_cfg.get("num_active_frames", 50),
                    base_filename="al_traditional_val")
    
            # update masks ---------------------------------------------------
            train_mask[sel_idx] = True
            val_mask[sel_idx]   = False
    
            # save to XYZ ----------------------------------------------------
            if len(sel_idx):
                val_positions  = np.array([all_frames[i].get_positions() for i in sel_idx])
                val_forces_arr = np.stack([all_true_F[i] for i in sel_idx])
                val_energies   = all_true_E[sel_idx]
                atom_types     = all_frames[sel_idx[0]].get_chemical_symbols() \
                                   if hasattr(all_frames[sel_idx[0]], 'get_chemical_symbols') else []
                save_stacked_xyz("to_label_from_val.xyz",
                                 val_energies, val_positions, val_forces_arr, atom_types)
                print(f"Saved {len(sel_idx)} validation frames to 'to_label_from_val.xyz'.")
            else:
                print("No validation frames selected; nothing to save.")

        if pool_xyz_path and os.path.exists(pool_xyz_path):
            print(f"[Pool-AL] parsing unlabeled pool from {pool_xyz_path}")
            _pool_E, _pool_F, _pool_pos = parse_extxyz(pool_xyz_path, "unlabeled_pool")
            pool_frames = read(pool_xyz_path, index=":", format="extxyz")

            # -------------------------------------------------
            # 0. Ensemble eval on pool
            # -------------------------------------------------
            print("[Pool-AL] running ensemble inference on pool ...")
            t_pool_eval_0 = time.perf_counter()

            ens_E_pool, ens_F_pool, ens_L_pool, _, _, _ = evaluate_and_cache_ensemble(
                pool_frames, ensemble_folder, device, batch_size, eval_log,
                config, neighbour_list, npz_path="ensemble_unlabel.npz")

            t_pool_eval_1 = time.perf_counter()
            print(f"[Pool-AL] ensemble eval finished in "
                  f"{t_pool_eval_1 - t_pool_eval_0:.2f} s")

            ens_E_pool = np.stack(ens_E_pool, axis=0)    # (n_models, n_pool)
            ens_L_pool = np.stack(ens_L_pool, axis=0)    # (n_models, n_pool, d_lat)

            mu_E_pool    = np.mean(ens_E_pool, axis=0)   # (n_pool,)
            sigma_E_pool = np.std(ens_E_pool, axis=0, ddof=1)
            mu_F_pool    = np.mean(ens_F_pool, axis=0)   # (n_pool * n_atoms, 3) or similar
            sigma_F_pool = np.std(ens_F_pool, axis=0, ddof=1)
            mu_L_pool    = np.mean(ens_L_pool, axis=0)   # (n_pool, d_lat)
            mean_L_pool  = np.mean(ens_L_pool, axis=0)   # alias, same as mu_L_pool really

            print(f"[Pool-AL] ens_F_pool shape: {ens_F_pool.shape}")
            print(f"[Pool-AL] n_pool_frames = {len(pool_frames)}")

            # -------------------------------------------------
            # 1. Rolling average plot prep
            # -------------------------------------------------
            steps  = np.arange(len(mu_E_pool))
            window = 50
            df     = pd.DataFrame({"mu": mu_E_pool, "sigma": sigma_E_pool})
            sm     = df.rolling(window, center=True, min_periods=1).mean()

            # -------------------------------------------------
            # 2. Indexing: train/val splits
            # -------------------------------------------------
            train_idx     = np.where(train_mask)[0]
            val_idx       = np.where(val_mask)[0]

            train_frames  = [all_frames[i] for i in train_idx]
            val_frames_ref = [all_frames[i] for i in val_idx]  # THIS is our DFT-trusted reference set

            # -------------------------------------------------
            # 3. Thinning the pool before heavy stuff
            # -------------------------------------------------
            pool_stride   = eval_cfg.get("pool_stride", 1)
            thin_idx      = np.arange(len(pool_frames))[::pool_stride]

            pool_frames_thin   = [pool_frames[i] for i in thin_idx]          # (n_thin,)
            F_pool_thin        = mu_L_pool[thin_idx].astype(float)           # (n_thin, d_lat)
            mu_E_pool_thin     = mu_E_pool[thin_idx].astype(float)           # (n_thin,)
            sigma_E_pool_thin  = sigma_E_pool[thin_idx].astype(float)        # (n_thin,)
            F_train_thin       = mean_L_frame[train_idx].astype(float)       # (n_train, d_lat)

            print(f"[Pool-AL] pool thinning stride={pool_stride} → {len(pool_frames_thin)} frames")

            # -------------------------------------------------
            # 4. RDF thresholds from VALIDATION frames
            # -------------------------------------------------
            rdf_cache_file = "rdf_thresholds_cache.npz"
            rdf_stride     = eval_cfg.get("rdf_stride", 5)  # allow control from YAML

            if os.path.exists(rdf_cache_file):
                print(f"[Pool-AL] Loading cached RDF thresholds from {rdf_cache_file}")
                data = np.load(rdf_cache_file, allow_pickle=True)
                rdf_thresholds = data["rdf_thresholds"].item()  # dict was pickled into object array
            else:
                print("[Pool-AL] computing RDF thresholds from validation frames...")
                t_rdf0 = time.perf_counter()
                rdf_thresholds = compute_rdf_thresholds_from_reference(val_frames_ref, stride=rdf_stride)
                print("[Pool-AL] RDF thresholds computed.")
                t_rdf1 = time.perf_counter()
                print(f"[Pool-AL] RDF threshold calc took {t_rdf1 - t_rdf0:.2f} s")
                print(f"[Pool-AL] Caching RDF thresholds to {rdf_cache_file}")
                np.savez_compressed(rdf_cache_file, rdf_thresholds=rdf_thresholds)

            debug_plot_rdfs(val_frames_ref, rdf_thresholds,
                r_max=6.0,
                dr=0.02,
                outprefix="rdf_DEBUG")

            # -------------------------------------------------
            # 5. Fast RDF filter ON THINNED POOL using KDTree
            # -------------------------------------------------
            print("[Pool-AL] applying fast RDF hard-cutoff filter to thinned pool...")
            t_filt0 = time.perf_counter()
            rdf_ok_mask_thin = fast_filter_by_rdf_kdtree(pool_frames_thin, rdf_thresholds)
            t_filt1 = time.perf_counter()
            print(f"[Pool-AL] RDF filter done in {t_filt1 - t_filt0:.2f} s. "
                  f"{rdf_ok_mask_thin.sum()}/{len(rdf_ok_mask_thin)} kept.")

            # mask for plotting: mark BAD frames at the *original* pool indices
            bad_mask = np.zeros_like(steps, dtype=bool)
            # thin_idx[k] is original index of pool_frames_thin[k]
            # rdf_ok_mask_thin[k] == False → it's bad
            for k, orig_i in enumerate(thin_idx):
                if not rdf_ok_mask_thin[k]:
                    bad_mask[orig_i] = True

            # -------------------------------------------------
            # (OPTIONAL) Diagnostics / plots of RDF distributions
            # -------------------------------------------------
            # If you still want to generate RDF comparison plots, do it on
            # a *small* subset so you don't blow up runtime:
            """
            diag_subset = pool_frames_thin[:100]  # first 100 thinned frames
            diag_subset_kept = [fr for fr, ok in zip(diag_subset, rdf_ok_mask_thin[:100]) if ok]

            # You would then write tiny helpers like collect_pair_distances(...) etc.
            # BUT DO NOT run that on all 5k frames; it's purely for QC.
            """

            # -------------------------------------------------
            # 6. Save pool energy trace + bad_mask
            # -------------------------------------------------
            np.savez_compressed(
                "pool_energy_trace.npz",
                steps = steps,
                mu    = sm["mu"].values,
                sigma = sm["sigma"].values,
                bad   = bad_mask
            )
            print("[Pool-AL] Saved data to pool_energy_trace.npz")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(steps, sm["mu"], label=f"{window}-pt MA of μE")
            ax.fill_between(steps,
                            sm["mu"]-2*sm["sigma"],
                            sm["mu"]+2*sm["sigma"],
                            alpha=0.3,
                            label="σ (smoothed)")
            ax.scatter(steps[bad_mask],
                       sm["mu"][bad_mask],
                       marker='x',
                       label='RDF hard-reject')
            ax.legend()
            ax.set_xlabel("Pool frame index")
            ax.set_ylabel("Predicted energy")
            ax.set_title("Pool energy ± uncertainty")
            fig.tight_layout()
            fig.savefig("pool_energy_uncertainty.png", dpi=200)
            plt.close(fig)
            print("[Pool-AL] Saved plot pool_energy_uncertainty.png")

            # ------------------ Isotonic calibration (clean, no fallbacks) ------------------
            from mlff_qd.postprocessing.uq_metrics_calculator import IsotonicCalibrator
           
            # --- 1) Latent GP calibration (unchanged) ----------------------------------------
            print("[Pool-AL] Calibrating α² and λ (GCV)...")
            F_train_full = mean_L_frame[train_idx].astype(float)
            y_train_full = stats_ens.delta_E_frame[train_idx].astype(float)
            
            good_rows = np.isfinite(F_train_full).all(axis=1) & np.isfinite(y_train_full)
            F_train = F_train_full[good_rows]
            y_train = y_train_full[good_rows]
            
            alpha_sq, lambda_opt, _, _, L_chol = calibrate_alpha_reg_gcv(
                F_eval=F_train,
                y=y_train,
                lambda_bounds=(1e-6, 1e4)
            )
            assert np.isfinite(L_chol).all(), "L_chol has NaNs/Infs!"
            print("[Pool-AL] Latent calibration done (alpha_sq, lambda_opt).")

            # --- UNCERTAINTY Calibrator (iso_unc) ---
            # --- Make sure you have these imports at the top ---
            from sklearn.isotonic import IsotonicRegression # <-- The ONLY calibrator you need
            # from mlff_qd.postprocessing.uq_metrics_calculator import IsotonicCalibrator # <-- You can remove this
            
            # --- UNCERTAINTY Calibrator (iso_unc) ---
            # --- 2) Isotonic calibration mapping ensemble σ -> expected |ΔE| -------------------
            print("[Pool-AL] Fitting UNCERTAINTY calibrator (sigma -> |ΔE|)...")
    
            # training arrays (already computed above)
            sigma_train_raw = sigma_E_raw[train_idx].astype(float)       # ensemble std on training frames
            delta_train_abs = np.abs(stats_ens.delta_E_frame[train_idx].astype(float)) # DFT - ML on training frames
            
            # fit isotonic calibrator (maps sigma -> expected |ΔE|)
            # --- FIX: Use standard sklearn class. It's robust and correct. ---
            iso_unc = IsotonicRegression(y_min=0.0, out_of_bounds='clip')
            iso_unc.fit(sigma_train_raw, delta_train_abs)
            print("[Pool-AL] Uncertainty calibrator fitted.")
            
            # diagnostics: transform training sigma and compute correlation + binned table
            # --- FIX: Use .predict() instead of .transform() ---
            sigma_train_cal = iso_unc.predict(sigma_train_raw)
            corr = np.corrcoef(sigma_train_cal, delta_train_abs)[0, 1]
            print(f"[Pool-AL] Calibration correlation corr(f(σ), |ΔE|) = {corr:.4f}")
    
            # --- 2. BIAS Calibrator (iso_bias) ---
            # GOAL: Map mu_E (X) -> expected_signed_error (y)
        
            print("[Pool-AL] Fitting BIAS calibrator (mu_E -> ΔE)...")
            # The input (X) is the predicted energy
            mu_E_train = mu_E_frame[train_idx].astype(float)
            # The target (y) is the SIGNED error
            delta_train_signed = stats_ens.delta_E_frame[train_idx].astype(float)
        
            # --- FIX: Use standard sklearn class to prevent crash ---
            iso_bias = IsotonicRegression(y_min=None, y_max=None, out_of_bounds='clip')
            iso_bias.fit(mu_E_train, delta_train_signed)
            print("[Pool-AL] Bias calibrator fitted.")
            
            # --- 3A. UNCERTAINTY Plots (Your existing code, now fixed) ---
            
            # binned diagnostics
            n_bins = 8
            bins = np.quantile(sigma_train_raw, np.linspace(0, 1, n_bins + 1))
            bin_idx = np.digitize(sigma_train_raw, bins, right=True) - 1
            bin_info = []
            for i in range(n_bins):
                sel = bin_idx == i
                if sel.any():
                    # --- FIX: Use delta_train_abs (already abs'd) ---
                    bin_info.append((i, np.median(sigma_train_raw[sel]), np.mean(delta_train_abs[sel]),
                                     np.median(sigma_train_cal[sel]), sel.sum()))
                    print(f"  UNCERT bin {i:2d}: median_sigma={bin_info[-1][1]:.4e}, mean|ΔE|={bin_info[-1][2]:.4e}, "
                          f"median_f(sigma)={bin_info[-1][3]:.4e}, count={bin_info[-1][4]}")
            
            # plots
            # 1) scatter: f(sigma) vs |delta|
            plt.figure(figsize=(5,4))
            # --- FIX: Use delta_train_abs (already abs'd) ---
            plt.scatter(sigma_train_raw, delta_train_abs, s=8, alpha=0.6, label="train")
            s_sorted = np.sort(sigma_train_raw)
            # --- FIX: Use .predict() ---
            plt.plot(s_sorted, iso_unc.predict(s_sorted), color="C1", lw=2, label="isotonic f(σ)")
            plt.plot([s_sorted.min(), s_sorted.max()], [s_sorted.min(), s_sorted.max()], 'k--', lw=1, label="identity")
            plt.xlabel("ensemble σ (training)")
            plt.ylabel("|ΔE| (training)")
            plt.legend()
            plt.tight_layout()
            plt.savefig("calibration_scatter.png", dpi=200)
            plt.close()
            print("[Pool-AL] Saved calibration_scatter.png")
            
            # 2) binned means plot
            medians_sigma = [b[1] for b in bin_info]
            means_absdelta = [b[2] for b in bin_info]
            medians_f = [b[3] for b in bin_info]
            plt.figure(figsize=(5,4))
            plt.plot(medians_sigma, means_absdelta, 'o-', label="mean |ΔE| (bin)")
            plt.plot(medians_sigma, medians_f, 's--', label="median f(σ) (bin)")
            plt.xlabel("median ensemble σ (bin)")
            plt.ylabel("energy (eV)")
            plt.legend()
            plt.tight_layout()
            plt.savefig("calibration_binned.png", dpi=200)
            plt.close()
            print("[Pool-AL] Saved calibration_binned.png")
    
            # ---------- Log-log Error vs Uncertainty plot (uses your plotting helper) ----------
            from mlff_qd.postprocessing.plotting import plot_swapped_final_tight
            
            # prepare arrays (use positive, finite values only)
            mask_plot = (sigma_train_raw > 0) & np.isfinite(sigma_train_raw) & np.isfinite(delta_train_abs)
            if np.any(mask_plot):
                x_plot = sigma_train_raw[mask_plot]
                y_plot = delta_train_abs[mask_plot]
                # figure and ax
                fig, ax = plt.subplots(1, 1, figsize=(6,5))
                # reuse your plotting function in log scale
                plot_swapped_final_tight(ax, x_plot, y_plot, scale="log",
                                         title="|Δ| vs σ (training) — isotonic fit (log-log)",
                                         xlabel="Predicted uncertainty σ", ylabel="|Δ|",
                                         q_low=0.005, q_high=0.995)
                # overlay isotonic fit as a solid line
                s_sorted = np.sort(x_plot)
                # --- FIX: Use .predict() ---
                ax.plot(s_sorted, iso_unc.predict(s_sorted), color="tab:orange", lw=2.2, label="Isotonic f(σ)")
                # overlay binned median for extra clarity
                bins = np.logspace(np.log10(s_sorted.min()), np.log10(s_sorted.max()), 30)
                from scipy.stats import binned_statistic
                bin_med_sigma, _, _ = binned_statistic(x_plot, x_plot, statistic="median", bins=bins)
                bin_med_absdelta, _, _ = binned_statistic(x_plot, y_plot, statistic="median", bins=bins)
                ok = ~np.isnan(bin_med_sigma) & ~np.isnan(bin_med_absdelta)
                if ok.sum() > 1:
                    ax.plot(bin_med_sigma[ok], bin_med_absdelta[ok], 's-', color="tab:purple", lw=1, markersize=4, label="binned median")
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig("calibration_abs_vs_sigma_log.png", dpi=200)
                plt.close(fig)
                print("[Pool-AL] Saved calibration_abs_vs_sigma_log.png")
            else:
                print("[Pool-AL] No valid points to plot log-log calibration.")
            # ---------- end plot ----------
    
            # --- 3B. --- NEW PLOTS --- BIAS Calibrator Diagnostics ---
            
            print("[Pool-AL] Generating diagnostic plots for BIAS calibrator...")
            
            # Get calibrated bias predictions for diagnostics
            bias_train_cal = iso_bias.predict(mu_E_train)
            
            # Plot 4) scatter: f(E) vs delta_E
            plt.figure(figsize=(5,4))
            plt.scatter(mu_E_train, delta_train_signed, s=8, alpha=0.6, label="train (signed error)")
            e_sorted = np.sort(mu_E_train)
            plt.plot(e_sorted, iso_bias.predict(e_sorted), color="C1", lw=2, label="isotonic bias f(E)")
            plt.plot([e_sorted.min(), e_sorted.max()], [0, 0], 'k--', lw=1, label="zero bias")
            plt.xlabel("Predicted Energy μE (training)")
            plt.ylabel("Signed Error ΔE (training)")
            plt.legend()
            plt.tight_layout()
            plt.savefig("bias_calibration_scatter.png", dpi=200)
            plt.close()
            print("[Pool-AL] Saved bias_calibration_scatter.png")
    
            # Binned diagnostics (for bias)
            # We bin by the predictor: mu_E_train
            bins_bias = np.quantile(mu_E_train, np.linspace(0, 1, n_bins + 1))
            bin_idx_bias = np.digitize(mu_E_train, bins_bias, right=True) - 1
            bin_info_bias = []
            for i in range(n_bins):
                sel_bias = bin_idx_bias == i
                if sel_bias.any():
                    med_E = np.median(mu_E_train[sel_bias])
                    mean_delta_E = np.mean(delta_train_signed[sel_bias])
                    med_f_E = np.median(bias_train_cal[sel_bias])
                    bin_info_bias.append((med_E, mean_delta_E, med_f_E, sel_bias.sum()))
                    print(f"  BIAS bin {i:2d}:   median_μE={bin_info_bias[-1][0]:.4e}, mean_ΔE={bin_info_bias[-1][1]:.4e}, "
                          f"median_f(E)={bin_info_bias[-1][2]:.4e}, count={bin_info_bias[-1][3]}")
    
            # Plot 5) binned means plot (for bias)
            medians_E = [b[0] for b in bin_info_bias]
            means_delta = [b[1] for b in bin_info_bias]
            medians_f_E = [b[2] for b in bin_info_bias]
            plt.figure(figsize=(5,4))
            plt.plot(medians_E, means_delta, 'o-', label="mean ΔE (bin)")
            plt.plot(medians_E, medians_f_E, 's--', label="median f(E) (bin)")
            plt.plot([min(medians_E), max(medians_E)], [0, 0], 'k--', lw=1, label="zero bias")
            plt.xlabel("median predicted μE (bin)")
            plt.ylabel("energy (eV)")
            plt.legend()
            plt.tight_layout()
            plt.savefig("bias_calibration_binned.png", dpi=200)
            plt.close()
            print("[Pool-AL] Saved bias_calibration_binned.png")
            
            # --- END OF NEW PLOTS ---
            
            print("[Pool-AL] Calibrating ENTIRE labeled set (train+val) sigmas...")
            # --- FIX: Use .predict() ---
            sigma_E_labeled_calibrated = iso_unc.predict(sigma_E_raw.astype(float))
        
            # apply isotonic transform to thinned pool ensemble sigmas
            # --- FIX: Use .predict() ---
            sigma_E_calibrated_thin = iso_unc.predict(sigma_E_pool_thin.astype(float))
            print(f"[Pool-AL] Calibrated sigma (thinned pool) stats: min/med/max = "
                  f"{np.nanmin(sigma_E_calibrated_thin):.4e}/{np.nanmedian(sigma_E_calibrated_thin):.4e}/{np.nanmax(sigma_E_calibrated_thin):.4e}")
    
            
            # --- THIS BLOCK IS UNCHANGED, AS YOU REQUESTED ---
            # run AL using calibrated sigma as the energy-uncertainty measure
            print("[Pool-AL] Running windowed active learning on thinned pool ...")
            Sel_objs_thin, sel_rel_thin = adaptive_learning_mig_pool_windowed(
                pool_frames_thin,
                F_pool_thin,
                F_train_thin,
                alpha_sq,
                L_chol,
                forces_train     = forces_train_arr,
                sigma_energy     = sigma_E_raw,  # calibrated σ' per thin-pool frame
                sigma_force      = sigma_comp,
                mu_E_frame_train = mu_E_frame[train_idx],
                mu_E_pool        = mu_E_pool_thin,
                sigma_E_pool     = sigma_E_pool_thin,
                mu_F_pool        = mu_F_pool,
                sigma_F_pool     = sigma_F_pool,
                rdf_thresholds   = rdf_thresholds,
                rho_eV           = eval_cfg.get("rho_eV", 0.002),
                min_k            = eval_cfg.get("pool_min_k", 5),
                window_size      = eval_cfg.get("pool_window", 100),
                base             = "al_pool_v1",
                budget_max       = budget_max,
                percentile_gamma = percentile_gamma,
                percentile_F_low = percentile_F_low,
                percentile_F_hi  = percentile_F_hi,
                hard_sigma_E_atom_min = eval_cfg.get("thr_sE_atom", thr_sE_atom),
                hard_sigma_F_mean_min = eval_cfg.get("thr_sF_mean", thr_sF_mean),
                hard_sigma_F_max_min  = eval_cfg.get("thr_sF_max", thr_sF_max),
                hard_Fmax_train_mult  = eval_cfg.get("thr_Fmax_mult", thr_Fmax_mult),
            )
            print("[Pool-AL] Windowed AL finished.")
    
            # -------------------------------------------------
            # 9. Log window metadata
            # -------------------------------------------------
            n_thin = len(thin_idx)
            w_size = eval_cfg.get("pool_window", 100)
            n_win  = (n_thin + w_size - 1) // w_size
            with open("pool_window_log.txt", "w") as fh:
                fh.write(f"total_frames_original\t{len(pool_frames)}\n")
                fh.write(f"thin_stride\t{pool_stride}\n")
                fh.write(f"frames_after_thin\t{n_thin}\n")
                fh.write(f"window_size\t{w_size}\n")
                fh.write(f"num_windows\t{n_win}\n")
            print("[Pool-AL] wrote pool_window_log.txt")
            
            # map thin indices back to full indices and save selected frames with comments
            sel_global_idx = thin_idx[sel_rel_thin]
            sel_objs       = [pool_frames[i] for i in sel_global_idx]
    
            # --- THIS BLOCK IS FOR APPLYING CALIBRATION TO THE POOL ---
            
            if sel_objs:
                with open("to_DFT_labelling_from_pool.xyz", "w") as fh:
                    for orig_idx in sel_global_idx:
                        atoms = pool_frames[orig_idx]
                        
                        # --- 1. Get raw predictions for this frame ---
                        e_pred_raw = float(mu_E_pool[orig_idx])
                        s_raw = float(sigma_E_pool[orig_idx])
    
                        # --- 2. Apply BOTH calibrations ---
                        # --- FIX: Use .predict() and np.array() to prevent crash ---
                        
                        # Get the bias correction by applying iso_bias to the raw energy
                        bias_correction = float(iso_bias.predict(np.array([e_pred_raw]))[0])
                        
                        # Get the calibrated uncertainty by applying iso_unc to the raw sigma
                        s_calibrated = float(iso_unc.predict(np.array([s_raw]))[0])
    
                        # --- 3. Create your new calibrated "ballpark" values ---
                        # (Assuming delta=E_ml - E_true, so E_true = E_ml - delta)
                        e_pred_calibrated = e_pred_raw - bias_correction 
                        
                        # --- 4. Write the rich, new comment ---
                        comment = (
                            f"frame={orig_idx}, "
                            f"e_pred_raw={e_pred_raw:.6f}, "
                            f"s_raw={s_raw:.6f}, "
                            f"bias_corr={-bias_correction:.6f}, " # Note: -bias_correction
                            f"s_calibrated={s_calibrated:.6f}, "
                            f"BALLPARK=[{e_pred_calibrated - s_calibrated:.6f}, {e_pred_calibrated + s_calibrated:.6f}]"
                        )
                        write(fh, atoms, format="xyz", comment=comment)
                            
                print(f"[Pool-AL] Saved {len(sel_objs)} pool frames to 'to_DFT_labelling_from_pool.xyz'.")
            else:
                print("[Pool-AL] No pool frames selected; nothing to save.")
            
            # ------------------ end isotonic replacement ------------------


    # 9) GMM UQ
    if "GMM" in uq_methods and stats_base is not None and all_latent is not None:
        print("\n--- GMM UQ Method ---")
        compute_per_atom = eval_cfg.get("gmm_compute_per_atom", False)
        latent_train = [l for i,l in enumerate(all_latent) if train_mask[i] and l.ndim==1]
        latent_all   = [l for l in all_latent if l.ndim==1]
        pa_train = ([l for i,l in enumerate(all_peratom_latent) if train_mask[i]] if compute_per_atom else None)
        pa_all   = all_peratom_latent if compute_per_atom else None
        results = fit_gmm_and_compute_uncertainty(
            latent_train, latent_all,
            pa_train, pa_all,
            compute_per_atom_uncertainty=compute_per_atom,
            max_components=eval_cfg.get("gmm_max_components",20),
            outlier_threshold=eval_cfg.get("gmm_outlier_threshold",5.0)
        )
        # unpack only needed entries
        train_eff_var = results[3]
        eval_eff_var  = results[4]
        per_atom_eff_var_train = None
        per_atom_eff_var_eval  = None
        if compute_per_atom and len(results) > 13:
            per_atom_eff_var_train = results[12]
            per_atom_eff_var_eval  = results[13]
        if compute_per_atom and per_atom_eff_var_train is not None:
            eff_var_all = np.full(stats_base.force_rmse_per_atom.shape, np.nan)
            mask_tr = stats_base._get_atom_mask(train_mask)
            mask_ev = stats_base._get_atom_mask(val_mask)
            eff_var_all[mask_tr] = per_atom_eff_var_train
            eff_var_all[mask_ev] = per_atom_eff_var_eval
            sigma_atom = np.sqrt(np.maximum(0, eff_var_all))
            sigma_comp = np.repeat(sigma_atom/np.sqrt(3), 3)
            calculate_uq_metrics(stats_base, sigma_comp, sigma_atom, None,
                                  "Train","GMM",eval_log)
            calculate_uq_metrics(stats_base, sigma_comp, sigma_atom, None,
                                  "Eval","GMM",eval_log)
        else:
            print("Skipping GMM per-atom metrics.")

    print("run_eval completed.")

