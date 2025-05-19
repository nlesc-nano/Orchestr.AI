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
import plotly.tools as tls
import plotly.io as pio

from ase.io import read, write  # Needed for reading training/eval data

# === Local Module Imports ===
from postprocessing.parsing import parse_extxyz, save_stacked_xyz
from postprocessing.calculator import setup_neighbor_list, assign_charges, evaluate_model
from postprocessing.stats import MLFFStats
from postprocessing.features import compute_features
from postprocessing.uq_models import train_uq_models, predict_uncertainties, fit_gmm_and_compute_uncertainty
#from postprocessing.uq_metrics_calculator import calculate_uq_metrics
from postprocessing.uq_metrics_calculator import VarianceScalingCalibrator, run_uq_metrics, calculate_uq_metrics
from postprocessing.mlff_plotting import plot_mlff_stats
from postprocessing.plotting import (
    generate_uq_plots, 
    generate_al_influence_plots, generate_al_traditional_plots
)
from postprocessing.active_learning import adaptive_learning, adaptive_learning_mig_calibrated, adaptive_learning_mig_pool,calibrate_alpha_reg_gcv, predict_sigma_from_L, adaptive_learning_mig_pool_windowed, filter_unrealistic_indices, compute_bond_thresholds, adaptive_learning_ensemble_calibrated 


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
            mdl = torch.load(mfile, map_location=device).to(torch.float32)
            mdl.eval()
            loaded_models.append(mdl)

            # evaluate_model now returns latent_per_atom
            pred_E, pred_F_per_frame, latent_per_frame, latent_per_atom, _ = evaluate_model(
                frames,
                dummy_E,
                dummy_F,
                mdl,
                device,
                batch_size,
                "ensemble",
                1,
                log_path,
                None,
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
    arr_E      = np.stack(energy_list).astype(np.float32)          # (n_models, n_frames)
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

    eval_cfg        = config.get("eval", {})
    error_estimate  = eval_cfg.get("error_estimate", False)
    al_flag         = eval_cfg.get("active_learning", None)
    ensemble_folder = eval_cfg.get("ensemble_folder", None)
    pool_xyz_path   = eval_cfg.get("unlabeled_pool_path", None)
    do_plot         = eval_cfg.get("plot", False)
    neighbour_list  = setup_neighbor_list(config)

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

    all_frames    = train_frames + val_frames
    all_true_E    = np.array(train_E + val_E)
    all_true_F    = train_F + val_F

    n_train = len(train_frames)
    n_val   = len(val_frames)
    train_mask = np.array([True]*n_train + [False]*n_val, dtype=bool)
    val_mask   = np.array([False]*n_train + [True]*n_val, dtype=bool)
    print(f"Total frames: {len(all_frames)} (train={n_train}, val={n_val})")

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
            base_model = torch.load(base_path, map_location=device).to(torch.float32)
            base_model.eval()
            print(f"Loaded base model from {base_path}")

    # 4) Base evaluation and feature computation
    stats_base = None
    features_all = None
    min_dists_all = None
    all_latent = None
    all_peratom_latent = None
    if need_base and base_model:
        pred_E, pred_F, all_latent, all_peratom_latent, _ = evaluate_model(
            all_frames,
            list(all_true_E),
            all_true_F,
            base_model,
            device,
            batch_size,
            tag="none",
            n_mc=1,
            log_path=eval_log,
            unique_log=None,
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
        ens_E_sel       = ens_E[:n_models]
        ens_F_sel       = ens_F[:n_models]
        ens_L_frame_sel = ens_L_frame[:n_models]
 
        mu_E          = np.mean(ens_E_sel, axis=0)
        sigma_mu_E    = np.std(ens_E_sel, axis=0)

        mu_F_flat     = np.mean(ens_F_sel, axis=0)

        mean_L_frame  = np.mean(ens_L_frame_sel, axis=0)   # (n_frames, d_frame)

        # build MLFFStats ----------------------------------------------------
        mf_list, idx = [], 0
        for fr in all_frames:
            n = len(fr)
            mf_list.append(mu_F_flat[idx:idx+n])
            idx += n
        stats_ens = MLFFStats(all_true_E, mu_E, all_true_F, mf_list, train_mask, val_mask)
        # ---- bookkeeping -------------------------------------------------
        atom_counts     = stats_ens.atom_counts              # shape (n_frames,)
        total_comp      = 3 * atom_counts.sum()
        comp_mask       = np.repeat(val_mask, atom_counts * 3)   # shape (total_comp,)
        latents_comp    = np.repeat(mean_L_atom, 3, axis=0)

        # Convert all_true_F list to NumPy array
        all_true_F = np.array(all_true_F)  # Shape: (n_frames, n_atoms, 3)
        n_frames, n_atoms = all_true_F.shape[:2]
        print(f"n_frames: {n_frames}")
        print(f"n_atoms: {n_atoms}")
        N_tot_atoms = n_frames * n_atoms
        
        force_res_flat = stats_ens.all_force_residuals.flatten()  # Shape: (3*N_tot_atoms,)
        res_per_frame_atom = force_res_flat.reshape(n_frames, n_atoms, 3)  # Shape: (n_frames, n_atoms, 3)
        
        # Compute RMSE (absolute residuals) per component per atom per frame
        rmse_force_comp = np.sqrt(res_per_frame_atom**2)  # Shape: (n_frames, n_atoms, 3)
        max_rmse_force_comp = np.max(rmse_force_comp, axis=(1, 2))  # Shape: (n_frames,)
        max_indices = np.argmax(rmse_force_comp.reshape(n_frames, -1), axis=1)  # Shape: (n_frames,)
        atom_indices = max_indices // 3  # Atom index
        comp_indices = max_indices % 3   # Component index (0, 1, 2 for x, y, z)
        
        # Find the frame with the overall maximum RMSE
        overall_max_frame_idx = np.argmax(max_rmse_force_comp)  # Index of frame with max RMSE
        overall_max_rmse = max_rmse_force_comp[overall_max_frame_idx]  # Scalar
        overall_atom_idx = atom_indices[overall_max_frame_idx]  # Atom index
        overall_comp_idx = comp_indices[overall_max_frame_idx]  # Component index
        
        # Extract true and predicted forces for the max RMSE component
        true_force = all_true_F[overall_max_frame_idx, overall_atom_idx, overall_comp_idx]  # Scalar (eV/Å)
        pred_force = true_force + res_per_frame_atom[overall_max_frame_idx, overall_atom_idx, overall_comp_idx]  # Scalar (eV/Å)
        
        # Compute absolute DFT forces
        abs_true_F = np.abs(all_true_F)  # Shape: (n_frames, n_atoms, 3)
        max_abs_true_F = abs_true_F[np.arange(n_frames), atom_indices, comp_indices]  # Shape: (n_frames,)
        
        # Compute relative RMSE for maximum force component per frame
        F0 = 0.01 * np.mean(abs_true_F)  # eV/Å floor
        print(f"F0: {F0} eV/Å")  # Debug F0
        denom = np.maximum(max_abs_true_F, F0)  # Shape: (n_frames,)
        rel_max_rmse_force_comp = max_rmse_force_comp / denom  # Shape: (n_frames,)
        
        # Debug max relative error
        max_rel_idx = np.argmax(rel_max_rmse_force_comp)
        print(f"Frame with max rel_max_rmse_force_comp: {max_rel_idx}")
        print(f"Max rel_max_rmse_force_comp value: {rel_max_rmse_force_comp[max_rel_idx]}")
        print(f"Corresponding max_rmse_force_comp: {max_rmse_force_comp[max_rel_idx]} eV/Å")
        print(f"Corresponding max_abs_true_F: {max_abs_true_F[max_rel_idx]} eV/Å")
        print(f"Corresponding denom: {denom[max_rel_idx]} eV/Å")
        print(f"Frames with rel_max : {rel_max_rmse_force_comp}")
        
        # Compute relative RMSE for all components
        rmse_force_comp = rmse_force_comp.flatten()  # Shape: (3*n_frames*n_atoms,)
        abs_true_F = abs_true_F.flatten()  # Shape: (3*n_frames*n_atoms,)
        denom_all = np.maximum(abs_true_F, F0)
        rel_rmse_force_comp = rmse_force_comp / denom_all  # Shape: (3*n_frames*n_atoms,)
        
        # Print diagnostics
        print(f"rmse_force_comp: {np.min(rmse_force_comp)} {np.max(rmse_force_comp)}")
        print(f"rel_rmse_force_comp: {np.min(rel_rmse_force_comp)} {np.max(rel_rmse_force_comp)}")
        print(f"max_rmse_force_comp: {np.min(max_rmse_force_comp)} {np.max(max_rmse_force_comp)}")
        print(f"rel_max_rmse_force_comp: {np.min(rel_max_rmse_force_comp)} {np.max(rel_max_rmse_force_comp)}")
        print(f"Overall max RMSE: {overall_max_rmse} eV/Å at frame {overall_max_frame_idx}, "
              f"atom {overall_atom_idx}, component {['x', 'y', 'z'][overall_comp_idx]}")
        print(f"True DFT force: {true_force} eV/Å")
        print(f"Predicted force: {pred_force} eV/Å")
        print(f"Absolute error: {abs(pred_force - true_force)} eV/Å")
        print(f"Relative error: {abs(pred_force - true_force) / np.maximum(abs(true_force), F0)}")
        
        if n_models > 1:
            # Ensemble-based spreads
            sigma_E_raw = np.std(ens_E_sel, axis=0, ddof=1)
            sigma_comp  = np.std(ens_F_sel, axis=0, ddof=1).flatten()
        else:
            # Energy uncertainty from GP calibration
            F_val_lat    = mean_L_frame[val_mask]
            y_val_E      = stats_ens.delta_E_frame[val_mask]
            print(f"Calibrating energy uncertainty on single model")
            alpha_sq_E, _, terms_lat_E, _, _ = calibrate_alpha_reg_gcv(F_val_lat, y_val_E)
            sigma_val_E  = np.sqrt(alpha_sq_E * terms_lat_E)
            sigma_E_raw  = np.full_like(all_true_E, np.nan)
            sigma_E_raw[val_mask] = sigma_val_E
    
            # -------- NEW: latent‑GP for forces -------------------------------
            F_val_lat_F = latents_comp[comp_mask]              # (n_val_comp, d_atom)
            y_val_F     = np.abs(force_res_flat[comp_mask])        # (n_val_comp,)
            print(f"Calibrating force uncertainty on single model")
            alpha_sq_F, _, _, _, L_F = calibrate_alpha_reg_gcv(F_val_lat_F, y_val_F)
            G_all_F     = scipy.linalg.solve_triangular(L_F, latents_comp.T, lower=True).T
            terms_lat_F_all = np.sum(G_all_F**2, axis=1)          # (total_comp,)
            sigma_comp_full = np.sqrt(alpha_sq_F * terms_lat_F_all)   # (total_comp,)
            sigma_comp = sigma_comp_full 
            
        sigma_atom = np.linalg.norm(sigma_comp.reshape(-1,3), axis=1)  
    
    
        # plotting & metrics -------------------------------------------------
        # ——— ensemble summary statistics ———
        # per-frame energy slices
        mu_E_frame  = mu_E                           # (n_frames,)
        std_E_frame = np.std(ens_E_sel, axis=0)     # (n_frames,)
        min_E_frame = np.min(ens_E_sel, axis=0)     # (n_frames,)
        max_E_frame = np.max(ens_E_sel, axis=0)     # (n_frames,)
    
        # per-component force slices (flattened)
        mu_F_comp   = mu_F_flat                      # (total_components,)
        std_F_comp  = np.std(ens_F_sel, axis=0).flatten()
        min_F_comp  = np.min(ens_F_sel, axis=0).flatten()
        max_F_comp  = np.max(ens_F_sel, axis=0).flatten()
    
        # overall summaries
        print("=== Ensemble summary ===")
        print(f"Energy: mean={mu_E_frame.mean():.4f}, "
              f"std={mu_E_frame.std():.4f}, "
              f"min={min_E_frame.min():.4f}, "
              f"max={max_E_frame.max():.4f}")
        print(f"Force : mean={mu_F_comp.mean():.4f},  "
              f"std={mu_F_comp.std():.4f},  "
              f"min={min_F_comp.min():.4f},  "
              f"max={max_F_comp.max():.4f}")
    
        # histograms if plotting
        do_plot = True 
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


        do_plot = False 
        if do_plot:
            features_all, min_dists_all, _, pca, scaler = compute_features(
                all_frames, config, train_xyz_path, train_mask, val_mask)
            plot_mlff_stats(stats_ens, min_dists_all,
                            "validation_results_ensemble", True, train_mask, val_mask)

        metrics_train = calculate_uq_metrics(stats_ens, sigma_comp, sigma_atom,
                                             sigma_E_raw, "Train", "ensemble", eval_log)
        if do_plot:
            generate_uq_plots(metrics_train["npz_path"], "Train", "error_model")

        metrics_eval  = calculate_uq_metrics(stats_ens, sigma_comp, sigma_atom,
                                             sigma_E_raw, "Eval", "ensemble", eval_log)
        if do_plot:
            generate_uq_plots(metrics_eval["npz_path"], "Eval", "error_model")
    
        # averaged latents across ensemble ----------------------------------
    
        if al_val_flag and al_val_flag.lower() in ["influence", "traditional"]:
    
            sel_objs, sel_idx = adaptive_learning_ensemble_calibrated(
                all_frames            = all_frames,
                eval_mask             = val_mask,
                delta_E_frame         = stats_ens.delta_E_frame,
                mean_l_al             = mean_L_frame,
                force_rmse_per_comp   = rmse_force_comp,
                denom_all             = all_true_F, 
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
            print(f"[Pool‑AL] parsing unlabeled pool from {pool_xyz_path}")
            _pool_E, _pool_F, _pool_pos = parse_extxyz(pool_xyz_path, "unlabeled_pool")
            pool_frames = read(pool_xyz_path, index=":", format="extxyz")
    
            # ensemble eval on pool -----------------------------------------
            ens_E_pool, ens_F_pool, ens_L_pool, _, _, _ = evaluate_and_cache_ensemble(
                pool_frames, ensemble_folder, device, batch_size, eval_log,
                config, neighbour_list, npz_path="ensemble_unlabel.npz")

            ens_E_pool = np.stack(ens_E_pool, axis=0)
            ens_L_pool = np.stack(ens_L_pool, axis=0)

            mu_E_pool    = np.mean(ens_E_pool, axis=0)
            sigma_E_pool = np.std(ens_E_pool, axis=0, ddof=1)
            mu_F_pool    = np.mean(ens_F_pool, axis=0)
            print(f"ens_F_pool shape: {ens_F_pool.shape}")

            sigma_F_pool = np.std(ens_F_pool, axis=0, ddof=1)
            mu_L_pool    = np.mean(ens_L_pool, axis=0)
            mean_L_frame = np.mean(ens_L_pool, axis=0)

            # rolling average plot (unchanged) ------------------------------
            steps  = np.arange(len(mu_E_pool))
            window = 50
            df     = pd.DataFrame({"mu": mu_E_pool, "sigma": sigma_E_pool})
            sm     = df.rolling(window, center=True, min_periods=1).mean()

    
            # bond‑length thresholds ----------------------------------------
            train_idx     = np.where(train_mask)[0]
            train_frames  = [all_frames[i] for i in train_idx]
            thresholds    = compute_bond_thresholds(train_frames, neighbour_list,
                                                    first_shell_cutoff=3.4)
            bad_global    = filter_unrealistic_indices(pool_frames, neighbour_list, thresholds)
    
            # thin for AL ----------------------------------------------------
            pool_stride   = eval_cfg.get("pool_stride", 50)
            thin_idx      = np.arange(len(pool_frames))[::pool_stride]
            pool_frames_thin   = [pool_frames[i] for i in thin_idx]
            F_pool_thin        = mu_L_pool[thin_idx].astype(float)
            mu_E_pool_thin     = mu_E_pool[thin_idx].astype(float)
            sigma_E_pool_thin  = sigma_E_pool[thin_idx].astype(float)
            F_train_thin       = mean_L_frame[thin_idx].astype(float)   # needed by AL
            
            # ---------------------------------------------------------------------
            # DROP NaN / Inf ROWS
            # ---------------------------------------------------------------------
            mask_pool  = np.isfinite(F_pool_thin).all(axis=1)
            mask_train = np.isfinite(F_train_thin).all(axis=1)
            finite_mask = mask_pool & mask_train
            
            if not finite_mask.all():
                dropped = np.where(~finite_mask)[0]
                print(f"[Pool-AL] dropping {len(dropped)} NaN/Inf rows: {dropped.tolist()}")
                thin_idx          = thin_idx[finite_mask]
                pool_frames_thin  = [pool_frames[i] for i in thin_idx]
                F_pool_thin       = F_pool_thin [finite_mask]
                F_train_thin      = F_train_thin[finite_mask]
                mu_E_pool_thin    = mu_E_pool_thin[finite_mask]
                sigma_E_pool_thin = sigma_E_pool_thin[finite_mask]

            # ---------------------------------------------------------------------
            # MAP bad_global TO THINNED COORDINATES
            # ---------------------------------------------------------------------
            bad_rel = set(np.nonzero(np.isin(thin_idx, list(bad_global)))[0])
            
            # highlight bond outliers on plot -------------------------------
            bad_thin = bad_global.intersection(set(thin_idx))
            bad_mask = np.isin(steps, list(bad_global))

            # Save everything for later re-plotting:
            np.savez_compressed(
                "pool_energy_trace.npz",
                steps = steps,
                mu    = sm["mu"].values,
                sigma = sm["sigma"].values,
                bad   = bad_mask
            )
            print("Saved data to pool_energy_trace.npz")

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(steps, sm["mu"], label=f"{window}-pt MA of μE")
            ax.fill_between(steps, sm["mu"]-2*sm["sigma"], sm["mu"]+2*sm["sigma"],
                             alpha=0.3, label="σ (smoothed)")
            ax.scatter(steps[bad_mask], sm["mu"][bad_mask], marker='x', label='bond outlier')
            ax.legend()
            ax.set_xlabel("Pool frame index")
            ax.set_ylabel("Predicted energy")
            ax.set_title("Pool energy ± uncertainty")
            fig.tight_layout()

            fig.savefig("pool_energy_uncertainty.png", dpi=200)

            # ----------------------------------------------------------------------
            # ❶  Build training design matrix and targets
            # ----------------------------------------------------------------------
            train_idx    = np.where(train_mask)[0]
            F_train_full = mean_L_frame[train_idx].astype(float)
            y_train_full = stats_ens.delta_E_frame[train_idx].astype(float)
            
            # guard: drop non-finite rows
            good_rows = np.isfinite(F_train_full).all(axis=1) & np.isfinite(y_train_full)
            F_train   = F_train_full[good_rows]
            y_train   = y_train_full[good_rows]
            
            # ----------------------------------------------------------------------
            # ❷  Fit λ and α² with (generalised) cross-validation
            # ----------------------------------------------------------------------
            alpha_sq, lambda_opt, _, _, L_chol = calibrate_alpha_reg_gcv(
                F_eval=F_train,
                y=y_train,
                lambda_bounds=(1e-6, 1e4)
            )
            
            assert np.isfinite(L_chol).all(), "L_chol still has NaNs/Infs!"

            # uncertainty for pool path -------------------------------------
            if n_models >= 2:
                calib_pool = VarianceScalingCalibrator().fit(
                                stats_ens.delta_E_frame[train_mask],
                                sigma_E_raw[train_mask])
                sigma_E_pool_cal = calib_pool.transform(sigma_E_pool)
            else:
                sigma_E_pool_cal = None  # latent path will compute its own σ
    
            # windowed pool AL ----------------------------------------------
            print(f"sigma_comp shape: {sigma_comp.shape}")
            print(f"mu_F_pool shape: {ens_F_pool.shape}")

            Sel_objs_thin, sel_rel_thin = adaptive_learning_mig_pool_windowed(
                pool_frames_thin,
                F_pool_thin,
                F_train_thin,
                alpha_sq,
                L_chol,
                forces_train  = all_true_F, 
                sigma_energy  = sigma_E_raw, 
                sigma_force   = sigma_comp, 
                mu_E_pool    = mu_E_pool_thin,
                sigma_E_pool = sigma_E_pool_thin,
                mu_F_pool    = mu_F_pool,
                sigma_F_pool = sigma_F_pool,
                bad_global   = bad_rel,
                rho_eV       = 0.002,
                min_k        = 5,
                window_size  = eval_cfg.get("pool_window", 100),
                base         = "al_pool_v1")
    
            # pool window log -----------------------------------------------
            n_thin = len(thin_idx)
            w_size = eval_cfg.get("pool_window", 100)
            n_win  = (n_thin + w_size - 1) // w_size
            with open("pool_window_log.txt", "w") as fh:
                fh.write(f"total_frames_original	{len(pool_frames)}")
                fh.write(f"thin_stride	{pool_stride}")
                fh.write(f"frames_after_thin	{n_thin}")
                fh.write(f"window_size	{w_size}")
                fh.write(f"num_windows	{n_win}")
            print("[Pool‑AL] wrote pool_window_log.txt")
    
            # Map thin indices back to original pool indices ----------------
            sel_global_idx = thin_idx[sel_rel_thin]
            sel_objs       = [pool_frames[i] for i in sel_global_idx]
    
            # Extend master data structures ---------------------------------
            if sel_objs:
                with open("to_DFT_labelling_from_pool.xyz", "w") as fh:
                    for orig_idx, atoms in zip(sel_global_idx, sel_objs):
                        e_pred = mu_E_pool[orig_idx]
                        comment = f"frame={orig_idx},  pred_E={e_pred:.5f}"
                        write(fh, atoms, format="xyz", comment=comment)
                print(f"Saved {len(sel_objs)} pool frames to 'to_DFT_labelling_from_pool.xyz'.")
            else:
                print("No pool frames selected; nothing to save.")

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

