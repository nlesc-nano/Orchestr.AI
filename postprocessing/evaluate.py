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

from ase.io import read  # Needed for reading training/eval data

# === Local Module Imports ===
from postprocessing.parsing import parse_extxyz, save_stacked_xyz
from postprocessing.calculator import setup_neighbor_list, assign_charges, evaluate_model
from postprocessing.stats import MLFFStats
from postprocessing.features import compute_features
from postprocessing.uq_models import train_uq_models, predict_uncertainties, fit_gmm_and_compute_uncertainty
#from postprocessing.uq_metrics_calculator import calculate_uq_metrics
from postprocessing.uq_metrics_calculator import run_uq_metrics, calculate_uq_metrics
from postprocessing.mlff_plotting import plot_mlff_stats
from postprocessing.plotting import (
    generate_uq_plots, 
    generate_al_influence_plots, generate_al_traditional_plots
)
from postprocessing.active_learning import adaptive_learning, adaptive_learning_mig_calibrated, adaptive_learning_mig_pool,calibrate_alpha_reg_gcv, predict_sigma_from_L, adaptive_learning_mig_pool_windowed, filter_unrealistic_indices, compute_bond_thresholds 


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
    tuple(np.ndarray, np.ndarray, np.ndarray, list[torch.nn.Module] | None)
        ``(energy_preds, force_preds, latent_preds, model_list_or_None)``
            * energy_preds : (n_models, n_frames)
            * force_preds  : (n_models, n_atoms_total, 3)
            * latent_preds : (n_models, n_frames, latent_dim)
            * model_list   : list of loaded models, or ``None`` if we
              loaded the *.npz* from disk.
    """

    # ------------------------------------------------------------------
    # 1. Fast‑path: load from cache
    # ------------------------------------------------------------------
    if os.path.isfile(npz_path):
        try:
            with np.load(npz_path, allow_pickle=True) as npz:
                return (npz["ensemble_energy_preds"],
                        npz["ensemble_force_preds"],
                        npz["ensemble_latent_preds"],
                        None)
        except Exception as e:
            print(f"[Ensemble‑cache] Failed to load '{npz_path}': {e}. Recomputing…")

    # ------------------------------------------------------------------
    # 2. Gather model files
    # ------------------------------------------------------------------
    model_files = []
    # 1) match by pattern (now '*' picks everything)
    for pat in model_glob:
        for f in glob.glob(os.path.join(model_folder, pat)):
            if os.path.isfile(f):
                model_files.append(f)
    # 2) fallback: any file at all
    if not model_files:
        for entry in os.listdir(model_folder):
            fp = os.path.join(model_folder, entry)
            if os.path.isfile(fp):
                model_files.append(fp)
    model_files = sorted(model_files)
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_folder}")

    n_frames = len(frames)
    print(f"[Ensemble] evaluating {len(model_files)} models on {n_frames} frames…")

    # ------------------------------------------------------------------
    # 3. Run each model
    # ------------------------------------------------------------------

    energy_list   = []   # each (n_frames,)
    force_list    = []   # each (n_atoms_total,3)
    latent_list   = []   # each (n_frames, latent_dim)
    loaded_models = []

    dummy_E = [0.0] * n_frames         # we do not need truth for inference
    dummy_F = [None]  * n_frames

    total_atoms = sum(len(fr) for fr in frames)

    for idx, mfile in enumerate(model_files, 1):
        try:
            mdl = torch.load(mfile, map_location=device).to(torch.float32)
            mdl.eval()
            loaded_models.append(mdl)

            pred_E, pred_F_per_frame, latent_per_frame, _, _ = evaluate_model(
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

            # ---- sanity & reshape -----------------------------------
            pred_E = np.asarray(pred_E, dtype=np.float64)  # (n_frames,)
            if pred_E.shape[0] != n_frames:
                raise ValueError("Energy array length mismatch.")
            energy_list.append(pred_E)

            # Forces: concatenate per‑frame arrays              (n_atoms_total,3)
            flat_F = np.concatenate(pred_F_per_frame, axis=0)
            if flat_F.shape != (total_atoms, 3):
                raise ValueError("Flattened force array has wrong shape.")
            force_list.append(flat_F)

            # Latent: stack to (n_frames, latent_dim)
            if latent_per_frame is None or len(latent_per_frame) != n_frames:
                raise ValueError("Latent array length mismatch.")
            latent_list.append(np.stack(latent_per_frame))

            print(f"  [{idx:>3}/{len(model_files)}] {os.path.basename(mfile)} done")

        except Exception as e:
            print(f"  [Warning] model '{mfile}' failed: {e}")
            traceback.print_exc()

    if not energy_list:
        raise RuntimeError("All ensemble evaluations failed.")

    # ------------------------------------------------------------------
    # 4. Stack & save
    # ------------------------------------------------------------------
    arr_E = np.stack(energy_list)                  # (n_models,n_frames)
    arr_F = np.stack(force_list)                  # (n_models,n_atoms_total,3)
    arr_L = np.stack(latent_list)                 # (n_models,n_frames,latent_dim)

    try:
        np.savez_compressed(npz_path,
                            ensemble_energy_preds=arr_E,
                            ensemble_force_preds=arr_F,
                            ensemble_latent_preds=arr_L)
        print(f"[Ensemble‑cache] saved → {npz_path}")
    except Exception as e:
        print(f"[Warning] Could not save ensemble cache '{npz_path}': {e}")

    return arr_E, arr_F, arr_L, loaded_models

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
    ensemble_models = None
    alpha_sq = None
    L_chol = None
    mean_L_al = None
    if "ensemble" in uq_methods and ensemble_folder:
        if os.path.isdir(ensemble_folder):
            ens_E, ens_F, ens_L, ensemble_models = evaluate_and_cache_ensemble(
                all_frames,
                ensemble_folder,
                device,
                batch_size,
                eval_log,
                config,
                neighbour_list,
                npz_path="ensemble.npz"
            )
            n_models = ens_E.shape[0]
            if n_models >= 2:
                mu_E = np.mean(ens_E, axis=0)
                sigma_E = np.std(ens_E, axis=0, ddof=1)
                mu_F_flat = np.mean(ens_F, axis=0)
                sigma_comp = np.std(ens_F, axis=0, ddof=1).flatten()
                sigma_atom = np.linalg.norm(sigma_comp.reshape(-1,3), axis=1)
                mf_list = []
                idx = 0
                for fr in all_frames:
                    n = len(fr)
                    mf_list.append(mu_F_flat[idx:idx+n])
                    idx += n
                stats_ens = MLFFStats(all_true_E, mu_E, all_true_F, mf_list, train_mask, val_mask)
                if do_plot:
                    features_all, min_dists_all, _, pca, scaler = compute_features(
                        all_frames, config, train_xyz_path, train_mask, val_mask)
                    plot_mlff_stats(stats_ens, min_dists_all, "validation_results_ensemble", True, train_mask, val_mask)
                metrics_train = calculate_uq_metrics(stats_ens, sigma_comp, sigma_atom, sigma_E, "Train", "ensemble", eval_log)
                if do_plot:
                    generate_uq_plots(metrics_train["npz_path"], "Train", "error_model")
                metrics_eval = calculate_uq_metrics(stats_ens, sigma_comp, sigma_atom, sigma_E, "Eval", "ensemble", eval_log)
                if do_plot:
                    generate_uq_plots(metrics_eval["npz_path"], "Eval", "error_model")
                mean_L_al = np.mean(ens_L, axis=0)
                F_train_lat = mean_L_al[~val_mask]
                F_val_lat   = mean_L_al[val_mask]
                y_val       = stats_ens.delta_E_frame[val_mask]
                # calibrate
                alpha_sq, lam, terms_lat, G_eval, L_chol = calibrate_alpha_reg_gcv(F_val_lat, y_val)
                # validation AL if no pool
                if al_val_flag and al_val_flag.lower() in ["influence","traditional"]:
                    if al_val_flag.lower() == "influence":
                        sel_objs, sel_idx = adaptive_learning_mig_calibrated(
                            all_frames, val_mask, sigma_E, stats_ens.delta_E_frame, mean_L_al,
                            target_rmse_conv=0.015,      # ← 10 meV / atom
                            beta=0.5,
                            drop_init=0.6,
                            min_k=5,
                            max_k=250,                  # hard upper limit
                            score_floor=None,           # let the code compute it
                            base="al_mig_val")
                    else:
                        train_atom_mask = stats_ens._get_atom_mask(train_mask)
                        sel_objs, sel_idx, _ = adaptive_learning(
                            all_frames, val_mask,
                            sigma_atom, stats_ens.force_rmse_per_atom,
                            train_atom_mask, None,
                            eval_cfg.get("num_active_frames",50),
                            base_filename="al_traditional_val"
                        )
                    train_mask[sel_idx] = True
                    val_mask[sel_idx]   = False
                     # Save selected validation frames using save_stacked_xyz
                    if sel_idx is not None and len(sel_idx) > 0:
                        val_positions = np.array([all_frames[i].get_positions() for i in sel_idx])
                        val_forces_arr = np.stack([all_true_F[i] for i in sel_idx])
                        val_energies = all_true_E[sel_idx]
                        atom_types = all_frames[sel_idx[0]].get_chemical_symbols() if hasattr(all_frames[sel_idx[0]], 'get_chemical_symbols') else []
                        save_stacked_xyz("to_label_from_val.xyz", val_energies, val_positions, val_forces_arr, atom_types)
                        print(f"Saved {len(sel_idx)} validation frames for labeling to 'to_label_from_val.xyz'.")
                    else:
                        print("No validation frames selected; nothing to save.")

                # 8) Unlabeled pool AL
                # 8) Unlabeled pool AL
                # Unlabeled pool AL
                if pool_xyz_path and os.path.exists(pool_xyz_path):
                    print(f"[Pool-AL] parsing unlabeled pool from {pool_xyz_path}")
                    pool_E, pool_F, pool_pos = parse_extxyz(pool_xyz_path, "unlabeled_pool")
                    pool_frames = read(pool_xyz_path, index=":", format="extxyz")
            
                    # Ensemble evaluation
                    ens_E_pool, ens_F_pool, ens_L_pool, _ = evaluate_and_cache_ensemble(
                        pool_frames, ensemble_folder, device, batch_size,
                        eval_log, config, neighbour_list,
                        npz_path="ensemble_unlabel.npz"
                    )
                    mu_L_pool = np.mean(ens_L_pool, axis=0)
                    mu_E_pool = np.mean(ens_E_pool, axis=0)
                    sigma_E_pool = np.std(ens_E_pool, axis=0)
            
                    # Plot smoothed energy ± uncertainty
                    steps = np.arange(len(mu_E_pool))
                    window = 50
                    df = pd.DataFrame({"mu": mu_E_pool, "sigma": sigma_E_pool})
                    sm = df.rolling(window, center=True, min_periods=1).mean()
            
                    plt.figure(figsize=(6,4))
                    plt.plot(steps, sm["mu"], label=f"{window}-pt MA of μE")
                    plt.fill_between(
                        steps,
                        sm["mu"] - 2 * sm["sigma"],
                        sm["mu"] + 2 * sm["sigma"],
                        alpha=0.3,
                        label="σ (smoothed)"
                    )
            
                    # Compute bond thresholds from training set
                    train_idx = np.where(train_mask)[0]
                    train_frames = [all_frames[i] for i in train_idx]
                    thresholds = compute_bond_thresholds(
                        train_frames,
                        neighbour_list,
                        first_shell_cutoff=3.4
                    )
            
                    # Decimate for AL
                    pool_stride = eval_cfg.get("pool_stride", 50)
                    all_idx = np.arange(len(mu_E_pool))
                    thin_idx = all_idx[::pool_stride]
                    pool_frames_thin = [pool_frames[i] for i in thin_idx]
                    F_pool_thin = mu_L_pool[thin_idx]
                    mu_E_pool_thin = mu_E_pool[thin_idx]
            
                    # Identify bad frames by bond-lengths
                    bad_global = filter_unrealistic_indices(pool_frames, neighbour_list, thresholds)
                    bad_thin = bad_global.intersection(set(thin_idx))
                    bad_mask = np.isin(steps, list(bad_thin))
                    plt.scatter(
                        steps[bad_mask],
                        sm["mu"][bad_mask],
                        marker='x',
                        label='bond outlier'
                    )
                    plt.legend()
            
                    plt.xlabel("Pool frame index")
                    plt.ylabel("Predicted energy")
                    plt.title("Pool energy ± uncertainty with bond outliers")
                    plt.tight_layout()
                    plt.savefig("pool_energy_uncertainty.png", dpi=200)
                    plt.close()
            
                    # Compute σ on training set (for AL sigmacap)
                    F_train = mean_L_al[~val_mask]
                    L_chol = L_chol
                    alpha_sq = alpha_sq
            
                    # Windowed pool AL with precomputed bad_global
                    Sel_objs_thin, sel_rel_thin = adaptive_learning_mig_pool_windowed(
                        pool_frames_thin,
                        F_pool_thin,
                        F_train,
                        alpha_sq,
                        L_chol,
                        mu_E_pool=mu_E_pool_thin,
                        thresholds=thresholds,
                        neighbor_list=neighbour_list,
                        bad_global=bad_thin,  # Pass bad_thin as bad_global
                        rho_eV=0.0025,
                        beta=0.0,
                        drop_init=1.0,
                        min_k=5,
                        window_size=eval_cfg.get("pool_window", 100),
                        base="al_mig_pool_v3"
                    )
            
                    # Window stats
                    n_thin = len(thin_idx)
                    w_size = eval_cfg.get("pool_window", 100)
                    n_win = (n_thin + w_size - 1) // w_size
                    with open("pool_window_log.txt", "w") as fh:
                        fh.write(f"total_frames_original\t{len(pool_frames)}\n")
                        fh.write(f"thin_stride\t{pool_stride}\n")
                        fh.write(f"frames_after_thin\t{n_thin}\n")
                        fh.write(f"window_size\t{w_size}\n")
                        fh.write(f"num_windows\t{n_win}\n")
                    print("[Pool-AL] wrote pool_window_log.txt")
            
                    # Map back and extend
                    sel_rel = thin_idx[sel_rel_thin].tolist()
                    sel_objs = [pool_frames[i] for i in sel_rel]
                    offset = len(all_frames)
                    sel_global = [offset + i for i in sel_rel]
                    all_frames.extend(sel_objs)
                    all_true_E = np.concatenate([all_true_E, np.full(len(sel_objs), np.nan)])
                    all_true_F.extend([None]*len(sel_objs))
                    train_mask = np.concatenate([train_mask, np.ones(len(sel_objs), dtype=bool)])
                    val_mask = np.concatenate([val_mask, np.zeros(len(sel_objs), dtype=bool)])
            
                    if sel_objs:
                        pool_positions = np.array([frame.get_positions() for frame in sel_objs])
                        pool_forces_arr = np.full(pool_positions.shape, np.nan)
                        pool_energies = np.full(len(sel_objs), np.nan)
                        atom_types = sel_objs[0].get_chemical_symbols() if hasattr(sel_objs[0], 'get_chemical_symbols') else []
                        save_stacked_xyz("to_label_from_pool.xyz", pool_energies, pool_positions, pool_forces_arr, atom_types)
                        print(f"Saved {len(sel_objs)} pool frames for labeling to 'to_label_from_pool.xyz'.")
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

