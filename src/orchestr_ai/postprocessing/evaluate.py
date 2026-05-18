"""
evaluate.py

This module orchestrates the evaluation process for ML force-field models.
Refactored in 2025: Fully object-oriented, removing all legacy code.
"""

import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.serialization
torch.serialization.add_safe_globals([slice])
import matplotlib.pyplot as plt
from ase.io import read, write
from sklearn.isotonic import IsotonicRegression

# === Local Module Imports ===
from orchestr_ai.postprocessing.parsing import parse_extxyz, save_stacked_xyz_schnetpack
from orchestr_ai.postprocessing.calculator import evaluate_model
from orchestr_ai.postprocessing.stats import MLFFStats
from orchestr_ai.postprocessing.features import compute_features
from orchestr_ai.postprocessing.uq_metrics_calculator import calculate_uq_metrics
from orchestr_ai.postprocessing.mlff_plotting import plot_mlff_stats
from orchestr_ai.postprocessing.plotting import generate_uq_plots
# Active Learning & Geometry Sanity
from orchestr_ai.postprocessing.active_learning import (
    calibrate_alpha_reg_gcv, 
    adaptive_learning_mig_pool_windowed, 
    adaptive_learning_ensemble_calibrated
)
from orchestr_ai.postprocessing.rdf import (
    compute_rdf_thresholds_from_reference,
    fast_filter_by_rdf_kdtree,
    debug_plot_rdfs
)

def _split_atom_vectors(flat_vectors, frames):
    """Split a concatenated atom vector array into one array per frame."""
    arr = np.asarray(flat_vectors, dtype=float)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"Expected force components divisible by 3, got shape {arr.shape}")
        arr = arr.reshape(-1, 3)
    counts = [len(fr) for fr in frames]
    if arr.shape[0] != sum(counts):
        raise ValueError(f"Atom-vector length mismatch: got {arr.shape[0]}, expected {sum(counts)}")
    splits = np.cumsum(counts)[:-1]
    return [x.astype(float, copy=False) for x in np.split(arr, splits)]


def _force_summary_from_flat(force_vectors, frames):
    force_frames = _split_atom_vectors(force_vectors, frames)
    mean_norm = np.array(
        [
            np.nanmean(np.linalg.norm(f, axis=1)) if np.asarray(f).size else np.nan
            for f in force_frames
        ],
        dtype=float,
    )
    max_norm = np.array(
        [
            np.nanmax(np.linalg.norm(f, axis=1)) if np.asarray(f).size else np.nan
            for f in force_frames
        ],
        dtype=float,
    )
    return mean_norm, max_norm


def _std_from_sums(sum_values, sum_sq_values, n_samples):
    if n_samples <= 1:
        return np.zeros_like(sum_values, dtype=float)
    mean_values = sum_values / n_samples
    var = (sum_sq_values - n_samples * mean_values**2) / (n_samples - 1)
    return np.sqrt(np.maximum(var, 0.0))


def _parse_bool_like(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "auto"}
    return bool(value)


def _configured_gpu_ids(eval_cfg):
    requested = eval_cfg.get("inference_gpus", eval_cfg.get("devices", None))
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if visible <= 0:
        return []

    if requested is None or requested == "auto":
        return list(range(visible))

    if isinstance(requested, int):
        return list(range(min(requested, visible)))

    if isinstance(requested, str):
        values = [v.strip() for v in requested.split(",") if v.strip()]
        if len(values) == 1 and values[0].isdigit():
            return list(range(min(int(values[0]), visible)))
        return [int(v) for v in values if int(v) < visible]

    if isinstance(requested, (list, tuple)):
        return [int(v) for v in requested if int(v) < visible]

    return list(range(visible))


def _load_eval_model(model_path, framework, device):
    framework = (framework or "schnetpack").lower()
    if framework == "allegro":
        framework = "nequip"
    if framework == "nequip":
        return model_path
    return torch.load(model_path, map_location=device, weights_only=False)


def _evaluate_model_chunk_worker(payload):
    model_path, framework, config, frames, true_E, true_F, batch_size, gpu_id = payload
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    model_obj = _load_eval_model(model_path, framework, device)
    preds = evaluate_model(
        frames=frames,
        true_energies=true_E,
        true_forces=true_F,
        model_obj=model_obj,
        device=device,
        batch_size=batch_size,
        eval_log_file=None,
        config=config,
        neighbor_list=None,
    )

    del model_obj
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()

    return preds

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


class DatasetManager:
    """Handles loading, purging, and masking of Train and Validation datasets."""
    def __init__(self, config):
        self.eval_cfg = config.get("eval", {})
        self.train_path = self.eval_cfg.get("training_data")
        self.eval_path = self.eval_cfg.get("eval_input_xyz")
        
    def load_datasets(self):
        print("\n--- Setting up Datasets ---")
        assert self.eval_path and os.path.exists(self.eval_path), f"Eval file not found: {self.eval_path}"
        
        val_E, val_F, val_pos = parse_extxyz(self.eval_path, "eval")
        val_frames = read(self.eval_path, index=":", format="extxyz")
        
        train_frames, train_E, train_F, train_pos = [], [], [], []
        if self.train_path and os.path.exists(self.train_path):
            train_E, train_F, train_pos = parse_extxyz(self.train_path, "training_data")
            train_frames = read(self.train_path, index=":", format="extxyz")
            
            # Redundancy Purge
            eval_mask = []
            energy_tol, pos_tol = 0.0001, 0.0001
            for i, (e_eval, p_eval) in enumerate(zip(val_E, val_pos)):
                is_redundant = False
                e_eval_rounded = round(e_eval, 5)
                for j, (e_train, p_train) in enumerate(zip(train_E, train_pos)):
                    e_train_rounded = round(e_train, 5)
                    if abs(e_eval_rounded - e_train_rounded) < energy_tol:
                        if p_eval.shape[0] >= 3 and p_train.shape[0] >= 3:
                            if np.allclose(p_eval[:3], p_train[:3], atol=pos_tol):
                                print(f"Redundant structure found: Eval frame {i} is redundant with Training frame {j}.")
                                is_redundant = True
                                break
                eval_mask.append(not is_redundant)
            
            val_frames = [f for f, k in zip(val_frames, eval_mask) if k]
            val_E = [e for e, k in zip(val_E, eval_mask) if k]
            val_F = [f for f, k in zip(val_F, eval_mask) if k]
            print(f"Validation frames after purge: {len(val_frames)}")

        all_frames = train_frames + val_frames
        n_train, n_val = len(train_frames), len(val_frames)
        train_mask = np.array([True]*n_train + [False]*n_val, dtype=bool)
        val_mask = np.array([False]*n_train + [True]*n_val, dtype=bool)

        print(f"Total labeled frames: {len(all_frames)} (train={n_train}, val={n_val})")

        return {
            "frames": all_frames, "E_true": np.array(train_E + val_E), "F_true": train_F + val_F,
            "train_mask": train_mask, "val_mask": val_mask,
            "train_idx": np.where(train_mask)[0], "val_idx": np.where(val_mask)[0],
            "val_frames_ref": val_frames
        }


class EnsembleRunner:
    """Handles loading, running, aggregating, and caching ensemble ML predictions."""
    def __init__(self, config, device, neighbor_list):
        self.config = config
        self.device = device
        self.neighbor_list = neighbor_list
        self.eval_cfg = config.get("eval", {})
        self.ensemble_folder = self.eval_cfg.get("ensemble_folder")
        self.n_models = self.eval_cfg.get("ensemble_size", 1)
        self.batch_size = self.eval_cfg.get("batch_size", 32)
        self.framework = self.config.get("model_framework", "schnetpack").lower()

    def _model_paths(self):
        valid_extensions = (".pth", ".pt", ".nequip.pth", ".model")
        found_models = []

        if os.path.exists(self.ensemble_folder):
            for filename in sorted(os.listdir(self.ensemble_folder)):
                if filename.endswith(valid_extensions):
                    found_models.append(os.path.join(self.ensemble_folder, filename))
        else:
            print(f"[EnsembleRunner] ERROR: Ensemble folder '{self.ensemble_folder}' does not exist.")

        model_paths_to_run = found_models[:self.n_models]

        if not model_paths_to_run:
            print(f"[EnsembleRunner] WARNING: No models found in {self.ensemble_folder} with extensions {valid_extensions}")

        return model_paths_to_run

    def _multi_gpu_enabled(self):
        flag = self.eval_cfg.get("multi_gpu_inference", "auto")
        if isinstance(flag, str) and flag.strip().lower() == "auto":
            return torch.cuda.is_available() and torch.cuda.device_count() > 1
        return _parse_bool_like(flag, default=False)

    def _evaluate_model_single_gpu(
        self,
        model_path,
        frames,
        true_E=None,
        true_F=None,
        log_file=None,
    ):
        try:
            model_obj = _load_eval_model(model_path, self.framework, self.device)
        except Exception as e:
            print(f"     Failed to load {model_path}: {e}")
            return None

        return evaluate_model(
            frames=frames,
            true_energies=true_E,
            true_forces=true_F,
            model_obj=model_obj,
            device=self.device,
            batch_size=self.batch_size,
            eval_log_file=log_file,
            config=self.config,
            neighbor_list=self.neighbor_list,
        )

    def _evaluate_model_multi_gpu(
        self,
        model_path,
        frames,
        true_E=None,
        true_F=None,
        log_file=None,
    ):
        gpu_ids = _configured_gpu_ids(self.eval_cfg)
        if len(gpu_ids) <= 1:
            return self._evaluate_model_single_gpu(
                model_path,
                frames,
                true_E,
                true_F,
                log_file=log_file,
            )

        n_frames = len(frames)
        chunks = [
            idx
            for idx in np.array_split(np.arange(n_frames), len(gpu_ids))
            if len(idx) > 0
        ]

        print(
            f"     Multi-GPU inference: {len(chunks)} worker(s), "
            f"GPUs={gpu_ids[:len(chunks)]}, per-GPU batch_size={self.batch_size}"
        )

        payloads = []
        for gpu_id, idx in zip(gpu_ids, chunks):
            chunk_frames = [frames[i] for i in idx]
            chunk_true_E = None if true_E is None else np.asarray(true_E)[idx]
            chunk_true_F = None if true_F is None else [true_F[i] for i in idx]
            payloads.append(
                (
                    model_path,
                    self.framework,
                    self.config,
                    chunk_frames,
                    chunk_true_E,
                    chunk_true_F,
                    self.batch_size,
                    gpu_id,
                )
            )

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(payloads)) as pool:
            chunk_results = pool.map(_evaluate_model_chunk_worker, payloads)

        energy_pred = [None] * n_frames
        forces_pred = [None] * n_frames
        latent_frame = [None] * n_frames
        latent_atom = [None] * n_frames

        for idx, result in zip(chunks, chunk_results):
            e_chunk, f_chunk, lf_chunk, la_chunk = result
            for local_i, global_i in enumerate(idx):
                energy_pred[int(global_i)] = e_chunk[local_i]
                forces_pred[int(global_i)] = f_chunk[local_i]
                latent_frame[int(global_i)] = lf_chunk[local_i]
                latent_atom[int(global_i)] = la_chunk[local_i]

        return energy_pred, forces_pred, latent_frame, latent_atom

    def evaluate(self, frames, true_E=None, true_F=None, cache_file="ensemble_cache.npz"):
        if os.path.exists(cache_file):
            print(f"\n[EnsembleRunner] Loading cached predictions from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True)

            # --- Safely load ens_L_atom only if it exists in the cache ---
            ens_L_atom_cached = data["ens_L_atom"] if "ens_L_atom" in data else None
            return (data["ens_E"], data["ens_F"], data["ens_L_frame"], ens_L_atom_cached)

        print(f"\n[EnsembleRunner] Inference for {len(frames)} frames. Scanning for models...")
        ens_E, ens_F, ens_L_frame, ens_L_atom = [], [], [], []

        model_paths_to_run = self._model_paths()

        # --- 2. Load and evaluate the found models ---
        for m_idx, model_path in enumerate(model_paths_to_run):
            print(f"  -> Loading Model {m_idx+1}/{len(model_paths_to_run)}: {model_path}")

            if self._multi_gpu_enabled():
                preds = self._evaluate_model_multi_gpu(
                    model_path,
                    frames,
                    true_E=true_E,
                    true_F=true_F,
                )
            else:
                preds = self._evaluate_model_single_gpu(
                    model_path,
                    frames,
                    true_E=true_E,
                    true_F=true_F,
                )

            if preds is None:
                continue

            preds_E, preds_F, preds_L_frame, preds_L_atom = preds
            ens_E.append(preds_E)
            ens_F.append(preds_F)
            ens_L_frame.append(preds_L_frame)
            ens_L_atom.append(preds_L_atom)

        # If no models were successfully evaluated, return empty lists to trigger the ValueError upstream
        if not ens_E:
            return np.array([]), np.array([]), np.array([]), np.array([])

        ens_E = np.array(ens_E)
        ens_F = np.array(ens_F, dtype=object)
        ens_L_frame = np.array(ens_L_frame)
        ens_L_atom = np.array(ens_L_atom, dtype=object)

        print(f"[EnsembleRunner] Saving uncompressed cache to {cache_file} (omitting atom latents)...")
        np.savez(
            cache_file,
            ens_E=ens_E,
            ens_F=ens_F,
            ens_L_frame=ens_L_frame
        )
        return ens_E, ens_F, ens_L_frame, ens_L_atom

    def evaluate_stats(self, frames, true_E=None, true_F=None, cache_file="ensemble.npz"):
        if os.path.exists(cache_file):
            print(f"\n[EnsembleRunner] Loading aggregate cache from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True)
            if "cache_format" in data and str(data["cache_format"]) == "ensemble_stats_v1":
                return {key: data[key] for key in data.files if key != "cache_format"}
            print("[EnsembleRunner] Existing cache is not aggregate stats; rebuilding.")

        print(f"\n[EnsembleRunner] Aggregate inference for {len(frames)} labeled frames...")
        model_paths_to_run = self._model_paths()

        sum_E = sum_E2 = None
        sum_F = sum_F2 = None
        sum_L = sum_L2 = None
        n_models_done = 0

        for m_idx, model_path in enumerate(model_paths_to_run):
            print(f"  -> Aggregating Model {m_idx+1}/{len(model_paths_to_run)}: {model_path}")

            if self._multi_gpu_enabled():
                preds = self._evaluate_model_multi_gpu(
                    model_path,
                    frames,
                    true_E=true_E,
                    true_F=true_F,
                )
            else:
                preds = self._evaluate_model_single_gpu(
                    model_path,
                    frames,
                    true_E=true_E,
                    true_F=true_F,
                )

            if preds is None:
                continue

            preds_E, preds_F, preds_L_frame, _ = preds
            e = np.asarray(preds_E, dtype=float)
            f = np.concatenate(preds_F, axis=0).astype(float, copy=False)
            l_frame = np.asarray(preds_L_frame, dtype=float)

            if sum_E is None:
                sum_E = np.zeros_like(e, dtype=float)
                sum_E2 = np.zeros_like(e, dtype=float)
                sum_F = np.zeros_like(f, dtype=float)
                sum_F2 = np.zeros_like(f, dtype=float)
                sum_L = np.zeros_like(l_frame, dtype=float)
                sum_L2 = np.zeros_like(l_frame, dtype=float)

            sum_E += e
            sum_E2 += e**2
            sum_F += f
            sum_F2 += f**2
            sum_L += l_frame
            sum_L2 += l_frame**2
            n_models_done += 1

            del preds, preds_E, preds_F, preds_L_frame, e, f, l_frame

        if n_models_done == 0:
            raise ValueError("No ensemble models were successfully evaluated.")

        mu_E = sum_E / n_models_done
        sigma_E = _std_from_sums(sum_E, sum_E2, n_models_done)
        mu_F = sum_F / n_models_done
        sigma_F = _std_from_sums(sum_F, sum_F2, n_models_done)
        mu_L_frame = sum_L / n_models_done
        sigma_L_frame = _std_from_sums(sum_L, sum_L2, n_models_done)
        n_atoms_per_frame = np.array([len(fr) for fr in frames], dtype=int)

        result = {
            "n_models": np.array(n_models_done, dtype=int),
            "mu_E": mu_E,
            "sigma_E": sigma_E,
            "mu_F": mu_F,
            "sigma_F": sigma_F,
            "mu_L_frame": mu_L_frame,
            "sigma_L_frame": sigma_L_frame,
            "n_atoms_per_frame": n_atoms_per_frame,
        }

        print(f"[EnsembleRunner] Saving aggregate cache to {cache_file}...")
        np.savez_compressed(cache_file, cache_format="ensemble_stats_v1", **result)
        return result

    def evaluate_pool_light(self, frames, cache_file="ensemble_unlabel.npz"):
        if os.path.exists(cache_file):
            print(f"\n[EnsembleRunner] Loading light pool cache from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True)
            if "cache_format" in data and str(data["cache_format"]) == "ensemble_pool_light_v1":
                return {key: data[key] for key in data.files if key != "cache_format"}
            print("[EnsembleRunner] Existing pool cache is not light format; rebuilding.")

        print(f"\n[EnsembleRunner] Light aggregate inference for {len(frames)} pool frames...")
        stats = self.evaluate_stats(frames, cache_file=cache_file)
        sigma_F_mean, sigma_F_max = _force_summary_from_flat(stats["sigma_F"], frames)
        _, frame_max_force = _force_summary_from_flat(stats["mu_F"], frames)

        result = {
            "n_models": stats["n_models"],
            "mu_E": stats["mu_E"],
            "sigma_E": stats["sigma_E"],
            "mu_L_frame": stats["mu_L_frame"],
            "sigma_F_mean": sigma_F_mean,
            "sigma_F_max": sigma_F_max,
            "frame_max_force": frame_max_force,
            "n_atoms_per_frame": stats["n_atoms_per_frame"],
        }

        print(f"[EnsembleRunner] Saving light pool cache to {cache_file}...")
        np.savez_compressed(cache_file, cache_format="ensemble_pool_light_v1", **result)
        return result

def plot_ensemble_histograms(mu_E, std_E, mu_F, std_F, out_dir="uq_plots"):
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(mu_E, bins=50); axes[0, 0].set_title("Mean Energy per Frame")
    axes[0, 1].hist(std_E, bins=50, color='orange'); axes[0, 1].set_title("Energy Uncertainty (Std)")
    axes[1, 0].hist(mu_F, bins=100); axes[1, 0].set_title("Mean Force (Component)")
    axes[1, 1].hist(std_F, bins=100, color='orange'); axes[1, 1].set_title("Force Uncertainty (Std)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/ensemble_distributions.png", dpi=200)
    plt.close()


def _safe_load_model(model_path: str, device: torch.device, force_dtype=torch.float32):
    try:
        mdl = torch.load(model_path, map_location=device, weights_only=False)
    except AttributeError:
        mdl = torch.load(model_path, map_location=device)
    if force_dtype is not None:
        mdl = mdl.to(dtype=force_dtype)
    mdl.eval()
    return mdl


class EvaluationPipeline:
    """Orchestrates the full MLFF evaluation and Active Learning pipeline."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_cfg = config.get("eval", {})
        
        uq_methods = self.eval_cfg.get("uncertainty", ["none"])
        self.uq_methods = [uq_methods] if not isinstance(uq_methods, list) else uq_methods
        
        # Directories & Logs
        os.makedirs("diagnostics", exist_ok=True)
        os.makedirs("uq_plots", exist_ok=True)
        self.eval_log = self.eval_cfg.get("eval_log_file", "eval_log.txt")
        open(self.eval_log, "w").close() 
        
        framework = self.config.get("model_framework", "schnetpack").lower()

        if framework == "allegro":
            framework = "nequip"

        self.neighbour_list = None

        if framework == "schnetpack":
            from orchestr_ai.postprocessing.neighbor_list import setup_neighbor_list

            self.neighbour_list = setup_neighbor_list(config)

        self.do_plot = self.eval_cfg.get("plot", False)
        
        self.pool_xyz_path = self.eval_cfg.get("unlabeled_pool_path", None)
        self.al_val_flag = None if self.pool_xyz_path else self.eval_cfg.get("active_learning", None)

    def run(self):
        # 1. Load Data
        data_mgr = DatasetManager(self.config)
        self.ds = data_mgr.load_datasets()
        
        # 2. Evaluate Base Model (Single Model Fallback)
        if "none" in self.uq_methods or self.eval_cfg.get("error_estimate", False):
            self._run_base_model()
            
        # 3. Evaluate Ensemble & Run Active Learning
        if "ensemble" in self.uq_methods:
            stats_ens, mean_L_frame, sigma_comp, sigma_E_raw = self._run_ensemble_labeled()
            
            if self.al_val_flag and self.al_val_flag.lower() == "influence":
                self._run_validation_al(stats_ens, mean_L_frame, sigma_comp)
                
            if self.pool_xyz_path and os.path.exists(self.pool_xyz_path):
                self._run_pool_al(stats_ens, mean_L_frame, sigma_E_raw, sigma_comp)

        print("Evaluation Pipeline Completed.")

    def _run_base_model(self):
        """Runs the standard single-model evaluation."""
        base_path = self.config.get("model_path", "")
        if not base_path or not os.path.exists(base_path):
            print("Base model not found, skipping base evaluation.")
            return

        runner = EnsembleRunner(self.config, self.device, self.neighbour_list)
        if runner._multi_gpu_enabled():
            preds = runner._evaluate_model_multi_gpu(
                base_path,
                self.ds["frames"],
                true_E=list(self.ds["E_true"]),
                true_F=self.ds["F_true"],
                log_file=self.eval_log,
            )
        else:
            print(f"Loaded base model from {base_path}")
            preds = runner._evaluate_model_single_gpu(
                base_path,
                self.ds["frames"],
                true_E=list(self.ds["E_true"]),
                true_F=self.ds["F_true"],
                log_file=self.eval_log,
            )

        if preds is None:
            print("Base model inference failed, skipping base evaluation.")
            return

        pred_E, pred_F, _, _ = preds
        
        if isinstance(pred_F, np.ndarray) and pred_F.ndim == 3:
            pf_list, idx = [], 0
            for fr in self.ds["frames"]:
                pf_list.append(pred_F[idx:idx+len(fr)])
                idx += len(fr)
            pred_F = pf_list
            
        stats_base = MLFFStats(self.ds["E_true"], pred_E, self.ds["F_true"], pred_F, self.ds["train_mask"], self.ds["val_mask"])
        
        if "none" in self.uq_methods:
            print("\n--- Evaluating Base Model Performance ---")
            features_all, min_dists_all, _, _, _ = compute_features(
                self.ds["frames"], self.config, self.eval_cfg.get("training_data"), 
                self.ds["train_mask"], self.ds["val_mask"]
            )
            plot_mlff_stats(stats_base, min_dists_all, "validation_results_base", True, self.ds["train_mask"], self.ds["val_mask"])

    def _run_ensemble_labeled(self):
        """Runs the ensemble on labeled datasets and computes UQ metrics."""
        runner = EnsembleRunner(self.config, self.device, self.neighbour_list)
        cache_mode = str(self.eval_cfg.get("ensemble_cache_mode", "stats")).lower()

        if cache_mode == "raw":
            ens_E_sel, ens_F_list, ens_L_frame_sel, _ = runner.evaluate(
                self.ds["frames"], self.ds["E_true"], self.ds["F_true"], cache_file="ensemble.npz"
            )

            ens_F_sel = np.array([np.concatenate(m_forces, axis=0) for m_forces in ens_F_list], dtype=float)

            if ens_E_sel.shape[0] < 2:
                raise ValueError("Ensemble UQ requested, but fewer than 2 models were loaded.")

            mu_E_frame = np.mean(ens_E_sel, axis=0)
            std_E_frame = np.std(ens_E_sel, axis=0, ddof=0)
            sigma_E_raw = np.std(ens_E_sel, axis=0, ddof=1)
            mu_F_comp = np.mean(ens_F_sel, axis=0)
            sigma_F_flat = np.std(ens_F_sel, axis=0, ddof=1)
            mean_L_frame = np.mean(ens_L_frame_sel, axis=0)
        else:
            stats_cache = runner.evaluate_stats(
                self.ds["frames"], self.ds["E_true"], self.ds["F_true"], cache_file="ensemble.npz"
            )

            if int(stats_cache["n_models"]) < 2:
                raise ValueError("Ensemble UQ requested, but fewer than 2 models were loaded.")

            mu_E_frame = stats_cache["mu_E"]
            std_E_frame = stats_cache["sigma_E"]
            sigma_E_raw = stats_cache["sigma_E"]
            mu_F_comp = stats_cache["mu_F"]
            sigma_F_flat = stats_cache["sigma_F"]
            mean_L_frame = stats_cache["mu_L_frame"]

        std_F_comp = sigma_F_flat.flatten()

        # Build Stats Object
        mf_list, idx = [], 0
        for fr in self.ds["frames"]:
            mf_list.append(mu_F_comp[idx:idx+len(fr)])
            idx += len(fr)
        stats_ens = MLFFStats(self.ds["E_true"], mu_E_frame, self.ds["F_true"], mf_list, self.ds["train_mask"], self.ds["val_mask"])
       
        sigma_comp = sigma_F_flat.flatten()
        sigma_atom = np.linalg.norm(sigma_comp.reshape(-1, 3), axis=1)

        print("\n=== Ensemble Summary ===")
        print(f"Energy: mean={mu_E_frame.mean():.4f}, std={mu_E_frame.std():.4f}")
        print(f"Force : mean={mu_F_comp.mean():.4f}, std={std_F_comp.std():.4f}")

        if self.do_plot:
            plot_ensemble_histograms(mu_E_frame, std_E_frame, mu_F_comp, std_F_comp)

        metrics_train = calculate_uq_metrics(stats_ens, sigma_comp, sigma_atom, sigma_E_raw, "Train", "ensemble", self.eval_log)
        metrics_eval  = calculate_uq_metrics(stats_ens, sigma_comp, sigma_atom, sigma_E_raw, "Eval", "ensemble", self.eval_log)
        
        if self.do_plot:
            generate_uq_plots(metrics_train["npz_path"], "Train", "error_model", calibration="var")
            generate_uq_plots(metrics_eval["npz_path"], "Eval", "error_model", calibration="var")

        return stats_ens, mean_L_frame, sigma_comp, sigma_E_raw

    def _run_validation_al(self, stats_ens, mean_L_frame, sigma_comp):
        """Active Learning on the validation set."""
        print("\n[Val-AL] Running Influence-based Active Learning on Validation Set...")
        _, sel_idx = adaptive_learning_ensemble_calibrated(
            all_frames=self.ds["frames"], eval_mask=self.ds["val_mask"], 
            delta_E_frame=stats_ens.delta_E_frame, mean_l_al=mean_L_frame, 
            force_rmse_per_comp=sigma_comp, denom_all=self.ds["F_true"], 
            reference_frames=self.ds["val_frames_ref"], base="al_ens_val"
        )

        if len(sel_idx):
            val_pos = [self.ds["frames"][i].get_positions() for i in sel_idx]
            val_forces = [self.ds["F_true"][i] for i in sel_idx]
            val_energies = self.ds["E_true"][sel_idx]
            atom_types = [self.ds["frames"][i].get_chemical_symbols() for i in sel_idx]
            
            save_stacked_xyz_schnetpack("to_label_from_val.xyz", val_energies, val_pos, val_forces, atom_types)
            print(f"[Val-AL] Saved {len(sel_idx)} validation frames to 'to_label_from_val.xyz'.")
        else:
            print("[Val-AL] No validation frames selected.")

    def _run_pool_al(self, stats_ens, mean_L_frame, sigma_E_raw, sigma_comp):
        """Active Learning on the unlabelled out-of-distribution pool."""
        print(f"\n[Pool-AL] Parsing unlabeled pool from {self.pool_xyz_path}")
        pool_frames = read(self.pool_xyz_path, index=":", format="extxyz")

        runner = EnsembleRunner(self.config, self.device, self.neighbour_list)
        pool_cache_mode = str(self.eval_cfg.get("pool_cache_mode", "light")).lower()

        if pool_cache_mode == "raw":
            ens_E_pool, ens_F_pool_list, ens_L_pool, _ = runner.evaluate(pool_frames, cache_file="ensemble_unlabel.npz")

            ens_F_pool = np.array([np.concatenate(m_forces, axis=0) for m_forces in ens_F_pool_list], dtype=float)

            mu_E_pool = np.mean(ens_E_pool, axis=0)
            sigma_E_pool = np.std(ens_E_pool, axis=0, ddof=1)
            mu_F_pool = np.mean(ens_F_pool, axis=0)
            sigma_F_pool = np.std(ens_F_pool, axis=0, ddof=1)
            mu_L_pool = np.mean(ens_L_pool, axis=0)
            sigma_F_pool_mean, sigma_F_pool_max = _force_summary_from_flat(sigma_F_pool, pool_frames)
            _, frame_max_force_pool = _force_summary_from_flat(mu_F_pool, pool_frames)
        elif pool_cache_mode == "stats":
            pool_stats = runner.evaluate_stats(pool_frames, cache_file="ensemble_unlabel.npz")

            mu_E_pool = pool_stats["mu_E"]
            sigma_E_pool = pool_stats["sigma_E"]
            mu_F_pool = pool_stats["mu_F"]
            sigma_F_pool = pool_stats["sigma_F"]
            mu_L_pool = pool_stats["mu_L_frame"]
            sigma_F_pool_mean, sigma_F_pool_max = _force_summary_from_flat(sigma_F_pool, pool_frames)
            _, frame_max_force_pool = _force_summary_from_flat(mu_F_pool, pool_frames)
        else:
            pool_light = runner.evaluate_pool_light(pool_frames, cache_file="ensemble_unlabel.npz")

            mu_E_pool = pool_light["mu_E"]
            sigma_E_pool = pool_light["sigma_E"]
            mu_L_pool = pool_light["mu_L_frame"]
            sigma_F_pool_mean = pool_light["sigma_F_mean"]
            sigma_F_pool_max = pool_light["sigma_F_max"]
            frame_max_force_pool = pool_light["frame_max_force"]

        # Thinning
        thin_idx = np.arange(len(pool_frames))[::self.eval_cfg.get("pool_stride", 1)]
        pool_frames_thin = [pool_frames[i] for i in thin_idx]
        F_pool_thin = mu_L_pool[thin_idx].astype(float)
        mu_E_pool_thin = mu_E_pool[thin_idx].astype(float)
        sigma_E_pool_thin = sigma_E_pool[thin_idx].astype(float)
        F_train_thin = mean_L_frame[self.ds["train_idx"]].astype(float)
        sigma_F_pool_mean_thin = sigma_F_pool_mean[thin_idx].astype(float)
        sigma_F_pool_max_thin = sigma_F_pool_max[thin_idx].astype(float)
        frame_max_force_pool_thin = frame_max_force_pool[thin_idx].astype(float)

        # RDF filtering
        rdf_cache = "rdf_thresholds_cache.npz"
        if os.path.exists(rdf_cache):
            print(f"[Pool-AL] Loading cached RDF thresholds...")
            data = np.load(rdf_cache, allow_pickle=True)
            if "rdf_thresholds" in data:
                rdf_thresholds = data["rdf_thresholds"].item()
            else:
                rdf_thresholds = {(str(r[0]), str(r[1])): (float(r[2]), float(r[3])) for r in data["thresholds"]}
        else:
            print("[Pool-AL] Computing RDF thresholds from validation frames...")
            rdf_thresholds = compute_rdf_thresholds_from_reference(self.ds["val_frames_ref"], stride=self.eval_cfg.get("rdf_stride", 5))
            np.savez_compressed(rdf_cache, rdf_thresholds=rdf_thresholds)

        debug_plot_rdfs(self.ds["val_frames_ref"], rdf_thresholds)
        rdf_ok_mask = fast_filter_by_rdf_kdtree(pool_frames_thin, rdf_thresholds)

        # Energy Trace Logging
        import scipy.spatial.distance
        df = pd.DataFrame({"mu": mu_E_pool, "sigma": sigma_E_pool})
        sm = df.rolling(50, center=True, min_periods=1).mean()
        bad_mask = np.zeros(len(mu_E_pool), dtype=bool)
        
        # Calculate baseline diameter
        initial_pos = [pool_frames_thin[i].get_positions() for i in range(min(10, len(pool_frames_thin)))]
        max_diam = np.median([scipy.spatial.distance.pdist(p).max() for p in initial_pos]) * 1.5

        for k, orig_i in enumerate(thin_idx):
            if not rdf_ok_mask[k]: 
                bad_mask[orig_i] = True
            else:
                diam = scipy.spatial.distance.pdist(pool_frames_thin[k].get_positions()).max()
                if diam > max_diam:
                    bad_mask[orig_i] = True
            
        np.savez_compressed("pool_energy_trace.npz", steps=np.arange(len(mu_E_pool)), mu=sm["mu"].values, sigma=sm["sigma"].values, bad=bad_mask)

        # Calibrations
        good_rows = np.isfinite(F_train_thin).all(axis=1) & np.isfinite(stats_ens.delta_E_frame[self.ds["train_idx"]])
        print(f"[Pool-AL] Extracted {good_rows.sum()} valid training frames for GP calibration.")
        train_atom_counts = np.array([len(self.ds["frames"][i]) for i in self.ds["train_idx"]], dtype=float)
        train_delta_E_atom = stats_ens.delta_E_frame[self.ds["train_idx"]] / train_atom_counts
        alpha_sq, _, _, _, L_chol = calibrate_alpha_reg_gcv(F_train_thin[good_rows], train_delta_E_atom[good_rows])

        calibrator = UQCalibrator()
        mu_E_train = stats_ens.pred_energies[self.ds["train_idx"]]
        mu_E_train_atom = mu_E_train / train_atom_counts
        sigma_E_train_atom = sigma_E_raw[self.ds["train_idx"]] / train_atom_counts
        calibrator.fit(mu_E_train_atom, sigma_E_train_atom, train_delta_E_atom)
        if self.do_plot:
            calibrator.plot_diagnostics(mu_E_train_atom, sigma_E_train_atom, train_delta_E_atom)

        sigma_force_frames = _split_atom_vectors(sigma_comp, self.ds["frames"])
        train_forces = [self.ds["F_true"][i] for i in self.ds["train_idx"]]
        sigma_force_train = [sigma_force_frames[i] for i in self.ds["train_idx"]]

        # Selection
        print("[Pool-AL] Running windowed active learning on thinned pool ...")
        _, sel_rel_thin = adaptive_learning_mig_pool_windowed(
            pool_frames_thin, F_pool_thin, F_train_thin, alpha_sq, L_chol,
            forces_train=train_forces, sigma_energy=sigma_E_raw[self.ds["train_idx"]], sigma_force=sigma_force_train,
            mu_E_frame_train=mu_E_train, mu_E_pool=mu_E_pool_thin, sigma_E_pool=sigma_E_pool_thin,
            rdf_thresholds=rdf_thresholds,
            sigma_F_pool_mean=sigma_F_pool_mean_thin, sigma_F_pool_max=sigma_F_pool_max_thin,
            frame_max_force_pool=frame_max_force_pool_thin,
            rho_eV=self.eval_cfg.get("rho_eV", 0.002), min_k=self.eval_cfg.get("pool_min_k", 5),
            window_size=self.eval_cfg.get("pool_window", 100), budget_max=self.eval_cfg.get("budget_max", 50),
            percentile_gamma=self.eval_cfg.get("percentile_gamma", 100),
            percentile_F_low=self.eval_cfg.get("percentile_F_low", 99.5),
            percentile_F_hi=self.eval_cfg.get("percentile_F_hi", 93),
            hard_sigma_E_atom_min=self.eval_cfg.get("thr_sE_atom", 0.001),
            hard_sigma_F_mean_min=self.eval_cfg.get("thr_sF_mean", 0.1),
            hard_sigma_F_max_min=self.eval_cfg.get("thr_sF_max", 0.1),
            hard_Fmax_train_mult=self.eval_cfg.get("thr_Fmax_mult", 1.5)
        )

        # Output
        sel_global_idx = thin_idx[sel_rel_thin]
        if len(sel_global_idx) > 0:
            with open("to_DFT_labelling_from_pool.xyz", "w") as fh:
                for orig_idx in sel_global_idx:
                    atoms = pool_frames[orig_idx]
                    e_raw, s_raw = float(mu_E_pool[orig_idx]), float(sigma_E_pool[orig_idx])
                    _, s_cal_arr, bias_corr_arr = calibrator.calibrate(np.array([e_raw]), np.array([s_raw]))
                    e_cal, s_cal = e_raw - float(bias_corr_arr[0]), float(s_cal_arr[0])
                    comment = f"frame={orig_idx}, e_pred_raw={e_raw:.6f}, s_raw={s_raw:.6f}, bias_corr={-float(bias_corr_arr[0]):.6f}, s_calibrated={s_cal:.6f}, BALLPARK=[{e_cal - s_cal:.6f}, {e_cal + s_cal:.6f}]"
                    write(fh, atoms, format="xyz", comment=comment)
            print(f"[Pool-AL] Saved {len(sel_global_idx)} pool frames to 'to_DFT_labelling_from_pool.xyz'.")


def run_eval(config):
    """Entry point for evaluation."""
    if config is None:
        print("Error: Config not loaded.")
        return
    pipeline = EvaluationPipeline(config)
    pipeline.run()
