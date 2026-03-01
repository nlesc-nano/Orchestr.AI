"""
evaluate.py

This module orchestrates the evaluation process for ML force-field models.
Refactored in 2025: Fully object-oriented, removing all legacy code.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ase.io import read, write
from sklearn.isotonic import IsotonicRegression

# === Local Module Imports ===
from mlff_qd.postprocessing.parsing import parse_extxyz, save_stacked_xyz_schnetpack
from mlff_qd.postprocessing.calculator import setup_neighbor_list, evaluate_model
from mlff_qd.postprocessing.stats import MLFFStats
from mlff_qd.postprocessing.features import compute_features
from mlff_qd.postprocessing.uq_metrics_calculator import calculate_uq_metrics
from mlff_qd.postprocessing.mlff_plotting import plot_mlff_stats
from mlff_qd.postprocessing.plotting import generate_uq_plots

# Active Learning & Geometry Sanity
from mlff_qd.postprocessing.active_learning import (
    calibrate_alpha_reg_gcv, 
    adaptive_learning_mig_pool_windowed, 
    adaptive_learning_ensemble_calibrated
)
from mlff_qd.postprocessing.rdf import (
    compute_rdf_thresholds_from_reference,
    fast_filter_by_rdf_kdtree,
    debug_plot_rdfs
)


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

        try:
            forces_train_arr = np.stack(train_F + val_F, axis=0).astype(float)
        except Exception:
            forces_train_arr = None

        print(f"Total labeled frames: {len(all_frames)} (train={n_train}, val={n_val})")

        return {
            "frames": all_frames, "E_true": np.array(train_E + val_E), "F_true": train_F + val_F,
            "F_train_arr": forces_train_arr, "train_mask": train_mask, "val_mask": val_mask,
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
        
    def evaluate(self, frames, true_E=None, true_F=None, cache_file="ensemble_cache.npz"):
        if os.path.exists(cache_file):
            print(f"\n[EnsembleRunner] Loading cached predictions from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True)
            return (data["ens_E"], data["ens_F"], data["ens_L_frame"], data["ens_L_atom"])
            
        print(f"\n[EnsembleRunner] Inference for {len(frames)} frames across {self.n_models} models...")
        ens_E, ens_F, ens_L_frame, ens_L_atom = [], [], [], []

        for m_idx in range(self.n_models):
            # Try both with and without the underscore before the number
            base_paths = [
                os.path.join(self.ensemble_folder, f"best_model_{m_idx+1}"),
                os.path.join(self.ensemble_folder, f"best_model{m_idx+1}")
            ]
            
            # Auto-detect extension
            model_path = None
            for base_path in base_paths:
                for ext in ["", ".pth", ".pt", ".nequip.pth"]:
                    if os.path.exists(base_path + ext):
                        model_path = base_path + ext
                        break
                if model_path:
                    break

            if model_path is None:
                print(f"     Failed to find model {m_idx+1}: {base_model_path}(.pth/.pt)")
                continue

            print(f"  -> Model {m_idx+1}/{self.n_models}: {model_path}")
            try:
                # --- NEW LOGIC: Pass the path directly if NequIP ---
                if self.config.get("model_framework", "schnetpack").lower() == "nequip":
                    model_obj = model_path
                else:
                    model_obj = torch.load(model_path, map_location=self.device, weights_only=False)
            except Exception as e:
                print(f"     Failed to load {model_path}: {e}")
                continue
                
            preds_E, preds_F, preds_L_frame, preds_L_atom = evaluate_model(
                frames=frames, true_energies=true_E, true_forces=true_F,
                model_obj=model_obj, device=self.device, batch_size=self.batch_size,
                eval_log_file=None, config=self.config, neighbor_list=self.neighbor_list
            )
            ens_E.append(preds_E)
            ens_F.append(preds_F)
            ens_L_frame.append(preds_L_frame)
            ens_L_atom.append(preds_L_atom)
            
        ens_E = np.array(ens_E)
        ens_F = np.array(ens_F, dtype=object)
        ens_L_frame = np.array(ens_L_frame)
        ens_L_atom = np.array(ens_L_atom, dtype=object)
        
        print(f"[EnsembleRunner] Saving compressed cache to {cache_file}...")
        np.savez_compressed(
            cache_file, 
            ens_E=ens_E, ens_F=ens_F, 
            ens_L_frame=ens_L_frame, ens_L_atom=ens_L_atom
        )
        return ens_E, ens_F, ens_L_frame, ens_L_atom


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

        framework = self.config.get("model_framework", "schnetpack").lower()
        if framework == "nequip":
            base_model = base_path
        else:
            base_model = _safe_load_model(base_path, device=self.device)
        print(f"Loaded base model from {base_path}")
        
        pred_E, pred_F, _, _ = evaluate_model(
            self.ds["frames"], list(self.ds["E_true"]), self.ds["F_true"],
            base_model, self.device, self.eval_cfg.get("batch_size", 32),
            log_path=self.eval_log, config=self.config, neighbor_list=self.neighbour_list
        )
        
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
        ens_E_sel, ens_F_list, ens_L_frame_sel, _ = runner.evaluate(
            self.ds["frames"], self.ds["E_true"], self.ds["F_true"], cache_file="ensemble.npz"
        )
        
        ens_F_sel = np.array([np.concatenate(m_forces, axis=0) for m_forces in ens_F_list], dtype=float)
        
        if ens_E_sel.shape[0] < 2:
            raise ValueError("Ensemble UQ requested, but fewer than 2 models were loaded.")

        # Compute Stats
        mu_E_frame = np.mean(ens_E_sel, axis=0)
        std_E_frame = np.std(ens_E_sel, axis=0, ddof=0)
        sigma_E_raw = np.std(ens_E_sel, axis=0, ddof=1)
        mu_F_comp = np.mean(ens_F_sel, axis=0)
        sigma_F_flat = np.std(ens_F_sel, axis=0, ddof=1)
        std_F_comp = sigma_F_flat.flatten()
        mean_L_frame = np.mean(ens_L_frame_sel, axis=0)

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
            val_pos = np.array([self.ds["frames"][i].get_positions() for i in sel_idx])
            val_forces = np.stack([self.ds["F_true"][i] for i in sel_idx])
            val_energies = self.ds["E_true"][sel_idx]
            atom_types = self.ds["frames"][sel_idx[0]].get_chemical_symbols()
            
            save_stacked_xyz_schnetpack("to_label_from_val.xyz", val_energies, val_pos, val_forces, atom_types)
            print(f"[Val-AL] Saved {len(sel_idx)} validation frames to 'to_label_from_val.xyz'.")
        else:
            print("[Val-AL] No validation frames selected.")

    def _run_pool_al(self, stats_ens, mean_L_frame, sigma_E_raw, sigma_comp):
        """Active Learning on the unlabelled out-of-distribution pool."""
        print(f"\n[Pool-AL] Parsing unlabeled pool from {self.pool_xyz_path}")
        pool_frames = read(self.pool_xyz_path, index=":", format="extxyz")

        runner = EnsembleRunner(self.config, self.device, self.neighbour_list)
        ens_E_pool, ens_F_pool_list, ens_L_pool, _ = runner.evaluate(pool_frames, cache_file="ensemble_unlabel.npz")

        ens_F_pool = np.array([np.concatenate(m_forces, axis=0) for m_forces in ens_F_pool_list], dtype=float)

        mu_E_pool = np.mean(ens_E_pool, axis=0)
        sigma_E_pool = np.std(ens_E_pool, axis=0, ddof=1)
        mu_F_pool = np.mean(ens_F_pool, axis=0)
        sigma_F_pool = np.std(ens_F_pool, axis=0, ddof=1)
        mu_L_pool = np.mean(ens_L_pool, axis=0)

        # Thinning
        thin_idx = np.arange(len(pool_frames))[::self.eval_cfg.get("pool_stride", 1)]
        pool_frames_thin = [pool_frames[i] for i in thin_idx]
        F_pool_thin = mu_L_pool[thin_idx].astype(float)
        mu_E_pool_thin = mu_E_pool[thin_idx].astype(float)
        sigma_E_pool_thin = sigma_E_pool[thin_idx].astype(float)
        F_train_thin = mean_L_frame[self.ds["train_idx"]].astype(float)

        # Thin the forces (reshape to 3D, slice, then pass)
        n_atoms_pool = len(pool_frames[0])
        mu_F_pool_thin = mu_F_pool.reshape(len(pool_frames), n_atoms_pool, 3)[thin_idx]
        sigma_F_pool_thin = sigma_F_pool.reshape(len(pool_frames), n_atoms_pool, 3)[thin_idx]

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
        df = pd.DataFrame({"mu": mu_E_pool, "sigma": sigma_E_pool})
        sm = df.rolling(50, center=True, min_periods=1).mean()
        bad_mask = np.zeros(len(mu_E_pool), dtype=bool)
        for k, orig_i in enumerate(thin_idx):
            if not rdf_ok_mask[k]: bad_mask[orig_i] = True
            
        np.savez_compressed("pool_energy_trace.npz", steps=np.arange(len(mu_E_pool)), mu=sm["mu"].values, sigma=sm["sigma"].values, bad=bad_mask)

        # Calibrations
        good_rows = np.isfinite(F_train_thin).all(axis=1) & np.isfinite(stats_ens.delta_E_frame[self.ds["train_idx"]])
        print(f"[Pool-AL] Extracted {good_rows.sum()} valid training frames for GP calibration.")
        alpha_sq, _, _, _, L_chol = calibrate_alpha_reg_gcv(F_train_thin[good_rows], stats_ens.delta_E_frame[self.ds["train_idx"]][good_rows])

        calibrator = UQCalibrator()
        mu_E_train = stats_ens.pred_energies[self.ds["train_idx"]]
        calibrator.fit(mu_E_train, sigma_E_raw[self.ds["train_idx"]], stats_ens.delta_E_frame[self.ds["train_idx"]])
        if self.do_plot: calibrator.plot_diagnostics(mu_E_train, sigma_E_raw[self.ds["train_idx"]], stats_ens.delta_E_frame[self.ds["train_idx"]])

        # Selection
        print("[Pool-AL] Running windowed active learning on thinned pool ...")
        _, sel_rel_thin = adaptive_learning_mig_pool_windowed(
            pool_frames_thin, F_pool_thin, F_train_thin, alpha_sq, L_chol,
            forces_train=self.ds["F_train_arr"], sigma_energy=sigma_E_raw, sigma_force=sigma_comp,
            mu_E_frame_train=mu_E_train, mu_E_pool=mu_E_pool_thin, sigma_E_pool=sigma_E_pool_thin,
            mu_F_pool=mu_F_pool_thin, sigma_F_pool=sigma_F_pool_thin, rdf_thresholds=rdf_thresholds,
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

