"""
evaluate.py

This module orchestrates the evaluation process for ML force-field models.
Refactored in 2025: Fully object-oriented, removing all legacy code.
"""

import os
import time
import shutil
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.serialization
torch.serialization.add_safe_globals([slice])
import matplotlib.pyplot as plt
from ase.io import read, write

# === Local Module Imports ===
from orchestr_ai.postprocessing.metrics import (
    _split_atom_vectors,
    _force_summary_from_flat,
    _std_from_sums
)
from orchestr_ai.postprocessing.parsing import parse_extxyz, save_stacked_xyz_schnetpack
from orchestr_ai.postprocessing.calculator import evaluate_model
from orchestr_ai.postprocessing.stats import MLFFStats
from orchestr_ai.postprocessing.features import compute_features
from orchestr_ai.postprocessing.uq_metrics_calculator import (
    VarianceScalingCalibrator,
    calculate_uq_metrics,
)
from orchestr_ai.postprocessing.mlff_plotting import plot_mlff_stats
from orchestr_ai.postprocessing.plotting import generate_uq_plots
# Active Learning & Geometry Sanity
from orchestr_ai.postprocessing.active_learning import (
    calibrate_alpha_reg_gcv,
    adaptive_learning_mig_pool_windowed,
    adaptive_learning_ensemble_calibrated,
    UQCalibrator,
    compute_soap_features,
    apply_sigma_comp_calibration,
    apply_sigma_energy_calibration,
    calibrate_sigma_force_frames,
    scale_pool_force_summaries,
    validate_consecutive_reference_deltas,
    write_pool_al_diagnostics_csv,
)
from orchestr_ai.postprocessing.plots.al_diagnostics import generate_al_diagnostic_plots
from orchestr_ai.postprocessing.rdf import (
    compute_rdf_thresholds_from_reference,
    fast_filter_by_rdf_kdtree,
    debug_plot_rdfs
)

def _parse_bool_like(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "auto"}
    return bool(value)


_GAUSSIAN_SIGMA_TO_ABS = float(np.sqrt(2.0 / np.pi))


def _metric_accepts_calibration(metrics_result, *, prefix, mode, min_spearman, max_ence, picp_tol):
    """Validate train-fitted calibration on eval metrics before pool use."""
    if not metrics_result:
        return False
    metrics = metrics_result.get("metrics", {})
    suffix = "" if prefix == "force" else "_E"
    label = "calVAR" if mode == "var" else "calISO"
    ence = metrics.get(f"ENCE_{label}{suffix}")
    picp95 = metrics.get(f"PICP95_{label}{suffix}")
    spearman = metrics.get(f"Spearman_{label}{suffix}")
    checks = [
        ence is not None and np.isfinite(ence) and ence <= max_ence,
        picp95 is not None and np.isfinite(picp95) and abs(picp95 - 0.95) <= picp_tol,
        spearman is not None and np.isfinite(spearman) and spearman >= min_spearman,
    ]
    return all(checks)


DEFAULT_MACE_HEADS_MAP = {
    "singlet": {"energy_key": "E_singlet", "forces_key": "f_singlet"},
    "triplet": {"energy_key": "E_triplet", "forces_key": "f_triplet"},
    "triplet_reconstructed": {"energy_key": "E_triplet", "forces_key": "f_triplet"}
}


def _build_calibration_policy(eval_cfg, metrics_eval, *, pool_cache_mode):
    requested = str(eval_cfg.get("selection_calibration", "var")).lower()
    if requested not in {"var", "iso"}:
        requested = "var"
    min_sp = float(eval_cfg.get("calibration_min_spearman", 0.4))
    max_ence = float(eval_cfg.get("calibration_max_ence", 0.20))
    picp_tol = float(eval_cfg.get("calibration_picp95_tol", 0.08))
    force_mode = requested
    if requested == "iso" and pool_cache_mode == "light":
        print("[Pool-AL] Light pool cache cannot apply component-wise isotonic force calibration; using VAR for forces.")
        force_mode = "var"

    accepted = {
        "force_mode": force_mode if _metric_accepts_calibration(
            metrics_eval, prefix="force", mode=force_mode, min_spearman=min_sp,
            max_ence=max_ence, picp_tol=picp_tol
        ) else None,
        "energy_mode": requested if _metric_accepts_calibration(
            metrics_eval, prefix="energy", mode=requested, min_spearman=min_sp,
            max_ence=max_ence, picp_tol=picp_tol
        ) else None,
    }
    print(
        "[Pool-AL] Eval-gated calibration policy: "
        f"force={accepted['force_mode'] or 'raw'}, energy={accepted['energy_mode'] or 'raw'}"
    )
    _print_calibration_policy_summary(
        metrics_eval,
        accepted,
        min_spearman=min_sp,
        max_ence=max_ence,
        picp_tol=picp_tol,
        pool_cache_mode=pool_cache_mode,
    )
    return accepted


def _print_calibration_policy_summary(
    metrics_result,
    accepted,
    *,
    min_spearman,
    max_ence,
    picp_tol,
    pool_cache_mode,
):
    """Print the eval UQ evidence used to choose AL calibration."""
    if not metrics_result:
        print("[Pool-AL] Calibration summary unavailable: no eval UQ metrics.")
        return

    metrics = metrics_result.get("metrics", {})
    if not metrics:
        print("[Pool-AL] Calibration summary unavailable: empty eval UQ metrics.")
        return

    print("[Pool-AL] Calibration quality gates for active learning:")
    print(
        f"          Spearman >= {min_spearman:.3f}, "
        f"ENCE <= {max_ence:.3f}, |PICP95 - 0.95| <= {picp_tol:.3f}; "
        f"pool_cache_mode={pool_cache_mode}"
    )

    def _fmt(value):
        if value is None or not np.isfinite(value):
            return "   n/a"
        return f"{float(value):7.4f}"

    def _row(prefix, label, suffix=""):
        ence = metrics.get(f"ENCE_{prefix}{suffix}")
        picp = metrics.get(f"PICP95_{prefix}{suffix}")
        spearman = metrics.get(f"Spearman_{prefix}{suffix}")
        nll = metrics.get(f"NLL_{prefix}{suffix}")
        ks = metrics.get(f"KS_z_{prefix}{suffix}")
        return (
            f"          {label:<6} "
            f"ENCE={_fmt(ence)}  PICP95={_fmt(picp)}  "
            f"Spearman={_fmt(spearman)}  NLL={_fmt(nll)}  KS_z={_fmt(ks)}"
        )

    print("[Pool-AL] Force calibration metrics on eval split:")
    print(_row("raw", "raw"))
    print(_row("calVAR", "var"))
    print(_row("calISO", "iso"))
    print(f"          -> selected for AL forces: {accepted.get('force_mode') or 'raw'}")

    has_energy_metrics = any(key.endswith("_E") for key in metrics)
    if has_energy_metrics:
        print("[Pool-AL] Energy calibration metrics on eval split:")
        print(_row("raw", "raw", "_E"))
        print(_row("calVAR", "var", "_E"))
        print(_row("calISO", "iso", "_E"))
        print(f"          -> selected for AL energy: {accepted.get('energy_mode') or 'raw'}")

    for quantity, mode in (("forces", accepted.get("force_mode")), ("energy", accepted.get("energy_mode"))):
        if mode is None:
            print(f"[Pool-AL] {quantity.capitalize()} AL selection will use raw uncertainties because requested calibration failed gates or is unavailable.")
        elif mode == "var":
            print(f"[Pool-AL] {quantity.capitalize()} AL selection will use variance-scaled uncertainties; ranking is preserved, scale is corrected.")
        elif mode == "iso":
            print(f"[Pool-AL] {quantity.capitalize()} AL selection will use isotonic-on-variance calibration in support, with VAR fallback where configured.")


def _range_with_margin(values, *, lo_pct=0.5, hi_pct=99.5, upper_mult=1.25):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -np.inf, np.inf
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))
    if hi >= 0:
        hi *= upper_mult
    return lo, hi


def _in_range(values, bounds):
    lo, hi = bounds
    arr = np.asarray(values, dtype=float)
    return np.isfinite(arr) & (arr >= lo) & (arr <= hi)


def _coerce_frame_latents(latents, n_frames, *, context="latents"):
    """Return a 2D latent matrix, replacing failed-frame placeholders with NaNs."""
    latent_list = list(latents) if latents is not None else []
    if len(latent_list) != n_frames:
        print(f"[EnsembleRunner] WARNING: {context} length {len(latent_list)} != {n_frames}; padding/truncating.")
        latent_list = (latent_list + [None] * n_frames)[:n_frames]

    latent_dim = None
    for item in latent_list:
        arr = np.asarray(item, dtype=float)
        if arr.ndim == 1 and arr.size > 0 and np.isfinite(arr).any():
            latent_dim = int(arr.size)
            break
    if latent_dim is None:
        print(f"[EnsembleRunner] WARNING: No valid {context}; using one NaN latent column.")
        latent_dim = 1

    out = np.full((n_frames, latent_dim), np.nan, dtype=float)
    n_bad = 0
    for i, item in enumerate(latent_list):
        arr = np.asarray(item, dtype=float)
        if arr.ndim == 1 and arr.size == latent_dim:
            out[i] = arr
        else:
            n_bad += 1
    if n_bad:
        print(f"[EnsembleRunner] WARNING: Replaced {n_bad} malformed {context} rows with NaNs.")
    return out


def _multi_gpu_frame_chunks(frames, n_workers, strategy="atom_balanced"):
    """Return global frame-index chunks for multi-GPU inference."""
    n_frames = len(frames)
    if n_workers <= 1:
        return [np.arange(n_frames, dtype=int)]

    strategy = str(strategy or "atom_balanced").strip().lower()
    if strategy in {"contiguous", "split", "array_split"}:
        return [
            idx.astype(int)
            for idx in np.array_split(np.arange(n_frames), n_workers)
            if len(idx) > 0
        ]

    atom_counts = np.array([max(1, len(fr)) for fr in frames], dtype=float)
    costs = atom_counts**2
    chunks = [[] for _ in range(n_workers)]
    loads = np.zeros(n_workers, dtype=float)

    for frame_idx in np.argsort(costs)[::-1]:
        worker_idx = int(np.argmin(loads))
        chunks[worker_idx].append(int(frame_idx))
        loads[worker_idx] += costs[frame_idx]

    out = []
    for chunk in chunks:
        if chunk:
            out.append(np.array(sorted(chunk), dtype=int))
    return out


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


_WORKER_GPU_ID = None
_RESIDENT_MODELS = {}


def _init_persistent_worker(gpu_queue):
    global _WORKER_GPU_ID, _RESIDENT_MODELS
    try:
        _WORKER_GPU_ID = gpu_queue.get()
        _RESIDENT_MODELS = {}
        import torch
        device = torch.device(f"cuda:{_WORKER_GPU_ID}")
        torch.cuda.set_device(device)
        print(f"[Persistent Worker] Initialized worker process on GPU {_WORKER_GPU_ID}")
    except Exception as e:
        print(f"[Persistent Worker] Error during worker initialization: {e}")


def _evaluate_model_chunk_worker(payload):
    include_multihead = False
    if len(payload) == 12:
        model_path, framework, config, frames, true_E, true_F, batch_size, gpu_id, frame_indices, E_singlet_true, E_triplet_true, include_multihead = payload
    elif len(payload) == 11:
        model_path, framework, config, frames, true_E, true_F, batch_size, gpu_id, frame_indices, E_singlet_true, E_triplet_true = payload
    else:
        model_path, framework, config, frames, true_E, true_F, batch_size, gpu_id, frame_indices = payload
        E_singlet_true, E_triplet_true = None, None
    
    global _WORKER_GPU_ID, _RESIDENT_MODELS
    if _WORKER_GPU_ID is not None:
        gpu_id = _WORKER_GPU_ID

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    keep_resident = _parse_bool_like(
        config.get("eval", {}).get("keep_worker_models_resident", False),
        default=False,
    )
    if not keep_resident:
        for cached_path in list(_RESIDENT_MODELS.keys()):
            if cached_path != model_path:
                del _RESIDENT_MODELS[cached_path]
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

    if model_path not in _RESIDENT_MODELS:
        if frame_indices is not None and len(frame_indices):
            frame_range = f"{int(frame_indices[0])}-{int(frame_indices[-1])}"
        else:
            frame_range = "unknown"
        print(
            f"[Worker GPU {gpu_id}] Loading model into GPU memory: "
            f"{os.path.basename(model_path)} | global frames {frame_range}"
        )
        _RESIDENT_MODELS[model_path] = _load_eval_model(model_path, framework, device)

    model_obj = _RESIDENT_MODELS[model_path]
    context_label = f"{os.path.basename(model_path)}|GPU{gpu_id}"
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
        frame_indices=frame_indices,
        context_label=context_label,
        E_singlet_true=E_singlet_true,
        E_triplet_true=E_triplet_true,
        include_multihead=include_multihead,
    )

    if not keep_resident:
        del _RESIDENT_MODELS[model_path]
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

    return preds

class DatasetManager:
    """Handles loading, purging, and masking of Train and Validation datasets."""
    def __init__(self, config):
        self.config = config
        self.eval_cfg = config.get("eval", {})
        self.train_path = self.eval_cfg.get("training_data")
        self.validation_path = self.eval_cfg.get("validation_data")
        if not self.validation_path:
            self.validation_path = self.eval_cfg.get("eval_input_xyz")
            if self.validation_path:
                print("[Eval] 'eval_input_xyz' is deprecated; use 'validation_data' instead.")
        self.eval_path = self.validation_path

    def _head_keys(self):
        mace_heads_map = DEFAULT_MACE_HEADS_MAP.copy()
        custom_heads = self.eval_cfg.get("mace_heads", self.config.get("mace_heads", {}))
        for head_name, head_cfg in custom_heads.items():
            if head_name not in mace_heads_map:
                mace_heads_map[head_name] = {}
            mace_heads_map[head_name].update(head_cfg)

        mace_head = self.config.get("mace_head", None)
        energy_key = None
        forces_key = None
        if mace_head:
            head_config = mace_heads_map.get(mace_head, {})
            energy_key = head_config.get("energy_key", f"E_{mace_head}")
            forces_key = head_config.get("forces_key", f"f_{mace_head}")

        singlet_cfg = mace_heads_map.get("singlet", {})
        triplet_cfg = mace_heads_map.get("triplet", {})
        return (
            energy_key,
            forces_key,
            singlet_cfg.get("energy_key", "E_singlet"),
            singlet_cfg.get("forces_key", "f_singlet"),
            triplet_cfg.get("energy_key", "E_triplet"),
            triplet_cfg.get("forces_key", "f_triplet"),
        )

    @staticmethod
    def _structure_key(energy, positions, symbols, decimals=4):
        pos = np.asarray(positions, dtype=float)
        return (
            tuple(symbols),
            round(float(energy), decimals) if np.isfinite(energy) else None,
            tuple(np.round(pos.reshape(-1), decimals)),
        )

    def _purge_redundant_validation(
        self,
        val_frames,
        val_E,
        val_F,
        val_E_singlet,
        val_F_singlet,
        val_E_triplet,
        val_F_triplet,
        train_E,
        train_pos,
        train_frames,
    ):
        if not _parse_bool_like(self.eval_cfg.get("purge_redundant_validation", True), default=True):
            return val_frames, val_E, val_F, val_E_singlet, val_F_singlet, val_E_triplet, val_F_triplet

        train_keys = {
            self._structure_key(e, p, fr.get_chemical_symbols())
            for e, p, fr in zip(train_E, train_pos, train_frames)
        }
        keep = []
        removed = []
        for i, (e, fr) in enumerate(zip(val_E, val_frames)):
            key = self._structure_key(e, fr.get_positions(), fr.get_chemical_symbols())
            redundant = key in train_keys
            keep.append(not redundant)
            if redundant:
                removed.append(i)

        print("[Dataset] Redundant validation purge enabled.")
        print(f"[Dataset] Training frames: {len(train_frames)}")
        print(f"[Dataset] Validation frames before purge: {len(val_frames)}")
        print(f"[Dataset] Removed redundant validation frames: {len(removed)}")
        if removed:
            with open("redundant_validation_frames.txt", "w") as fh:
                fh.write("# validation_frame_index redundant_with_training\n")
                for idx in removed:
                    fh.write(f"{idx} 1\n")

        val_frames = [f for f, k in zip(val_frames, keep) if k]
        val_E = [e for e, k in zip(val_E, keep) if k]
        val_F = [f for f, k in zip(val_F, keep) if k]
        if val_E_singlet and len(val_E_singlet) == len(keep):
            val_E_singlet = [e for e, k in zip(val_E_singlet, keep) if k]
        if val_F_singlet and len(val_F_singlet) == len(keep):
            val_F_singlet = [f for f, k in zip(val_F_singlet, keep) if k]
        if val_E_triplet and len(val_E_triplet) == len(keep):
            val_E_triplet = [e for e, k in zip(val_E_triplet, keep) if k]
        if val_F_triplet and len(val_F_triplet) == len(keep):
            val_F_triplet = [f for f, k in zip(val_F_triplet, keep) if k]
        print(f"[Dataset] Validation frames after purge: {len(val_frames)}")
        return val_frames, val_E, val_F, val_E_singlet, val_F_singlet, val_E_triplet, val_F_triplet

    def load_datasets(self):
        print("\n--- Setting up Datasets ---")
        assert self.eval_path and os.path.exists(self.eval_path), f"Eval file not found: {self.eval_path}"

        energy_key, forces_key, s_e_key, s_f_key, t_e_key, t_f_key = self._head_keys()
        val_E, val_F, val_pos = parse_extxyz(self.eval_path, "eval", energy_key=energy_key, forces_key=forces_key)
        val_E_singlet, val_F_singlet, _ = parse_extxyz(self.eval_path, "eval_singlet", energy_key=s_e_key, forces_key=s_f_key)
        val_E_triplet, val_F_triplet, _ = parse_extxyz(self.eval_path, "eval_triplet", energy_key=t_e_key, forces_key=t_f_key)
        
        val_frames = read(self.eval_path, index=":", format="extxyz")
        
        train_frames, train_E, train_F, train_pos = [], [], [], []
        train_E_singlet, train_E_triplet = [], []
        train_F_singlet, train_F_triplet = [], []
        if self.train_path and os.path.exists(self.train_path):
            train_E, train_F, train_pos = parse_extxyz(self.train_path, "training_data", energy_key=energy_key, forces_key=forces_key)
            train_E_singlet, train_F_singlet, _ = parse_extxyz(self.train_path, "training_singlet", energy_key=s_e_key, forces_key=s_f_key)
            train_E_triplet, train_F_triplet, _ = parse_extxyz(self.train_path, "training_triplet", energy_key=t_e_key, forces_key=t_f_key)
            train_frames = read(self.train_path, index=":", format="extxyz")
            
            (
                val_frames, val_E, val_F,
                val_E_singlet, val_F_singlet,
                val_E_triplet, val_F_triplet,
            ) = self._purge_redundant_validation(
                val_frames, val_E, val_F,
                val_E_singlet, val_F_singlet,
                val_E_triplet, val_F_triplet,
                train_E, train_pos, train_frames,
            )

        all_frames = train_frames + val_frames
        n_train, n_val = len(train_frames), len(val_frames)
        train_mask = np.array([True]*n_train + [False]*n_val, dtype=bool)
        val_mask = np.array([False]*n_train + [True]*n_val, dtype=bool)

        print(f"Total labeled frames: {len(all_frames)} (train={n_train}, val={n_val})")

        return {
            "frames": all_frames, "E_true": np.array(train_E + val_E), "F_true": train_F + val_F,
            "train_mask": train_mask, "val_mask": val_mask,
            "train_idx": np.where(train_mask)[0], "val_idx": np.where(val_mask)[0],
            "val_frames_ref": val_frames,
            "E_singlet_true": np.array(train_E_singlet + val_E_singlet) if (train_E_singlet or val_E_singlet) else None,
            "F_singlet_true": train_F_singlet + val_F_singlet if (train_F_singlet or val_F_singlet) else None,
            "E_triplet_true": np.array(train_E_triplet + val_E_triplet) if (train_E_triplet or val_E_triplet) else None,
            "F_triplet_true": train_F_triplet + val_F_triplet if (train_F_triplet or val_F_triplet) else None,
        }

    def load_active_learning_datasets(self):
        print("\n--- Setting up Active Learning Calibration Dataset ---")
        if not self.validation_path or not os.path.exists(self.validation_path):
            raise ValueError("eval.mode='active_learning' requires eval.validation_data (or legacy eval.eval_input_xyz).")
        pool_path = self.eval_cfg.get("unlabeled_pool_path")
        if not pool_path or not os.path.exists(pool_path):
            raise ValueError("eval.mode='active_learning' requires eval.unlabeled_pool_path.")

        purge = _parse_bool_like(self.eval_cfg.get("purge_redundant_validation", True), default=True)
        if purge and (not self.train_path or not os.path.exists(self.train_path)):
            raise ValueError("purge_redundant_validation=true requires eval.training_data for active_learning mode.")

        calibration_source = str(self.eval_cfg.get("calibration_source", "validation")).lower()
        if calibration_source != "validation":
            raise ValueError("active_learning mode currently supports calibration_source='validation'.")

        energy_key, forces_key, s_e_key, s_f_key, t_e_key, t_f_key = self._head_keys()
        val_E, val_F, _ = parse_extxyz(self.validation_path, "validation", energy_key=energy_key, forces_key=forces_key)
        val_E_singlet, val_F_singlet, _ = parse_extxyz(self.validation_path, "validation_singlet", energy_key=s_e_key, forces_key=s_f_key)
        val_E_triplet, val_F_triplet, _ = parse_extxyz(self.validation_path, "validation_triplet", energy_key=t_e_key, forces_key=t_f_key)
        val_frames = read(self.validation_path, index=":", format="extxyz")

        if purge:
            train_E, _, train_pos = parse_extxyz(self.train_path, "training_data", energy_key=energy_key, forces_key=forces_key)
            train_frames = read(self.train_path, index=":", format="extxyz")
            (
                val_frames, val_E, val_F,
                val_E_singlet, val_F_singlet,
                val_E_triplet, val_F_triplet,
            ) = self._purge_redundant_validation(
                val_frames, val_E, val_F,
                val_E_singlet, val_F_singlet,
                val_E_triplet, val_F_triplet,
                train_E, train_pos, train_frames,
            )

        n_val = len(val_frames)
        if n_val < 2:
            raise ValueError("active_learning mode requires at least two non-redundant validation frames for calibration.")

        train_mask = np.ones(n_val, dtype=bool)
        val_mask = np.zeros(n_val, dtype=bool)
        print(f"[Dataset] Calibration source: validation ({n_val} frames).")
        print("[Dataset] Active-learning thresholds and calibration will be based on validation data.")
        return {
            "frames": val_frames,
            "E_true": np.array(val_E),
            "F_true": val_F,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "train_idx": np.arange(n_val, dtype=int),
            "val_idx": np.array([], dtype=int),
            "calibration_idx": np.arange(n_val, dtype=int),
            "validation_idx": np.arange(n_val, dtype=int),
            "val_frames_ref": val_frames,
            "E_singlet_true": np.array(val_E_singlet) if val_E_singlet else None,
            "F_singlet_true": val_F_singlet if val_F_singlet else None,
            "E_triplet_true": np.array(val_E_triplet) if val_E_triplet else None,
            "F_triplet_true": val_F_triplet if val_F_triplet else None,
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

    def _force_recompute(self):
        return _parse_bool_like(self.eval_cfg.get("ensemble_force_recompute", False), default=False)

    @staticmethod
    def _model_key(model_path):
        try:
            stat = os.stat(model_path)
            return f"{os.path.abspath(model_path)}|{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            return os.path.abspath(model_path)

    @staticmethod
    def _atomic_savez(path, compressed=True, **arrays):
        tmp_path = f"{path}.tmp.npz"
        if compressed:
            np.savez_compressed(tmp_path, **arrays)
        else:
            np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)

    def _clear_resume_artifacts(self, cache_file):
        checkpoint_file = f"{cache_file}.checkpoint.npz"
        parts_dir = f"{cache_file}.parts"
        for path in (cache_file, checkpoint_file):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                print(f"[EnsembleRunner] WARNING: Could not remove {path}: {exc}")
        if os.path.isdir(parts_dir):
            try:
                shutil.rmtree(parts_dir)
            except OSError as exc:
                print(f"[EnsembleRunner] WARNING: Could not remove {parts_dir}: {exc}")

    def _evaluate_model_single_gpu(
        self,
        model_path,
        frames,
        true_E=None,
        true_F=None,
        log_file=None,
        frame_indices=None,
        E_singlet_true=None,
        E_triplet_true=None,
        include_multihead=False,
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
            frame_indices=frame_indices,
            context_label=f"{os.path.basename(model_path)}|single",
            E_singlet_true=E_singlet_true,
            E_triplet_true=E_triplet_true,
            include_multihead=include_multihead,
        )

    def _evaluate_model_multi_gpu(
        self,
        model_path,
        frames,
        true_E=None,
        true_F=None,
        log_file=None,
        pool=None,
        E_singlet_true=None,
        E_triplet_true=None,
        include_multihead=False,
    ):
        gpu_ids = _configured_gpu_ids(self.eval_cfg)
        if len(gpu_ids) <= 1:
            return self._evaluate_model_single_gpu(
                model_path,
                frames,
                true_E,
                true_F,
                log_file=log_file,
                frame_indices=np.arange(len(frames), dtype=int),
                E_singlet_true=E_singlet_true,
                E_triplet_true=E_triplet_true,
                include_multihead=include_multihead,
            )

        n_frames = len(frames)
        chunk_strategy = self.eval_cfg.get("multi_gpu_chunk_strategy", "atom_balanced")
        chunks = _multi_gpu_frame_chunks(frames, len(gpu_ids), strategy=chunk_strategy)

        print(
            f"     Multi-GPU inference: {len(chunks)} worker(s), "
            f"GPUs={gpu_ids[:len(chunks)]}, per-GPU batch_size={self.batch_size}, "
            f"chunk_strategy={chunk_strategy}"
        )
        assigned = np.concatenate(chunks) if chunks else np.array([], dtype=int)
        unique_assigned = np.unique(assigned)
        overlap = int(len(assigned) - len(unique_assigned))
        missing = int(n_frames - len(unique_assigned))
        print(
            f"       Chunk coverage: assigned={len(assigned)}, unique={len(unique_assigned)}, "
            f"missing={missing}, overlap={overlap}. "
            "Atom-balanced chunks are non-contiguous by design."
        )

        payloads = []
        for gpu_id, idx in zip(gpu_ids, chunks):
            chunk_frames = [frames[i] for i in idx]
            chunk_true_E = None if true_E is None else np.asarray(true_E)[idx]
            chunk_true_F = None if true_F is None else [true_F[i] for i in idx]
            chunk_E_singlet = None if E_singlet_true is None else np.asarray(E_singlet_true)[idx]
            chunk_E_triplet = None if E_triplet_true is None else np.asarray(E_triplet_true)[idx]
            chunk_cost = int(np.sum([max(1, len(frames[i])) ** 2 for i in idx]))
            preview = ",".join(str(int(i)) for i in idx[:6])
            tail = ",".join(str(int(i)) for i in idx[-3:])
            print(
                f"       Planned chunk for GPU {gpu_id}: n={len(idx)}, "
                f"min={int(idx.min())}, max={int(idx.max())}, cost~{chunk_cost}, "
                f"sample=[{preview}...{tail}]"
            )
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
                    idx.astype(int),
                    chunk_E_singlet,
                    chunk_E_triplet,
                    include_multihead,
                )
            )

        if pool is not None:
            chunk_results = pool.map(_evaluate_model_chunk_worker, payloads)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(payloads)) as pool_temp:
                chunk_results = pool_temp.map(_evaluate_model_chunk_worker, payloads)

        energy_pred = [None] * n_frames
        forces_pred = [None] * n_frames
        latent_frame = [None] * n_frames
        latent_atom = [None] * n_frames
        mh_by_head = {}

        for idx, result in zip(chunks, chunk_results):
            if include_multihead:
                e_chunk, f_chunk, lf_chunk, la_chunk, mh_chunk = result
            else:
                e_chunk, f_chunk, lf_chunk, la_chunk = result
                mh_chunk = None
            for local_i, global_i in enumerate(idx):
                energy_pred[int(global_i)] = e_chunk[local_i]
                forces_pred[int(global_i)] = f_chunk[local_i]
                latent_frame[int(global_i)] = lf_chunk[local_i]
                latent_atom[int(global_i)] = la_chunk[local_i]
            if mh_chunk:
                for head, values in mh_chunk.items():
                    slot = mh_by_head.setdefault(
                        head,
                        {"energy": [None] * n_frames, "forces": [None] * n_frames},
                    )
                    for local_i, global_i in enumerate(idx):
                        slot["energy"][int(global_i)] = values["energy"][local_i]
                        slot["forces"][int(global_i)] = values["forces"][local_i]

        if include_multihead:
            mh_out = {
                head: {
                    "energy": np.asarray(values["energy"], dtype=float),
                    "forces": values["forces"],
                }
                for head, values in mh_by_head.items()
            }
            return energy_pred, forces_pred, latent_frame, latent_atom, mh_out
        return energy_pred, forces_pred, latent_frame, latent_atom

    def evaluate(self, frames, true_E=None, true_F=None, cache_file="ensemble_cache.npz", E_singlet_true=None, E_triplet_true=None):
        force_recompute = self._force_recompute()
        if force_recompute:
            print(f"\n[EnsembleRunner] ensemble_force_recompute=true; rebuilding {cache_file} from scratch.")
            self._clear_resume_artifacts(cache_file)

        if not force_recompute and os.path.exists(cache_file):
            print(f"\n[EnsembleRunner] Loading cached predictions from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True)

            # --- Safely load ens_L_atom only if it exists in the cache ---
            ens_L_atom_cached = data["ens_L_atom"] if "ens_L_atom" in data else None
            return (data["ens_E"], data["ens_F"], data["ens_L_frame"], ens_L_atom_cached)

        print(f"\n[EnsembleRunner] Inference for {len(frames)} frames. Scanning for models...")
        ens_E, ens_F, ens_L_frame, ens_L_atom = [], [], [], []

        model_paths_to_run = self._model_paths()
        parts_dir = f"{cache_file}.parts"
        os.makedirs(parts_dir, exist_ok=True)

        # Initialize persistent worker pool if multi-GPU is enabled
        pool = None
        if self._multi_gpu_enabled() and len(model_paths_to_run) > 0:
            gpu_ids = _configured_gpu_ids(self.eval_cfg)
            if len(gpu_ids) > 1:
                ctx = mp.get_context("spawn")
                gpu_queue = ctx.SimpleQueue()
                for gid in gpu_ids:
                    gpu_queue.put(gid)
                
                print(f"[EnsembleRunner] Spawning persistent worker pool with {len(gpu_ids)} GPU(s): {gpu_ids}")
                pool = ctx.Pool(
                    processes=len(gpu_ids),
                    initializer=_init_persistent_worker,
                    initargs=(gpu_queue,)
                )

        try:
            # --- 2. Load and evaluate the found models ---
            for m_idx, model_path in enumerate(model_paths_to_run):
                model_key = self._model_key(model_path)
                part_file = os.path.join(parts_dir, f"model_{m_idx:04d}.npz")
                if not force_recompute and os.path.exists(part_file):
                    try:
                        part = np.load(part_file, allow_pickle=True)
                        cached_key = str(part["model_key"]) if "model_key" in part else ""
                        if cached_key == model_key:
                            print(f"  -> Reusing cached raw predictions for Model {m_idx+1}/{len(model_paths_to_run)}: {model_path}")
                            ens_E.append(part["E"])
                            ens_F.append(list(part["F"]))
                            ens_L_frame.append(part["L_frame"])
                            ens_L_atom.append(None)
                            continue
                        print(f"  -> Raw part cache mismatch for Model {m_idx+1}; recomputing.")
                    except Exception as exc:
                        print(f"  -> Failed to load raw part cache {part_file}: {exc}; recomputing.")

                print(f"  -> Loading Model {m_idx+1}/{len(model_paths_to_run)}: {model_path}")

                if self._multi_gpu_enabled():
                    preds = self._evaluate_model_multi_gpu(
                        model_path,
                        frames,
                        true_E=true_E,
                        true_F=true_F,
                        pool=pool,
                        E_singlet_true=E_singlet_true,
                        E_triplet_true=E_triplet_true,
                    )
                else:
                    preds = self._evaluate_model_single_gpu(
                        model_path,
                        frames,
                        true_E=true_E,
                        true_F=true_F,
                        E_singlet_true=E_singlet_true,
                        E_triplet_true=E_triplet_true,
                    )

                if preds is None:
                    continue

                preds_E, preds_F, preds_L_frame, preds_L_atom = preds
                ens_E.append(preds_E)
                ens_F.append(preds_F)
                ens_L_frame.append(preds_L_frame)
                ens_L_atom.append(preds_L_atom)
                self._atomic_savez(
                    part_file,
                    compressed=True,
                    cache_format="ensemble_raw_model_v1",
                    model_key=np.array(model_key),
                    model_path=np.array(model_path),
                    model_index=np.array(m_idx, dtype=int),
                    E=np.asarray(preds_E, dtype=float),
                    F=np.asarray(preds_F, dtype=object),
                    L_frame=_coerce_frame_latents(
                        preds_L_frame,
                        len(frames),
                        context=f"raw model {m_idx+1} frame latents",
                    ),
                )
                print(f"     Saved raw model checkpoint to {part_file}")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
                print("[EnsembleRunner] Persistent worker pool closed successfully.")

        # If no models were successfully evaluated, return empty lists to trigger the ValueError upstream
        if not ens_E:
            return np.array([]), np.array([]), np.array([]), np.array([])

        ens_E = np.array(ens_E)
        ens_F = np.array(ens_F, dtype=object)
        ens_L_frame = np.array([
            _coerce_frame_latents(lat, len(frames), context=f"raw model {i+1} frame latents")
            for i, lat in enumerate(ens_L_frame)
        ])
        ens_L_atom = np.array(ens_L_atom, dtype=object)

        print(f"[EnsembleRunner] Saving uncompressed cache to {cache_file} (omitting atom latents)...")
        self._atomic_savez(
            cache_file,
            compressed=False,
            ens_E=ens_E,
            ens_F=ens_F,
            ens_L_frame=ens_L_frame
        )
        try:
            shutil.rmtree(parts_dir)
        except OSError:
            pass
        return ens_E, ens_F, ens_L_frame, ens_L_atom

    def evaluate_stats(self, frames, true_E=None, true_F=None, cache_file="ensemble.npz", E_singlet_true=None, E_triplet_true=None):
        force_recompute = self._force_recompute()
        checkpoint_file = f"{cache_file}.checkpoint.npz"
        if force_recompute:
            print(f"\n[EnsembleRunner] ensemble_force_recompute=true; rebuilding {cache_file} from scratch.")
            self._clear_resume_artifacts(cache_file)

        if not force_recompute and os.path.exists(cache_file):
            print(f"\n[EnsembleRunner] Loading aggregate cache from {cache_file}...")
            data = np.load(cache_file, allow_pickle=True)
            if "cache_format" in data and str(data["cache_format"]) == "ensemble_stats_v1":
                return {key: data[key] for key in data.files if key != "cache_format"}
            print("[EnsembleRunner] Existing cache is not aggregate stats; rebuilding.")

        print(f"\n[EnsembleRunner] Aggregate inference for {len(frames)} labeled frames...")
        model_paths_to_run = self._model_paths()
        current_model_keys = {self._model_key(path) for path in model_paths_to_run}

        sum_E = sum_E2 = None
        sum_F = sum_F2 = None
        sum_L = sum_L2 = None
        count_E = count_F = count_L = None
        n_models_done = 0
        model_keys_done = []
        model_paths_done = []

        if not force_recompute and os.path.exists(checkpoint_file):
            try:
                ckpt = np.load(checkpoint_file, allow_pickle=True)
                if "cache_format" in ckpt and str(ckpt["cache_format"]) == "ensemble_stats_checkpoint_v1":
                    sum_E = ckpt["sum_E"]
                    sum_E2 = ckpt["sum_E2"]
                    sum_F = ckpt["sum_F"]
                    sum_F2 = ckpt["sum_F2"]
                    sum_L = ckpt["sum_L"]
                    sum_L2 = ckpt["sum_L2"]
                    count_E = ckpt["count_E"]
                    count_F = ckpt["count_F"]
                    count_L = ckpt["count_L"]
                    loaded_model_keys_done = [str(x) for x in ckpt["model_keys_done"]]
                    if not set(loaded_model_keys_done).issubset(current_model_keys):
                        raise ValueError("checkpoint model list does not match current ensemble files")
                    n_models_done = int(ckpt["n_models_done"])
                    model_keys_done = loaded_model_keys_done
                    model_paths_done = [str(x) for x in ckpt["model_paths_done"]]
                    print(
                        f"[EnsembleRunner] Resuming aggregate cache from {checkpoint_file}: "
                        f"{n_models_done} model(s) already complete."
                    )
                else:
                    print("[EnsembleRunner] Existing checkpoint has unknown format; ignoring.")
            except Exception as exc:
                print(f"[EnsembleRunner] Failed to load checkpoint {checkpoint_file}: {exc}; rebuilding.")
                sum_E = sum_E2 = None
                sum_F = sum_F2 = None
                sum_L = sum_L2 = None
                count_E = count_F = count_L = None
                n_models_done = 0
                model_keys_done = []
                model_paths_done = []

        # Initialize persistent worker pool if multi-GPU is enabled
        pool = None
        if self._multi_gpu_enabled() and len(model_paths_to_run) > 0:
            gpu_ids = _configured_gpu_ids(self.eval_cfg)
            if len(gpu_ids) > 1:
                ctx = mp.get_context("spawn")
                gpu_queue = ctx.SimpleQueue()
                for gid in gpu_ids:
                    gpu_queue.put(gid)
                
                print(f"[EnsembleRunner] Spawning persistent worker pool with {len(gpu_ids)} GPU(s): {gpu_ids}")
                pool = ctx.Pool(
                    processes=len(gpu_ids),
                    initializer=_init_persistent_worker,
                    initargs=(gpu_queue,)
                )

        try:
            for m_idx, model_path in enumerate(model_paths_to_run):
                model_key = self._model_key(model_path)
                if model_key in model_keys_done:
                    print(f"  -> Skipping completed Model {m_idx+1}/{len(model_paths_to_run)}: {model_path}")
                    continue

                print(f"  -> Aggregating Model {m_idx+1}/{len(model_paths_to_run)}: {model_path}")

                if self._multi_gpu_enabled():
                    preds = self._evaluate_model_multi_gpu(
                        model_path,
                        frames,
                        true_E=true_E,
                        true_F=true_F,
                        pool=pool,
                        E_singlet_true=E_singlet_true,
                        E_triplet_true=E_triplet_true,
                    )
                else:
                    preds = self._evaluate_model_single_gpu(
                        model_path,
                        frames,
                        true_E=true_E,
                        true_F=true_F,
                        E_singlet_true=E_singlet_true,
                        E_triplet_true=E_triplet_true,
                    )

                if preds is None:
                    continue

                preds_E, preds_F, preds_L_frame, _ = preds
                e = np.asarray(preds_E, dtype=float)
                f = np.concatenate(preds_F, axis=0).astype(float, copy=False)
                l_frame = _coerce_frame_latents(
                    preds_L_frame,
                    len(frames),
                    context=f"model {m_idx+1} frame latents",
                )

                if sum_E is None:
                    sum_E = np.zeros_like(e, dtype=float)
                    sum_E2 = np.zeros_like(e, dtype=float)
                    sum_F = np.zeros_like(f, dtype=float)
                    sum_F2 = np.zeros_like(f, dtype=float)
                    sum_L = np.zeros_like(l_frame, dtype=float)
                    sum_L2 = np.zeros_like(l_frame, dtype=float)
                    count_E = np.zeros_like(e, dtype=float)
                    count_F = np.zeros_like(f, dtype=float)
                    count_L = np.zeros_like(l_frame, dtype=float)

                e_mask = np.isfinite(e)
                f_mask = np.isfinite(f)
                l_mask = np.isfinite(l_frame)
                sum_E += np.where(e_mask, e, 0.0)
                sum_E2 += np.where(e_mask, e**2, 0.0)
                sum_F += np.where(f_mask, f, 0.0)
                sum_F2 += np.where(f_mask, f**2, 0.0)
                sum_L += np.where(l_mask, l_frame, 0.0)
                sum_L2 += np.where(l_mask, l_frame**2, 0.0)
                count_E += e_mask.astype(float)
                count_F += f_mask.astype(float)
                count_L += l_mask.astype(float)
                n_models_done += 1
                model_keys_done.append(model_key)
                model_paths_done.append(model_path)

                self._atomic_savez(
                    checkpoint_file,
                    compressed=True,
                    cache_format="ensemble_stats_checkpoint_v1",
                    cache_file=np.array(cache_file),
                    model_keys_done=np.asarray(model_keys_done, dtype=str),
                    model_paths_done=np.asarray(model_paths_done, dtype=str),
                    n_models_done=np.array(n_models_done, dtype=int),
                    sum_E=sum_E,
                    sum_E2=sum_E2,
                    sum_F=sum_F,
                    sum_F2=sum_F2,
                    sum_L=sum_L,
                    sum_L2=sum_L2,
                    count_E=count_E,
                    count_F=count_F,
                    count_L=count_L,
                )
                print(f"     Saved aggregate checkpoint to {checkpoint_file}")

                del preds, preds_E, preds_F, preds_L_frame, e, f, l_frame
        finally:
            if pool is not None:
                pool.close()
                pool.join()
                print("[EnsembleRunner] Persistent worker pool closed successfully.")

        if n_models_done == 0:
            raise ValueError("No ensemble models were successfully evaluated.")

        if np.any(count_E < n_models_done) or np.any(count_F < n_models_done) or np.any(count_L < n_models_done):
            print(
                "[EnsembleRunner] WARNING: Some model/frame predictions failed; "
                "aggregate statistics use finite predictions only."
            )

        with np.errstate(invalid="ignore", divide="ignore"):
            mu_E = np.where(count_E > 0, sum_E / count_E, np.nan)
            mu_F = np.where(count_F > 0, sum_F / count_F, np.nan)
            mu_L_frame = np.where(count_L > 0, sum_L / count_L, np.nan)
            var_E = np.where(count_E > 1, (sum_E2 - count_E * mu_E**2) / (count_E - 1), 0.0)
            var_F = np.where(count_F > 1, (sum_F2 - count_F * mu_F**2) / (count_F - 1), 0.0)
            var_L = np.where(count_L > 1, (sum_L2 - count_L * mu_L_frame**2) / (count_L - 1), 0.0)
        sigma_E = np.sqrt(np.maximum(var_E, 0.0))
        sigma_F = np.sqrt(np.maximum(var_F, 0.0))
        sigma_L_frame = np.sqrt(np.maximum(var_L, 0.0))
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
        self._atomic_savez(cache_file, compressed=True, cache_format="ensemble_stats_v1", **result)
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
        except OSError:
            pass
        return result

    def _aggregate_prediction_sets(self, pred_sets, frames):
        sum_E = sum_E2 = None
        sum_F = sum_F2 = None
        sum_L = sum_L2 = None
        count_E = count_F = count_L = None
        n_models_done = 0

        for preds_E, preds_F, preds_L_frame in pred_sets:
            e = np.asarray(preds_E, dtype=float)
            f = np.concatenate(preds_F, axis=0).astype(float, copy=False)
            l_frame = _coerce_frame_latents(
                preds_L_frame,
                len(frames),
                context=f"model {n_models_done+1} frame latents",
            )
            if sum_E is None:
                sum_E = np.zeros_like(e, dtype=float)
                sum_E2 = np.zeros_like(e, dtype=float)
                sum_F = np.zeros_like(f, dtype=float)
                sum_F2 = np.zeros_like(f, dtype=float)
                sum_L = np.zeros_like(l_frame, dtype=float)
                sum_L2 = np.zeros_like(l_frame, dtype=float)
                count_E = np.zeros_like(e, dtype=float)
                count_F = np.zeros_like(f, dtype=float)
                count_L = np.zeros_like(l_frame, dtype=float)

            e_mask = np.isfinite(e)
            f_mask = np.isfinite(f)
            l_mask = np.isfinite(l_frame)
            sum_E += np.where(e_mask, e, 0.0)
            sum_E2 += np.where(e_mask, e**2, 0.0)
            sum_F += np.where(f_mask, f, 0.0)
            sum_F2 += np.where(f_mask, f**2, 0.0)
            sum_L += np.where(l_mask, l_frame, 0.0)
            sum_L2 += np.where(l_mask, l_frame**2, 0.0)
            count_E += e_mask.astype(float)
            count_F += f_mask.astype(float)
            count_L += l_mask.astype(float)
            n_models_done += 1

        if n_models_done == 0:
            raise ValueError("No ensemble models were successfully evaluated.")

        with np.errstate(invalid="ignore", divide="ignore"):
            mu_E = np.where(count_E > 0, sum_E / count_E, np.nan)
            mu_F = np.where(count_F > 0, sum_F / count_F, np.nan)
            mu_L_frame = np.where(count_L > 0, sum_L / count_L, np.nan)
            var_E = np.where(count_E > 1, (sum_E2 - count_E * mu_E**2) / (count_E - 1), 0.0)
            var_F = np.where(count_F > 1, (sum_F2 - count_F * mu_F**2) / (count_F - 1), 0.0)
            var_L = np.where(count_L > 1, (sum_L2 - count_L * mu_L_frame**2) / (count_L - 1), 0.0)

        return {
            "n_models": np.array(n_models_done, dtype=int),
            "mu_E": mu_E,
            "sigma_E": np.sqrt(np.maximum(var_E, 0.0)),
            "mu_F": mu_F,
            "sigma_F": np.sqrt(np.maximum(var_F, 0.0)),
            "mu_L_frame": mu_L_frame,
            "sigma_L_frame": np.sqrt(np.maximum(var_L, 0.0)),
            "n_atoms_per_frame": np.array([len(fr) for fr in frames], dtype=int),
        }

    def evaluate_stats_multihead_once(
        self,
        frames,
        cache_file="ensemble_calibration.npz",
        E_singlet_true=None,
        E_triplet_true=None,
        other_head="triplet_reconstructed",
        other_cache_file=None,
    ):
        other_cache_file = other_cache_file or f"ensemble_calibration_{other_head}.npz"
        force_recompute = self._force_recompute()
        checkpoint_file = f"{cache_file}.multihead_checkpoint.npz"
        if force_recompute:
            for path in (cache_file, other_cache_file, checkpoint_file):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError as exc:
                    print(f"[EnsembleRunner] WARNING: Could not remove {path}: {exc}")

        if not force_recompute and os.path.exists(cache_file) and os.path.exists(other_cache_file):
            data_a = np.load(cache_file, allow_pickle=True)
            data_b = np.load(other_cache_file, allow_pickle=True)
            if (
                "cache_format" in data_a and str(data_a["cache_format"]) == "ensemble_stats_v1"
                and "cache_format" in data_b and str(data_b["cache_format"]) == "ensemble_stats_v1"
            ):
                print(f"[EnsembleRunner] Loading dual-head aggregate caches from {cache_file} and {other_cache_file}.")
                return (
                    {key: data_a[key] for key in data_a.files if key != "cache_format"},
                    {key: data_b[key] for key in data_b.files if key != "cache_format"},
                )

        print(f"\n[EnsembleRunner] One-pass dual-head aggregate inference for {len(frames)} frames...")
        primary_sets = []
        other_sets = []
        model_keys_done = []
        model_paths_done = []
        model_paths_to_run = self._model_paths()

        if not force_recompute and os.path.exists(checkpoint_file):
            try:
                ckpt = np.load(checkpoint_file, allow_pickle=True)
                if "cache_format" in ckpt and str(ckpt["cache_format"]) == "ensemble_multihead_parts_v1":
                    current_keys = {self._model_key(path) for path in model_paths_to_run}
                    loaded_keys = [str(x) for x in ckpt["model_keys_done"]]
                    if not set(loaded_keys).issubset(current_keys):
                        raise ValueError("checkpoint model list does not match current ensemble files")
                    model_keys_done = loaded_keys
                    model_paths_done = [str(x) for x in ckpt["model_paths_done"]]
                    primary_E_parts = list(ckpt["primary_E_parts"])
                    primary_F_parts = list(ckpt["primary_F_parts"])
                    primary_L_parts = list(ckpt["primary_L_parts"])
                    other_E_parts = list(ckpt["other_E_parts"])
                    other_F_parts = list(ckpt["other_F_parts"])
                    other_L_parts = list(ckpt["other_L_parts"])
                    primary_sets = [
                        (primary_E_parts[i], list(primary_F_parts[i]), primary_L_parts[i])
                        for i in range(len(primary_E_parts))
                    ]
                    other_sets = [
                        (other_E_parts[i], list(other_F_parts[i]), other_L_parts[i])
                        for i in range(len(other_E_parts))
                    ]
                    print(
                        f"[EnsembleRunner] Resuming one-pass dual-head checkpoint "
                        f"from {checkpoint_file}: {len(model_keys_done)} model(s) complete."
                    )
                else:
                    print("[EnsembleRunner] Existing multihead checkpoint has unknown format; ignoring.")
            except Exception as exc:
                print(f"[EnsembleRunner] Failed to load multihead checkpoint {checkpoint_file}: {exc}; rebuilding.")
                primary_sets = []
                other_sets = []
                model_keys_done = []
                model_paths_done = []

        pool = None
        if self._multi_gpu_enabled() and len(model_paths_to_run) > 0:
            gpu_ids = _configured_gpu_ids(self.eval_cfg)
            if len(gpu_ids) > 1:
                ctx = mp.get_context("spawn")
                gpu_queue = ctx.SimpleQueue()
                for gid in gpu_ids:
                    gpu_queue.put(gid)
                pool = ctx.Pool(
                    processes=len(gpu_ids),
                    initializer=_init_persistent_worker,
                    initargs=(gpu_queue,),
                )
        try:
            for m_idx, model_path in enumerate(model_paths_to_run):
                model_key = self._model_key(model_path)
                if model_key in model_keys_done:
                    print(f"  -> Skipping completed dual-head Model {m_idx+1}/{len(model_paths_to_run)}: {model_path}")
                    continue

                print(f"  -> Dual-head aggregating Model {m_idx+1}/{len(model_paths_to_run)}: {model_path}")
                if self._multi_gpu_enabled():
                    preds = self._evaluate_model_multi_gpu(
                        model_path,
                        frames,
                        pool=pool,
                        E_singlet_true=E_singlet_true,
                        E_triplet_true=E_triplet_true,
                        include_multihead=True,
                    )
                else:
                    preds = self._evaluate_model_single_gpu(
                        model_path,
                        frames,
                        E_singlet_true=E_singlet_true,
                        E_triplet_true=E_triplet_true,
                        include_multihead=True,
                    )
                if preds is None:
                    continue
                preds_E, preds_F, preds_L_frame, _, mh = preds
                primary_sets.append((preds_E, preds_F, preds_L_frame))
                if not mh or other_head not in mh:
                    raise ValueError(f"Multihead inference did not return head '{other_head}'.")
                other_sets.append((mh[other_head]["energy"], mh[other_head]["forces"], preds_L_frame))
                model_keys_done.append(model_key)
                model_paths_done.append(model_path)
                self._atomic_savez(
                    checkpoint_file,
                    compressed=True,
                    cache_format="ensemble_multihead_parts_v1",
                    cache_file=np.array(cache_file),
                    other_cache_file=np.array(other_cache_file),
                    other_head=np.array(other_head),
                    model_keys_done=np.asarray(model_keys_done, dtype=str),
                    model_paths_done=np.asarray(model_paths_done, dtype=str),
                    primary_E_parts=np.asarray([p[0] for p in primary_sets], dtype=object),
                    primary_F_parts=np.asarray([p[1] for p in primary_sets], dtype=object),
                    primary_L_parts=np.asarray([p[2] for p in primary_sets], dtype=object),
                    other_E_parts=np.asarray([p[0] for p in other_sets], dtype=object),
                    other_F_parts=np.asarray([p[1] for p in other_sets], dtype=object),
                    other_L_parts=np.asarray([p[2] for p in other_sets], dtype=object),
                )
                print(f"     Saved one-pass dual-head checkpoint to {checkpoint_file}")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        primary = self._aggregate_prediction_sets(primary_sets, frames)
        other = self._aggregate_prediction_sets(other_sets, frames)
        self._atomic_savez(cache_file, compressed=True, cache_format="ensemble_stats_v1", **primary)
        self._atomic_savez(other_cache_file, compressed=True, cache_format="ensemble_stats_v1", **other)
        try:
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
        except OSError:
            pass
        return primary, other

    def evaluate_pool_light(self, frames, cache_file="ensemble_unlabel.npz"):
        force_recompute = self._force_recompute()
        if force_recompute:
            print(f"\n[EnsembleRunner] ensemble_force_recompute=true; rebuilding {cache_file} from scratch.")
            self._clear_resume_artifacts(cache_file)

        if not force_recompute and os.path.exists(cache_file):
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
        self._atomic_savez(cache_file, compressed=True, cache_format="ensemble_pool_light_v1", **result)
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
        force_recompute = _parse_bool_like(self.eval_cfg.get("ensemble_force_recompute", False), default=False)
        has_resume_artifacts = False
        if not force_recompute:
            try:
                has_resume_artifacts = any(
                    name.endswith(".checkpoint.npz") or name.endswith(".npz.parts")
                    for name in os.listdir(".")
                )
            except OSError:
                has_resume_artifacts = False
        if has_resume_artifacts:
            print(f"[EvaluationPipeline] Resume artifacts found; appending to {self.eval_log}.")
            open(self.eval_log, "a").close()
        else:
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
        
        self.stats_ens_other = None
        self.sigma_comp_other = None
        self.sigma_E_raw_other = None

    def run(self):
        data_mgr = DatasetManager(self.config)
        mode = str(self.eval_cfg.get("mode", "all")).lower()

        if mode == "active_learning":
            self.ds = data_mgr.load_active_learning_datasets()
            if "ensemble" not in self.uq_methods:
                raise ValueError("eval.mode='active_learning' requires eval.uncertainty to include 'ensemble'.")
            stats_ens, mean_L_frame, sigma_comp, sigma_E_raw, uq_calibrators, metrics_cal = (
                self._run_ensemble_calibration()
            )
            soap_species = self._maybe_replace_latents_with_soap(mean_L_frame)
            if soap_species is not None:
                mean_L_frame = self._soap_labeled
            self._run_pool_al(
                stats_ens,
                mean_L_frame,
                sigma_E_raw,
                sigma_comp,
                soap_species=soap_species,
                uq_calibrators=uq_calibrators,
                metrics_train=metrics_cal,
                metrics_eval=metrics_cal,
            )
            print("Evaluation Pipeline Completed.")
            return

        if mode == "uq_stats":
            if not self.eval_cfg.get("training_data"):
                raise ValueError("eval.mode='uq_stats' requires eval.training_data.")
            if not (self.eval_cfg.get("validation_data") or self.eval_cfg.get("eval_input_xyz")):
                raise ValueError("eval.mode='uq_stats' requires eval.validation_data (or legacy eval.eval_input_xyz).")

        if mode not in {"all", "uq_stats"}:
            raise ValueError("eval.mode must be one of: 'active_learning', 'uq_stats', 'all'.")

        self.ds = data_mgr.load_datasets()
        
        # 2. Evaluate Base Model (Single Model Fallback)
        if "none" in self.uq_methods or self.eval_cfg.get("error_estimate", False):
            self._run_base_model()
            
        # 3. Evaluate Ensemble & Run Active Learning
        if "ensemble" in self.uq_methods:
            stats_ens, mean_L_frame, sigma_comp, sigma_E_raw, uq_calibrators, metrics_train, metrics_eval = (
                self._run_ensemble_labeled()
            )
            
            soap_species = None
            if mode == "all":
                soap_species = self._maybe_replace_latents_with_soap(mean_L_frame)
                if soap_species is not None:
                    mean_L_frame = self._soap_labeled
            
            if mode == "all" and self.al_val_flag and self.al_val_flag.lower() == "influence":
                self._run_validation_al(stats_ens, mean_L_frame, sigma_comp)
                
            if mode == "all" and self.pool_xyz_path and os.path.exists(self.pool_xyz_path):
                self._run_pool_al(
                    stats_ens,
                    mean_L_frame,
                    sigma_E_raw,
                    sigma_comp,
                    soap_species=soap_species,
                    uq_calibrators=uq_calibrators,
                    metrics_train=metrics_train,
                    metrics_eval=metrics_eval,
                )

        print("Evaluation Pipeline Completed.")

    def _maybe_replace_latents_with_soap(self, mean_L_frame):
        use_soap = self.eval_cfg.get("use_soap", True)
        if not use_soap:
            return None
        print("\n[Active Learning] Computing model-independent SOAP descriptors for active learning latent space...")
        soap_all, soap_species = compute_soap_features(
            self.ds["frames"],
            r_cut=self.eval_cfg.get("soap_rcut", 4.0),
            n_max=self.eval_cfg.get("soap_nmax", 4),
            l_max=self.eval_cfg.get("soap_lmax", 4),
        )
        if soap_all is not None:
            self._soap_labeled = soap_all
            return soap_species
        self._soap_labeled = mean_L_frame
        return None

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
                E_singlet_true=self.ds.get("E_singlet_true"),
                E_triplet_true=self.ds.get("E_triplet_true"),
            )
        else:
            print(f"Loaded base model from {base_path}")
            preds = runner._evaluate_model_single_gpu(
                base_path,
                self.ds["frames"],
                true_E=list(self.ds["E_true"]),
                true_F=self.ds["F_true"],
                log_file=self.eval_log,
                E_singlet_true=self.ds.get("E_singlet_true"),
                E_triplet_true=self.ds.get("E_triplet_true"),
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

    def _run_ensemble_calibration(self):
        """Run ensemble on the calibration set only for pool active learning."""
        orig_mace_head = self.config.get("mace_head")
        other_head = self._dual_head_other_head(orig_mace_head)
        if other_head is not None:
            print(
                "\n[Calibration] Dual-head AL: collecting primary and secondary "
                "validation calibration from one ensemble pass."
            )
            runner = EnsembleRunner(self.config, self.device, self.neighbour_list)
            primary_cache, other_cache = runner.evaluate_stats_multihead_once(
                self.ds["frames"],
                cache_file="ensemble_calibration.npz",
                E_singlet_true=self.ds.get("E_singlet_true"),
                E_triplet_true=self.ds.get("E_triplet_true"),
                other_head=other_head,
                other_cache_file=f"ensemble_calibration_{other_head}.npz",
            )
            other_E_true, other_F_true = self._truth_for_head(other_head)
            primary = self._calibration_from_stats_cache(
                primary_cache,
                self.ds["E_true"],
                self.ds["F_true"],
                tag="ensemble_calibration",
            )
            other = self._calibration_from_stats_cache(
                other_cache,
                other_E_true,
                other_F_true,
                tag=f"ensemble_calibration_{other_head}",
            )
            self.stats_ens_other = other[0]
            self.sigma_comp_other = other[2]
            self.sigma_E_raw_other = other[3]
            self.metrics_train_other = other[5]
            self.metrics_eval_other = other[5]
            shared_calibrators = self._build_shared_var_calibrators(
                primary[0], primary[2], primary[3], other[0], other[2], other[3]
            )
            primary[5]["calibrators"] = shared_calibrators
            self.metrics_train_other["calibrators"] = shared_calibrators
            if str(self.eval_cfg.get("selection_calibration", "var")).lower() != "var":
                print("[Calibration] Shared dual-head calibration is VAR-based; using selection_calibration='var'.")
                self.eval_cfg["selection_calibration"] = "var"
            print("[Calibration] Dual-head AL uses one shared VAR calibration for both heads.")
            return primary[0], primary[1], primary[2], primary[3], shared_calibrators, primary[5]

        primary = self._evaluate_calibration_head(
            cache_file="ensemble_calibration.npz",
            tag="ensemble_calibration",
        )
        return primary[0], primary[1], primary[2], primary[3], primary[4], primary[5]

    def _calibration_from_stats_cache(self, stats_cache, true_E, true_F, tag):
        if int(stats_cache["n_models"]) < 2:
            raise ValueError("Ensemble UQ requested, but fewer than 2 models were loaded.")
        mu_E_frame = stats_cache["mu_E"]
        sigma_E_raw = stats_cache["sigma_E"]
        mu_F_comp = stats_cache["mu_F"]
        sigma_F_flat = stats_cache["sigma_F"]
        mean_L_frame = stats_cache["mu_L_frame"]

        mf_list, idx = [], 0
        for fr in self.ds["frames"]:
            mf_list.append(mu_F_comp[idx:idx+len(fr)])
            idx += len(fr)
        stats_ens = MLFFStats(true_E, mu_E_frame, true_F, mf_list, self.ds["train_mask"], self.ds["val_mask"])
        sigma_comp = sigma_F_flat.flatten()
        sigma_atom = np.linalg.norm(sigma_comp.reshape(-1, 3), axis=1)
        metrics_cal = calculate_uq_metrics(
            stats_ens,
            sigma_comp,
            sigma_atom,
            sigma_E_raw,
            "Train",
            tag,
            self.eval_log,
            energy_per_atom=True,
            save_plot_data=False,
        )
        uq_calibrators = metrics_cal.get("calibrators", {})
        self._print_active_learning_calibration_summary(metrics_cal)
        self._run_reference_slope_validation(stats_ens)
        return stats_ens, mean_L_frame, sigma_comp, sigma_E_raw, uq_calibrators, metrics_cal

    def _evaluate_calibration_head(self, cache_file, tag, true_E=None, true_F=None):
        """Evaluate one head on validation/calibration frames and fit diagnostics."""
        runner = EnsembleRunner(self.config, self.device, self.neighbour_list)
        cache_mode = str(self.eval_cfg.get("ensemble_cache_mode", "stats")).lower()
        true_E = self.ds["E_true"] if true_E is None else true_E
        true_F = self.ds["F_true"] if true_F is None else true_F

        if cache_mode == "raw":
            ens_E_sel, ens_F_list, ens_L_frame_sel, _ = runner.evaluate(
                self.ds["frames"],
                true_E,
                true_F,
                cache_file=cache_file,
                E_singlet_true=self.ds.get("E_singlet_true"),
                E_triplet_true=self.ds.get("E_triplet_true"),
            )
            ens_F_sel = np.array([np.concatenate(m_forces, axis=0) for m_forces in ens_F_list], dtype=float)
            if ens_E_sel.shape[0] < 2:
                raise ValueError("Ensemble UQ requested, but fewer than 2 models were loaded.")
            mu_E_frame = np.mean(ens_E_sel, axis=0)
            sigma_E_raw = np.std(ens_E_sel, axis=0, ddof=1)
            mu_F_comp = np.mean(ens_F_sel, axis=0)
            sigma_F_flat = np.std(ens_F_sel, axis=0, ddof=1)
            mean_L_frame = np.mean(ens_L_frame_sel, axis=0)
        else:
            stats_cache = runner.evaluate_stats(
                self.ds["frames"],
                true_E,
                true_F,
                cache_file=cache_file,
                E_singlet_true=self.ds.get("E_singlet_true"),
                E_triplet_true=self.ds.get("E_triplet_true"),
            )
            if int(stats_cache["n_models"]) < 2:
                raise ValueError("Ensemble UQ requested, but fewer than 2 models were loaded.")
            mu_E_frame = stats_cache["mu_E"]
            sigma_E_raw = stats_cache["sigma_E"]
            mu_F_comp = stats_cache["mu_F"]
            sigma_F_flat = stats_cache["sigma_F"]
            mean_L_frame = stats_cache["mu_L_frame"]

        mf_list, idx = [], 0
        for fr in self.ds["frames"]:
            mf_list.append(mu_F_comp[idx:idx+len(fr)])
            idx += len(fr)
        stats_ens = MLFFStats(
            true_E,
            mu_E_frame,
            true_F,
            mf_list,
            self.ds["train_mask"],
            self.ds["val_mask"],
        )
        sigma_comp = sigma_F_flat.flatten()
        sigma_atom = np.linalg.norm(sigma_comp.reshape(-1, 3), axis=1)

        metrics_cal = calculate_uq_metrics(
            stats_ens,
            sigma_comp,
            sigma_atom,
            sigma_E_raw,
            "Train",
            tag,
            self.eval_log,
            energy_per_atom=True,
            save_plot_data=False,
        )
        uq_calibrators = metrics_cal.get("calibrators", {})
        self._print_active_learning_calibration_summary(metrics_cal)
        self._run_reference_slope_validation(stats_ens)
        return stats_ens, mean_L_frame, sigma_comp, sigma_E_raw, uq_calibrators, metrics_cal

    def _dual_head_other_head(self, orig_mace_head):
        al_multihead_mode = self.eval_cfg.get("al_multihead_mode", "reconstructed").lower()
        model_fw = self.config.get("model_framework", "").lower()
        if model_fw != "mace" or al_multihead_mode != "dual_head_or":
            return None
        mace_heads_map = DEFAULT_MACE_HEADS_MAP.copy()
        custom_heads = self.eval_cfg.get("mace_heads", self.config.get("mace_heads", {}))
        for head_name, head_cfg in custom_heads.items():
            mace_heads_map.setdefault(head_name, {}).update(head_cfg)
        has_multihead = (
            "singlet" in mace_heads_map
            and any(k in mace_heads_map for k in ["triplet", "triplet_reconstructed", "delta"])
        )
        if not has_multihead:
            return None
        if orig_mace_head == "singlet":
            return "triplet_reconstructed" if "triplet_reconstructed" in mace_heads_map else "triplet"
        return "singlet"

    def _truth_for_head(self, head):
        if head == "singlet":
            true_E = self.ds.get("E_singlet_true")
            true_F = self.ds.get("F_singlet_true")
        else:
            true_E = self.ds.get("E_triplet_true")
            true_F = self.ds.get("F_triplet_true")
        if true_E is None:
            true_E = self.ds["E_true"]
        if true_F is None:
            true_F = self.ds["F_true"]
        return true_E, true_F

    def _build_shared_var_calibrators(
        self,
        stats_a,
        sigma_comp_a,
        sigma_energy_a,
        stats_b,
        sigma_comp_b,
        sigma_energy_b,
    ):
        atom_counts_a = np.asarray(stats_a.atom_counts, dtype=float)
        atom_counts_b = np.asarray(stats_b.atom_counts, dtype=float)
        delta_e = np.concatenate([
            stats_a.delta_E_frame / atom_counts_a,
            stats_b.delta_E_frame / atom_counts_b,
        ])
        sigma_e = np.concatenate([
            np.asarray(sigma_energy_a, dtype=float) / atom_counts_a,
            np.asarray(sigma_energy_b, dtype=float) / atom_counts_b,
        ])
        delta_f = np.concatenate([
            stats_a.all_force_residuals.reshape(-1),
            stats_b.all_force_residuals.reshape(-1),
        ])
        sigma_f = np.concatenate([
            np.asarray(sigma_comp_a, dtype=float).reshape(-1),
            np.asarray(sigma_comp_b, dtype=float).reshape(-1),
        ])
        cal_var_F = VarianceScalingCalibrator().fit(delta_f, sigma_f)
        cal_var_E = VarianceScalingCalibrator().fit(delta_e, sigma_e)
        print(
            "[Calibration] Shared VAR scales: "
            f"forces={cal_var_F.s:.4f}, energy_per_atom={cal_var_E.s:.4f}"
        )
        return {"cal_var_F": cal_var_F, "cal_var_E": cal_var_E}

    def _print_active_learning_calibration_summary(self, metrics_cal):
        metrics = metrics_cal.get("metrics", {}) if metrics_cal else {}
        mode = str(self.eval_cfg.get("selection_calibration", "var")).lower()
        if mode not in {"var", "iso"}:
            mode = "var"
        print("\n[Calibration] Active-learning calibration summary")
        print("[Calibration] Source: validation")
        print(f"[Calibration] Frames used: {len(self.ds['frames'])}")
        print(f"[Calibration] Method requested for AL selection: {mode.upper()}")
        print("[Calibration] Diagnostics below are computed on the calibration source itself.")
        def _fmt(value):
            return "n/a" if value is None or not np.isfinite(value) else f"{float(value):.4f}"
        for label, prefix in (("raw", "raw"), ("VAR", "calVAR"), ("ISO", "calISO")):
            print(
                f"[Calibration] {label:>3} force: "
                f"Spearman={_fmt(metrics.get(f'Spearman_{prefix}'))}, "
                f"ENCE={_fmt(metrics.get(f'ENCE_{prefix}'))}, "
                f"PICP95={_fmt(metrics.get(f'PICP95_{prefix}'))}"
            )
        has_energy = any(key.endswith("_E") for key in metrics)
        if has_energy:
            for label, prefix in (("raw", "raw_E"), ("VAR", "calVAR_E"), ("ISO", "calISO_E")):
                print(
                    f"[Calibration] {label:>3} energy: "
                    f"Spearman={_fmt(metrics.get(f'Spearman_{prefix}'))}, "
                    f"ENCE={_fmt(metrics.get(f'ENCE_{prefix}'))}, "
                    f"PICP95={_fmt(metrics.get(f'PICP95_{prefix}'))}"
                )

    def _run_ensemble_labeled(self):
        """Runs the ensemble on labeled datasets and computes UQ metrics."""
        runner = EnsembleRunner(self.config, self.device, self.neighbour_list)
        cache_mode = str(self.eval_cfg.get("ensemble_cache_mode", "stats")).lower()

        if cache_mode == "raw":
            ens_E_sel, ens_F_list, ens_L_frame_sel, _ = runner.evaluate(
                self.ds["frames"], self.ds["E_true"], self.ds["F_true"], cache_file="ensemble.npz",
                E_singlet_true=self.ds.get("E_singlet_true"),
                E_triplet_true=self.ds.get("E_triplet_true"),
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
                self.ds["frames"], self.ds["E_true"], self.ds["F_true"], cache_file="ensemble.npz",
                E_singlet_true=self.ds.get("E_singlet_true"),
                E_triplet_true=self.ds.get("E_triplet_true"),
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

        metrics_train = calculate_uq_metrics(
            stats_ens,
            sigma_comp,
            sigma_atom,
            sigma_E_raw,
            "Train",
            "ensemble",
            self.eval_log,
            energy_per_atom=True,
        )
        uq_calibrators = metrics_train.get("calibrators", {})
        metrics_eval = calculate_uq_metrics(
            stats_ens,
            sigma_comp,
            sigma_atom,
            sigma_E_raw,
            "Eval",
            "ensemble",
            self.eval_log,
            calibrators=uq_calibrators,
            energy_per_atom=True,
        )

        if self.do_plot:
            generate_uq_plots(metrics_train["npz_path"], "Train", "error_model", calibration="var")
            generate_uq_plots(metrics_eval["npz_path"], "Eval", "error_model", calibration="var")

        # --- Evaluate and save UQ metrics/plots for other heads if dual_head_or is active ---
        try:
            al_multihead_mode = self.eval_cfg.get("al_multihead_mode", "reconstructed").lower()
            model_fw = self.config.get("model_framework", "").lower()
            if model_fw == "mace" and al_multihead_mode == "dual_head_or":
                mace_heads_map = DEFAULT_MACE_HEADS_MAP.copy()
                custom_heads = self.eval_cfg.get("mace_heads", self.config.get("mace_heads", {}))
                for head_name, head_cfg in custom_heads.items():
                    if head_name not in mace_heads_map:
                        mace_heads_map[head_name] = {}
                    mace_heads_map[head_name].update(head_cfg)
                
                has_multihead = (
                    "singlet" in mace_heads_map
                    and any(k in mace_heads_map for k in ["triplet", "triplet_reconstructed", "delta"])
                )
                if has_multihead:
                    orig_mace_head = self.config.get("mace_head")
                    if orig_mace_head == "singlet":
                        other_head = "triplet_reconstructed" if "triplet_reconstructed" in mace_heads_map else "triplet"
                    else:
                        other_head = "singlet"
                    
                    print(f"\n[Ensemble-UQ] Evaluating secondary head '{other_head}' on labeled splits...")
                    self.config["mace_head"] = other_head
                    
                    if other_head == "singlet":
                        other_E_true = self.ds.get("E_singlet_true")
                        other_F_true = self.ds.get("F_singlet_true")
                    else:
                        other_E_true = self.ds.get("E_triplet_true")
                        other_F_true = self.ds.get("F_triplet_true")
                    
                    if other_E_true is None:
                        other_E_true = self.ds["E_true"]
                    if other_F_true is None:
                        other_F_true = self.ds["F_true"]
                        
                    runner_other = EnsembleRunner(self.config, self.device, self.neighbour_list)
                    ens_E_other, ens_F_list_other, _, _ = runner_other.evaluate(
                        self.ds["frames"], other_E_true, other_F_true, cache_file=f"ensemble_{other_head}.npz",
                        E_singlet_true=self.ds.get("E_singlet_true"),
                        E_triplet_true=self.ds.get("E_triplet_true"),
                    )
                    
                    self.config["mace_head"] = orig_mace_head
                    
                    if len(ens_E_other) > 0 and len(ens_F_list_other) > 0:
                        ens_F_other = np.array([np.concatenate(m_forces, axis=0) for m_forces in ens_F_list_other], dtype=float)
                        mu_E_frame_other = np.mean(ens_E_other, axis=0)
                        std_E_frame_other = np.std(ens_E_other, axis=0, ddof=0)
                        sigma_E_raw_other = np.std(ens_E_other, axis=0, ddof=1)
                        mu_F_comp_other = np.mean(ens_F_other, axis=0)
                        sigma_F_flat_other = np.std(ens_F_other, axis=0, ddof=1)
                        std_F_comp_other = sigma_F_flat_other.flatten()
                        
                        mf_list_other, idx = [], 0
                        for fr in self.ds["frames"]:
                            mf_list_other.append(mu_F_comp_other[idx:idx+len(fr)])
                            idx += len(fr)
                        stats_ens_other = MLFFStats(other_E_true, mu_E_frame_other, other_F_true, mf_list_other, self.ds["train_mask"], self.ds["val_mask"])
                        
                        sigma_comp_other = sigma_F_flat_other.flatten()
                        sigma_atom_other = np.linalg.norm(sigma_comp_other.reshape(-1, 3), axis=1)
                        
                        self.stats_ens_other = stats_ens_other
                        self.sigma_comp_other = sigma_comp_other
                        self.sigma_E_raw_other = sigma_E_raw_other
                        
                        self.metrics_train_other = calculate_uq_metrics(
                            stats_ens_other, sigma_comp_other, sigma_atom_other, sigma_E_raw_other,
                            "Train", f"ensemble_{other_head}", self.eval_log,
                            energy_per_atom=True,
                        )
                        uq_calibrators_other = self.metrics_train_other.get("calibrators", {})
                        self.metrics_eval_other = calculate_uq_metrics(
                            stats_ens_other, sigma_comp_other, sigma_atom_other, sigma_E_raw_other,
                            "Eval", f"ensemble_{other_head}", self.eval_log,
                            calibrators=uq_calibrators_other,
                            energy_per_atom=True,
                        )
                        
                        if self.do_plot:
                            generate_uq_plots(self.metrics_train_other["npz_path"], "Train", f"error_model_{other_head}", calibration="var")
                            generate_uq_plots(self.metrics_eval_other["npz_path"], "Eval", f"error_model_{other_head}", calibration="var")
        except Exception as e:
            print(f"[Ensemble-UQ] WARNING: Failed to compute UQ metrics/plots for secondary head: {e}")

        self._run_reference_slope_validation(stats_ens)

        return stats_ens, mean_L_frame, sigma_comp, sigma_E_raw, uq_calibrators, metrics_train, metrics_eval

    def _run_reference_slope_validation(self, stats_ens):
        """Consecutive ΔE/force transition checks on labeled train/eval (by size class)."""
        if not _parse_bool_like(self.eval_cfg.get("reference_slope_validation"), False):
            return

        size_thr = int(self.eval_cfg.get("size_split_atoms", 300))
        min_frames = int(self.eval_cfg.get("slope_validation_min_frames", 2))
        pred_forces = stats_ens.pred_forces
        true_forces = stats_ens.true_forces
        pred_E = stats_ens.pred_energies
        true_E = stats_ens.true_energies
        atom_counts = stats_ens.atom_counts

        splits = [
            ("train_all", self.ds["train_mask"], None),
            ("eval_all", self.ds["val_mask"], None),
            ("train_small", self.ds["train_mask"], atom_counts < size_thr),
            ("train_large", self.ds["train_mask"], atom_counts >= size_thr),
            ("eval_small", self.ds["val_mask"], atom_counts < size_thr),
            ("eval_large", self.ds["val_mask"], atom_counts >= size_thr),
        ]

        for tag, frame_mask, size_cond in splits:
            if size_cond is not None:
                idx = np.where(frame_mask & size_cond)[0]
            else:
                idx = np.where(frame_mask)[0]
            if len(idx) < min_frames:
                print(f"[SlopeVal] Skipping {tag}: only {len(idx)} frames.")
                continue
            validate_consecutive_reference_deltas(
                [self.ds["frames"][i] for i in idx],
                pred_E[idx],
                true_E[idx],
                [pred_forces[i] for i in idx],
                [true_forces[i] for i in idx],
                system_tag=tag,
            )

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

    def _run_pool_al(
        self,
        stats_ens,
        mean_L_frame,
        sigma_E_raw,
        sigma_comp,
        soap_species=None,
        uq_calibrators=None,
        metrics_train=None,
        metrics_eval=None,
    ):
        """Active Learning on the unlabelled out-of-distribution pool."""
        print(f"\n[Pool-AL] Parsing unlabeled pool from {self.pool_xyz_path}")
        pool_frames = read(self.pool_xyz_path, index=":", format="extxyz")

        runner = EnsembleRunner(self.config, self.device, self.neighbour_list)
        pool_cache_mode = str(self.eval_cfg.get("pool_cache_mode", "light")).lower()
        orig_mace_head = self.config.get("mace_head", "triplet_reconstructed")
        other_head = self._dual_head_other_head(orig_mace_head)
        has_multihead = other_head is not None
        used_one_pass_dual_pool = False
        al_diagnostics_runs = []
        mu_E_pool_other = None
        sigma_E_pool_other = None
        sigma_F_pool_other = None

        if has_multihead and str(self.eval_cfg.get("mode", "all")).lower() == "active_learning":
            print("[Pool-AL] Dual-head AL: collecting primary and secondary pool predictions from one ensemble pass.")
            pool_stats, pool_stats_other = runner.evaluate_stats_multihead_once(
                pool_frames,
                cache_file="ensemble_unlabel.npz",
                other_head=other_head,
                other_cache_file=f"ensemble_unlabel_{other_head}.npz",
            )
            mu_E_pool = pool_stats["mu_E"]
            sigma_E_pool = pool_stats["sigma_E"]
            mu_F_pool = pool_stats["mu_F"]
            sigma_F_pool = pool_stats["sigma_F"]
            mu_L_pool = pool_stats["mu_L_frame"]
            sigma_F_pool_mean, sigma_F_pool_max = _force_summary_from_flat(sigma_F_pool, pool_frames)
            _, frame_max_force_pool = _force_summary_from_flat(mu_F_pool, pool_frames)
            mu_E_pool_other = pool_stats_other["mu_E"]
            sigma_E_pool_other = pool_stats_other["sigma_E"]
            sigma_F_pool_other = pool_stats_other["sigma_F"]
            used_one_pass_dual_pool = True
            pool_cache_mode = "stats"
        elif pool_cache_mode == "raw":
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
            sigma_F_pool = None

        pool_has_full_sigma = pool_cache_mode in ("raw", "stats")

        # Thinning (Adaptive Striding Option)
        adaptive_striding = self.eval_cfg.get("adaptive_striding", False)
        if adaptive_striding:
            coarse_stride = self.eval_cfg.get("coarse_stride", 20)
            fine_stride = self.eval_cfg.get("fine_stride", 2)
            unc_threshold = self.eval_cfg.get("adaptive_uncertainty_threshold", None)
            
            if unc_threshold is None:
                # Use 30th percentile of pool force uncertainties as a transition threshold
                unc_threshold = float(np.percentile(sigma_F_pool_mean, 30))
                print(f"[Pool-AL] Adaptive striding transition threshold determined from pool: {unc_threshold:.5f} eV/Å")
            else:
                print(f"[Pool-AL] User-defined adaptive striding threshold: {unc_threshold:.5f} eV/Å")
                
            thin_idx_list = []
            curr_i = 0
            n_pool = len(pool_frames)
            while curr_i < n_pool:
                thin_idx_list.append(curr_i)
                # Check average force uncertainty at current frame
                unc = sigma_F_pool_mean[curr_i]
                if unc > unc_threshold:
                    curr_i += fine_stride
                else:
                    curr_i += coarse_stride
            thin_idx = np.array(thin_idx_list, dtype=int)
            print(f"[Pool-AL] Adaptive striding thinned pool from {n_pool} to {len(thin_idx)} frames.")
        else:
            thin_idx = np.arange(len(pool_frames))[::self.eval_cfg.get("pool_stride", 1)]

        # Store original head uncertainties for diagnostic separation
        sigma_E_pool_orig = sigma_E_pool.copy() if sigma_E_pool is not None else None
        sigma_F_pool_orig = sigma_F_pool.copy() if sigma_F_pool is not None else None
        if not used_one_pass_dual_pool:
            sigma_E_pool_other = None
            sigma_F_pool_other = None
            mu_E_pool_other = None

        # Handle multi-head OR active learning if enabled
        al_multihead_mode = self.eval_cfg.get("al_multihead_mode", "reconstructed").lower()
        model_fw = self.config.get("model_framework", "").lower()
        mace_heads_map = {}
        if model_fw == "mace":
            mace_heads_map = DEFAULT_MACE_HEADS_MAP.copy()
            custom_heads = self.eval_cfg.get("mace_heads", self.config.get("mace_heads", {}))
            for head_name, head_cfg in custom_heads.items():
                if head_name not in mace_heads_map:
                    mace_heads_map[head_name] = {}
                mace_heads_map[head_name].update(head_cfg)
            has_multihead = has_multihead or (
                "singlet" in mace_heads_map
                and any(k in mace_heads_map for k in ["triplet", "triplet_reconstructed", "delta"])
            )

        if has_multihead and al_multihead_mode == "dual_head_or" and not used_one_pass_dual_pool:
            if orig_mace_head == "singlet":
                other_head = "triplet_reconstructed" if "triplet_reconstructed" in mace_heads_map else "triplet"
            else:
                other_head = "singlet"

            print(f"[Pool-AL] Dual-head Active Learning enabled. Evaluating other head '{other_head}'...")
            self.config["mace_head"] = other_head
            
            runner_s = EnsembleRunner(self.config, self.device, self.neighbour_list)
            ens_E_pool_s, ens_F_pool_list_s, _, _ = runner_s.evaluate(pool_frames, cache_file=f"ensemble_unlabel_{other_head}.npz")
            
            self.config["mace_head"] = orig_mace_head
            
            if len(ens_E_pool_s) > 0 and len(ens_F_pool_list_s) > 0:
                ens_F_pool_s = np.array([np.concatenate(m_forces, axis=0) for m_forces in ens_F_pool_list_s], dtype=float)
                sigma_E_pool_other = np.std(ens_E_pool_s, axis=0, ddof=1)
                sigma_F_pool_other = np.std(ens_F_pool_s, axis=0, ddof=1)
                mu_E_pool_other = np.mean(ens_E_pool_s, axis=0)
                
                print(f"[Pool-AL] Successfully evaluated original head '{orig_mace_head}' and other head '{other_head}' uncertainties.")
        pool_frames_thin = [pool_frames[i] for i in thin_idx]
        F_pool_thin = mu_L_pool[thin_idx].astype(float)

        # --- Compute SOAP for thinned pool frames ---
        if soap_species is not None:
            print("[Pool-AL] Computing model-independent SOAP descriptors for thinned pool frames...")
            soap_pool_thin, _ = compute_soap_features(
                pool_frames_thin,
                species=soap_species,
                r_cut=self.eval_cfg.get("soap_rcut", 4.0),
                n_max=self.eval_cfg.get("soap_nmax", 4),
                l_max=self.eval_cfg.get("soap_lmax", 4),
            )
            if soap_pool_thin is not None:
                F_pool_thin = soap_pool_thin

        mu_E_pool_thin = mu_E_pool[thin_idx].astype(float)
        sigma_E_pool_thin = sigma_E_pool[thin_idx].astype(float)
        F_train_thin = mean_L_frame[self.ds["train_idx"]].astype(float)
        sigma_F_pool_mean_thin = sigma_F_pool_mean[thin_idx].astype(float)
        sigma_F_pool_max_thin = sigma_F_pool_max[thin_idx].astype(float)
        frame_max_force_pool_thin = frame_max_force_pool[thin_idx].astype(float)

        use_cal = _parse_bool_like(self.eval_cfg.get("use_calibrated_selection"), True)
        calibration_policy = {"force_mode": None, "energy_mode": None}
        if use_cal and uq_calibrators:
            calibration_policy = _build_calibration_policy(
                self.eval_cfg, metrics_eval, pool_cache_mode=pool_cache_mode
            )
        elif use_cal:
            print("[Pool-AL] Calibrated selection requested but no train-fitted calibrators are available.")

        train_idx = self.ds["train_idx"]
        train_atom_counts = np.array([len(self.ds["frames"][i]) for i in train_idx], dtype=float)
        pool_atom_counts_thin = np.array([len(fr) for fr in pool_frames_thin], dtype=float)
        sigma_E_train_atom_raw = sigma_E_raw[train_idx] / train_atom_counts
        sigma_E_pool_atom_raw = sigma_E_pool_thin / pool_atom_counts_thin
        sigma_force_frames_all = _split_atom_vectors(sigma_comp, self.ds["frames"])
        sigma_force_train = [sigma_force_frames_all[i] for i in train_idx]
        sigma_F_train_mean_raw = np.array([
            np.nanmean(np.linalg.norm(f, axis=1)) for f in sigma_force_train
        ], dtype=float)
        sigma_F_train_max_raw = np.array([
            np.nanmax(np.linalg.norm(f, axis=1)) for f in sigma_force_train
        ], dtype=float)
        frame_max_force_train_raw = np.array([
            np.nanmax(np.linalg.norm(self.ds["F_true"][i], axis=1)) for i in train_idx
        ], dtype=float)

        support_mult = float(self.eval_cfg.get("calibration_support_upper_mult", 1.25))
        train_count_min = float(np.nanmin(train_atom_counts))
        train_count_max = float(np.nanmax(train_atom_counts))
        support_E = _in_range(
            sigma_E_pool_atom_raw,
            _range_with_margin(sigma_E_train_atom_raw, upper_mult=support_mult),
        )
        support_Fmean = _in_range(
            sigma_F_pool_mean_thin,
            _range_with_margin(sigma_F_train_mean_raw, upper_mult=support_mult),
        )
        support_Fmax = _in_range(
            sigma_F_pool_max_thin,
            _range_with_margin(sigma_F_train_max_raw, upper_mult=support_mult),
        )
        support_count = (pool_atom_counts_thin >= train_count_min) & (pool_atom_counts_thin <= train_count_max)
        support_Fphys = frame_max_force_pool_thin <= (
            np.nanmax(frame_max_force_train_raw) * float(self.eval_cfg.get("calibration_support_force_mult", 1.5))
        )
        calibration_in_support = support_E & support_Fmean & support_Fmax & support_count & support_Fphys
        ood_risk_mask = ~calibration_in_support
        print(
            "[Pool-AL] Calibration support on thinned pool: "
            f"{int(calibration_in_support.sum())}/{len(calibration_in_support)} in-domain "
            f"({np.mean(calibration_in_support):.3f})."
        )
        print(
            "[Pool-AL] Calibration support failures: "
            f"sigma_E={int((~support_E).sum())}, "
            f"sigma_F_mean={int((~support_Fmean).sum())}, "
            f"sigma_F_max={int((~support_Fmax).sum())}, "
            f"atom_count={int((~support_count).sum())}, "
            f"force_magnitude={int((~support_Fphys).sum())}."
        )

        sigma_energy_train = sigma_E_raw[train_idx]
        expected_abs_E_atom = sigma_E_pool_atom_raw * _GAUSSIAN_SIGMA_TO_ABS
        expected_abs_F_mean = sigma_F_pool_mean_thin * _GAUSSIAN_SIGMA_TO_ABS
        expected_abs_F_max = sigma_F_pool_max_thin * _GAUSSIAN_SIGMA_TO_ABS

        energy_mode = calibration_policy["energy_mode"]
        force_mode = calibration_policy["force_mode"]

        if energy_mode:
            print(f"[Pool-AL] Applying eval-accepted '{energy_mode}' energy calibration in per-atom units.")
            sigma_energy_train_atom = apply_sigma_energy_calibration(
                sigma_E_train_atom_raw, uq_calibrators, energy_mode
            )
            sigma_energy_train = sigma_energy_train_atom * train_atom_counts
            sigma_E_pool_atom_cal = apply_sigma_energy_calibration(
                sigma_E_pool_atom_raw, uq_calibrators, energy_mode
            )
            if energy_mode == "iso":
                sigma_E_pool_atom_var = apply_sigma_energy_calibration(
                    sigma_E_pool_atom_raw, uq_calibrators, "var"
                )
                sigma_E_pool_atom_cal = np.where(
                    calibration_in_support, sigma_E_pool_atom_cal, sigma_E_pool_atom_var
                )
            sigma_E_pool_thin = sigma_E_pool_atom_cal * pool_atom_counts_thin
            expected_abs_E_atom = sigma_E_pool_atom_cal * _GAUSSIAN_SIGMA_TO_ABS

        if force_mode:
            print(f"[Pool-AL] Applying eval-accepted '{force_mode}' force calibration.")
            sigma_force_train = calibrate_sigma_force_frames(
                sigma_force_train, uq_calibrators, force_mode
            )
            if pool_has_full_sigma:
                sigma_F_pool_shape = np.asarray(sigma_F_pool).shape
                sigma_F_pool_flat = np.asarray(sigma_F_pool, dtype=float).reshape(-1)
                sigma_F_pool_var = apply_sigma_comp_calibration(
                    sigma_F_pool_flat, uq_calibrators, "var"
                )
                if force_mode == "iso":
                    sigma_F_pool_iso = apply_sigma_comp_calibration(
                        sigma_F_pool_flat, uq_calibrators, "iso"
                    )
                    pool_counts = np.array([len(fr) for fr in pool_frames], dtype=int)
                    frame_ids = np.repeat(np.arange(len(pool_frames)), pool_counts * 3)
                    thin_support_global = np.zeros(len(pool_frames), dtype=bool)
                    thin_support_global[thin_idx] = calibration_in_support
                    component_support = thin_support_global[frame_ids]
                    sigma_F_pool = np.where(component_support, sigma_F_pool_iso, sigma_F_pool_var)
                else:
                    sigma_F_pool = sigma_F_pool_var
                sigma_F_pool = sigma_F_pool.reshape(sigma_F_pool_shape)
                sigma_F_pool_mean, sigma_F_pool_max = _force_summary_from_flat(
                    sigma_F_pool, pool_frames
                )
                sigma_F_pool_mean_thin = sigma_F_pool_mean[thin_idx].astype(float)
                sigma_F_pool_max_thin = sigma_F_pool_max[thin_idx].astype(float)
            else:
                sigma_F_pool_mean_thin, sigma_F_pool_max_thin = scale_pool_force_summaries(
                    sigma_F_pool_mean_thin, sigma_F_pool_max_thin, uq_calibrators
                )
            expected_abs_F_mean = sigma_F_pool_mean_thin * _GAUSSIAN_SIGMA_TO_ABS
            expected_abs_F_max = sigma_F_pool_max_thin * _GAUSSIAN_SIGMA_TO_ABS

        sigma_E_pool_orig_thin = sigma_E_pool_orig[thin_idx].astype(float) if sigma_E_pool_orig is not None else None
        sigma_F_pool_orig_thin = None

        sigma_E_pool_other_thin = None
        mu_E_pool_other_thin = None
        sigma_F_pool_other_thin = None
        if sigma_E_pool_other is not None:
            sigma_E_pool_other_thin = sigma_E_pool_other[thin_idx].astype(float)
        if mu_E_pool_other is not None:
            mu_E_pool_other_thin = mu_E_pool_other[thin_idx].astype(float)

        # --- Secondary head calibration setup if dual_head_or is active ---
        calibration_in_support_other = calibration_in_support
        ood_risk_mask_other = ood_risk_mask
        expected_abs_E_atom_other = expected_abs_E_atom
        expected_abs_F_mean_other = expected_abs_F_mean
        expected_abs_F_max_other = expected_abs_F_max
        sigma_energy_train_other = None
        sigma_force_train_other = None
        sigma_F_pool_mean_other_thin = None
        sigma_F_pool_max_other_thin = None

        if has_multihead and al_multihead_mode == "dual_head_or" and getattr(self, "metrics_train_other", None) is not None:
            if sigma_E_pool_other is not None:
                sigma_F_pool_mean_other, sigma_F_pool_max_other = _force_summary_from_flat(sigma_F_pool_other, pool_frames)
                sigma_F_pool_mean_other_thin = sigma_F_pool_mean_other[thin_idx].astype(float)
                sigma_F_pool_max_other_thin = sigma_F_pool_max_other[thin_idx].astype(float)

            uq_calibrators_other = self.metrics_train_other.get("calibrators", {})
            metrics_eval_other = getattr(self, "metrics_eval_other", {})
            
            use_cal_other = _parse_bool_like(self.eval_cfg.get("use_calibrated_selection"), True)
            calibration_policy_other = {"force_mode": None, "energy_mode": None}
            if use_cal_other and uq_calibrators_other:
                calibration_policy_other = _build_calibration_policy(
                    self.eval_cfg, metrics_eval_other, pool_cache_mode=pool_cache_mode
                )
            
            # support check for other head
            sigma_E_train_atom_raw_other = self.sigma_E_raw_other[train_idx] / train_atom_counts
            sigma_E_pool_atom_raw_other = sigma_E_pool_other_thin / pool_atom_counts_thin
            sigma_force_frames_all_other = _split_atom_vectors(self.sigma_comp_other, self.ds["frames"])
            sigma_force_train_other = [sigma_force_frames_all_other[i] for i in train_idx]
            sigma_F_train_mean_raw_other = np.array([
                np.nanmean(np.linalg.norm(f, axis=1)) for f in sigma_force_train_other
            ], dtype=float)
            sigma_F_train_max_raw_other = np.array([
                np.nanmax(np.linalg.norm(f, axis=1)) for f in sigma_force_train_other
            ], dtype=float)
            
            support_E_other = _in_range(
                sigma_E_pool_atom_raw_other,
                _range_with_margin(sigma_E_train_atom_raw_other, upper_mult=support_mult),
            )
            if sigma_F_pool_mean_other_thin is not None:
                support_Fmean_other = _in_range(
                    sigma_F_pool_mean_other_thin,
                    _range_with_margin(sigma_F_train_mean_raw_other, upper_mult=support_mult),
                )
                support_Fmax_other = _in_range(
                    sigma_F_pool_max_other_thin,
                    _range_with_margin(sigma_F_train_max_raw_other, upper_mult=support_mult),
                )
                calibration_in_support_other = support_E_other & support_Fmean_other & support_Fmax_other & support_count & support_Fphys
            else:
                calibration_in_support_other = support_E_other & support_count & support_Fphys
            ood_risk_mask_other = ~calibration_in_support_other
            
            sigma_energy_train_other = self.sigma_E_raw_other[train_idx]
            expected_abs_E_atom_other = sigma_E_pool_atom_raw_other * _GAUSSIAN_SIGMA_TO_ABS
            if sigma_F_pool_mean_other_thin is not None:
                expected_abs_F_mean_other = sigma_F_pool_mean_other_thin * _GAUSSIAN_SIGMA_TO_ABS
                expected_abs_F_max_other = sigma_F_pool_max_other_thin * _GAUSSIAN_SIGMA_TO_ABS
            
            energy_mode_other = calibration_policy_other["energy_mode"]
            force_mode_other = calibration_policy_other["force_mode"]
            
            if energy_mode_other:
                print(f"[Pool-AL] Applying other head eval-accepted '{energy_mode_other}' energy calibration.")
                sigma_energy_train_atom_other = apply_sigma_energy_calibration(
                    sigma_E_train_atom_raw_other, uq_calibrators_other, energy_mode_other
                )
                sigma_energy_train_other = sigma_energy_train_atom_other * train_atom_counts
                sigma_E_pool_atom_cal_other = apply_sigma_energy_calibration(
                    sigma_E_pool_atom_raw_other, uq_calibrators_other, energy_mode_other
                )
                if energy_mode_other == "iso":
                    sigma_E_pool_atom_var_other = apply_sigma_energy_calibration(
                        sigma_E_pool_atom_raw_other, uq_calibrators_other, "var"
                    )
                    sigma_E_pool_atom_cal_other = np.where(
                        calibration_in_support_other, sigma_E_pool_atom_cal_other, sigma_E_pool_atom_var_other
                    )
                sigma_E_pool_other_thin = sigma_E_pool_atom_cal_other * pool_atom_counts_thin
                expected_abs_E_atom_other = sigma_E_pool_atom_cal_other * _GAUSSIAN_SIGMA_TO_ABS
                
            if force_mode_other:
                print(f"[Pool-AL] Applying other head eval-accepted '{force_mode_other}' force calibration.")
                sigma_force_train_other = calibrate_sigma_force_frames(
                    sigma_force_train_other, uq_calibrators_other, force_mode_other
                )
                if pool_has_full_sigma:
                    sigma_F_pool_shape_other = np.asarray(sigma_F_pool_other).shape
                    sigma_F_pool_flat_other = np.asarray(sigma_F_pool_other, dtype=float).reshape(-1)
                    sigma_F_pool_var_other = apply_sigma_comp_calibration(
                        sigma_F_pool_flat_other, uq_calibrators_other, "var"
                    )
                    if force_mode_other == "iso":
                        sigma_F_pool_iso_other = apply_sigma_comp_calibration(
                            sigma_F_pool_flat_other, uq_calibrators_other, "iso"
                        )
                        pool_counts = np.array([len(fr) for fr in pool_frames], dtype=int)
                        frame_ids = np.repeat(np.arange(len(pool_frames)), pool_counts * 3)
                        thin_support_global_other = np.zeros(len(pool_frames), dtype=bool)
                        thin_support_global_other[thin_idx] = calibration_in_support_other
                        component_support_other = thin_support_global_other[frame_ids]
                        sigma_F_pool_other_cal = np.where(component_support_other, sigma_F_pool_iso_other, sigma_F_pool_var_other)
                    else:
                        sigma_F_pool_other_cal = sigma_F_pool_var_other
                    sigma_F_pool_other_cal = sigma_F_pool_other_cal.reshape(sigma_F_pool_shape_other)
                    sigma_F_pool_mean_other, sigma_F_pool_max_other = _force_summary_from_flat(
                        sigma_F_pool_other_cal, pool_frames
                    )
                    sigma_F_pool_mean_other_thin = sigma_F_pool_mean_other[thin_idx].astype(float)
                    sigma_F_pool_max_other_thin = sigma_F_pool_max_other[thin_idx].astype(float)
                else:
                    sigma_F_pool_mean_other_thin, sigma_F_pool_max_other_thin = scale_pool_force_summaries(
                        sigma_F_pool_mean_other_thin, sigma_F_pool_max_other_thin, uq_calibrators_other
                    )
                expected_abs_F_mean_other = sigma_F_pool_mean_other_thin * _GAUSSIAN_SIGMA_TO_ABS
                expected_abs_F_max_other = sigma_F_pool_max_other_thin * _GAUSSIAN_SIGMA_TO_ABS

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
        df = pd.DataFrame({"mu": mu_E_pool, "sigma": sigma_E_pool_orig})
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
        good_rows = np.isfinite(F_train_thin).all(axis=1) & np.isfinite(stats_ens.delta_E_frame[train_idx])
        print(f"[Pool-AL] Extracted {good_rows.sum()} valid training frames for GP calibration.")
        if int(good_rows.sum()) < 2:
            raise ValueError("Pool AL needs at least two finite training latent rows for GP calibration")
        train_delta_E_atom = stats_ens.delta_E_frame[train_idx] / train_atom_counts
        alpha_sq, _, _, _, L_chol = calibrate_alpha_reg_gcv(F_train_thin[good_rows], train_delta_E_atom[good_rows])

        calibrator = UQCalibrator()
        mu_E_train = stats_ens.pred_energies[train_idx]
        mu_E_train_atom = mu_E_train / train_atom_counts
        sigma_E_train_atom = sigma_E_raw[train_idx] / train_atom_counts
        calibrator.fit(mu_E_train_atom, sigma_E_train_atom, train_delta_E_atom)
        if self.do_plot:
            calibrator.plot_diagnostics(mu_E_train_atom, sigma_E_train_atom, train_delta_E_atom)

        train_forces = [self.ds["F_true"][i] for i in train_idx]
        train_frames = [self.ds["frames"][i] for i in train_idx]

        # Selection
        print("[Pool-AL] Running windowed active learning on thinned pool ...")
        # Helper to write head-specific selection files
        def save_selected_frames(filename, sel_rel_indices, sigma_E_source, head_name):
            sel_global_idx = thin_idx[sel_rel_indices]
            if len(sel_global_idx) > 0:
                with open(filename, "w") as fh:
                    for orig_idx in sel_global_idx:
                        atoms = pool_frames[orig_idx]
                        if head_name == orig_mace_head:
                            e_raw = float(mu_E_pool[orig_idx])
                        elif mu_E_pool_other is not None:
                            e_raw = float(mu_E_pool_other[orig_idx])
                        else:
                            e_raw = float(mu_E_pool[orig_idx])
                        s_raw = float(sigma_E_source[orig_idx])
                        comment = f"frame={orig_idx}, head={head_name}, e_pred_raw={e_raw:.6f}, s_raw={s_raw:.6f}"
                        write(fh, atoms, format="xyz", comment=comment)
                print(f"[Pool-AL] Saved {len(sel_global_idx)} pool frames to '{filename}'.")

        sel_rel_thin_orig = []
        sel_rel_thin_other = []

        # Diagnostic separate runs for each head if in dual_head_or mode
        if has_multihead and al_multihead_mode == "dual_head_or":
            # 1. Run for original/primary head only
            try:
                print(f"[Pool-AL] Running diagnostic selection for primary head '{orig_mace_head}'...")
                _, sel_rel_thin_orig = adaptive_learning_mig_pool_windowed(
                    pool_frames_thin, F_pool_thin, F_train_thin, alpha_sq, L_chol,
                    forces_train=train_forces, sigma_energy=sigma_energy_train, sigma_force=sigma_force_train,
                    mu_E_frame_train=mu_E_train, mu_E_pool=mu_E_pool_thin, sigma_E_pool=sigma_E_pool_orig_thin,
                    rdf_thresholds=rdf_thresholds,
                    sigma_F_pool_mean=sigma_F_pool_mean_thin, sigma_F_pool_max=sigma_F_pool_max_thin,
                    frame_max_force_pool=frame_max_force_pool_thin,
                    calibration_in_support=calibration_in_support,
                    ood_risk_mask=ood_risk_mask,
                    expected_abs_E_atom=expected_abs_E_atom,
                    expected_abs_F_mean=expected_abs_F_mean,
                    expected_abs_F_max=expected_abs_F_max,
                    train_frames=train_frames,
                    rho_eV=self.eval_cfg.get("rho_eV", 0.002), min_k=self.eval_cfg.get("pool_min_k", 5),
                    window_size=self.eval_cfg.get("pool_window", 100), budget_max=self.eval_cfg.get("budget_max", 50),
                    percentile_gamma=self.eval_cfg.get("percentile_gamma", 99),
                    percentile_F_low=self.eval_cfg.get("percentile_F_low", 99.5),
                    percentile_F_hi=self.eval_cfg.get("percentile_F_hi", 93),
                    hard_sigma_E_atom_min=self.eval_cfg.get("thr_sE_atom", 0.001),
                    hard_sigma_F_mean_min=self.eval_cfg.get("thr_sF_mean", 0.1),
                    hard_sigma_F_max_min=self.eval_cfg.get("thr_sF_max", 0.1),
                    hard_Fmax_train_mult=self.eval_cfg.get("thr_Fmax_mult", 1.5),
                    large_cluster_threshold=self.eval_cfg.get("large_cluster_threshold", 300),
                    surface_relax_factor=self.eval_cfg.get("surface_relax_factor", None),
                    stratify_train_by_size=_parse_bool_like(
                        self.eval_cfg.get("stratify_train_by_size"), True
                    ),
                    size_split_atoms=self.eval_cfg.get(
                        "size_split_atoms", self.eval_cfg.get("large_cluster_threshold", 300)
                    ),
                    hard_floors_from_calibrated_train=_parse_bool_like(
                        self.eval_cfg.get("hard_floors_from_calibrated_train"), bool(force_mode or energy_mode)
                    ),
                    base=f"al_pool_{orig_mace_head}", state=orig_mace_head,
                    pool_indices=thin_idx, diagnostics_collector=al_diagnostics_runs
                )
                save_selected_frames(f"to_DFT_labelling_from_pool_{orig_mace_head}.xyz", sel_rel_thin_orig, sigma_E_pool_orig, orig_mace_head)
            except Exception as e:
                print(f"[Pool-AL] WARNING: Failed to run separate AL diagnostic for head '{orig_mace_head}': {e}")

            # 2. Run for secondary head only
            if sigma_E_pool_other_thin is not None and self.stats_ens_other is not None and self.sigma_E_raw_other is not None and self.sigma_comp_other is not None:
                try:
                    print(f"[Pool-AL] Running diagnostic selection for secondary head '{other_head}'...")
                    good_rows_other = np.isfinite(F_train_thin).all(axis=1) & np.isfinite(self.stats_ens_other.delta_E_frame[self.ds["train_idx"]])
                    alpha_sq_other, _, _, _, L_chol_other = calibrate_alpha_reg_gcv(F_train_thin[good_rows_other], self.stats_ens_other.delta_E_frame[self.ds["train_idx"]][good_rows_other])
                    mu_E_train_other = self.stats_ens_other.pred_energies[self.ds["train_idx"]]
                    
                    other_F_true = self.ds.get("F_singlet_true") if other_head == "singlet" else self.ds.get("F_triplet_true")
                    if other_F_true is None:
                        other_F_true = self.ds["F_true"]
                    train_forces_other = [other_F_true[i] for i in train_idx]

                    _, sel_rel_thin_other = adaptive_learning_mig_pool_windowed(
                        pool_frames_thin, F_pool_thin, F_train_thin, alpha_sq_other, L_chol_other,
                        forces_train=train_forces_other, sigma_energy=sigma_energy_train_other, sigma_force=sigma_force_train_other,
                        mu_E_frame_train=mu_E_train_other, mu_E_pool=mu_E_pool_other_thin, sigma_E_pool=sigma_E_pool_other_thin,
                        rdf_thresholds=rdf_thresholds,
                        sigma_F_pool_mean=sigma_F_pool_mean_other_thin, sigma_F_pool_max=sigma_F_pool_max_other_thin,
                        frame_max_force_pool=frame_max_force_pool_thin,
                        calibration_in_support=calibration_in_support_other,
                        ood_risk_mask=ood_risk_mask_other,
                        expected_abs_E_atom=expected_abs_E_atom_other,
                        expected_abs_F_mean=expected_abs_F_mean_other,
                        expected_abs_F_max=expected_abs_F_max_other,
                        train_frames=train_frames,
                        rho_eV=self.eval_cfg.get("rho_eV", 0.002), min_k=self.eval_cfg.get("pool_min_k", 5),
                        window_size=self.eval_cfg.get("pool_window", 100), budget_max=self.eval_cfg.get("budget_max", 50),
                        percentile_gamma=self.eval_cfg.get("percentile_gamma", 99),
                        percentile_F_low=self.eval_cfg.get("percentile_F_low", 99.5),
                        percentile_F_hi=self.eval_cfg.get("percentile_F_hi", 93),
                        hard_sigma_E_atom_min=self.eval_cfg.get("thr_sE_atom", 0.001),
                        hard_sigma_F_mean_min=self.eval_cfg.get("thr_sF_mean", 0.1),
                        hard_sigma_F_max_min=self.eval_cfg.get("thr_sF_max", 0.1),
                        hard_Fmax_train_mult=self.eval_cfg.get("thr_Fmax_mult", 1.5),
                        large_cluster_threshold=self.eval_cfg.get("large_cluster_threshold", 300),
                        surface_relax_factor=self.eval_cfg.get("surface_relax_factor", None),
                        stratify_train_by_size=_parse_bool_like(
                            self.eval_cfg.get("stratify_train_by_size"), True
                        ),
                        size_split_atoms=self.eval_cfg.get(
                            "size_split_atoms", self.eval_cfg.get("large_cluster_threshold", 300)
                        ),
                        hard_floors_from_calibrated_train=_parse_bool_like(
                            self.eval_cfg.get("hard_floors_from_calibrated_train"), bool(force_mode_other or energy_mode_other)
                        ),
                        base=f"al_pool_{other_head}", state=other_head,
                        pool_indices=thin_idx, diagnostics_collector=al_diagnostics_runs
                    )
                    save_selected_frames(f"to_DFT_labelling_from_pool_{other_head}.xyz", sel_rel_thin_other, sigma_E_pool_other, other_head)
                except Exception as e:
                    print(f"[Pool-AL] WARNING: Failed to run separate AL diagnostic for head '{other_head}': {e}")

            # Merge choices
            merge_selections = self.eval_cfg.get("merge_al_selections", True)
            if merge_selections:
                sel_rel_thin = sorted(list(set(sel_rel_thin_orig) | set(sel_rel_thin_other)))
                print(f"[Pool-AL] Merging selections: Union of {orig_mace_head} and {other_head} contains {len(sel_rel_thin)} unique frames.")
            else:
                sel_rel_thin = sel_rel_thin_orig
                print(f"[Pool-AL] merge_al_selections is False. Final selection set to primary head selector pass ({len(sel_rel_thin)} frames).")

        else:
            # Single-head fallback
            try:
                _, sel_rel_thin_orig = adaptive_learning_mig_pool_windowed(
                    pool_frames_thin, F_pool_thin, F_train_thin, alpha_sq, L_chol,
                    forces_train=train_forces, sigma_energy=sigma_energy_train, sigma_force=sigma_force_train,
                    mu_E_frame_train=mu_E_train, mu_E_pool=mu_E_pool_thin, sigma_E_pool=sigma_E_pool_thin,
                    rdf_thresholds=rdf_thresholds,
                    sigma_F_pool_mean=sigma_F_pool_mean_thin, sigma_F_pool_max=sigma_F_pool_max_thin,
                    frame_max_force_pool=frame_max_force_pool_thin,
                    calibration_in_support=calibration_in_support,
                    ood_risk_mask=ood_risk_mask,
                    expected_abs_E_atom=expected_abs_E_atom,
                    expected_abs_F_mean=expected_abs_F_mean,
                    expected_abs_F_max=expected_abs_F_max,
                    train_frames=train_frames,
                    rho_eV=self.eval_cfg.get("rho_eV", 0.002), min_k=self.eval_cfg.get("pool_min_k", 5),
                    window_size=self.eval_cfg.get("pool_window", 100), budget_max=self.eval_cfg.get("budget_max", 50),
                    percentile_gamma=self.eval_cfg.get("percentile_gamma", 99),
                    percentile_F_low=self.eval_cfg.get("percentile_F_low", 99.5),
                    percentile_F_hi=self.eval_cfg.get("percentile_F_hi", 93),
                    hard_sigma_E_atom_min=self.eval_cfg.get("thr_sE_atom", 0.001),
                    hard_sigma_F_mean_min=self.eval_cfg.get("thr_sF_mean", 0.1),
                    hard_sigma_F_max_min=self.eval_cfg.get("thr_sF_max", 0.1),
                    hard_Fmax_train_mult=self.eval_cfg.get("thr_Fmax_mult", 1.5),
                    large_cluster_threshold=self.eval_cfg.get("large_cluster_threshold", 300),
                    surface_relax_factor=self.eval_cfg.get("surface_relax_factor", None),
                    stratify_train_by_size=_parse_bool_like(
                        self.eval_cfg.get("stratify_train_by_size"), True
                    ),
                    size_split_atoms=self.eval_cfg.get(
                        "size_split_atoms", self.eval_cfg.get("large_cluster_threshold", 300)
                    ),
                    hard_floors_from_calibrated_train=_parse_bool_like(
                        self.eval_cfg.get("hard_floors_from_calibrated_train"), bool(force_mode or energy_mode)
                    ),
                    state=orig_mace_head, pool_indices=thin_idx,
                    diagnostics_collector=al_diagnostics_runs,
                )
                save_selected_frames(f"to_DFT_labelling_from_pool_{orig_mace_head}.xyz", sel_rel_thin_orig, sigma_E_pool, orig_mace_head)
                sel_rel_thin = sel_rel_thin_orig
            except Exception as e:
                print(f"[Pool-AL] WARNING: Failed to run single head selection pass: {e}")
                sel_rel_thin = []

        if al_diagnostics_runs:
            al_csv_path = self.eval_cfg.get("al_diagnostics_csv", "al_pool_diagnostics.csv")
            write_pool_al_diagnostics_csv(al_diagnostics_runs, al_csv_path)
            if _parse_bool_like(self.eval_cfg.get("plot_AL"), False):
                al_plot_dir = self.eval_cfg.get("al_plot_dir", "al_plots")
                os.makedirs(al_plot_dir, exist_ok=True)
                generate_al_diagnostic_plots(
                    al_csv_path,
                    out_dir=al_plot_dir,
                    window=int(self.eval_cfg.get("al_plot_smoothing_window", 50)),
                    drop_first=_parse_bool_like(self.eval_cfg.get("al_plot_drop_first"), True),
                    dpi=int(self.eval_cfg.get("al_plot_dpi", 300)),
                )

        # Output
        sel_global_idx = thin_idx[sel_rel_thin]
        if len(sel_global_idx) > 0:
            thin_lookup = {int(orig): int(rel) for rel, orig in enumerate(thin_idx)}
            with open("to_DFT_labelling_from_pool.xyz", "w") as fh:
                for orig_idx in sel_global_idx:
                    atoms = pool_frames[orig_idx]
                    if has_multihead and al_multihead_mode == "dual_head_or":
                        if orig_mace_head == "singlet":
                            e_singlet = float(mu_E_pool[orig_idx])
                            s_singlet = float(sigma_E_pool_orig[orig_idx])
                            e_triplet = float(mu_E_pool_other[orig_idx]) if mu_E_pool_other is not None else 0.0
                            s_triplet = float(sigma_E_pool_other[orig_idx]) if sigma_E_pool_other is not None else 0.0
                        else:
                            e_triplet = float(mu_E_pool[orig_idx])
                            s_triplet = float(sigma_E_pool_orig[orig_idx])
                            e_singlet = float(mu_E_pool_other[orig_idx]) if mu_E_pool_other is not None else 0.0
                            s_singlet = float(sigma_E_pool_other[orig_idx]) if sigma_E_pool_other is not None else 0.0
                        comment = f"frame={orig_idx}, e_singlet={e_singlet:.4f}, s_singlet={s_singlet:.4f}, e_triplet={e_triplet:.4f}, s_triplet={s_triplet:.4f}"
                    else:
                        rel_idx = thin_lookup[int(orig_idx)]
                        e_raw, s_raw = float(mu_E_pool[orig_idx]), float(sigma_E_pool[orig_idx])
                        n_atoms = float(len(atoms))
                        e_atom_raw = e_raw / n_atoms
                        s_atom_raw = s_raw / n_atoms
                        _, _, bias_corr_arr = calibrator.calibrate(
                            np.array([e_atom_raw]), np.array([s_atom_raw])
                        )
                        e_atom_cal = e_atom_raw - float(bias_corr_arr[0])
                        exp_abs_e_atom = float(expected_abs_E_atom[rel_idx])
                        comment = (
                            f"frame={orig_idx}, e_pred_raw={e_raw:.6f}, "
                            f"e_pred_atom={e_atom_raw:.8f}, sigma_E_atom_raw={s_atom_raw:.8f}, "
                            f"bias_corr_atom={-float(bias_corr_arr[0]):.8f}, "
                            f"expected_abs_E_atom={exp_abs_e_atom:.8f}, "
                            f"expected_abs_F_mean={float(expected_abs_F_mean[rel_idx]):.8f}, "
                            f"calibration_in_support={int(calibration_in_support[rel_idx])}, "
                            f"ood_risk={int(ood_risk_mask[rel_idx])}, "
                            f"BALLPARK_E_atom=[{e_atom_cal - exp_abs_e_atom:.8f}, {e_atom_cal + exp_abs_e_atom:.8f}]"
                        )
                    write(fh, atoms, format="xyz", comment=comment)
            print(f"[Pool-AL] Saved {len(sel_global_idx)} pool frames to 'to_DFT_labelling_from_pool.xyz'.")


def run_eval(config):
    """Entry point for evaluation."""
    if config is None:
        print("Error: Config not loaded.")
        return
    pipeline = EvaluationPipeline(config)
    pipeline.run()
