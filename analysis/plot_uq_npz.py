#!/usr/bin/env python3
"""Regenerate UQ diagnostic plots from a saved UQ NPZ archive.

Example:
    python analysis/plot_uq_npz.py uq_plots/uq_plot_data_eval_ensemble.npz

The main plot set is the same one produced during postprocessing runtime via
``generate_uq_plots``. Optional extra plots add compact overconfidence and
error-quantile diagnostics directly from the same NPZ arrays.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def import_generate_uq_plots():
    """Import repo plotting code when this script is run inside or outside the repo.

    Resolution order:
    1. Normal Python import, useful when ``orchestr_ai`` is installed or PYTHONPATH is set.
    2. ``ORCHESTR_AI_ROOT=/path/to/Orchestr.AI`` environment variable.
    3. Parent directories around this script containing ``src/orchestr_ai``.
    """
    try:
        from orchestr_ai.postprocessing.plotting import generate_uq_plots as func

        return func
    except ImportError as first_error:
        import os

        candidates = []
        env_root = os.environ.get("ORCHESTR_AI_ROOT")
        if env_root:
            candidates.append(Path(env_root).expanduser())

        script_path = Path(__file__).resolve()
        candidates.extend(script_path.parents)

        for root in candidates:
            src_dir = root / "src"
            if not (src_dir / "orchestr_ai").exists():
                continue
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            try:
                from orchestr_ai.postprocessing.plotting import generate_uq_plots as func

                return func
            except ImportError:
                continue

        print(
            "[Warning] Could not import orchestr_ai plotting code. "
            "Using the self-contained fallback plots instead."
        )
        return standalone_generate_uq_plots


def normal_interval_coverage(delta, sigma, p_values):
    """Empirical coverage for central Gaussian intervals without scipy."""
    delta = np.asarray(delta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mask = np.isfinite(delta) & np.isfinite(sigma) & (sigma > 0)
    delta = np.abs(delta[mask])
    sigma = sigma[mask]
    if delta.size == 0:
        return np.full_like(p_values, np.nan, dtype=float)

    # Normal two-sided central interval multipliers for common p grid.
    z_lookup = {
        0.00: 0.0,
        0.05: 0.0627068,
        0.10: 0.1256613,
        0.15: 0.1891184,
        0.20: 0.2533471,
        0.25: 0.3186394,
        0.30: 0.3853205,
        0.35: 0.4537622,
        0.40: 0.5244005,
        0.45: 0.5977601,
        0.50: 0.6744898,
        0.55: 0.7554150,
        0.60: 0.8416212,
        0.65: 0.9345893,
        0.70: 1.0364334,
        0.75: 1.1503494,
        0.80: 1.2815516,
        0.85: 1.4395315,
        0.90: 1.6448536,
        0.95: 1.9599640,
        1.00: np.inf,
    }
    out = []
    for p in p_values:
        z = z_lookup.get(round(float(p), 2))
        if z is None:
            z = np.nan
        out.append(float(np.mean(delta <= z * sigma)) if np.isfinite(z) else 1.0)
    return np.asarray(out, dtype=float)


def rmse_rmv_bins(delta, sigma, n_bins=10):
    delta = np.asarray(delta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mask = np.isfinite(delta) & np.isfinite(sigma) & (sigma > 0)
    delta = delta[mask]
    sigma = sigma[mask]
    if delta.size < n_bins:
        return np.array([]), np.array([])
    order = np.argsort(sigma)
    chunks = np.array_split(order, n_bins)
    rmse = np.array([np.sqrt(np.mean(delta[idx] ** 2)) for idx in chunks if idx.size], dtype=float)
    rmv = np.array([np.sqrt(np.mean(sigma[idx] ** 2)) for idx in chunks if idx.size], dtype=float)
    return rmse, rmv


def standalone_pick(data, raw_base, calibration):
    key = {
        "var": f"{raw_base}_cal_var",
        "iso": f"{raw_base}_cal_iso",
        "legacy": f"{raw_base}_cal",
    }[calibration]
    if key in data.files:
        return data[key]
    fallback = f"{raw_base}_uncal"
    print(f"[Warning] Missing {key}; falling back to {fallback}.")
    return data[fallback]


def scatter_error_sigma(delta, sigma_raw, sigma_cal, title, out_path):
    err = np.abs(np.asarray(delta, dtype=float))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, sigma, label, color in (
        (axs[0], sigma_raw, "Raw", "royalblue"),
        (axs[1], sigma_cal, "Calibrated", "crimson"),
    ):
        sigma, err_plot = finite_positive_pair(sigma, err)
        ax.scatter(sigma, err_plot, s=8, alpha=0.25, color=color)
        if sigma.size:
            lo = min(float(np.nanmin(sigma)), float(np.nanmin(err_plot)))
            hi = max(float(np.nanmax(sigma)), float(np.nanmax(err_plot)))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(label)
        ax.set_xlabel("Predicted uncertainty sigma")
        ax.set_ylabel("Absolute error")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def overlay_error_sigma(delta, sigma_raw, sigma_cal, title, out_path):
    """Overlay raw and calibrated |error| vs sigma on one log-log axis."""
    err = np.abs(np.asarray(delta, dtype=float))
    raw_sigma, raw_err = finite_positive_pair(sigma_raw, err)
    cal_sigma, cal_err = finite_positive_pair(sigma_cal, err)
    if raw_sigma.size == 0 or cal_sigma.size == 0:
        print(f"[Warning] Skipping {out_path.name}: no finite positive sigma values.")
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(raw_sigma, raw_err, s=7, alpha=0.18, color="royalblue", label="Raw")
    ax.scatter(cal_sigma, cal_err, s=7, alpha=0.18, color="crimson", label="Calibrated")

    all_x = np.concatenate([raw_sigma, cal_sigma])
    all_y = np.concatenate([raw_err, cal_err])
    lo = min(float(np.nanmin(all_x)), float(np.nanmin(all_y)))
    hi = max(float(np.nanmax(all_x)), float(np.nanmax(all_y)))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="error = sigma")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted uncertainty sigma")
    ax.set_ylabel("Absolute error")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_coverage(p, raw, cal, title, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(p, p, "k--", linewidth=1, label="Ideal")
    ax.plot(p, raw, marker="o", label="Raw", color="royalblue")
    ax.plot(p, cal, marker="o", label="Calibrated", color="crimson")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_sigma_hist(sigma_raw, sigma_cal, title, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for sigma, label, color in (
        (sigma_raw, "Raw", "royalblue"),
        (sigma_cal, "Calibrated", "crimson"),
    ):
        sigma = np.asarray(sigma, dtype=float)
        sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
        ax.hist(sigma, bins=60, density=True, alpha=0.45, label=label, color=color)
    ax.set_xlabel("Predicted uncertainty sigma")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_rmse_rmv(delta, sigma_raw, sigma_cal, title, out_path):
    rmse_raw, rmv_raw = rmse_rmv_bins(delta, sigma_raw)
    rmse_cal, rmv_cal = rmse_rmv_bins(delta, sigma_cal)
    if rmse_raw.size == 0 or rmse_cal.size == 0:
        print(f"[Warning] Skipping {out_path.name}: not enough finite data.")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rmv_raw, rmse_raw, marker="o", label="Raw", color="royalblue")
    ax.plot(rmv_cal, rmse_cal, marker="o", label="Calibrated", color="crimson")
    hi = max(np.nanmax(rmse_raw), np.nanmax(rmv_raw), np.nanmax(rmse_cal), np.nanmax(rmv_cal))
    ax.plot([0, hi], [0, hi], "k--", linewidth=1, label="Ideal")
    ax.set_xlabel("RMV per uncertainty bin")
    ax.set_ylabel("RMSE per uncertainty bin")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_z_hist(delta, sigma_raw, sigma_cal, title, out_path):
    err = np.asarray(delta, dtype=float)
    raw_sigma, raw_delta = finite_positive_pair(sigma_raw, err)
    cal_sigma, cal_delta = finite_positive_pair(sigma_cal, err)
    if raw_sigma.size == 0 or cal_sigma.size == 0:
        print(f"[Warning] Skipping {out_path.name}: no finite positive sigma values.")
        return
    z_raw = raw_delta / raw_sigma
    z_cal = cal_delta / cal_sigma
    lim = np.nanpercentile(np.abs(np.concatenate([z_raw, z_cal])), 99)
    bins = np.linspace(-max(lim, 3.0), max(lim, 3.0), 80)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(z_raw, bins=bins, density=True, alpha=0.45, label="Raw", color="royalblue")
    ax.hist(z_cal, bins=bins, density=True, alpha=0.45, label="Calibrated", color="crimson")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("z = error / sigma")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def standalone_generate_uq_plots(
    npz_plot_data_path,
    set_name,
    set_uq,
    ensemble_size=None,
    norm_energy=False,
    calibration="var",
):
    """Small dependency-free fallback for copied use outside Orchestr.AI."""
    data = np.load(npz_plot_data_path, allow_pickle=True)
    plot_dir = Path(npz_plot_data_path).resolve().parent
    ens = f"_ens{ensemble_size}" if ensemble_size else ""
    base = f"{set_name.lower()}_{set_uq.lower()}_{calibration}{ens}_standalone"
    title = f"{set_name} ({set_uq}, {calibration}{ens})"

    delta_c = data["delta_comp"]
    sigma_c_raw = data["sigma_comp_uncal"]
    sigma_c_cal = standalone_pick(data, "sigma_comp", calibration)

    p = data["p_thresholds"] if "p_thresholds" in data.files else np.linspace(0, 1, 21)
    cov_raw = data["coverage_uncal"] if "coverage_uncal" in data.files else normal_interval_coverage(delta_c, sigma_c_raw, p)
    cov_cal_key = f"coverage_cal_{calibration}"
    cov_cal = data[cov_cal_key] if cov_cal_key in data.files else normal_interval_coverage(delta_c, sigma_c_cal, p)

    plot_coverage(p, cov_raw, cov_cal, f"{title} - coverage forces", plot_dir / f"{base}_coverage_forces.png")
    plot_sigma_hist(sigma_c_raw, sigma_c_cal, f"{title} - sigma density forces", plot_dir / f"{base}_sigma_density_forces.png")
    scatter_error_sigma(delta_c, sigma_c_raw, sigma_c_cal, f"{title} - |force error| vs sigma", plot_dir / f"{base}_err_unc_forces.png")
    overlay_error_sigma(delta_c, sigma_c_raw, sigma_c_cal, f"{title} - force overlay", plot_dir / f"{base}_err_unc_forces_overlay.png")
    plot_rmse_rmv(delta_c, sigma_c_raw, sigma_c_cal, f"{title} - RMSE vs RMV forces", plot_dir / f"{base}_rmse_rmv_forces.png")
    plot_z_hist(delta_c, sigma_c_raw, sigma_c_cal, f"{title} - z-score forces", plot_dir / f"{base}_z_hist_forces.png")

    has_energy = all(k in data.files for k in ("delta_energy", "sigma_energy_uncal")) and data["delta_energy"].size
    if has_energy:
        delta_e = data["delta_energy"]
        sigma_e_raw = data["sigma_energy_uncal"]
        sigma_e_cal = standalone_pick(data, "sigma_energy", calibration)
        if norm_energy and "n_atoms_per_frame" in data.files:
            n_atoms = data["n_atoms_per_frame"]
            delta_e = delta_e / n_atoms
            sigma_e_raw = sigma_e_raw / n_atoms
            sigma_e_cal = sigma_e_cal / n_atoms

        cov_raw_e = data["coverage_uncal_e"] if "coverage_uncal_e" in data.files else normal_interval_coverage(delta_e, sigma_e_raw, p)
        cov_cal_e_key = f"coverage_cal_{calibration}_e"
        cov_cal_e = data[cov_cal_e_key] if cov_cal_e_key in data.files else normal_interval_coverage(delta_e, sigma_e_cal, p)

        plot_coverage(p, cov_raw_e, cov_cal_e, f"{title} - coverage energy", plot_dir / f"{base}_coverage_energy.png")
        plot_sigma_hist(sigma_e_raw, sigma_e_cal, f"{title} - sigma density energy", plot_dir / f"{base}_sigma_density_energy.png")
        scatter_error_sigma(delta_e, sigma_e_raw, sigma_e_cal, f"{title} - |energy error| vs sigma", plot_dir / f"{base}_err_unc_energy.png")
        overlay_error_sigma(delta_e, sigma_e_raw, sigma_e_cal, f"{title} - energy overlay", plot_dir / f"{base}_err_unc_energy_overlay.png")
        plot_rmse_rmv(delta_e, sigma_e_raw, sigma_e_cal, f"{title} - RMSE vs RMV energy", plot_dir / f"{base}_rmse_rmv_energy.png")
        plot_z_hist(delta_e, sigma_e_raw, sigma_e_cal, f"{title} - z-score energy", plot_dir / f"{base}_z_hist_energy.png")

    print(f"[INFO] Finished standalone UQ plotting for {npz_plot_data_path}")


generate_uq_plots = import_generate_uq_plots()


def infer_metadata(npz_path):
    """Infer split name, UQ tag, and ensemble size from standard filenames."""
    stem = Path(npz_path).stem
    if stem.startswith("uq_plot_data_"):
        stem = stem[len("uq_plot_data_") :]

    parts = stem.split("_")
    set_name = parts[0].capitalize() if parts else "Eval"
    set_uq = parts[1] if len(parts) > 1 else "ensemble"
    ensemble_size = None

    for part in parts:
        if part.startswith("ens"):
            try:
                ensemble_size = int(part[3:])
            except ValueError:
                pass

    return set_name, set_uq, ensemble_size


def available_calibrations(data):
    """Return calibration modes available in this archive."""
    modes = []
    if "sigma_comp_cal_var" in data.files or "sigma_energy_cal_var" in data.files:
        modes.append("var")
    if "sigma_comp_cal_iso" in data.files or "sigma_energy_cal_iso" in data.files:
        modes.append("iso")
    if "sigma_comp_cal" in data.files or "sigma_energy_cal" in data.files:
        modes.append("legacy")
    return modes or ["var"]


def pick_calibrated(data, base_name, mode):
    if mode == "var":
        key = f"{base_name}_cal_var"
    elif mode == "iso":
        key = f"{base_name}_cal_iso"
    elif mode == "legacy":
        key = f"{base_name}_cal"
    else:
        raise ValueError(f"Unsupported calibration mode: {mode}")
    return data[key] if key in data.files else None


def finite_positive_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return x[mask], y[mask]


def finite_regression_arrays(delta, sigma):
    delta = np.asarray(delta, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mask = np.isfinite(delta) & np.isfinite(sigma) & (sigma > 0)
    return delta[mask], sigma[mask]


def calibration_errors(p_nominal, empirical):
    p_nominal = np.asarray(p_nominal, dtype=float)
    empirical = np.asarray(empirical, dtype=float)
    mask = np.isfinite(p_nominal) & np.isfinite(empirical)
    if not np.any(mask):
        return np.nan, np.nan
    gap = empirical[mask] - p_nominal[mask]
    return float(np.mean(np.abs(gap))), float(np.max(np.abs(gap)))


def nearest_coverage(p_nominal, empirical, target=0.95):
    p_nominal = np.asarray(p_nominal, dtype=float)
    empirical = np.asarray(empirical, dtype=float)
    if p_nominal.size == 0 or empirical.size == 0:
        return np.nan
    idx = int(np.nanargmin(np.abs(p_nominal - target)))
    return float(empirical[idx])


def gaussian_nll(delta, sigma):
    delta, sigma = finite_regression_arrays(delta, sigma)
    if delta.size == 0:
        return np.nan
    var = np.maximum(sigma ** 2, 1e-30)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * var) + 0.5 * delta ** 2 / var))


def regression_metrics(delta, sigma, p_nominal, empirical_coverage):
    delta, sigma = finite_regression_arrays(delta, sigma)
    if delta.size == 0:
        return {}
    abs_delta = np.abs(delta)
    z = delta / sigma
    ce, mce = calibration_errors(p_nominal, empirical_coverage)
    return {
        "n": int(delta.size),
        "mae": float(np.mean(abs_delta)),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "mean_sigma": float(np.mean(sigma)),
        "median_sigma": float(np.median(sigma)),
        "rmv_sharpness": float(np.sqrt(np.mean(sigma ** 2))),
        "sharpness_var": float(np.mean(sigma ** 2)),
        "coverage_95": nearest_coverage(p_nominal, empirical_coverage, 0.95),
        "coverage_ce": ce,
        "coverage_mce": mce,
        "nll": gaussian_nll(delta, sigma),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "mean_abs_z": float(np.mean(np.abs(z))),
        "frac_abs_z_gt_1": float(np.mean(np.abs(z) > 1.0)),
        "frac_abs_z_gt_1p96": float(np.mean(np.abs(z) > 1.96)),
        "frac_abs_z_gt_3": float(np.mean(np.abs(z) > 3.0)),
    }


def trusted_stats(delta, sigma, threshold):
    if threshold is None:
        return None
    delta, sigma = finite_regression_arrays(delta, sigma)
    if delta.size == 0:
        return None
    trusted = 2.0 * sigma < float(threshold)
    high_error = np.abs(delta) > float(threshold)
    trusted_count = int(np.sum(trusted))
    false_trusted = int(np.sum(trusted & high_error))
    high_error_total = int(np.sum(high_error))
    return {
        "threshold": float(threshold),
        "trusted_count": trusted_count,
        "trusted_fraction": float(trusted_count / max(1, delta.size)),
        "false_trusted_count": false_trusted,
        "false_trusted_rate": float(false_trusted / max(1, trusted_count)),
        "high_error_total": high_error_total,
        "high_error_fraction": float(high_error_total / max(1, delta.size)),
    }


def plot_average_miscalibration(p, raw, cal, title, out_path):
    ce_raw, _ = calibration_errors(p, raw)
    ce_cal, _ = calibration_errors(p, cal)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot(p, p, "k--", linewidth=1, label="Ideal")
    ax.plot(p, raw, marker="o", color="royalblue", label=f"Raw CE={ce_raw:.3f}")
    ax.plot(p, cal, marker="o", color="crimson", label=f"Cal CE={ce_cal:.3f}")
    ax.fill_between(p, raw, p, color="royalblue", alpha=0.12)
    ax.fill_between(p, cal, p, color="crimson", alpha=0.12)
    ax.set_xlabel("Predicted proportion in interval")
    ax.set_ylabel("Observed proportion in interval")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_coverage_gap(p, raw, cal, title, out_path):
    raw_gap = np.asarray(raw, dtype=float) - np.asarray(p, dtype=float)
    cal_gap = np.asarray(cal, dtype=float) - np.asarray(p, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.fill_between(p, raw_gap, 0, where=raw_gap < 0, color="tab:red", alpha=0.12)
    ax.fill_between(p, raw_gap, 0, where=raw_gap > 0, color="tab:green", alpha=0.12)
    ax.plot(p, raw_gap, marker="o", color="royalblue", label="Raw")
    ax.plot(p, cal_gap, marker="o", color="crimson", label="Calibrated")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical - nominal coverage")
    ax.set_title(title)
    ax.text(0.02, 0.96, "< 0 overconfident\n> 0 underconfident", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def variance_bins(delta, sigma, n_bins=12):
    delta, sigma = finite_regression_arrays(delta, sigma)
    if delta.size < n_bins:
        return np.array([]), np.array([])
    order = np.argsort(sigma)
    chunks = np.array_split(order, n_bins)
    pred_var = np.array([np.mean(sigma[idx] ** 2) for idx in chunks if idx.size], dtype=float)
    obs_var = np.array([np.mean(delta[idx] ** 2) for idx in chunks if idx.size], dtype=float)
    return pred_var, obs_var


def plot_variance_calibration(delta, sigma_raw, sigma_cal, title, out_path):
    raw_x, raw_y = variance_bins(delta, sigma_raw)
    cal_x, cal_y = variance_bins(delta, sigma_cal)
    if raw_x.size == 0 or cal_x.size == 0:
        print(f"[Warning] Skipping {out_path.name}: not enough finite data.")
        return
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(raw_x, raw_y, marker="o", color="royalblue", label="Raw")
    ax.plot(cal_x, cal_y, marker="o", color="crimson", label="Calibrated")
    all_vals = np.concatenate([raw_x, raw_y, cal_x, cal_y])
    lo = max(float(np.nanmin(all_vals[all_vals > 0])) * 0.8, 1e-30)
    hi = float(np.nanmax(all_vals)) * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="ideal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean predicted variance, mean(sigma^2)")
    ax.set_ylabel("Mean squared error, mean(error^2)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_trusted_summary(raw_stats, cal_stats, title, out_path):
    if raw_stats is None or cal_stats is None:
        return
    labels = ["Raw", "Calibrated"]
    trusted = [raw_stats["trusted_fraction"] * 100.0, cal_stats["trusted_fraction"] * 100.0]
    false = [raw_stats["false_trusted_rate"] * 100.0, cal_stats["false_trusted_rate"] * 100.0]
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axs[0].bar(labels, trusted, color=["royalblue", "crimson"], alpha=0.8)
    axs[0].set_ylabel("Trusted points (%)")
    axs[0].set_title("2 sigma below threshold")
    axs[1].bar(labels, false, color=["royalblue", "crimson"], alpha=0.8)
    axs[1].set_ylabel("False trusted (%)")
    axs[1].set_title("High-error among trusted")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_overconfidence(delta, sigma_raw, sigma_cal, title, out_path):
    """Plot |error| / sigma distributions for raw and calibrated uncertainties."""
    err = np.abs(np.asarray(delta, dtype=float))
    raw_sigma, raw_err = finite_positive_pair(sigma_raw, err)
    cal_sigma, cal_err = finite_positive_pair(sigma_cal, err)
    if raw_sigma.size == 0 or cal_sigma.size == 0:
        print(f"[Warning] Skipping {out_path.name}: no finite positive sigma values.")
        return

    raw_ratio = raw_err / raw_sigma
    cal_ratio = cal_err / cal_sigma
    upper = np.nanpercentile(np.concatenate([raw_ratio, cal_ratio]), 99.5)
    bins = np.linspace(0.0, max(upper, 3.0), 80)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw_ratio, bins=bins, density=True, alpha=0.45, label="Raw", color="royalblue")
    ax.hist(cal_ratio, bins=bins, density=True, alpha=0.45, label="Calibrated", color="crimson")
    for threshold in (1, 2, 3):
        ax.axvline(threshold, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("|error| / sigma")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def plot_error_quantiles(delta, sigma, title, out_path, n_bins=12):
    """Plot absolute-error quantiles as a function of predicted uncertainty."""
    err = np.abs(np.asarray(delta, dtype=float))
    sigma, err = finite_positive_pair(sigma, err)
    if sigma.size < n_bins:
        print(f"[Warning] Skipping {out_path.name}: not enough finite points.")
        return

    order = np.argsort(sigma)
    sigma = sigma[order]
    err = err[order]
    chunks = np.array_split(np.arange(sigma.size), n_bins)

    centers, q50, q68, q90, q95 = [], [], [], [], []
    for chunk in chunks:
        if chunk.size == 0:
            continue
        centers.append(float(np.median(sigma[chunk])))
        q50.append(float(np.percentile(err[chunk], 50)))
        q68.append(float(np.percentile(err[chunk], 68)))
        q90.append(float(np.percentile(err[chunk], 90)))
        q95.append(float(np.percentile(err[chunk], 95)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, q50, marker="o", label="median |error|")
    ax.plot(centers, q68, marker="o", label="68% |error|")
    ax.plot(centers, q90, marker="o", label="90% |error|")
    ax.plot(centers, q95, marker="o", label="95% |error|")
    ax.plot(centers, centers, "k--", linewidth=1, label="error = sigma")
    ax.set_xlabel("Predicted uncertainty sigma")
    ax.set_ylabel("Absolute error quantile")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[INFO] Saved {out_path}")


def generate_extra_plots(npz_path, calibration, set_name, set_uq):
    data = np.load(npz_path, allow_pickle=True)
    plot_dir = Path(npz_path).resolve().parent
    base = f"{set_name.lower()}_{set_uq.lower()}_{calibration}_extra"

    if "delta_comp" in data.files and "sigma_comp_uncal" in data.files:
        sigma_cal = pick_calibrated(data, "sigma_comp", calibration)
        if sigma_cal is not None:
            plot_overconfidence(
                data["delta_comp"],
                data["sigma_comp_uncal"],
                sigma_cal,
                f"{set_name} ({set_uq}, {calibration}) - force overconfidence",
                plot_dir / f"{base}_force_overconfidence.png",
            )
            plot_error_quantiles(
                data["delta_comp"],
                sigma_cal,
                f"{set_name} ({set_uq}, {calibration}) - force error quantiles",
                plot_dir / f"{base}_force_error_quantiles.png",
            )

    has_energy = all(k in data.files for k in ("delta_energy", "sigma_energy_uncal"))
    if has_energy and data["delta_energy"].size:
        sigma_cal_e = pick_calibrated(data, "sigma_energy", calibration)
        if sigma_cal_e is not None:
            plot_overconfidence(
                data["delta_energy"],
                data["sigma_energy_uncal"],
                sigma_cal_e,
                f"{set_name} ({set_uq}, {calibration}) - energy overconfidence",
                plot_dir / f"{base}_energy_overconfidence.png",
            )
            plot_error_quantiles(
                data["delta_energy"],
                sigma_cal_e,
                f"{set_name} ({set_uq}, {calibration}) - energy error quantiles",
                plot_dir / f"{base}_energy_error_quantiles.png",
            )


def coverage_from_npz_or_compute(data, target, mode, delta, sigma):
    p = data["p_thresholds"] if "p_thresholds" in data.files else np.linspace(0, 1, 21)
    if mode == "raw":
        key = "coverage_uncal_e" if target == "energy" else "coverage_uncal"
    else:
        key = f"coverage_cal_{mode}_e" if target == "energy" else f"coverage_cal_{mode}"
    cov = data[key] if key in data.files else normal_interval_coverage(delta, sigma, p)
    return np.asarray(p, dtype=float), np.asarray(cov, dtype=float)


def format_metric(value):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{float(value):.6g}"


def interpretation_block(target, raw_metrics, cal_metrics, mode):
    lines = [f"{target.upper()} INTERPRETATION"]
    raw_cov = raw_metrics.get("coverage_95", np.nan)
    cal_cov = cal_metrics.get("coverage_95", np.nan)
    raw_z = raw_metrics.get("z_std", np.nan)
    cal_z = cal_metrics.get("z_std", np.nan)
    raw_ce = raw_metrics.get("coverage_ce", np.nan)
    cal_ce = cal_metrics.get("coverage_ce", np.nan)

    if np.isfinite(raw_cov):
        if raw_cov < 0.95:
            lines.append(f"- Raw uncertainties are overconfident at 95% coverage ({raw_cov:.3f} < 0.950): intervals are too narrow.")
        elif raw_cov > 0.95:
            lines.append(f"- Raw uncertainties are underconfident at 95% coverage ({raw_cov:.3f} > 0.950): intervals are too wide.")
        else:
            lines.append("- Raw uncertainties match the 95% coverage target exactly on this grid.")
    if np.isfinite(cal_cov):
        direction = "overconfident" if cal_cov < 0.95 else "underconfident" if cal_cov > 0.95 else "well aligned"
        lines.append(f"- {mode} calibration gives 95% coverage {cal_cov:.3f}, so the calibrated intervals are {direction} at this level.")
    if np.isfinite(raw_ce) and np.isfinite(cal_ce):
        if cal_ce < raw_ce:
            lines.append(f"- Coverage calibration improves: CE decreases from {raw_ce:.3f} to {cal_ce:.3f}.")
        else:
            lines.append(f"- Coverage CE does not improve for this mode: {raw_ce:.3f} -> {cal_ce:.3f}.")
    if np.isfinite(raw_z) and np.isfinite(cal_z):
        lines.append(f"- z-score std moves from {raw_z:.3f} to {cal_z:.3f}; calibrated Gaussian UQ should be close to 1.")
    lines.append("- Scatter shapes can remain similar because calibration often rescales sigma without changing prediction errors.")
    return lines


def write_report(report_path, rows, text_blocks):
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("UQ Calibration Analysis Report\n")
        fh.write("================================\n\n")
        for block in text_blocks:
            for line in block:
                fh.write(line + "\n")
            fh.write("\n")
        fh.write("Metric Table\n")
        fh.write("------------\n")
        for row in rows:
            fh.write(
                f"{row['target']} {row['method']}: "
                f"RMSE={format_metric(row.get('rmse'))}, "
                f"RMV={format_metric(row.get('rmv_sharpness'))}, "
                f"CE={format_metric(row.get('coverage_ce'))}, "
                f"MCE={format_metric(row.get('coverage_mce'))}, "
                f"cov95={format_metric(row.get('coverage_95'))}, "
                f"z_std={format_metric(row.get('z_std'))}, "
                f"NLL={format_metric(row.get('nll'))}\n"
            )
        fh.write("\nGaussian reference exceedance rates: |z|>1 = 0.317, |z|>1.96 = 0.050, |z|>3 = 0.0027.\n")
    print(f"[INFO] Saved {report_path}")


def write_summary_csv(csv_path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved {csv_path}")


def add_trusted_to_row(row, stats):
    if stats is None:
        return row
    for key, value in stats.items():
        row[f"trusted_{key}"] = value
    return row


def generate_paper_style_analysis(npz_path, calibration, set_name, set_uq, force_threshold=None, energy_threshold=None):
    data = np.load(npz_path, allow_pickle=True)
    plot_dir = Path(npz_path).resolve().parent
    base = f"{set_name.lower()}_{set_uq.lower()}_{calibration}_calibration"
    rows = []
    text_blocks = []

    targets = []
    if "delta_comp" in data.files and "sigma_comp_uncal" in data.files:
        targets.append(("forces", "force", data["delta_comp"], data["sigma_comp_uncal"], pick_calibrated(data, "sigma_comp", calibration), force_threshold))
    if all(k in data.files for k in ("delta_energy", "sigma_energy_uncal")) and data["delta_energy"].size:
        targets.append(("energy", "energy", data["delta_energy"], data["sigma_energy_uncal"], pick_calibrated(data, "sigma_energy", calibration), energy_threshold))

    for target_label, target_key, delta, sigma_raw, sigma_cal, threshold in targets:
        if sigma_cal is None:
            print(f"[Warning] Missing calibrated sigma for {target_label} ({calibration}); skipping paper-style analysis.")
            continue
        p, cov_raw = coverage_from_npz_or_compute(data, target_key, "raw", delta, sigma_raw)
        _, cov_cal = coverage_from_npz_or_compute(data, target_key, calibration, delta, sigma_cal)

        plot_average_miscalibration(
            p,
            cov_raw,
            cov_cal,
            f"{set_name} ({set_uq}, {calibration}) - average miscalibration {target_label}",
            plot_dir / f"{base}_{target_label}_average_miscalibration.png",
        )
        plot_coverage_gap(
            p,
            cov_raw,
            cov_cal,
            f"{set_name} ({set_uq}, {calibration}) - coverage gap {target_label}",
            plot_dir / f"{base}_{target_label}_coverage_gap.png",
        )
        plot_variance_calibration(
            delta,
            sigma_raw,
            sigma_cal,
            f"{set_name} ({set_uq}, {calibration}) - variance calibration {target_label}",
            plot_dir / f"{base}_{target_label}_squared_error_vs_squared_uncertainty.png",
        )

        raw_metrics = regression_metrics(delta, sigma_raw, p, cov_raw)
        cal_metrics = regression_metrics(delta, sigma_cal, p, cov_cal)
        raw_trusted = trusted_stats(delta, sigma_raw, threshold)
        cal_trusted = trusted_stats(delta, sigma_cal, threshold)
        if raw_trusted is not None and cal_trusted is not None:
            plot_trusted_summary(
                raw_trusted,
                cal_trusted,
                f"{set_name} ({set_uq}, {calibration}) - trusted threshold {target_label}",
                plot_dir / f"{base}_{target_label}_trusted_threshold.png",
            )

        raw_row = {"target": target_label, "method": "raw", "calibration_mode": calibration, **raw_metrics}
        cal_row = {"target": target_label, "method": calibration, "calibration_mode": calibration, **cal_metrics}
        rows.append(add_trusted_to_row(raw_row, raw_trusted))
        rows.append(add_trusted_to_row(cal_row, cal_trusted))
        text_blocks.append(interpretation_block(target_label, raw_metrics, cal_metrics, calibration))

        raw_delta, raw_sigma = finite_regression_arrays(delta, sigma_raw)
        cal_delta, cal_sigma = finite_regression_arrays(delta, sigma_cal)
        if raw_sigma.size and cal_sigma.size:
            ratio = cal_sigma[: min(cal_sigma.size, raw_sigma.size)] / raw_sigma[: min(cal_sigma.size, raw_sigma.size)]
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            if ratio.size:
                text_blocks.append([
                    f"{target_label.upper()} CALIBRATION SCALE",
                    f"- median calibrated/raw sigma ratio: {np.median(ratio):.6g}",
                    f"- 5-95% calibrated/raw sigma ratio: {np.percentile(ratio, 5):.6g} to {np.percentile(ratio, 95):.6g}",
                ])

    report_path = plot_dir / f"{base}_report.txt"
    csv_path = plot_dir / f"{base}_summary.csv"
    write_report(report_path, rows, text_blocks)
    write_summary_csv(csv_path, rows)


def print_npz_summary(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    print(f"[INFO] Loaded {npz_path}")
    print(f"[INFO] Keys: {', '.join(data.files)}")
    if "scalar_metrics" in data.files:
        try:
            metrics = data["scalar_metrics"].item()
            print("[INFO] Scalar metrics:")
            for key in sorted(metrics):
                value = metrics[key]
                if isinstance(value, (int, float, np.floating)):
                    print(f"  {key}: {float(value):.6g}")
                else:
                    print(f"  {key}: {value}")
        except Exception as exc:
            print(f"[Warning] Could not read scalar_metrics: {exc}")


def prepare_npz_path(input_path, output_dir):
    input_path = Path(input_path).expanduser().resolve()
    if output_dir is None:
        return input_path

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / input_path.name
    if target != input_path:
        shutil.copy2(input_path, target)
        print(f"[INFO] Copied NPZ to {target} so generated plots are written there.")
    return target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate UQ uncertainty diagnostic plots from uq_plot_data_*.npz."
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        default="uq_plots/uq_plot_data_eval_ensemble.npz",
        help="Path to the saved UQ NPZ archive.",
    )
    parser.add_argument("--set-name", default=None, help="Plot label for the split, e.g. Eval or Train.")
    parser.add_argument("--uq", default=None, help="Plot label for the UQ method, e.g. ensemble.")
    parser.add_argument("--ensemble-size", type=int, default=None, help="Optional ensemble size label.")
    parser.add_argument(
        "--calibration",
        choices=("var", "iso", "legacy", "all"),
        default="var",
        help="Calibration mode to plot. Use 'all' to generate every available mode.",
    )
    parser.add_argument(
        "--norm-energy",
        action="store_true",
        help="Normalize energy error/uncertainty per atom when n_atoms_per_frame is present.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for generated plots. The NPZ is copied there before plotting.",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Also generate overconfidence and error-quantile diagnostic plots.",
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        default=None,
        help="Optional force-error threshold for paper-style trusted-point analysis; uses 2*sigma < threshold.",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=None,
        help="Optional energy-error threshold for paper-style trusted-point analysis; uses 2*sigma < threshold.",
    )
    parser.add_argument(
        "--skip-paper-analysis",
        action="store_true",
        help="Only regenerate the standard plots; skip report, CSV, coverage-gap, and variance-calibration outputs.",
    )
    parser.add_argument("--no-summary", action="store_true", help="Do not print NPZ keys and scalar metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    npz_path = prepare_npz_path(args.npz_path, args.output_dir)
    if not npz_path.exists():
        raise FileNotFoundError(f"UQ NPZ file not found: {npz_path}")

    inferred_set_name, inferred_uq, inferred_ensemble_size = infer_metadata(npz_path)
    set_name = args.set_name or inferred_set_name
    set_uq = args.uq or inferred_uq
    ensemble_size = args.ensemble_size if args.ensemble_size is not None else inferred_ensemble_size

    if not args.no_summary:
        print_npz_summary(npz_path)

    with np.load(npz_path, allow_pickle=True) as data:
        modes = available_calibrations(data) if args.calibration == "all" else [args.calibration]

    for mode in modes:
        print(f"[INFO] Generating standard UQ plots with calibration='{mode}'")
        generate_uq_plots(
            str(npz_path),
            set_name,
            set_uq,
            ensemble_size=ensemble_size,
            norm_energy=args.norm_energy,
            calibration=mode,
        )
        if args.extra:
            generate_extra_plots(npz_path, mode, set_name, set_uq)
        if not args.skip_paper_analysis:
            generate_paper_style_analysis(
                npz_path,
                mode,
                set_name,
                set_uq,
                force_threshold=args.force_threshold,
                energy_threshold=args.energy_threshold,
            )


if __name__ == "__main__":
    main()
