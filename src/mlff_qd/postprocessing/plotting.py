# plotting.py  –  April 2025 refactor
"""Plotting utilities for UQ and MLFF evaluation.

Additions in this version
=========================
* Scalar‑metric bar charts (CRPS, ENCE, RLL, Sharpness, CV)
* Coverage curves (nominal vs empirical) for forces **and** energy
* σ‑density histograms (uncal & calibrated)
* Enhanced |Δ| vs σ and Δ² vs σ² plots:
  ─ empirical quantile lines (parameterised)                           [1, 7]
  ─ automatic switch to hex‑bin for very large N                       [2]
  ─ distinct colours for calibrated vs non‑calibrated panels           [3]
  ─ option to colour by residual sign                                  [4]
  ─ common axis limits across paired panels                            [8]
  ─ optional per‑atom normalisation for energy                         [9]
  ─ theoretical / empirical envelope data saved to NPZ                 [10]
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import halfnorm, norm, spearmanr

# -------------------------------------------------------------------
#  Generic helpers (new)
# -------------------------------------------------------------------

def _ideal_colour(is_calibrated: bool) -> str:
    """Return plot colour depending on calibration state."""
    return "crimson" if is_calibrated else "royalblue"


# -------------------------------------------------------------------
#  Scalar‑metric bar chart
# -------------------------------------------------------------------

def plot_scalar_metrics(metrics_dict: dict, title: str, filename: str):
    pairs = [
        ("CRPS", "CRPS"), ("ENCE", "ENCE_cal"),
        ("RLL", "RLL_cal"), ("Sharpness", "Sharpness"), ("CV", "CV"),
    ]
    labels, uncal, cal = [], [], []
    for k_unc, k_cal in pairs:
        if k_unc in metrics_dict and k_cal in metrics_dict:
            labels.append(k_unc)
            uncal.append(metrics_dict[k_unc])
            cal.append(metrics_dict[k_cal])
    if not labels:
        print("No scalar metrics – skip bar chart.")
        return
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, uncal, width, label="uncal")
    plt.bar(x + width/2, cal,   width, label="cal")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.yscale("log")
    plt.ylabel("value  (↓ better for CRPS/ENCE/RLL)")
    plt.title(title)
    for i, v in enumerate(uncal):
        plt.text(x[i]-width/2, v, f"{v:.3g}", va="bottom", ha="center", fontsize=7, rotation=90)
    for i, v in enumerate(cal):
        plt.text(x[i]+width/2, v, f"{v:.3g}", va="bottom", ha="center", fontsize=7, rotation=90)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Generated scalar bar ➜ {filename}")


# -------------------------------------------------------------------
#  Coverage curve
# -------------------------------------------------------------------

def plot_coverage_curve(p_nom, cov_uncal, cov_cal, title, filename):
    if p_nom is None or cov_uncal is None or len(p_nom) == 0:
        print("No coverage data – skip plot.")
        return
    plt.figure(figsize=(5, 5))
    plt.plot(p_nom, p_nom, "k--", lw=1, label="ideal")
    plt.plot(p_nom, cov_uncal, "-o", ms=3, label="uncal")
    if cov_cal is not None and len(cov_cal):
        plt.plot(p_nom, cov_cal, "-s", ms=3, label="cal")
    plt.xlabel("nominal interval prob.")
    plt.ylabel("empirical coverage")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Generated coverage ➜ {filename}")


# -------------------------------------------------------------------
#  σ density histogram
# -------------------------------------------------------------------

def plot_sigma_density(sig_uncal, sig_cal, title, filename, bins=60):
    if sig_uncal is None or len(sig_uncal) == 0:
        print("No σ data – skip density.")
        return
    plt.figure(figsize=(6, 4))
    plt.hist(sig_uncal, bins=bins, density=True, alpha=0.5, label="uncal")
    if sig_cal is not None and len(sig_cal):
        plt.hist(sig_cal, bins=bins, density=True, alpha=0.5, label="cal")
    plt.yscale("log")
    plt.xlabel("σ value")
    plt.ylabel("density (log)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Generated σ density ➜ {filename}")


# =============================================================================
#  Updated swapped‑axis |Δ| vs σ
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import halfnorm
import matplotlib.colors as mcolors

def plot_swapped_final_tight(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    scale: str = "linear",
    title: str = "",
    xlabel: str = "Predicted Uncertainty (σ)",
    ylabel: str = "|Δ|",
    q_low: float = 0.005,
    q_high: float = 0.995,
    hexbin_threshold: int = 20000,
    colour: str = "royalblue",
):
    """
    Error vs Uncertainty with:
      - identity line
      - theoretical envelopes (half-normal)
      - empirical quantiles
      - hexbin on linear scale, scatter on log-log
    """
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    x_plot, y_plot = x[mask], y[mask]

    # line grid for envelopes
    if scale == "linear":
        x_min, x_max = x_plot.min(), x_plot.max()
        x_line = np.linspace(x_min, x_max, 300)
        xlim_min = 0
    else:
        x_min_log, x_max_log = np.log10(x_plot.min()), np.log10(x_plot.max())
        x_line = np.logspace(x_min_log, x_max_log, 300)
        xlim_min = 10**x_min_log * 0.9

    # theoretical envelopes
    c_low, c_high = halfnorm.ppf(q_low), halfnorm.ppf(q_high)
    lower_theo = c_low  * x_line
    upper_theo = c_high * x_line

    # identity + theory
    ax.plot(x_line, x_line,      "k--", lw=1, label="Error = Unc")
    ax.plot(x_line, lower_theo,  color="grey", lw=0.8, label=f"{q_low*100:.1f}% Theo")
    ax.plot(x_line, upper_theo,  color="grey", lw=0.8, label=f"{q_high*100:.1f}% Theo")

    # plot data: HEX on linear, SCATTER on log
    if scale == "linear" and len(x_plot) > hexbin_threshold:
        hb = ax.hexbin(
            x_plot, y_plot,
            gridsize=80,
            cmap="plasma",
            bins="log",
            norm=mcolors.LogNorm(vmin=1, vmax=None),
            mincnt=1,
            edgecolors="none",
        )
        plt.colorbar(hb, ax=ax, pad=0.02, label="log(count)")
    else:
        ax.scatter(x_plot, y_plot, c=colour, alpha=0.6, s=25, edgecolors="none")

    # empirical quantiles
    ratio = y_plot / x_plot
    emp_low, emp_high = np.quantile(ratio, [q_low, q_high])
    ax.plot(
        x_line, emp_low * x_line,
        "--", color="tab:green", lw=1.2, label=f"Emp {q_low*100:.1f}%"
    )
    ax.plot(
        x_line, emp_high * x_line,
        "--", color="tab:green", lw=1.2, label=f"Emp {q_high*100:.1f}%"
    )

    # finalize
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if scale == "log":
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(left=xlim_min, right=x_line.max()*1.05)
    y_min = min(y_plot.min(), lower_theo.min()) * (0.9 if scale=="log" else 1.0)
    y_max = max(y_plot.max(), upper_theo.max()) * 1.05
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.3)

    # compact legend
    handles, labels = ax.get_legend_handles_labels()
    keep = ["Error = Unc",
            f"{q_low*100:.1f}% Theo", f"{q_high*100:.1f}% Theo",
            f"Emp {q_low*100:.1f}%", f"Emp {q_high*100:.1f}%"]
    sel = [(h,l) for h,l in zip(handles,labels) if l in keep]
    if sel:
        hs, ls = zip(*sel)
        ax.legend(hs, ls, fontsize=8, loc="upper left")


def plot_original_final_tight(
    ax,
    x_sq: np.ndarray,
    y_sq: np.ndarray,
    *,
    scale: str = "linear",
    title: str = "",
    xlabel: str = "Predicted Uncertainty² (σ²)",
    ylabel: str = "Squared Error (Δ²)",
    q_low: float = 0.005,
    q_high: float = 0.995,
    hexbin_threshold: int = 20000,
    colour: str = "royalblue",
    colour_by_sign: bool = False,
    raw_delta: np.ndarray = None
):
    """
    Δ² vs σ² with:
      - identity line
      - theory & empirical envelopes
      - hexbin on linear, scatter on log
      - optional sign‐colouring (if raw_delta provided)
    """
    mask = (x_sq > 0) & (y_sq > 0) & np.isfinite(x_sq) & np.isfinite(y_sq)
    if not np.any(mask):
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    x_plot, y_plot = x_sq[mask], y_sq[mask]

    if scale == "linear":
        x_min, x_max = x_plot.min(), x_plot.max()
        x_line = np.linspace(x_min, x_max, 300)
        xlim_min = 0
    else:
        x_min_log, x_max_log = np.log10(x_plot.min()), np.log10(x_plot.max())
        x_line = np.logspace(x_min_log, x_max_log, 300)
        xlim_min = 10**x_min_log * 0.9

    # theory on z²
    c_low, c_high = halfnorm.ppf(q_low), halfnorm.ppf(q_high)
    lower_theo = (c_low**2) * x_line
    upper_theo = (c_high**2) * x_line

    ax.plot(x_line, x_line,        "k--", lw=1, label="Error² = Unc²")
    ax.plot(x_line, lower_theo,    color="grey", lw=0.8, label=f"{q_low*100:.1f}% Theo²")
    ax.plot(x_line, upper_theo,    color="grey", lw=0.8, label=f"{q_high*100:.1f}% Theo²")

    # data display
    if colour_by_sign and raw_delta is not None:
        sign = np.sign(raw_delta[mask])
        cols = np.where(sign>=0, "tab:green", "tab:red")
        ax.scatter(x_plot, y_plot, c=cols, alpha=0.4, s=25, edgecolors="none")
    elif scale == "linear" and len(x_plot) > hexbin_threshold:
        hb = ax.hexbin(
            x_plot, y_plot,
            gridsize=80,
            cmap="plasma",
            bins="log",
            norm=mcolors.LogNorm(vmin=1, vmax=None),
            mincnt=1,
            edgecolors="none",
        )
        plt.colorbar(hb, ax=ax, pad=0.02, label="log(count)")
    else:
        ax.scatter(x_plot, y_plot, c=colour, alpha=0.6, s=25, edgecolors="none")

    # empirical on z²
    ratio = y_plot / x_plot
    emp_low, emp_high = np.quantile(ratio, [q_low, q_high])
    ax.plot(
        x_line, emp_low * x_line,
        "--", color="tab:green", lw=1.2, label=f"Emp {q_low*100:.1f}%²"
    )
    ax.plot(
        x_line, emp_high * x_line,
        "--", color="tab:green", lw=1.2, label=f"Emp {q_high*100:.1f}%²"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if scale == "log":
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(left=xlim_min, right=x_line.max()*1.05)
    y_min = min(y_plot.min(), lower_theo.min()) * (0.9 if scale=="log" else 1.0)
    y_max = max(y_plot.max(), upper_theo.max()) * 1.05
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    keep = [
        "Error² = Unc²",
        f"{q_low*100:.1f}% Theo²", f"{q_high*100:.1f}% Theo²",
        f"Emp {q_low*100:.1f}%²", f"Emp {q_high*100:.1f}%²"
    ]
    sel = [(h,l) for h,l in zip(handles,labels) if l in keep]
    if sel:
        hs, ls = zip(*sel)
        ax.legend(hs, ls, fontsize=8, loc="upper left")

# =============================================================================
# Updated generate_uq_plots   (re‑implemented with new calls)
# =============================================================================

import os
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# assume these come from your module
# from ..plot_helpers import (
#     plot_swapped_final_tight,
#     plot_original_final_tight,
#     plot_scalar_metrics,
#     plot_coverage_curve,
#     plot_sigma_density,
#     _ideal_colour
# )


def generate_uq_plots(npz_plot_data_path, set_name, set_uq, ensemble_size=None,
                      norm_energy=False):
    print(f"\n--- Generating UQ Plots for: {set_name} Set ({set_uq}) from {npz_plot_data_path} ---")
    if not os.path.exists(npz_plot_data_path):
        print(f"Error: Plot data file not found: {npz_plot_data_path}. Skipping plots.")
        return

    # ------------------------------------------------------------------
    try:
        data = np.load(npz_plot_data_path, allow_pickle=True)

        # component‐level
        sigma_c_uncal = data["sigma_comp_uncal"]
        sigma_c_cal   = data["sigma_comp_cal"]
        delta_c       = data["delta_comp"]
        err_c_abs     = np.abs(delta_c)

        # frame‐level energy
        sigma_E_uncal = data.get("sigma_energy_uncal")
        sigma_E_cal   = data.get("sigma_energy_cal")
        delta_E       = data.get("delta_energy")
        err_E_abs     = np.abs(delta_E) if delta_E is not None else None

        # precomputed scalars & coverage
        scalar_metrics = data.get("scalar_metrics", {}).item()
        p_nominal      = data.get("p_thresholds")
        cov_uncal      = data.get("coverage_uncal")
        cov_cal        = data.get("coverage_cal")
        cov_uncal_e    = data.get("coverage_uncal_e")
        cov_cal_e      = data.get("coverage_cal_e")
        n_atoms_frame  = data.get("n_atoms_per_frame")
    except Exception as e:
        print(f"Error loading NPZ: {e}")
        traceback.print_exc()
        return

    plot_dir = os.path.dirname(npz_plot_data_path)
    os.makedirs(plot_dir, exist_ok=True)
    ens_str     = f"_ens{ensemble_size}" if ensemble_size else ""
    base_part   = f"{set_name.lower()}_{set_uq.lower()}{ens_str}"
    title_base  = f"{set_name} ({set_uq}{ens_str})"

    # 1. scalar metrics bar‐chart ----------------------------------------
    try:
        plot_scalar_metrics(
            scalar_metrics,
            f"{title_base} – scalar metrics",
            os.path.join(plot_dir, f"{base_part}_scalar_metrics.png")
        )
    except Exception as e:
        print(f"Error plotting scalar metrics: {e}")

    # 2. coverage curves -------------------------------------------------
    try:
        if p_nominal is not None and cov_uncal is not None:
            plot_coverage_curve(
                p_nominal, cov_uncal, cov_cal,
                f"{title_base} – coverage (forces)",
                os.path.join(plot_dir, f"{base_part}_coverage_forces.png")
            )
        if p_nominal is not None and cov_uncal_e is not None and len(cov_uncal_e):
            plot_coverage_curve(
                p_nominal, cov_uncal_e, cov_cal_e,
                f"{title_base} – coverage (energy)",
                os.path.join(plot_dir, f"{base_part}_coverage_energy.png")
            )
    except Exception as e:
        print(f"Error plotting coverage curves: {e}")

    # 3. sigma density histograms ---------------------------------------
    try:
        plot_sigma_density(
            sigma_c_uncal, sigma_c_cal,
            f"{title_base} – σ distribution (forces)",
            os.path.join(plot_dir, f"{base_part}_sigma_density_forces.png")
        )
        if sigma_E_uncal is not None and len(sigma_E_uncal):
            plot_sigma_density(
                sigma_E_uncal, sigma_E_cal,
                f"{title_base} – σ distribution (energy)",
                os.path.join(plot_dir, f"{base_part}_sigma_density_energy.png")
            )
    except Exception as e:
        print(f"Error plotting σ density: {e}")

    # 4. optional energy normalisation per atom -------------------------
    if norm_energy and n_atoms_frame is not None and err_E_abs is not None:
        sigma_E_uncal = sigma_E_uncal / n_atoms_frame
        sigma_E_cal   = sigma_E_cal   / n_atoms_frame
        err_E_abs     = err_E_abs     / n_atoms_frame

    # 5. error vs uncertainty plots -------------------------------------
    try:
        # --- Overlay both clouds on one axes (forces) ---
        if len(sigma_c_uncal) and len(err_c_abs):
            fig_ov, ax_ov = plt.subplots(1,1,figsize=(6,5))
            ax_ov.set_xscale("log"); ax_ov.set_yscale("log")
            ax_ov.scatter(
                sigma_c_uncal, err_c_abs,
                s=10, alpha=0.2, c="royalblue", label="Uncalibrated"
            )
            ax_ov.scatter(
                sigma_c_cal, err_c_abs,
                s=10, alpha=0.2, c="crimson", label="Calibrated"
            )
            x_line = np.logspace(
                np.log10(min(sigma_c_uncal.min(), sigma_c_cal.min())),
                np.log10(max(sigma_c_uncal.max(), sigma_c_cal.max())),
                200
            )
            ax_ov.plot(x_line, x_line, "k--", lw=1, label="Error = Unc")
            ax_ov.set_title(f"{title_base} – overlay")
            ax_ov.set_xlabel("Predicted Uncertainty (σ)")
            ax_ov.set_ylabel("|Δ|")
            ax_ov.legend(fontsize=8, loc="upper left")
            fig_ov.tight_layout()
            fig_ov.savefig(os.path.join(plot_dir, f"force_overlay_{base_part}.png"))
            plt.close(fig_ov)
        else:
            print("Skipping overlay (forces) – no data.")

        # Prepare shared limits for two‑panel views
        if len(sigma_c_uncal):
            # valid indices
            mask_unc = (sigma_c_uncal>0)&(err_c_abs>0)
            mask_cal = (sigma_c_cal  >0)&(err_c_abs>0)
            x_u, y_u = sigma_c_uncal[mask_unc], err_c_abs[mask_unc]
            x_c, y_c = sigma_c_cal[mask_cal],   err_c_abs[mask_cal]
            all_x = np.hstack([x_u, x_c])
            all_y = np.hstack([y_u, y_c])
            x_min, x_max = all_x.min()*0.95, all_x.max()*1.05
            y_min, y_max = all_y.min()*0.95, all_y.max()*1.05

            # a) |Δ| vs σ — Lin & Log
            for scale in ("linear", "log"):
                fig, axs = plt.subplots(1, 2, figsize=(13, 5))
                plot_swapped_final_tight(
                    axs[0], sigma_c_uncal, err_c_abs,
                    scale=scale, title="Non‑calib",
                    colour=_ideal_colour(False),
                    q_low=0.005, q_high=0.995
                )
                plot_swapped_final_tight(
                    axs[1], sigma_c_cal,   err_c_abs,
                    scale=scale, title="Calib",
                    colour=_ideal_colour(True),
                    q_low=0.005, q_high=0.995
                )
                for ax in axs:
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                fig.suptitle(f"{title_base} – |Δ| vs σ ({scale})")
                fig.tight_layout(rect=[0, 0, 1, 0.93])
                fig.savefig(f"{Path(plot_dir)/('force_' + base_part)}_ErrUnc_{scale.title()}.png")
                plt.close(fig)

            # b) Δ² vs σ² — Lin & Log
            for scale in ("linear", "log"):
                fig, axs = plt.subplots(1, 2, figsize=(13, 5))
                x_sq_all = np.hstack([sigma_c_uncal**2, sigma_c_cal**2])
                y_sq_all = err_c_abs**2
                x2_min, x2_max = x_sq_all.min()*0.95, x_sq_all.max()*1.05
                y2_min, y2_max = y_sq_all.min()*0.95, y_sq_all.max()*1.05

                plot_original_final_tight(
                    axs[0], sigma_c_uncal**2, err_c_abs**2,
                    scale=scale, title="Non‑calib",
                    colour=_ideal_colour(False)
                )
                plot_original_final_tight(
                    axs[1], sigma_c_cal**2,   err_c_abs**2,
                    scale=scale, title="Calib",
                    colour=_ideal_colour(True)
                )
                for ax in axs:
                    ax.set_xlim(x2_min, x2_max)
                    ax.set_ylim(y2_min, y2_max)
                fig.suptitle(f"{title_base} – Δ² vs σ² ({scale})")
                fig.tight_layout(rect=[0, 0, 1, 0.93])
                fig.savefig(f"{Path(plot_dir)/('force_' + base_part)}_ErrSqUncSq_{scale.title()}.png")
                plt.close(fig)

            print(f"Generated force UQ plots for {set_name} ({set_uq})")
        else:
            print("Skipping force UQ plots – no data.")
    except Exception as plot_err:
        print(f"Error during force UQ plotting: {plot_err}")
        traceback.print_exc()
        plt.close("all")

    # 6. energy Err vs σ + Δ² vs σ² if available -----------------------
    try:
        if sigma_E_uncal is not None and len(sigma_E_uncal):
            # shared limits for energy
            mask_eu = (sigma_E_uncal>0)&(err_E_abs>0)
            mask_ec = (sigma_E_cal  >0)&(err_E_abs>0)
            x_eu, y_eu = sigma_E_uncal[mask_eu], err_E_abs[mask_eu]
            x_ec, y_ec = sigma_E_cal[mask_ec],   err_E_abs[mask_ec]
            all_xe = np.hstack([x_eu, x_ec])
            all_ye = np.hstack([y_eu, y_ec])
            xe_min, xe_max = all_xe.min()*0.95, all_xe.max()*1.05
            ye_min, ye_max = all_ye.min()*0.95, all_ye.max()*1.05

            # |ΔE| vs σ_E
            for scale in ("linear", "log"):
                fig, axs = plt.subplots(1, 2, figsize=(13, 5))
                plot_swapped_final_tight(
                    axs[0], sigma_E_uncal, err_E_abs,
                    scale=scale, title="Non‑calib",
                    colour=_ideal_colour(False)
                )
                plot_swapped_final_tight(
                    axs[1], sigma_E_cal,   err_E_abs,
                    scale=scale, title="Calib",
                    colour=_ideal_colour(True)
                )
                for ax in axs:
                    ax.set_xlim(xe_min, xe_max)
                    ax.set_ylim(ye_min, ye_max)
                fig.suptitle(f"{title_base} – |ΔE| vs σ_E ({scale})")
                fig.tight_layout(rect=[0, 0, 1, 0.93])
                fig.savefig(f"{Path(plot_dir)/('energy_' + base_part)}_ErrUnc_{scale.title()}.png")
                plt.close(fig)

            # ΔE² vs σ_E²
            for scale in ("linear", "log"):
                fig, axs = plt.subplots(1, 2, figsize=(13, 5))
                x2e = np.hstack([sigma_E_uncal**2, sigma_E_cal**2])
                y2e = err_E_abs**2
                x2emin, x2emax = x2e.min()*0.95, x2e.max()*1.05
                y2emin, y2emax = y2e.min()*0.95, y2e.max()*1.05

                plot_original_final_tight(
                    axs[0], sigma_E_uncal**2, err_E_abs**2,
                    scale=scale, title="Non‑calib",
                    colour=_ideal_colour(False)
                )
                plot_original_final_tight(
                    axs[1], sigma_E_cal**2,   err_E_abs**2,
                    scale=scale, title="Calib",
                    colour=_ideal_colour(True)
                )
                for ax in axs:
                    ax.set_xlim(x2emin, x2emax)
                    ax.set_ylim(y2emin, y2emax)
                fig.suptitle(f"{title_base} – ΔE² vs σ_E² ({scale})")
                fig.tight_layout(rect=[0, 0, 1, 0.93])
                fig.savefig(f"{Path(plot_dir)/('energy_' + base_part)}_ErrSqUncSq_{scale.title()}.png")
                plt.close(fig)

            print(f"Generated energy UQ plots for {set_name} ({set_uq})")
        else:
            print("Skipping energy UQ plots – no data.")
    except Exception as plot_err:
        print(f"Error during energy UQ plotting: {plot_err}")
        traceback.print_exc()
        plt.close("all")


# === Plotting Helpers Specific to Traditional Active Learning ===

def plot_histogram(scores, title, xlabel, filename, bins=50, vline_values=None, vline_labels=None, vline_colors=None):
    """
    Plots and saves a histogram with optional vertical reference lines.

    Parameters:
        scores (np.ndarray): Data to plot.
        title (str): Plot title.
        xlabel (str): X-axis label.
        filename (str): Output filename.
        bins (int): Number of bins (default 50).
        vline_values (list): X-values for vertical lines.
        vline_labels (list): Corresponding labels.
        vline_colors (list): Colors for each vertical line.

    Returns:
        None
    """
    plt.figure(figsize=(8, 5))
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) == 0:
        print(f"Warning: No valid scores for histogram {filename}")
        plt.close()
        return
    plt.hist(valid_scores, bins=bins, alpha=0.7, color='teal', label='_nolegend_')
    all_handles, all_labels = [], []
    if vline_values and vline_labels and vline_colors and len(vline_values) == len(vline_labels) == len(vline_colors):
        for val, label, color in zip(vline_values, vline_labels, vline_colors):
            if np.isfinite(val):
                line = plt.axvline(val, color=color, linestyle='--', label=f'{label} ({val:.3f})')
                all_handles.append(line)
                all_labels.append(f'{label} ({val:.3f})')
            else:
                print(f"Warning: Skipping NaN/inf vline for {label} in {filename}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    if all_handles:
        plt.legend(handles=all_handles, labels=all_labels, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Generated plot: {filename}")
    plt.close()


def plot_overall_score_vs_clusters(overall_score, hdbscan_cluster_labels, unique_eval_frames, selected_indices):
    """
    Plots overall score versus evaluation frame index, colored by HDBSCAN cluster.

    Parameters:
        overall_score (np.ndarray): Array of overall scores for evaluation frames.
        hdbscan_cluster_labels (np.ndarray): Cluster labels from HDBSCAN.
        unique_eval_frames (np.ndarray): Array of evaluation frame indices.
        selected_indices (list): List of frame indices selected by an AL procedure.

    Returns:
        None
    """
    print("Generating overall score vs. cluster plot...")
    plt.figure(figsize=(12, 7))
    x_values = np.arange(len(unique_eval_frames))
    unique_clusters = np.unique(hdbscan_cluster_labels)
    cmap = plt.cm.viridis
    if len(overall_score) != len(hdbscan_cluster_labels) or len(hdbscan_cluster_labels) != len(unique_eval_frames):
        print("Error plotting clusters: Length mismatch!")
        plt.close()
        return

    noise_color = 'grey'
    cluster_ids = sorted([c for c in unique_clusters if c >= 0])
    num_real_clusters = len(cluster_ids)
    cluster_colors = {cid: cmap(i / max(1, num_real_clusters - 1)) for i, cid in enumerate(cluster_ids)}
    cluster_colors[-1] = noise_color

    plotted_labels = set()
    for cluster_id in unique_clusters:
        mask = (hdbscan_cluster_labels == cluster_id)
        label_text = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
        color = cluster_colors[cluster_id]
        current_label = label_text if label_text not in plotted_labels else None
        plt.scatter(x_values[mask], overall_score[mask], color=color, alpha=0.6, label=current_label, s=20)
        if current_label:
            plotted_labels.add(current_label)

    frame_idx_to_pos = {f_idx: pos for pos, f_idx in enumerate(unique_eval_frames)}
    selected_pos_indices = [frame_idx_to_pos[idx] for idx in selected_indices if idx in frame_idx_to_pos]
    selected_mask = np.zeros(len(unique_eval_frames), dtype=bool)
    if selected_pos_indices:
        selected_mask[selected_pos_indices] = True
    if np.any(selected_mask):
        plt.scatter(x_values[selected_mask], overall_score[selected_mask],
                    c='black', marker='x', s=50, label='Selected', zorder=5)

    plt.xlabel('Evaluation Frame Index (Position)')
    plt.ylabel('Overall Score')
    plt.title('Overall Score vs. HDBSCAN Clustering')
    handles, labels_leg = plt.gca().get_legend_handles_labels()
    max_legend = 15
    if len(handles) > max_legend:
        step = max(1, len(handles) // max_legend)
        handles = handles[::step]
        labels_leg = labels_leg[::step]
    if handles:
        plt.legend(handles, labels_leg, bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('trad_overall_score_vs_clusters.png')
    plt.close()
    print("Generated overall score vs. cluster plot.")


def plot_rmse_vs_sorted_overall_score(overall_score, agg_error, frame_conf_labels, frame_error_labels):
    """
    Plots Aggregated Error (RMSE) versus sorted overall score, colored by confidence and error status.

    Parameters:
        overall_score (np.ndarray): Overall scores per evaluation frame.
        agg_error (np.ndarray): Aggregated error (RMSE) per frame.
        frame_conf_labels (list): Confidence labels per frame.
        frame_error_labels (list): Error labels per frame.

    Returns:
        None
    """
    print("Generating detailed RMSE vs sorted overall score plot...")
    plt.figure(figsize=(12, 7))
    valid_mask = ~np.isnan(overall_score) & ~np.isnan(agg_error)
    if not np.any(valid_mask):
        print("No valid data for RMSE vs Score plot.")
        plt.close()
        return
    overall_score_v = overall_score[valid_mask]
    agg_error_v = agg_error[valid_mask]
    frame_conf_labels_v = np.array(frame_conf_labels)[valid_mask]
    frame_error_labels_v = np.array(frame_error_labels)[valid_mask]
    sort_indices = np.argsort(overall_score_v)
    sorted_scores = overall_score_v[sort_indices]
    sorted_rmse = agg_error_v[sort_indices]
    sorted_conf = frame_conf_labels_v[sort_indices]
    sorted_err = frame_error_labels_v[sort_indices]
    masks = {
        'W/N': (sorted_conf == "within") & (sorted_err == "normal"),
        'W/H': (sorted_conf == "within") & (sorted_err == "high"),
        'Ov/N': (sorted_conf == "over") & (sorted_err == "normal"),
        'Ov/H': (sorted_conf == "over") & (sorted_err == "high"),
        'Un/N': (sorted_conf == "under") & (sorted_err == "normal"),
        'Un/H': (sorted_conf == "under") & (sorted_err == "high")
    }
    colors = {
        'W/N': 'grey',
        'W/H': 'orange',
        'Ov/N': 'deepskyblue',
        'Ov/H': 'mediumblue',
        'Un/N': 'lightcoral',
        'Un/H': 'firebrick'
    }
    alphas = {
        'W/N': 0.4,
        'W/H': 0.6,
        'Ov/N': 0.5,
        'Ov/H': 0.7,
        'Un/N': 0.5,
        'Un/H': 0.7
    }
    sizes = {
        'W/N': 15,
        'W/H': 25,
        'Ov/N': 20,
        'Ov/H': 30,
        'Un/N': 20,
        'Un/H': 30
    }
    full_labels = {
        'W/N': 'Within CI / Normal Error',
        'W/H': 'Within CI / High Error',
        'Ov/N': 'Overconfident / Normal Error',
        'Ov/H': 'Overconfident / High Error',
        'Un/N': 'Underconfident / Normal Error',
        'Un/H': 'Underconfident / High Error'
    }
    for key, mask in masks.items():
        if np.any(mask):
            plt.scatter(sorted_scores[mask], sorted_rmse[mask], c=colors[key],
                        alpha=alphas[key], s=sizes[key], label=full_labels[key])
    plt.xlabel('Sorted Overall Score')
    plt.ylabel('Aggregated Error (RMSE per Frame)')
    plt.title('RMSE vs. Sorted Overall Score (Colored by Confidence & Error Status)')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('trad_rmse_vs_sorted_overall_score.png')
    plt.close()
    print("Generated detailed RMSE vs sorted overall score plot.")


def plot_error_vs_uncertainty(agg_unc, agg_error, selected_indices, lower_threshold, upper_threshold, av_score, unique_eval_frames):
    """
    Plots Aggregated Error vs Aggregated Uncertainty on a log-log scale.

    Parameters:
        agg_unc (np.ndarray): Aggregated uncertainty per frame.
        agg_error (np.ndarray): Aggregated error (RMSE) per frame.
        selected_indices (list): List of selected frame indices.
        lower_threshold (float): Lower threshold for confidence.
        upper_threshold (float): Upper threshold for confidence.
        av_score (float): Average combined score.
        unique_eval_frames (np.ndarray): Array of evaluation frame indices.

    Returns:
        None
    """
    print("Generating error vs uncertainty plot...")
    plt.figure(figsize=(8, 6))
    valid_mask = ~np.isnan(agg_unc) & ~np.isnan(agg_error)
    if not np.any(valid_mask):
        print("No valid data for Error vs Uncertainty plot.")
        plt.close()
        return
    agg_unc_v = agg_unc[valid_mask]
    agg_error_v = agg_error[valid_mask]
    unique_eval_frames_v = unique_eval_frames[valid_mask]
    
    frame_idx_to_pos_v = {f_idx: pos for pos, f_idx in enumerate(unique_eval_frames_v)}
    selected_pos_indices = [frame_idx_to_pos_v[idx] for idx in selected_indices if idx in frame_idx_to_pos_v]
    selected_mask = np.zeros(len(unique_eval_frames_v), dtype=bool)
    if selected_pos_indices:
        selected_mask[selected_pos_indices] = True

    plt.scatter(agg_unc_v, agg_error_v, color='blue', alpha=0.3, label='All Eval Frames', s=15)
    plt.scatter(agg_unc_v[selected_mask], agg_error_v[selected_mask],
                color='orange', alpha=0.8, label='Selected Frames', s=30, edgecolors='k', lw=0.5)
    eps = 1e-9
    unc_min = np.nanmin(agg_unc_v[agg_unc_v > eps]) if np.any(agg_unc_v > eps) else eps
    unc_max = np.nanmax(agg_unc_v) if np.any(agg_unc_v > eps) else eps * 10
    unc_range = np.logspace(np.log10(unc_min), np.log10(unc_max), 200)
    ideal_line = unc_range
    lower_line = unc_range / max(eps, upper_threshold)
    upper_line = unc_range / max(eps, lower_threshold)
    plt.plot(unc_range, ideal_line, 'k--', label='Error = Unc', lw=1)
    plt.plot(unc_range, lower_line, 'r--', label=f'Underconf Thr ({upper_threshold:.2f})', lw=1)
    plt.plot(unc_range, upper_line, 'b--', label=f'Overconf Thr ({lower_threshold:.2f})', lw=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Aggregated Uncertainty (Mean per Frame)')
    plt.ylabel('Aggregated Error (Mean RMSE per Frame)')
    plt.title('Error vs. Uncertainty (Log-Log)')
    y_min = max(eps, np.nanmin(agg_error_v[agg_error_v > eps]) if np.any(agg_error_v > eps) else eps) * 0.5
    y_max = max(eps * 10, np.nanmax(agg_error_v)) * 2.0
    plt.ylim(max(y_min, 1e-7), y_max)
    plt.xlim(unc_min * 0.9, unc_max * 1.1)
    plt.fill_between(unc_range, upper_line, y_max, where=upper_line < y_max,
                     color='blue', alpha=0.1, label='Overconfident Region')
    plt.fill_between(unc_range, y_min, lower_line, where=lower_line > y_min,
                     color='red', alpha=0.1, label='Underconfident Region')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('trad_error_vs_uncertainty_loglog.png')
    plt.close()
    print("Generated error vs uncertainty plot.")


# === New Plotting Functions for Active Learning Data ===

def generate_al_influence_plots(npz_path):
    """
    Generates plots specific to the Influence Active Learning method using saved NPZ data.

    Parameters:
        npz_path (str): Path to the NPZ file containing plot data.

    Returns:
        None
    """
    if not npz_path or not os.path.exists(npz_path):
        print(f"Warning: Influence AL plot data not found at {npz_path}. Skipping plots.")
        return

    print(f"\n--- Generating Plots for Influence AL from {npz_path} ---")
    try:
        data = np.load(npz_path)
        x_sorted_indices = np.argsort(data['X_valid'])
        x_plot = data['X_valid'][x_sorted_indices]
        y_calibrated_plot = data['y_calibrated_plot']
        corr_raw = data['corr_raw'].item()
        corr_calib = data['corr_calib'].item()

        plt.figure(figsize=(8, 6))
        plt.scatter(data['X_valid'], data['y_valid'], alpha=0.5, label='Raw Data', s=10)
        plt.plot(x_plot, y_calibrated_plot, color='red', linewidth=2, label='Isotonic Fit')
        plt.xlabel("Raw Uncertainty (Avg per Frame)")
        plt.ylabel("Actual Error (Avg per Frame)")
        title = f"Isotonic Regression Calibration\nCorr(Raw): {corr_raw:.3f}, Corr(Calib): {corr_calib:.3f}"
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.6)
        plot_filename = npz_path.replace("_plot_data.npz", "_calibration_plot.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved calibration plot: {plot_filename}")
    except Exception as e:
        print(f"Error generating influence AL plots: {e}")
        traceback.print_exc()
        plt.close("all")


def generate_al_traditional_plots(npz_path):
    """
    Generates plots specific to the Traditional Active Learning method using saved NPZ data.
    
    Parameters:
        npz_path (str): Path to the NPZ file with Traditional AL plot data.
    
    Returns:
        None
    """
    if not npz_path or not os.path.exists(npz_path):
        print(f"Warning: Traditional AL plot data not found at {npz_path}. Skipping plots.")
        return

    print(f"\n--- Generating Plots for Traditional AL from {npz_path} ---")
    try:
        data = np.load(npz_path)
        overall_score = data["overall_score"]
        agg_error = data["agg_error"]
        agg_unc = data["agg_unc"]
        combined_score = data["combined_score"]
        error_score = data["error_score"]
        frame_conf_labels = data["frame_conf_labels"]
        frame_error_labels = data["frame_error_labels"]
        hdbscan_cluster_labels = data["hdbscan_cluster_labels"]
        unique_eval_frames = data["unique_eval_frames"]
        selected_indices = data["selected_indices"]
        lower_thr_cs = data["lower_thr_cs"].item()
        upper_thr_cs = data["upper_thr_cs"].item()
        av_cs = data["av_cs"].item()
        z_threshold_high_es = data["z_threshold_high_es"].item()
        mean_es = data["mean_es"].item()
        std_es = data["std_es"].item()
        lower_perc_overall = np.percentile(overall_score[~np.isnan(overall_score)], 5)
        upper_perc_overall = np.percentile(overall_score[~np.isnan(overall_score)], 95)
        threshold_es_value = mean_es + z_threshold_high_es * std_es if std_es > 1e-9 else np.nan

        plot_dir = os.path.dirname(npz_path)
        base_filename = os.path.basename(npz_path).replace("_plot_data.npz", "")

        # Call the plotting helpers for Traditional AL.
        plot_overall_score_vs_clusters(overall_score, hdbscan_cluster_labels, unique_eval_frames, selected_indices)
        plot_rmse_vs_sorted_overall_score(overall_score, agg_error, frame_conf_labels, frame_error_labels)
        plot_error_vs_uncertainty(agg_unc, agg_error, selected_indices, lower_thr_cs, upper_thr_cs, av_cs, unique_eval_frames)
        plot_histogram(combined_score, 'Dist Combined Score (CS)', 'CS',
                       os.path.join(plot_dir, f"{base_filename}_cs_dist.png"),
                       vline_values=[lower_thr_cs, upper_thr_cs],
                       vline_labels=['LowCI', 'HighCI'],
                       vline_colors=['blue', 'red'])
        plot_histogram(error_score, 'Dist Error Score (ES)', 'ES',
                       os.path.join(plot_dir, f"{base_filename}_es_dist.png"),
                       vline_values=[threshold_es_value],
                       vline_labels=['HighErrThr'],
                       vline_colors=['red'])
        plot_histogram(overall_score, 'Dist Overall Score', 'Overall Score',
                       os.path.join(plot_dir, f"{base_filename}_overall_dist.png"),
                       vline_values=[lower_perc_overall, upper_perc_overall],
                       vline_labels=['5th %ile', '95th %ile'],
                       vline_colors=['purple', 'purple'])
        print("Traditional AL plot generation finished.")
    except KeyError as e_key:
        print(f"Error loading data from {npz_path}: Missing key {e_key}. Cannot generate plots.")
    except Exception as e:
        print(f"Error generating traditional AL plots: {e}")
        traceback.print_exc()
        plt.close("all")


def plot_mlff_stats(stats: "MLFFStats", min_distances_all, log_file_base,
                    compare_with_training, train_mask, eval_mask):
    """
    Generates MLFF statistics plots including residual vs. distance and parity plots.
    
    Parameters:
        stats (MLFFStats): MLFFStats object containing prediction errors and statistics.
        min_distances_all (np.ndarray): Array of minimum distances per frame.
        log_file_base (str): Base filename or identifier used in logging.
        compare_with_training (bool): Whether to compare training and evaluation sets.
        train_mask (np.ndarray): Boolean array for training frames.
        eval_mask (np.ndarray): Boolean array for evaluation frames.
    
    Returns:
        None
    """
    print(f"\n--- Generating MLFF Stats Plots (Base: {log_file_base}) ---")
    if stats is None:
        print("Error: MLFFStats object is missing. Skipping plots.")
        return
    diag_dir = "diagnostics"
    os.makedirs(diag_dir, exist_ok=True)
    save_path_res = os.path.join(diag_dir, f"{os.path.basename(log_file_base)}_mlff_residuals.png")
    save_path_par = os.path.join(diag_dir, f"{os.path.basename(log_file_base)}_mlff_parity.png")

    try:
        x_distances = np.array(min_distances_all).flatten() if min_distances_all is not None else None
        can_plot_residuals = x_distances is not None and len(x_distances) == len(stats.true_energies)
        if not can_plot_residuals:
            print("Warning: Cannot plot residuals vs. distance (length mismatch or no distances).")
    
        energy_error_per_atom = np.abs(stats.delta_E_frame) / stats.atom_counts
        force_rmse_per_frame = stats.force_rmse_per_frame
        force_mae_per_frame = stats.force_mae_per_frame
        mae_energy_atom_mean = stats.mae_energy / np.mean(stats.atom_counts)
        rmse_energy_atom_mean = stats.rmse_energy / np.mean(stats.atom_counts)
        mae_force_mean = stats.mae_force_comp
        rmse_force_mean = stats.rmse_force_comp
        optimal_energy_atom = 0.002
        optimal_force = 0.02
        title_suffix = " (Train/Eval)" if compare_with_training else ""
        train_idx = np.where(train_mask)[0]
        eval_idx = np.where(eval_mask)[0]
    
        if can_plot_residuals:
            fig_res, axes_res = plt.subplots(2, 2, figsize=(14, 11))
            fig_res.suptitle(f'MLFF Residuals vs. Distance{title_suffix}', fontsize=16)
            # Energy MAE
            ax = axes_res[0, 0]
            ax.set_title("Energy MAE")
            if compare_with_training:
                ax.scatter(x_distances[train_idx], energy_error_per_atom[train_idx],
                           c='red', alpha=0.5, label='Train', s=10)
                ax.scatter(x_distances[eval_idx], energy_error_per_atom[eval_idx],
                           c='blue', alpha=0.5, label='Eval', s=10)
            else:
                ax.scatter(x_distances, energy_error_per_atom, c='blue', alpha=0.5, label='Data', s=10)
            ax.axhline(mae_energy_atom_mean, color='green', linestyle='-',
                       label=f'Mean:{mae_energy_atom_mean:.4f}')
            ax.axhline(optimal_energy_atom, color='black', linestyle='--',
                       label=f'Opt:{optimal_energy_atom:.4f}')
            ax.set_xlabel('Min Dist (SOAP PCA)')
            ax.set_ylabel('Energy MAE (eV/atom)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            # Energy RMSE
            ax = axes_res[0, 1]
            ax.set_title("Energy RMSE")
            if compare_with_training:
                ax.scatter(x_distances[train_idx], energy_error_per_atom[train_idx],
                           c='red', alpha=0.5, label='Train', s=10)
                ax.scatter(x_distances[eval_idx], energy_error_per_atom[eval_idx],
                           c='blue', alpha=0.5, label='Eval', s=10)
            else:
                ax.scatter(x_distances, energy_error_per_atom, c='blue', alpha=0.5, label='Data', s=10)
            ax.axhline(rmse_energy_atom_mean, color='purple', linestyle='-',
                       label=f'Mean:{rmse_energy_atom_mean:.4f}')
            ax.axhline(optimal_energy_atom, color='black', linestyle='--',
                       label=f'Opt:{optimal_energy_atom:.4f}')
            ax.set_xlabel('Min Dist (SOAP PCA)')
            ax.set_ylabel('Energy RMSE (eV/atom)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            # Force MAE
            ax = axes_res[1, 0]
            ax.set_title("Force MAE (per Frame)")
            if compare_with_training:
                ax.scatter(x_distances[train_idx], force_mae_per_frame[train_idx],
                           c='red', alpha=0.5, label='Train', s=10)
                ax.scatter(x_distances[eval_idx], force_mae_per_frame[eval_idx],
                           c='blue', alpha=0.5, label='Eval', s=10)
            else:
                ax.scatter(x_distances, force_mae_per_frame, c='blue', alpha=0.5, label='Data', s=10)
            ax.axhline(mae_force_mean, color='green', linestyle='-',
                       label=f'Mean Comp:{mae_force_mean:.3f}')
            ax.axhline(optimal_force, color='black', linestyle='--',
                       label=f'Opt:{optimal_force:.3f}')
            ax.set_xlabel('Min Dist (SOAP PCA)')
            ax.set_ylabel('Force MAE (eV/Å)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            # Force RMSE
            ax = axes_res[1, 1]
            ax.set_title("Force RMSE (per Frame)")
            if compare_with_training:
                ax.scatter(x_distances[train_idx], force_rmse_per_frame[train_idx],
                           c='red', alpha=0.5, label='Train', s=10)
                ax.scatter(x_distances[eval_idx], force_rmse_per_frame[eval_idx],
                           c='blue', alpha=0.5, label='Eval', s=10)
            else:
                ax.scatter(x_distances, force_rmse_per_frame, c='blue', alpha=0.5, label='Data', s=10)
            ax.axhline(rmse_force_mean, color='purple', linestyle='-',
                       label=f'Mean Comp:{rmse_force_mean:.3f}')
            ax.axhline(optimal_force, color='black', linestyle='--',
                       label=f'Opt:{optimal_force:.3f}')
            ax.set_xlabel('Min Dist (SOAP PCA)')
            ax.set_ylabel('Force RMSE (eV/Å)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(save_path_res, dpi=150)
            plt.close(fig_res)
            print(f"Saved residual plots to {save_path_res}")
        else:
            print("Skipping residual vs. distance plots.")

        # --- Parity Plots ---
        fig_parity, axes_parity = plt.subplots(1, 3, figsize=(18, 5.5))
        fig_parity.suptitle(f'MLFF Parity Plots{title_suffix}', fontsize=16)
        # Energy Parity
        true_e_atom = stats.true_energy / stats.atom_counts
        pred_e_atom = stats.pred_energy / stats.atom_counts
        min_true_e = np.min(true_e_atom)
        true_e_rel = true_e_atom - min_true_e
        pred_e_rel = pred_e_atom - min_true_e
        ax = axes_parity[0]
        ax.set_title('Energy (eV/atom, Relative)')
        if compare_with_training:
            ax.scatter(true_e_rel[train_idx], pred_e_rel[train_idx],
                       alpha=0.5, c='red', label='Train', s=10)
            ax.scatter(true_e_rel[eval_idx], pred_e_rel[eval_idx],
                       alpha=0.5, c='blue', label='Eval', s=10)
        else:
            ax.scatter(true_e_rel, pred_e_rel, alpha=0.5, c='blue', label='Data', s=10)
        min_v = np.min(true_e_rel)
        max_v = np.max(true_e_rel)
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', label='y=x')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        # Force Norm Parity
        true_f_norm = stats.true_force_per_atom_norm
        pred_f_norm = stats.pred_force_per_atom_norm
        ax = axes_parity[1]
        ax.set_title('Force Norm (eV/Å, per Atom)')
        if compare_with_training:
            atom_train_mask = stats._get_atom_mask(train_mask)
            atom_eval_mask = stats._get_atom_mask(eval_mask)
            ax.scatter(true_f_norm[atom_train_mask], pred_f_norm[atom_train_mask],
                       alpha=0.1, c='red', label='Train', s=5, rasterized=True)
            ax.scatter(true_f_norm[atom_eval_mask], pred_f_norm[atom_eval_mask],
                       alpha=0.1, c='blue', label='Eval', s=5, rasterized=True)
        else:
            ax.scatter(true_f_norm, pred_f_norm, alpha=0.1, c='blue', label='Data', s=5, rasterized=True)
        min_v = min(np.min(true_f_norm), np.min(pred_f_norm))
        max_v = max(np.max(true_f_norm), np.max(pred_f_norm))
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', label='y=x')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        # Force Component Parity
        true_f_comp = (stats.all_force_residuals + stats.pred_forces_flat).flatten()
        pred_f_comp = stats.pred_forces_flat.flatten()
        ax = axes_parity[2]
        ax.set_title('Force Component (eV/Å)')
        if compare_with_training:
            comp_train_mask = np.repeat(stats._get_atom_mask(train_mask), 3)
            comp_eval_mask = np.repeat(stats._get_atom_mask(eval_mask), 3)
            step_tr = max(1, len(true_f_comp[comp_train_mask]) // 100000)
            step_ev = max(1, len(true_f_comp[comp_eval_mask]) // 100000)
            ax.scatter(true_f_comp[comp_train_mask][::step_tr],
                       pred_f_comp[comp_train_mask][::step_tr],
                       alpha=0.05, c='red', label='Train', s=1, rasterized=True)
            ax.scatter(true_f_comp[comp_eval_mask][::step_ev],
                       pred_f_comp[comp_eval_mask][::step_ev],
                       alpha=0.05, c='blue', label='Eval', s=1, rasterized=True)
        else:
            step = max(1, len(true_f_comp) // 200000)
            ax.scatter(true_f_comp[::step], pred_f_comp[::step],
                       alpha=0.05, c='blue', label='Data', s=1, rasterized=True)
        min_v = min(np.min(true_f_comp), np.min(pred_f_comp))
        max_v = max(np.max(true_f_comp), np.max(pred_f_comp))
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', label='y=x')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path_par, dpi=150)
        plt.close(fig_parity)
        print(f"Saved parity plots to {save_path_par}")
    
    except Exception as plot_err:
        print(f"Error during MLFF stats plotting: {plot_err}")
        traceback.print_exc()
        plt.close("all")


# End of plotting.py

