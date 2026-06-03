import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import halfnorm, norm, spearmanr
import matplotlib.colors as mcolors

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


def _pick(var_arr, iso_arr, legacy_arr, mode):
    """Return the chosen calibration array."""
    if mode == "var" and var_arr is not None:
        return var_arr
    if mode == "iso" and iso_arr is not None:
        return iso_arr
    # fallback – legacy NPZs or wrong mode
    return legacy_arr


def plot_rmse_rmv_per_bin(delta, sigma, label, color, ax, n_bins=20):
    idx = np.argsort(sigma)
    delta = delta[idx]
    sigma = sigma[idx]
    N = len(sigma)
    bins = np.array_split(np.arange(N), n_bins)
    rmses, rmvs = [], []
    for bin_idx in bins:
        d = delta[bin_idx]
        s = sigma[bin_idx]
        if len(d) == 0:
            continue
        rmses.append(np.sqrt(np.mean(d**2)))
        rmvs.append(np.sqrt(np.mean(s**2)))
    ax.plot(rmvs, rmses, '+', color=color, label=label, markersize=7)
    return np.array(rmses), np.array(rmvs)

def compute_ence(rmses, rmvs):
    # Prevent division by zero
    rmvs_nonzero = np.where(rmvs == 0, 1e-8, rmvs)
    return np.mean(np.abs(rmses - rmvs) / rmvs_nonzero)


def _plot_reliability_gap(p_nom, cov_u, cov_c, title, filename):
    if p_nom is None or cov_u is None or len(p_nom) == 0: return
    plt.figure(figsize=(5, 4))
    plt.axhline(0, color="k", lw=1)
    plt.plot(p_nom, cov_u - p_nom, "-o", ms=3, label="uncal")
    if cov_c is not None and len(cov_c):
        plt.plot(p_nom, cov_c - p_nom, "-s", ms=3, label="cal")
    plt.xlabel("nominal interval prob.");  plt.ylabel("coverage − nominal")
    plt.title(title);  plt.grid(alpha=0.3);  plt.legend(fontsize=8)
    plt.tight_layout();  plt.savefig(filename, dpi=150);  plt.close()
    print(f"Generated reliability gap ➜ {filename}")


def _plot_zscore_hist_qq_compare(delta, sigma_raw, sigma_cal, title_base, f_hist, f_qq, bins=60):
    if delta is None or sigma_raw is None or len(delta) == 0: return
    z_raw = delta / sigma_raw
    z_cal = delta / sigma_cal

    # Histogram
    plt.figure(figsize=(5, 4))
    plt.hist(z_raw, bins=bins, density=True, alpha=0.5, color="royalblue", label="Raw z-scores")
    plt.hist(z_cal, bins=bins, density=True, alpha=0.5, color="crimson", label="Calibrated z-scores")
    xs = np.linspace(-4, 4, 400)
    plt.plot(xs, 1/np.sqrt(2*np.pi)*np.exp(-0.5*xs**2), "k--", lw=1, label="N(0,1)")
    plt.xlabel("z");  plt.ylabel("density")
    plt.title(title_base + " – hist")
    plt.legend(fontsize=8);  plt.tight_layout();  plt.savefig(f_hist, dpi=150);  plt.close()

    # QQ plot (use reduced quantiles for speed)
    n_quantiles = min(400, len(z_raw), len(z_cal))
    per = np.linspace(0, 1, n_quantiles+2)[1:-1]
    q_emp_raw  = np.quantile(z_raw, per)
    q_emp_cal  = np.quantile(z_cal, per)
    q_theo     = norm.ppf(per)
    plt.figure(figsize=(4, 4))
    plt.scatter(q_theo, q_emp_raw, s=8, alpha=0.5, color="royalblue", label="Raw")
    plt.scatter(q_theo, q_emp_cal, s=8, alpha=0.5, color="crimson", label="Calibrated")
    lim = [min(q_emp_raw.min(), q_emp_cal.min(), q_theo.min()), max(q_emp_raw.max(), q_emp_cal.max(), q_theo.max())]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("theoretical N(0,1) quantile");  plt.ylabel("empirical quantile")
    plt.title(title_base + " – QQ");  plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f_qq, dpi=150);  plt.close()
    print(f"Generated z-score hist & QQ (compare) ➜ {f_hist}, {f_qq}")



# =============================================================================
#  Updated swapped‑axis |Δ| vs σ
# =============================================================================


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

