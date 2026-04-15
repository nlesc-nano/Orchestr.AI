import os
import traceback
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

###############################################################
#                    Helper plotting utilities               #
###############################################################

def _compute_basic_stats(true_arr: np.ndarray, pred_arr: np.ndarray):
    """Return MAE, RMSE and R² (coefficient of determination)."""
    residuals = pred_arr - true_arr
    mae = float(np.nanmean(np.abs(residuals)))
    rmse = float(np.sqrt(np.nanmean(residuals ** 2)))
    # R² (manual to avoid sklearn dependency)
    ss_res = float(np.nansum(residuals ** 2))
    ss_tot = float(np.nansum((true_arr - np.nanmean(true_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2


def _downsample(true_arr, pred_arr, mask: Optional[np.ndarray], max_points: int):
    """Sub‑sample large arrays for faster and lighter plotting."""
    n = len(true_arr)
    if n <= max_points:
        return true_arr, pred_arr, mask
    step = n // max_points
    idx = slice(None, None, step)
    if mask is None:
        return true_arr[idx], pred_arr[idx], None
    return true_arr[idx], pred_arr[idx], mask[idx]


def _plot_parity(ax, true_arr, pred_arr, label: str,
                 color, alpha=0.3, s=10, max_points=200_000,
                 diag_kwargs=None, subtitle_stats=True):
    """Scatter parity plot with stats annotation and y=x line."""
    true_arr, pred_arr, _ = _downsample(true_arr, pred_arr, None, max_points)
    ax.scatter(true_arr, pred_arr, c=color, alpha=alpha, s=s, label=label, rasterized=True)

    # Diagonal y=x line
    min_v = np.nanmin(np.concatenate([true_arr, pred_arr]))
    max_v = np.nanmax(np.concatenate([true_arr, pred_arr]))
    ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=1)

    mae, rmse, r2 = _compute_basic_stats(true_arr, pred_arr)
    if subtitle_stats:
        ax.set_title(f"{label}\nMAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.grid(alpha=0.3)


def _plot_residual_hist(ax, residuals, title: str, bins: int = 100, log: bool = True):
    """Histogram (optionally log‑scaled) of residuals."""
    residuals = residuals[~np.isnan(residuals)]
    ax.hist(residuals, bins=bins, log=log, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(5))


def _plot_ecdf(ax, abs_errors, title: str):
    """Empirical CDF of absolute errors."""
    abs_errors = abs_errors[~np.isnan(abs_errors)]
    sorted_err = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax.plot(sorted_err, cdf)
    ax.set_title(title)
    ax.set_xlabel("|Error|")
    ax.set_ylabel("Fraction ≤ x")
    ax.grid(alpha=0.3)


###############################################################
#                     Main entry point                        #
###############################################################

def plot_mlff_stats(
    stats: "MLFFStats",
    min_distances_all: Optional[np.ndarray],
    log_file_base: str,
    compare_with_training: bool,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    save_dir: str = "diagnostics",
):
    """Generate and save a standard set of MLFF diagnostic figures.

    Parameters
    ----------
    stats : MLFFStats
        Populated statistics object.
    min_distances_all : array‑like or None
        Per‑frame descriptor used for extrapolation diagnostics (optional).
    log_file_base : str
        Identifier used in output filenames.
    compare_with_training : bool
        Whether to split plots into train/eval subsets.
    train_mask / eval_mask : np.ndarray
        Boolean masks for frames.
    save_dir : str, default "diagnostics"
        Where to save the PNGs.
    """
    print(f"\n— Generating MLFF diagnostic plots for {log_file_base} —")
    os.makedirs(save_dir, exist_ok=True)

    try:
        # ------------------------------------------------------------------
        # 1. Energy & force parity plots
        # ------------------------------------------------------------------
        fig_par, axes_par = plt.subplots(1, 2, figsize=(12, 5))
        title_suffix = " (Train/Eval)" if compare_with_training else ""
        fig_par.suptitle(f"MLFF Parity Plots{title_suffix}", fontsize=15)

        # Energy per atom parity
        true_e_atom = stats.true_energy / stats.atom_counts
        pred_e_atom = stats.pred_energy / stats.atom_counts

        if compare_with_training:
            _plot_parity(
                axes_par[0],
                true_e_atom[train_mask],
                pred_e_atom[train_mask],
                label="Energy – Train",
                color="red",
            )
            _plot_parity(
                axes_par[0],
                true_e_atom[eval_mask],
                pred_e_atom[eval_mask],
                label="Energy – Eval",
                color="blue",
                alpha=0.3,
            )
            axes_par[0].legend(fontsize=7)
        else:
            _plot_parity(
                axes_par[0], true_e_atom, pred_e_atom, label="Energy", color="blue"
            )

        # Force component parity
        true_f_comp = (stats.all_force_residuals + stats.pred_forces_flat).flatten()
        pred_f_comp = stats.pred_forces_flat.flatten()

        if compare_with_training:
            comp_train_mask = np.repeat(stats._get_atom_mask(train_mask), 3)
            comp_eval_mask = np.repeat(stats._get_atom_mask(eval_mask), 3)
            _plot_parity(
                axes_par[1],
                true_f_comp[comp_train_mask],
                pred_f_comp[comp_train_mask],
                label="Force comp – Train",
                color="red",
                alpha=0.15,
                s=3,
            )
            _plot_parity(
                axes_par[1],
                true_f_comp[comp_eval_mask],
                pred_f_comp[comp_eval_mask],
                label="Force comp – Eval",
                color="blue",
                alpha=0.15,
                s=3,
            )
            axes_par[1].legend(fontsize=7)
        else:
            _plot_parity(
                axes_par[1],
                true_f_comp,
                pred_f_comp,
                label="Force component",
                color="blue",
                alpha=0.15,
                s=3,
            )

        fig_par.tight_layout(rect=[0, 0, 1, 0.94])
        fname_par = os.path.join(save_dir, f"{os.path.basename(log_file_base)}_parity.png")
        fig_par.savefig(fname_par, dpi=150)
        plt.close(fig_par)
        print(f"  • Saved parity plots to {fname_par}")

        # ------------------------------------------------------------------
        # 2. Residual histograms + ECDFs
        # ------------------------------------------------------------------
        fig_hist, axes_hist = plt.subplots(2, 2, figsize=(10, 8))
        fig_hist.suptitle("Residual distributions", fontsize=15)

        # Energy residuals per atom
        energy_residuals = pred_e_atom - true_e_atom
        _plot_residual_hist(axes_hist[0, 0], energy_residuals, "Energy (eV/atom)")
        _plot_ecdf(axes_hist[0, 1], np.abs(energy_residuals), "Energy |error| ECDF")

        # Force component residuals
        force_residuals = pred_f_comp - true_f_comp
        _plot_residual_hist(axes_hist[1, 0], force_residuals, "Force comp (eV/Å)")
        _plot_ecdf(axes_hist[1, 1], np.abs(force_residuals), "Force component |error| ECDF")

        fig_hist.tight_layout(rect=[0, 0, 1, 0.94])
        fname_hist = os.path.join(save_dir, f"{os.path.basename(log_file_base)}_residual_dists.png")
        fig_hist.savefig(fname_hist, dpi=150)
        plt.close(fig_hist)
        print(f"  • Saved residual histograms to {fname_hist}")

        # ------------------------------------------------------------------
        # 3. Error vs. min‑distance (optional extrapolation diagnostic)
        # ------------------------------------------------------------------
        if min_distances_all is not None and len(min_distances_all) == len(stats.true_energy):
            fig_scatter, ax_sc = plt.subplots(figsize=(6, 5))
            x_vals = np.asarray(min_distances_all).flatten()
            energy_mae_per_frame = np.abs(stats.delta_E_frame) / stats.atom_counts

            if compare_with_training:
                ax_sc.scatter(
                    x_vals[train_mask],
                    energy_mae_per_frame[train_mask],
                    c="red",
                    alpha=0.4,
                    s=10,
                    label="Train",
                )
                ax_sc.scatter(
                    x_vals[eval_mask],
                    energy_mae_per_frame[eval_mask],
                    c="blue",
                    alpha=0.4,
                    s=10,
                    label="Eval",
                )
            else:
                ax_sc.scatter(
                    x_vals, energy_mae_per_frame, c="blue", alpha=0.4, s=10, label="Data"
                )
            ax_sc.set_xlabel("Min SOAP distance")
            ax_sc.set_ylabel("Energy MAE per atom (eV)")
            ax_sc.set_title("Error vs. structural novelty")
            ax_sc.grid(alpha=0.3)
            ax_sc.legend(fontsize=7)
            fig_scatter.tight_layout()
            fname_sc = os.path.join(
                save_dir, f"{os.path.basename(log_file_base)}_error_vs_distance.png"
            )
            fig_scatter.savefig(fname_sc, dpi=150)
            plt.close(fig_scatter)
            print(f"  • Saved error‑vs‑distance plot to {fname_sc}")
        else:
            print("  • Skipping error‑vs‑distance plot (no distances or mismatch).")

    except Exception as err:
        print(f"[plot_mlff_stats] Error while generating plots: {err}")
        traceback.print_exc()
        plt.close("all")



