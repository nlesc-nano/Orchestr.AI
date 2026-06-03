import os
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .utils import (
    _ideal_colour,
    plot_scalar_metrics,
    plot_coverage_curve,
    plot_sigma_density,
    _pick,
    plot_rmse_rmv_per_bin,
    compute_ence,
    _plot_reliability_gap,
    _plot_zscore_hist_qq_compare,
    plot_swapped_final_tight,
    plot_original_final_tight
)

def generate_uq_plots(npz_plot_data_path, set_name, set_uq,
                      ensemble_size=None, norm_energy=False,
                      calibration="var"):  #  NEW ARG
    """
    calibration: "var", "iso" or "legacy"
    """
    print(f"\n--- Generating UQ Plots for: {set_name} Set ({set_uq}, {calibration}) from {npz_plot_data_path} ---")
    if not os.path.exists(npz_plot_data_path):
        print(f"Error: Plot data file not found: {npz_plot_data_path}. Skipping plots.");  return

    # --------------------------------------------------------------
    try:
        data = np.load(npz_plot_data_path, allow_pickle=True)

        # component-level
        sigma_c_uncal    = data.get("sigma_comp_uncal")
        sigma_c_cal_var  = data.get("sigma_comp_cal_var")
        sigma_c_cal_iso  = data.get("sigma_comp_cal_iso")
        sigma_c_cal_legacy = data.get("sigma_comp_cal")             # for old archives
        delta_c          = data["delta_comp"]
        err_c_abs        = np.abs(delta_c)

        # frame-level energy
        sigma_E_uncal    = data.get("sigma_energy_uncal")
        sigma_E_cal_var  = data.get("sigma_energy_cal_var")
        sigma_E_cal_iso  = data.get("sigma_energy_cal_iso")
        sigma_E_cal_legacy = data.get("sigma_energy_cal")
        delta_E          = data.get("delta_energy")
        print(delta_E.mean(), np.median(delta_E))
        err_E_abs        = np.abs(delta_E) if delta_E is not None else None

        # precomputed scalars & coverage
        scalar_metrics   = data.get("scalar_metrics", {}).item()
        p_nominal        = data.get("p_thresholds")
        cov_uncal        = data.get("coverage_uncal")
        cov_cal_var      = data.get("coverage_cal_var")
        cov_cal_iso      = data.get("coverage_cal_iso")
        cov_cal_legacy   = data.get("coverage_cal")
        cov_uncal_e      = data.get("coverage_uncal_e")
        cov_cal_var_e    = data.get("coverage_cal_var_e")
        cov_cal_iso_e    = data.get("coverage_cal_iso_e")
        cov_cal_legacy_e = data.get("coverage_cal_e")
        n_atoms_frame    = data.get("n_atoms_per_frame")
    except Exception as e:
        print(f"Error loading NPZ: {e}");  traceback.print_exc();  return

    # Pick which calibrated arrays to feed downstream  -----------------
    sigma_c_cal = _pick(sigma_c_cal_var,  sigma_c_cal_iso,  sigma_c_cal_legacy,  calibration)
    sigma_E_cal = _pick(sigma_E_cal_var,  sigma_E_cal_iso,  sigma_E_cal_legacy,  calibration)
    cov_cal     = _pick(cov_cal_var,      cov_cal_iso,      cov_cal_legacy,      calibration)
    cov_cal_e   = _pick(cov_cal_var_e,    cov_cal_iso_e,    cov_cal_legacy_e,    calibration)

    plot_dir   = os.path.dirname(npz_plot_data_path);  os.makedirs(plot_dir, exist_ok=True)
    ens_str    = f"_ens{ensemble_size}" if ensemble_size else ""
    base_part  = f"{set_name.lower()}_{set_uq.lower()}_{calibration}{ens_str}"
    title_base = f"{set_name} ({set_uq}, {calibration}{ens_str})"

    # ------------------------------------------------------------------
    # 1. scalar metrics bar‐chart (unchanged) ---------------------------
    try:
        plot_scalar_metrics(
            scalar_metrics,
            f"{title_base} – scalar metrics",
            os.path.join(plot_dir, f"{base_part}_scalar_metrics.png")
        )
    except Exception as e:
        print(f"Error plotting scalar metrics: {e}")

    # ------------------------------------------------------------------
    # 2. coverage & reliability curves ---------------------------------
    try:
        if p_nominal is not None and cov_uncal is not None:
            # coverage
            plot_coverage_curve(
                p_nominal, cov_uncal, cov_cal,
                f"{title_base} – coverage (forces)",
                os.path.join(plot_dir, f"{base_part}_coverage_forces.png")
            )
            # reliability gap
            _plot_reliability_gap(
                p_nominal, cov_uncal, cov_cal,
                f"{title_base} – reliability gap (forces)",
                os.path.join(plot_dir, f"{base_part}_reliability_forces.png")
            )
        if p_nominal is not None and cov_uncal_e is not None and len(cov_uncal_e):
            plot_coverage_curve(
                p_nominal, cov_uncal_e, cov_cal_e,
                f"{title_base} – coverage (energy)",
                os.path.join(plot_dir, f"{base_part}_coverage_energy.png")
            )
            _plot_reliability_gap(
                p_nominal, cov_uncal_e, cov_cal_e,
                f"{title_base} – reliability gap (energy)",
                os.path.join(plot_dir, f"{base_part}_reliability_energy.png")
            )
    except Exception as e:
        print(f"Error plotting coverage curves: {e}")

    # ------------------------------------------------------------------
    # 3. σ density histograms (unchanged) ------------------------------
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

    # 7. RMSE versus RMV  ----------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(4,4))
        # Per-bin points for RAW and CAL
        rmses_raw, rmvs_raw = plot_rmse_rmv_per_bin(delta_c, sigma_c_uncal, "Raw", "royalblue", ax)
        rmses_cal, rmvs_cal = plot_rmse_rmv_per_bin(delta_c, sigma_c_cal, "Calibrated", "crimson", ax)
        # Ideal line
        lim = (0, max(np.max(rmvs_raw), np.max(rmses_raw), np.max(rmvs_cal), np.max(rmses_cal))*1.05)
        ax.plot([lim[0], lim[1]], [lim[0], lim[1]], "k--", lw=1, label="ideal")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("RMV (per bin)")
        ax.set_ylabel("RMSE (per bin)")
        ax.set_title(f"{title_base} – RMSE vs RMV (forces)")
        ax.legend(fontsize=8)
        # ENCE annotation
        ence = compute_ence(rmses_cal, rmvs_cal)
        ax.text(0.05, 0.90, f"ENCE = {ence:.2f}", transform=ax.transAxes, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{base_part}_rmse_rmv_forces_perbin.png"), dpi=150)
        plt.close()
        print(f"Generated RMSE-RMV (per bin) ➜ {os.path.join(plot_dir, f'{base_part}_rmse_rmv_forces_perbin.png')}")
    
        # If energies are available
        if sigma_E_uncal is not None and len(sigma_E_uncal):
            fig, ax = plt.subplots(figsize=(4,4))
            rmses_raw_e, rmvs_raw_e = plot_rmse_rmv_per_bin(delta_E, sigma_E_uncal, "Raw", "royalblue", ax)
            rmses_cal_e, rmvs_cal_e = plot_rmse_rmv_per_bin(delta_E, sigma_E_cal, "Calibrated", "crimson", ax)
            lim_e = (0, max(np.max(rmvs_raw_e), np.max(rmses_raw_e), np.max(rmvs_cal_e), np.max(rmses_cal_e))*1.05)
            ax.plot([lim_e[0], lim_e[1]], [lim_e[0], lim_e[1]], "k--", lw=1, label="ideal")
            ax.set_xlim(lim_e)
            ax.set_ylim(lim_e)
            ax.set_xlabel("RMV (per bin)")
            ax.set_ylabel("RMSE (per bin)")
            ax.set_title(f"{title_base} – RMSE vs RMV (energy)")
            ax.legend(fontsize=8)
            ence_e = compute_ence(rmses_cal_e, rmvs_cal_e)
            ax.text(0.05, 0.90, f"ENCE = {ence_e:.2f}", transform=ax.transAxes, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{base_part}_rmse_rmv_energy_perbin.png"), dpi=150)
            plt.close()
            print(f"Generated RMSE-RMV (per bin) ➜ {os.path.join(plot_dir, f'{base_part}_rmse_rmv_energy_perbin.png')}")
    
    except Exception as e:
        print(f"Error plotting RMSE-RMV: {e}")
    
    # ------------------------------------------------------------------
    # 8. z-score diagnostics  ------------------------------------------
    try:
        # Forces
        _plot_zscore_hist_qq_compare(
            delta_c, sigma_c_uncal, sigma_c_cal,
            title_base + " (forces)",
            os.path.join(plot_dir, f"{base_part}_z_hist_forces_compare.png"),
            os.path.join(plot_dir, f"{base_part}_z_qq_forces_compare.png")
        )
        # Energies
        if sigma_E_cal is not None and len(sigma_E_cal):
            _plot_zscore_hist_qq_compare(
                delta_E, sigma_E_uncal, sigma_E_cal,
                title_base + " (energy)",
                os.path.join(plot_dir, f"{base_part}_z_hist_energy_compare.png"),
                os.path.join(plot_dir, f"{base_part}_z_qq_energy_compare.png")
            )
    except Exception as e:
        print(f"Error plotting z-scores: {e}")
    
    print(f"Finished UQ plotting for {set_name} ({set_uq}, {calibration})")
    plt.close("all")

# === Plotting Helpers Specific to Traditional Active Learning ===

