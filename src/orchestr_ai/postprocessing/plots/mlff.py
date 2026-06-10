import os
import traceback
import numpy as np
import matplotlib.pyplot as plt

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

