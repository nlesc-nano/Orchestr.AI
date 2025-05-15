"""
stats.py

This module defines the MLFFStats class, which computes and stores statistics for 
ML force-field evaluation. It calculates energy and force residuals (e.g. RMSE, MAE) 
for per-frame, per-atom, and component-level errors. The class also provides a range 
of properties for easy access to computed metrics.
"""

import numpy as np


class MLFFStats:
    """
    Computes and stores statistics for ML force-field evaluation.

    Attributes:
        true_energies (np.ndarray): Reference energies for each frame.
        pred_energies (np.ndarray): Predicted energies for each frame.
        true_forces (list of np.ndarray): List of true force arrays for each frame.
        pred_forces (list of np.ndarray): List of predicted force arrays for each frame.
        train_mask (np.ndarray): Boolean mask for training frames.
        eval_mask (np.ndarray): Boolean mask for evaluation frames.
        atom_counts (np.ndarray): Number of atoms per frame.
        n_atoms_per_frame (np.ndarray): Alias for atom_counts.
        delta_E_frame (np.ndarray): Per-frame energy residuals.
        force_rmse_per_frame (np.ndarray): Per-frame force RMSE (unnormalized).
        force_mae_per_frame (np.ndarray): Per-frame force MAE.
        force_rmse_per_atom (np.ndarray): Per-atom force RMSE.
        all_force_residuals (np.ndarray): Concatenated force residuals (N_total_atoms, 3).
        pred_forces_flat (np.ndarray): Flattened predicted forces (used for plotting).
        energy_metrics (dict): Aggregated energy metrics.
        force_metrics (dict): Aggregated force metrics.
    """

    def __init__(self, true_energies, pred_energies, true_forces, pred_forces, train_mask, eval_mask):
        # Convert inputs and store masks
        self.true_energies = np.asarray(true_energies, dtype=float)
        self.pred_energies = np.asarray(pred_energies, dtype=float)

        # Forces can be lists of arrays if atom counts vary; convert carefully
        self.true_forces = [np.asarray(f, dtype=float) for f in true_forces]
        self.pred_forces = [np.asarray(f, dtype=float) for f in pred_forces]

        self.train_mask = np.asarray(train_mask, dtype=bool)
        self.eval_mask = np.asarray(eval_mask, dtype=bool)

        # Validate that all input lists/arrays have the same number of frames
        n_frames = len(self.true_energies)
        if not (len(self.pred_energies) == n_frames and
                len(self.true_forces) == n_frames and
                len(self.pred_forces) == n_frames and
                len(self.train_mask) == n_frames and
                len(self.eval_mask) == n_frames):
            raise ValueError("Input array/list lengths do not match number of frames.")

        # Store atom counts per frame
        self.atom_counts = np.array([len(f) for f in self.true_forces], dtype=int)
        self.n_atoms_per_frame = self.atom_counts  # Alias for consistency

        # Check that force arrays match the expected shapes based on atom counts
        for i in range(n_frames):
            if self.atom_counts[i] > 0:
                if (self.true_forces[i].shape != (self.atom_counts[i], 3) or
                        self.pred_forces[i].shape != (self.atom_counts[i], 3)):
                    raise ValueError(
                        f"Force shape mismatch for frame {i}. Expected ({self.atom_counts[i]}, 3), "
                        f"Got True: {self.true_forces[i].shape}, Pred: {self.pred_forces[i].shape}"
                    )
            elif self.true_forces[i].size != 0 or self.pred_forces[i].size != 0:
                raise ValueError(
                    f"Frame {i} has 0 atoms but non-empty force arrays."
                )

        # Compute metrics
        self.delta_E_frame = self._compute_delta_e()
        self.force_rmse_per_frame = self._compute_force_rmse_per_frame()
        self.force_mae_per_frame = self._compute_force_mae_per_frame()
        self.force_rmse_per_atom = self._compute_force_rmse_per_atom()
        self.all_force_residuals = self._compute_force_residuals()

        # Flatten predicted forces (used for parity plots)
        if any(f.size > 0 for f in self.pred_forces):
            self.pred_forces_flat = np.concatenate([p for p in self.pred_forces if p.size > 0], axis=0)
        else:
            self.pred_forces_flat = np.empty((0, 3))

        self.energy_metrics = self._compute_energy_metrics()
        self.force_metrics = self._compute_force_metrics()

    def _compute_delta_e(self):
        """Compute per-frame energy residuals."""
        delta = self.pred_energies - self.true_energies
        return delta

    def _compute_force_rmse_per_frame(self):
        """Compute per-frame force RMSE (unnormalized)."""
        rmses = []
        for true, pred in zip(self.true_forces, self.pred_forces):
            if true.size == 0:
                rmses.append(np.nan)
                continue
            frame_residuals_sq = (true - pred) ** 2
            mean_sq_error = np.nanmean(frame_residuals_sq)
            rmse = np.sqrt(mean_sq_error) if not np.isnan(mean_sq_error) else np.nan
            rmses.append(rmse)
        result = np.array(rmses)
        return result

    def _compute_force_mae_per_frame(self):
        """Compute per-frame force MAE."""
        maes = []
        for true, pred in zip(self.true_forces, self.pred_forces):
            if true.size == 0:
                maes.append(np.nan)
                continue
            frame_abs_residuals = np.abs(true - pred)
            mae = np.nanmean(frame_abs_residuals)
            maes.append(mae)
        result = np.array(maes)
        return result

    def _compute_force_rmse_per_atom(self):
        """Compute per-atom force RMSE."""
        rmse_list = []
        for true, pred in zip(self.true_forces, self.pred_forces):
            if true.size == 0:
                continue
            per_atom_sq_error = (true - pred) ** 2
            per_atom_mean_sq_error = np.nanmean(per_atom_sq_error, axis=1)
            per_atom_rmse = np.sqrt(per_atom_mean_sq_error)
            rmse_list.append(per_atom_rmse)
        if rmse_list:
            all_rmse = np.concatenate(rmse_list)
        else:
            all_rmse = np.array([])
        print(f"Per-atom force RMSE: mean={np.nanmean(all_rmse):.4f}, NaNs={np.isnan(all_rmse).sum()}, total={len(all_rmse)}")
        return all_rmse

    def _compute_force_residuals(self):
        """Concatenate force residuals across all frames."""
        residuals = [t - p for t, p in zip(self.true_forces, self.pred_forces) if t.size > 0]
        if not residuals:
            return np.empty((0, 3), dtype=float)
        result = np.concatenate(residuals, axis=0)
        return result

    def _compute_metrics_subset(self, metric_array, mask):
        """
        Compute the mean and standard deviation of a metric for a subset defined by mask.
        """
        if metric_array is None or len(metric_array) == 0 or len(mask) != len(metric_array):
            return np.nan, np.nan
        subset = metric_array[mask]
        if subset.size == 0:
            return np.nan, np.nan
        mean_val = np.nanmean(subset)
        std_val = np.nanstd(subset)
        return mean_val, std_val

    def _get_atom_mask(self, frame_mask):
        """
        Create a per-atom mask from a per-frame mask.

        Parameters:
            frame_mask (array-like): Boolean array for frames.

        Returns:
            np.ndarray: Boolean array for each atom in all frames.
        """
        if len(frame_mask) != len(self.atom_counts):
            raise ValueError("Frame mask length doesn't match number of frames in atom_counts.")
        atom_mask_list = [
            np.ones(count, dtype=bool) if fm else np.zeros(count, dtype=bool)
            for fm, count in zip(frame_mask, self.atom_counts)
        ]
        if not atom_mask_list:
            return np.array([], dtype=bool)
        return np.concatenate(atom_mask_list)

    def _compute_energy_metrics(self):
        """Compute aggregated energy metrics (MAE, RMSE, etc.)."""
        abs_errors = np.abs(self.delta_E_frame)
        mae_comb, std_mae_comb = self._compute_metrics_subset(abs_errors, np.ones_like(self.train_mask, dtype=bool))
        sq_errors = self.delta_E_frame ** 2
        mse_comb, std_mse_comb = self._compute_metrics_subset(sq_errors, np.ones_like(self.train_mask, dtype=bool))
        rmse_comb = np.sqrt(mse_comb) if not np.isnan(mse_comb) else np.nan
        _, std_delta_e_comb = self._compute_metrics_subset(self.delta_E_frame, np.ones_like(self.train_mask, dtype=bool))

        mae_train, std_mae_train = self._compute_metrics_subset(abs_errors, self.train_mask)
        mse_train, _ = self._compute_metrics_subset(sq_errors, self.train_mask)
        rmse_train = np.sqrt(mse_train) if not np.isnan(mse_train) else np.nan
        _, std_delta_e_train = self._compute_metrics_subset(self.delta_E_frame, self.train_mask)

        mae_eval, std_mae_eval = self._compute_metrics_subset(abs_errors, self.eval_mask)
        mse_eval, _ = self._compute_metrics_subset(sq_errors, self.eval_mask)
        rmse_eval = np.sqrt(mse_eval) if not np.isnan(mse_eval) else np.nan
        _, std_delta_e_eval = self._compute_metrics_subset(self.delta_E_frame, self.eval_mask)

        metrics = {
            "mae_combined": mae_comb, "std_mae_combined": std_mae_comb,
            "rmse_combined": rmse_comb, "std_delta_e_combined": std_delta_e_comb,
            "mae_train": mae_train, "std_mae_train": std_mae_train,
            "rmse_train": rmse_train, "std_delta_e_train": std_delta_e_train,
            "mae_eval": mae_eval, "std_mae_eval": std_mae_eval,
            "rmse_eval": rmse_eval, "std_delta_e_eval": std_delta_e_eval
        }
        print("Energy metrics computed:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        return metrics

    def _compute_force_metrics(self):
        """Compute aggregated force metrics (frame-wise, atom-wise, and component-wise)."""
        rmse_f_comb, _ = self._compute_metrics_subset(self.force_rmse_per_frame, np.ones_like(self.train_mask, dtype=bool))
        rmse_f_train, _ = self._compute_metrics_subset(self.force_rmse_per_frame, self.train_mask)
        rmse_f_eval, _ = self._compute_metrics_subset(self.force_rmse_per_frame, self.eval_mask)
        mae_f_comb, _ = self._compute_metrics_subset(self.force_mae_per_frame, np.ones_like(self.train_mask, dtype=bool))
        mae_f_train, _ = self._compute_metrics_subset(self.force_mae_per_frame, self.train_mask)
        mae_f_eval, _ = self._compute_metrics_subset(self.force_mae_per_frame, self.eval_mask)

        atom_train_mask = self._get_atom_mask(self.train_mask)
        atom_eval_mask = self._get_atom_mask(self.eval_mask)
        atom_all_mask = np.ones_like(atom_train_mask, dtype=bool)

        rmse_atom_comb, _ = self._compute_metrics_subset(self.force_rmse_per_atom, atom_all_mask)
        rmse_atom_train, _ = self._compute_metrics_subset(self.force_rmse_per_atom, atom_train_mask)
        rmse_atom_eval, _ = self._compute_metrics_subset(self.force_rmse_per_atom, atom_eval_mask)

        flat_residuals = self.all_force_residuals.flatten()
        flat_abs_residuals = np.abs(flat_residuals)
        comp_train_mask = np.repeat(atom_train_mask, 3)
        comp_eval_mask = np.repeat(atom_eval_mask, 3)
        comp_all_mask = np.ones_like(comp_train_mask, dtype=bool)

        mae_forces_comb, std_mae_forces_comb = self._compute_metrics_subset(flat_abs_residuals, comp_all_mask)
        mse_forces_comb, std_mse_forces_comb = self._compute_metrics_subset(flat_residuals**2, comp_all_mask)
        rmse_forces_comb = np.sqrt(mse_forces_comb) if not np.isnan(mse_forces_comb) else np.nan
        _, std_res_force_comb = self._compute_metrics_subset(flat_residuals, comp_all_mask)

        mae_forces_train, std_mae_forces_train = self._compute_metrics_subset(flat_abs_residuals, comp_train_mask)
        mse_forces_train, _ = self._compute_metrics_subset(flat_residuals**2, comp_train_mask)
        rmse_forces_train = np.sqrt(mse_forces_train) if not np.isnan(mse_forces_train) else np.nan
        _, std_res_force_train = self._compute_metrics_subset(flat_residuals, comp_train_mask)

        mae_forces_eval, std_mae_forces_eval = self._compute_metrics_subset(flat_abs_residuals, comp_eval_mask)
        mse_forces_eval, _ = self._compute_metrics_subset(flat_residuals**2, comp_eval_mask)
        rmse_forces_eval = np.sqrt(mse_forces_eval) if not np.isnan(mse_forces_eval) else np.nan
        _, std_res_force_eval = self._compute_metrics_subset(flat_residuals, comp_eval_mask)

        metrics = {
            "rmse_frame_combined": rmse_f_comb, "mae_frame_combined": mae_f_comb,
            "rmse_frame_train": rmse_f_train, "mae_frame_train": mae_f_train,
            "rmse_frame_eval": rmse_f_eval, "mae_frame_eval": mae_f_eval,
            "rmse_atom_combined": rmse_atom_comb,
            "rmse_atom_train": rmse_atom_train,
            "rmse_atom_eval": rmse_atom_eval,
            "mae_comp_combined": mae_forces_comb, "std_mae_comp_combined": std_mae_forces_comb,
            "rmse_comp_combined": rmse_forces_comb, "std_res_comp_combined": std_res_force_comb,
            "mae_comp_train": mae_forces_train, "std_mae_comp_train": std_mae_forces_train,
            "rmse_comp_train": rmse_forces_train, "std_res_comp_train": std_res_force_train,
            "mae_comp_eval": mae_forces_eval, "std_mae_comp_eval": std_mae_forces_eval,
            "rmse_comp_eval": rmse_forces_eval, "std_res_comp_eval": std_res_force_eval
        }
        print("Force metrics computed:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
        return metrics

    # ---------------------- PROPERTY DEFINITIONS ---------------------- #
    # Energy metrics
    @property
    def mae_energy(self):
        return self.energy_metrics["mae_combined"]

    @property
    def std_mae_energy(self):
        return self.energy_metrics["std_mae_combined"]

    @property
    def rmse_energy(self):
        return self.energy_metrics["rmse_combined"]

    @property
    def std_delta_e(self):
        return self.energy_metrics["std_delta_e_combined"]

    @property
    def mae_energy_train(self):
        return self.energy_metrics["mae_train"]

    @property
    def std_mae_energy_train(self):
        return self.energy_metrics["std_mae_train"]

    @property
    def rmse_energy_train(self):
        return self.energy_metrics["rmse_train"]

    @property
    def std_delta_e_train(self):
        return self.energy_metrics["std_delta_e_train"]

    @property
    def mae_energy_eval(self):
        return self.energy_metrics["mae_eval"]

    @property
    def std_mae_energy_eval(self):
        return self.energy_metrics["std_mae_eval"]

    @property
    def rmse_energy_eval(self):
        return self.energy_metrics["rmse_eval"]

    @property
    def std_delta_e_eval(self):
        return self.energy_metrics["std_delta_e_eval"]

    @property
    def true_energy(self):
        return self.true_energies

    @property
    def pred_energy(self):
        return self.pred_energies

    # Force metrics (Frame-level)
    @property
    def force_rmse_frame_array(self):
        return self.force_rmse_per_frame

    @property
    def force_mae_frame_array(self):
        return self.force_mae_per_frame

    @property
    def rmse_force_frame_train(self):
        return self.force_metrics["rmse_frame_train"]

    @property
    def rmse_force_frame_eval(self):
        return self.force_metrics["rmse_frame_eval"]

    @property
    def rmse_force_frame(self):
        return self.force_metrics["rmse_frame_combined"]

    @property
    def mae_force_frame_train(self):
        return self.force_metrics["mae_frame_train"]

    @property
    def mae_force_frame_eval(self):
        return self.force_metrics["mae_frame_eval"]

    @property
    def mae_force_frame(self):
        return self.force_metrics["mae_frame_combined"]

    # Force metrics (Atom-level)
    @property
    def force_rmse_atom_array(self):
        return self.force_rmse_per_atom

    @property
    def rmse_force_atom_train(self):
        return self.force_metrics["rmse_atom_train"]

    @property
    def rmse_force_atom_eval(self):
        return self.force_metrics["rmse_atom_eval"]

    @property
    def rmse_force_atom(self):
        return self.force_metrics["rmse_atom_combined"]

    # Force metrics (Component-level)
    @property
    def mae_force_comp(self):
        return self.force_metrics["mae_comp_combined"]

    @property
    def std_mae_force_comp(self):
        return self.force_metrics["std_mae_comp_combined"]

    @property
    def rmse_force_comp(self):
        return self.force_metrics["rmse_comp_combined"]

    @property
    def std_res_force_comp(self):
        return self.force_metrics["std_res_comp_combined"]

    @property
    def mae_force_comp_train(self):
        return self.force_metrics["mae_comp_train"]

    @property
    def std_mae_force_comp_train(self):
        return self.force_metrics["std_mae_comp_train"]

    @property
    def rmse_force_comp_train(self):
        return self.force_metrics["rmse_comp_train"]

    @property
    def std_res_force_comp_train(self):
        return self.force_metrics["std_res_comp_train"]

    @property
    def mae_force_comp_eval(self):
        return self.force_metrics["mae_comp_eval"]

    @property
    def std_mae_force_comp_eval(self):
        return self.force_metrics["std_mae_comp_eval"]

    @property
    def rmse_force_comp_eval(self):
        return self.force_metrics["rmse_comp_eval"]

    @property
    def std_res_force_comp_eval(self):
        return self.force_metrics["std_res_comp_eval"]

    # For parity plots (Per-atom force norms)
    @property
    def true_force_per_atom_norm(self):
        if any(f.size > 0 for f in self.true_forces):
            return np.concatenate([np.linalg.norm(f, axis=1) for f in self.true_forces if f.size > 0])
        else:
            return np.array([])

    @property
    def pred_force_per_atom_norm(self):
        if any(f.size > 0 for f in self.pred_forces):
            return np.concatenate([np.linalg.norm(f, axis=1) for f in self.pred_forces if f.size > 0])
        else:
            return np.array([])

