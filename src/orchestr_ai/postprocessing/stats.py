"""
stats.py

This module defines the MLFFStats class, which computes and stores statistics for 
ML force-field evaluation. It calculates energy and force residuals (e.g. RMSE, MAE) 
for per-frame, per-atom, and component-level errors.
"""

import numpy as np

class MLFFStats:
    def __init__(self, true_energies, pred_energies, true_forces, pred_forces, train_mask, eval_mask):
        self.true_energies = np.asarray(true_energies, dtype=float)
        self.pred_energies = np.asarray(pred_energies, dtype=float)
        self.true_forces = [np.asarray(f, dtype=float) for f in true_forces]
        self.pred_forces = [np.asarray(f, dtype=float) for f in pred_forces]

        self.train_mask = np.asarray(train_mask, dtype=bool)
        self.eval_mask = np.asarray(eval_mask, dtype=bool)

        self.atom_counts = np.array([len(f) for f in self.true_forces], dtype=int)
        self.n_atoms_per_frame = self.atom_counts 

        self.delta_E_frame = self._compute_delta_e()
        self.force_rmse_per_frame = self._compute_force_rmse_per_frame()
        self.force_mae_per_frame = self._compute_force_mae_per_frame()
        self.force_rmse_per_atom = self._compute_force_rmse_per_atom()
        self.all_force_residuals = self._compute_force_residuals()

        self.energy_metrics = self._compute_energy_metrics()
        self.force_metrics = self._compute_force_metrics()

    def _compute_delta_e(self): return self.pred_energies - self.true_energies

    def _compute_force_rmse_per_frame(self):
        return np.array([np.sqrt(np.nanmean((t - p) ** 2)) if t.size > 0 else np.nan for t, p in zip(self.true_forces, self.pred_forces)])

    def _compute_force_mae_per_frame(self):
        return np.array([np.nanmean(np.abs(t - p)) if t.size > 0 else np.nan for t, p in zip(self.true_forces, self.pred_forces)])

    def _compute_force_rmse_per_atom(self):
        rmse_list = [np.sqrt(np.nanmean((t - p) ** 2, axis=1)) for t, p in zip(self.true_forces, self.pred_forces) if t.size > 0]
        return np.concatenate(rmse_list) if rmse_list else np.array([])

    def _compute_force_residuals(self):
        residuals = [t - p for t, p in zip(self.true_forces, self.pred_forces) if t.size > 0]
        return np.concatenate(residuals, axis=0) if residuals else np.empty((0, 3), dtype=float)

    def _get_atom_mask(self, frame_mask):
        return np.concatenate([np.ones(c, dtype=bool) if m else np.zeros(c, dtype=bool) for m, c in zip(frame_mask, self.atom_counts)])

    def _compute_metrics_subset(self, metric_array, mask):
        if metric_array is None or len(metric_array) == 0 or len(mask) != len(metric_array): return np.nan, np.nan
        subset = metric_array[mask]
        return (np.nan, np.nan) if subset.size == 0 else (np.nanmean(subset), np.nanstd(subset))

    def _compute_energy_metrics(self):
        abs_e, sq_e = np.abs(self.delta_E_frame), self.delta_E_frame ** 2
        all_m = np.ones_like(self.train_mask, dtype=bool)
        
        mae_c, _ = self._compute_metrics_subset(abs_e, all_m)
        rmse_c = np.sqrt(self._compute_metrics_subset(sq_e, all_m)[0])
        mae_t, _ = self._compute_metrics_subset(abs_e, self.train_mask)
        rmse_t = np.sqrt(self._compute_metrics_subset(sq_e, self.train_mask)[0])
        mae_e, _ = self._compute_metrics_subset(abs_e, self.eval_mask)
        rmse_e = np.sqrt(self._compute_metrics_subset(sq_e, self.eval_mask)[0])

        print("\n" + "="*50)
        print(f"{'ENERGY METRICS (eV)':^50}")
        print("="*50)
        print(f"{'Metric':<15} | {'Train':<10} | {'Eval':<10} | {'Combined':<10}")
        print("-" * 50)
        print(f"{'MAE':<15} | {mae_t:<10.5f} | {mae_e:<10.5f} | {mae_c:<10.5f}")
        print(f"{'RMSE':<15} | {rmse_t:<10.5f} | {rmse_e:<10.5f} | {rmse_c:<10.5f}")

        return {"mae_train": mae_t, "rmse_train": rmse_t, "mae_eval": mae_e, "rmse_eval": rmse_e}

    def _compute_force_metrics(self):
        atom_t_mask, atom_e_mask = self._get_atom_mask(self.train_mask), self._get_atom_mask(self.eval_mask)
        comp_t_mask, comp_e_mask = np.repeat(atom_t_mask, 3), np.repeat(atom_e_mask, 3)
        all_comp_mask = np.ones_like(comp_t_mask, dtype=bool)
        all_atom_mask = np.ones_like(atom_t_mask, dtype=bool)

        abs_res, sq_res = np.abs(self.all_force_residuals.flatten()), self.all_force_residuals.flatten()**2

        rmse_a_t, _ = self._compute_metrics_subset(self.force_rmse_per_atom, atom_t_mask)
        rmse_a_e, _ = self._compute_metrics_subset(self.force_rmse_per_atom, atom_e_mask)
        rmse_a_c, _ = self._compute_metrics_subset(self.force_rmse_per_atom, all_atom_mask)

        mae_c_t, _ = self._compute_metrics_subset(abs_res, comp_t_mask)
        mae_c_e, _ = self._compute_metrics_subset(abs_res, comp_e_mask)
        mae_c_c, _ = self._compute_metrics_subset(abs_res, all_comp_mask)

        print("\n" + "="*50)
        print(f"{'FORCE METRICS (eV/Å)':^50}")
        print("="*50)
        print(f"{'Metric':<15} | {'Train':<10} | {'Eval':<10} | {'Combined':<10}")
        print("-" * 50)
        print(f"{'MAE (Comp)':<15} | {mae_c_t:<10.5f} | {mae_c_e:<10.5f} | {mae_c_c:<10.5f}")
        print(f"{'RMSE (Atom)':<15} | {rmse_a_t:<10.5f} | {rmse_a_e:<10.5f} | {rmse_a_c:<10.5f}")
        print("="*50 + "\n")

        return {"mae_comp_train": mae_c_t, "rmse_atom_train": rmse_a_t, "mae_comp_eval": mae_c_e, "rmse_atom_eval": rmse_a_e}

