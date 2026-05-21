# src/orchestr_ai/postprocessing/inference_runner.py

from __future__ import annotations

import time
import traceback

import numpy as np


class InferenceRunner:
    """
    Orchestrates batching, timing, logging, and standardized output collection.
    """

    def __init__(self, calculator, batch_size: int, log_file: str | None = None):
        self.calculator = calculator
        self.batch_size = batch_size
        self.log_file = log_file

    def run(self, frames, true_energies=None, true_forces=None, E_singlet_true=None, E_triplet_true=None):
        n_frames = len(frames)

        all_energy_pred = []
        all_forces_pred = []
        all_latent_frame = []
        all_latent_atom = []

        cum_eval_time = 0.0
        batches_processed = 0
        log_lines_buffer = []

        print(
            f"Starting generic inference for {n_frames} frames "
            f"(Batch Size: {self.batch_size})..."
        )

        # Check if the calculator is a multi-head MACE calculator with singlet/triplet heads
        has_multihead = (
            hasattr(self.calculator, "available_heads")
            and "singlet" in self.calculator.available_heads
            and "triplet" in self.calculator.available_heads
        )

        for batch_start in range(0, n_frames, self.batch_size):
            batch_frames = frames[batch_start : batch_start + self.batch_size]
            actual_size = len(batch_frames)
            n_atoms_list = [len(f) for f in batch_frames]

            batch_start_time = time.time()

            try:
                t0_prep = time.time()
                
                if has_multihead:
                    # Run singlet
                    orig_head = self.calculator.head
                    self.calculator.head = "singlet"
                    inputs_s = self.calculator.prepare_batch(batch_frames)
                    prep_time = time.time() - t0_prep
                    
                    t0_forward = time.time()
                    energies_s, forces_s, lat_frame_s, lat_atom_s = self.calculator.forward(
                        inputs_s,
                        n_atoms_list,
                    )
                    forward_time_s = time.time() - t0_forward
                    
                    # Run triplet
                    t0_prep_t = time.time()
                    self.calculator.head = "triplet"
                    inputs_t = self.calculator.prepare_batch(batch_frames)
                    prep_time_t = time.time() - t0_prep_t
                    
                    t0_forward_t = time.time()
                    energies_t, forces_t, lat_frame_t, lat_atom_t = self.calculator.forward(
                        inputs_t,
                        n_atoms_list,
                    )
                    forward_time_t = time.time() - t0_forward_t
                    
                    # Restore original head
                    self.calculator.head = orig_head
                    
                    # Compute prep and forward times for stats
                    prep_time = (prep_time + prep_time_t) / 2.0
                    forward_time = (forward_time_s + forward_time_t) / 2.0
                    
                    # Set current batch predictions for the requested/original head
                    if orig_head == "singlet":
                        energies = energies_s
                        forces_list = forces_s
                        lat_frame = lat_frame_s
                        lat_atom = lat_atom_s
                    else:
                        energies = energies_t
                        forces_list = forces_t
                        lat_frame = lat_frame_t
                        lat_atom = lat_atom_t
                else:
                    inputs = self.calculator.prepare_batch(batch_frames)
                    prep_time = time.time() - t0_prep

                    t0_forward = time.time()
                    energies, forces_list, lat_frame, lat_atom = self.calculator.forward(
                        inputs,
                        n_atoms_list,
                    )
                    forward_time = time.time() - t0_forward

                all_energy_pred.extend(energies)
                all_forces_pred.extend(forces_list)
                all_latent_frame.extend(lat_frame)
                all_latent_atom.extend(lat_atom)

                if self.log_file:
                    for i in range(actual_size):
                        global_idx = batch_start + i

                        if has_multihead:
                            true_e_s = E_singlet_true[global_idx] if E_singlet_true is not None and global_idx < len(E_singlet_true) else np.nan
                            pred_e_s = energies_s[i]
                            diff_e_s = pred_e_s - true_e_s
                            
                            true_e_t = E_triplet_true[global_idx] if E_triplet_true is not None and global_idx < len(E_triplet_true) else np.nan
                            pred_e_t = energies_t[i]
                            diff_e_t = pred_e_t - true_e_t
                            
                            true_diff = true_e_s - true_e_t
                            pred_diff = pred_e_s - pred_e_t
                            diff_diff = pred_diff - true_diff
                            
                            if not log_lines_buffer:
                                header = (
                                    f"{'Frame':>6s} | "
                                    f"{'True_E_s(eV)':>15s} | {'Pred_E_s(eV)':>15s} | {'Diff_s(eV)':>12s} | "
                                    f"{'True_E_t(eV)':>15s} | {'Pred_E_t(eV)':>15s} | {'Diff_t(eV)':>12s} | "
                                    f"{'True_delta_gap(eV)':>18s} | {'Pred_delta_gap(eV)':>18s} | {'Diff_delta_gap(eV)':>15s}\n"
                                )
                                log_lines_buffer.append(header)
                                
                            log_lines_buffer.append(
                                f"{global_idx:6d} | "
                                f"{true_e_s:15.6f} | {pred_e_s:15.6f} | {diff_e_s:12.6f} | "
                                f"{true_e_t:15.6f} | {pred_e_t:15.6f} | {diff_e_t:12.6f} | "
                                f"{true_diff:15.6f} | {pred_diff:15.6f} | {diff_diff:12.6f}\n"
                            )
                        else:
                            true_e = (
                                true_energies[global_idx]
                                if true_energies is not None
                                and global_idx < len(true_energies)
                                else np.nan
                            )

                            pred_e = energies[i]
                            diff_e = pred_e - true_e

                            if not log_lines_buffer:
                                header = (
                                    f"{'Frame':>6s} | {'True_E(eV)':>15s} | "
                                    f"{'Pred_E(eV)':>15s} | {'Diff(eV)':>12s} | "
                                    f"{'PrepTime(s)':>12s} | {'FwdTime(s)':>12s}\n"
                                )
                                log_lines_buffer.append(header)

                            log_lines_buffer.append(
                                f"{global_idx:6d} | {true_e:15.6f} | "
                                f"{pred_e:15.6f} | {diff_e:12.6f} | "
                                f"{prep_time / actual_size:12.6f} | "
                                f"{forward_time / actual_size:12.6f}\n"
                            )

                if actual_size > 0:
                    first_idx = batch_start

                    if has_multihead:
                        # Log singlet
                        true_e_s = E_singlet_true[first_idx] if E_singlet_true is not None and first_idx < len(E_singlet_true) else np.nan
                        pred_e_s = energies_s[0]
                        diff_s = pred_e_s - true_e_s
                        print(
                            f"  [Batch {batches_processed + 1}] Frame {first_idx:5d} | "
                            f"Pred E_singlet: {pred_e_s:12.4f} eV | "
                            f"True E_singlet: {true_e_s:12.4f} eV | "
                            f"Diff: {diff_s:10.4f} eV"
                        )
                        
                        # Log triplet
                        true_e_t = E_triplet_true[first_idx] if E_triplet_true is not None and first_idx < len(E_triplet_true) else np.nan
                        pred_e_t = energies_t[0]
                        diff_t = pred_e_t - true_e_t
                        print(
                            f"  [Batch {batches_processed + 1}] Frame {first_idx:5d} | "
                            f"Pred E_triplet: {pred_e_t:12.4f} eV | "
                            f"True E_triplet: {true_e_t:12.4f} eV | "
                            f"Diff: {diff_t:10.4f} eV"
                        )
                        
                        # Log singlet-triplet difference
                        pred_diff = pred_e_s - pred_e_t
                        true_diff = true_e_s - true_e_t
                        diff_diff = pred_diff - true_diff
                        print(
                            f"  [Batch {batches_processed + 1}] Frame {first_idx:5d} | "
                            f"Pred delta_gap: {pred_diff:12.4f} eV | "
                            f"True delta_gap: {true_diff:12.4f} eV | "
                            f"Diff: {diff_diff:10.4f} eV"
                        )
                    else:
                        true_e_first = (
                            true_energies[first_idx]
                            if true_energies is not None
                            and first_idx < len(true_energies)
                            else np.nan
                        )

                        pred_e_first = energies[0]
                        diff_first = pred_e_first - true_e_first

                        print(
                            f"  [Batch {batches_processed + 1}] "
                            f"Frame {first_idx:5d} | "
                            f"Pred E: {pred_e_first:12.4f} eV | "
                            f"True E: {true_e_first:12.4f} eV | "
                            f"Diff: {diff_first:10.4f} eV"
                        )

            except Exception as e:
                print(f"Error processing batch {batches_processed}: {e}")
                traceback.print_exc()

                all_energy_pred.extend([np.nan] * actual_size)
                all_forces_pred.extend(
                    [np.full((n, 3), np.nan) for n in n_atoms_list]
                )
                all_latent_frame.extend([np.nan] * actual_size)
                all_latent_atom.extend([np.nan] * actual_size)

            cum_eval_time += time.time() - batch_start_time
            batches_processed += 1

        if self.log_file and log_lines_buffer:
            try:
                with open(self.log_file, "w", encoding="utf-8") as elog:
                    elog.writelines(log_lines_buffer)
            except IOError as log_e:
                print(f"Warning: Failed to write log file: {log_e}")

        print("\n--- Inference Summary ---")
        print(
            f"Total Time: {cum_eval_time:.3f}s | "
            f"Avg Time/Frame: {cum_eval_time / max(1, n_frames):.5f}s"
        )
        print("-------------------------\n")

        return all_energy_pred, all_forces_pred, all_latent_frame, all_latent_atom