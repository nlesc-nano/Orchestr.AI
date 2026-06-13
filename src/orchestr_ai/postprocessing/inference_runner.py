# src/orchestr_ai/postprocessing/inference_runner.py

from __future__ import annotations

import gc
import time
import traceback

import numpy as np
import torch


class InferenceRunner:
    """
    Orchestrates batching, timing, logging, and standardized output collection.
    """

    def __init__(
        self,
        calculator,
        batch_size: int,
        log_file: str | None = None,
        clear_cuda_cache: bool = False,
        context_label: str = "InferenceRunner",
    ):
        self.calculator = calculator
        self.batch_size = batch_size
        self.log_file = log_file
        self.clear_cuda_cache = clear_cuda_cache
        self.context_label = context_label

    def _log(self, message):
        print(f"[{self.context_label}] {message}", flush=True)

    @staticmethod
    def _is_cuda_oom(exc):
        msg = str(exc).lower()
        return "cuda out of memory" in msg or "outofmemoryerror" in msg

    def _run_single_batch(self, batch_frames, n_atoms_list, has_multihead):
        t0_prep = time.time()
        
        if has_multihead:
            if hasattr(self.calculator, "k_E"):
                # Auto-scaled reconstruction mode:
                # Prepare the joint batch (batch_base, batch_delta)
                inputs = self.calculator.prepare_batch(batch_frames)
                prep_time = time.time() - t0_prep
                
                t0_forward = time.time()
                energies, forces_list, lat_frame, lat_atom = self.calculator.forward(
                    inputs,
                    n_atoms_list,
                )
                forward_time = time.time() - t0_forward
                
                # Retrieve singlet and triplet reconstructed predictions
                energies_s = self.calculator.last_E_singlet
                forces_s = self.calculator.last_F_singlet
                energies_t = energies
                forces_t = forces_list
                
                # Use base_head config settings for the runner's main outputs
                orig_head = self.calculator.head
                if orig_head == self.calculator.base_head:
                    energies_out = energies_s
                    forces_out = forces_s
                else:
                    energies_out = energies_t
                    forces_out = forces_t
            else:
                # Original direct singlet/triplet mode:
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
                    energies_out = energies_s
                    forces_out = forces_s
                    lat_frame = lat_frame_s
                    lat_atom = lat_atom_s
                else:
                    energies_out = energies_t
                    forces_out = forces_t
                    lat_frame = lat_frame_t
                    lat_atom = lat_atom_t
            
            return (
                energies_out,
                forces_out,
                lat_frame,
                lat_atom,
                prep_time,
                forward_time,
                energies_s,
                forces_s,
                energies_t,
                forces_t,
            )
        else:
            inputs = self.calculator.prepare_batch(batch_frames)
            prep_time = time.time() - t0_prep

            t0_forward = time.time()
            energies, forces_list, lat_frame, lat_atom = self.calculator.forward(
                inputs,
                n_atoms_list,
            )
            forward_time = time.time() - t0_forward
            
            return energies, forces_list, lat_frame, lat_atom, prep_time, forward_time, None, None, None, None

    def _run_batch_recursive(self, batch_frames, n_atoms_list, batch_start, has_multihead, depth=0):
        try:
            return self._run_single_batch(batch_frames, n_atoms_list, has_multihead)
        except Exception as exc:
            if not self._is_cuda_oom(exc) or len(batch_frames) <= 1:
                raise
            exc.__traceback__ = None
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            mid = len(batch_frames) // 2
            self._log(
                f"CUDA OOM for local batch starting at {batch_start} "
                f"(size={len(batch_frames)}). Retrying as {mid}+{len(batch_frames)-mid}."
            )
            left = self._run_batch_recursive(
                batch_frames[:mid],
                n_atoms_list[:mid],
                batch_start,
                has_multihead,
                depth + 1,
            )
            right = self._run_batch_recursive(
                batch_frames[mid:],
                n_atoms_list[mid:],
                batch_start + mid,
                has_multihead,
                depth + 1,
            )
            
            energies = np.concatenate([np.asarray(left[0]), np.asarray(right[0])])
            forces_list = list(left[1]) + list(right[1])
            lat_frame = list(left[2]) + list(right[2])
            lat_atom = list(left[3]) + list(right[3])
            
            prep_time = float(left[4]) + float(right[4])
            forward_time = float(left[5]) + float(right[5])
            
            if has_multihead:
                energies_s = np.concatenate([np.asarray(left[6]), np.asarray(right[6])])
                forces_s = list(left[7]) + list(right[7])
                energies_t = np.concatenate([np.asarray(left[8]), np.asarray(right[8])])
                forces_t = list(left[9]) + list(right[9])
            else:
                energies_s, forces_s, energies_t, forces_t = None, None, None, None
                
            return (
                energies,
                forces_list,
                lat_frame,
                lat_atom,
                prep_time,
                forward_time,
                energies_s,
                forces_s,
                energies_t,
                forces_t,
            )

    def run(self, frames, true_energies=None, true_forces=None, E_singlet_true=None, E_triplet_true=None, frame_indices=None):
        n_frames = len(frames)
        if frame_indices is None:
            frame_indices = np.arange(n_frames, dtype=int)
        else:
            frame_indices = np.asarray(frame_indices, dtype=int)
            if len(frame_indices) != n_frames:
                raise ValueError("frame_indices must contain one entry per frame")

        all_energy_pred = []
        all_forces_pred = []
        all_latent_frame = []
        all_latent_atom = []

        cum_eval_time = 0.0
        batches_processed = 0
        log_lines_buffer = []

        self._log(
            f"Starting generic inference for {n_frames} frames "
            f"(Batch Size: {self.batch_size}, "
            f"global frame span {int(frame_indices[0]) if n_frames else 0}-"
            f"{int(frame_indices[-1]) if n_frames else -1})..."
        )

        # Check if the calculator is a multi-head MACE calculator with singlet/triplet heads or singlet/delta heads
        has_multihead = (
            hasattr(self.calculator, "available_heads")
            and "singlet" in self.calculator.available_heads
            and ("triplet" in self.calculator.available_heads or "delta" in self.calculator.available_heads)
        )

        for batch_start in range(0, n_frames, self.batch_size):
            batch_frames = frames[batch_start : batch_start + self.batch_size]
            actual_size = len(batch_frames)
            n_atoms_list = [len(f) for f in batch_frames]

            batch_start_time = time.time()

            try:
                (
                    energies,
                    forces_list,
                    lat_frame,
                    lat_atom,
                    prep_time,
                    forward_time,
                    energies_s,
                    forces_s,
                    energies_t,
                    forces_t,
                ) = self._run_batch_recursive(
                    batch_frames,
                    n_atoms_list,
                    batch_start,
                    has_multihead,
                )

                all_energy_pred.extend(energies)
                all_forces_pred.extend(forces_list)
                all_latent_frame.extend(lat_frame)
                all_latent_atom.extend(lat_atom)

                if self.log_file:
                    for i in range(actual_size):
                        local_idx = batch_start + i
                        global_idx = int(frame_indices[local_idx])

                        if has_multihead:
                            true_e_s = E_singlet_true[local_idx] if E_singlet_true is not None and local_idx < len(E_singlet_true) else np.nan
                            pred_e_s = energies_s[i]
                            diff_e_s = pred_e_s - true_e_s
                            
                            true_e_t = E_triplet_true[local_idx] if E_triplet_true is not None and local_idx < len(E_triplet_true) else np.nan
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
                                true_energies[local_idx]
                                if true_energies is not None
                                and local_idx < len(true_energies)
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
                    first_local_idx = batch_start
                    first_global_idx = int(frame_indices[first_local_idx])

                    if has_multihead:
                        # Log singlet
                        true_e_s = E_singlet_true[first_local_idx] if E_singlet_true is not None and first_local_idx < len(E_singlet_true) else np.nan
                        pred_e_s = energies_s[0]
                        diff_s = pred_e_s - true_e_s
                        print(
                            f"[{self.context_label}]   [Batch {batches_processed + 1}] Frame {first_global_idx:5d} | "
                            f"Pred E_singlet: {pred_e_s:12.4f} eV | "
                            f"True E_singlet: {true_e_s:12.4f} eV | "
                            f"Diff: {diff_s:10.4f} eV",
                            flush=True,
                        )
                        
                        # Log triplet
                        true_e_t = E_triplet_true[first_local_idx] if E_triplet_true is not None and first_local_idx < len(E_triplet_true) else np.nan
                        pred_e_t = energies_t[0]
                        diff_t = pred_e_t - true_e_t
                        print(
                            f"[{self.context_label}]   [Batch {batches_processed + 1}] Frame {first_global_idx:5d} | "
                            f"Pred E_triplet: {pred_e_t:12.4f} eV | "
                            f"True E_triplet: {true_e_t:12.4f} eV | "
                            f"Diff: {diff_t:10.4f} eV",
                            flush=True,
                        )
                        
                        # Log singlet-triplet difference
                        pred_diff = pred_e_s - pred_e_t
                        true_diff = true_e_s - true_e_t
                        diff_diff = pred_diff - true_diff
                        print(
                            f"[{self.context_label}]   [Batch {batches_processed + 1}] Frame {first_global_idx:5d} | "
                            f"Pred delta_gap: {pred_diff:12.4f} eV | "
                            f"True delta_gap: {true_diff:12.4f} eV | "
                            f"Diff: {diff_diff:10.4f} eV",
                            flush=True,
                        )
                    else:
                        true_e_first = (
                            true_energies[first_local_idx]
                            if true_energies is not None
                            and first_local_idx < len(true_energies)
                            else np.nan
                        )

                        pred_e_first = energies[0]
                        diff_first = pred_e_first - true_e_first

                        print(
                            f"[{self.context_label}]   [Batch {batches_processed + 1}] "
                            f"Frame {first_global_idx:5d} | "
                            f"Pred E: {pred_e_first:12.4f} eV | "
                            f"True E: {true_e_first:12.4f} eV | "
                            f"Diff: {diff_first:10.4f} eV",
                            flush=True,
                        )

            except Exception as e:
                self._log(f"Error processing batch {batches_processed}: {e}")
                if self._is_cuda_oom(e):
                    e.__traceback__ = None
                    self._log(
                        f"CUDA OOM persisted for frame range "
                        f"{int(frame_indices[batch_start])}-{int(frame_indices[batch_start + actual_size - 1])}; "
                        "marking this batch as NaN. Reduce eval.batch_size."
                    )
                    if torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    traceback.print_exc()

                all_energy_pred.extend([np.nan] * actual_size)
                all_forces_pred.extend(
                    [np.full((n, 3), np.nan) for n in n_atoms_list]
                )
                all_latent_frame.extend([np.nan] * actual_size)
                all_latent_atom.extend([np.nan] * actual_size)

            finally:
                if self.clear_cuda_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            cum_eval_time += time.time() - batch_start_time
            batches_processed += 1

        if self.log_file and log_lines_buffer:
            try:
                with open(self.log_file, "w", encoding="utf-8") as elog:
                    elog.writelines(log_lines_buffer)
            except IOError as log_e:
                print(f"Warning: Failed to write log file: {log_e}")

        self._log("--- Inference Summary ---")
        self._log(
            f"Total Time: {cum_eval_time:.3f}s | "
            f"Avg Time/Frame: {cum_eval_time / max(1, n_frames):.5f}s"
        )
        self._log("-------------------------")

        return all_energy_pred, all_forces_pred, all_latent_frame, all_latent_atom
