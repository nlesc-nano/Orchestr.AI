# src/orchestr_ai/postprocessing/inference_runner.py

from __future__ import annotations

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
    ):
        self.calculator = calculator
        self.batch_size = batch_size
        self.log_file = log_file
        self.clear_cuda_cache = clear_cuda_cache

    @staticmethod
    def _is_cuda_oom(exc):
        msg = str(exc).lower()
        return "cuda out of memory" in msg or "outofmemoryerror" in msg

    def _run_batch_recursive(self, batch_frames, n_atoms_list, batch_start, depth=0):
        try:
            t0_prep = time.time()
            inputs = self.calculator.prepare_batch(batch_frames)
            prep_time = time.time() - t0_prep

            t0_forward = time.time()
            energies, forces_list, lat_frame, lat_atom = self.calculator.forward(
                inputs,
                n_atoms_list,
            )
            forward_time = time.time() - t0_forward
            return energies, forces_list, lat_frame, lat_atom, prep_time, forward_time
        except Exception as exc:
            if not self._is_cuda_oom(exc) or len(batch_frames) <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = len(batch_frames) // 2
            print(
                f"[InferenceRunner] CUDA OOM for batch starting at frame {batch_start} "
                f"(size={len(batch_frames)}). Retrying as {mid}+{len(batch_frames)-mid}."
            )
            left = self._run_batch_recursive(
                batch_frames[:mid],
                n_atoms_list[:mid],
                batch_start,
                depth + 1,
            )
            right = self._run_batch_recursive(
                batch_frames[mid:],
                n_atoms_list[mid:],
                batch_start + mid,
                depth + 1,
            )
            energies = np.concatenate([np.asarray(left[0]), np.asarray(right[0])])
            forces_list = list(left[1]) + list(right[1])
            lat_frame = list(left[2]) + list(right[2])
            lat_atom = list(left[3]) + list(right[3])
            return (
                energies,
                forces_list,
                lat_frame,
                lat_atom,
                float(left[4]) + float(right[4]),
                float(left[5]) + float(right[5]),
            )

    def run(self, frames, true_energies=None, true_forces=None):
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

        for batch_start in range(0, n_frames, self.batch_size):
            batch_frames = frames[batch_start : batch_start + self.batch_size]
            actual_size = len(batch_frames)
            n_atoms_list = [len(f) for f in batch_frames]

            batch_start_time = time.time()

            try:
                energies, forces_list, lat_frame, lat_atom, prep_time, forward_time = (
                    self._run_batch_recursive(
                        batch_frames,
                        n_atoms_list,
                        batch_start,
                    )
                )

                all_energy_pred.extend(energies)
                all_forces_pred.extend(forces_list)
                all_latent_frame.extend(lat_frame)
                all_latent_atom.extend(lat_atom)

                if self.log_file:
                    for i in range(actual_size):
                        global_idx = batch_start + i

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

        print("\n--- Inference Summary ---")
        print(
            f"Total Time: {cum_eval_time:.3f}s | "
            f"Avg Time/Frame: {cum_eval_time / max(1, n_frames):.5f}s"
        )
        print("-------------------------\n")

        return all_energy_pred, all_forces_pred, all_latent_frame, all_latent_atom
