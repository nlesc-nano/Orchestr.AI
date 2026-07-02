"""
simulation.py

This module contains simulation driver functions for molecular dynamics (MD),
geometry optimization, and vibrational analysis using ASE and a custom PyTorch model.
It also provides utilities for status logging during simulations.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import traceback # Make sure traceback is imported
from types import MethodType

from pathlib import Path
from ase import units
from ase.io import read, write
from ase.md import VelocityVerlet, Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.neighborlist import neighbor_list
from ase.optimize import BFGSLineSearch, FIRE, LBFGS
from ase.vibrations import Vibrations

# --- Global Timing Variables ---
last_call_time = None
cumulative_time = 0.0

def get_ase_calculator(model, config, device, neighbor_list=None):
    """Returns the official ASE calculator for the chosen ML framework."""
    framework = config.get("model_framework", "schnetpack").lower()

    if framework in {"schnet", "painn", "so3net", "field_schnet", "fusion"}:
        framework = "schnetpack"
        if neighbor_list is None:
            raise ValueError(
                "SchNetPack ASE calculator requires neighbor_list, "
                "but neighbor_list=None was passed."
            )

        from orchestr_ai.postprocessing.calculators.schnetpack_ase import LegacyOffsetSpkCalculator

        # Return our newly wrapped calculator
        return LegacyOffsetSpkCalculator(
            model_obj=model,
            device=device,
            energy="energy",
            forces="forces",
            energy_units="eV",
            forces_units="eV/Angstrom",
            neighbor_list=neighbor_list
        )

    elif framework == "mace":
        from mace.calculators import MACECalculator
        from mace.tools import utils
        import torch

        # 1. Robustly extract and fix z_table for compiled models
        # The official calculator expects a z_table object, not a Tensor.
        z_table = None
        for attr_name in ["z_table", "atomic_numbers"]:
            if hasattr(model, attr_name):
                val = getattr(model, attr_name)
                # If it's a Tensor, convert it to the AtomicNumberTable MACE expects
                if isinstance(val, torch.Tensor):
                    z_table = utils.get_atomic_number_table_from_zs(val.detach().cpu().numpy().astype(int).tolist())
                    # Overwrite the attribute on the model so the ASE calculator finds it
                    model.z_table = z_table
                else:
                    z_table = val
                break    

        # 2. Return the official calculator using the correct plural 'models' keyword.
        mace_head = config.get("mace_head", None)
        if isinstance(mace_head, str):
            mace_head = mace_head.strip()

        # Check for user-defined dtype or default to float32
        default_dtype = config.get("default_dtype") or config.get("default_precision") or "float32"

        print(f"[MACE] Initializing calculator with precision default_dtype='{default_dtype}'")

        if mace_head == "triplet_reconstructed":
            from orchestr_ai.postprocessing.calculators.mace_calculator import (
                ReconstructedMACECalculator,
            )
            scale_metadata_path = config.get("scale_metadata_path") or config.get("mace_scale_metadata")
            if not scale_metadata_path:
                scale_metadata_path = "mace_scale_metadata.json"

            return ReconstructedMACECalculator(
                model=model,
                device=device,
                scale_metadata_path=scale_metadata_path,
                default_dtype=default_dtype,
            )

        kwargs = {
            "models": [model],
            "device": str(device),
            "default_dtype": default_dtype,
        }

        if mace_head:
            print(f"[MACE] Requested head from config: {mace_head}")
            kwargs["head"] = mace_head
        else:
            print("[MACE] No mace_head specified. Using MACE default head.")

        return MACECalculator(**kwargs)

    elif framework in {"nequip", "allegro"}:
        from nequip.ase import NequIPCalculator
        from ase.io import read
        try:
            atoms_temp = read(config.get("initial_xyz"))
            species = sorted(list(set(atoms_temp.get_chemical_symbols())))
            chemical_map = {s: s for s in species}
        except Exception:
            chemical_map = None

        return NequIPCalculator.from_compiled_model(
            compile_path=model,
            device=str(device),
            chemical_species_to_atom_type_map=chemical_map
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")


def _enable_calculator_profiling(calc, *, label="calculator", sync_cuda=False):
    """Attach low-overhead call counting/timing to an ASE calculator instance."""
    if getattr(calc, "_orchestr_profile_enabled", False):
        return calc

    original_calculate = calc.calculate
    profile = {
        "label": label,
        "calls": 0,
        "total_s": 0.0,
        "last_s": 0.0,
        "max_s": 0.0,
        "last_print_calls": 0,
        "last_print_total_s": 0.0,
        "sync_cuda": bool(sync_cuda),
    }

    def profiled_calculate(self, *args, **kwargs):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        try:
            return original_calculate(*args, **kwargs)
        finally:
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            profile["calls"] += 1
            profile["total_s"] += elapsed
            profile["last_s"] = elapsed
            if elapsed > profile["max_s"]:
                profile["max_s"] = elapsed

    calc.calculate = MethodType(profiled_calculate, calc)
    calc._orchestr_profile = profile
    calc._orchestr_profile_enabled = True
    print(
        f"[MD-PROFILE] Enabled {label} calculate() profiler "
        f"(sync_cuda={bool(sync_cuda)})."
    )
    return calc


def _get_structure_profile(atoms, cutoff):
    """Return cheap structural diagnostics for MD profiling."""
    min_distance = np.nan
    edge_count = -1

    try:
        if len(atoms) > 1:
            use_mic = bool(np.any(atoms.get_pbc()))
            distances = atoms.get_all_distances(mic=use_mic)
            distances[distances <= 0.0] = np.inf
            min_distance = float(np.min(distances))
    except Exception:
        min_distance = np.nan

    try:
        if cutoff and cutoff > 0:
            # ASE returns directed i->j pairs; this still tracks neighbor-graph size.
            edge_count = int(len(neighbor_list("i", atoms, cutoff)))
    except Exception:
        edge_count = -1

    return min_distance, edge_count


def _print_calculator_profile(
    calc,
    dyn,
    atoms,
    write_queue=None,
    cutoff=None,
    profile_file=None,
    print_to_screen=True,
):
    """Emit a profiler snapshot without calling energy/force getters."""
    profile = getattr(calc, "_orchestr_profile", None)
    if not profile:
        return

    calls = profile["calls"]
    total_s = profile["total_s"]
    delta_calls = calls - profile["last_print_calls"]
    delta_s = total_s - profile["last_print_total_s"]
    avg_s = total_s / calls if calls else 0.0
    delta_avg_s = delta_s / delta_calls if delta_calls else 0.0

    profile["last_print_calls"] = calls
    profile["last_print_total_s"] = total_s

    queue_size = "n/a"
    if write_queue is not None:
        try:
            queue_size = str(write_queue.qsize())
        except Exception:
            queue_size = "unknown"

    allocated_mb = np.nan
    reserved_mb = np.nan
    max_allocated_mb = np.nan
    cuda_msg = "cuda=n/a"
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / 1024**2
        reserved_mb = torch.cuda.memory_reserved() / 1024**2
        max_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
        cuda_msg = (
            f"cuda_alloc={allocated_mb:.1f}MB "
            f"cuda_reserved={reserved_mb:.1f}MB "
            f"cuda_max_alloc={max_allocated_mb:.1f}MB"
        )

    step = dyn.get_number_of_steps()
    min_distance, edge_count = _get_structure_profile(atoms, cutoff)
    line = (
        f"[MD-PROFILE] step={step} "
        f"{profile['label']}_calls={calls} delta_calls={delta_calls} "
        f"last_calc={profile['last_s']:.6f}s "
        f"avg_calc={avg_s:.6f}s delta_avg={delta_avg_s:.6f}s "
        f"max_calc={profile['max_s']:.6f}s queue={queue_size} "
        f"min_dist={min_distance:.4f}A edges={edge_count} {cuda_msg}"
    )

    if profile_file:
        try:
            fresh = not Path(profile_file).exists() or Path(profile_file).stat().st_size == 0
            with open(profile_file, "a") as fh:
                if fresh:
                    fh.write(
                        "# step calls delta_calls last_calc_s avg_calc_s "
                        "delta_avg_s max_calc_s queue cuda_alloc_MB "
                        "cuda_reserved_MB cuda_max_alloc_MB min_distance_A "
                        "neighbor_edges\n"
                    )
                fh.write(
                    f"{step} {calls} {delta_calls} "
                    f"{profile['last_s']:.6f} {avg_s:.6f} "
                    f"{delta_avg_s:.6f} {profile['max_s']:.6f} "
                    f"{queue_size} {allocated_mb:.1f} {reserved_mb:.1f} "
                    f"{max_allocated_mb:.1f} {min_distance:.6f} "
                    f"{edge_count}\n"
                )
        except IOError as exc:
            print(f"Warning: Failed to write MD profile line: {exc}", flush=True)

    if print_to_screen:
        print(line, flush=True)

def _reset_timers():
    """
    Resets the global timers for logging execution time.
    """
    global last_call_time, cumulative_time
    last_call_time = None
    cumulative_time = 0.0


def _log_status_line(log_file, header, values_format, values):
    """
    Helper to write a status line to a log file.

    Parameters:
      log_file (str): Path to the log file.
      header (str): Header line to write if file is empty.
      values_format (str): A format string for the values.
      values (tuple): Tuple of values to log.
    """
    if not log_file:
        return

    write_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0

    try:
        with open(log_file, "a") as lf:
            if write_header:
                lf.write(header + "\n")
            lf.write(values_format.format(*values) + "\n")
    except IOError as e:
        print(f"Warning: Failed write to log {log_file}: {e}")

def log_geo_opt_status(optimizer, atoms, log_file, trajectory_file, config=None):
    """
    Logs geometry optimization status to a log file.

    Parameters:
      optimizer: Optimizer object.
      atoms (ase.Atoms): The atomic structure.
      log_file (str): Path to the log file.
      trajectory_file (str): File name for the geometry optimization trajectory.
      config (dict, optional): Configuration parameters.
    """
    global last_call_time, cumulative_time
    now = time.time()
    step_total_time = now - last_call_time if last_call_time is not None else 0.0
    last_call_time = now
    cumulative_time += step_total_time

    step = optimizer.get_number_of_steps()
    
    if step == 0:
        log_geo_opt_status.last_epot = None
        
    e_pot_raw = atoms.get_potential_energy()
    e_pot = _get_logged_potential_energy(atoms, fallback=e_pot_raw)
    forces = atoms.get_forces(apply_constraint=False) # Get forces after energy
    max_force = np.sqrt((forces**2).sum(axis=1).max()) if len(forces) > 0 else 0.0

    last_epot = getattr(log_geo_opt_status, "last_epot", None)
    dE = e_pot - last_epot if last_epot is not None else 0.0
    log_geo_opt_status.last_epot = e_pot

    # Helper to rename long head name
    def log_name(h):
        if h == "triplet_reconstructed":
            return "trip_delta"
        return h

    # Detect multi-head MACE run
    is_mace = False
    if config is not None:
        framework = config.get("model_framework", "").lower()
        if framework == "mace":
            is_mace = True

    has_multihead = False
    selected_head = None
    other_head = None
    available_heads = []

    if is_mace and atoms.calc is not None:
        if hasattr(atoms.calc, "models") and len(atoms.calc.models) > 0:
            available_heads = getattr(atoms.calc.models[0], "heads", [])
        if not available_heads and hasattr(atoms.calc, "available_heads"):
            available_heads = atoms.calc.available_heads
        
        if available_heads:
            available_heads = [h.strip() for h in available_heads if isinstance(h, str)]

        mace_head = config.get("mace_head", None)
        if isinstance(mace_head, str):
            mace_head = mace_head.strip()

        is_reconstructed = (mace_head == "triplet_reconstructed")
        is_multi_model = ("singlet" in available_heads and ("triplet" in available_heads or "delta" in available_heads))
        
        if is_reconstructed or is_multi_model:
            has_multihead = True
            selected_head = mace_head if mace_head else "singlet"

            if selected_head == "singlet":
                other_head = "triplet" if "triplet" in available_heads else "delta"
            elif selected_head == "triplet":
                other_head = "singlet"
            elif selected_head == "delta":
                other_head = "singlet"
            elif selected_head == "triplet_reconstructed":
                other_head = "singlet"
            else:
                other_head = "singlet" if "singlet" in available_heads else None

            if other_head == "delta":
                k_E = _load_scale_metadata(config)
                if k_E is not None:
                    other_head = "triplet_reconstructed"

    epot_other = np.nan
    delta_gap = np.nan
    if has_multihead and other_head:
        epot_other = _get_other_head_energy(atoms, other_head, config)
        
        e_singlet_val = np.nan
        e_triplet_val = np.nan
        
        # Check active/selected head
        if selected_head == "singlet":
            e_singlet_val = e_pot
        elif selected_head in ("triplet", "triplet_reconstructed"):
            e_triplet_val = e_pot
            
        # Check other head
        if other_head == "singlet":
            e_singlet_val = epot_other
        elif other_head in ("triplet", "triplet_reconstructed"):
            e_triplet_val = epot_other
            
        if not np.isnan(e_singlet_val) and not np.isnan(e_triplet_val):
            delta_gap = e_singlet_val - e_triplet_val

    if has_multihead and other_head:
        col1_name = f"Epot_{log_name(selected_head)}(eV)"
        col2_name = f"Epot_{log_name(other_head)}(eV)"
        
        val_epot_str = f"{e_pot:.6f}"
        val_maxforce_str = f"{max_force:.6f}"
        val_dE_str = f"{dE:.6f}"
        val_epot_other_str = f"{epot_other:.6f}"
        val_delta_gap_str = f"{delta_gap:.6f}"
        
        col1_w = max(11, len(col1_name), len(val_epot_str))
        col2_w = max(11, len(col2_name), len(val_epot_other_str))
        gap_w = max(13, len("delta_gap(eV)"), len(val_delta_gap_str))
        force_w = max(14, len("MaxForce(eV/A)"), len(val_maxforce_str))
        de_w = max(11, len("dE(eV)"), len(val_dE_str))

        header = (
            f"{'Step':>6} | {col1_name:>{col1_w}} | {col2_name:>{col2_w}} | "
            f"{'delta_gap(eV)':>{gap_w}} | {'MaxForce(eV/A)':>{force_w}} | "
            f"{'dE(eV)':>{de_w}} | {'dt(s)':>8} | {'cum(s)':>9}"
        )
        fmt = (
            "{:6d} | "
            f"{{:{col1_w}.6f}} | {{:{col2_w}.6f}} | "
            f"{{:{gap_w}.6f}} | {{:{force_w}.6f}} | "
            f"{{:{de_w}.6f}} | {{:8.4f}} | {{:9.4f}}"
        )
        values = (
            step, e_pot, epot_other, delta_gap,
            max_force, dE, step_total_time, cumulative_time
        )
    else:
        col_name = f"Epot_{log_name(selected_head)}(eV)" if selected_head else "Epot(eV)"
        
        val_epot_str = f"{e_pot:.6f}"
        val_maxforce_str = f"{max_force:.6f}"
        val_dE_str = f"{dE:.6f}"
        
        col_w = max(11, len(col_name), len(val_epot_str))
        force_w = max(14, len("MaxForce(eV/A)"), len(val_maxforce_str))
        de_w = max(11, len("dE(eV)"), len(val_dE_str))

        header = (
            f"{'Step':>6} | {col_name:>{col_w}} | {'MaxForce(eV/A)':>{force_w}} | "
            f"{'dE(eV)':>{de_w}} | {'dt(s)':>8} | {'cum(s)':>9}"
        )
        fmt = (
            "{:6d} | "
            f"{{:{col_w}.6f}} | {{:{force_w}.6f}} | "
            f"{{:{de_w}.6f}} | {{:8.4f}} | {{:9.4f}}"
        )
        values = (
            step, e_pot, max_force, dE,
            step_total_time, cumulative_time
        )

    _log_status_line(log_file, header, fmt, values)

    if trajectory_file:
        try:
            # Write using ASE's write function for extxyz format
            write(trajectory_file, atoms, append=True, format='extxyz')
        except IOError as e:
            print(f"Warning: Failed to write trajectory frame {step}: {e}")


def log_vib_opt_status(optimizer, atoms, log_file, trajectory_file, config=None):
    """
    Logs vibrational optimization status by reusing the geometry optimization logger.
    """
    log_geo_opt_status(optimizer, atoms, log_file, trajectory_file, config=config)


# === Simulation Drivers ===

def run_geo_opt(atoms, model_obj, device, config, neighbor_list=None):
    """
    Runs geometry optimization using ASE's BFGSLineSearch optimizer.

    Parameters:
      atoms (ase.Atoms): The atomic structure to be optimized.
      model_obj: Pre-trained PyTorch model.
      device: Torch device.
      neighbor_list: Neighbor list object.
      config (dict): Configuration parameters.
    """
    _reset_timers()
    geo_config = config.get("geo_opt", {})
    geo_opt_fmax = geo_config.get("geo_opt_fmax", 0.02)
    geo_opt_steps = geo_config.get("geo_opt_steps", 500)
    trajectory_file = geo_config.get("trajectory_file_geo_opt", "geo_opt_trajectory.xyz")
    log_file = geo_config.get("log_file_geo_opt", "simulation_opt.log")

    print("Setting up calculator for Geometry Optimization...")
    # Correct instantiation using the signature from calculator.py
    calc = get_ase_calculator(model_obj, config, device, neighbor_list)
    atoms.calc = calc

    print(f"Running Geometry Optimization (fmax={geo_opt_fmax}, steps={geo_opt_steps})...")
    # Use atoms directly, no need for Optimizable wrapper unless constraints change
    opt_type = str(geo_config.get("optimizer", "bfgs")).lower().strip()
    if opt_type == "lbfgs":
        print("Using LBFGS optimizer (forces-only).")
        optimizer = LBFGS(atoms, logfile=None, maxstep=0.04)
    elif opt_type == "fire":
        print("Using FIRE optimizer (forces-only).")
        optimizer = FIRE(atoms, logfile=None, maxstep=0.04)
    else:
        print("Using BFGSLineSearch optimizer (energy + forces).")
        optimizer = BFGSLineSearch(atoms, logfile=None, maxstep=0.04)

    # Pass optimizer itself to the logger function
    optimizer.attach(
        lambda opt=optimizer: log_geo_opt_status(opt, atoms, log_file, trajectory_file, config=config),
        interval=1
    )

    try:
        optimizer.run(fmax=geo_opt_fmax, steps=geo_opt_steps)
    except Exception as e:
        print(f"Error during geometry optimization: {e}")
        import traceback
        traceback.print_exc()

    print("Geometry Optimization Finished.")


def _log_status_line(log_file, header, fmt, values):
    """Append one nicely formatted line to *log_file* (create if absent)."""
    if log_file and hasattr(log_file, "put"):
        # Push message to background thread queue
        log_file.put(("log", (header, fmt, values)))
        return

    line = fmt.format(*values)
    if not log_file:
        print(header)
        print(line)
        return

    if isinstance(log_file, (str, Path)):
        fresh = not Path(log_file).exists() or Path(log_file).stat().st_size == 0
        with open(log_file, "a") as fh:
            if fresh:
                fh.write(header + "\n")
            fh.write(line + "\n")
    else:
        # It's an open file handle
        try:
            is_empty = log_file.tell() == 0
        except Exception:
            is_empty = False
        if is_empty:
            log_file.write(header + "\n")
        log_file.write(line + "\n")
        log_file.flush()


def _write_xyz_frame(atoms, step, md_time, T_set, friction, e_pot, file_handle):
    """Append one extended-XYZ frame with positions, velocities, and forces."""
    if not file_handle:
        return

    frame = atoms.copy()
    velocities = atoms.get_velocities()
    if velocities is None:
        velocities = np.full((len(atoms), 3), np.nan, dtype=float)

    frame.set_array("velocities", np.asarray(velocities, dtype=float))
    frame.set_array("forces", np.asarray(atoms.get_forces(), dtype=float))
    frame.info.update(
        {
            "step": int(step),
            "time_fs": float(md_time),
            "temperature_set_K": float(T_set) if np.isfinite(T_set) else np.nan,
            "friction_fs_inv": float(friction),
            "energy": float(e_pot),
        }
    )

    try:
        write(file_handle, frame, format="extxyz")
        file_handle.flush()
    except IOError as exc:
        print(f"Warning: Failed to write MD trajectory frame {step}: {exc}")

def _load_scale_metadata(config):
    """Loads scale factor k_E from mace_scale_metadata.json (or config override) if available."""
    if not isinstance(config, dict):
        return None
    import json
    import os
    scale_metadata_path = config.get("scale_metadata_path") or config.get("mace_scale_metadata")
    if not scale_metadata_path:
        scale_metadata_path = "mace_scale_metadata.json"
    if os.path.exists(scale_metadata_path):
        try:
            with open(scale_metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return float(meta.get("k_E", 1.0))
        except Exception as e:
            print(f"Warning: Failed to load scale metadata from {scale_metadata_path}: {e}")
    return None


def _sum_result_energy(results, *, per_atom_key="energies", scalar_key="energy"):
    """Return a float64 log energy from calculator results when possible."""
    if not isinstance(results, dict):
        return np.nan

    per_atom = results.get(per_atom_key)
    if per_atom is not None:
        arr = np.asarray(per_atom, dtype=np.float64)
        if arr.size and np.all(np.isfinite(arr)):
            return float(arr.sum(dtype=np.float64))

    scalar = results.get(scalar_key)
    if scalar is None:
        return np.nan
    try:
        return float(np.asarray(scalar, dtype=np.float64).reshape(-1)[0])
    except Exception:
        return np.nan


def _get_logged_potential_energy(atoms, fallback=None):
    """Prefer per-atom MACE energies to avoid float32 total-energy quantization in logs."""
    if atoms.calc is not None and hasattr(atoms.calc, "results"):
        energy = _sum_result_energy(atoms.calc.results)
        if np.isfinite(energy):
            return energy
    return float(fallback) if fallback is not None else np.nan


def _get_other_head_energy(atoms, other_head, config):
    """
    Safely calculates or retrieves potential energy for other_head.
    Restores the original calculator head and results to keep MD forces/cache intact.
    """
    if atoms.calc is None:
        return np.nan

    # If it is ReconstructedMACECalculator, we might already have the cached energies:
    calc_results = getattr(atoms.calc, "results", {})
    if other_head == "singlet" and "energies_singlet" in calc_results:
        return _sum_result_energy(
            calc_results,
            per_atom_key="energies_singlet",
            scalar_key="energy_singlet",
        )
    if other_head == "triplet_reconstructed" and "energies_triplet_reconstructed" in calc_results:
        return _sum_result_energy(
            calc_results,
            per_atom_key="energies_triplet_reconstructed",
            scalar_key="energy_triplet_reconstructed",
        )
    if other_head == "singlet" and "energy_singlet" in calc_results:
        return calc_results["energy_singlet"]
    if other_head == "triplet_reconstructed" and "energy_triplet_reconstructed" in calc_results:
        return calc_results["energy_triplet_reconstructed"]

    reconstruct_triplet = (other_head == "triplet_reconstructed" or other_head == "triplet")

    try:
        # Save original calculator state
        old_results = dict(atoms.calc.results) if hasattr(atoms.calc, "results") else {}
        old_head = getattr(atoms.calc, "head", None)

        from ase.calculators.calculator import all_changes
        
        # Calculate energy for delta or standard head
        target_head = "delta" if (reconstruct_triplet and not hasattr(atoms.calc, "available_heads")) else other_head
        if hasattr(atoms.calc, "available_heads") and target_head not in atoms.calc.available_heads:
            if target_head == "triplet_reconstructed" and "delta" in atoms.calc.available_heads:
                target_head = "delta"

        atoms.calc.head = target_head
        atoms.calc.calculate(atoms, properties=["energy"], system_changes=all_changes)
        calculated_energy = _sum_result_energy(atoms.calc.results)

        if reconstruct_triplet and target_head == "delta":
            k_E = _load_scale_metadata(config)
            E_singlet = _sum_result_energy(old_results)
            if k_E is not None and np.isfinite(E_singlet):
                E_delta = calculated_energy
                calculated_energy = E_singlet - (E_delta / k_E)
            else:
                calculated_energy = np.nan

        return calculated_energy

    except Exception as e:
        print(f"Warning: Failed to compute energy for other head '{other_head}': {e}")
        return np.nan
    finally:
        # Restore original calculator state
        if old_head is not None:
            atoms.calc.head = old_head
        if hasattr(atoms.calc, "results"):
            atoms.calc.results = old_results

def _load_md_restart_frame(restart_file, reference_atoms=None):
    """Load the last MD frame and restore velocities for a restart."""
    if not restart_file:
        raise ValueError("MD restart requested but no restart_file or trajectory_file_md was provided.")

    restart_path = Path(restart_file)
    if not restart_path.exists() or restart_path.stat().st_size == 0:
        raise FileNotFoundError(f"MD restart file is missing or empty: {restart_file}")

    atoms = read(str(restart_path), index=-1)
    if reference_atoms is not None:
        if len(atoms) != len(reference_atoms):
            raise ValueError(
                f"MD restart frame has {len(atoms)} atoms, but initial_xyz has "
                f"{len(reference_atoms)} atoms."
            )
        if not atoms.cell.rank and reference_atoms.cell.rank:
            atoms.set_cell(reference_atoms.get_cell())
        if not np.any(atoms.get_pbc()) and np.any(reference_atoms.get_pbc()):
            atoms.set_pbc(reference_atoms.get_pbc())

    velocities = atoms.arrays.get("velocities")
    if velocities is None:
        velocities = atoms.get_velocities()
    if velocities is None:
        raise ValueError(
            f"MD restart file '{restart_file}' does not contain velocities. "
            "Restart requires a trajectory written by this MD driver."
        )

    atoms.set_velocities(np.asarray(velocities, dtype=float))
    step_offset = int(atoms.info.get("step", 0) or 0)
    return atoms, step_offset


def _set_md_initial_velocities_from_atoms(atoms, mode="auto"):
    """Restore velocities already present on the ASE Atoms object."""
    mode = str(mode or "auto").strip().lower()
    if mode in {"temperature", "maxwell", "maxwell_boltzmann"}:
        return False
    if mode not in {"auto", "file", "extxyz"}:
        raise ValueError(
            f"Unsupported md.initial_velocities '{mode}'. "
            "Use 'auto', 'file', or 'temperature'."
        )

    velocities = atoms.arrays.get("velocities")
    if velocities is None:
        velocities = atoms.get_velocities()

    if velocities is None:
        if mode in {"file", "extxyz"}:
            raise ValueError(
                "md.initial_velocities is set to 'file', but the input structure "
                "does not contain velocities."
            )
        return False

    velocities = np.asarray(velocities, dtype=float)
    if velocities.shape != (len(atoms), 3):
        raise ValueError(
            "Input velocities must have shape "
            f"({len(atoms)}, 3), got {velocities.shape}."
        )

    atoms.set_velocities(velocities)
    return True


def print_md_status(
    dyn,
    atoms,
    log_file,
    dt_fs,
    friction,
    config=None,
    step_offset=0,
):
    """
    Log one line with energies, timing, etc.  **Do not** write XYZ here –
    that now lives in a separate callback.
    """
    global last_call_time, cumulative_time

    now = time.time()
    step_time = now - last_call_time if last_call_time else 0.0
    last_call_time = now
    cumulative_time += step_time

    step = step_offset + dyn.get_number_of_steps()
    md_time = step * dt_fs
    e_pot_raw = atoms.get_potential_energy()
    e_pot = _get_logged_potential_energy(atoms, fallback=e_pot_raw)
    e_kin = atoms.get_kinetic_energy()
    e_tot = e_pot + e_kin
    temp_inst = e_kin / (1.5 * units.kB * len(atoms)) if len(atoms) else 0.0
    T_set = getattr(dyn, "temperature_K", np.nan)

    # Compute max force
    try:
        forces = atoms.get_forces()
        max_force = np.sqrt((forces**2).sum(axis=1).max()) if len(forces) > 0 else 0.0
    except Exception:
        max_force = np.nan

    # Compute simulated steps per second -> ns/day
    step_diff = step - getattr(print_md_status, "last_step", 0)
    print_md_status.last_step = step

    if step_time > 0 and step_diff > 0:
        speed = (step_diff * dt_fs * 0.0864) / step_time
    else:
        speed = 0.0

    # Helper to rename long head name
    def log_name(h):
        if h == "triplet_reconstructed":
            return "trip_delta"
        return h

    # Detect multi-head MACE run
    is_mace = False
    if config is not None:
        framework = config.get("model_framework", "").lower()
        if framework == "mace":
            is_mace = True

    has_multihead = False
    selected_head = None
    other_head = None
    available_heads = []

    if is_mace and atoms.calc is not None:
        if hasattr(atoms.calc, "models") and len(atoms.calc.models) > 0:
            available_heads = getattr(atoms.calc.models[0], "heads", [])
        if not available_heads and hasattr(atoms.calc, "available_heads"):
            available_heads = atoms.calc.available_heads
        
        if available_heads:
            available_heads = [h.strip() for h in available_heads if isinstance(h, str)]

        mace_head = config.get("mace_head", None)
        if isinstance(mace_head, str):
            mace_head = mace_head.strip()

        is_reconstructed = (mace_head == "triplet_reconstructed")
        is_multi_model = ("singlet" in available_heads and ("triplet" in available_heads or "delta" in available_heads))
        
        if is_reconstructed or is_multi_model:
            has_multihead = True
            selected_head = mace_head if mace_head else "singlet"

            if selected_head == "singlet":
                other_head = "triplet" if "triplet" in available_heads else "delta"
            elif selected_head == "triplet":
                other_head = "singlet"
            elif selected_head == "delta":
                other_head = "singlet"
            elif selected_head == "triplet_reconstructed":
                other_head = "singlet"
            else:
                other_head = "singlet" if "singlet" in available_heads else None

            if other_head == "delta":
                k_E = _load_scale_metadata(config)
                if k_E is not None:
                    other_head = "triplet_reconstructed"

    epot_other = np.nan
    if has_multihead and other_head:
        epot_other = _get_other_head_energy(atoms, other_head, config)

    # Compute delta_gap = Epot_singlet - Epot_triplet
    delta_gap = np.nan
    if has_multihead and other_head:
        e_singlet_val = np.nan
        e_triplet_val = np.nan
        
        # Check active/selected head
        if selected_head == "singlet":
            e_singlet_val = e_pot
        elif selected_head in ("triplet", "triplet_reconstructed"):
            e_triplet_val = e_pot
            
        # Check other head
        if other_head == "singlet":
            e_singlet_val = epot_other
        elif other_head in ("triplet", "triplet_reconstructed"):
            e_triplet_val = epot_other
            
        if not np.isnan(e_singlet_val) and not np.isnan(e_triplet_val):
            delta_gap = e_singlet_val - e_triplet_val

    if has_multihead and other_head:
        col1_name = f"Epot_{log_name(selected_head)}(eV)"
        col2_name = f"Epot_{log_name(other_head)}(eV)"
        
        val_epot_str = f"{e_pot:.6f}"
        val_ekin_str = f"{e_kin:.6f}"
        val_etot_str = f"{e_pot + e_kin:.6f}"
        val_maxforce_str = f"{max_force:.6f}"
        val_epot_other_str = f"{epot_other:.6f}"
        val_delta_gap_str = f"{delta_gap:.6f}"
        
        col1_w = max(11, len(col1_name), len(val_epot_str))
        ekin_w = max(11, len("Ekin(eV)"), len(val_ekin_str))
        etot_w = max(11, len("Etot(eV)"), len(val_etot_str))
        force_w = max(14, len("MaxForce(eV/A)"), len(val_maxforce_str))
        col2_w = max(11, len(col2_name), len(val_epot_other_str))
        gap_w = max(13, len("delta_gap(eV)"), len(val_delta_gap_str))

        header = (
            f"{'Step':>6} | {'MD_Time(fs)':>11} | {'T_inst(K)':>9} | {'T_set(K)':>8} | "
            f"{col1_name:>{col1_w}} | {'Ekin(eV)':>{ekin_w}} | {'Etot(eV)':>{etot_w}} | "
            f"{'MaxForce(eV/A)':>{force_w}} | {'dt(s)':>8} | {col2_name:>{col2_w}} | "
            f"{'delta_gap(eV)':>{gap_w}} | {'cum(s)':>9} | {'Speed(ns/day)':>13}"
        )
        fmt = (
            "{:6d} | {:11.2f} | {:9.2f} | {:8.2f} | "
            f"{{:{col1_w}.6f}} | {{:{ekin_w}.6f}} | {{:{etot_w}.6f}} | "
            f"{{:{force_w}.6f}} | {{:8.4f}} | {{:{col2_w}.6f}} | "
            f"{{:{gap_w}.6f}} | {{:9.4f}} | {{:13.4f}}"
        )
        values = (
            step, md_time, temp_inst, T_set,
            e_pot, e_kin, e_pot + e_kin,
            max_force, step_time, epot_other,
            delta_gap, cumulative_time, speed
        )
    else:
        col_name = f"Epot_{log_name(selected_head)}(eV)" if selected_head else "Epot(eV)"
        
        val_epot_str = f"{e_pot:.6f}"
        val_ekin_str = f"{e_kin:.6f}"
        val_etot_str = f"{e_pot + e_kin:.6f}"
        val_maxforce_str = f"{max_force:.6f}"
        
        col_w = max(11, len(col_name), len(val_epot_str))
        ekin_w = max(11, len("Ekin(eV)"), len(val_ekin_str))
        etot_w = max(11, len("Etot(eV)"), len(val_etot_str))
        force_w = max(14, len("MaxForce(eV/A)"), len(val_maxforce_str))

        header = (
            f"{'Step':>6} | {'MD_Time(fs)':>11} | {'T_inst(K)':>9} | {'T_set(K)':>8} | "
            f"{col_name:>{col_w}} | {'Ekin(eV)':>{ekin_w}} | {'Etot(eV)':>{etot_w}} | "
            f"{'MaxForce(eV/A)':>{force_w}} | {'dt(s)':>8} | {'cum(s)':>9} | {'Speed(ns/day)':>13}"
        )
        fmt = (
            "{:6d} | {:11.2f} | {:9.2f} | {:8.2f} | "
            f"{{:{col_w}.6f}} | {{:{ekin_w}.6f}} | {{:{etot_w}.6f}} | "
            f"{{:{force_w}.6f}} | {{:8.4f}} | {{:9.4f}} | {{:13.4f}}"
        )
        values = (
            step, md_time, temp_inst, T_set,
            e_pot, e_kin, e_pot + e_kin,
            max_force, step_time, cumulative_time, speed
        )

    _log_status_line(
        log_file,
        header,
        fmt,
        values,
    )




from threading import Thread
from queue import Queue

class MDWriterThread(Thread):
    """
    Background worker thread to write trajectory frames and status logs asynchronously,
    preventing disk I/O operations from blocking the main simulation integration loop.
    """
    def __init__(self, traj_file, log_file, q):
        super().__init__(daemon=True)
        self.traj_file = traj_file
        self.log_file = log_file
        self.q = q

    def run(self):
        from contextlib import ExitStack
        from ase import Atoms
        from ase.io import write
        
        with ExitStack() as stack:
            f_out = None
            f_log = None
            
            if self.traj_file:
                try:
                    f_out = stack.enter_context(open(self.traj_file, "a"))
                except IOError as exc:
                    print(f"Warning: Background writer failed to open trajectory file {self.traj_file}: {exc}")
                    
            if self.log_file:
                try:
                    f_log = stack.enter_context(open(self.log_file, "a"))
                except IOError as exc:
                    print(f"Warning: Background writer failed to open log file {self.log_file}: {exc}")
                    
            # Track if log file header needs to be written
            fresh_log = False
            if self.log_file:
                try:
                    fresh_log = not Path(self.log_file).exists() or Path(self.log_file).stat().st_size == 0
                except Exception:
                    fresh_log = True

            while True:
                try:
                    msg_type, msg_data = self.q.get()
                except Exception:
                    break
                
                if msg_type == "stop":
                    self.q.task_done()
                    break
                    
                elif msg_type == "traj":
                    if f_out:
                        symbols, positions, velocities, forces, info, cell, pbc = msg_data
                        frame = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
                        if velocities is not None:
                            frame.set_array("velocities", velocities)
                        if forces is not None:
                            frame.set_array("forces", forces)
                        frame.info.update(info)
                        
                        try:
                            write(f_out, frame, format="extxyz")
                            f_out.flush()
                        except Exception as exc:
                            print(f"Warning: Background writer failed to write traj frame: {exc}")
                            
                elif msg_type == "log":
                    header, fmt, values = msg_data
                    line = fmt.format(*values)
                    
                    if f_log:
                        try:
                            if fresh_log:
                                f_log.write(header + "\n")
                                fresh_log = False
                            f_log.write(line + "\n")
                            f_log.flush()
                        except Exception as exc:
                            print(f"Warning: Background writer failed to write status line: {exc}")
                    else:
                        print(header)
                        print(line)
                        
                self.q.task_done()


def run_md(atoms, model_obj, device, config, neighbor_list=None):
    """Top-level MD driver with tidy, non-overlapping callbacks."""

    md            = config["md"]
    dt_fs         = md.get("timestep_fs",      2.0)
    nsteps        = md.get("steps",            5000)
    log_int       = md.get("log_interval",     5)
    xyz_int       = md.get("xyz_print_interval", 50)
    T0            = md.get("temperature_K",    300.0)
    traj_file     = md.get("trajectory_file_md")
    log_file      = md.get("log_file")
    framework     = config.get("model_framework", "schnetpack").lower()
    cutoff        = config.get("cutoff", md.get("cutoff"))
    restart       = bool(md.get("restart", False))
    restart_file  = md.get("restart_file", traj_file)
    step_offset   = 0

    if restart:
        atoms, step_offset = _load_md_restart_frame(restart_file, atoms)
        print(
            f"Restarting MD from {restart_file} "
            f"at saved step {step_offset}; running {nsteps} additional steps."
        )

    # -----------------------------------------------------------------
    #  calculator + starting velocities (preventing zero kinetic energy)
    # -----------------------------------------------------------------
    calc = get_ase_calculator(model_obj, config, device, neighbor_list)
    profile_mace = bool(md.get("profile_mace", framework == "mace"))
    profile_interval = int(md.get("profile_interval", log_int if log_int else 50))
    profile_sync_cuda = bool(md.get("profile_sync_cuda", False))
    profile_file = md.get("profile_file")
    profile_print = bool(md.get("profile_print", profile_file is None))
    cleanup_interval = int(md.get("clear_cuda_cache_interval", 0) or 0)
    remove_translation = bool(md.get("remove_translation", False))
    remove_rotation = bool(md.get("remove_rotation", False))
    remove_drift_interval = int(md.get("remove_drift_interval", 100) or 0)

    if framework == "mace" and profile_mace:
        calc = _enable_calculator_profiling(
            calc,
            label="mace",
            sync_cuda=profile_sync_cuda,
        )

    atoms.calc = calc

    # -----------------------------------------------------------------
    #  choose ensemble and integrator/thermostat
    # -----------------------------------------------------------------
    ensemble = str(md.get("ensemble", "NVT")).strip().upper()
    thermostat = md.get("thermostat", "langevin").lower()
    if "thermostat" not in md and "use_langevin" in md:
        thermostat = "langevin" if md.get("use_langevin") else "verlet"
    if ensemble == "NVE":
        thermostat = "verlet"
    elif ensemble != "NVT":
        raise ValueError(f"Unsupported MD ensemble '{ensemble}'. Supported ensembles are NVE and NVT.")

    heating_steps = md.get("heating_steps", 0)
    T_start = md.get("heating_T_start", 10.0)
    initial_velocity_mode = md.get("initial_velocities", "auto")
    loaded_initial_velocities = False

    if restart:
        print("MD restart: using velocities from restart frame; skipping velocity initialization and heating ramp.")
        heating_steps = 0
        loaded_initial_velocities = True
    else:
        loaded_initial_velocities = _set_md_initial_velocities_from_atoms(
            atoms,
            mode=initial_velocity_mode,
        )

    if loaded_initial_velocities:
        if not restart:
            print("Using velocities from input structure; temperature_K will not reinitialize velocities.")
        if heating_steps > 0 and ensemble == "NVE":
            print("NVE ensemble selected: disabling heating ramp because no thermostat is active.")
            heating_steps = 0
    elif heating_steps > 0 and ensemble == "NVE":
        print(
            "NVE ensemble selected: initializing velocities at target temperature "
            f"{T0} K and disabling heating ramp."
        )
        heating_steps = 0
        MaxwellBoltzmannDistribution(atoms, temperature_K=T0)
    elif heating_steps > 0:
        print(f"Heating initialized: starting at {T_start} K, ramping to {T0} K over {heating_steps} steps.")
        MaxwellBoltzmannDistribution(atoms, temperature_K=T_start)
    else:
        print(f"No heating ramp: starting at target temperature {T0} K.")
        MaxwellBoltzmannDistribution(atoms, temperature_K=T0)

    if thermostat in {"bussi", "csvr"}:
        from ase.md.bussi import Bussi
        taut_fs = md.get("taut_fs", 100.0)
        dyn = Bussi(
            atoms,
            timestep      = dt_fs * units.fs,
            temperature_K = T0,
            taut          = taut_fs * units.fs,
        )
        dyn.temperature_K = T0
        gamma_fs = 0.0  # not Langevin, no friction logging
        thermostat_desc = f"Bussi/CSVR τ = {taut_fs} fs"
    elif thermostat == "langevin":
        gamma_fs = md.get("friction_coefficient", 0.01)
        dyn = Langevin(
            atoms,
            timestep      = dt_fs * units.fs,
            temperature_K = T0,
            friction      = gamma_fs,
        )
        dyn.temperature_K = T0
        thermostat_desc = f"Langevin γ = {gamma_fs}"
    else:
        from ase.md.verlet import VelocityVerlet
        dyn = VelocityVerlet(atoms, timestep = dt_fs * units.fs)
        dyn.temperature_K = np.nan
        gamma_fs = 0.0
        thermostat_desc = "NVE VelocityVerlet" if ensemble == "NVE" else "VelocityVerlet"

    # -----------------------------------------------------------------
    #  temperature heating ramp callback
    # -----------------------------------------------------------------
    if heating_steps > 0:
        def ramp_callback():
            step = dyn.get_number_of_steps()
            if step <= heating_steps:
                current_T = T_start + (T0 - T_start) * (step / heating_steps)
            else:
                current_T = T0

            # Set temperature on thermostat
            if hasattr(dyn, "set_temperature"):
                dyn.set_temperature(temperature_K=current_T)
            elif hasattr(dyn, "temp"):
                dyn.temp = current_T * units.kB
                if hasattr(dyn, "ndof"):
                    dyn.target_kinetic_energy = 0.5 * dyn.temp * dyn.ndof
            
            dyn.temperature_K = current_T

        dyn.attach(ramp_callback, interval=1)
        # Execute once at step 0 to ensure initialization starts at T_start
        ramp_callback()

    if (remove_translation or remove_rotation) and remove_drift_interval > 0:
        def remove_drift_callback():
            if remove_translation:
                Stationary(atoms, preserve_temperature=True)
            if remove_rotation:
                ZeroRotation(atoms, preserve_temperature=True)

        remove_drift_callback()
        dyn.attach(remove_drift_callback, interval=remove_drift_interval)
        active = []
        if remove_translation:
            active.append("translation")
        if remove_rotation:
            active.append("rotation")
        print(
            f"MD drift removal enabled: removing {' and '.join(active)} "
            f"every {remove_drift_interval} steps."
        )

    # -----------------------------------------------------------------
    #  callbacks & run!
    # -----------------------------------------------------------------
    from queue import Queue
    import gc

    if cleanup_interval > 0:
        def periodic_cleanup():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        dyn.attach(periodic_cleanup, interval=cleanup_interval)
        print(
            f"MD cleanup enabled: clearing Python/CUDA caches every "
            f"{cleanup_interval} steps."
        )

    # Start background writer thread if writing/logging is enabled
    use_writer = (traj_file and xyz_int > 0) or log_file
    if use_writer:
        write_queue = Queue()
        writer_thread = MDWriterThread(traj_file, log_file, write_queue)
        writer_thread.start()

        if framework == "mace" and profile_mace and profile_interval > 0:
            dyn.attach(
                lambda: _print_calculator_profile(
                    calc,
                    dyn,
                    atoms,
                    write_queue,
                    cutoff=cutoff,
                    profile_file=profile_file,
                    print_to_screen=profile_print,
                ),
                interval=profile_interval,
            )

        # Attach log status callback (using write_queue for async logging)
        if log_file:
            dyn.attach(
                lambda: print_md_status(
                    dyn,
                    atoms,
                    write_queue,
                    dt_fs,
                    gamma_fs,
                    config=config,
                    step_offset=step_offset,
                ),
                interval=log_int,
            )

        # Attach trajectory print callback
        if traj_file and xyz_int > 0:
            def async_write_xyz_frame():
                step = step_offset + dyn.get_number_of_steps()
                md_time = step * dt_fs
                T_set = getattr(dyn, "temperature_K", np.nan)
                
                velocities = atoms.get_velocities()
                if velocities is not None:
                    velocities = velocities.copy()
                    
                forces = atoms.get_forces()
                if forces is not None:
                    forces = forces.copy()
                    
                info = {
                    "step": int(step),
                    "time_fs": float(md_time),
                    "temperature_set_K": float(T_set) if np.isfinite(T_set) else np.nan,
                    "friction_fs_inv": float(gamma_fs),
                    "energy": float(atoms.get_potential_energy()),
                }
                
                write_queue.put((
                    "traj", 
                    (
                        atoms.get_chemical_symbols(),
                        atoms.get_positions().copy(),
                        velocities,
                        forces,
                        info,
                        atoms.get_cell().array.copy(),
                        atoms.get_pbc().copy(),
                    )
                ))

            dyn.attach(async_write_xyz_frame, interval=xyz_int)

        try:
            print(f"Running MD: {nsteps} steps · Δt = {dt_fs} fs · ensemble = {ensemble} · thermostat = {thermostat_desc}")
            dyn.run(nsteps)
        finally:
            # Signal the background writer thread to flush and stop
            write_queue.put(("stop", None))
            writer_thread.join()
    else:
        if framework == "mace" and profile_mace and profile_interval > 0:
            dyn.attach(
                lambda: _print_calculator_profile(
                    calc,
                    dyn,
                    atoms,
                    cutoff=cutoff,
                    profile_file=profile_file,
                    print_to_screen=profile_print,
                ),
                interval=profile_interval,
            )

        # No files to write or log, run standard
        print(f"Running MD: {nsteps} steps · Δt = {dt_fs} fs · ensemble = {ensemble} · thermostat = {thermostat_desc}")
        dyn.run(nsteps)

    print("MD finished.")



def run_vibrational_analysis(atoms, model_obj, device, config, neighbor_list=None):
    """
    Runs vibrational analysis, including tight geometry optimization and frequency calculation.

    Parameters:
      atoms (ase.Atoms): The atomic structure.
      model_obj: Pre-trained PyTorch model.
      device: Torch device.
      neighbor_list: Neighbor list object.
      config (dict): Configuration for vibrational analysis.

    Returns:
      np.ndarray: Array of vibrational frequencies in cm^-1, or None if failed.
    """
    _reset_timers()
    vib_config = config.get("vib", {})
    vib_opt_fmax = vib_config.get("vib_opt_fmax", 0.001)
    vib_opt_steps = vib_config.get("vib_opt_steps", 1000)
    trajectory_file_vib = vib_config.get("trajectory_file_vib", "vib_trajectory.xyz")
    log_file_vib = vib_config.get("log_file_vib", "vib_opt.log")
    vib_output_file = vib_config.get("vib_output_file", "vibrational_frequencies.txt")
    vdos_plot_file = vib_config.get("vdos_plot_file", "vdos_plot.png")
    delta = vib_config.get("delta", 0.01)
    vib_cache_name = vib_config.get("cache_name", "vib")
    normal_modes_file = vib_config.get(
        "normal_modes_file",
        vib_output_file.replace(".txt", "_normal_modes.npz"),
    )

    print("Setting up calculator for Vibrational Analysis...")
    # Correct instantiation
    calc = get_ase_calculator(model_obj, config, device, neighbor_list)
    atoms.calc = calc

    print(f"Running tight Geometry Optimization for Vibrations (fmax={vib_opt_fmax}, steps={vib_opt_steps})...")
    
    geo_config = config.get("geo_opt", {})
    opt_type = str(vib_config.get("optimizer", geo_config.get("optimizer", "bfgs"))).lower().strip()
    if opt_type == "lbfgs":
        print("Using LBFGS optimizer (forces-only) for pre-vibrational optimization.")
        optimizer = LBFGS(atoms, logfile=None, maxstep=0.02)
    elif opt_type == "fire":
        print("Using FIRE optimizer (forces-only) for pre-vibrational optimization.")
        optimizer = FIRE(atoms, logfile=None, maxstep=0.02)
    else:
        print("Using BFGSLineSearch optimizer (energy + forces) for pre-vibrational optimization.")
        optimizer = BFGSLineSearch(atoms, logfile=None, maxstep=0.02)

    # Attach logger using the vib log file
    optimizer.attach(
        lambda opt=optimizer: log_vib_opt_status(opt, atoms, log_file_vib, trajectory_file_vib, config=config),
        interval=1
    )

    try:
        optimizer.run(fmax=vib_opt_fmax, steps=vib_opt_steps)
    except Exception as e:
        print(f"Error during tight geometry optimization for vibrations: {e}")
        import traceback
        traceback.print_exc()
        print("Cannot proceed with vibration calculation.")
        return None

    print("Tight Geometry Optimization Finished.")

    print(f"Calculating Vibrations (delta={delta} Ang, cache='{vib_cache_name}')...")
    try:
        vib = Vibrations(atoms, delta=delta, name=vib_cache_name)
        vib.run()
        print("Vibrations calculation finished.")
    except Exception as e:
        print(f"Error during vibrations calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

    # === Process Vibrational Modes ===
    # ASE Vibrations.get_frequencies() already returns cm^-1.
    ase_frequencies_cm = vib.get_frequencies()
    frequencies_cm = []
    imag_modes_count = 0

    for freq in ase_frequencies_cm:
        if isinstance(freq, complex):
            # Check imaginary part magnitude - threshold might need adjustment
            if abs(freq.imag) > 1e-4:
                # Mark imaginary modes with negative sign
                frequencies_cm.append(-abs(freq.imag))
                imag_modes_count += 1
            else:
                # Treat as real if imaginary part is negligible
                frequencies_cm.append(freq.real)
        else:
            # Handle real frequencies directly
            frequencies_cm.append(freq)

    frequencies_cm = np.array(frequencies_cm)
    print(f"Found {imag_modes_count} imaginary modes (marked negative).")

    # === Save Frequencies ===
    try:
        with open(vib_output_file, "w") as f:
            f.write("# Vibrational Frequencies (cm^-1)\n")
            f.write("# (Imaginary modes denoted by negative values)\n")
            for i, freq in enumerate(frequencies_cm):
                f.write(f"Mode {i + 1}: {freq:.4f}\n")
        print(f"Vibrational frequencies saved to {vib_output_file}")
    except IOError as e:
        print(f"Warning: Failed to write frequencies file: {e}")

    # === Save Normal Modes ===
    try:
        vib_data = vib.get_vibrations()
        modes = np.asarray(vib_data.get_modes(all_atoms=True), dtype=np.float64)
        mode_norm2_by_atom = np.sum(modes**2, axis=2)
        energies_eV = np.asarray(vib_data.get_energies(), dtype=np.complex128)
        raw_frequencies_cm = np.asarray(ase_frequencies_cm, dtype=np.complex128)
        field_names = np.asarray([
            "frequencies_cm",
            "raw_frequencies_cm",
            "energies_eV",
            "modes_cartesian",
            "mode_norm2_by_atom",
            "symbols",
            "masses",
            "positions",
            "atomic_numbers",
            "vib_indices",
            "delta_angstrom",
        ])
        field_descriptions = np.asarray([
            "Signed frequencies in cm^-1; imaginary modes are stored as negative real values.",
            "Raw ASE frequencies in cm^-1, preserving complex values for imaginary modes.",
            "Raw ASE vibrational energies in eV, preserving complex values.",
            "Cartesian normal modes with shape (n_modes, n_atoms, 3).",
            "Per-mode per-atom squared displacement weights with shape (n_modes, n_atoms).",
            "Chemical symbols for all atoms in the optimized structure.",
            "Atomic masses in amu for all atoms in the optimized structure.",
            "Optimized Cartesian positions in Angstrom.",
            "Atomic numbers for all atoms in the optimized structure.",
            "Atom indices included in the ASE finite-difference Hessian.",
            "Finite-difference displacement used by ASE Vibrations, in Angstrom.",
        ])

        np.savez_compressed(
            normal_modes_file,
            schema=np.asarray("orchestr_ai.vibrational_modes.v1"),
            field_names=field_names,
            field_descriptions=field_descriptions,
            frequencies_cm=frequencies_cm.astype(np.float64),
            raw_frequencies_cm=raw_frequencies_cm,
            energies_eV=energies_eV,
            modes_cartesian=modes,
            mode_norm2_by_atom=mode_norm2_by_atom,
            symbols=np.asarray(atoms.get_chemical_symbols()),
            masses=np.asarray(atoms.get_masses(), dtype=np.float64),
            positions=np.asarray(atoms.get_positions(), dtype=np.float64),
            atomic_numbers=np.asarray(atoms.get_atomic_numbers(), dtype=np.int64),
            vib_indices=np.asarray(vib.indices, dtype=np.int64),
            delta_angstrom=float(delta),
        )
        print(f"Normal modes saved to {normal_modes_file}")
    except Exception as e:
        print(f"Warning: Failed to write normal modes file: {e}")
        traceback.print_exc()

    # === Save Molden File (Use ASE's built-in method if possible) ===
    molden_file = vib_output_file.replace(".txt", ".molden")
    try:
        # ASE's write_molden uses atomic units (Bohr) by default
        vib.write_molden(molden_file)
        print(f"Molden file saved to {molden_file}")
    except AttributeError:
         print(f"Warning: Current ASE version might not support vib.write_molden(). Skipping Molden file.")
    except Exception as e:
        print(f"Warning: Failed to write Molden file: {e}")
        traceback.print_exc() # Print traceback for Molden write errors


    # === Plot VDOS ===
    try:
        # Filter out imaginary frequencies for VDOS plot
        real_frequencies_cm = frequencies_cm[frequencies_cm >= 0]
        if len(real_frequencies_cm) == 0:
            print("Warning: No real frequencies found for VDOS plot.")
        else:
            # Determine frequency range dynamically
            freq_min_plot = 0
            freq_max_plot = max(real_frequencies_cm) * 1.1 if len(real_frequencies_cm) > 0 else 100
            freq_range = np.linspace(freq_min_plot, freq_max_plot, 1000)
            vdos = np.zeros_like(freq_range)
            # Get broadening from config or use default
            sigma_vdos = vib_config.get("vdos_broadening_cm", 10.0)

            # Gaussian broadening
            for freq in real_frequencies_cm:
                vdos += np.exp(-((freq_range - freq)**2) / (2 * sigma_vdos**2))

            # Normalize VDOS if it's not all zero
            max_vdos = np.max(vdos)
            if max_vdos > 1e-9:
                vdos /= max_vdos
            else:
                print("Warning: VDOS intensity is near zero. Plot may be empty.")


            plt.figure(figsize=(10, 6))
            plt.plot(freq_range, vdos, 'b-', label=f'VDOS ($\\sigma={sigma_vdos:.1f}$ cm$^{{-1}}$)')
            plt.xlabel('Frequency (cm$^{-1}$)')
            plt.ylabel('Density of States (Normalized)')
            plt.title('Vibrational Density of States')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.xlim(left=freq_min_plot)
            plt.ylim(bottom=0) # Start y-axis at 0
            plt.legend()
            plt.savefig(vdos_plot_file)
            plt.close()
            print(f"VDOS plot saved to {vdos_plot_file}")

    except Exception as e:
        print(f"Warning: Failed to generate VDOS plot: {e}")
        traceback.print_exc() # Print traceback for VDOS errors

    return frequencies_cm
