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

from ase import units
from ase.io import write
from ase.io.extxyz import write_extxyz
from ase.md import VelocityVerlet, Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGSLineSearch
from ase.vibrations import Vibrations

# === Local module imports ===
# Use the calculator from the calculator module
from mlff_qd.postprocessing.calculator import MyTorchCalculator, ANGSTROM_TO_BOHR

# --- Global Timing Variables ---
last_call_time = None
cumulative_time = 0.0


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

def log_geo_opt_status(optimizer, atoms, log_file, trajectory_file):
    """
    Logs geometry optimization status to a log file.

    Parameters:
      optimizer: Optimizer object.
      atoms (ase.Atoms): The atomic structure.
      log_file (str): Path to the log file.
      trajectory_file (str): File name for the geometry optimization trajectory.
    """
    global last_call_time, cumulative_time
    now = time.time()
    step_total_time = now - last_call_time if last_call_time is not None else 0.0
    last_call_time = now
    cumulative_time += step_total_time

    step = optimizer.get_number_of_steps()
    e_pot = atoms.get_potential_energy()
    forces = atoms.get_forces(apply_constraint=False) # Get forces after energy
    max_force = np.sqrt((forces**2).sum(axis=1).max()) if len(forces) > 0 else 0.0

    # Access results safely from the calculator
    calc_results = {}
    if hasattr(atoms, 'calc') and atoms.calc is not None and hasattr(atoms.calc, 'results'):
         calc_results = atoms.calc.results

    E_ml_only = calc_results.get("E_ml_avg", np.nan) # Use E_ml_avg which is ML energy
    E_coul = calc_results.get("coul_fn_energy", np.nan) # Use coul_fn_energy
    ml_time = calc_results.get("ml_time", 0.0)
    coul_fn_time = calc_results.get("coul_fn_time", 0.0)

    header = (
        f"{'Step':>5s} | {'Epot(eV)':>14s} | {'E_ML_only(eV)':>14s} | {'E_Coul(eV)':>12s} | "
        f"{'MLtime(s)':>10s} | {'CoulFn(s)':>10s} | {'StepTime(s)':>12s} | "
        f"{'CumTime(s)':>12s} | {'MaxForce(eV/A)':>14s}"
    )
    values_format = (
        "{:5d} | {:14.6f} | {:14.6f} | {:12.6f} | {:10.4f} | "
        "{:10.4f} | {:12.4f} | {:12.4f} | {:14.6f}"
    )
    values = (
        step, e_pot, E_ml_only, E_coul, ml_time, coul_fn_time,
        step_total_time, cumulative_time, max_force
    )
    _log_status_line(log_file, header, values_format, values)

    if trajectory_file:
        try:
            # Write using ASE's write function for extxyz format
            write(trajectory_file, atoms, append=True, format='extxyz')
        except IOError as e:
            print(f"Warning: Failed to write trajectory frame {step}: {e}")


def log_vib_opt_status(optimizer, atoms, log_file, trajectory_file):
    """
    Logs vibrational optimization status by reusing the geometry optimization logger.
    """
    log_geo_opt_status(optimizer, atoms, log_file, trajectory_file)


# === Simulation Drivers ===

def run_geo_opt(atoms, model_obj, device, neighbor_list, config):
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
    calc = MyTorchCalculator(model_obj, device, neighbor_list, config)
    atoms.set_calculator(calc)

    print(f"Running Geometry Optimization (fmax={geo_opt_fmax}, steps={geo_opt_steps})...")
    # Use atoms directly, no need for Optimizable wrapper unless constraints change
    optimizer = BFGSLineSearch(atoms, logfile=None, maxstep=0.04)
    # Pass optimizer itself to the logger function
    optimizer.attach(
        lambda opt=optimizer: log_geo_opt_status(opt, atoms, log_file, trajectory_file),
        interval=1
    )

    try:
        optimizer.run(fmax=geo_opt_fmax, steps=geo_opt_steps)
    except Exception as e:
        print(f"Error during geometry optimization: {e}")
        import traceback
        traceback.print_exc()

    print("Geometry Optimization Finished.")

def print_md_status(
    dyn,
    atoms,
    log_file,
    dt_fs,
    xyz_print_interval,
    trajectory_file_with_forces,
    friction,
    temperature_K
):
    global last_call_time, cumulative_time

    now = time.time()
    step_time = now - last_call_time if last_call_time is not None else 0.0
    last_call_time = now
    cumulative_time += step_time

    step = dyn.get_number_of_steps()
    md_time = step * dt_fs
    e_pot = atoms.get_potential_energy()
    e_kin = atoms.get_kinetic_energy()
    forces = atoms.get_forces()
    temp_inst = (
        e_kin / (1.5 * units.kB * len(atoms))
        if len(atoms) > 0 else 0.0
    )

    # Use stored temperature_K or dyn.temperature_K if available
    T_set = getattr(dyn, 'temperature_K', temperature_K)

    if friction == 0.0 and isinstance(dyn, Langevin):
        print(f"Warning: Friction coefficient is 0.0 at step {step}. Expected non-zero for Langevin.")

    calc_results = {}
    if hasattr(atoms, "calc") and atoms.calc and hasattr(atoms.calc, "results"):
        calc_results = atoms.calc.results

    E_ml_only = calc_results.get("E_ml_avg", np.nan)
    E_coul = calc_results.get("coul_fn_energy", np.nan)
    ml_time = calc_results.get("ml_time", 0.0)
    coul_fn_time = calc_results.get("coul_fn_time", 0.0)

    header = (
        f"{'Step':>5s} | {'MD_Time(fs)':>12s} | {'T_inst(K)':>9s} | {'T_set(K)':>9s} | "
        f"{'Friction(fs⁻¹)':>13s} | {'Epot(eV)':>10s} | {'Ekin(eV)':>10s} | "
        f"{'E_ML(eV)':>10s} | {'E_Coul(eV)':>11s} | {'ML_time(s)':>10s} | "
        f"{'Coul_time(s)':>12s} | {'StepTime(s)':>12s} | {'CumTime(s)':>12s}"
    )
    values_fmt = (
        "{:5d} | {:12.2f} | {:9.2f} | {:9.2f} | {:13.6f} | "
        "{:10.6f} | {:10.6f} | {:10.6f} | {:11.6f} | "
        "{:10.4f} | {:12.4f} | {:12.4f} | {:12.4f}"
    )
    values = (
        step, md_time, temp_inst, T_set, friction,
        e_pot, e_kin, E_ml_only, E_coul,
        ml_time, coul_fn_time, step_time, cumulative_time
    )
    _log_status_line(log_file, header, values_fmt, values)

    if step % xyz_print_interval == 0 and trajectory_file_with_forces:
        try:
            with open(trajectory_file_with_forces, "a") as f:
                f.write(f"{len(atoms)}\n")
                f.write(
                    f"Step = {step}, MD time = {md_time:.2f} fs, "
                    f"T_set = {T_set:.1f} K, Friction = {friction:.6f} fs⁻¹, "
                    f"Epot (eV) = {e_pot:.8f}\n"
                )
                symbols = atoms.get_chemical_symbols()
                positions = atoms.get_positions()
                for i, sym in enumerate(symbols):
                    x, y, z = positions[i]
                    fx, fy, fz = forces[i]
                    f.write(
                        f"{sym:<2s} "
                        f"{x:15.8f} {y:15.8f} {z:15.8f} "
                        f"{fx:15.8f} {fy:15.8f} {fz:15.8f}\n"
                    )
        except IOError as e:
            print(f"Warning: Failed to write trajectory frame {step}: {e}")

def run_md(atoms, model_obj, device, neighbor_list, config):
    md = config["md"]
    dt_fs = md.get("timestep_fs", 2.0)
    nsteps = md.get("steps", 5000)
    print_int = md.get("xyz_print_interval", 50)
    base_T = md.get("temperature_K", 300.0)
    use_ramp = md.get("use_ramp", False)
    temps = md.get("ramp_temps", [base_T])
    ramp_steps = md.get("ramp_steps", 0)
    plat_steps = md.get("plateau_steps", 0)
    cycle = temps + temps[-2:0:-1] if use_ramp and len(temps) > 1 else [base_T]
    traj_file = md.get("trajectory_file_md", None)
    log_file = md.get("log_file", None)

    _reset_timers()
    print("Setting up calculator …")
    atoms.calc = MyTorchCalculator(model_obj, device, neighbor_list, config)
    MaxwellBoltzmannDistribution(atoms, temperature_K=cycle[0] / 8)

    print(f"MD: {nsteps} steps, dt={dt_fs} fs, start T={cycle[0]} K")
    global stored_friction, stored_temperature_K
    if md.get("use_langevin", True):
        gamma_fs = md.get("friction_coefficient", 0.01)
        dyn = Langevin(
            atoms,
            timestep=dt_fs * units.fs,
            temperature_K=cycle[0],
            friction=gamma_fs
        )
        stored_friction = gamma_fs
        stored_temperature_K = cycle[0]
        print(f"Langevin friction set to: {stored_friction:.6f} fs⁻¹")
        print(f"Langevin temperature set to: {stored_temperature_K:.2f} K")
        attributes = dir(dyn)
        friction_attrs = [attr for attr in attributes if 'friction' in attr.lower()]
        print(f"Langevin attributes with 'friction': {friction_attrs}")
        for attr in friction_attrs:
            try:
                value = getattr(dyn, attr)
                print(f"Value of {attr}: {value}")
            except AttributeError:
                print(f"Could not access {attr}")
    else:
        from ase.md.verlet import VelocityVerlet
        dyn = VelocityVerlet(atoms, timestep=dt_fs * units.fs)
        stored_friction = 0.0
        stored_temperature_K = 0.0
        print("Using VelocityVerlet, friction set to: 0.000000 fs⁻¹")

    if len(cycle) > 1 and use_ramp:
        seg_len = ramp_steps + plat_steps
        n_seg = len(cycle) - 1

        def _ramp_cb():
            i = dyn.get_number_of_steps()
            seg = (i // seg_len) % n_seg
            pos = i % seg_len
            T0, T1 = cycle[seg], cycle[seg + 1]
            if pos < ramp_steps and ramp_steps > 0:
                frac = pos / ramp_steps
                Tn = T0 + frac * (T1 - T0)
            else:
                Tn = T1
            dyn.temperature_K = Tn
            print(f"Step {i}: Setting temperature to {Tn:.2f} K")
            global stored_temperature_K
            stored_temperature_K = Tn

        dyn.attach(_ramp_cb, interval=1)

    dyn.attach(lambda: print_md_status(
        dyn, atoms, log_file, dt_fs, print_int, traj_file, stored_friction, stored_temperature_K),
        interval=5)

    try:
        dyn.run(nsteps)
    except Exception as e:
        print("MD error:", e)
        raise
    print("MD finished.")

def run_vibrational_analysis(atoms, model_obj, device, neighbor_list, config):
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

    print("Setting up calculator for Vibrational Analysis...")
    # Correct instantiation
    calc = MyTorchCalculator(model_obj, device, neighbor_list, config)
    atoms.set_calculator(calc)

    print(f"Running tight Geometry Optimization for Vibrations (fmax={vib_opt_fmax}, steps={vib_opt_steps})...")
    optimizer = BFGSLineSearch(atoms, logfile=None, maxstep=0.02) # Smaller maxstep for tighter opt
    # Attach logger using the vib log file
    optimizer.attach(
        lambda opt=optimizer: log_vib_opt_status(opt, atoms, log_file_vib, trajectory_file_vib),
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

    print(f"Calculating Vibrations (delta={delta} Ang)...")
    try:
        vib = Vibrations(atoms, delta=delta)
        vib.run()
        print("Vibrations calculation finished.")
    except Exception as e:
        print(f"Error during vibrations calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

    # === Process Vibrational Modes ===
    # Standard ASE conversion factor for Vibrations frequencies (which are in meV)
    meV_to_cm1 = units.invcm # Should be ~8.06554
    frequencies_meV = vib.get_frequencies()
    frequencies_cm = []
    imag_modes_count = 0

    for f_meV in frequencies_meV:
        if isinstance(f_meV, complex):
            # Check imaginary part magnitude - threshold might need adjustment
            if abs(f_meV.imag) > 1e-4:
                # Mark imaginary modes with negative sign
                frequencies_cm.append(-abs(f_meV.imag * meV_to_cm1))
                imag_modes_count += 1
            else:
                # Treat as real if imaginary part is negligible
                frequencies_cm.append(f_meV.real * meV_to_cm1)
        else:
            # Handle real frequencies directly
            frequencies_cm.append(f_meV * meV_to_cm1)

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

