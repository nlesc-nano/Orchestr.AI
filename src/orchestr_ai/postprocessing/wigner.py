from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import units
from ase.io import write
from ase.md.velocitydistribution import Stationary, ZeroRotation


def _coth(x):
    if x > 50.0:
        return 1.0
    return 1.0 / np.tanh(x)


def _mass_weighted_modes_from_ase(vib_data):
    atoms = vib_data.get_atoms()
    indices = np.asarray(vib_data.get_indices(), dtype=np.int64)
    active_atoms = atoms[indices]
    masses = np.asarray(active_atoms.get_masses(), dtype=np.float64)
    if np.any(masses <= 0.0):
        raise ValueError("Wigner sampling requires strictly positive atomic masses.")

    mass_weights = np.repeat(masses**-0.5, 3)
    hessian = np.asarray(vib_data.get_hessian_2d(), dtype=np.float64)
    dynmat = mass_weights[:, None] * hessian * mass_weights[None, :]
    omega2, vectors = np.linalg.eigh(dynmat)

    cart_modes_active = vectors.T.reshape(len(indices) * 3, len(indices), 3)
    cart_modes_active = cart_modes_active * masses[np.newaxis, :, np.newaxis]**-0.5

    cart_modes = np.zeros((len(indices) * 3, len(atoms), 3), dtype=np.float64)
    cart_modes[:, indices, :] = cart_modes_active

    unit_conversion = units._hbar * units.m / np.sqrt(units._e * units._amu)
    energies_eV = unit_conversion * omega2.astype(np.complex128) ** 0.5
    frequencies_cm = energies_eV / units.invcm

    return omega2, frequencies_cm, cart_modes, indices


def _mode_selection(frequencies_cm, cutoff_cm, skip_imaginary):
    real_freq = np.real(frequencies_cm)
    imag_freq = np.abs(np.imag(frequencies_cm))
    has_imaginary = imag_freq > 1.0e-8

    if skip_imaginary:
        valid = ~has_imaginary
    else:
        valid = np.ones(len(frequencies_cm), dtype=bool)

    valid &= real_freq > float(cutoff_cm)
    skipped_imaginary = int(np.count_nonzero(has_imaginary))
    skipped_low = int(np.count_nonzero((~has_imaginary) & (real_freq <= float(cutoff_cm))))
    return np.where(valid)[0], skipped_imaginary, skipped_low


def _sample_mode_amplitudes(frequencies_cm, mode_indices, temperature_K, rng):
    q = np.zeros(len(frequencies_cm), dtype=np.float64)
    qdot_fs = np.zeros(len(frequencies_cm), dtype=np.float64)

    temperature_K = float(temperature_K)
    for mode_index in mode_indices:
        nu_cm = float(np.real(frequencies_cm[mode_index]))
        omega = 2.0 * np.pi * units._c * nu_cm * 100.0
        if omega <= 0.0:
            continue

        if temperature_K <= 0.0:
            thermal_factor = 1.0
        else:
            x = units._hbar * omega / (2.0 * units._k * temperature_K)
            thermal_factor = _coth(x)

        sigma_q_si2 = units._hbar / (2.0 * omega) * thermal_factor
        sigma_qdot_si2 = units._hbar * omega / 2.0 * thermal_factor

        # q: sqrt(amu) Angstrom. qdot_fs: sqrt(amu) Angstrom / fs.
        sigma_q = np.sqrt(sigma_q_si2 / units._amu * 1.0e20)
        sigma_qdot_fs = np.sqrt(sigma_qdot_si2 / units._amu * 1.0e-10)

        q[mode_index] = rng.normal(0.0, sigma_q)
        qdot_fs[mode_index] = rng.normal(0.0, sigma_qdot_fs)

    return q, qdot_fs


def _print_wigner_summary(config, mode_count, skipped_imaginary, skipped_low, output_dir):
    print("Wigner phase-space sampling")
    print(f"  initial conditions       : {int(config.get('n_initial_conditions', 0))}")
    print(f"  temperature              : {float(config.get('temperature_K', 0.0)):.6f} K")
    print(f"  random seed              : {config.get('random_seed', None)}")
    print(f"  frequency cutoff         : {float(config.get('frequency_cutoff_cm', 20.0)):.6f} cm^-1")
    print(f"  real modes included      : {mode_count}")
    print(f"  imaginary modes skipped  : {skipped_imaginary}")
    print(f"  low-frequency skipped    : {skipped_low}")
    print(f"  output directory         : {output_dir}")
    print("")
    print("ASE convention:")
    print("  D = M^(-1/2) H M^(-1/2)")
    print("  Cartesian mode L_j = U_j / sqrt(m)")
    print("  normalization: sum_a m_a L_i(a) dot L_j(a) = delta_ij")
    print("")
    print("Wigner distribution:")
    print("  Q_j    ~ N(0, hbar/(2 omega_j) coth(hbar omega_j/(2 kBT)))")
    print("  Qdot_j ~ N(0, hbar omega_j/2 coth(hbar omega_j/(2 kBT)))")
    print("")
    print("Cartesian reconstruction:")
    print("  R = R0 + sum_j L_j Q_j")
    print("  V =      sum_j L_j Qdot_j")
    print("  extxyz velocities are written in ASE velocity units for direct restart.")


def generate_wigner_initial_conditions(atoms, vib_data, config):
    """Generate Wigner phase-space initial conditions from ASE vibrations."""
    wigner_config = config.get("vib", {}).get("wigner", {})
    if not wigner_config.get("enabled", False):
        return []

    n_initial_conditions = int(wigner_config.get("n_initial_conditions", 1))
    if n_initial_conditions <= 0:
        print("Wigner sampling enabled but n_initial_conditions <= 0; skipping.")
        return []

    output_dir = Path(wigner_config.get("output_dir", "wigner_ic"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(wigner_config.get("output_prefix", "ic"))
    combined_file = wigner_config.get("combined_file", "wigner_initial_conditions.xyz")
    write_combined = bool(wigner_config.get("write_combined_trajectory", True))
    combined_path = Path(combined_file) if combined_file else None
    if combined_path and not combined_path.is_absolute():
        combined_path = output_dir / combined_path

    temperature_K = float(wigner_config.get("temperature_K", 0.0))
    cutoff_cm = float(wigner_config.get("frequency_cutoff_cm", 20.0))
    skip_imaginary = bool(wigner_config.get("skip_imaginary_modes", True))
    remove_translation = bool(wigner_config.get("remove_translation", False))
    remove_rotation = bool(wigner_config.get("remove_rotation", False))
    rng = np.random.default_rng(wigner_config.get("random_seed", None))

    _, frequencies_cm, cart_modes, _ = _mass_weighted_modes_from_ase(vib_data)
    mode_indices, skipped_imaginary, skipped_low = _mode_selection(
        frequencies_cm,
        cutoff_cm,
        skip_imaginary,
    )
    if len(mode_indices) == 0:
        raise ValueError("No real vibrational modes remain for Wigner sampling after filtering.")

    _print_wigner_summary(
        wigner_config,
        len(mode_indices),
        skipped_imaginary,
        skipped_low,
        output_dir,
    )

    base_positions = np.asarray(vib_data.get_atoms().get_positions(), dtype=np.float64)
    written_files = []
    combined_frames = []

    for i in range(n_initial_conditions):
        q, qdot_fs = _sample_mode_amplitudes(frequencies_cm, mode_indices, temperature_K, rng)
        displacement = np.einsum("m,mij->ij", q, cart_modes)
        velocities_fs = np.einsum("m,mij->ij", qdot_fs, cart_modes)
        velocities_ase = velocities_fs / units.fs

        frame = atoms.copy()
        frame.set_positions(base_positions + displacement)
        frame.set_velocities(velocities_ase)
        frame.set_array("velocities", np.asarray(velocities_ase, dtype=np.float64))

        if remove_translation:
            Stationary(frame, preserve_temperature=False)
            frame.set_array("velocities", np.asarray(frame.get_velocities(), dtype=np.float64))
        if remove_rotation:
            ZeroRotation(frame, preserve_temperature=False)
            frame.set_array("velocities", np.asarray(frame.get_velocities(), dtype=np.float64))

        frame.info.update(
            {
                "wigner_index": i + 1,
                "wigner_temperature_K": temperature_K,
                "wigner_frequency_cutoff_cm": cutoff_cm,
                "wigner_modes_included": int(len(mode_indices)),
                "wigner_seed": str(wigner_config.get("random_seed", None)),
            }
        )

        output_file = output_dir / f"{output_prefix}_{i + 1:06d}.xyz"
        write(output_file, frame, format="extxyz")
        written_files.append(str(output_file))
        if write_combined:
            combined_frames.append(frame)

    if write_combined and combined_path is not None:
        write(combined_path, combined_frames, format="extxyz")
        print(f"Combined Wigner initial conditions saved to {combined_path}")

    print(f"Wigner initial conditions written: {len(written_files)}")
    return written_files
