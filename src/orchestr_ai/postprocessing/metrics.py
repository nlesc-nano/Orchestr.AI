import numpy as np

def _split_atom_vectors(flat_vectors, frames):
    """Split a concatenated atom vector array into one array per frame."""
    arr = np.asarray(flat_vectors, dtype=float)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError(f"Expected force components divisible by 3, got shape {arr.shape}")
        arr = arr.reshape(-1, 3)
    counts = [len(fr) for fr in frames]
    if arr.shape[0] != sum(counts):
        raise ValueError(f"Atom-vector length mismatch: got {arr.shape[0]}, expected {sum(counts)}")
    splits = np.cumsum(counts)[:-1]
    return [x.astype(float, copy=False) for x in np.split(arr, splits)]


def _force_summary_from_flat(force_vectors, frames):
    force_frames = _split_atom_vectors(force_vectors, frames)
    mean_norm = np.array(
        [
            np.nanmean(np.linalg.norm(f, axis=1)) if np.asarray(f).size else np.nan
            for f in force_frames
        ],
        dtype=float,
    )
    max_norm = np.array(
        [
            np.nanmax(np.linalg.norm(f, axis=1)) if np.asarray(f).size else np.nan
            for f in force_frames
        ],
        dtype=float,
    )
    return mean_norm, max_norm


def _std_from_sums(sum_values, sum_sq_values, n_samples):
    if n_samples <= 1:
        return np.zeros_like(sum_values, dtype=float)
    mean_values = sum_values / n_samples
    var = (sum_sq_values - n_samples * mean_values**2) / (n_samples - 1)
    return np.sqrt(np.maximum(var, 0.0))


def write_per_atom_uncertainties(sigma_F, sigma_E, frames, output_path, mu_E=None, n_atoms_per_frame=None, mu_F=None):
    """Write per-atom force uncertainties to an XYZ-like file.

    Parameters
    ----------
    sigma_F : ndarray
        Flat array of per-atom force uncertainty components (n_total_atoms * 3).
    sigma_E : ndarray
        Per-frame energy uncertainty (n_frames,).
    frames : list of ase.Atoms
        Pool frames with atom symbols and positions.
    output_path : str
        Path for the output XYZ file.
    mu_F : ndarray or None
        Flat array of per-atom mean force components (n_total_atoms * 3).
        Written as an extra muF_norm column when provided (enables
        relative-force gamma plots).
    n_atoms_per_frame : ndarray or None
        Atom counts per frame. Computed from frames if None.
    """
    if sigma_F is None:
        print("[PerAtomUQ] sigma_F is None; cannot write per-atom uncertainties.")
        return

    sigma_F_frames = _split_atom_vectors(sigma_F, frames)

    if mu_F is not None:
        mu_F_frames = _split_atom_vectors(mu_F, frames)
        if len(mu_F_frames) != len(sigma_F_frames):
            print("[PerAtomUQ] mu_F atom count mismatch; ignoring mu_F.")
            mu_F_frames = None
    else:
        mu_F_frames = None

    if n_atoms_per_frame is None:
        n_atoms_per_frame = np.array([len(fr) for fr in frames], dtype=int)

    sigma_E_per_frame = np.asarray(sigma_E, dtype=float)

    with open(output_path, "w") as fh:
        for i, (frame, sF_frame) in enumerate(zip(frames, sigma_F_frames)):
            n_atoms = len(frame)
            symbols = frame.get_chemical_symbols()
            positions = frame.get_positions()

            # Per-atom force uncertainty magnitude (L2 norm)
            sF_norm = np.linalg.norm(sF_frame, axis=1)
            muF_norm = np.linalg.norm(mu_F_frames[i], axis=1) if mu_F_frames is not None else None

            # Header line with energy uncertainty
            if mu_E is not None and i < len(mu_E):
                energy_val = float(mu_E[i])
            else:
                energy_val = float(getattr(frame, "info", {}).get("energy", 0.0) or 0.0)
            sigma_e = float(sigma_E_per_frame[i]) if i < len(sigma_E_per_frame) else 0.0
            fh.write(f"{n_atoms}\n")
            fh.write(f"frame={i} energy={energy_val:.6f} sigma_E={sigma_e:.6f}\n")

            for j in range(n_atoms):
                line = (f"{symbols[j]:<2} "
                        f"{positions[j, 0]:12.6f} {positions[j, 1]:12.6f} {positions[j, 2]:12.6f} "
                        f"{sF_frame[j, 0]:12.6f} {sF_frame[j, 1]:12.6f} {sF_frame[j, 2]:12.6f} "
                        f"{sF_norm[j]:12.6f}")
                if muF_norm is not None:
                    line += f" {muF_norm[j]:12.6f}"
                fh.write(line + "\n")

    n_frames = len(frames)
    n_atoms_total = int(sum(n_atoms_per_frame))
    print(f"[PerAtomUQ] Wrote {n_frames} frames, {n_atoms_total} atoms to {output_path}")

