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
