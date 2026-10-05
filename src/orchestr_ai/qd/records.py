"""
Library-record helpers the QD properties pipeline needs, copied from QD_builder
(builder.library_record, builder.analysis) so this package does not depend on
the builder. They move to the shared qd-schema package once it is published.
"""
from __future__ import annotations

import hashlib
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def read_xyz_first_frame(path: str) -> Tuple[List[str], NDArray[np.float64]]:
    """Symbols and coordinates of the first frame of an XYZ file."""
    with open(path) as fh:
        n = int(fh.readline().split()[0])
        fh.readline()
        symbols: List[str] = []
        coords: List[List[float]] = []
        for _ in range(n):
            parts = fh.readline().split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return symbols, np.asarray(coords, float)


def fingerprint(symbols: Sequence[str], pts: NDArray[np.float64], resolution: float = 0.05) -> str:
    """
    Rotation- and translation-invariant structure hash: composition plus the
    per-species sorted distances to the centroid, binned to `resolution` Å
    (identical to builder.library_record.fingerprint).
    """
    pts = np.asarray(pts, float)
    cen = pts.mean(axis=0)
    parts = []
    for s in sorted(set(symbols)):
        d = np.linalg.norm(pts[[i for i, x in enumerate(symbols) if x == s]] - cen, axis=1)
        bins = np.round(np.sort(d) / resolution).astype(int)
        parts.append(s + ":" + ",".join(map(str, bins)))
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _cov_radius(sym: str) -> float:
    """Covalent radius (Å): pymatgen's, else ASE's (builder.analysis uses pymatgen's)."""
    try:
        from pymatgen.core.periodic_table import Element
        r = Element(sym).covalent_radius
        if r is not None:
            return float(r)
    except Exception:
        pass
    from ase.data import atomic_numbers, covalent_radii
    return float(covalent_radii[atomic_numbers[sym]])


def pair_cut(a: str, b: str) -> float:
    """Bond cutoff for a ligand pair: 1.25 x the covalent-radius sum (builder.analysis._pair_cut)."""
    return 1.25 * (_cov_radius(a) + _cov_radius(b))
