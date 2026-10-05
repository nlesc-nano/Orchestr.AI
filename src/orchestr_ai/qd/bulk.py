# src/orchestr_ai/qd/bulk.py
"""
MACE-MH-1 bulk reference of a record's CIF, for comparisons in the report:
the cell-relaxed primitive cell, its cation–anion nearest-neighbour bond
length and its Gamma-point optical frequencies (primitive-cell force
constants, i.e. q = 0; no LO–TO splitting, which needs Born charges).
Cached per MACE head and CIF in ~/.cache/qdprops/references/<head>/bulk/.

EXPERIMENT holds measured room-temperature lattice parameters of the bulk
phases in the library, keyed by CIF stem, with the bond length that follows
from them, for the same plots.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

# Experimental bulk structures (room temperature).  Bond: zinc blende a√3/4;
# wurtzite with the ideal u = 3/8, mean of the axial u·c and the three basal bonds.
EXPERIMENT = {
    "CdSe_zb": {"phase": "zinc blende", "a_A": 6.05,
                "ref": "Landolt–Börnstein III/41B, CdSe: lattice parameters"},
    "CdSe_wur": {"phase": "wurtzite", "a_A": 4.299, "c_A": 7.010,
                 "ref": "Landolt–Börnstein III/41B, CdSe: lattice parameters"},
}


def experimental_bond(cif_stem: str):
    """(bond length Å, reference) of the measured bulk, or None."""
    e = EXPERIMENT.get(cif_stem)
    if not e:
        return None
    if "c_A" in e:
        a, c, u = e["a_A"], e["c_A"], 0.375
        axial = u * c
        basal = math.sqrt(a * a / 3 + ((0.5 - u) * c) ** 2)
        return (axial + 3 * basal) / 4, e["ref"]
    return e["a_A"] * math.sqrt(3) / 4, e["ref"]


def mace_bulk(cif, settings) -> dict:
    """Cell-relaxed primitive cell: lattice, nearest-neighbour bond, Gamma optical frequencies (cached)."""
    from .engines import resolve_device
    from .references import REFS_DIR
    dev = resolve_device(settings.device)
    src = {"head": settings.head, "model": Path(settings.model).name, "device": dev,
           "dtype": "float32" if dev == "mps" else settings.dtype,
           "cif": hashlib.sha256(Path(cif).read_bytes()).hexdigest()[:16], "v": 1}
    path = REFS_DIR / settings.head / "bulk" / (hashlib.sha256(json.dumps(src, sort_keys=True).encode())
                                                .hexdigest()[:12] + ".json")
    if path.is_file():
        return json.loads(path.read_text())
    out = {"key": src, "cif": Path(cif).name, **_compute(cif, settings)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def _compute(cif, settings) -> dict:
    from ase.filters import FrechetCellFilter
    from ase.neighborlist import neighbor_list
    from ase.optimize import BFGS
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from .engines import mace_calculator
    from .steps.hessian import EIG_TO_CM1

    atoms = AseAtomsAdaptor.get_atoms(Structure.from_file(str(cif)).get_primitive_structure())
    for key in list(atoms.arrays):
        if key not in ("numbers", "positions"):
            del atoms.arrays[key]
    atoms.info = {}
    atoms.calc = mace_calculator(settings.head, settings.model, settings.device, settings.dtype)
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-3, steps=1000)
    # nearest-neighbour bond: shortest distances between unlike species (tetrahedral cation–anion)
    i, j, dist = neighbor_list("ijd", atoms, 4.0)
    sym = np.array(atoms.get_chemical_symbols())
    unlike = sym[i] != sym[j]
    dmin = dist[unlike].min()
    bonds = dist[unlike & (dist < dmin * 1.05)]
    # Gamma-point force constants: displacing an atom of the primitive cell moves its whole sublattice
    n, d = len(atoms), 0.01
    pos0 = atoms.get_positions().copy()
    h = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        a_, ax = divmod(k, 3)
        f = []
        for s in (1.0, -1.0):
            p = pos0.copy()
            p[a_, ax] += s * d
            atoms.set_positions(p)
            f.append(atoms.get_forces().ravel())
        h[:, k] = -(f[0] - f[1]) / (2 * d)
    atoms.set_positions(pos0)
    w = 1 / np.sqrt(np.repeat(atoms.get_masses(), 3))
    lam = np.linalg.eigvalsh(0.5 * (h + h.T) * w[:, None] * w[None, :])
    nu = np.sign(lam) * np.sqrt(np.abs(lam)) * EIG_TO_CM1
    acoustic = set(np.argsort(np.abs(nu))[:3])          # the three modes nearest zero
    return {"cell_lengths_A": [float(x) for x in atoms.cell.lengths()],
            "cell_angles_deg": [float(x) for x in atoms.cell.angles()],
            "bond_A": float(bonds.mean()), "bond_spread_A": float(bonds.max() - bonds.min()),
            "optical_cm1": sorted(float(x) for k, x in enumerate(nu) if k not in acoustic)}
