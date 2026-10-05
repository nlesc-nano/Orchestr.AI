# src/orchestr_ai/qd/references.py
"""
Reference species for dot energetics, computed with the same MACE-MH-1 head
as the dot and cached per head and material.

A charge-balanced binary dot M_a A_b X_c (cation charge q, anion charge -q,
monovalent ligand X, so a = b + c/q) is b MA units plus m = c/q Z-type MX_q
units, [MA]_b(MX_q)_m.  Its energetics are referred to:

  * MA, bulk: the thermodynamic sink of the inorganic core.  Dots ripen and
    grow towards the bulk, so the free energy relative to bulk MA is the
    driving force for growth and dissolution.  Cell-relaxed bulk from the
    record's CIF, with phonons (ASE finite displacements in a supercell) for
    F_vib(T).
  * MA, monomer: the diatomic molecule.  Not a species found in solution,
    but the usual reference for cluster binding (cohesive) energies.
  * MX_q, monomer: the molecular Z-type ligand (linear CdCl2, trigonal InCl3,
    ...).  Detached Z-type ligands stay in solution as molecular complexes,
    so the monomer, not its bulk crystal, is the relevant sink.  It is
    treated as an ideal gas at a 1 M standard state; solvation and binding of
    L-type donors (amines, phosphines) are not included, which makes
    detachment free energies an upper bound.

Free energies use the harmonic approximation; molecules and dots add
ideal-gas translation and rotation (ASE IdealGasThermo).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .engines import mace_calculator, mace_hessian, mace_provenance

REFS_DIR = Path(os.environ.get("QDPROPS_REFS", str(Path.home() / ".cache/qdprops/references")))
R_GAS = 8.314462618       # J / (mol K)
STD_CONC = 1000.0         # mol / m³ (1 M)
KB_EV = 8.617333262e-5
CM1_TO_EV = 1.2398419843320026e-4


def standard_pressure(t: float) -> float:
    """Pressure (Pa) of an ideal gas at 1 mol/L and temperature t."""
    return STD_CONC * R_GAS * t


def ideal_gas_g(atoms, energy: float, freqs_cm1: np.ndarray, temperatures, symmetry_number: int,
                spin: float = 0.0) -> List[float]:
    """
    Gibbs free energy (eV) per temperature at the 1 M standard state:
    translation, rotation (rigid rotor, symmetry number) and pV from ASE's
    IdealGasThermo, plus the vibrational free energy of qdprops thermo()
    (harmonic U, quasi-RRHO S; the same treatment as for every other species).
    `freqs_cm1` are the vibrations only (translations and rotations projected out).
    """
    from ase.thermochemistry import IdealGasThermo
    from .steps.hessian import thermo
    n = len(atoms)
    if n == 1:
        geometry = "monatomic"
    else:
        pos = atoms.get_positions() - atoms.get_positions().mean(axis=0)
        sv = np.linalg.svd(pos, compute_uv=False)
        geometry = "linear" if sv[1] < 1e-3 * sv[0] else "nonlinear"
    from .steps.hessian import thermo_modes
    nu = thermo_modes(freqs_cm1) if len(freqs_cm1) else np.zeros(0)
    th = thermo(nu, temperatures) if nu.size else None
    # ASE needs the vibrations: give it the same (floored) modes and remove its harmonic
    # vibrational free energy, keeping translation, rotation and pV.
    gas = IdealGasThermo(vib_energies=nu * CM1_TO_EV, geometry=geometry, potentialenergy=energy, atoms=atoms,
                         symmetrynumber=symmetry_number, spin=spin)
    out = []
    for i, t in enumerate(temperatures):
        g = float(gas.get_gibbs_energy(t, standard_pressure(t), verbose=False))
        if th is not None:
            g += th["F_vib_eV"][i] - th["F_vib_harmonic_eV"][i]
        out.append(g)
    return out


def rotational_symmetry_number(symbols, pts) -> int:
    try:
        from pymatgen.core import Molecule
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer
        return int(PointGroupAnalyzer(Molecule(list(symbols), np.asarray(pts, float)), tolerance=0.1)
                   .get_rotational_symmetry_number())
    except Exception:
        return 1


def binary_units(symbols, charges: Dict[str, int], native: List[str]) -> Optional[dict]:
    """(M, A, X, q, n_MA, m_MXq) for a charge-balanced [MA]_n(MX_q)_m dot, else None."""
    comp = {e: list(symbols).count(e) for e in set(symbols)}
    cats = [e for e in native if charges.get(e, 0) > 0]
    ans = [e for e in native if charges.get(e, 0) < 0]
    ligs = [e for e in comp if e not in native]
    if len(cats) != 1 or len(ans) != 1 or len(ligs) > 1:
        return None
    m_, a_ = cats[0], ans[0]
    q = int(charges[m_])
    if int(charges[a_]) != -q:
        return None
    x = ligs[0] if ligs else None
    if x is not None and int(charges.get(x, 0)) != -1:
        return None
    nx = comp.get(x, 0) if x else 0
    if nx % q or comp[m_] - comp[a_] != nx // q:
        return None
    return {"M": m_, "A": a_, "X": x, "q": q, "n_MA": comp[a_], "m_MXq": nx // q}


# --------------------------------------------------------------------------
# Molecules
# --------------------------------------------------------------------------

def _molecule_guess(m: str, ligands: List[str], bond: float):
    """M bonded to len(ligands) atoms at `bond`, in the VSEPR arrangement."""
    from ase import Atoms
    k = len(ligands)
    dirs = {1: [(0, 0, 1)], 2: [(0, 0, 1), (0, 0, -1)],
            3: [(1, 0, 0), (-0.5, math.sqrt(3) / 2, 0), (-0.5, -math.sqrt(3) / 2, 0)],
            4: [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]}[k]
    pos = [(0.0, 0.0, 0.0)] + [tuple(bond * np.asarray(d) / np.linalg.norm(d)) for d in dirs]
    return Atoms([m] + ligands, positions=pos)


def _molecule(name: str, atoms, settings, temperatures) -> dict:
    from ase.optimize import BFGS
    from .steps.hessian import vibrations
    calc = mace_calculator(settings.head, settings.model, settings.device, settings.dtype)
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=1e-3, steps=500)
    e = float(atoms.get_potential_energy())
    h = mace_hessian(atoms, calc)
    freqs, _, _ = vibrations(h, atoms.get_positions(), atoms.get_masses())
    sigma = rotational_symmetry_number(atoms.get_chemical_symbols(), atoms.get_positions())
    g = ideal_gas_g(atoms, e, freqs, temperatures, sigma)
    from .steps.hessian import thermo_modes
    real = thermo_modes(freqs)
    return {"name": name, "symbols": atoms.get_chemical_symbols(), "positions": atoms.get_positions().tolist(),
            "energy_eV": e, "frequencies_cm1": freqs.tolist(), "zpe_eV": float(0.5 * real.sum() * CM1_TO_EV),
            "symmetry_number": sigma, "G_1M_eV": g}


# --------------------------------------------------------------------------
# Bulk
# --------------------------------------------------------------------------

def _bulk(cif: Path, ma: Tuple[str, str], settings, temperatures, workdir: Path) -> dict:
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    from ase.phonons import Phonons
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    st = Structure.from_file(str(cif)).get_primitive_structure()
    atoms = AseAtomsAdaptor.get_atoms(st)
    for key in list(atoms.arrays):
        if key not in ("numbers", "positions"):
            del atoms.arrays[key]
    atoms.info = {}
    calc = mace_calculator(settings.head, settings.model, settings.device, settings.dtype)
    atoms.calc = calc
    BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=1e-3, steps=1000)
    n_fu = sum(1 for s in atoms.get_chemical_symbols() if s == ma[1])
    e_fu = float(atoms.get_potential_energy()) / n_fu
    lengths = atoms.cell.lengths()
    sc = tuple(int(max(2, math.ceil(12.0 / L))) for L in lengths)
    ph = Phonons(atoms, calc, supercell=sc, delta=0.01, name=str(workdir / "phonon"))
    ph.run()
    ph.read(acoustic=True)
    ph.clean()
    dos = ph.get_dos(kpts=(24, 24, 24)).sample_grid(npts=1500, width=1e-3)
    eps = np.asarray(dos.get_energies(), float)
    w = np.asarray(dos.get_weights(), float)
    keep = eps > 1e-5
    eps, w = eps[keep], w[keep]
    de = eps[1] - eps[0] if len(eps) > 1 else 1.0
    w = w / (w.sum() * de) * 3 * len(atoms)          # normalise to 3 N_cell modes
    zpe = float((0.5 * eps * w).sum() * de) / n_fu
    f_vib = []
    for t in temperatures:
        x = eps / (KB_EV * t)
        f_cell = ((0.5 * eps + KB_EV * t * np.log1p(-np.exp(-x))) * w).sum() * de
        f_vib.append(float(f_cell) / n_fu)
    a, b, c = (float(v) for v in lengths)
    return {"cif": cif.name, "formula_unit": "".join(ma), "n_fu_cell": n_fu, "energy_per_fu_eV": e_fu,
            "cell_lengths_A": [a, b, c], "cell_angles_deg": [float(v) for v in atoms.cell.angles()],
            "phonon_supercell": list(sc), "zpe_per_fu_eV": zpe, "F_vib_per_fu_eV": f_vib,
            "G_per_fu_eV": [e_fu + f for f in f_vib],
            # phonon DOS (eV; states per eV per cell, normalised to 3 N_cell) for F_vib at any T
            "phonon_dos": {"energy_eV": eps.tolist(), "weight": w.tolist(), "de_eV": float(de)}}


def bulk_free_energy(bulk: dict, temperatures) -> np.ndarray:
    """E + F_vib per formula unit of the bulk reference at any temperature (eV)."""
    dos = bulk["phonon_dos"]
    eps, w, de = np.asarray(dos["energy_eV"]), np.asarray(dos["weight"]), dos["de_eV"]
    out = []
    for t in temperatures:
        x = eps / (KB_EV * t)
        f_cell = ((0.5 * eps + KB_EV * t * np.log1p(-np.exp(-x))) * w).sum() * de
        out.append(bulk["energy_per_fu_eV"] + float(f_cell) / bulk["n_fu_cell"])
    return np.asarray(out)


# --------------------------------------------------------------------------
# Reference set
# --------------------------------------------------------------------------

def reference_set(ctx, units: dict) -> dict:
    """Bulk MA, MA monomer and MX_q monomer for this record's material (cached)."""
    s = ctx.settings
    if ctx.cif is None:
        raise RuntimeError("no bulk CIF for this record (pass --cif)")
    from .engines import resolve_device
    dev = resolve_device(s.device)
    key_src = {"head": s.head, "model": Path(s.model).name, "device": dev,
               "dtype": "float32" if dev == "mps" else s.dtype,
               "cif": hashlib.sha256(Path(ctx.cif).read_bytes()).hexdigest()[:16],
               "units": {k: units[k] for k in ("M", "A", "X", "q")}, "T": s.temperatures, "v": 3}
    key = hashlib.sha256(json.dumps(key_src, sort_keys=True).encode()).hexdigest()[:12]
    m_, a_, x_, q = units["M"], units["A"], units["X"], units["q"]
    out_dir = REFS_DIR / s.head / f"{m_}{a_}-{Path(ctx.cif).stem}-{key}"
    path = out_dir / "references.json"
    if path.is_file():
        return json.loads(path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    from .steps.structure import _bulk_reference
    _cn, bond = _bulk_reference(ctx.cif, ctx.charges)
    bond = bond or 2.5
    refs = {"key": key_src, "temperatures": s.temperatures,
            "bulk_MA": _bulk(Path(ctx.cif), (m_, a_), s, s.temperatures, out_dir),
            "MA_monomer": _molecule(f"{m_}{a_}", _molecule_guess(m_, [a_], 0.9 * bond), s, s.temperatures)}
    if x_:
        refs["MXq_monomer"] = _molecule(f"{m_}{x_}{q if q > 1 else ''}",
                                        _molecule_guess(m_, [x_] * q, 0.9 * bond), s, s.temperatures)
    refs["provenance"] = mace_provenance(s.head, s.model, s.device, s.dtype)
    path.write_text(json.dumps(refs, indent=1))
    return refs
