# src/orchestr_ai/qd/steps/relax.py
"""
MACE-MH-1 relaxation of the builder start geometry.

BFGS to fmax; if it stalls (step limit or a non-finite step), FIRE continues
from the last geometry.  The relaxed geometry is written to props/relaxed.xyz
(centred at the centre of mass, like the library start files).
"""
from __future__ import annotations

import numpy as np

from ..engines import mace_calculator, mace_provenance


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD after optimal superposition (translation + rotation)."""
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(((a @ r - b) ** 2).sum(axis=1).mean()))


def run(ctx) -> dict:
    from ase import Atoms
    from ase.optimize import BFGS, FIRE

    s = ctx.settings
    calc = mace_calculator(s.head, s.model, s.device, s.dtype)
    atoms = Atoms(ctx.symbols, positions=ctx.start_pts)
    atoms.calc = calc
    e_start = float(atoms.get_potential_energy())
    f_start = float(np.linalg.norm(atoms.get_forces(), axis=1).max())

    trace = []

    def record():
        trace.append(float(atoms.get_potential_energy()))

    opt = BFGS(atoms, logfile=None)
    opt.attach(record, interval=1)
    converged = bool(opt.run(fmax=s.fmax, steps=s.max_steps))
    steps = opt.get_number_of_steps()
    optimizer = "BFGS"
    if not converged:
        opt = FIRE(atoms, logfile=None)
        opt.attach(record, interval=1)
        converged = bool(opt.run(fmax=s.fmax, steps=s.max_steps))
        steps += opt.get_number_of_steps()
        optimizer = "BFGS+FIRE"

    e_relax = float(atoms.get_potential_energy())
    f_relax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    pos = atoms.get_positions()
    pos = pos - pos.mean(axis=0)
    disp = np.linalg.norm(pos - (ctx.start_pts - ctx.start_pts.mean(axis=0)), axis=1)

    lines = [str(len(atoms)), f"{ctx.record.get('id', '')} MACE-MH-1/{s.head} relaxed E={e_relax:.6f} eV"]
    lines += [f"{sym:2s} {x:14.8f} {y:14.8f} {z:14.8f}" for sym, (x, y, z) in zip(ctx.symbols, pos)]
    (ctx.props / "relaxed.xyz").write_text("\n".join(lines) + "\n")

    n = len(atoms)
    return {
        "summary": {
            "converged": converged,
            "optimizer": optimizer,
            "steps": int(steps),
            "fmax_eV_A": f_relax,
            "fmax_start_eV_A": f_start,
            "energy_start_eV": e_start,
            "energy_eV": e_relax,
            "relaxation_energy_eV": e_relax - e_start,
            "relaxation_energy_meV_atom": 1000.0 * (e_relax - e_start) / n,
            "rmsd_A": _kabsch_rmsd(ctx.start_pts, pos),
            "max_displacement_A": float(disp.max()),
        },
        "energy_trace_eV": trace,
        "provenance": mace_provenance(s.head, s.model, s.device, s.dtype),
    }
