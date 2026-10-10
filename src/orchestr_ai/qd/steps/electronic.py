# src/orchestr_ai/qd/steps/electronic.py
"""
Tight-binding electronic structure at the MACE-MH-1 minimum: g-xTB, or
GFN2-xTB when the g-xTB SCF fails (`Settings.xtb_method` "auto"; "gxtb" or
"gfn2" fix one).  The neutral, cation and anion runs always share a method.

Neutral single point: total energy, frontier orbitals and the orbital
(HOMO-LUMO) gap, atomic partial charges, dipole moment and, optionally, the
xtb gradient as a cross-check of how close the MACE minimum is to the xtb
one.  Vertical IP and EA come from the N-1 and N+1 doublets (same
geometry), giving the fundamental gap IP - EA, which for a tight-binding
method is more meaningful than the orbital gap.
"""
from __future__ import annotations

import numpy as np

from ..engines import XTB_FAST_FAIL, XTB_ORDER, XtbRunner


def run(ctx) -> dict:
    s = ctx.settings
    symbols, pts = ctx.relaxed()
    order = XTB_ORDER[s.xtb_method]
    for i, method in enumerate(order):
        runner = XtbRunner(method)
        # a method with a fallback gets one short try (see engines.XTB_FAST_FAIL)
        kw = dict(attempts=1, extra=XTB_FAST_FAIL) if i < len(order) - 1 else {}
        try:
            neutral = runner.run(symbols, pts, charge=0, uhf=0, gradient=s.xtb_gradient, **kw)
            ip_ea = _ip_ea(runner, symbols, pts, neutral, kw) if s.xtb_ip_ea else None
            break
        except RuntimeError:
            if i == len(order) - 1:
                raise
    summary = {
        "method": runner.provenance()["engine"],
        "energy_eV": neutral.energy_eV,
        "homo_eV": neutral.homo_eV,
        "lumo_eV": neutral.lumo_eV,
        "orbital_gap_eV": neutral.gap_eV,
        "dipole_debye": neutral.dipole_debye,
        "xtb_method": method,
        "xtb_methods_tried": list(order[:i + 1]),
    }
    charges = np.asarray(neutral.charges, float)
    by_element = {}
    for e in sorted(set(symbols)):
        q = charges[[i for i, x in enumerate(symbols) if x == e]] if charges.size else np.array([])
        if q.size:
            by_element[e] = {"mean": float(q.mean()), "min": float(q.min()), "max": float(q.max())}
    role = ctx.results.get("structure", {}).get("role")
    by_role = {}
    if role and charges.size:
        for r in ("core", "surface", "ligand"):
            idx = [i for i, x in enumerate(role) if x == r]
            if idx:
                by_role[r] = {"mean": float(charges[idx].mean()), "sum": float(charges[idx].sum())}
    if neutral.gradient_eV_A is not None:
        f = np.linalg.norm(neutral.gradient_eV_A, axis=1)
        summary["xtb_fmax_at_mace_min_eV_A"] = float(f.max())
        summary["xtb_frms_at_mace_min_eV_A"] = float(np.sqrt((f ** 2).mean()))
    if ip_ea:
        ip, ea, how = ip_ea
        summary.update({"ip_vertical_eV": ip, "ea_vertical_eV": ea, "fundamental_gap_eV": ip - ea,
                        "ip_ea_method": how})
    return {
        "summary": summary,
        "charges": charges.tolist(),
        "charges_by_element": by_element,
        "charges_by_role": by_role,
        "dipole_au": neutral.dipole_au,
        "provenance": runner.provenance(),
    }


def _ip_ea(runner, symbols, pts, neutral, kw) -> tuple:
    """Vertical IP and EA (eV) and how they were obtained, in the runner's method."""
    if runner.method == "gfn2":
        # GFN2 absolute levels are shifted; IPEA-xTB delta-SCC carries the empirical correction.
        v = runner.vipea(symbols, pts)
        return v["ip_eV"], v["ea_eV"], "IPEA-xTB delta-SCC (xtb --vipea)"
    cation = runner.run(symbols, pts, charge=1, uhf=1, **kw)
    anion = runner.run(symbols, pts, charge=-1, uhf=1, **kw)
    return (cation.energy_eV - neutral.energy_eV, neutral.energy_eV - anion.energy_eV,
            "delta-SCF N-1 / N+1 doublets")
