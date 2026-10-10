# src/orchestr_ai/qd/steps/electronic.py
"""
Tight-binding electronic structure at the MACE-MH-1 minimum (GFN2-xTB by
default; `Settings.xtb_method` = "gxtb" selects g-xTB).

Neutral single point: total energy, frontier orbitals and the orbital
(HOMO-LUMO) gap, atomic partial charges, dipole moment and, optionally, the
xtb gradient as a cross-check of how close the MACE minimum is to the xtb
one.  Vertical IP and EA come from the N-1 and N+1 doublets (same
geometry), giving the fundamental gap IP - EA, which for a tight-binding
method is more meaningful than the orbital gap.
"""
from __future__ import annotations

import numpy as np

from ..engines import XtbRunner


def run(ctx) -> dict:
    s = ctx.settings
    symbols, pts = ctx.relaxed()
    runner = XtbRunner(s.xtb_method)
    neutral = runner.run(symbols, pts, charge=0, uhf=0, gradient=s.xtb_gradient)
    summary = {
        "method": runner.provenance()["engine"],
        "energy_eV": neutral.energy_eV,
        "homo_eV": neutral.homo_eV,
        "lumo_eV": neutral.lumo_eV,
        "orbital_gap_eV": neutral.gap_eV,
        "dipole_debye": neutral.dipole_debye,
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
    if s.xtb_ip_ea:
        if s.xtb_method == "gfn2":
            # GFN2 absolute levels are shifted; IPEA-xTB delta-SCC carries the empirical correction.
            v = runner.vipea(symbols, pts)
            ip, ea, how = v["ip_eV"], v["ea_eV"], "IPEA-xTB delta-SCC (xtb --vipea)"
        else:
            cation = runner.run(symbols, pts, charge=1, uhf=1)
            anion = runner.run(symbols, pts, charge=-1, uhf=1)
            ip = cation.energy_eV - neutral.energy_eV
            ea = neutral.energy_eV - anion.energy_eV
            how = "delta-SCF N-1 / N+1 doublets"
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
