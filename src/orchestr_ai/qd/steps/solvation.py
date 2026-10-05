# src/orchestr_ai/qd/steps/solvation.py
"""
Implicit-solvent free energies of the dot, its stripped states and the
monomer references, at the MACE-MH-1 geometries.

Model used by the dashboards: Generalized Born electrostatics of the
GFN2-xTB (gas-phase) charge distribution,

    dG_solv(eps) = -(1/2) (1 - 1/eps) sum_ij q_i q_j / f_GB(r_ij),
    f_GB = sqrt(r^2 + R_i R_j exp(-r^2 / 4 R_i R_j))

with Hawkins-Cramer-Truhlar effective Born radii from Bondi radii scaled by
RADIUS_SCALE.  It is exactly continuous in eps (one number per structure,
G_inf, times 1 - 1/eps) and deterministic.  Self-consistent GFN2-xTB
ddCOSMO, which also lets the electrons polarise, proved numerically unstable
for the ligand-stripped dots (diverging or jumping between SCC solutions
from one eps to the next); RADIUS_SCALE = 1.15 reproduces ddCOSMO within
~0.1 eV for the monomers and Cd16Se13Cl6, while for larger, more polarisable
dots GB gives less (no electronic polarisation).  Non-electrostatic
(cavity, dispersion) terms are not included.

All geometries are the gas-phase MACE-MH-1 ones; solvation is a correction
from one gas-phase GFN2-xTB single point per structure, independent of T.
With Settings.solvation_checks (--solvation-checks), ddCOSMO (eps 2.4 and 80)
and ALPB (named solvents) are also computed for every structure, each
restarted from the converged gas-phase density; runs that fail are null.
"""
from __future__ import annotations

import numpy as np

from ..records import read_xyz_first_frame

from ..engines import XtbRunner
from ..references import binary_units, reference_set

RADIUS_SCALE = 1.15
COULOMB_EV_A = 14.3996454784
BONDI = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85,
         "I": 1.98, "Se": 1.90, "Te": 2.06, "As": 1.85, "Sb": 2.06, "Zn": 1.39, "Cd": 1.58, "Hg": 1.55,
         "Ga": 1.87, "In": 1.93, "Al": 1.84, "Pb": 2.02, "Cs": 3.43}
COSMO_CHECK = [2.4, 80.0]
ALPB_SOLVENTS = {"hexane": 1.88, "toluene": 2.38, "chcl3": 4.71, "thf": 7.43, "acetone": 20.7,
                 "dmso": 46.7, "water": 78.4}
# Gas-phase single point: plain first; Fermi smearing only if the SCC does not converge.
SMEAR = []
SMEAR_FALLBACKS = [["--etemp", "1500", "--iterations", "500"], ["--etemp", "3000", "--iterations", "500"]]


def hct_born_radii(symbols, pts, scale=RADIUS_SCALE, offset=0.09, sfac=0.8) -> np.ndarray:
    """Hawkins-Cramer-Truhlar pairwise-descreening effective Born radii (Å)."""
    p = np.asarray(pts, float)
    r0 = np.array([BONDI.get(s, 2.0) * scale for s in symbols]) - offset
    R = np.empty(len(symbols))
    for i in range(len(symbols)):
        acc = 0.0
        for j in range(len(symbols)):
            if i == j:
                continue
            d = float(np.linalg.norm(p[i] - p[j]))
            sj = sfac * r0[j]
            if d + sj <= r0[i]:
                continue
            L, U = max(r0[i], abs(d - sj)), d + sj
            acc += 0.5 * (1 / L - 1 / U + d / 4 * (1 / U ** 2 - 1 / L ** 2) + np.log(L / U) / (2 * d)
                          + sj ** 2 / (4 * d) * (1 / L ** 2 - 1 / U ** 2))
        R[i] = 1.0 / max(1.0 / r0[i] - acc, 1e-3)
    return R


def gb_conductor_energy(symbols, pts, charges) -> float:
    """-(1/2) sum_ij q_i q_j / f_GB in eV: dG_solv(eps) = (1 - 1/eps) times this."""
    p = np.asarray(pts, float)
    q = np.asarray(charges, float)
    R = hct_born_radii(symbols, p)
    r2 = ((p[:, None] - p[None]) ** 2).sum(-1)
    RR = R[:, None] * R[None]
    f = np.sqrt(r2 + RR * np.exp(-r2 / (4 * RR)))
    return float(-0.5 * COULOMB_EV_A * (q[:, None] * q[None] / f).sum())


def run(ctx) -> dict:
    u = binary_units(ctx.symbols, ctx.charges, ctx.native)
    runner = XtbRunner("gfn2")                  # gas-phase single points: multithreaded is fine
    structures = [("dot_0", *ctx.relaxed())]
    for st in ctx.results.get("detachment", {}).get("steps", []):
        sym, pts = read_xyz_first_frame(str(ctx.props / f"detach_{st['k']}.xyz"))
        structures.append((f"dot_{st['k']}", list(sym), np.asarray(pts, float)))
    if u is not None:
        refs = reference_set(ctx, u)
        structures.append(("MA", refs["MA_monomer"]["symbols"], np.asarray(refs["MA_monomer"]["positions"])))
        if "MXq_monomer" in refs:
            structures.append(("MX", refs["MXq_monomer"]["symbols"], np.asarray(refs["MXq_monomer"]["positions"])))
    # ALPB reference state gsolv: 1 M ideal gas -> 1 M solution, as used throughout
    checks = ctx.settings.solvation_checks
    flag_sets = ([["--cosmo", f"{e:g}"] for e in COSMO_CHECK] + [["--alpb", s, "gsolv"] for s in ALPB_SOLVENTS]
                 if checks else [])
    gb, cosmo, alpb, smearing = {}, {}, {s: {} for s in ALPB_SOLVENTS}, {}
    for name, sym, pts in structures:
        gas, energies, used = runner.run_series(list(sym), pts, flag_sets, base=SMEAR, fallbacks=SMEAR_FALLBACKS)
        smearing[name] = " ".join(used) or "none"
        gb[name] = gb_conductor_energy(list(sym), pts, gas.charges)
        e_gas = gas.energy_eV
        diff = [None if e is None else e - e_gas for e in energies]
        if checks:
            cosmo[name] = dict(zip([str(e) for e in COSMO_CHECK], diff[:len(COSMO_CHECK)]))
            for s, v in zip(ALPB_SOLVENTS, diff[len(COSMO_CHECK):]):
                alpb[s][name] = v
    return {
        "summary": {
            "model": f"Generalized Born on GFN2-xTB charges (HCT radii, Bondi x {RADIUS_SCALE})",
            "dG_solv_dot_eps2.4_eV": (1 - 1 / 2.4) * gb["dot_0"],
            "dG_solv_dot_eps80_eV": (1 - 1 / 80) * gb["dot_0"],
            "cosmo_dot_eps2.4_eV": cosmo["dot_0"]["2.4"] if checks else None,
            "n_structures": len(structures),
            "n_cosmo_failed": sum(v is None for c in cosmo.values() for v in c.values()),
            "n_alpb_failed": sum(v is None for a in alpb.values() for v in a.values()),
        },
        "gb_inf": gb,
        "radius_scale": RADIUS_SCALE,
        "smearing": smearing,           # xtb electronic-temperature flags used per structure
        "cosmo_check": cosmo,
        "alpb": {"solvents": ALPB_SOLVENTS if checks else {}, "dG": alpb if checks else {}},
        "provenance": runner.provenance(),
    }
