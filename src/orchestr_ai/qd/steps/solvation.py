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

With the DFT chain's solvation.json (qd-workflows `qdw cp2k esp`, the model of qdw reaction) and
solvation_charges "dft", the model is instead

    dG_solv(eps) = (1 - 1/eps) E_inf + (alpha + gamma)(eps) S
    E_inf  COSMO (conductor) energy of the DFT electrostatic potential on the electron-density
           isosurface rho_0 = 5e-4 e/bohr^3 (CP2K PBE cubes, no atomic charges)
    S      the area of that isosurface; (alpha + gamma)(eps) = 2.16 ln eps - 14.79 mN/m (Hille et
           al., J. Chem. Phys. 150, 041710 (2019), beta = 0 fit over 67 non-aqueous solvents)

for the intact dot (k = 0), the fully stripped dot (k = K, the last ladder state) and the MA and
MX_q monomers; the states in between are linear in k between k = 0 and K (E_inf and S). Below
eps = 1.8, where the fit has no data, the cavity term goes to zero with (1 - 1/eps).

All geometries are the gas-phase MACE-MH-1 ones; solvation is a correction
from one gas-phase xTB single point per structure, independent of T: g-xTB,
or GFN2-xTB for every structure of the dot when g-xTB fails on any of them
(Settings.xtb_method; the GB radii were calibrated on GFN2 charges).
With Settings.solvation_checks (--solvation-checks), ddCOSMO (eps 2.4 and 80)
and ALPB (named solvents) are also computed for every structure, each
restarted from the converged gas-phase density; runs that fail are null.
"""
from __future__ import annotations

import numpy as np

from ..records import read_xyz_first_frame

from ..engines import xtb_pool, xtb_series
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


BETA0_MN_M = (2.16, -14.79)      # (alpha + gamma)(eps) = a ln eps + b, mN/m (Hille et al. 2019, beta = 0)
NONEL_MIN_EPS = 1.8
MN_M_EV_A2 = 1e-3 / 1.602176634e-19 * 1e-20


def cavity_energy(eps, area_A2) -> float:
    """(alpha + gamma)(eps) S in eV; 0 in the gas phase or without an area."""
    if area_A2 is None or eps is None or eps <= 1.0:
        return 0.0
    e = max(float(eps), NONEL_MIN_EPS)
    v = (BETA0_MN_M[0] * np.log(e) + BETA0_MN_M[1]) * MN_M_EV_A2 * float(area_A2)
    if eps < NONEL_MIN_EPS:                       # no data below: fade to the gas phase
        v *= (1 - 1 / eps) / (1 - 1 / NONEL_MIN_EPS)
    return float(v)


def cosmo_file(ctx):
    """The DFT chain's solvation.json (qdw cp2k esp collect)."""
    from pathlib import Path
    d = Path(ctx.settings.dft_dir)
    return (d if d.is_absolute() else ctx.record_dir / d) / "solvation.json"


def cosmo_data(ctx):
    import json
    f = cosmo_file(ctx)
    return json.loads(f.read_text()) if f.is_file() else None


def cosmo_at_k(data: dict, k: int) -> tuple:
    """(E_inf, S, how) of the dot after k removals: computed at k = 0 and the last state K, linear
    in k in between."""
    sp = data["species"]
    if f"dot_{k}" in sp:
        return sp[f"dot_{k}"]["E_inf_eV"], sp[f"dot_{k}"]["S_A2"], "DFT"
    ks = sorted(v["k"] for n, v in sp.items() if n.startswith("dot_") and v.get("k", 0) > 0)
    a = sp["dot_0"]
    if not ks or k > ks[-1]:
        return a["E_inf_eV"], a["S_A2"], "intact dot's value (no stripped state computed)"
    b, K = sp[f"dot_{ks[-1]}"], ks[-1]
    w = k / K
    return ((1 - w) * a["E_inf_eV"] + w * b["E_inf_eV"], (1 - w) * a["S_A2"] + w * b["S_A2"],
            f"linear in k between 0 and {K}")


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
    structures = [("dot_0", *ctx.relaxed())]
    for st in ctx.results.get("detachment", {}).get("steps", []):
        sym, pts = read_xyz_first_frame(str(ctx.props / f"detach_{st['k']}.xyz"))
        structures.append((f"dot_{st['k']}", list(sym), np.asarray(pts, float)))
    if u is not None:
        refs = reference_set(ctx, u)
        structures.append(("MA", refs["MA_monomer"]["symbols"], np.asarray(refs["MA_monomer"]["positions"])))
        if "MXq_monomer" in refs:
            structures.append(("MX", refs["MXq_monomer"]["symbols"], np.asarray(refs["MXq_monomer"]["positions"])))
    if ctx.settings.solvation_charges == "dft":
        data = cosmo_data(ctx)
        return _run_cosmo(ctx, structures, data) if data else _run_dft(ctx, structures)
    # ALPB reference state gsolv: 1 M ideal gas -> 1 M solution, as used throughout
    checks = ctx.settings.solvation_checks
    flag_sets = ([["--cosmo", f"{e:g}"] for e in COSMO_CHECK] + [["--alpb", s, "gsolv"] for s in ALPB_SOLVENTS]
                 if checks else [])
    gb, cosmo, alpb, smearing = {}, {}, {s: {} for s in ALPB_SOLVENTS}, {}
    # the structures are independent: run them side by side on the job's cores
    workers, threads = xtb_pool(len(structures))
    runner, series = xtb_series([(st[1], st[2]) for st in structures], flag_sets, workers, threads,
                                ctx.settings.xtb_method, base=SMEAR, fallbacks=SMEAR_FALLBACKS,
                                gxtb_max_atoms=ctx.settings.gxtb_max_atoms)
    for (name, sym, pts), (gas, energies, used) in zip(structures, series):
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
            "model": f"Generalized Born on {runner.provenance()['engine']} charges (HCT radii, Bondi x {RADIUS_SCALE})",
            "xtb_method": runner.method,
            "dG_solv_dot_eps2.4_eV": (1 - 1 / 2.4) * gb["dot_0"],
            "dG_solv_dot_eps80_eV": (1 - 1 / 80) * gb["dot_0"],
            "cosmo_dot_eps2.4_eV": cosmo["dot_0"]["2.4"] if checks else None,
            "n_structures": len(structures),
            "n_cosmo_failed": sum(v is None for c in cosmo.values() for v in c.values()),
            "n_alpb_failed": sum(v is None for a in alpb.values() for v in a.values()),
            "xtb_parallel": f"{workers} x {threads} threads",
        },
        "gb_inf": gb,
        "radius_scale": RADIUS_SCALE,
        "smearing": smearing,           # xtb electronic-temperature flags used per structure
        "cosmo_check": cosmo,
        "alpb": {"solvents": ALPB_SOLVENTS if checks else {}, "dG": alpb if checks else {}},
        "provenance": runner.provenance(),
    }


def dft_charges(ctx) -> np.ndarray:
    """Hirshfeld charges of the intact dot from the field step (CP2K PBE)."""
    q = ctx.results.get("field", {}).get("charges")
    if not q:
        raise RuntimeError("solvation_charges is dft but the field step has no Hirshfeld charges")
    return np.asarray(q, float)


def frozen_charges(q0, keep) -> np.ndarray:
    """The intact dot's charges on the atoms kept, re-neutralised (no new SCF)."""
    qk = np.asarray(q0, float)[list(keep)]
    return qk - qk.sum() / len(qk)


def _run_dft(ctx, structures) -> dict:
    """GB on DFT charges: the intact dot's Hirshfeld charges, the desorption products' frozen from
    them (atoms left after the path's removals), the monomers' from their own single points."""
    q0 = dft_charges(ctx)
    mono = ctx.results["field"].get("monomers", {})
    removed, keep_k = set(), {}
    for st in ctx.results.get("detachment", {}).get("steps", []):
        c, lig = st["unit"]
        removed |= {c, *lig}
        keep_k[f"dot_{st['k']}"] = [i for i in range(len(q0)) if i not in removed]
    gb = {}
    for name, sym, pts in structures:
        if name == "dot_0":
            q = q0
        elif name in keep_k:
            q = frozen_charges(q0, keep_k[name])
        else:
            if name not in mono or mono[name]["symbols"] != list(sym):
                raise RuntimeError(f"no DFT charges for the {name} monomer (field step monomers: {sorted(mono)})")
            q = np.asarray(mono[name]["charges"], float)
        if len(q) != len(sym):
            raise RuntimeError(f"{name}: {len(q)} charges for {len(sym)} atoms")
        gb[name] = gb_conductor_energy(list(sym), pts, q)
    return {
        "summary": {
            "model": f"Generalized Born on CP2K PBE Hirshfeld charges (HCT radii, Bondi x {RADIUS_SCALE}; "
                     "desorption products: the intact dot's charges, re-neutralised)",
            "charges": "dft",
            "dG_solv_dot_eps2.4_eV": (1 - 1 / 2.4) * gb["dot_0"],
            "dG_solv_dot_eps80_eV": (1 - 1 / 80) * gb["dot_0"],
            "n_structures": len(structures),
        },
        "gb_inf": gb,
        "radius_scale": RADIUS_SCALE,
        "alpb": {"solvents": {}, "dG": {}},
        "provenance": ctx.results["field"].get("provenance", {}),
    }


def _run_cosmo(ctx, structures, data: dict) -> dict:
    """COSMO on the DFT potential + the cavity/dispersion term on the isosurface area (solvation.json)."""
    sp = data["species"]
    e_inf, area, how = {}, {}, {}
    for name, sym, _pts in structures:
        if name.startswith("dot_"):
            e_inf[name], area[name], how[name] = cosmo_at_k(data, int(name.split("_")[1]))
        else:
            if name not in sp:
                raise RuntimeError(f"solvation.json has no {name} monomer (has {sorted(sp)})")
            if sp[name]["n_atoms"] != len(sym):
                raise RuntimeError(f"solvation.json {name}: {sp[name]['n_atoms']} atoms, the reference {len(sym)}")
            e_inf[name], area[name], how[name] = sp[name]["E_inf_eV"], sp[name]["S_A2"], "DFT"
    d0 = e_inf["dot_0"], area["dot_0"]
    return {
        "summary": {
            "model": "COSMO on the CP2K PBE electrostatic potential at the density isosurface "
                     f"({data['rho0_e_bohr3']:.1e} e/bohr^3) + cavity/dispersion (alpha+gamma)(eps) S (Hille 2019, beta = 0); "
                     "stripped states linear in k between the intact and the fully stripped dot",
            "charges": "dft",
            "dG_solv_dot_eps2.4_eV": (1 - 1 / 2.4) * d0[0] + cavity_energy(2.4, d0[1]),
            "dG_solv_dot_eps80_eV": (1 - 1 / 80) * d0[0] + cavity_energy(80, d0[1]),
            "dG_el_dot_eps2.4_eV": (1 - 1 / 2.4) * d0[0],
            "S_dot_A2": d0[1],
            "computed": sorted(sp),
            "n_structures": len(structures),
        },
        "gb_inf": e_inf,                 # conductor energy E_inf (the key name kept for the exports)
        "cav_S": area,
        "source": how,
        "beta0_alpha_gamma_mN_m": list(BETA0_MN_M),
        "alpb": {"solvents": {}, "dG": {}},
        "provenance": {"engine": "CP2K PBE cubes + qd-workflows COSMO", "file": str(cosmo_file(ctx)),
                       **ctx.results.get("field", {}).get("provenance", {})},
    }
