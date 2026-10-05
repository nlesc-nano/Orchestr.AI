# src/orchestr_ai/qd/solution.py
"""
Solution thermodynamics of a [MA]_n(MX_q)_m dot, its ligand-stripped states
and the monomers, assembled for the dashboards (which evaluate the formulas
below live, with sliders for eps, concentrations and T).

Species i (dot after k removals, MA and MX_q monomers) are ideal solutes at a
1 M standard state with harmonic vibrations and rigid-rotor rotation:

    G_i^sol(T, eps) = G_i°(T) + dG_solv,i(eps),  dG_solv,i = (1 - 1/eps) G_GB,i
                      (Generalized Born on the GFN2-xTB charges, see steps/solvation.py)
    mu_i = G_i^sol + kT ln(c_i / 1 M)

and bulk MA is a solid (no solvation, no concentration term).  Then

    dG_dec  = [G_dot^sol - n G_bulk - m mu_MX] / n                 per MA unit
    dG_bind = [G_dot^sol - n mu_MA  - m mu_MX] / (n + m)           per unit
    T_diss : dG_bind(T_diss) = 0
    dG_k    = G_k^sol - G_(k-1)^sol + mu_MX                        stepwise MX_q loss

and, for a population of species {i: n_i MA + m_i MX_q} with total
concentrations C_MA, C_MX, the coupled equilibria

    c_i = exp(-dG°_i / kT) c_MA^n_i c_MX^m_i,   dG°_i = G_i^sol - n_i G_MA^sol - m_i G_MX^sol
    C_MA = c_MA + sum_i n_i c_i,   C_MX = c_MX + sum_i m_i c_i

are solved for the free monomer concentrations (2-D Newton in log space).
"""
from __future__ import annotations

import json

import numpy as np

from .records import read_xyz_first_frame

from .references import binary_units, bulk_free_energy, ideal_gas_g, reference_set, rotational_symmetry_number

EXPORT_T = [float(t) for t in range(50, 1501, 10)]


def _gas_g(symbols, pts, energy, freqs):
    from ase import Atoms
    atoms = Atoms(list(symbols), positions=np.asarray(pts, float))
    sigma = rotational_symmetry_number(symbols, pts)
    return np.asarray(ideal_gas_g(atoms, energy, np.asarray(freqs, float), EXPORT_T, sigma))


def export(ctx) -> dict | None:
    """All species of one record on EXPORT_T, with their solvation free energies."""
    u = binary_units(ctx.symbols, ctx.charges, ctx.native)
    if u is None or "solvation" not in ctx.results:
        return None
    refs = reference_set(ctx, u)
    sol = ctx.results["solvation"]
    e_full = ctx.results["relax"]["summary"]["energy_eV"]
    sym0, pts0 = ctx.relaxed()
    dots = [{"k": 0, "n": u["n_MA"], "m": u["m_MXq"], "sigma": rotational_symmetry_number(sym0, pts0),
             "G": _gas_g(sym0, pts0, e_full, ctx.results["hessian"]["frequencies_cm1"]).tolist(),
             "solv_inf": sol["gb_inf"]["dot_0"], "alpb": {s: v["dot_0"] for s, v in sol["alpb"]["dG"].items()}}]
    det = ctx.results.get("detachment", {})
    for st in det.get("steps", []):
        if st.get("frequencies_cm1") is None:
            break
        sym, pts = read_xyz_first_frame(str(ctx.props / f"detach_{st['k']}.xyz"))
        name = f"dot_{st['k']}"
        dots.append({"k": st["k"], "n": u["n_MA"], "m": u["m_MXq"] - st["k"],
                     "sigma": rotational_symmetry_number(sym, pts),
                     "G": _gas_g(sym, pts, e_full + st["E_mace_eV"], st["frequencies_cm1"]).tolist(),
                     "solv_inf": sol["gb_inf"][name], "alpb": {s: v[name] for s, v in sol["alpb"]["dG"].items()}})
    ma, mx = refs["MA_monomer"], refs["MXq_monomer"]
    # The evaluated configurations, as offsets from the ladder state of the same k (gas phase).
    ladder_e = {0: 0.0, **{st["k"]: st["E_mace_eV"] for st in det.get("steps", [])}}
    ens = []
    for k, e, sigma in det.get("ensemble_list", []):
        if k in ladder_e:
            ens.append([k, e - ladder_e[k], sigma])
    q = u["q"]
    return {
        "id": ctx.record.get("id"), "formula": ctx.record.get("formula"), "material": ctx.record.get("material"),
        "units": {"MA": f"{u['M']}{u['A']}", "MX": f"{u['M']}{u['X']}{q if q > 1 else ''}", "n": u["n_MA"],
                  "m": u["m_MXq"]},
        "T": EXPORT_T,
        "solvation_model": ctx.results["solvation"]["summary"]["model"],
        "alpb_solvents": {s: e for s, e in sol["alpb"]["solvents"].items()
                          if all(v is not None for v in sol["alpb"]["dG"][s].values())},
        "dots": dots,
        "MA": {"G": _gas_g(ma["symbols"], ma["positions"], ma["energy_eV"], ma["frequencies_cm1"]).tolist(),
               "solv_inf": sol["gb_inf"]["MA"], "alpb": {s: v["MA"] for s, v in sol["alpb"]["dG"].items()}},
        "MX": {"G": _gas_g(mx["symbols"], mx["positions"], mx["energy_eV"], mx["frequencies_cm1"]).tolist(),
               "solv_inf": sol["gb_inf"]["MX"], "alpb": {s: v["MX"] for s, v in sol["alpb"]["dG"].items()}},
        "bulk": bulk_free_energy(refs["bulk_MA"], EXPORT_T).tolist(),
        "ensemble": ens,                 # [k, E - E(ladder_k), rotational symmetry number], gas phase
        "warnings": ([] if len(dots) == len(det.get("steps", [])) + 1 else
                     ["detachment thermochemistry was skipped (dot too large): no stripped states"]),
    }


# --------------------------------------------------------------------------
# Formulas (mirrored in JS_LIB for the live dashboards)
# --------------------------------------------------------------------------

KB_EV = 8.617333262e-5


def solv_at(species: dict, eps_grid, solvent):
    """dG_solv of a species: solvent = float eps (Generalized Born, (1 - 1/eps) G_GB),
    a named ALPB solvent, or None (gas).  eps_grid is unused (kept for the call signature)."""
    if solvent is None:
        return 0.0
    if isinstance(solvent, str):
        v = species["alpb"].get(solvent)
        return float("nan") if v is None else v
    return (1.0 - 1.0 / float(solvent)) * species["solv_inf"]


def g_sol(species: dict, eps_grid, solvent, prec_shift=0.0) -> np.ndarray:
    """G_i^sol(T) at 1 M on the export grid; prec_shift (eV) is added (used for the MA monomer)."""
    return np.asarray(species["G"], float) + solv_at(species, eps_grid, solvent) + prec_shift


def curves(ex: dict, solvent=2.4, c_ma=1e-2, c_mx=1e-2, prec_shift=0.0) -> dict:
    """dG_dec(T), dG_bind(T), T_diss and the stepwise ladder for one dot."""
    T = np.asarray(ex["T"], float)
    kT = KB_EV * T
    eps = None
    n, m = ex["units"]["n"], ex["units"]["m"]
    g0 = g_sol(ex["dots"][0], eps, solvent)
    mu_ma = g_sol(ex["MA"], eps, solvent, prec_shift) + kT * np.log(c_ma)
    mu_mx = g_sol(ex["MX"], eps, solvent) + kT * np.log(c_mx)
    bulk = np.asarray(ex["bulk"], float)
    dec = (g0 - n * bulk - m * mu_mx) / n
    bind = (g0 - n * mu_ma - m * mu_mx) / (n + m)
    t_diss = None
    j = np.where((bind[:-1] < 0) & (bind[1:] >= 0))[0]
    if j.size:
        i = int(j[0])
        t_diss = float(T[i] + (0 - bind[i]) * (T[i + 1] - T[i]) / (bind[i + 1] - bind[i]))
    gs = [g_sol(d, eps, solvent) for d in ex["dots"]]
    ladder = np.array([gs[k] - gs[k - 1] + mu_mx for k in range(1, len(gs))])   # (K, nT)
    return {"T": T, "dec": dec, "bind": bind, "t_diss": t_diss, "ladder": ladder}


def mean_removed(ex: dict, solvent=2.4, c_mx=1e-2) -> np.ndarray:
    """<k>(T): Boltzmann average over the evaluated configurations at MX_q activity c_mx."""
    T = np.asarray(ex["T"], float)
    kT = KB_EV * T
    eps = None
    mu_mx = g_sol(ex["MX"], eps, solvent) + kT * np.log(c_mx)
    gs = {d["k"]: g_sol(d, eps, solvent) for d in ex["dots"]}
    sig = {d["k"]: d["sigma"] for d in ex["dots"]}
    confs = [(0, 0.0, sig[0])] + [tuple(c) for c in ex["ensemble"] if c[0] in gs]
    # G of a configuration: the ladder dot_k's G with its own symmetry number in place of dot_k's.
    lg = np.array([-(gs[k] - kT * np.log(sig[k]) + kT * np.log(sc) + off + k * mu_mx) / kT for k, off, sc in confs])
    ks = np.array([c[0] for c in confs], float)
    lg -= lg.max(axis=0, keepdims=True)
    p = np.exp(lg)
    return (p * ks[:, None]).sum(axis=0) / p.sum(axis=0)


def species_list(exports: list) -> list:
    """Every dot state of every export as {family, k, n, m, G, solv, alpb}."""
    out = []
    for ex in exports:
        for d in ex["dots"]:
            out.append({"family": ex["formula"], **d})
    return out


def _bisect(f, lo, hi, n=200, tol=1e-12):
    """Root of an increasing function; expands the bracket downwards if needed."""
    for _ in range(200):
        if f(lo) < 0:
            break
        lo -= max(50.0, abs(lo))
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def equilibrium(exports: list, ti: int, solvent=2.4, C_ma=1e-2, C_mx=1e-2, prec_shift=0.0,
                allow_bulk=False, guess=None) -> dict:
    """
    Free monomer concentrations and species populations at T index ti for total
    MA and MX_q concentrations C_ma, C_mx (M).  Both mass-balance residuals are
    increasing log-sum-exps, so they are solved by nested bisection in
    (a, b) = (ln c_MA, ln c_MX): for each b the MA balance gives a(b), and along
    that curve the MX balance is increasing in b.  With allow_bulk, c_MA is capped
    at the bulk solubility and the excess MA precipitates.  `guess` is unused
    (kept for the call signature).
    """
    ex0 = exports[0]
    t = ex0["T"][ti]
    kT = KB_EV * t
    g_ma = g_sol(ex0["MA"], None, solvent, prec_shift)[ti]
    g_mx = g_sol(ex0["MX"], None, solvent)[ti]
    sp = species_list(exports)
    nn = np.array([s["n"] for s in sp], float)
    mm = np.array([s["m"] for s in sp], float)
    g0 = np.array([(g_sol(s, None, solvent)[ti] - s["n"] * g_ma - s["m"] * g_mx) / kT for s in sp])
    ln_sat = -(g_ma - np.asarray(ex0["bulk"])[ti]) / kT
    lnA, lnX = np.log(C_ma), np.log(C_mx)

    def lse(v):
        top = v.max()
        return top + np.log(np.exp(v - top).sum())

    def lA(a, b):
        return lse(np.concatenate([[a], np.log(nn) - g0 + nn * a + mm * b]))

    def lX(a, b):
        mask = mm > 0
        return lse(np.concatenate([[b], np.log(mm[mask]) - g0[mask] + nn[mask] * a + mm[mask] * b]))

    def a_of(b):
        return _bisect(lambda a: lA(a, b) - lnA, lnA - 60.0, lnA)

    b = _bisect(lambda b: lX(a_of(b), b) - lnX, lnX - 60.0, lnX)
    a = a_of(b)
    bulk_frac, conserved = 0.0, True
    if allow_bulk and a > ln_sat:
        a = ln_sat
        b = _bisect(lambda b: lX(a, b) - lnX, lnX - 60.0, lnX)
        excess = 1.0 - np.exp(lA(a, b) - lnA)
        conserved = excess >= -1e-9
        bulk_frac = max(0.0, excess)
    lc = -g0 + nn * a + mm * b
    c = np.exp(lc)
    resid = (abs(np.exp(lA(a, b) - lnA) + bulk_frac - 1.0), abs(np.exp(lX(a, b) - lnX) - 1.0))
    fam = {}
    for s_, ci in zip(sp, c):
        f = fam.setdefault(s_["family"], {"frac_MA": 0.0, "conc": 0.0, "k_sum": 0.0})
        f["frac_MA"] += s_["n"] * ci / C_ma
        f["conc"] += ci
        f["k_sum"] += s_["k"] * ci
    for f in fam.values():
        f["mean_k"] = f["k_sum"] / f["conc"] if f["conc"] > 0 else None
    return {"ln_c_ma": float(a), "ln_c_mx": float(b), "monomer_frac": float(np.exp(a) / C_ma),
            "bulk_frac": float(bulk_frac), "families": fam, "supersaturation": float(np.exp(a - ln_sat)),
            "converged": max(resid) < 1e-6 and conserved, "residual": max(resid), "guess": (float(a), float(b))}
