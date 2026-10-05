# src/orchestr_ai/qd/steps/stability.py
"""
Thermodynamic stability of a [MA]_n(MX_q)_m dot (see qdprops.references).

  decomposition   [MA]_n(MX_q)_m -> n MA(bulk) + m MX_q(1 M)
                  the excess (surface) free energy of the dot relative to
                  bulk growth and ligands in solution; > 0 for every finite
                  dot, decreasing per MA unit as the dot grows.
  binding         [MA]_n(MX_q)_m -> n MA(monomer) + m MX_q(1 M), per unit:
                  the cluster binding (cohesive) energy.

Each as dE (electronic), dE + dZPE and dG(T); the dot is a rigid-rotor
ideal-gas solute at 1 M with harmonic vibrations from the hessian step.
"""
from __future__ import annotations

import numpy as np

from ..references import binary_units, ideal_gas_g, reference_set, rotational_symmetry_number


def dot_free_energy(ctx, symbols, pts, energy: float, freqs) -> list:
    from ase import Atoms
    atoms = Atoms(list(symbols), positions=np.asarray(pts, float))
    sigma = rotational_symmetry_number(symbols, pts)
    return ideal_gas_g(atoms, energy, np.asarray(freqs, float), ctx.settings.temperatures, sigma), sigma


def run(ctx) -> dict:
    u = binary_units(ctx.symbols, ctx.charges, ctx.native)
    if u is None:
        return {"summary": {"skipped": "not a charge-balanced binary [MA]_n(MX_q)_m dot"}}
    refs = reference_set(ctx, u)
    T = ctx.settings.temperatures
    n, m = u["n_MA"], u["m_MXq"]
    e = ctx.results["relax"]["summary"]["energy_eV"]
    hs = ctx.results["hessian"]
    freqs = np.asarray(hs["frequencies_cm1"], float)
    zpe = hs["summary"]["zpe_eV"]
    symbols, pts = ctx.relaxed()
    g_dot, sigma = dot_free_energy(ctx, symbols, pts, e, freqs)

    bulk, ma = refs["bulk_MA"], refs["MA_monomer"]
    mx = refs.get("MXq_monomer")
    e_mx = mx["energy_eV"] if mx else 0.0
    zpe_mx = mx["zpe_eV"] if mx else 0.0
    g_mx = np.asarray(mx["G_1M_eV"]) if mx else np.zeros(len(T))

    de_dec = e - n * bulk["energy_per_fu_eV"] - m * e_mx
    dzpe_dec = zpe - n * bulk["zpe_per_fu_eV"] - m * zpe_mx
    dg_dec = np.asarray(g_dot) - n * np.asarray(bulk["G_per_fu_eV"]) - m * g_mx
    units = n + m
    de_bind = (e - n * ma["energy_eV"] - m * e_mx) / units
    dzpe_bind = (zpe - n * ma["zpe_eV"] - m * zpe_mx) / units
    dg_bind = (np.asarray(g_dot) - n * np.asarray(ma["G_1M_eV"]) - m * g_mx) / units
    n_surf = sum(1 for r in ctx.results["structure"]["role"] if r != "core")

    i300 = T.index(300.0) if 300.0 in T else None
    at300 = lambda arr: float(arr[i300]) if i300 is not None else None
    return {
        "summary": {
            "units": u,
            "decomposition_dE_eV": de_dec,
            "decomposition_dE_per_MA_eV": de_dec / n,
            "decomposition_dE_ZPE_eV": de_dec + dzpe_dec,
            "decomposition_dG_300K_eV": at300(dg_dec),
            "decomposition_dG_300K_per_MA_eV": at300(dg_dec / n),
            "excess_dE_per_surface_atom_eV": de_dec / n_surf if n_surf else None,
            "binding_dE_per_unit_eV": de_bind,
            "binding_dE_ZPE_per_unit_eV": de_bind + dzpe_bind,
            "binding_dG_300K_per_unit_eV": at300(dg_bind),
            "rotational_symmetry_number": sigma,
            "bulk_energy_per_fu_eV": bulk["energy_per_fu_eV"],
            "bulk_cell_lengths_A": bulk["cell_lengths_A"],
        },
        "temperatures": T,
        "G_dot_1M_eV": g_dot,
        "decomposition_dG_eV": dg_dec.tolist(),
        "binding_dG_per_unit_eV": dg_bind.tolist(),
        "references": {k: {kk: vv for kk, vv in v.items() if kk not in ("positions",)}
                       for k, v in refs.items() if isinstance(v, dict) and k != "key"},
        "provenance": refs.get("provenance"),
    }
