# src/orchestr_ai/qd/steps/dft_field.py
"""
DFT response of the dot from the CP2K chain's finite-field single points (qd.field):
Born charges, Raman tensors, polarizability, dipole and Hirshfeld charges, plus the
Hirshfeld charges of the MA and MX_q monomers. The tensors go to props/field.npz for the
vibspec step (IR/Raman on the MACE modes); the charges feed solvation and sites.

The runs are found at <record>/<settings.dft_dir>/field/ (an absolute dft_dir is used as
given); their outputs enter this step's cache key (run.STEP_EXTERNAL), so new DFT results
invalidate it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import field as fld

BOHR_A = 0.529177210903
AU_DEBYE = 2.541746473


def field_dir(ctx) -> Path:
    d = Path(ctx.settings.dft_dir)
    return (d if d.is_absolute() else ctx.record_dir / d) / "field"


def run(ctx) -> dict:
    d = field_dir(ctx)
    if not (d / "runs.txt").is_file():
        return {"summary": {"skipped": f"no DFT field runs in {d}"}}
    res = fld.analyse(str(d))
    symbols = ctx.relaxed()[0]
    if res["n_atoms"] != len(symbols):
        raise RuntimeError(f"field runs have {res['n_atoms']} atoms, the dot {len(symbols)}")
    arrays = {k: res[k] for k in ("born", "raman_tensor", "alpha", "dipole", "charges") if res[k] is not None}
    np.savez_compressed(ctx.props / "field.npz", **arrays)
    born = res["born"]
    trace = {e: float(np.mean([np.trace(born[i]) / 3 for i, x in enumerate(symbols) if x == e]))
             for e in sorted(set(symbols))}
    q = res["charges"]
    summary = {
        "source": "CP2K PBE finite field (13 single points)",
        "field_au": res["field_au"],
        "sum_rule_born_e": round(res["sum_rule_born_e"], 6),
        "sum_rule_raman_relative": round(res["sum_rule_raman_rel"], 6),
        "born_mean_trace_by_element_e": {k: round(v, 4) for k, v in trace.items()},
        "alpha_iso_A3": None if res["alpha"] is None else float(np.trace(res["alpha"]) / 3 * BOHR_A ** 3),
        "dipole_D": None if res["dipole"] is None else float(np.linalg.norm(res["dipole"]) * AU_DEBYE),
        "hirshfeld_by_element_e": None if q is None else {
            e: round(float(np.mean([q[i] for i, x in enumerate(symbols) if x == e])), 4) for e in sorted(set(symbols))},
        "monomers": sorted(res["monomers"]),
    }
    return {"summary": summary, "charges": None if q is None else q.tolist(), "monomers": res["monomers"],
            "provenance": {"engine": "CP2K", "functional": "PBE", "dir": str(d), "field_au": res["field_au"]}}
