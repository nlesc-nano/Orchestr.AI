# src/orchestr_ai/qd/synthesis.py
"""
Multi-dot synthesis dashboard: several dots (and all their ligand-stripped
states) in equilibrium with the MA and MX_q monomers.

    python -m orchestr_ai.qd synthesis <record_dir> [<record_dir> ...] -o synthesis.html

Each record must have been run through the report step (props/solution.json).
Writes the interactive page and a static PNG at default conditions (toluene-like
eps = 2.4, total [MA] = 10 mM, [MX_q]/[MA] = 0.25, no bulk precipitation).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dashboards import INK, INK2, INK3, GRID, SITE, synthesis_page
from .solution import equilibrium

DEFAULTS = {"solvent": 2.4, "C_ma": 1e-2, "ratio": 0.25, "prec_shift": 0.0}


def load_exports(records):
    exports = []
    for r in records:
        path = Path(r) / "props" / "solution.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing: run `python -m orchestr_ai.qd run {r}` first")
        exports.append(json.loads(path.read_text()))
    ref = exports[0]
    for ex in exports[1:]:
        if ex["units"]["MA"] != ref["units"]["MA"] or ex["units"]["MX"] != ref["units"]["MX"]:
            raise ValueError("all dots must share the same MA and MX_q monomers")
        if not (np.allclose(ex["MA"]["G"], ref["MA"]["G"]) and np.allclose(ex["MX"]["G"], ref["MX"]["G"])
                and np.allclose(ex["bulk"], ref["bulk"])):
            raise ValueError("references differ between records (different MACE head or CIF?)")
    common = set(exports[0]["alpb_solvents"])
    for ex in exports[1:]:
        common &= set(ex["alpb_solvents"])
    for ex in exports:
        ex["alpb_solvents"] = {k: v for k, v in ex["alpb_solvents"].items() if k in common}
    return exports


def scan_temperature(exports, solvent, C_ma, C_mx, prec_shift=0.0, allow_bulk=False):
    T = exports[0]["T"]
    out, guess = [None] * len(T), None
    for i in range(len(T) - 1, -1, -1):
        out[i] = equilibrium(exports, i, solvent, C_ma, C_mx, prec_shift, allow_bulk, guess)
        guess = out[i]["guess"]
    return out


def write_synthesis(records, out: Path) -> None:
    exports = load_exports(records)
    out.write_text(synthesis_page(exports))
    _write_png(exports, out.with_suffix(".png"))
    print(f"[qdprops] synthesis: {out} and {out.with_suffix('.png')}")


def _write_png(exports, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import re
    tex = lambda f: re.sub(r"(\d+)", r"$_{\1}$", f)
    T = np.asarray(exports[0]["T"])
    fams = [e["formula"] for e in exports]
    ma, mx = exports[0]["units"]["MA"], exports[0]["units"]["MX"]
    d = DEFAULTS
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 9,
                         "axes.titlesize": 10.5, "axes.titleweight": "bold"})
    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5), facecolor="white")
    fig.suptitle(f"Synthesis thermodynamics: {', '.join(tex(f) for f in fams)}  "
                 f"(ε = {d['solvent']}, [{tex(ma)}]$_\\mathrm{{tot}}$ = {d['C_ma'] * 1e3:g} mM, "
                 f"[{tex(mx)}]/[{tex(ma)}] = {d['ratio']})", fontsize=12)
    for bulk, ax, title in ((False, axs[0, 0], "Yield, no bulk precipitation"),
                            (True, axs[0, 1], f"Yield, bulk {tex(ma)} allowed to precipitate")):
        res = scan_temperature(exports, d["solvent"], d["C_ma"], d["C_ma"] * d["ratio"], d["prec_shift"], bulk)
        for j, f in enumerate(fams):
            ax.plot(T, [r["families"][f]["frac_MA"] for r in res], color=SITE[j % 3], lw=2.2, label=tex(f))
        ax.plot(T, [r["monomer_frac"] for r in res], color=INK3, lw=1.8, ls=":", label=f"free {tex(ma)}")
        if bulk:
            ax.plot(T, [r["bulk_frac"] for r in res], color=INK, lw=1.8, ls="--", label=f"bulk {tex(ma)}")
        ax.set(xlabel="T (K)", ylabel=f"fraction of {tex(ma)}", ylim=(-0.02, 1.02))
        ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
        ax.set_title(title, pad=24, loc="left")
    ax = axs[1, 0]
    rr = np.arange(-3, 2.0001, 0.05)
    ti = list(T).index(500.0)
    g, rs = None, []
    for v in rr:
        r = equilibrium(exports, ti, d["solvent"], d["C_ma"], d["C_ma"] * 10 ** v, d["prec_shift"], False, g)
        g = r["guess"]
        rs.append(r)
    for j, f in enumerate(fams):
        ax.plot(rr, [r["families"][f]["frac_MA"] for r in rs], color=SITE[j % 3], lw=2.2, label=tex(f))
    ax.plot(rr, [r["monomer_frac"] for r in rs], color=INK3, lw=1.8, ls=":", label=f"free {tex(ma)}")
    ax.set(xlabel=f"log$_{{10}}$([{tex(mx)}]/[{tex(ma)}])", ylabel=f"fraction of {tex(ma)} at 500 K", ylim=(-0.02, 1.02))
    ax.set_title("Populations against the ligand-to-monomer ratio", pad=24, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    ax = axs[1, 1]
    res = scan_temperature(exports, d["solvent"], d["C_ma"], d["C_ma"] * d["ratio"], d["prec_shift"], False)
    for j, f in enumerate(fams):
        m = exports[j]["units"]["m"]
        ax.plot(T, [m - r["families"][f]["mean_k"] if r["families"][f]["mean_k"] is not None else np.nan for r in res],
                color=SITE[j % 3], lw=2.2, label=f"{tex(f)} (of {m})")
    ax.set(xlabel="T (K)", ylabel=f"{tex(mx)} bound per dot")
    ax.set_title("Ligand shell in the equilibrium population", pad=24, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)
    for a in axs.ravel():
        a.grid(True, color=GRID, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
