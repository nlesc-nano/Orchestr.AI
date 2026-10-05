# src/orchestr_ai/qd/steps/structure.py
"""
Structure analysis of the start and relaxed geometries.

Bonds are opposite-charge pairs closer than BOND_TOL x the bulk
cation-anion bond of the CIF for native pairs (relaxation stretches surface
and centre bonds by several per cent, beyond the builder's tight cut-geometry
cutoffs) and 1.25 x the covalent-radius sum for ligand pairs.  Native
atoms with their bulk native coordination are "core", the rest "surface";
ligands are classified by the number of cations they bind (mu1/mu2/mu3).
The relaxed bond graph is compared with the start one so that ligand
detachment or migration during the relaxation is flagged.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..records import pair_cut as _pair_cut

HIST_STEP = 0.02  # Å, bond-length histogram bin
BOND_TOL = 1.2    # native bond cutoff / bulk bond


def _bulk_reference(cif: Optional[str], charges: Dict[str, int]) -> Tuple[Dict[str, int], Optional[float]]:
    """(bulk CN per element, bulk cation-anion bond in Å) from the periodic CIF."""
    if not cif:
        return {}, None
    from pymatgen.core import Structure
    st = Structure.from_file(str(cif))
    syms = [str(s.specie.symbol) for s in st.sites]
    bonds = [n.nn_distance for i, nbrs in enumerate(st.get_all_neighbors(6.0)) for n in nbrs
             if charges.get(syms[i], 0) * charges.get(str(n.specie.symbol), 0) < 0]
    if not bonds:
        return {}, None
    bond = float(min(bonds))
    per_el: Dict[str, Counter] = {}
    for i, nbrs in enumerate(st.get_all_neighbors(BOND_TOL * bond)):
        cn = sum(1 for n in nbrs if charges.get(syms[i], 0) * charges.get(str(n.specie.symbol), 0) < 0)
        per_el.setdefault(syms[i], Counter())[cn] += 1
    return {s: c.most_common(1)[0][0] for s, c in per_el.items()}, bond


def _cutoffs(symbols, charges, native, bulk_bond) -> Dict[Tuple[str, str], float]:
    nat = set(native)
    elems = sorted(set(symbols))
    cut = {}
    for a in elems:
        for b in elems:
            if charges.get(a, 0) * charges.get(b, 0) >= 0:
                continue
            cut[(a, b)] = BOND_TOL * bulk_bond if (bulk_bond and a in nat and b in nat) else _pair_cut(a, b)
    return cut


def _bonds(symbols: List[str], pts: np.ndarray, cut: Dict[Tuple[str, str], float]) -> List[Set[int]]:
    nb: List[Set[int]] = [set() for _ in symbols]
    if not cut:
        return nb
    for i, j in cKDTree(pts).query_pairs(max(cut.values())):
        c = cut.get((symbols[i], symbols[j]))
        if c is not None and np.linalg.norm(pts[i] - pts[j]) <= c:
            nb[i].add(j)
            nb[j].add(i)
    return nb


def _stats(x: List[float]) -> dict:
    if not x:
        return {"n": 0}
    a = np.asarray(x, float)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max())}


def _hist(x: List[float], lo: float, hi: float) -> dict:
    edges = np.arange(lo, hi + HIST_STEP, HIST_STEP)
    h, e = np.histogram(x, bins=edges)
    return {"left": [round(v, 4) for v in e[:-1].tolist()], "count": h.tolist()}


def analyse(symbols, pts, charges, native, ligands, bulk_cn, cut) -> dict:
    nb = _bonds(symbols, pts, cut)
    nat = set(native)
    cations = {e for e in symbols if charges.get(e, 0) > 0}
    native_cn = [sum(1 for j in nb[i] if symbols[j] in nat) for i in range(len(symbols))]
    role = []
    for i, s in enumerate(symbols):
        if s in nat:
            role.append("core" if bulk_cn.get(s) and native_cn[i] >= bulk_cn[s] else "surface")
        else:
            role.append("ligand")
    cn_hist = {e: dict(sorted(Counter(len(nb[i]) for i, s in enumerate(symbols) if s == e).items()))
               for e in sorted(set(symbols))}
    mu = Counter()
    for i, s in enumerate(symbols):
        if s in set(ligands):
            mu[f"mu{sum(1 for j in nb[i] if symbols[j] in cations)}"] += 1
    bonds: Dict[str, Dict[str, List[float]]] = {}
    for i in range(len(symbols)):
        for j in nb[i]:
            if j <= i:
                continue
            a, b = sorted((symbols[i], symbols[j]), key=lambda e: (charges.get(e, 0) < 0, e))
            kind = "core" if role[i] == role[j] == "core" else "shell"
            bonds.setdefault(f"{a}-{b}", {"core": [], "shell": []})[kind].append(
                float(np.linalg.norm(pts[i] - pts[j])))
    return {"nb": nb, "role": role, "cn_hist": cn_hist, "ligand_mu": dict(sorted(mu.items())), "bonds": bonds}


def run(ctx) -> dict:
    bulk_cn, bulk_bond = _bulk_reference(ctx.cif, ctx.charges)
    cut = _cutoffs(ctx.symbols, ctx.charges, ctx.native, bulk_bond)
    out = {}
    for tag, (syms, pts) in (("start", (ctx.symbols, ctx.start_pts)), ("relaxed", ctx.relaxed())):
        out[tag] = analyse(list(syms), np.asarray(pts, float), ctx.charges, ctx.native, ctx.ligands, bulk_cn, cut)

    s, r = out["start"], out["relaxed"]
    edges = lambda nb: {(i, j) for i, js in enumerate(nb) for j in js if i < j}
    e0, e1 = edges(s["nb"]), edges(r["nb"])
    lig = set(ctx.ligands)
    cations = {e for e in ctx.symbols if ctx.charges.get(e, 0) > 0}
    detached, migrated = [], []
    for i, sym in enumerate(ctx.symbols):
        if sym not in lig:
            continue
        c0 = {j for j in s["nb"][i] if ctx.symbols[j] in cations}
        c1 = {j for j in r["nb"][i] if ctx.symbols[j] in cations}
        if not c1:
            detached.append(i)
        elif c1 != c0:
            migrated.append(i)

    all_lengths = [d for g in r["bonds"].values() for d in g["core"] + g["shell"]]
    all_lengths += [d for g in s["bonds"].values() for d in g["core"] + g["shell"]]
    lo = np.floor(min(all_lengths) / HIST_STEP) * HIST_STEP if all_lengths else 2.0
    hi = np.ceil(max(all_lengths) / HIST_STEP) * HIST_STEP if all_lengths else 3.5

    def bond_block(a):
        return {pair: {kind: {**_stats(v), "hist": _hist(v, lo, hi)} for kind, v in g.items()}
                for pair, g in sorted(a["bonds"].items())}

    core_strain = None
    native_pairs = [p for p in r["bonds"] if all(e in set(ctx.native) for e in p.split("-"))]
    core_native = [d for p in native_pairs for d in r["bonds"][p]["core"]]
    if bulk_bond and core_native:
        core_strain = 100.0 * (float(np.mean(core_native)) - bulk_bond) / bulk_bond

    counts = Counter(r["role"])
    return {
        "summary": {
            "topology_preserved": e0 == e1,
            "bonds_broken": len(e0 - e1),
            "bonds_formed": len(e1 - e0),
            "ligands_detached": len(detached),
            "ligands_migrated": len(migrated),
            "n_core": counts.get("core", 0),
            "n_surface": counts.get("surface", 0),
            "n_ligand": counts.get("ligand", 0),
            "ligand_mu": r["ligand_mu"],
            "ligand_mu_start": s["ligand_mu"],
            "bulk_bond_A": bulk_bond,
            "core_strain_pct": core_strain,
        },
        "bulk_cn": bulk_cn,
        "cutoffs_A": {f"{a}-{b}": c for (a, b), c in cut.items() if a < b},
        "cif": str(ctx.cif) if ctx.cif else None,
        "detached": detached,
        "migrated": migrated,
        "role": r["role"],
        "cn_hist": {"start": s["cn_hist"], "relaxed": r["cn_hist"]},
        "bonds": {"start": bond_block(s), "relaxed": bond_block(r)},
    }
