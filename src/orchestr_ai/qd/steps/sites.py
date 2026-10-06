# src/orchestr_ai/qd/steps/sites.py
"""
Site-resolved Z-type binding: one free energy per symmetry-distinct surface
unit of the intact dot, for the binding-site map of the report.

For every site class of the detachment step (first removal from the full
shell) the relaxed product dot − unit is taken from the detachment cache and
one GFN2-xTB single point gives its Generalized Born solvation.  The thermal
part (vibrations, rotation, translation) is by default that of the first
step of the desorption path, shifted by each site's own energy:

    G(dot − unit_i) ~ G(dot − unit_1) + E(dot − unit_i) − E(dot − unit_1)

so no Hessian beyond the path's is needed (the map is meant to be
qualitative; vibrational differences between sites are small next to the
spread of dE).  With settings.sites_hessian_max_atoms >= N every product
gets its own MACE Hessian instead.  Validated on Cd68Se55Cl26 (25 sites, exact
spread 0.88 eV at 300 K, eps 2.4, 10 mM): rms 0.04 eV, rank correlation 0.994.
settings.sites_solvation = "frozen" also skips the per-site GFN2-xTB single
points (the dot's charges on the remaining atoms): rms 0.12 eV, rank 0.90.  With the unit in solution

    dG_site(T, eps, c) = G^sol(dot − unit) − G^sol(dot) + mu_MXq(T, eps, c)
    G^sol = E + G_trans+rot+vib(T, 1 M) + (1 − 1/eps) G_GB
    mu_MXq = G^sol_MXq + kT ln(c / 1 M)

evaluated live in the dashboard.  A site is a labelled position, so the
rotational symmetry numbers are left out (sigma = 1 for the dot and every
product): their ratio counts equivalent sites, which the map shows explicitly.

Also stored: the symmetry orbit of every class (which atoms to colour), the
atoms of the dot with their roles, and the faceted shape for the 2D map, the
intersection of the recipe's facet planes (each family expanded with the
crystal's proper rotations, placed at the outermost native atom).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..records import read_xyz_first_frame

from ..engines import XtbRunner, parallel_map, xtb_pool
from ..references import binary_units, ideal_gas_g, reference_set
from ..solution import EXPORT_T
from .detachment import RelaxCache, _product
from .solvation import SMEAR, SMEAR_FALLBACKS, gb_conductor_energy


def _g_label(symbols, pts, energy, freqs):
    """G(T) on EXPORT_T at 1 M with sigma = 1 (a labelled site, not a counted species)."""
    from ase import Atoms
    atoms = Atoms(list(symbols), positions=np.asarray(pts, float))
    return np.asarray(ideal_gas_g(atoms, energy, np.asarray(freqs, float), EXPORT_T, 1))


def _orbits(symbols, pts, classes, tol=0.1):
    """For each class, the atom sets of all symmetry-equivalent units (point group of the relaxed dot)."""
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    n = len(symbols)
    xc = pts - pts.mean(0)
    try:
        ops = PointGroupAnalyzer(Molecule(list(symbols), xc), tolerance=tol).get_symmetry_operations()
    except Exception:
        ops = []
    perms = [np.arange(n)]
    for op in ops:
        y = xc @ op.rotation_matrix.T
        p = np.array([int(np.argmin(np.linalg.norm(xc - y[i], axis=1))) for i in range(n)])
        if np.abs(y - xc[p]).max() < 3 * tol and len(set(p)) == n:
            perms.append(p)
    out = []
    for c in classes:
        atoms = [c["cation"]] + list(c["ligands"])
        orb = sorted({(int(p[atoms[0]]), *sorted(int(p[a]) for a in atoms[1:])) for p in perms})
        out.append([list(o) for o in orb])
    return out


def _parse_hkl(t: str):
    out, i = [], 0
    while i < len(t):
        if t[i] == "-":
            out.append(-int(t[i + 1]))
            i += 2
        else:
            out.append(int(t[i]))
            i += 1
    return out


def facet_shape(ctx, pts_native_c):
    """
    Faceted shape from the recipe's facet families: faces [{label, normal, vertices}] (centred frame).
    Polar families (a family and its inverse both in the recipe, e.g. {111} / {-1-1-1}) are named by
    their termination, as the builder and the detachment step do: a plane whose outer layer of the
    builder geometry is cation-rich takes the label of the recipe's cation-rich entry.  The sign of
    the crystallographic hkl depends on the CIF's origin and is not used for the name.
    """
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from scipy.spatial import HalfspaceIntersection
    facets = ((ctx.record.get("origin") or {}).get("recipe") or {}).get("facets") or []
    if ctx.cif is None or not facets:
        return []
    st = Structure.from_file(str(ctx.cif))
    B = st.lattice.reciprocal_lattice_crystallographic.matrix
    rots = [o.rotation_matrix for o in SpacegroupAnalyzer(st).get_point_group_operations(cartesian=True)
            if np.linalg.det(o.rotation_matrix) > 0]
    nat = [i for i, x in enumerate(ctx.symbols) if x in set(ctx.native)]
    p_start = ctx.start_pts[nat] - ctx.start_pts[nat].mean(0)
    cation = ctx.native[0] if ctx.charges.get(ctx.native[0], 0) > 0 else ctx.native[-1]
    is_cat = np.array([ctx.symbols[i] == cation for i in nat])
    term = {str(f["hkl"]): str(f.get("termination", "")) for f in facets}
    planes = []
    for f in facets:
        n0 = np.array(_parse_hkl(str(f["hkl"]))) @ B
        n0 /= np.linalg.norm(n0)
        seen = set()
        for R in rots:
            nv = R @ n0
            key = tuple(np.round(nv, 5))
            if key in seen:
                continue
            seen.add(key)
            label = str(f["hkl"])
            inverse = "".join(str(-h) for h in _parse_hkl(label))
            if inverse in term and term[inverse] != term[label]:
                proj = p_start @ nv
                layer = proj > proj.max() - 0.5
                cat_rich = is_cat[layer].sum() >= (~is_cat[layer]).sum()
                label = next((h for h in (label, inverse) if ("cation" in term[h]) == cat_rich), label)
            planes.append((f"{{{label}}}", nv, float((pts_native_c @ nv).max())))
    hs = np.array([[*nv, -d] for _l, nv, d in planes])
    try:
        verts = HalfspaceIntersection(hs, np.zeros(3)).intersections
    except Exception:
        return []
    faces = []
    for label, nv, d in planes:
        on = verts[np.abs(verts @ nv - d) < 1e-6]
        if len(on) < 3:
            continue
        on = np.unique(on.round(6), axis=0)
        c = on.mean(0)
        e1 = on[0] - c
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nv, e1)
        order = np.argsort(np.arctan2((on - c) @ e2, (on - c) @ e1))
        faces.append({"label": label, "normal": nv.round(6).tolist(), "vertices": on[order].round(4).tolist()})
    return faces


def run(ctx) -> dict:
    det = ctx.results.get("detachment", {})
    classes = det.get("site_classes") or []
    u = binary_units(ctx.symbols, ctx.charges, ctx.native)
    if not classes or u is None:
        return {"summary": {"skipped": "no site classes"}}
    s = ctx.settings
    sym0, pts0 = ctx.relaxed()
    sym0, pts0 = list(sym0), np.asarray(pts0, float)
    e_full = ctx.results["relax"]["summary"]["energy_eV"]
    refs = reference_set(ctx, u)
    method = det.get("hessian_method") or ("analytic" if len(sym0) <= s.analytic_max_atoms else "fd")
    s_hess = replace(s, hessian=method)          # the detachment step's Hessian method, so its cache hits
    exact = len(sym0) <= min(s.sites_hessian_max_atoms, s.detach_thermo_max_atoms)
    # default: the thermal part of the first path step, shifted by each site's energy
    steps = det.get("steps") or []
    g1 = None
    if not exact and steps and steps[0].get("frequencies_cm1") is not None:
        s1, p1 = read_xyz_first_frame(str(ctx.props / "detach_1.xyz"))
        e1 = e_full + steps[0]["E_mace_eV"]
        g1 = _g_label(s1, p1, e1, steps[0]["frequencies_cm1"]) - e1          # thermal part only
    thermo_ok = exact or g1 is not None
    cache = RelaxCache(ctx)
    frozen = s.sites_solvation == "frozen"
    workers, threads = xtb_pool(1 + (0 if frozen else len(classes)))
    runner = XtbRunner("gfn2", threads=threads)
    gas0, _e, _u = runner.run_series(sym0, pts0, [], base=SMEAR, fallbacks=SMEAR_FALLBACKS)
    gb0 = gb_conductor_energy(sym0, pts0, gas0.charges)
    g0 = _g_label(sym0, pts0, e_full, ctx.results["hessian"]["frequencies_cm1"])
    out, products = [], []
    try:
        removed = [[c["cation"]] + list(c["ligands"]) for c in classes]
        prods = [_product(sym0, pts0, r) for r in removed]
        relaxed = cache.relax_many([(psym, ppts) for psym, ppts in prods], s)   # cached by the detachment search
        for c, r, (psym, _p), (atoms, e_prod, ok) in zip(classes, removed, prods, relaxed):
            pos = atoms.get_positions()
            row = {**c, "relaxed_converged": ok, "E_prod_eV": e_prod}
            if exact:
                fr = np.asarray(cache.frequencies(psym, pos, s_hess))
                row["G_prod_eV"] = _g_label(psym, pos, e_prod, fr).round(6).tolist()
                row["n_imaginary"] = int((fr < -10.0).sum())
            elif g1 is not None:
                row["G_prod_eV"] = (g1 + e_prod).round(6).tolist()
            if frozen:
                # the intact dot's charges on the remaining atoms, re-neutralised (no new SCF)
                keep = [i for i in range(len(sym0)) if i not in set(r)]
                qk = np.asarray(gas0.charges)[keep]
                row["solv_inf_eV"] = gb_conductor_energy(list(psym), pos, qk - qk.sum() / len(qk))
            out.append(row)
            products.append((list(psym), pos))
    finally:
        cache.close()
    if not frozen:
        # per-site GFN2-xTB charges: independent single points, side by side on the job's cores
        series = parallel_map(lambda pp: runner.run_series(pp[0], pp[1], [], base=SMEAR, fallbacks=SMEAR_FALLBACKS),
                              products, workers)
        for row, (psym, pos), (gas, _e, used) in zip(out, products, series):
            row["solv_inf_eV"] = gb_conductor_energy(psym, pos, gas.charges)
            row["smearing"] = " ".join(used) or "none"
    orbits = _orbits(sym0, pts0, classes)
    for row, orb in zip(out, orbits):
        row["orbit"] = orb
    centre = pts0.mean(0)
    native = [i for i, x in enumerate(sym0) if x in set(ctx.native)]
    faces = facet_shape(ctx, pts0[native] - centre)
    role = ctx.results.get("structure", {}).get("role") or ["surface"] * len(sym0)
    return {
        "summary": {"n_classes": len(out), "thermo": thermo_ok,
                    "thermal": "per-site Hessians" if exact else ("first path step" if g1 is not None else "none"),
                    "solvation": "GB on per-site GFN2-xTB charges" if s.sites_solvation != "frozen"
                    else "GB on the intact dot's charges (frozen)",
                    "hessian_method": method,
                    "n_atoms_coloured": len({a for r in out for o in r["orbit"] for a in o}),
                    "dE_eV_range": [min(r["dE_eV"] for r in out), max(r["dE_eV"] for r in out)],
                    "n_new_relaxations": cache.n_new,
                    "xtb_parallel": f"{workers} x {threads} threads"},
        "T": EXPORT_T,
        "dot": {"symbols": sym0, "positions": (pts0 - centre).round(4).tolist(), "role": role,
                "G_label_eV": g0.round(6).tolist(), "solv_inf_eV": gb0},
        "MX_energy_eV": refs["MXq_monomer"]["energy_eV"],
        "classes": out,
        "faces": faces,
    }
