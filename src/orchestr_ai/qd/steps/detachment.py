# src/orchestr_ai/qd/steps/detachment.py
"""
Z-type ligand (MX_q) desorption by a beam search with local updates.

Units.  MX_q leaves preferably as one molecular unit: a surface cation M
with q bonded ligands X gives units of its own X only.  A cation with fewer
bonded X (e.g. CdCl with the second Cl bridging elsewhere) is completed by
the closest other X in space, whatever the distance (flagged non-molecular).

Search.  R_k is a relaxed structure after k removals and U(R_k) the units
enumerated on it.  Removing u costs

    dE_k(u) = E[relax(R_k - u)] + E(MX_q) - E(R_k)

The search keeps the BEAM best structures per level,

    S_(k+1) = best_BEAM { relax(R - u) : R in S_k, u in U(R) }

(deduplicated by fingerprint), so the units are re-defined on the relaxed,
reconstructed surface after every removal.  Local update: a removal only
perturbs its surroundings, so for a unit farther than r_c = max(LOCAL_R,
LOCAL_FRAC x the dot's span) from the last removed cation dE_(k+1)(u) is
estimated by dE_k(u); every other unit is
relaxed (one relaxation per symmetry class).  Candidates are taken lazily:
an estimated one is relaxed before it can be chosen, and estimated units
within REFRESH_WINDOW of the best relaxed candidate are relaxed anyway (stale
estimates are typically too high: a removal can make a distant unit much
cheaper), so every committed step is a real MACE relaxation.

Thermodynamics.  Along the best path, harmonic Hessians give the vibrational
free-energy change of each step, and with MX_q in solution at
mu = G°_MXq(T, 1 M) + dmu (dmu = kT ln(c / 1 M))

    dG_k(T) = dE_k - E(MX_q) + dF_vib,k(T) + G°_MXq(T)

The equilibrium number of removed units <k>(T, dmu) is the Boltzmann average
over every distinct relaxed configuration the search evaluated, each counted
sigma_full / sigma_conf times (its labelled multiplicity); at high coverage it
covers the states near the best paths only.  dG_k here holds the vibrational
change only; qdprops.solution adds the change in translation, rotation and
symmetry number of the dot (ideal-solute G of every dot_k) and solvation.
"""
from __future__ import annotations

import gc
import itertools
import json
import multiprocessing

import numpy as np

from ..records import fingerprint

from ..engines import mace_calculator, mace_hessian
from ..references import binary_units, reference_set, rotational_symmetry_number
from .hessian import _fd_hessian, thermo, vibrations
from .structure import _bonds, _cutoffs

TRAIN_FMAX = 0.02        # eV/Å, relaxations of candidate products
TRAIN_STEPS = 1000
TIE_TOL = 0.05           # Å, completing ligands this close to the closest one are equivalent choices
BEAM = 2                 # structures kept per level
LOCAL_R = 7.0            # Å, minimum radius around the last removal within which units are re-evaluated ...
LOCAL_FRAC = 0.6         # ... or this fraction of the dot's largest native-atom distance, if larger
                         # (in small dots a removal relaxes the whole cluster)
REFRESH_WINDOW = 1.0     # eV, estimated units within this of the best relaxed candidate are relaxed too
                         # (a removal can make a distant unit much cheaper; stale estimates are too high)
TASKS_PER_WORKER = 15    # GPU tasks per worker process before it is replaced
KB_EV = 8.617333262e-5


# --------------------------------------------------------------------------
# MACE relaxations: GPU worker process and checkpoint cache
# --------------------------------------------------------------------------

def _relax(symbols, pts, settings, fmax=TRAIN_FMAX, steps=TRAIN_STEPS):
    from ase import Atoms
    from ase.optimize import BFGS, FIRE
    atoms = Atoms(list(symbols), positions=np.asarray(pts, float))
    atoms.calc = mace_calculator(settings.head, settings.model, settings.device, settings.dtype)
    ok = bool(BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps))
    if not ok:
        ok = bool(FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps))
    e = float(atoms.get_potential_energy())
    atoms.calc = None
    return atoms, e, ok


def _relax_task(args):
    symbols, pts, settings = args
    atoms, e, ok = _relax(symbols, pts, settings)
    return atoms.get_positions(), e, ok


def _frequencies_local(symbols, pts, settings) -> np.ndarray:
    from ase import Atoms
    atoms = Atoms(list(symbols), positions=np.asarray(pts, float))
    atoms.calc = mace_calculator(settings.head, settings.model, settings.device, settings.dtype)
    method = settings.hessian
    if method == "auto":
        method = "analytic" if len(atoms) <= settings.analytic_max_atoms else "fd"
    h = mace_hessian(atoms, atoms.calc) if method == "analytic" else _fd_hessian(atoms, settings.fd_step)
    freqs, _, _ = vibrations(h, atoms.get_positions(), atoms.get_masses())
    return freqs


def _frequencies_task(args):
    return _frequencies_local(*args)


def _free_device_memory(settings) -> None:
    gc.collect()
    from ..engines import resolve_device
    if resolve_device(settings.device) == "mps":
        import torch
        torch.mps.empty_cache()


class _Worker:
    """
    On the Apple GPU every new input shape leaves compiled Metal kernels behind
    (the driver's 'other allocations' grow by ~1 GB per relaxation of a 150-atom
    dot and are not released by torch.mps.empty_cache), so relaxations run in a
    spawned worker process that is replaced every TASKS_PER_WORKER tasks.
    """

    def __init__(self):
        self.pool = None

    def _pool(self):
        if self.pool is None:
            self.pool = multiprocessing.get_context("spawn").Pool(1, maxtasksperchild=TASKS_PER_WORKER)
        return self.pool

    def relax(self, symbols, pts, settings):
        return self._pool().apply(_relax_task, ((list(symbols), np.asarray(pts, float), settings),))

    def frequencies(self, symbols, pts, settings):
        return self._pool().apply(_frequencies_task, ((list(symbols), np.asarray(pts, float), settings),))

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


def geometry_key(symbols, pts) -> str:
    """Ordering-sensitive hash of an input geometry (symbols, coordinates to 1e-4 Å)."""
    import hashlib
    blob = ";".join(symbols) + "|" + np.round(np.asarray(pts, float), 4).tobytes().hex()
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


class RelaxCache:
    """
    Relaxed configurations and Hessians, checkpointed to props/detach_cache.json so an
    interrupted run resumes.  Keyed by the exact input geometry (ordered atoms), never
    by the rotation/permutation-invariant fingerprint: a fingerprint hit could return an
    equivalent structure with its atoms in another order.
    """

    def __init__(self, ctx):
        from ..engines import resolve_device
        self.path = ctx.props / "detach_cache.json"
        s = ctx.settings
        self.tag = f"v2|{s.head}|{s.model}|{s.device}|{s.dtype}|{TRAIN_FMAX}|{TRAIN_STEPS}"
        self.data = json.loads(self.path.read_text()) if self.path.is_file() else {}
        self.worker = _Worker() if resolve_device(s.device) == "mps" else None
        self.n_new = 0

    def relax(self, symbols, pts, settings):
        from ase import Atoms
        key = f"{self.tag}|{geometry_key(symbols, pts)}"
        hit = self.data.get(key)
        if hit is None:
            if self.worker is not None:
                pos, e, ok = self.worker.relax(symbols, pts, settings)
            else:
                atoms, e, ok = _relax(symbols, pts, settings)
                pos = atoms.get_positions()
            hit = {"energy": e, "converged": ok, "positions": np.asarray(pos).round(6).tolist()}
            self.data[key] = hit
            self.path.write_text(json.dumps(self.data))
            self.n_new += 1
            _free_device_memory(settings)
        atoms = Atoms(list(symbols), positions=np.asarray(hit["positions"], float))
        return atoms, float(hit["energy"]), bool(hit["converged"])

    def frequencies(self, symbols, pts, settings):
        key = f"freq|{self.tag}|{settings.hessian}|{geometry_key(symbols, pts)}"
        if key in self.data:
            return np.asarray(self.data[key])
        if self.worker is not None:
            freqs = np.asarray(self.worker.frequencies(symbols, pts, settings))
        else:
            freqs = _frequencies_local(symbols, pts, settings)
            _free_device_memory(settings)
        self.data[key] = np.asarray(freqs).round(4).tolist()
        self.path.write_text(json.dumps(self.data))
        return freqs

    def close(self):
        if self.worker is not None:
            self.worker.close()


# --------------------------------------------------------------------------
# Compact units
# --------------------------------------------------------------------------

def enumerate_units(symbols, pts, charges, native, bulk_bond, q):
    """
    MX_q units, molecular ones preferred: a cation M with at least q bonded
    ligands X gives units of its own X only (MX_q leaving as one molecule);
    a cation with fewer is completed by the closest other X in space (ties
    within TIE_TOL kept, so symmetry-equivalent choices are all present).
    Dicts with cation, ligands, atoms and descriptors.
    """
    nat = set(native)
    p = np.asarray(pts, float)
    nb = _bonds(list(symbols), p, _cutoffs(symbols, charges, native, bulk_bond))
    ligs = [i for i, s in enumerate(symbols) if s not in nat]
    lig_set = set(ligs)
    cats = [i for i, s in enumerate(symbols) if charges.get(s, 0) > 0]
    seen, units = set(), []
    for c in cats:
        own = sorted(j for j in nb[c] if j in lig_set)
        if not own:
            continue
        if len(own) >= q:
            combos = list(itertools.combinations(own, q))
        else:
            need = q - len(own)
            d = {x: float(np.linalg.norm(p[x] - p[c])) for x in ligs if x not in own}
            if len(d) < need:
                continue
            others = sorted(d, key=d.get)
            cut = d[others[need - 1]] + TIE_TOL
            pool = [x for x in others if d[x] <= cut]
            combos = [tuple(sorted(own + list(e))) for e in itertools.combinations(pool, need)]
        for lig in combos:
            key = (c, tuple(sorted(lig)))
            if key in seen:
                continue
            seen.add(key)
            units.append({"cation": c, "ligands": list(key[1]), "atoms": frozenset((c, *key[1])),
                          "molecular": len(own) >= q,
                          "cation_native_cn": sum(1 for j in nb[c] if j not in lig_set),
                          "cation_ligands": len(own),
                          "ligand_mu": [sum(1 for k in nb[x] if charges.get(symbols[k], 0) > 0) for x in key[1]],
                          "M_X_A": [round(float(np.linalg.norm(p[x] - p[c])), 3) for x in key[1]]})
    return units


def _product(symbols, pts, remove):
    keep = [i for i in range(len(symbols)) if i not in remove]
    return [symbols[i] for i in keep], np.asarray(pts, float)[keep]


def site_locations(start_symbols, start_pts, native, indices):
    """facet / edge / vertex of atoms on the ideal (builder) geometry, from the outer
    native layers along <100> and <111>; a direction whose outer layer has fewer than
    three atoms is a corner."""
    p = np.asarray(start_pts, float)
    nat = [i for i, s in enumerate(start_symbols) if s in set(native)]
    dirs = [np.array(v, float) for v in itertools.product((1, -1), repeat=3)] + \
           [np.array(v, float) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    planes = []
    for v in dirs:
        n = v / np.linalg.norm(v)
        proj = p[nat] @ n
        layer = {nat[k] for k in np.where(proj > proj.max() - 0.1)[0]}
        if np.count_nonzero(v) == 1:
            fam = "{100}"
        else:
            cat = sum(1 for i in layer if start_symbols[i] == native[0])
            fam = "{111}" if cat >= len(layer) - cat else "{-1-1-1}"
        planes.append((fam, len(layer) >= 3, layer))
    out = {}
    for i in indices:
        on = [(f, ok) for f, ok, layer in planes if i in layer]
        fac = sorted({f for f, ok in on if ok})
        corner = any(not ok for _f, ok in on)
        kind = "vertex" if corner or len(fac) >= 3 else {0: "interior", 1: "facet", 2: "edge"}[len(fac)]
        out[i] = (kind, fac)
    return out


# --------------------------------------------------------------------------
# Beam search with local updates
# --------------------------------------------------------------------------

class State:
    """A relaxed structure after k removals; ids map atoms back to the full dot."""

    def __init__(self, sym, pts, ids, energy, path, parent=None, last_center=None, parent_table=None):
        self.sym, self.pts, self.ids = list(sym), np.asarray(pts, float), list(ids)
        self.energy = energy            # E(state) - E(full dot), eV
        self.path = path                # [(cation id, ligand ids), ...] in removal order
        self.parent = parent
        self.last_center = last_center
        self.parent_table = parent_table or {}
        self.table = {}                 # unit key -> dict(J, exact, rep, atoms_local, unit)
        self.fp = fingerprint(self.sym, self.pts)

    @property
    def k(self):
        return len(self.path)


def _key(state, unit):
    return (state.ids[unit["cation"]], tuple(sorted(state.ids[x] for x in unit["ligands"])))


def _add_configuration(ensemble, atoms, k, energy):
    """Distinct relaxed configurations (keyed by the relaxed fingerprint), with their
    rotational symmetry numbers for the configurational partition function."""
    sym, pts = atoms.get_chemical_symbols(), atoms.get_positions()
    fp = fingerprint(sym, pts)
    if fp not in ensemble or energy < ensemble[fp][1]:
        ensemble[fp] = (k, energy, rotational_symmetry_number(sym, pts))


def evaluate(state, units, cache, settings, e_full, ensemble, stats, local_r=LOCAL_R):
    """dE table of the state's units: relaxed near the last removal (one per
    symmetry class), estimated from the parent elsewhere."""
    exact_groups = {}
    for un in units:
        key = _key(state, un)
        near = state.last_center is None or \
            float(np.linalg.norm(state.pts[un["cation"]] - state.last_center)) <= local_r
        if not near and key in state.parent_table:
            state.table[key] = {"J": state.parent_table[key], "exact": False, "unit": un}
            continue
        fp = fingerprint(*_product(state.sym, state.pts, un["atoms"]))
        exact_groups.setdefault(fp, []).append((key, un))
    for fp, members in exact_groups.items():
        key0, un0 = members[0]
        atoms, e, ok = cache.relax(*_product(state.sym, state.pts, un0["atoms"]), settings)
        stats["relaxations"] += 1
        J = e - e_full - state.energy
        for key, un in members:
            state.table[key] = {"J": J, "exact": True, "rep": key0, "unit": un, "converged": ok,
                                "multiplicity": len(members)}
        state.table[key0]["product"] = atoms
        _add_configuration(ensemble, atoms, state.k + 1, state.energy + J)


def relax_estimate(state, key, cache, settings, e_full, ensemble, stats):
    entry = state.table[key]
    un = entry["unit"]
    sym, pts = _product(state.sym, state.pts, un["atoms"])
    atoms, e, ok = cache.relax(sym, pts, settings)
    stats["relaxations"] += 1
    stats["lazy"] += 1
    entry.update(J=e - e_full - state.energy, exact=True, rep=key, product=atoms, converged=ok, multiplicity=1)
    _add_configuration(ensemble, atoms, state.k + 1, state.energy + entry["J"])


def expand(beam, cache, settings, e_full, ensemble, stats, ctx, bulk_bond, q, local_r=LOCAL_R):
    """Next level: the BEAM best children of the current beam (lazy, deduplicated)."""
    import heapq
    heap = []
    for si, st in enumerate(beam):
        units = enumerate_units(st.sym, st.pts, ctx.charges, ctx.native, bulk_bond, q)
        evaluate(st, units, cache, settings, e_full, ensemble, stats, local_r)
        for key, entry in st.table.items():
            heapq.heappush(heap, (st.energy + entry["J"], si, key))
    # Refresh: relax estimated candidates close to the best relaxed one, per beam state.
    for si, st in enumerate(beam):
        exact = [v["J"] for v in st.table.values() if v["exact"]]
        if not exact:
            continue
        best = min(exact)
        for key, entry in list(st.table.items()):
            if not entry["exact"] and entry["J"] <= best + REFRESH_WINDOW:
                relax_estimate(st, key, cache, settings, e_full, ensemble, stats)
                heapq.heappush(heap, (st.energy + st.table[key]["J"], si, key))
    children, seen = [], set()
    while heap and len(children) < BEAM:
        e_child, si, key = heapq.heappop(heap)
        st = beam[si]
        entry = st.table[key]
        if entry["exact"] and abs(e_child - (st.energy + entry["J"])) > 1e-9:
            continue                     # stale entry: the unit was relaxed since it was pushed
        if not entry["exact"]:
            relax_estimate(st, key, cache, settings, e_full, ensemble, stats)
            heapq.heappush(heap, (st.energy + st.table[key]["J"], si, key))
            continue
        rep = st.table[entry["rep"]]
        un = rep["unit"]
        removed = set(un["atoms"])
        keep = [i for i in range(len(st.sym)) if i not in removed]
        atoms = rep["product"]
        child = State(atoms.get_chemical_symbols(), atoms.get_positions(), [st.ids[i] for i in keep],
                      st.energy + rep["J"], st.path + [entry["rep"]], parent=st,
                      last_center=st.pts[un["cation"]].copy(),
                      parent_table={k_: v["J"] for k_, v in st.table.items()
                                    if not (set((k_[0], *k_[1])) & {st.ids[i] for i in removed})})
        child.step = {"J_eV": rep["J"], "unit": entry["rep"], "molecular": un["molecular"],
                      "cation_native_cn": un["cation_native_cn"], "cation_ligands": un["cation_ligands"],
                      "ligand_mu": un["ligand_mu"], "M_X_A": un["M_X_A"],
                      "n_candidates": len(st.table),
                      "n_exact": sum(1 for v in st.table.values() if v["exact"])}
        if child.fp in seen:
            continue
        seen.add(child.fp)
        children.append(child)
    return children


# --------------------------------------------------------------------------
# Step
# --------------------------------------------------------------------------

def _write_xyz(path, atoms_sym, atoms_pts, comment):
    p = np.asarray(atoms_pts, float)
    p = p - p.mean(axis=0)
    lines = [str(len(atoms_sym)), comment]
    lines += [f"{x:2s} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}" for x, c in zip(atoms_sym, p)]
    path.write_text("\n".join(lines) + "\n")


def run(ctx) -> dict:
    u = binary_units(ctx.symbols, ctx.charges, ctx.native)
    if u is None or not u["m_MXq"]:
        return {"summary": {"skipped": "no Z-type MX_q units"}}
    s = ctx.settings
    refs = reference_set(ctx, u)
    mx = refs["MXq_monomer"]
    q, m = u["q"], u["m_MXq"]
    T = s.temperatures
    i300 = T.index(300.0) if 300.0 in T else None
    unit = f"{u['M']}{u['X']}{q if q > 1 else ''}"
    bulk_bond = ctx.results["structure"]["summary"].get("bulk_bond_A")
    sym0, pts0 = ctx.relaxed()
    e_full = ctx.results["relax"]["summary"]["energy_eV"]
    f_full = np.asarray(thermo(np.asarray(ctx.results["hessian"]["frequencies_cm1"]), T)["F_vib_eV"])
    g_mx = np.asarray(mx["G_1M_eV"], float)        # MX_q at 1 M: vibrations, rotation, translation
    thermo_ok = len(sym0) <= s.detach_thermo_max_atoms
    from dataclasses import replace
    method = s.hessian if s.hessian != "auto" else ("analytic" if len(sym0) <= s.analytic_max_atoms else "fd")
    s_hess = replace(s, hessian=method)          # one Hessian method for the full dot and the whole path
    cache = RelaxCache(ctx)
    stats = {"relaxations": 0, "lazy": 0}
    ensemble = {}                                    # relaxed fingerprint -> (k, E - E_full, symmetry number)

    nat_pts = np.asarray(pts0, float)[[i for i, x in enumerate(sym0) if x in set(ctx.native)]]
    span = float(np.sqrt(((nat_pts[:, None] - nat_pts[None]) ** 2).sum(-1).max()))
    local_r = max(LOCAL_R, LOCAL_FRAC * span)
    root = State(sym0, pts0, range(len(sym0)), 0.0, [])
    levels = [[root]]
    first_table = None
    while levels[-1] and levels[-1][0].k < m:
        children = expand(levels[-1], cache, s, e_full, ensemble, stats, ctx, bulk_bond, q, local_r)
        if first_table is None:
            first_table = root.table
        if not children:
            break
        levels.append(children)

    # Best path: the lowest structure of the last level and its ancestors.
    best = min(levels[-1], key=lambda st: st.energy)
    path = []
    st = best
    while st.parent is not None:
        path.append(st)
        st = st.parent
    path.reverse()

    # Harmonic free energies along the best path.
    f_prev = f_full
    ladder = []
    prev_e = 0.0
    for st in path:
        fr = None
        if thermo_ok:
            fr = np.asarray(cache.frequencies(st.sym, st.pts, s_hess))
            f_st = np.asarray(thermo(fr, T)["F_vib_eV"])
            n_imag = int((fr < -10.0).sum())
        else:
            f_st, n_imag = f_prev, None
        dF = f_st - f_prev
        dE = st.energy - prev_e + mx["energy_eV"]
        dG = st.energy - prev_e + dF + g_mx
        ladder.append({"k": st.k, "E_mace_eV": st.energy, "dE_eV": dE, "dG_eV": dG.tolist(),
                       "dG_300K_eV": float(dG[i300]) if i300 is not None else None,
                       "dF_vib_300K_eV": float(dF[i300]) if i300 is not None else None,
                       "dF_vib_eV": dF.tolist(),
                       "n_imaginary": n_imag, **st.step,
                       "frequencies_cm1": fr.round(3).tolist() if fr is not None else None,
                       "symbols": st.sym,
                       "best_at_level_eV": min(x.energy for x in levels[st.k])})
        _write_xyz(ctx.props / f"detach_{st.k}.xyz", st.sym, st.pts,
                   f"best path, {st.k} {unit} removed, E - E_full = {st.energy:.6f} eV")
        prev_e, f_prev = st.energy, f_st
    cache.close()

    # Equilibrium <k>(T, dmu) over every evaluated configuration.
    mu = np.asarray(s.mu_grid, float)
    dF_cum = {0: np.zeros(len(T))}          # vibrational change after k removals (best path)
    for st in ladder:
        dF_cum[st["k"]] = dF_cum[st["k"] - 1] + np.asarray(st["dF_vib_eV"])
    sigma0 = rotational_symmetry_number(sym0, pts0)
    confs = [(0, 0.0, sigma0)] + [v for v in ensemble.values()]
    ks = np.array([c[0] for c in confs], float)
    es = np.array([c[1] for c in confs], float)
    lw = np.log(sigma0 / np.array([c[2] for c in confs], float))   # labelled count of each distinct configuration
    dfs = np.vstack([dF_cum[min(int(k), len(ladder))] for k in ks])
    kmean = np.zeros((len(T), len(mu)))
    for ti, t in enumerate(T):
        kT = KB_EV * t
        base = es + dfs[:, ti] + ks * g_mx[ti]
        lg = lw[:, None] - (base[:, None] + ks[:, None] * mu[None, :]) / kT
        lg -= lg.max(axis=0, keepdims=True)
        p = np.exp(lg)
        kmean[ti] = (p * ks[:, None]).sum(axis=0) / p.sum(axis=0)

    # Site-resolved first step (all symmetry classes of the intact dot).
    sites = []
    if first_table:
        reps = {}
        for key, entry in first_table.items():
            if entry.get("exact") and entry.get("rep") == key:
                reps[key] = entry
        loc = site_locations(ctx.symbols, ctx.start_pts, ctx.native, [k_[0] for k_ in reps])
        for key, entry in sorted(reps.items(), key=lambda kv: kv[1]["J"]):
            un = entry["unit"]
            sites.append({"cation": key[0], "ligands": list(key[1]), "multiplicity": entry["multiplicity"],
                          "J_eV": entry["J"], "dE_eV": entry["J"] + mx["energy_eV"], "molecular": un["molecular"],
                          "cation_native_cn": un["cation_native_cn"], "cation_ligands": un["cation_ligands"],
                          "ligand_mu": un["ligand_mu"], "M_X_A": un["M_X_A"],
                          "site": loc[key[0]][0], "facets": loc[key[0]][1]})

    return {
        "summary": {
            "method": f"beam search (width {BEAM}) with local updates", "unit": unit, "n_units": m,
            "n_steps": len(ladder), "beam": BEAM, "local_radius_A": local_r,
            "n_relaxations": stats["relaxations"], "n_lazy_checks": stats["lazy"],
            "n_relaxations_new": cache.n_new, "n_configurations": len(confs),
            "ensemble": "every relaxed configuration evaluated by the search",
            "dE_eV": [st["dE_eV"] for st in ladder],
            "dG_300K_eV": [st["dG_300K_eV"] for st in ladder],
            "cumulative_dE_eV": [st["E_mace_eV"] + st["k"] * mx["energy_eV"] for st in ladder],
            "stopped": None if len(ladder) == m else "no unit left",
            "thermo": bool(thermo_ok),
        },
        "site_classes": sites,
        "steps": ladder,
        "ensemble_list": [list(v) for v in ensemble.values()],      # [k, E - E_full, rotational symmetry number]
        "sigma_full": sigma0,
        "hessian_method": method,
        "levels": [[{"E_eV": x.energy, "path": [list(map(lambda v: v if isinstance(v, int) else list(v), p_))
                                               for p_ in x.path]} for x in lv] for lv in levels[1:]],
        "map": {"T": T, "dmu_eV": mu.tolist(), "k_mean": kmean.tolist(), "ensemble": "evaluated configurations"},
        "reference": {"MXq_energy_eV": mx["energy_eV"], "MXq_G_1M_eV": mx["G_1M_eV"]},
    }
