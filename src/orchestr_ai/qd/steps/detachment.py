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
import os
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
POLISH_FMAX = 0.005      # eV/Å: candidates that can enter the beam are re-relaxed to this before ranking.
POLISH_WINDOW = 0.3      # eV above the level's best. A product can pass 0.02 eV/Å in a flat valley
POLISH_STEPS = 1000      # (soft surface / Cl motion) 0.1 eV above its minimum (Cd68, level 5), and
                         # then the beam follows another branch; polishing makes the ranking robust.
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


def _relax_batch(structures, settings, polish=False):
    """[(positions, energy, converged), ...]: all structures relaxed together (batch_relax)."""
    from ..batch_relax import BatchMACE, relax_many
    calc = mace_calculator(settings.head, settings.model, settings.device, settings.dtype)
    kw = {"fmax": POLISH_FMAX, "plateau_fmax": 0.05, "max_steps": POLISH_STEPS} if polish else {}
    res = relax_many(structures, BatchMACE(calc, settings.relax_batch_atoms), **kw)
    return [(r.positions, r.energy, r.converged) for r in res]


def _relax_batch_task(args):
    structures, settings, polish = (*args, False)[:3]
    return _relax_batch(structures, settings, polish)


def _frequencies_local(symbols, pts, settings) -> np.ndarray:
    from ase import Atoms
    atoms = Atoms(list(symbols), positions=np.asarray(pts, float))
    atoms.calc = mace_calculator(settings.head, settings.model, settings.device, settings.dtype)
    method = settings.hessian
    if method == "auto":
        method = "analytic" if len(atoms) <= settings.analytic_max_atoms else "fd"
    h = (mace_hessian(atoms, atoms.calc) if method == "analytic"
         else _fd_hessian(atoms, settings.fd_step, settings.relax_batch_atoms))
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

    def relax_many(self, structures, settings, polish=False):
        return self._pool().apply(_relax_batch_task, (([(list(a), np.asarray(b, float)) for a, b in structures],
                                                       settings, polish),))

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
        self.tag = (f"v2|{s.head}|{s.model}|{s.device}|{s.dtype}|{TRAIN_FMAX}|{TRAIN_STEPS}" if not s.relax_batch_atoms
                    else f"v3|{s.head}|{s.model}|{s.device}|{s.dtype}|{s.relaxer()}")
        self.data = json.loads(self.path.read_text()) if self.path.is_file() else {}
        self.perms = None                # rotations of the intact dot (set by run) for configuration_sigma
        self.worker = _Worker() if resolve_device(s.device) == "mps" else None
        self.n_new = 0
        # Optional random displacement of every candidate's starting geometry (reproducibility
        # tests of the search): seeded by detach_seed and the record directory, so copies of a
        # record in one job take different perturbations.
        self.perturb = float(getattr(s, "detach_perturb_A", 0.0) or 0.0)
        self.perturb_seed = f"{getattr(s, 'detach_seed', 0)}|{ctx.record_dir.name}"
        if self.perturb:
            self.tag += f"|perturb-{self.perturb}-{self.perturb_seed}"

    def _perturbed(self, structures):
        if not self.perturb:
            return structures
        import hashlib
        out = []
        for sy, p in structures:
            h = hashlib.sha256(f"{self.perturb_seed}|{geometry_key(sy, p)}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            p = np.asarray(p, float)
            out.append((sy, p + rng.normal(0.0, self.perturb, p.shape)))
        return out

    def sigma(self, remaining_ids):
        return configuration_sigma(self.perms, remaining_ids) if self.perms is not None else None

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, self.path)             # atomic: a job killed at its time limit keeps the checkpoint

    def relax(self, symbols, pts, settings):
        return self.relax_many([(symbols, pts)], settings)[0]

    def relax_many(self, structures, settings, polish=False):
        """
        [(atoms, energy, converged), ...] for every (symbols, positions): cached ones
        are reused, the others relaxed together in one batch (settings.relax_batch_atoms > 0)
        or one at a time (ASE BFGS), and checkpointed. polish: to POLISH_FMAX.
        """
        from ase import Atoms
        tag = f"{self.tag}|polish-{POLISH_FMAX}" if polish else self.tag
        if not polish:
            structures = self._perturbed(structures)
        keys = [f"{tag}|{geometry_key(sy, p)}" for sy, p in structures]
        todo = {}
        for k, (sy, p) in zip(keys, structures):
            if k not in self.data and k not in todo:
                todo[k] = (list(sy), np.asarray(p, float))
        if todo:
            new = list(todo.values())
            if settings.relax_batch_atoms:
                res = (self.worker.relax_many(new, settings, polish) if self.worker is not None
                       else _relax_batch(new, settings, polish))
            else:
                res = []
                for sy, p in new:
                    if self.worker is not None and not polish:
                        res.append(self.worker.relax(sy, p, settings))
                    else:
                        atoms, e, ok = (_relax(sy, p, settings, fmax=POLISH_FMAX, steps=POLISH_STEPS) if polish
                                        else _relax(sy, p, settings))
                        res.append((atoms.get_positions(), e, ok))
            for k, (pos, e, ok) in zip(todo, res):
                self.data[k] = {"energy": float(e), "converged": bool(ok),
                                "positions": np.asarray(pos).round(6).tolist()}
            self._save()
            self.n_new += len(todo)
            _free_device_memory(settings)
        out = []
        for k, (sy, _p) in zip(keys, structures):
            hit = self.data[k]
            out.append((Atoms(list(sy), positions=np.asarray(hit["positions"], float)),
                        float(hit["energy"]), bool(hit["converged"])))
        return out

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
        self._save()
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


def rotation_permutations(symbols, pts, tol: float = 0.1) -> list:
    """The proper rotations of a structure as atom permutations (index arrays), identity
    first: one point-group analysis per dot (pymatgen takes 2-6 s on 300-500 atoms)."""
    sym, p = list(symbols), np.asarray(pts, float)
    n = len(sym)
    ident = np.arange(n)
    try:
        from pymatgen.core import Molecule
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer
        mol = Molecule(sym, p)
        ops = PointGroupAnalyzer(mol, tolerance=tol).get_symmetry_operations()
        x = p - mol.center_of_mass
    except Exception:
        return [ident]
    out = [ident]
    species = np.array(sym)
    for op in ops:
        r = op.rotation_matrix
        if np.linalg.det(r) < 0 or np.allclose(r, np.eye(3)):
            continue
        y = x @ r.T
        dm = np.linalg.norm(y[:, None] - x[None], axis=-1)
        dm[species[:, None] != species[None]] = np.inf
        j = dm.argmin(1)
        if dm[ident, j].max() < 3 * tol and len(set(j.tolist())) == n:
            out.append(j)
    return out


def configuration_sigma(perms, remaining_ids) -> int:
    """Rotational symmetry number of a configuration left by removals from the intact dot:
    the intact dot's rotations that map its remaining atoms onto themselves."""
    keep = np.zeros(len(perms[0]), bool)
    keep[list(remaining_ids)] = True
    return int(sum(1 for pm in perms if keep[pm[keep]].all()))


def _remaining(state, removed_local) -> list:
    return [state.ids[i] for i in range(len(state.sym)) if i not in removed_local]


def _add_configuration(ensemble, atoms, k, energy, sigma=None):
    """Distinct relaxed configurations (keyed by the relaxed fingerprint), with their
    rotational symmetry numbers for the configurational partition function."""
    sym, pts = atoms.get_chemical_symbols(), atoms.get_positions()
    fp = fingerprint(sym, pts)
    if fp not in ensemble or energy < ensemble[fp][1]:
        ensemble[fp] = (k, energy, sigma if sigma is not None else rotational_symmetry_number(sym, pts))


def plan(state, units, local_r=LOCAL_R):
    """
    The state's dE table: units far from the last removal keep the parent's value as
    an estimate (filled here); the others are grouped by symmetry (product fingerprint),
    one relaxation per group. Returns {fingerprint: [(key, unit), ...]} to relax.
    """
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
    return exact_groups


def evaluate_many(beam, groups, cache, settings, e_full, ensemble, stats):
    """Relax the representative of every group of every beam state in one batch and fill the tables."""
    reqs = [(si, members) for si, g in enumerate(groups) for members in g.values()]
    res = cache.relax_many([_product(beam[si].sym, beam[si].pts, members[0][1]["atoms"]) for si, members in reqs],
                           settings)
    stats["relaxations"] += len(reqs)
    for (si, members), (atoms, e, ok) in zip(reqs, res):
        state = beam[si]
        key0, _un0 = members[0]
        J = e - e_full - state.energy
        for key, un in members:
            state.table[key] = {"J": J, "exact": True, "rep": key0, "unit": un, "converged": ok,
                                "multiplicity": len(members)}
        state.table[key0]["product"] = atoms
        _add_configuration(ensemble, atoms, state.k + 1, state.energy + J,
                           cache.sigma(_remaining(state, members[0][1]["atoms"])))


def relax_estimates(beam, todo, cache, settings, e_full, ensemble, stats):
    """Relax estimated entries [(state index, key), ...] in one batch."""
    if not todo:
        return
    res = cache.relax_many([_product(beam[si].sym, beam[si].pts, beam[si].table[key]["unit"]["atoms"])
                            for si, key in todo], settings)
    stats["relaxations"] += len(todo)
    stats["lazy"] += len(todo)
    for (si, key), (atoms, e, ok) in zip(todo, res):
        state = beam[si]
        entry = state.table[key]
        entry.update(J=e - e_full - state.energy, exact=True, rep=key, product=atoms, converged=ok, multiplicity=1)
        _add_configuration(ensemble, atoms, state.k + 1, state.energy + entry["J"],
                           cache.sigma(_remaining(state, entry["unit"]["atoms"])))


def polish(beam, todo, cache, settings, e_full, ensemble, stats):
    """Re-relax relaxed candidates [(state index, representative key), ...] to POLISH_FMAX (one batch)."""
    if not todo:
        return
    res = cache.relax_many([(beam[si].table[key]["product"].get_chemical_symbols(),
                             beam[si].table[key]["product"].get_positions()) for si, key in todo], settings, polish=True)
    stats["polished"] = stats.get("polished", 0) + len(todo)
    for (si, key), (atoms, e, ok) in zip(todo, res):
        state = beam[si]
        J = e - e_full - state.energy
        for entry in state.table.values():
            if entry.get("rep") == key:
                entry["J"] = J
        state.table[key].update(product=atoms, converged=ok, polished=True)
        _add_configuration(ensemble, atoms, state.k + 1, state.energy + J,
                           cache.sigma(_remaining(state, state.table[key]["unit"]["atoms"])))


def polish_level(beam, cache, settings, e_full, ensemble, stats):
    """Polish every relaxed representative within POLISH_WINDOW of the level's best."""
    reps = [(si, key) for si, st in enumerate(beam) for key, e in st.table.items()
            if e["exact"] and e.get("rep") == key and "product" in e]
    if not reps:
        return
    energy = {c: beam[c[0]].energy + beam[c[0]].table[c[1]]["J"] for c in reps}
    best = min(energy.values())
    polish(beam, [c for c in reps if energy[c] <= best + POLISH_WINDOW and not beam[c[0]].table[c[1]].get("polished")],
           cache, settings, e_full, ensemble, stats)


def relax_estimate(state, key, cache, settings, e_full, ensemble, stats):
    relax_estimates([state], [(0, key)], cache, settings, e_full, ensemble, stats)


def expand(beam, cache, settings, e_full, ensemble, stats, ctx, bulk_bond, q, local_r=LOCAL_R):
    """Next level: the BEAM best children of the current beam (lazy, deduplicated)."""
    import heapq
    heap = []
    # Every candidate of the level (all beam states) is relaxed in one batch.
    groups = [plan(st, enumerate_units(st.sym, st.pts, ctx.charges, ctx.native, bulk_bond, q), local_r)
              for st in beam]
    evaluate_many(beam, groups, cache, settings, e_full, ensemble, stats)
    # Refresh: relax estimated candidates close to the best relaxed one, per beam state (one batch).
    todo = []
    for si, st in enumerate(beam):
        exact = [v["J"] for v in st.table.values() if v["exact"]]
        if not exact:
            continue
        best = min(exact)
        todo += [(si, key) for key, entry in st.table.items()
                 if not entry["exact"] and entry["J"] <= best + REFRESH_WINDOW]
    relax_estimates(beam, todo, cache, settings, e_full, ensemble, stats)
    # Polish the candidates that can enter the beam, then rank.
    polish_level(beam, cache, settings, e_full, ensemble, stats)
    for si, st in enumerate(beam):
        for key, entry in st.table.items():
            heapq.heappush(heap, (st.energy + entry["J"], si, key))
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
        if not rep.get("polished"):      # relaxed after the level was polished (lazy estimate)
            polish(beam, [(si, entry["rep"])], cache, settings, e_full, ensemble, stats)
            for k2, e2 in st.table.items():      # its equivalent units share the new energy
                if e2.get("rep") == entry["rep"]:
                    heapq.heappush(heap, (st.energy + e2["J"], si, k2))
            continue
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
# Symmetric moves (default search)
# --------------------------------------------------------------------------
#
# Every move removes a set of symmetry-equivalent units at once (one per equivalent facet),
# then relaxes: relax -> remove the set -> relax -> ...  The equivalent sets are the orbits of
# the units' cations under the 2-fold rotations about the cube axes of the builder core (D2 in
# a Td or Oh core: one site on each of the four {111} facets). The builder's ligand shell need
# not have the core's symmetry, so a move takes, per unit, the largest such set whose cations
# all carry a unit of the same ligand topology (D2 orbit, else a C2 pair, else the unit alone).
# Equivalent units on opposite facets do not interact (Cd68: four removed at once 33.97 eV,
# the four singly 34.10 eV), and the moves cut the relaxations six- to elevenfold against the
# one-unit beam search (Cd68 82 vs 482, InP 316 645 vs ~7,000) for a path ~0.1 eV per unit
# higher. Moves are ranked by energy per removed unit; a child re-relaxes only the moves within
# SYM_R_LOCAL of a cation it just removed and keeps its parent's dE for the others (re-relaxed
# when within SYM_REFRESH of the best). Along the final path every move of n units is filled
# with n - 1 intermediate states (its first j units removed from the parent at once, relaxed), so the
# ladder, its Hessians and the dot_k structures exist for every k.

SYM_R_LOCAL = 7.0         # Å
SYM_REFRESH = 0.1         # eV per removed unit (0.3 re-relaxed ~2/3 of the estimates of PbS, whose units are within 0.2 eV)
SYM_POLISH_N = 3          # best moves per expansion re-relaxed to POLISH_FMAX


def core_c2_permutations(symbols, start_pts, native) -> list:
    """Atom permutations ({atom: image}) of the identity and the 2-fold rotations about the
    cube axes that map the builder core onto itself; [identity] if there are none."""
    p0 = np.asarray(start_pts, float)
    core = [i for i, x in enumerate(symbols) if x in set(native)]
    ident = {i: i for i in core}
    try:
        from pymatgen.core import Molecule
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer
        mol = Molecule([symbols[i] for i in core], p0[core])
        ops = PointGroupAnalyzer(mol, tolerance=0.05).get_symmetry_operations()
        x = p0[core] - mol.center_of_mass
    except Exception:          # no symmetry analysis: single removals only
        return [ident]
    perms = []
    for op in ops:
        r = np.round(op.rotation_matrix, 6)
        if not (np.allclose(r, np.diag(np.diag(r))) and np.isclose(np.linalg.det(r), 1.0)):
            continue
        dm = np.linalg.norm((x @ r.T)[:, None] - x[None], axis=-1)
        j = dm.argmin(1)
        if dm[np.arange(len(core)), j].max() < 0.15 and len(set(j)) == len(core):
            perms.append({core[a]: core[j[a]] for a in range(len(core))})
    return perms if any(any(k != v for k, v in pm.items()) for pm in perms) else [ident]


def _topology(un):
    return (un["molecular"], un["cation_ligands"], tuple(sorted(un["ligand_mu"])))


def symmetric_moves(state, units, perms) -> list:
    """Candidate moves of a state: lists of units, one per cation of the largest equivalent set."""
    ident = [pm for pm in perms if all(k == v for k, v in pm.items())] or [{}]
    others = [pm for pm in perms if pm not in ident]
    groups = [perms] + [ident + [pm] for pm in others] + [ident]
    by_cat = {}
    for un in units:
        by_cat.setdefault(state.ids[un["cation"]], []).append(un)
    out, seen = [], set()
    for un in units:
        c = state.ids[un["cation"]]
        for g in groups:
            orbit = sorted({pm.get(c, c) for pm in g})
            if any(o not in by_cat for o in orbit):
                continue
            chosen, taken = [], set()
            for o in [c] + [o for o in orbit if o != c]:
                opts = [un] if o == c else [u for u in by_cat[o] if _topology(u) == _topology(un)]
                opts = [u for u in opts if not set(u["ligands"]) & taken]
                if not opts:
                    break
                chosen.append(opts[0])
                taken |= set(opts[0]["ligands"])
            else:
                key = (tuple(orbit), _topology(un))
                if key not in seen:
                    seen.add(key)
                    out.append(chosen)
                break
    return out


def _move_key(state, mv):
    return tuple(sorted(_key(state, un) for un in mv))


def _step_info(state, un, J, n_candidates, n_exact, move_units, intermediate):
    return {"J_eV": J, "unit": _key(state, un), "molecular": un["molecular"],
            "cation_native_cn": un["cation_native_cn"], "cation_ligands": un["cation_ligands"],
            "ligand_mu": un["ligand_mu"], "M_X_A": un["M_X_A"], "n_candidates": n_candidates,
            "n_exact": n_exact, "move_units": move_units, "intermediate": intermediate}


def symmetric_search(ctx, settings, cache, e_full, bulk_bond, q, perms, ensemble, stats):
    """(path with one state per k, beam states per move, first-step table of single units)."""
    s = settings

    def units_of(st):
        return enumerate_units(st.sym, st.pts, ctx.charges, ctx.native, bulk_bond, q)

    def relax(cands, polish=False):
        if not cands:
            return
        if polish:
            res = cache.relax_many([(c["atoms"].get_chemical_symbols(), c["atoms"].get_positions()) for c in cands],
                                   s, polish=True)
            stats["polished"] = stats.get("polished", 0) + len(cands)
        else:
            for c in cands:
                c["product"] = _product(c["st"].sym, c["st"].pts, set().union(*(un["atoms"] for un in c["mv"])))
            res = cache.relax_many([c["product"] for c in cands], s)
            stats["relaxations"] += len(cands)
        for c, (atoms, e, ok) in zip(cands, res):
            c.update(atoms=atoms, exact=True, converged=ok, dE=e - e_full - c["st"].energy)
            _add_configuration(ensemble, atoms, c["st"].k + len(c["mv"]), c["st"].energy + c["dE"],
                               cache.sigma(_remaining(c["st"], set().union(*(un["atoms"] for un in c["mv"])))))

    root = State(ctx.relaxed()[0], ctx.relaxed()[1], range(len(ctx.symbols)), 0.0, [])
    root.table, root.last_removed = {}, None
    # Site classes: one single unit per equivalent set of the intact dot.
    root_moves = symmetric_moves(root, units_of(root), perms)
    singles = [{"st": root, "mv": [mv[0]], "n": len(mv)} for mv in root_moves]
    relax(singles)
    first_table = {}
    for c in singles:
        un = c["mv"][0]
        key = _key(root, un)
        first_table[key] = {"J": c["dE"], "exact": True, "rep": key, "unit": un, "multiplicity": c["n"],
                            "converged": c["converged"], "product": c["atoms"]}

    beam, levels = [root], []
    while True:
        cands = []
        for st in beam:
            for mv in symmetric_moves(st, units_of(st), perms):
                key = _move_key(st, mv)
                near = st.last_removed is None or any(
                    float(np.linalg.norm(st.pts[un["cation"]] - c)) <= SYM_R_LOCAL for un in mv for c in st.last_removed)
                cands.append({"st": st, "mv": mv, "key": key, "exact": False,
                              "dE": None if near or key not in st.table else st.table[key]})
        if not cands:
            break
        relax([c for c in cands if c["dE"] is None])
        exact = [c for c in cands if c["exact"]]
        best = min(c["dE"] / len(c["mv"]) for c in (exact or cands))
        relax([c for c in cands if not c["exact"] and c["dE"] / len(c["mv"]) <= best + SYM_REFRESH])
        stats["lazy"] += sum(1 for c in cands if not c["exact"])
        exact = sorted((c for c in cands if c["exact"]), key=lambda c: c["dE"] / len(c["mv"]))
        relax(exact[:SYM_POLISH_N], polish=True)
        exact.sort(key=lambda c: c["dE"] / len(c["mv"]))
        n_per = {}
        for c in cands:
            n_per.setdefault(id(c["st"]), [0, 0])
            n_per[id(c["st"])][0] += 1
            n_per[id(c["st"])][1] += int(c["exact"])
        children, seen = [], set()
        for c in exact:
            st, mv, atoms = c["st"], c["mv"], c["atoms"]
            removed = set().union(*(un["atoms"] for un in mv))
            keep = [i for i in range(len(st.sym)) if i not in removed]
            child = State(atoms.get_chemical_symbols(), atoms.get_positions(), [st.ids[i] for i in keep],
                          st.energy + c["dE"], st.path + [_key(st, un) for un in mv], parent=st)
            if child.fp in seen:
                continue
            seen.add(child.fp)
            gone = {st.ids[i] for i in removed}
            child.table = {cc["key"]: cc["dE"] for cc in cands if cc["st"] is st and cc["dE"] is not None
                           and not {x for k_ in cc["key"] for x in (k_[0], *k_[1])} & gone}
            child.last_removed = [st.pts[un["cation"]].copy() for un in mv]
            child.move = mv
            child.move_stats = tuple(n_per[id(st)])
            children.append(child)
            if len(children) == BEAM:
                break
        levels.append(children)
        beam = children
    final = min(beam, key=lambda x: (-x.k, x.energy))

    # One state per k: state k + j of a move removes the move's first j units from its parent at
    # once and relaxes, as the move itself does for all n (equivalent units are independent).
    moves_path, st = [], final
    while st.parent is not None:
        moves_path.append(st)
        st = st.parent
    moves_path.reverse()
    path = []
    for ms in moves_path:
        parent, mv = ms.parent, ms.move
        nc, ne = ms.move_stats
        partial = [mv[:j] for j in range(1, len(mv))]
        res = cache.relax_many([_product(parent.sym, parent.pts, set().union(*(un["atoms"] for un in sub)))
                                for sub in partial], s)
        res = cache.relax_many([(a.get_chemical_symbols(), a.get_positions()) for a, _e, _ok in res], s, polish=True)
        stats["relaxations"] += len(partial)
        stats["intermediate"] = stats.get("intermediate", 0) + len(partial)
        cur = parent
        for sub, (atoms, e, _ok) in zip(partial, res):
            removed = set().union(*(un["atoms"] for un in sub))
            keep = [i for i in range(len(parent.sym)) if i not in removed]
            nxt = State(atoms.get_chemical_symbols(), atoms.get_positions(), [parent.ids[i] for i in keep],
                        e - e_full, parent.path + [_key(parent, un) for un in sub], parent=cur)
            nxt.step = _step_info(parent, sub[-1], nxt.energy - cur.energy, nc, ne, len(mv), True)
            _add_configuration(ensemble, atoms, nxt.k, nxt.energy, cache.sigma(nxt.ids))
            path.append(nxt)
            cur = nxt
        ms.step = _step_info(parent, mv[-1], ms.energy - cur.energy, nc, ne, len(mv), False)
        path.append(ms)
    return path, levels, first_table


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
    if len(ctx.symbols) > s.detach_max_atoms:
        return {"summary": {"skipped": f"{len(ctx.symbols)} atoms > detach_max_atoms = {s.detach_max_atoms}"}}
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
    cache.perms = rotation_permutations(sym0, pts0)
    stats = {"relaxations": 0, "lazy": 0}
    ensemble = {}                                    # relaxed fingerprint -> (k, E - E_full, symmetry number)

    nat_pts = np.asarray(pts0, float)[[i for i, x in enumerate(sym0) if x in set(ctx.native)]]
    span = float(np.sqrt(((nat_pts[:, None] - nat_pts[None]) ** 2).sum(-1).max()))
    n_ops = 1
    if s.detach_search == "symmetric":
        perms = core_c2_permutations(ctx.symbols, ctx.start_pts, ctx.native)
        n_ops = len(perms)
        local_r = SYM_R_LOCAL
        path, levels_out, first_table = symmetric_search(ctx, s, cache, e_full, bulk_bond, q, perms, ensemble, stats)
    else:
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
        levels_out = levels[1:]
        # Best path: the lowest structure of the last level and its ancestors.
        best = min(levels[-1], key=lambda st: st.energy)
        path = []
        st = best
        while st.parent is not None:
            path.append(st)
            st = st.parent
        path.reverse()
    best_at = {}
    for k_, e_, _sg in ensemble.values():
        best_at[k_] = min(best_at.get(k_, e_), e_)

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
                       "best_at_level_eV": min(best_at.get(st.k, st.energy), st.energy)})
        _write_xyz(ctx.props / f"detach_{st.k}.xyz", st.sym, st.pts,
                   f"best path, {st.k} {unit} removed, E - E_full = {st.energy:.6f} eV")
        prev_e, f_prev = st.energy, f_st
    cache.close()

    # Equilibrium <k>(T, dmu) over every evaluated configuration.
    mu = np.asarray(s.mu_grid, float)
    dF_cum = {0: np.zeros(len(T))}          # vibrational change after k removals (best path)
    for st in ladder:
        dF_cum[st["k"]] = dF_cum[st["k"] - 1] + np.asarray(st["dF_vib_eV"])
    sigma0 = len(cache.perms)
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
            "method": (f"symmetric moves (width {BEAM}) with local updates" if s.detach_search == "symmetric"
                       else f"beam search (width {BEAM}) with local updates"),
            "search": s.detach_search, "core_c2_operations": n_ops, "unit": unit, "n_units": m,
            "n_steps": len(ladder), "n_moves": len(levels_out), "beam": BEAM, "local_radius_A": local_r,
            "n_intermediate": stats.get("intermediate", 0),
            "n_relaxations": stats["relaxations"], "n_lazy_checks": stats["lazy"],
            "n_polished": stats.get("polished", 0), "polish_fmax_eV_A": POLISH_FMAX,
            "polish_window_eV": POLISH_WINDOW,
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
                                               for p_ in x.path]} for x in lv] for lv in levels_out],
        "map": {"T": T, "dmu_eV": mu.tolist(), "k_mean": kmean.tolist(), "ensemble": "evaluated configurations"},
        "reference": {"MXq_energy_eV": mx["energy_eV"], "MXq_G_1M_eV": mx["G_1M_eV"]},
    }
