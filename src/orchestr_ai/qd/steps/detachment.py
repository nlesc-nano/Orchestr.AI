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
# Quick mode (settings.detach_quick, the default): a qualitative ladder at a fraction of the cost.
# Candidates relaxed to QUICK_FMAX, no polishing, and one Hessian per path (the last state): the
# vibrational free energy of the states in between is linear in k from the intact dot. A MACE-MH-1
# force call costs ~80 ms whatever the size (A100, 149-1946 atoms: per-call overhead), so the
# number of optimiser steps sets the cost.
QUICK_FMAX = 0.05        # eV/Å
TRAIN_STEPS = 1000
TIE_TOL = 0.05           # Å, completing ligands this close to the closest one are equivalent choices
BEAM = 2                 # structures kept per level (beam search)
# Greedy search (the default): the beam search with one structure per level, candidates re-relaxed
# only near the last removal, and a cap on the number of removals. A qualitative ladder at a
# bounded cost: the symmetric moves follow symmetric paths up to 2.2 eV higher (Cd68), the full
# beam search costs ~7,000 relaxations at 316 atoms.
GREEDY_R_LOCAL = 7.0     # Å, units re-relaxed after a removal (the rest keep their previous dE)
GREEDY_REFRESH = 0.3     # eV, estimated units this close to the best relaxed one are relaxed too
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
CHECKPOINT_CALLS = 4     # candidates relaxed together, in model calls' worth of atoms (relax_batch_atoms each),
                         # then checkpointed: 8 candidates of 2,000 atoms, ~50 of 300
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


def _candidate_fmax(settings) -> float:
    return QUICK_FMAX if getattr(settings, "detach_quick", False) else TRAIN_FMAX


def _relax_task(args):
    symbols, pts, settings = args
    atoms, e, ok = _relax(symbols, pts, settings)
    return atoms.get_positions(), e, ok


def _relax_batch(structures, settings, polish=False, snapshots=()):
    """[(positions, energy, converged, frames), ...]: all structures relaxed together (batch_relax);
    frames: snapshots along each relaxation (batch_relax.relax_many(snapshots=...))."""
    from ..batch_relax import BatchMACE, relax_many
    calc = mace_calculator(settings.head, settings.model, settings.device, settings.dtype)
    kw = ({"fmax": POLISH_FMAX, "plateau_fmax": 0.05, "max_steps": POLISH_STEPS} if polish
          else {"fmax": QUICK_FMAX} if getattr(settings, "detach_quick", False) else {})
    res = relax_many(structures, BatchMACE(calc, settings.relax_batch_atoms), snapshots=snapshots, **kw)
    return [(r.positions, r.energy, r.converged, r.frames) for r in res]


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
        # Training frames off the minimum: snapshots along every candidate relaxation, spooled to
        # disk as they come (an interrupted run keeps them) and selected by training_frames().
        self.snapshots = tuple(s.detach_frames_fractions) if getattr(s, "detach_frames", False) else ()
        self.spool = ctx.props / "ensemble" / "detach_spool.extxyz"
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
        are reused, the others relaxed together in checkpointed batches (settings.relax_batch_atoms > 0)
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
        if todo and settings.relax_batch_atoms and self.worker is None:
            # Batches of at most CHECKPOINT_CALLS model calls' worth of atoms, each checkpointed: the first
            # level of a large dot is hundreds of candidates (2 per call at 2,000 atoms), longer than a
            # short-QoS job piece, and an unsaved batch would start over in every piece.
            limit = CHECKPOINT_CALLS * settings.relax_batch_atoms
            keys_left = list(todo)
            while keys_left:
                group, n = [], 0
                while keys_left and (not group or n + len(todo[keys_left[0]][0]) <= limit):
                    k = keys_left.pop(0)
                    group.append(k)
                    n += len(todo[k][0])
                res = _relax_batch([todo[k] for k in group], settings, polish, () if polish else self.snapshots)
                self._store({k: todo[k] for k in group}, res)
            _free_device_memory(settings)
            todo = {}
        if todo:
            new = list(todo.values())
            if settings.relax_batch_atoms:
                res = self.worker.relax_many(new, settings, polish)
            else:
                res = []
                for sy, p in new:
                    if self.worker is not None and not polish:
                        res.append(self.worker.relax(sy, p, settings))
                    else:
                        atoms, e, ok = (_relax(sy, p, settings, fmax=POLISH_FMAX, steps=POLISH_STEPS) if polish
                                        else _relax(sy, p, settings, fmax=_candidate_fmax(settings)))
                        res.append((atoms.get_positions(), e, ok))
            self._store(todo, res)
            _free_device_memory(settings)
        out = []
        for k, (sy, _p) in zip(keys, structures):
            hit = self.data[k]
            out.append((Atoms(list(sy), positions=np.asarray(hit["positions"], float)),
                        float(hit["energy"]), bool(hit["converged"])))
        return out

    def _store(self, todo, res):
        """Cache (and checkpoint) the relaxations `res` of the inputs `todo` (key -> (symbols, positions));
        their snapshots go to the spool."""
        spooled = []
        for k, r in zip(todo, res):
            pos, e, ok = r[:3]
            self.data[k] = {"energy": float(e), "converged": bool(ok),
                            "positions": np.asarray(pos).round(6).tolist()}
            if len(r) > 3 and r[3]:
                sy = todo[k][0]
                final = geometry_key(sy, np.asarray(pos).round(6))
                spooled += [(sy, fr, k, final) for fr in r[3]]
        if spooled:
            self._spool(spooled)
        self._save()
        self.n_new += len(todo)

    def _spool(self, frames):
        from ase import Atoms
        from ase.io import write
        self.spool.parent.mkdir(exist_ok=True)
        out = []
        for sy, fr, start, final in frames:
            a = Atoms(list(sy), positions=np.asarray(fr["positions"], float))
            a.arrays["forces"] = np.asarray(fr["forces"], float)
            a.info.update({"energy": float(fr["energy"]), "fraction": fr["fraction"], "relax_step": fr["step"],
                           "start_key": start.rsplit("|", 1)[-1], "final_key": final})
            out.append(a)
        write(str(self.spool), out, format="extxyz", append=True)

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

    def __init__(self, sym, pts, ids, energy, path, parent=None, last_center=None, parent_table=None, sign=1):
        self.sym, self.pts, self.ids = list(sym), np.asarray(pts, float), list(ids)
        self.sign = sign                # +1: units removed (ladder down), -1: units added (ladder up)
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
    if "add" in unit:
        return unit["key"]
    return (state.ids[unit["cation"]], tuple(sorted(state.ids[x] for x in unit["ligands"])))


def _apply(state, unit):
    """(symbols, positions) of the state with the unit removed (ladder down) or added (ladder up)."""
    if "add" in unit:
        sym, pos = unit["add"]
        return list(state.sym) + list(sym), np.vstack([state.pts, pos])
    return _product(state.sym, state.pts, unit["atoms"])


def _center(state, unit):
    return np.asarray(unit["center"]) if "add" in unit else state.pts[unit["cation"]]


def _unit_sigma(cache, state, unit):
    """Rotational symmetry number of the product. Removals: from the intact dot's rotations (precomputed).
    Additions: 1 when the intact dot has no rotation (an added unit does not create symmetry the dot
    lacks, except by accident), else None (computed from the structure). The point-group analysis
    of every addition product took 81 % of a PbS ladder (17 s per 507-atom configuration)."""
    if "add" in unit:
        return 1 if cache.perms is not None and len(cache.perms) <= 1 else None
    return cache.sigma(_remaining(state, unit["atoms"]))


# --------------------------------------------------------------------------
# Ladder up: sites where one more MX_q can bind
# --------------------------------------------------------------------------
#
# The rules of the builder's Z-type post-treatment (QD_builder z_type_ligand_posttreat), applied
# to the relaxed dot: M takes the empty cation site above an under-coordinated surface anion A
# (the direction of one of A's missing bonds, at the M-A bond length), and its q ligands X take
# free sites around M in the cation's bulk coordination (tetrahedral or octahedral), preferring
# sites that bridge to surface metals, then outward ones. No atom may be closer to a new atom
# than ADD_CLEAR times its bond.

ADD_CLEAR = 0.8
ADD_PARTNER = 1.15       # x the M-X bond: metals an X site would bridge to


def _unit_vec(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _perp_basis(d):
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit_vec(np.cross(d, a))
    return e1, np.cross(d, e1)


def _missing_directions(us, cn):
    """Directions of the missing bonds of a site whose bonds point along the unit vectors `us`
    (bulk coordination cn: 4 tetrahedral, otherwise octahedral)."""
    us = [_unit_vec(u) for u in us]
    if cn >= 6:
        return [-u for u in us]
    if len(us) >= 3:
        return [_unit_vec(-np.sum(us, axis=0))]
    if len(us) == 2:
        b = _unit_vec(-(us[0] + us[1]))
        n = _unit_vec(np.cross(us[0], us[1]))
        half = np.arccos(-1.0 / 3.0) / 2
        return [np.cos(half) * b + s * np.sin(half) * n for s in (1.0, -1.0)]
    if len(us) == 1:
        e1, e2 = _perp_basis(us[0])
        r = np.sqrt(8.0) / 3.0
        return [-us[0] / 3.0 + r * (np.cos(f) * e1 + np.sin(f) * e2) for f in np.deg2rad([0.0, 120.0, 240.0])]
    return []


def _ligand_directions(d, cn, phi):
    """Bond directions of the new M other than the one to its host (-d), rotated by phi about d."""
    e1, e2 = _perp_basis(d)
    if cn >= 6:
        return [d] + [np.cos(phi + f) * e1 + np.sin(phi + f) * e2 for f in np.deg2rad([0.0, 90.0, 180.0, 270.0])]
    r = np.sqrt(8.0) / 3.0
    return [d / 3.0 + r * (np.cos(phi + f) * e1 + np.sin(phi + f) * e2) for f in np.deg2rad([0.0, 120.0, 240.0])]


def enumerate_sites(state, charges, native, bulk_bond, units, d_mx):
    """
    Candidate MX_q additions on `state`: dicts with the atoms to add, their centre (the new M),
    the host anion and a key that stays the same from one ladder state to the next.
    """
    from scipy.spatial import cKDTree
    sym, p = list(state.sym), state.pts
    nb = _bonds(sym, p, _cutoffs(sym, charges, native, bulk_bond))
    nat = set(native)
    cat_n = [e for e in native if charges.get(e, 0) > 0][0]
    an_n = [e for e in native if charges.get(e, 0) < 0][0]
    m, x, q = units["M"], units["X"], units["q"]
    anions = [i for i, s in enumerate(sym) if s == an_n]
    cn_of = {i: [j for j in nb[i] if charges.get(sym[j], 0) > 0] for i in anions}
    cn_bulk = max(len(v) for v in cn_of.values())
    centroid = p.mean(axis=0)
    tree = cKDTree(p)
    d_ma = float(bulk_bond)
    out = []
    for a in anions:
        if len(cn_of[a]) >= cn_bulk:
            continue
        outward = _unit_vec(p[a] - centroid)
        for d in _missing_directions([p[j] - p[a] for j in cn_of[a]], cn_bulk):
            if float(np.dot(d, outward)) < 0.1:
                continue
            mpos = p[a] + d_ma * d
            if any(j != a for j in tree.query_ball_point(mpos, ADD_CLEAR * d_ma)):
                continue
            best = None
            for phi in np.deg2rad(np.arange(0.0, 120.0, 10.0)):
                sites = []
                for e in _ligand_directions(d, cn_bulk, phi):
                    xpos = mpos + d_mx * e
                    if tree.query_ball_point(xpos, ADD_CLEAR * d_mx):
                        continue
                    partners = [j for j in tree.query_ball_point(xpos, ADD_PARTNER * d_mx) if charges.get(sym[j], 0) > 0]
                    sites.append((len(partners), float(np.dot(e, outward)), xpos, partners))
                sites.sort(key=lambda s: (-s[0], -s[1]))
                if len(sites) < q:
                    continue
                score = (sum(s[0] for s in sites[:q]), sum(s[1] for s in sites[:q]))
                if best is None or score > best[0]:
                    best = (score, sites[:q])
            if best is None:
                continue
            xs = best[1]
            partners = sorted({state.ids[j] for s in xs for j in s[3]})
            out.append({"add": ([m] + [x] * q, np.vstack([mpos] + [s[2] for s in xs])), "center": mpos,
                        "host": a, "key": ("+", state.ids[a], tuple(partners)),
                        "bridges": [s[0] for s in xs], "molecular": True})
    return out


def _surface_mx_bond(symbols, pts, charges, native, bulk_bond, units, fallback):
    """Median M-X bond of the dot's ligands (the length of a surface-bound X), or `fallback`."""
    nb = _bonds(list(symbols), np.asarray(pts, float), _cutoffs(symbols, charges, native, bulk_bond))
    d = [float(np.linalg.norm(pts[i] - pts[j])) for i, s in enumerate(symbols) if s == units["M"]
         for j in nb[i] if symbols[j] == units["X"]]
    return float(np.median(d)) if d else fallback


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
            float(np.linalg.norm(_center(state, un) - state.last_center)) <= local_r
        if not near and key in state.parent_table:
            j_par, jr_par = state.parent_table[key]
            state.table[key] = {"J": j_par, "exact": False, "unit": un, "J_parent_rigid": jr_par}
            continue
        fp = fingerprint(*_apply(state, un))
        exact_groups.setdefault(fp, []).append((key, un))
    return exact_groups


def evaluate_many(beam, groups, cache, settings, e_full, ensemble, stats):
    """Relax the representative of every group of every beam state in one batch and fill the tables."""
    reqs = [(si, members) for si, g in enumerate(groups) for members in g.values()]
    res = cache.relax_many([_apply(beam[si], members[0][1]) for si, members in reqs], settings)
    stats["relaxations"] += len(reqs)
    for (si, members), (atoms, e, ok) in zip(reqs, res):
        state = beam[si]
        key0, _un0 = members[0]
        J = e - e_full - state.energy
        for key, un in members:
            state.table[key] = {"J": J, "exact": True, "rep": key0, "unit": un, "converged": ok,
                                "multiplicity": len(members)}
        state.table[key0]["product"] = atoms
        _add_configuration(ensemble, atoms, state.sign * (state.k + 1), state.energy + J,
                           _unit_sigma(cache, state, members[0][1]))


def rigid_level(beam, groups, settings, e_full, stats):
    """
    Unrelaxed single points of every candidate of the level (detach_rigid_refresh): one per group
    to relax and one per estimate, batched. An estimate carried over from the previous level is
    corrected by the change of its unrelaxed energy since then,

        J_k(u) ~ J_(k-1)(u) + [J_k^rigid(u) - J_(k-1)^rigid(u)],

    so a removal or addition that changes a unit beyond the local radius is seen (Cd68: a unit
    7.6 A from the first removal became 0.42 eV cheaper). Returns {(state index, key): J^rigid}.
    """
    from ..batch_relax import BatchMACE
    ev = BatchMACE(mace_calculator(settings.head, settings.model, settings.device, settings.dtype),
                   settings.relax_batch_atoms)
    reqs = []                                     # (state index, [keys sharing the value], unit)
    for si, g in enumerate(groups):
        st = beam[si]
        reqs += [(si, [key for key, _u in members], members[0][1]) for members in g.values()]
        reqs += [(si, [key], e["unit"]) for key, e in st.table.items() if not e["exact"]]
    res = ev([_apply(beam[si], un) for si, _keys, un in reqs]) if reqs else []
    stats["rigid_single_points"] = stats.get("rigid_single_points", 0) + len(reqs)
    rig = {}
    for (si, keys, _un), (e, _f) in zip(reqs, res):
        for key in keys:
            rig[(si, key)] = e - e_full - beam[si].energy
    for si, st in enumerate(beam):
        for key, entry in st.table.items():
            if entry["exact"] or (si, key) not in rig:
                continue
            if entry.get("J_parent_rigid") is not None:
                entry["J"] += rig[(si, key)] - entry["J_parent_rigid"]
                stats["rigid_corrected"] = stats.get("rigid_corrected", 0) + 1
            entry["J_rigid"] = rig[(si, key)]
    return rig


def screen(beam, groups, settings, e_full, stats, rig=None):
    """
    Screening of a level with many candidates (settings.detach_screen): every group's product is
    evaluated as removed, without relaxing it (one batched single point each), and only the
    detach_screen_keep lowest groups and those within detach_screen_window of the lowest are kept
    for relaxation. Returns the kept groups and the screened-out ones [(state index, members,
    J_rigid)], whose estimates are set by estimate_screened once the kept ones are relaxed.
    """
    if not getattr(settings, "detach_screen", False):
        return groups, [], {}
    from ..batch_relax import BatchMACE
    kept_all, deferred, rigid = [], [], {}
    ev = None
    for si, g in enumerate(groups):
        if len(g) <= settings.detach_screen_min:
            kept_all.append(g)
            continue
        if ev is None and rig is None:
            ev = BatchMACE(mace_calculator(settings.head, settings.model, settings.device, settings.dtype),
                           settings.relax_batch_atoms)
        st = beam[si]
        fps = list(g)
        if rig is not None:                       # the level's single points (rigid_level)
            J = {fp: rig[(si, g[fp][0][0])] for fp in fps}
        else:
            res = ev([_apply(st, g[fp][0][1]) for fp in fps])
            stats["screen_single_points"] = stats.get("screen_single_points", 0) + len(fps)
            J = {fp: e - e_full - st.energy for fp, (e, _f) in zip(fps, res)}
        order = sorted(fps, key=J.get)
        within = [fp for fp in order if J[fp] <= J[order[0]] + settings.detach_screen_window]
        # dense levels (InAs In844As715Cl387: ~300 of 557 within 0.5 eV) keep at most detach_screen_max
        keep = set(order[:settings.detach_screen_keep]) | set(within[:settings.detach_screen_max])
        kept_all.append({fp: g[fp] for fp in fps if fp in keep})
        for fp in fps:
            rigid[(si, g[fp][0][0])] = J[fp]
            if fp not in keep:
                deferred.append((si, g[fp], J[fp]))
        stats["screened_out"] = stats.get("screened_out", 0) + len(fps) - len(keep)
    return kept_all, deferred, rigid


def estimate_screened(beam, deferred, rigid, settings):
    """
    Estimates of the screened-out groups: their unrelaxed J minus a high quantile
    (detach_screen_quantile) of the relaxation energies of the state's relaxed candidates, an
    optimistic value, so the refresh and the lazy check relax every one that might compete.
    """
    for si in {d[0] for d in deferred}:
        st = beam[si]
        relax_e = [rigid[(si, key)] - st.table[key]["J"] for (s_, key) in rigid if s_ == si
                   and key in st.table and st.table[key].get("exact")]
        shift = float(np.quantile(relax_e, settings.detach_screen_quantile)) if relax_e else 0.0
        for s_, members, j_rigid in deferred:
            if s_ != si:
                continue
            group = [key for key, _un in members]
            for key, un in members:
                st.table[key] = {"J": j_rigid - shift, "exact": False, "unit": un, "screened": True,
                                 "J_rigid": j_rigid, "group": group}


def relax_estimates(beam, todo, cache, settings, e_full, ensemble, stats):
    """Relax estimated entries [(state index, key), ...] in one batch; a screened group's members
    (symmetry-equivalent units) share the relaxed value of the one relaxed."""
    # one relaxation per screened group
    seen, uniq = set(), []
    for si, key in todo:
        grp = beam[si].table[key].get("group")
        gid = (si, tuple(grp)) if grp else (si, key)
        if gid not in seen:
            seen.add(gid)
            uniq.append((si, key))
    todo = uniq
    if not todo:
        return
    res = cache.relax_many([_apply(beam[si], beam[si].table[key]["unit"]) for si, key in todo], settings)
    stats["relaxations"] += len(todo)
    stats["lazy"] += len(todo)
    for (si, key), (atoms, e, ok) in zip(todo, res):
        state = beam[si]
        entry = state.table[key]
        group = entry.get("group") or [key]
        J = e - e_full - state.energy
        for k2 in group:
            state.table[k2].update(J=J, exact=True, rep=key, converged=ok, multiplicity=len(group))
        entry["product"] = atoms
        _add_configuration(ensemble, atoms, state.sign * (state.k + 1), state.energy + entry["J"],
                           _unit_sigma(cache, state, entry["unit"]))


def polish(beam, todo, cache, settings, e_full, ensemble, stats):
    """Re-relax relaxed candidates [(state index, representative key), ...] to POLISH_FMAX (one batch).
    Quick mode: no re-relaxation, the candidates are only marked as final."""
    if not todo:
        return
    if getattr(settings, "detach_quick", False):
        for si, key in todo:
            beam[si].table[key]["polished"] = True
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
        _add_configuration(ensemble, atoms, state.sign * (state.k + 1), state.energy + J,
                           _unit_sigma(cache, state, state.table[key]["unit"]))


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


def expand(beam, cache, settings, e_full, ensemble, stats, enum, local_r=LOCAL_R, width=BEAM,
           refresh=REFRESH_WINDOW):
    """Next level: the `width` best children of the current beam (lazy, deduplicated); enum(state)
    lists the state's candidate units (enumerate_units: removals, enumerate_sites: additions)."""
    import heapq
    heap = []
    # Every candidate of the level (all beam states) is relaxed in one batch.
    groups = [plan(st, enum(st), local_r) for st in beam]
    rig = rigid_level(beam, groups, settings, e_full, stats) if getattr(settings, "detach_rigid_refresh", False) else None
    groups, deferred, rigid = screen(beam, groups, settings, e_full, stats, rig)
    evaluate_many(beam, groups, cache, settings, e_full, ensemble, stats)
    estimate_screened(beam, deferred, rigid, settings)
    if rig is not None:
        for (si, key), jr in rig.items():
            if key in beam[si].table:
                beam[si].table[key].setdefault("J_rigid", jr)
    # Refresh: relax estimated candidates close to the best relaxed one, per beam state (one batch).
    todo = []
    for si, st in enumerate(beam):
        exact = [v["J"] for v in st.table.values() if v["exact"]]
        if not exact:
            continue
        best = min(exact)
        cand = sorted(((entry["J"], key) for key, entry in st.table.items()
                       if not entry["exact"] and entry["J"] <= best + refresh), key=lambda x: x[0])
        cap = getattr(settings, "detach_refresh_max", 0)
        if cap and len(cand) > cap:                # near-degenerate units: the lowest estimates only
            stats["refresh_capped"] = stats.get("refresh_capped", 0) + len(cand) - cap
            cand = cand[:cap]
        todo += [(si, key) for _j, key in cand]
    relax_estimates(beam, todo, cache, settings, e_full, ensemble, stats)
    # Lazy checks in batches: every estimate below the best relaxed candidate, or within
    # detach_lazy_margin above it, is relaxed, all together, until none is left (one at a time each
    # ran alone on the GPU: 266 on InAs In844As715Cl387). The margin settles near-ties with real
    # relaxations: on Cd68 a unit screened out at step 2 was 3 meV below the chosen one, and the
    # other branch cost 0.55 eV more at step 3.
    margin = float(getattr(settings, "detach_lazy_margin", 0.0) or 0.0)
    while True:
        exact = [st.energy + e["J"] for st in beam for e in st.table.values() if e["exact"]]
        if not exact:
            break
        todo = [(si, key) for si, st in enumerate(beam) for key, e in st.table.items()
                if not e["exact"] and st.energy + e["J"] < min(exact) + margin]
        if not todo:
            break
        stats["lazy_batches"] = stats.get("lazy_batches", 0) + 1
        relax_estimates(beam, todo, cache, settings, e_full, ensemble, stats)
    # Polish the candidates that can enter the beam, then rank.
    polish_level(beam, cache, settings, e_full, ensemble, stats)
    for si, st in enumerate(beam):
        for key, entry in st.table.items():
            heapq.heappush(heap, (st.energy + entry["J"], si, key))
    children, seen = [], set()
    while heap and len(children) < width:
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
        atoms = rep["product"]
        if "add" in un:                  # ladder up: the new atoms get ids after every id used so far
            first = max(st.ids) + 1
            child = State(atoms.get_chemical_symbols(), atoms.get_positions(),
                          st.ids + list(range(first, first + len(un["add"][0]))),
                          st.energy + rep["J"], st.path + [entry["rep"]], parent=st,
                          last_center=np.asarray(un["center"]).copy(),
                          parent_table={k_: (v["J"], v.get("J_rigid")) for k_, v in st.table.items()}, sign=-1)
            child.step = {"J_eV": rep["J"], "unit": entry["rep"], "host": st.ids[un["host"]], "molecular": True,
                          "bridges": un["bridges"], "n_candidates": len(st.table),
                          "n_exact": sum(1 for v in st.table.values() if v["exact"])}
        else:
            removed = set(un["atoms"])
            keep = [i for i in range(len(st.sym)) if i not in removed]
            child = State(atoms.get_chemical_symbols(), atoms.get_positions(), [st.ids[i] for i in keep],
                          st.energy + rep["J"], st.path + [entry["rep"]], parent=st,
                          last_center=st.pts[un["cation"]].copy(),
                          parent_table={k_: (v["J"], v.get("J_rigid")) for k_, v in st.table.items()
                                        if k_[0] != "+" and not (set((k_[0], *k_[1])) & {st.ids[i] for i in removed})})
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

LIGAND_SITE_MERGE = 0.8     # A: candidate X positions of neighbouring metals this close are one (bridging) site


def ligand_sites(symbols, pts, charges, native, bulk_bond, x_sym, d_mx, cn_bulk):
    """
    (x, y, occupied): the places a ligand X can sit on the dot, the ligands, and how many of the
    places hold one. Every surface metal misses bonds to the anion sublattice; along each missing
    bond (bulk coordination cn_bulk) an X would sit at the M-X distance, as the builder places it.
    Positions of neighbouring metals that coincide (within LIGAND_SITE_MERGE) are one site that
    bridges them; the others are terminal. Positions inside the dot (closer than 0.8 x the bond to
    a core atom) are dropped. Each site holds at most one X.
    """
    from scipy.spatial import cKDTree
    sym, p = list(symbols), np.asarray(pts, float)
    nb = _bonds(sym, p, _cutoffs(sym, charges, native, bulk_bond))
    an_n = {e for e in native if charges.get(e, 0) < 0}
    centroid = p.mean(axis=0)
    core = np.array([q for s_, q in zip(sym, p) if s_ != x_sym])
    core_tree = cKDTree(core)
    cand = []
    for i, s_ in enumerate(sym):
        if charges.get(s_, 0) <= 0:
            continue
        us = [p[j] - p[i] for j in nb[i] if sym[j] in an_n]
        if not us or len(us) >= cn_bulk:
            continue
        for d in _missing_directions(us, cn_bulk):
            if float(np.dot(d, _unit_vec(p[i] - centroid))) < -0.2:
                continue
            xp = p[i] + d_mx * d
            if any(float(np.linalg.norm(core[j] - p[i])) > 1e-6 for j in core_tree.query_ball_point(xp, 0.8 * d_mx)):
                continue
            cand.append(xp)
    if not cand:
        return 0, sum(1 for s_ in sym if s_ == x_sym), 0
    cand = np.asarray(cand)
    root = list(range(len(cand)))

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a
    for a, b in cKDTree(cand).query_pairs(LIGAND_SITE_MERGE):
        ra, rb = find(a), find(b)
        root[max(ra, rb)] = min(ra, rb)
    sites = np.array([cand[[j for j in range(len(cand)) if find(j) == r]].mean(axis=0) for r in sorted(set(map(find, range(len(cand)))))])
    xs = np.array([q for s_, q in zip(sym, p) if s_ == x_sym])
    occupied = 0
    if len(xs):                                  # every X sits on a site: one not matching a predicted
        dist, idx = cKDTree(sites).query(xs)     # position (within 1 A, one X per site) adds its own
        taken = set()
        for d_, j in sorted(zip(dist, idx)):
            if d_ < 1.0 and j not in taken:
                taken.add(j)
        occupied = len(taken)
    return len(sites) + len(xs) - occupied, len(xs), occupied


def training_frames(ctx, settings, path, q, unit) -> dict:
    """
    props/ensemble/detach_relax.extxyz: frames off the minimum from the search, for the ML set.
    Every candidate relaxation left snapshots in the spool (its start, right after the removal,
    and the first steps that release detach_frames_fractions of its energy drop); kept are all
    frames of the relaxations that made the best path and a seeded random subset of the others
    (at most detach_frames_max_other). Energies and forces are exact MACE-MH-1 labels.
    """
    from ase.io import read
    from .ensemble import _write_extxyz
    spool = ctx.props / "ensemble" / "detach_spool.extxyz"
    out = ctx.props / "ensemble" / "detach_relax.extxyz"
    if not spool.is_file():
        return {"frames": 0, "note": "no new candidate relaxations (all cached)" if not out.is_file()
                else "kept the existing detach_relax.extxyz (all relaxations cached)"}
    n0 = len(ctx.symbols)
    on_path = {geometry_key(st.sym, st.pts): st.k for st in path}
    frames, seen = [], set()
    for a in read(str(spool), index=":"):
        key = (a.info["start_key"], a.info["relax_step"])
        if key in seen:                              # re-relaxed after an interruption
            continue
        seen.add(key)
        frames.append({"symbols": a.get_chemical_symbols(), "positions": a.get_positions(),
                       "forces": a.get_forces(), "energy": a.get_potential_energy(),
                       "info": {"source": "detach_relax", "k": int(round((n0 - len(a)) / (q + 1))),
                                "fraction": float(a.info["fraction"]), "relax_step": int(a.info["relax_step"]),
                                "on_path": a.info["final_key"] in on_path}})
    path_frames = [f for f in frames if f["info"]["on_path"]]
    others = [f for f in frames if not f["info"]["on_path"]]
    import hashlib
    seed = hashlib.sha256(f"{settings.detach_seed}|{ctx.record_dir.name}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], "little"))
    if len(others) > settings.detach_frames_max_other:
        pick = rng.choice(len(others), settings.detach_frames_max_other, replace=False)
        others = [others[i] for i in sorted(pick)]
    kept = sorted(path_frames + others, key=lambda f: (f["info"]["k"], not f["info"]["on_path"]))
    _write_extxyz(out, None, kept, f"MACE-MH-1 {settings.head}")
    spool.unlink()
    return {"frames": len(kept), "on_path": len(path_frames), "other": len(others),
            "candidates_spooled": len(frames), "fractions": list(settings.detach_frames_fractions),
            "file": "ensemble/detach_relax.extxyz", "unit": unit}


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
        greedy = s.detach_search == "greedy"
        width = 1 if greedy else BEAM
        local_r = GREEDY_R_LOCAL if greedy else max(LOCAL_R, LOCAL_FRAC * span)
        refresh = GREEDY_REFRESH if greedy else REFRESH_WINDOW
        k_max = min(m, s.detach_max_levels) if greedy and s.detach_max_levels else m
        root = State(sym0, pts0, range(len(sym0)), 0.0, [])
        levels = [[root]]
        first_table = None
        while levels[-1] and levels[-1][0].k < k_max:
            children = expand(levels[-1], cache, s, e_full, ensemble, stats,
                              lambda st: enumerate_units(st.sym, st.pts, ctx.charges, ctx.native, bulk_bond, q),
                              local_r, width, refresh)
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
    # Ladder up: MX_q added one at a time on free surface anions (the reverse step), with the same
    # greedy machinery (screening, local updates, batched relaxations).
    path_up = []
    if getattr(s, "detach_up", False) and u.get("X"):
        mono = np.asarray(mx["positions"], float)
        d_mx = _surface_mx_bond(sym0, np.asarray(pts0, float), ctx.charges, ctx.native, bulk_bond, u,
                                1.08 * float(np.linalg.norm(mono[1:] - mono[0], axis=1).min()))
        up_levels = [[State(sym0, pts0, range(len(sym0)), 0.0, [], sign=-1)]]
        while up_levels[-1] and up_levels[-1][0].k < s.detach_up_max_levels:
            kids = expand(up_levels[-1], cache, s, e_full, ensemble, stats,
                          lambda st: enumerate_sites(st, ctx.charges, ctx.native, bulk_bond, u, d_mx),
                          GREEDY_R_LOCAL, 1, GREEDY_REFRESH)
            if not kids:
                break
            up_levels.append(kids)
        st = min(up_levels[-1], key=lambda x: x.energy) if len(up_levels) > 1 else None
        while st is not None and st.parent is not None:
            path_up.append(st)
            st = st.parent
        path_up.reverse()

    best_at = {}
    for k_, e_, _sg in ensemble.values():
        best_at[k_] = min(best_at.get(k_, e_), e_)

    # Harmonic free energies along the best path.
    f_prev = f_full
    ladder = []
    prev_e = 0.0
    quick = bool(getattr(s, "detach_quick", False)) and thermo_ok and len(path) > 1
    if quick:                                        # one Hessian: the last state; F_vib linear in k
        fr_last = np.asarray(cache.frequencies(path[-1].sym, path[-1].pts, s_hess))
        f_last = np.asarray(thermo(fr_last, T)["F_vib_eV"])
        k_last = path[-1].k
    for st in path:
        fr = None
        if quick:
            fr = fr_last if st is path[-1] else None
            f_st = f_last if st is path[-1] else f_full + (f_last - f_full) * st.k / k_last
            n_imag = int((fr < -10.0).sum()) if fr is not None else None
        elif thermo_ok:
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
                       "F_vib_source": ("Hessian" if fr is not None else
                                        "linear in k (quick mode)" if quick else "not computed"),
                       "symbols": st.sym,
                       "best_at_level_eV": min(best_at.get(st.k, st.energy), st.energy)})
        _write_xyz(ctx.props / f"detach_{st.k}.xyz", st.sym, st.pts,
                   f"best path, {st.k} {unit} removed, E - E_full = {st.energy:.6f} eV")
        prev_e, f_prev = st.energy, f_st
    # The ladder up: adsorption energies dE_add = E(dot_-j) - E(dot_-(j-1)) - E(MX_q) (negative: the unit binds).
    ladder_up = []
    f_prev, prev_e = f_full, 0.0
    if path_up:
        fr_up = (np.asarray(cache.frequencies(path_up[-1].sym, path_up[-1].pts, s_hess)) if thermo_ok else None)
        f_up = np.asarray(thermo(fr_up, T)["F_vib_eV"]) if fr_up is not None else None
        for st in path_up:
            last = st is path_up[-1]
            f_st = (f_up if last else f_full + (f_up - f_full) * st.k / path_up[-1].k) if f_up is not None else f_prev
            dF = f_st - f_prev
            dE = st.energy - prev_e - mx["energy_eV"]
            dG = st.energy - prev_e + dF - g_mx
            ladder_up.append({"k": -st.k, "E_mace_eV": st.energy, "dE_eV": dE, "dG_eV": dG.tolist(),
                              "dG_300K_eV": float(dG[i300]) if i300 is not None else None,
                              "dF_vib_300K_eV": float(dF[i300]) if i300 is not None else None,
                              "dF_vib_eV": dF.tolist(), **st.step,
                              "frequencies_cm1": fr_up.round(3).tolist() if (last and fr_up is not None) else None,
                              "F_vib_source": ("Hessian" if (last and fr_up is not None) else
                                               "linear in k (quick mode)" if fr_up is not None else "not computed"),
                              "symbols": st.sym})
            _write_xyz(ctx.props / f"adsorb_{st.k}.xyz", st.sym, st.pts,
                       f"ladder up, {st.k} {unit} added, E - E_full = {st.energy:.6f} eV")
            prev_e, f_prev = st.energy, f_st
    cache.close()
    frames = training_frames(ctx, s, path + path_up, q, unit) if getattr(s, "detach_frames", False) else None

    # Configurational entropy of every ladder state from its own structure: the ways to place its
    # y ligands X on the x places an X can sit (ligand_sites: the distinct anion-lattice positions
    # next to the surface metals, one per position whether it bridges several metals or is
    # terminal), S_conf = k_B ln C(x, y). An ideal count (every arrangement equally likely): an
    # upper bound, S_conf = 0 the lower one. It includes the sites left free at maximum coverage.
    from math import lgamma

    def ln_binom(n_, k_):
        return lgamma(n_ + 1) - lgamma(k_ + 1) - lgamma(n_ - k_ + 1) if 0 <= k_ <= n_ else 0.0

    an_n = [e for e in ctx.native if ctx.charges.get(e, 0) < 0]
    nb0 = _bonds(list(sym0), np.asarray(pts0, float), _cutoffs(sym0, ctx.charges, ctx.native, bulk_bond))
    cats0 = [i for i, s_ in enumerate(sym0) if ctx.charges.get(s_, 0) > 0]
    cn_bulk = max(sum(1 for j in nb0[i] if sym0[j] in an_n) for i in cats0)

    mono = np.asarray(mx["positions"], float)
    d_mx_conf = _surface_mx_bond(sym0, np.asarray(pts0, float), ctx.charges, ctx.native, bulk_bond, u,
                                 1.08 * float(np.linalg.norm(mono[1:] - mono[0], axis=1).min()))

    def sites_and_ligands(sym, pts):
        x_, y_, _occ = ligand_sites(sym, pts, ctx.charges, ctx.native, bulk_bond, u["X"], d_mx_conf, cn_bulk)
        return x_, y_

    x0, y0 = sites_and_ligands(sym0, pts0)
    xy = {0: (x0, y0)}
    pairs = [(st_, e_) for st_, e_ in zip(path, ladder)] + [(st_, e_) for st_, e_ in zip(path_up, ladder_up)]
    for st, lad in pairs:
        xy[lad["k"]] = sites_and_ligands(st.sym, st.pts)
    # The raw count moves by a few sites from state to state (relaxation reconstructs the surface and
    # moves predicted positions across the merge tolerance), which made single steps noisy by
    # 0.1-0.2 eV. x(k) is therefore a straight line on each side of the intact dot, through its exact
    # count x0 (least squares slope: sites lost per removal, gained per addition); y is exact.
    def slope(side):
        ks_ = np.array([k_ for k_ in xy if (k_ > 0 if side > 0 else k_ < 0)], float)
        if not len(ks_):
            return 0.0
        xs_ = np.array([xy[int(k_)][0] for k_ in ks_], float) - x0
        return float((ks_ * xs_).sum() / (ks_ * ks_).sum())
    a_down, a_up = slope(1), slope(-1)

    def x_fit(k_):
        return x0 + (a_down if k_ > 0 else a_up) * k_

    def ln_binom_real(n_, k_):
        return lgamma(n_ + 1) - lgamma(k_ + 1) - lgamma(n_ - k_ + 1) if 0 <= k_ <= n_ else 0.0
    s_conf = {k_: ln_binom_real(x_fit(k_), y_) for k_, (_x, y_) in xy.items()}
    for st, lad in pairs:
        xk, yk = xy[lad["k"]]
        prev = lad["k"] - 1 if lad["k"] > 0 else lad["k"] + 1    # the state one step closer to the intact dot
        lad["sites_x"], lad["sites_x_fit"], lad["ligands_y"] = int(xk), round(x_fit(lad["k"]), 2), int(yk)
        lad["S_conf_kB"] = s_conf[lad["k"]] - s_conf[0]           # relative to the intact dot
        lad["dS_conf_kB"] = s_conf[lad["k"]] - s_conf[prev]       # change in this step

    # Equilibrium <k>(T, dmu) over every evaluated configuration.
    mu = np.asarray(s.mu_grid, float)
    dF_cum = {0: np.zeros(len(T))}          # vibrational change after k removals (best path)
    for st in ladder:
        dF_cum[st["k"]] = dF_cum[st["k"] - 1] + np.asarray(st["dF_vib_eV"])
    for st in ladder_up:                     # k = -1, -2, ...: units added
        dF_cum[st["k"]] = dF_cum[st["k"] + 1] + np.asarray(st["dF_vib_eV"])
    sigma0 = len(cache.perms)
    confs = [(0, 0.0, sigma0)] + [v for v in ensemble.values()]
    ks = np.array([c[0] for c in confs], float)
    es = np.array([c[1] for c in confs], float)
    lw = np.log(sigma0 / np.array([c[2] for c in confs], float))   # labelled count of each distinct configuration
    dfs = np.vstack([dF_cum[min(int(k), len(ladder)) if k >= 0 else max(int(k), -len(ladder_up))] for k in ks])
    kmean = np.zeros((len(T), len(mu)))
    for ti, t in enumerate(T):
        kT = KB_EV * t
        base = es + dfs[:, ti] + ks * g_mx[ti]
        lg = lw[:, None] - (base[:, None] + ks[:, None] * mu[None, :]) / kT
        lg -= lg.max(axis=0, keepdims=True)
        p = np.exp(lg)
        kmean[ti] = (p * ks[:, None]).sum(axis=0) / p.sum(axis=0)

    # <k> over the ladder states with their configurational entropy (lattice gas): state k (removed,
    # k > 0, or added, k < 0) has G_k(T) = E_k + F_vib,k + k G_MXq - T S_conf(k).
    states = [(0, 0.0)] + [(st["k"], st["E_mace_eV"]) for st in ladder] + [(st["k"], st["E_mace_eV"]) for st in ladder_up]
    kk = np.array([k_ for k_, _e in states], float)
    ee = np.array([e_ for _k, e_ in states], float)
    def ladder_kmean(entropy):                      # entropy -> S_conf/k_B of every state in kk
        out = np.zeros((len(T), len(mu)))
        for ti, t_ in enumerate(T):
            kT = KB_EV * t_
            base = ee + np.array([dF_cum[int(k_)][ti] for k_ in kk]) + kk * g_mx[ti] - kT * entropy
            lg = -(base[:, None] + kk[:, None] * mu[None, :]) / kT
            lg -= lg.max(axis=0, keepdims=True)
            p_ = np.exp(lg)
            out[ti] = (p_ * kk[:, None]).sum(axis=0) / p_.sum(axis=0)
        return out
    kmean_conf = ladder_kmean(np.array([s_conf[int(k_)] - s_conf[0] for k_ in kk]))
    kmean_noconf = ladder_kmean(np.zeros(len(kk)))

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
            "method": ({"symmetric": f"symmetric moves (width {BEAM}) with local updates",
                        "greedy": f"greedy search (one unit per step, re-relaxed within {GREEDY_R_LOCAL:g} Å, "
                                  f"at most {s.detach_max_levels or m} removals)"}
                       .get(s.detach_search, f"beam search (width {BEAM}) with local updates")),
            "search": s.detach_search, "core_c2_operations": n_ops, "unit": unit, "n_units": m,
            "n_steps": len(ladder), "n_moves": len(levels_out), "beam": 1 if s.detach_search == "greedy" else BEAM,
            "local_radius_A": local_r,
            "n_intermediate": stats.get("intermediate", 0),
            "n_relaxations": stats["relaxations"], "n_lazy_checks": stats["lazy"], "n_lazy_batches": stats.get("lazy_batches", 0),
            "n_polished": stats.get("polished", 0), "polish_fmax_eV_A": POLISH_FMAX,
            "quick": bool(getattr(s, "detach_quick", False)),
            "candidate_fmax_eV_A": _candidate_fmax(s),
            "polish_window_eV": POLISH_WINDOW,
            "n_relaxations_new": cache.n_new, "n_configurations": len(confs),
            "refresh_max": s.detach_refresh_max, "refresh_capped": stats.get("refresh_capped", 0),
            "rigid_refresh": ({"single_points": stats.get("rigid_single_points", 0),
                               "estimates_corrected": stats.get("rigid_corrected", 0)}
                              if getattr(s, "detach_rigid_refresh", False) else None),
            "screen": ({"single_points": stats.get("screen_single_points", 0), "screened_out": stats.get("screened_out", 0),
                        "min_groups": s.detach_screen_min, "keep": s.detach_screen_keep, "max": s.detach_screen_max,
                        "window_eV": s.detach_screen_window, "quantile": s.detach_screen_quantile}
                       if getattr(s, "detach_screen", False) else None),
            "ensemble": "every relaxed configuration evaluated by the search",
            "dE_eV": [st["dE_eV"] for st in ladder],
            "dG_300K_eV": [st["dG_300K_eV"] for st in ladder],
            "cumulative_dE_eV": [st["E_mace_eV"] + st["k"] * mx["energy_eV"] for st in ladder],
            "stopped": (None if len(ladder) == m else
                        f"after {len(ladder)} of {m} units (detach_max_levels)" if s.detach_search == "greedy"
                        and s.detach_max_levels and len(ladder) >= s.detach_max_levels else "no unit left"),
            "thermo": bool(thermo_ok),
            "up": ({"dE_add_eV": [st["dE_eV"] for st in ladder_up],
                    "dG_add_300K_eV": [st["dG_300K_eV"] for st in ladder_up],
                    "cumulative_dE_add_eV": [st["E_mace_eV"] - abs(st["k"]) * mx["energy_eV"] for st in ladder_up],
                    "n_steps": len(ladder_up), "max_levels": s.detach_up_max_levels}
                   if getattr(s, "detach_up", False) else None),
            "training_frames": frames,
        },
        "site_classes": sites,
        "steps": ladder,
        "steps_up": ladder_up,
        "ensemble_list": [list(v) for v in ensemble.values()],      # [k, E - E_full, rotational symmetry number]
        "sigma_full": sigma0,
        "hessian_method": method,
        "levels": [[{"E_eV": x.energy, "path": [list(map(lambda v: v if isinstance(v, int) else list(v), p_))
                                               for p_ in x.path]} for x in lv] for lv in levels_out],
        "map": {"T": T, "dmu_eV": mu.tolist(), "k_mean": kmean_conf.tolist(),
                "model": "ladder states (down and up) with the configurational entropy k_B ln C(x, y) of each",
                "k_mean_no_conf": kmean_noconf.tolist(),
                "bounds": "k_mean: S_conf = k_B ln C(x, y) of every state (upper bound); k_mean_no_conf: S_conf = 0",
                "k_mean_ensemble": kmean.tolist(), "ensemble": "evaluated configurations (no configurational entropy)"},
        "configurational": {"model": ("k_B ln C(x, y): y ligands X on the x distinct anion-lattice sites next to the "
                                      "surface metals (terminal or bridging, as the builder places X)"),
                            "cn_bulk": int(cn_bulk), "intact": {"sites_x": int(x0), "ligands_y": int(y0),
                                                                "S_conf_kB": float(s_conf[0])},
                            "x_fit": {"sites_per_removal": a_down, "sites_per_addition": -a_up,
                                      "note": "x(k) = x0 + slope k on each side of the intact dot (least squares)"},
                            "per_state": {str(k_): {"sites_x": int(v[0]), "ligands_y": int(v[1])} for k_, v in sorted(xy.items())}},
        "reference": {"MXq_energy_eV": mx["energy_eV"], "MXq_G_1M_eV": mx["G_1M_eV"]},
    }
