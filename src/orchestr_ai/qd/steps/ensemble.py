# src/orchestr_ai/qd/steps/ensemble.py
"""
Structure ensembles for DFT labelling and fine-tuning: Wigner samples and short MD.

wigner  For each temperature, `wigner_samples` structures drawn from the quantum
        harmonic (Wigner) distribution of the MACE-MH-1 normal modes (props/modes.npz):
        Q_j ~ N(0, hbar/(2 w_j) coth(hbar w_j / 2kT)), Qdot_j ~ N(0, hbar w_j/2 coth(...)).
        Imaginary modes and modes below wigner_cutoff_cm are left out. Soft modes
        (below wigner_soft_cm: ligand rotations, floppy surface motion) are sampled
        with the width of a wigner_soft_cm mode, because a straight-line displacement
        along a rotation-like mode at its own (huge) amplitude stretches bonds.

md      Short MD with MACE-MH-1 in the canonical ensemble: velocity Verlet with the CSVR
        thermostat (Bussi, Donadio, Parrinello, J. Chem. Phys. 126, 014101 (2007): the
        kinetic energy is resampled from its canonical distribution with coupling time
        md_tau_fs, half a step before and after each Verlet step), or Langevin BAOAB
        (md_thermostat: langevin). Every replica follows its own
        temperature schedule, constant ([T]) or a linear ramp ([T0, T1]); all replicas
        share one batched model call per step, so six replicas cost little more than
        one. Replicas start from Wigner samples at their first temperature and are
        stopped if they become unphysical (forces above md_max_force, or an atom more
        than 10 A outside the starting structure). Frames every md_stride_fs after
        md_skip_ps carry their exact MACE energy and forces.

detach_ensemble  Short MD (the same integrator) from states of the desorption ladder,
        props/detach_<k>.xyz, with Maxwell-Boltzmann start velocities: the newly exposed,
        reconstructed surfaces at finite temperature (props/ensemble/detach_md.extxyz).
        The search itself leaves frames off the minimum (detach_relax.extxyz, see
        detachment.training_frames).

Latent features.  Next to every extxyz file, <name>.latent.npz holds the per-atom MACE-MH-1
latent features of its frames (the invariant channels of every interaction layer) projected on
frozen 64-component PCA bases, one per element (data/latent_pca64_macemh1_omat_pbe.npz, fitted on
all library materials), in float16: `features` (all atoms of all frames, in order), `symbols` and
`offsets` (frame f is features[offsets[f]:offsets[f+1]]). The same bases for the whole library,
so the atoms of every record can be compared and selected (farthest-point sampling, clustering,
entropy) element by element in one space.

Every frame stores its positions, the MACE-MH-1 energy (eV, also per atom), forces (eV/A)
and velocities where it has them (extxyz: info energy, energy_per_atom, label; arrays forces,
vel_A_fs), so the files are a training set as written and DFT labels can be added beside them.
Every structure gets energy, forces and MACE invariant descriptors (per element
averaged; per atom for the optimised structure), labelled in batched calls. Files go
to props/ensemble/: optimised.xyz, wigner_<T>K.extxyz, md_<name>.extxyz,
embeddings.npz and manifest.json.
"""
from __future__ import annotations

import functools
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..engines import mace_calculator, mace_provenance
from ..records import read_xyz_first_frame

HBAR = 1.054571817e-34         # J s
KB_J = 1.380649e-23            # J/K
KB_EV = 8.617333262e-5         # eV/K
AMU = 1.66053906660e-27        # kg
C_CM = 2.99792458e10           # cm/s
ACC = 9.648533212e-3           # (eV/A/amu) -> A/fs^2


def _masses(symbols):
    from ase.data import atomic_masses, atomic_numbers
    return np.array([atomic_masses[atomic_numbers[s]] for s in symbols])


def wigner_samples(pos0, masses, freqs_cm, modes_mw, T, n, rng, cutoff_cm=20.0, soft_cm=50.0):
    """n (positions A, velocities A/fs) from the Wigner distribution at T (modes: mass-weighted, (3N, m))."""
    keep = np.flatnonzero(freqs_cm > cutoff_cm)
    w = 2 * np.pi * C_CM * np.maximum(freqs_cm[keep], soft_cm)          # rad/s
    x = HBAR * w / (2 * KB_J * max(T, 1e-6))
    coth = 1.0 / np.tanh(np.minimum(x, 350.0))
    sig_q = np.sqrt(HBAR / (2 * w) * coth / (AMU * 1e-20))             # amu^1/2 A
    sig_v = np.sqrt(HBAR * w / 2 * coth / (AMU * 1e-20) * 1e-30)       # amu^1/2 A/fs
    L = modes_mw[:, keep]
    inv_sqrt_m = np.repeat(1 / np.sqrt(masses), 3)
    Q = rng.standard_normal((n, len(keep))) * sig_q
    V = rng.standard_normal((n, len(keep))) * sig_v
    dx = (Q @ L.T) * inv_sqrt_m
    dv = (V @ L.T) * inv_sqrt_m
    return pos0[None] + dx.reshape(n, -1, 3), dv.reshape(n, -1, 3)


def _element_means(symbols, desc):
    els = sorted(set(symbols))
    sym = np.asarray(symbols)
    return els, np.stack([desc[sym == e].mean(axis=0) for e in els]).astype(np.float16)


def harmonic_potential_energy(freqs_cm, T, cutoff_cm, soft_cm):
    """<V> of the sampled Wigner distribution: sum_j hbar w_j / 4 coth(hbar w_j / 2kT), in eV."""
    w = 2 * np.pi * C_CM * np.maximum(freqs_cm[freqs_cm > cutoff_cm], soft_cm)
    x = np.minimum(HBAR * w / (2 * KB_J * max(T, 1e-6)), 350.0)
    return float(np.sum(HBAR * w / 4 / np.tanh(x)) / 1.602176634e-19)


def _write_extxyz(path, symbols, frames, label="MACE-MH-1"):
    """frames: dicts with positions, forces, energy (+ velocities, info); label: the method of energy and forces."""
    from ase import Atoms
    from ase.io import write
    out = []
    for fr in frames:
        a = Atoms(fr.get("symbols", symbols), positions=fr["positions"])
        a.arrays["forces"] = np.asarray(fr["forces"], float)
        if fr.get("velocities") is not None:
            a.arrays["vel_A_fs"] = np.asarray(fr["velocities"], float)
        a.info.update({"energy": float(fr["energy"]), "energy_per_atom": float(fr["energy"]) / len(a),
                       "label": label, **fr.get("info", {})})
        out.append(a)
    write(str(path), out, format="extxyz")


LATENT_BASIS = Path(__file__).resolve().parents[1] / "data" / "latent_pca64_macemh1_omat_pbe.npz"


@functools.lru_cache(maxsize=4)
def _latent_basis(model: str, head: str):
    """{element: (mean, components)} of the frozen per-element PCA bases and their provenance, or None."""
    if not LATENT_BASIS.is_file():
        return None
    from ..engines import file_sha256
    b = np.load(LATENT_BASIS)
    sha = file_sha256(model)[:16] if Path(model).is_file() else "?"
    bases = {str(e): (b[f"mean_{e}"].astype(np.float32), b[f"components_{e}"].astype(np.float32)) for e in b["elements"]}
    prov = {"basis": LATENT_BASIS.name, "kind": str(b["kind"]), "basis_model_sha256_16": str(b["model_sha256_16"]),
            "basis_head": str(b["head"]), "model_sha256_16": sha, "head": head,
            "consistent": sha == str(b["model_sha256_16"]) and head == str(b["head"])}
    return bases, prov


def write_latent(path, descriptors, symbols, s) -> Optional[dict]:
    """
    <path stem>.latent.npz: per-atom latent features (float16) of the frames written to `path`:
    `descriptors` the per-atom invariant features of each frame and `symbols` its atoms, in the
    file's order. Every atom is projected on its element's PCA basis (an element without a
    basis gets NaN), so atoms are compared with atoms of the same element.
    """
    basis = _latent_basis(str(s.model), s.head)
    if basis is None or not len(descriptors):
        return None
    bases, prov = basis
    n_comp = next(iter(bases.values()))[1].shape[0]
    feats = []
    for d, sym in zip(descriptors, symbols):
        d, sym = np.asarray(d, np.float32), np.asarray(sym)
        f = np.full((len(sym), n_comp), np.nan, np.float32)
        for e in set(sym.tolist()):
            if e in bases:
                m, c = bases[e]
                f[sym == e] = (d[sym == e] - m) @ c.T
        feats.append(f.astype(np.float16))
    offsets = np.cumsum([0] + [len(f) for f in feats])
    path = Path(path)
    np.savez_compressed(path.with_name(path.name.split(".")[0] + ".latent.npz"), features=np.concatenate(feats),
                        offsets=offsets, symbols=np.concatenate([np.asarray(x) for x in symbols]),
                        **{k: str(v) for k, v in prov.items()})
    return prov


def _evaluator(s, descriptors=False):
    from ..batch_relax import BatchMACE
    calc = mace_calculator(s.head, s.model, s.device, s.dtype)
    return BatchMACE(calc, s.relax_batch_atoms or 4000, descriptors=descriptors)


def _label(ev, symbols, positions):
    return ev([(symbols, p) for p in positions])


def run_wigner(ctx) -> dict:
    s = ctx.settings
    out_dir = ctx.props / "ensemble"
    out_dir.mkdir(exist_ok=True)
    symbols, pos0 = ctx.relaxed()
    symbols, pos0 = list(symbols), np.asarray(pos0, float)
    m = np.load(ctx.props / "modes.npz")
    freqs, modes = np.asarray(m["frequencies_cm1"], float), np.asarray(m["modes_mass_weighted"], float)
    masses = _masses(symbols)
    ev = _evaluator(s, descriptors=True)
    label = f"MACE-MH-1 {s.head}"
    t0 = time.time()
    (e0, f0, d0), = _label(ev, symbols, [pos0])
    _write_extxyz(out_dir / "optimised.xyz", symbols, [{"positions": pos0, "forces": f0, "energy": e0}], label)
    latent = write_latent(out_dir / "optimised.xyz", [d0], [symbols], s)
    emb = {"optimised_per_atom": d0.astype(np.float16)}
    rng = np.random.default_rng(s.wigner_seed)
    per_T = {}
    for T in s.wigner_temperatures:
        P, V = wigner_samples(pos0, masses, freqs, modes, T, s.wigner_samples, rng,
                              s.wigner_cutoff_cm, s.wigner_soft_cm)
        lab = _label(ev, symbols, P)
        frames = [{"positions": P[k], "velocities": V[k], "forces": f, "energy": e,
                   "info": {"T_K": float(T), "sample": k, "source": "wigner"}} for k, (e, f, _d) in enumerate(lab)]
        _write_extxyz(out_dir / f"wigner_{int(T)}K.extxyz", symbols, frames, label)
        write_latent(out_dir / f"wigner_{int(T)}K.extxyz", [d for _e, _f, d in lab], [symbols] * len(lab), s)
        els, _ = _element_means(symbols, lab[0][2])
        emb[f"wigner_{int(T)}K"] = np.stack([_element_means(symbols, d)[1] for _e, _f, d in lab])
        dE = np.array([e for e, _f, _d in lab]) - e0
        fmax = np.array([np.sqrt((f ** 2).sum(axis=1)).max() for _e, f, _d in lab])
        rmsd = np.sqrt(((P - pos0[None]) ** 2).sum(axis=2).mean(axis=1))
        per_T[str(int(T))] = {"n": len(lab), "dE_mean_eV": float(dE.mean()), "dE_std_eV": float(dE.std()),
                              "dE_harmonic_eV": harmonic_potential_energy(freqs, T, s.wigner_cutoff_cm, s.wigner_soft_cm),
                              "fmax_max_eV_A": float(fmax.max()), "rmsd_mean_A": float(rmsd.mean())}
    els = sorted(set(symbols))
    np.savez_compressed(out_dir / "embeddings.npz", elements=np.array(els), **emb)
    n_used = int(np.sum(freqs > s.wigner_cutoff_cm))
    n_soft = int(np.sum((freqs > s.wigner_cutoff_cm) & (freqs < s.wigner_soft_cm)))
    manifest = {"record": ctx.record.get("id"), "n_atoms": len(symbols), "E_optimised_eV": e0,
                "wigner": {"temperatures_K": list(map(float, s.wigner_temperatures)), "samples": s.wigner_samples,
                           "seed": s.wigner_seed, "cutoff_cm": s.wigner_cutoff_cm, "soft_cm": s.wigner_soft_cm,
                           "modes_used": n_used, "soft_modes": n_soft,
                           "imaginary_modes": int(np.sum(freqs < 0))},
                "latent": latent,
                "model": mace_provenance(s.head, s.model, s.device, s.dtype)}
    _update_manifest(out_dir, manifest)
    return {"summary": {"n_samples": int(sum(v["n"] for v in per_T.values())), "modes_used": n_used,
                        "soft_modes": n_soft, "seconds_labelling": round(time.time() - t0, 1),
                        "descriptor_dim": int(d0.shape[1])},
            "per_temperature": per_T}


def _update_manifest(out_dir, new):
    path = out_dir / "manifest.json"
    old = json.loads(path.read_text()) if path.is_file() else {}
    old.update(new)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(old, indent=1, default=float))
    tmp.replace(path)


def csvr_scale(v, masses, kT, dt, tau, n_dof, rng):
    """Velocity factor of one CSVR step of length dt (Bussi et al. 2007, eq. A7): the kinetic energy K
    relaxes to n_dof kT/2 with time constant tau, with the noise that keeps the ensemble canonical."""
    K = 0.5 * float((masses[:, None] * v ** 2).sum()) / ACC          # eV
    if K <= 0.0:
        return 1.0
    K_target = 0.5 * n_dof * kT
    c = np.exp(-dt / tau)
    r1 = rng.standard_normal()
    s = rng.chisquare(n_dof - 1) if n_dof > 1 else 0.0
    K_new = (K + (1 - c) * (K_target / n_dof * (r1 ** 2 + s) - K)
             + 2 * r1 * np.sqrt(c * (1 - c) * K_target / n_dof * K))
    return float(np.sqrt(max(K_new, 0.0) / K))


def _schedule_T(sched, t_frac):
    return float(sched[0]) if len(sched) == 1 else float(sched[0] + (sched[1] - sched[0]) * t_frac)


def maxwell_velocities(masses, T, rng):
    """Velocities (A/fs) from the Maxwell-Boltzmann distribution at T, without centre-of-mass motion."""
    v = rng.standard_normal((len(masses), 3)) * np.sqrt(KB_EV * T * ACC / masses)[:, None]
    return v - (masses[:, None] * v).sum(axis=0) / masses.sum()


def md_replicas(ev, systems, velocities, scheds, names, s, n_steps, skip, stride, rng, info=None):
    """
    Independent MD replicas in one batched model call per step, each with its own structure
    (symbols, positions), start velocities and temperature schedule ([T] or [T0, T1]).
    Velocity Verlet with CSVR (s.md_thermostat "csvr") or Langevin BAOAB ("langevin").
    A replica stops when its forces exceed s.md_max_force or an atom leaves its start by more
    than 10 A. Returns per replica the frames (every `stride` steps from step `skip`: positions,
    velocities, exact energy and forces, info) and a summary.
    """
    if s.md_thermostat not in ("csvr", "langevin"):
        raise ValueError(f"md_thermostat {s.md_thermostat!r}: csvr or langevin")
    csvr = s.md_thermostat == "csvr"
    dt = float(s.md_timestep_fs)
    R = len(systems)
    syms = [list(sy) for sy, _p in systems]
    X = [np.asarray(p, float).copy() for _sy, p in systems]
    V = [np.asarray(v, float).copy() for v in velocities]
    M = [_masses(sy) for sy in syms]
    inv_m = [(ACC / m)[:, None] for m in M]
    n_dof = [3 * len(sy) - 3 for sy in syms]          # centre-of-mass momentum is removed
    c1 = np.exp(-s.md_friction_fs * dt)
    c2 = np.sqrt(1 - c1 ** 2)
    centre0 = [x.mean(axis=0) for x in X]
    radius0 = [float(np.linalg.norm(x - c, axis=1).max()) for x, c in zip(X, centre0)]
    active = np.ones(R, bool)
    stop_reason = [None] * R
    frames = [[] for _ in range(R)]
    T_sum, T_n = np.zeros(R), np.zeros(R)

    def forces(idx):
        return dict(zip(idx, ev([(syms[r], X[r]) for r in idx])))

    cur = forces(list(range(R)))
    for step in range(n_steps + 1):
        idx = [r for r in range(R) if active[r]]
        if not idx:
            break
        if step > 0:
            for r in idx:
                kT = KB_EV * _schedule_T(scheds[r], step / n_steps)
                if csvr:                       # O(dt/2) B A
                    V[r] *= csvr_scale(V[r], M[r], kT, 0.5 * dt, s.md_tau_fs, n_dof[r], rng)
                    V[r] += 0.5 * dt * cur[r][1] * inv_m[r]
                    X[r] += dt * V[r]
                else:                          # B A O A
                    V[r] += 0.5 * dt * cur[r][1] * inv_m[r]
                    X[r] += 0.5 * dt * V[r]
                    sig = np.sqrt(kT * ACC / M[r])[:, None]
                    V[r] = c1 * V[r] + c2 * sig * rng.standard_normal(V[r].shape)
                    X[r] += 0.5 * dt * V[r]
            cur.update(forces(idx))
            for r in idx:                      # B (O(dt/2) for CSVR)
                V[r] += 0.5 * dt * cur[r][1] * inv_m[r]
                if csvr:
                    kT = KB_EV * _schedule_T(scheds[r], step / n_steps)
                    V[r] *= csvr_scale(V[r], M[r], kT, 0.5 * dt, s.md_tau_fs, n_dof[r], rng)
                V[r] -= (M[r][:, None] * V[r]).sum(axis=0) / M[r].sum()   # no centre-of-mass drift
        for r in idx:
            e, f = cur[r]
            fmax = float(np.sqrt((f ** 2).sum(axis=1)).max())
            far = float(np.linalg.norm(X[r] - centre0[r], axis=1).max())
            if not np.isfinite(e) or fmax > s.md_max_force or far > radius0[r] + 10.0:
                active[r] = False
                stop_reason[r] = (f"step {step}: max |F| {fmax:.1f} eV/A" if fmax > s.md_max_force
                                  else f"step {step}: atom {far - radius0[r]:.1f} A outside the start")
                continue
            T_inst = float((M[r][:, None] * V[r] ** 2).sum() / ACC / n_dof[r] / KB_EV)
            if step >= skip:
                T_sum[r] += T_inst
                T_n[r] += 1
                if (step - skip) % stride == 0:
                    frames[r].append({"symbols": syms[r], "positions": X[r].copy(), "velocities": V[r].copy(),
                                      "forces": f.copy(), "energy": e,
                                      "info": {"time_fs": step * dt, "T_target_K": _schedule_T(scheds[r], step / n_steps),
                                               "T_inst_K": T_inst, "replica": names[r], **((info or [{}] * R)[r])}})
    summary = [{"schedule_K": scheds[r], "frames": len(frames[r]), "T_inst_mean_K": float(T_sum[r] / max(T_n[r], 1)),
                "stopped": stop_reason[r]} for r in range(R)]
    return frames, summary


def _thermostat_name(s):
    return "CSVR (Bussi-Donadio-Parrinello), velocity Verlet" if s.md_thermostat == "csvr" else "Langevin BAOAB"


def run_md(ctx) -> dict:
    s = ctx.settings
    if not s.md_enabled:
        return {"summary": {"skipped": "md_enabled is false (MD runs on a subset of the library)"}}
    out_dir = ctx.props / "ensemble"
    out_dir.mkdir(exist_ok=True)
    symbols, pos0 = ctx.relaxed()
    symbols, pos0 = list(symbols), np.asarray(pos0, float)
    masses = _masses(symbols)
    m = np.load(ctx.props / "modes.npz")
    freqs, modes = np.asarray(m["frequencies_cm1"], float), np.asarray(m["modes_mass_weighted"], float)
    dt = float(s.md_timestep_fs)
    n_steps = int(round(s.md_ps * 1000 / dt))
    stride = max(1, int(round(s.md_stride_fs / dt)))
    skip = int(round(s.md_skip_ps * 1000 / dt))
    scheds = [list(map(float, sc if isinstance(sc, (list, tuple)) else [sc])) for sc in s.md_schedule]
    names = [f"{int(sc[0])}K" if len(sc) == 1 else f"ramp_{int(sc[0])}-{int(sc[1])}K" for sc in scheds]
    names = [n if names.count(n) == 1 else f"{n}_{k}" for k, n in enumerate(names)]
    rng = np.random.default_rng(s.md_seed)
    systems, vels = [], []
    for sc in scheds:                          # replicas start from Wigner samples at their first temperature
        P, Vw = wigner_samples(pos0, masses, freqs, modes, sc[0], 1, rng, s.wigner_cutoff_cm, s.wigner_soft_cm)
        systems.append((symbols, P[0]))
        vels.append(Vw[0])
    t0 = time.time()
    frames, summ = md_replicas(_evaluator(s), systems, vels, scheds, names, s, n_steps, skip, stride, rng,
                               info=[{"source": "md"}] * len(scheds))
    seconds_md = time.time() - t0
    evd = _evaluator(s, descriptors=True)
    emb = dict(np.load(out_dir / "embeddings.npz")) if (out_dir / "embeddings.npz").is_file() else {}
    per = {}
    for r, name in enumerate(names):
        if frames[r]:
            _write_extxyz(out_dir / f"md_{name}.extxyz", symbols, frames[r], f"MACE-MH-1 {s.head}")
            lab = _label(evd, symbols, [fr["positions"] for fr in frames[r]])
            emb[f"md_{name}"] = np.stack([_element_means(symbols, d)[1] for _e, _f, d in lab])
            write_latent(out_dir / f"md_{name}.extxyz", [d for _e, _f, d in lab], [symbols] * len(lab), s)
        rmsd = [float(np.sqrt(((fr["positions"] - pos0) ** 2).sum(axis=1).mean())) for fr in frames[r]]
        per[name] = {**summ[r], "rmsd_max_A": max(rmsd) if rmsd else None}
    emb.setdefault("elements", np.array(sorted(set(symbols))))
    np.savez_compressed(out_dir / "embeddings.npz", **emb)
    _update_manifest(out_dir, {"md": {"schedule_K": scheds, "names": names, "ps": s.md_ps, "timestep_fs": dt,
                                      "friction_fs": s.md_friction_fs, "tau_fs": s.md_tau_fs, "stride_fs": s.md_stride_fs,
                                      "skip_ps": s.md_skip_ps, "seed": s.md_seed, "thermostat": _thermostat_name(s)}})
    return {"summary": {"replicas": len(scheds), "frames": int(sum(len(f) for f in frames)), "steps": n_steps,
                        "timestep_fs": dt, "seconds_md": round(seconds_md, 1),
                        "ms_per_step": round(1e3 * seconds_md / max(n_steps, 1), 2),
                        "stopped": int(sum(x["stopped"] is not None for x in summ))},
            "replicas": per}


def run_detach_ensemble(ctx) -> dict:
    """
    Training frames from the desorption ladder: short MD (md_replicas, CSVR) from the states
    props/detach_<k>.xyz for k in detach_ensemble_levels (those the search reached; the last
    state is always included), at each of detach_ensemble_temperatures, starting from
    Maxwell-Boltzmann velocities at 2T (the ladder states have no Hessian; at a minimum half the
    kinetic energy goes into the potential at once). Frames go to
    props/ensemble/detach_md.extxyz; the per-element MACE descriptors of these frames and of the
    search's detach_relax.extxyz go to embeddings.npz (detach_md, detach_relax).
    """
    from ase.io import read
    s = ctx.settings
    det = ctx.results.get("detachment", {})
    ks = sorted(st["k"] for st in det.get("steps", []))
    ups = sorted(-st["k"] for st in det.get("steps_up", []) or [])
    if not ks and not ups:
        return {"summary": {"skipped": "no desorption ladder (detachment skipped or empty)"}}
    # ladder states: removals k > 0 (props/detach_<k>.xyz) and additions k < 0 (props/adsorb_<j>.xyz)
    levels = (sorted({k for k in s.detach_ensemble_levels if k in ks} | ({ks[-1]} if ks else set())) +
              [-j for j in sorted({j for j in s.detach_ensemble_levels if j in ups} | ({ups[-1]} if ups else set()))])
    out_dir = ctx.props / "ensemble"
    out_dir.mkdir(exist_ok=True)
    dt = float(s.md_timestep_fs)
    n_steps = int(round(s.detach_ensemble_ps * 1000 / dt))
    stride = max(1, int(round(s.detach_ensemble_stride_fs / dt)))
    skip = int(round(s.detach_ensemble_skip_ps * 1000 / dt))
    rng = np.random.default_rng(s.md_seed + 1)
    systems, vels, scheds, names, info = [], [], [], [], []
    def state_file(k):
        return ctx.props / (f"detach_{k}.xyz" if k > 0 else f"adsorb_{-k}.xyz")

    for k in levels:
        a = read(state_file(k))
        sy, p = a.get_chemical_symbols(), a.get_positions()
        for T in s.detach_ensemble_temperatures:
            systems.append((sy, p))
            # at 2T: starting at a minimum, equipartition moves half the kinetic energy into the potential
            vels.append(maxwell_velocities(_masses(sy), 2.0 * T, rng))
            scheds.append([float(T)])
            names.append(f"k{k}_{int(T)}K" if k > 0 else f"a{-k}_{int(T)}K")
            info.append({"source": "detach_md", "k": int(k)})
    t0 = time.time()
    frames, summ = md_replicas(_evaluator(s), systems, vels, scheds, names, s, n_steps, skip, stride, rng, info)
    seconds_md = time.time() - t0
    flat = [fr for rep in frames for fr in rep]
    label = f"MACE-MH-1 {s.head}"
    if flat:
        _write_extxyz(out_dir / "detach_md.extxyz", None, flat, label)
    # rattled and strained copies of the same ladder states
    starts = [(int(k), read(state_file(k))) for k in levels]
    rattled, n_rejected = rattle_strain([(k, a.get_chemical_symbols(), a.get_positions()) for k, a in starts], s, rng)
    if rattled:
        res = _evaluator(s)([(fr["symbols"], fr["positions"]) for fr in rattled])
        keep = []
        for fr, (e, f) in zip(rattled, res):
            if np.isfinite(e) and float(np.sqrt((f ** 2).sum(axis=1)).max()) <= s.md_max_force:
                keep.append({**fr, "energy": e, "forces": f})
            else:
                n_rejected += 1
        rattled = keep
        _write_extxyz(out_dir / "detach_rattle.extxyz", None, rattled, label)
    # descriptors (per element averaged) of the ladder MD frames and of the search's relaxation frames
    evd = _evaluator(s, descriptors=True)
    emb = dict(np.load(out_dir / "embeddings.npz")) if (out_dir / "embeddings.npz").is_file() else {}
    sets = {"detach_md": [(fr["symbols"], fr["positions"]) for fr in flat],
            "detach_rattle": [(fr["symbols"], fr["positions"]) for fr in rattled]}
    relax_file = out_dir / "detach_relax.extxyz"
    if relax_file.is_file():
        sets["detach_relax"] = [(x.get_chemical_symbols(), x.get_positions()) for x in read(relax_file, index=":")]
    els = sorted(set(ctx.symbols))
    for key, structs in sets.items():
        if structs:
            labels = evd(structs)
            emb[key] = np.stack([_element_means_fixed(sy, d, els) for (sy, _p), (_e, _f, d) in zip(structs, labels)])
            write_latent(out_dir / f"{key}.extxyz", [d for _e, _f, d in labels], [sy for sy, _p in structs], s)
    emb.setdefault("elements", np.array(els))
    np.savez_compressed(out_dir / "embeddings.npz", **emb)
    _update_manifest(out_dir, {"detach_ensemble": {"levels": levels, "temperatures_K": list(map(float, s.detach_ensemble_temperatures)),
                                                   "ps": s.detach_ensemble_ps, "skip_ps": s.detach_ensemble_skip_ps,
                                                   "stride_fs": s.detach_ensemble_stride_fs, "timestep_fs": dt,
                                                   "tau_fs": s.md_tau_fs, "thermostat": _thermostat_name(s),
                                                   "start_velocities": "Maxwell-Boltzmann at 2T (from a minimum)",
                                                   "rattle_sigmas_A": list(s.detach_rattle_sigmas), "rattle_n": s.detach_rattle_n,
                                                   "strains": list(s.detach_strains), "strain_n": s.detach_strain_n}})
    return {"summary": {"levels": levels, "replicas": len(systems), "frames_md": len(flat),
                        "frames_rattle": len(rattled), "rattle_rejected": n_rejected,
                        "frames_relax": len(sets.get("detach_relax", [])), "seconds_md": round(seconds_md, 1),
                        "stopped": int(sum(x["stopped"] is not None for x in summ))},
            "replicas": dict(zip(names, summ))}


def _min_distance(pts):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(d[:, 1].min())


def rattle_strain(states, s, rng, tries=20):
    """
    Rattled and strained copies of structures [(k, symbols, positions)]: s.detach_rattle_n copies per
    Gaussian width in s.detach_rattle_sigmas (independent displacements of every atom, sigma per
    Cartesian component), and s.detach_strain_n copies per uniform strain in s.detach_strains (the
    structure scaled about its centroid, then rattled by RATTLE_ON_STRAIN). A copy whose shortest
    interatomic distance falls below s.detach_rattle_min_frac of the original's is drawn again (up to
    `tries` times, then dropped). Returns the frames (without labels) and the number dropped.
    """
    out, dropped = [], 0
    for k, sym, pts in states:
        pts = np.asarray(pts, float)
        d_min = s.detach_rattle_min_frac * _min_distance(pts)
        c = pts.mean(axis=0)
        jobs = ([("rattle", sg, 0.0) for sg in s.detach_rattle_sigmas for _ in range(s.detach_rattle_n)] +
                [("strain", RATTLE_ON_STRAIN, eps) for eps in s.detach_strains for _ in range(s.detach_strain_n)])
        for kind, sg, eps in jobs:
            base = c + (pts - c) * (1.0 + eps)
            for _ in range(tries):
                p = base + rng.normal(0.0, sg, pts.shape)
                if _min_distance(p) >= d_min:
                    out.append({"symbols": list(sym), "positions": p,
                                "info": {"source": f"detach_{kind}", "k": int(k), "sigma_A": float(sg), "strain": float(eps)}})
                    break
            else:
                dropped += 1
    return out, dropped


RATTLE_ON_STRAIN = 0.02     # A, displacements on top of a uniform strain


def run_monomer_ensemble(ctx) -> dict:
    """
    Training frames of the free monomers of props/monomers.xyz (the MA diatomic and the Z-type MX_q,
    MACE geometries from the stability step): the MA bond and the M-X bonds of MX_q scanned
    (monomer_scan factors of the equilibrium length; for MX_q also one bond alone, its dissociation),
    rattled copies of MX_q (monomer_rattle_sigmas) and short CSVR MD of MX_q at
    monomer_md_temperatures. The seed comes from the formula, so every record of a material writes
    the same frames (deduplicated when the set is assembled): props/ensemble/monomers.extxyz.
    """
    import hashlib
    from ase.io import read
    s = ctx.settings
    path = ctx.props / "monomers.xyz"
    if not path.is_file():
        return {"summary": {"skipped": "no props/monomers.xyz (stability step)"}}
    out_dir = ctx.props / "ensemble"
    out_dir.mkdir(exist_ok=True)
    frames = []
    counts = {}
    for mol in read(path, index=":"):
        sym, pts = mol.get_chemical_symbols(), mol.get_positions()
        formula = mol.get_chemical_formula()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{s.md_seed}|{formula}".encode()).digest()[:8], "little"))
        m_idx = 0 if len(sym) == 2 else _centre_atom(pts)
        mine = []
        others = [i for i in range(len(sym)) if i != m_idx]
        for f_ in s.monomer_scan:                       # all bonds together
            p = pts.copy()
            p[others] = pts[m_idx] + (pts[others] - pts[m_idx]) * f_
            mine.append({"positions": p, "info": {"source": "monomer_scan", "scale": float(f_), "bonds": "all"}})
        if len(others) > 1:                              # one bond alone: dissociation of MX_q
            for f_ in s.monomer_scan:
                if f_ > 1.0:
                    p = pts.copy()
                    p[others[0]] = pts[m_idx] + (pts[others[0]] - pts[m_idx]) * f_
                    mine.append({"positions": p, "info": {"source": "monomer_scan", "scale": float(f_), "bonds": "one"}})
            for sg in s.monomer_rattle_sigmas:
                for _ in range(s.monomer_rattle_n):
                    mine.append({"positions": pts + rng.normal(0.0, sg, pts.shape),
                                 "info": {"source": "monomer_rattle", "sigma_A": float(sg)}})
        res = _evaluator(s)([(sym, fr["positions"]) for fr in mine])
        for fr, (e, f) in zip(mine, res):
            fr.update({"symbols": sym, "energy": e, "forces": f, "info": {**fr["info"], "monomer": formula}})
        if len(others) > 1 and s.monomer_md_temperatures:  # MD of MX_q (a diatomic has no internal motion to sample)
            dt = float(s.md_timestep_fs)
            n_steps = int(round(s.detach_ensemble_ps * 1000 / dt))
            stride = max(1, int(round(s.detach_ensemble_stride_fs / dt)))
            skip = int(round(s.detach_ensemble_skip_ps * 1000 / dt))
            temps = list(s.monomer_md_temperatures)
            md_frames, _summ = md_replicas(_evaluator(s), [(sym, pts)] * len(temps),
                                           [maxwell_velocities(_masses(sym), 2.0 * T, rng) for T in temps],
                                           [[float(T)] for T in temps], [f"{formula}_{int(T)}K" for T in temps],
                                           s, n_steps, skip, stride, rng,
                                           [{"source": "monomer_md", "monomer": formula}] * len(temps))
            mine += [fr for rep in md_frames for fr in rep]
        counts[formula] = len(mine)
        frames += mine
    _write_extxyz(out_dir / "monomers.extxyz", None, frames, f"MACE-MH-1 {s.head}")
    if frames:
        labels = _evaluator(s, descriptors=True)([(fr["symbols"], fr["positions"]) for fr in frames])
        write_latent(out_dir / "monomers.extxyz", [d for _e, _f, d in labels], [fr["symbols"] for fr in frames], s)
    _update_manifest(out_dir, {"monomers": {"scan": list(s.monomer_scan), "rattle_sigmas_A": list(s.monomer_rattle_sigmas),
                                            "rattle_n": s.monomer_rattle_n,
                                            "md_temperatures_K": list(map(float, s.monomer_md_temperatures))}})
    return {"summary": {"frames": len(frames), "per_monomer": counts}}


def _centre_atom(pts):
    """Index of the atom nearest the centroid (the cation of MX_q)."""
    return int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))


def _element_means_fixed(symbols, desc, elements):
    """Per-element mean descriptor on a fixed element list (NaN for an element the frame lacks)."""
    sym = np.asarray(symbols)
    return np.stack([desc[sym == e].mean(axis=0) if np.any(sym == e) else np.full(desc.shape[1], np.nan)
                     for e in elements]).astype(np.float16)
