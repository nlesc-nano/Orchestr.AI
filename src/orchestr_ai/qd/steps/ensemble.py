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

md      Short Langevin MD (BAOAB) with MACE-MH-1. Every replica follows its own
        temperature schedule, constant ([T]) or a linear ramp ([T0, T1]); all replicas
        share one batched model call per step, so six replicas cost little more than
        one. Replicas start from Wigner samples at their first temperature and are
        stopped if they become unphysical (forces above md_max_force, or an atom more
        than 10 A outside the starting structure). Frames every md_stride_fs after
        md_skip_ps carry their exact MACE energy and forces.

Every structure gets energy, forces and MACE invariant descriptors (per element
averaged; per atom for the optimised structure), labelled in batched calls. Files go
to props/ensemble/: optimised.xyz, wigner_<T>K.extxyz, md_<name>.extxyz,
embeddings.npz and manifest.json.
"""
from __future__ import annotations

import json
import time

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


def _write_extxyz(path, symbols, frames):
    """frames: dicts with positions, forces, energy (+ velocities, info)."""
    from ase import Atoms
    from ase.io import write
    out = []
    for fr in frames:
        a = Atoms(symbols, positions=fr["positions"])
        a.arrays["forces"] = np.asarray(fr["forces"], float)
        if fr.get("velocities") is not None:
            a.arrays["vel_A_fs"] = np.asarray(fr["velocities"], float)
        a.info.update({"energy": float(fr["energy"]), **fr.get("info", {})})
        out.append(a)
    write(str(path), out, format="extxyz")


def _evaluator(s, descriptors=False):
    from ..batch_relax import BatchMACE
    calc = mace_calculator(s.head, s.model, s.device, s.dtype)
    return BatchMACE(calc, s.relax_batch_atoms or 8000, descriptors=descriptors)


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
    t0 = time.time()
    (e0, f0, d0), = _label(ev, symbols, [pos0])
    _write_extxyz(out_dir / "optimised.xyz", symbols, [{"positions": pos0, "forces": f0, "energy": e0}])
    emb = {"optimised_per_atom": d0.astype(np.float16)}
    rng = np.random.default_rng(s.wigner_seed)
    per_T = {}
    for T in s.wigner_temperatures:
        P, V = wigner_samples(pos0, masses, freqs, modes, T, s.wigner_samples, rng,
                              s.wigner_cutoff_cm, s.wigner_soft_cm)
        lab = _label(ev, symbols, P)
        frames = [{"positions": P[k], "velocities": V[k], "forces": f, "energy": e,
                   "info": {"T_K": float(T), "sample": k, "source": "wigner"}} for k, (e, f, _d) in enumerate(lab)]
        _write_extxyz(out_dir / f"wigner_{int(T)}K.extxyz", symbols, frames)
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


def _schedule_T(sched, t_frac):
    return float(sched[0]) if len(sched) == 1 else float(sched[0] + (sched[1] - sched[0]) * t_frac)


def run_md(ctx) -> dict:
    s = ctx.settings
    if not s.md_enabled:
        return {"summary": {"skipped": "md_enabled is false (MD runs on a subset of the library)"}}
    out_dir = ctx.props / "ensemble"
    out_dir.mkdir(exist_ok=True)
    symbols, pos0 = ctx.relaxed()
    symbols, pos0 = list(symbols), np.asarray(pos0, float)
    n_at = len(symbols)
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
    R = len(scheds)
    X = np.empty((R, n_at, 3))
    V = np.empty((R, n_at, 3))
    for r, sc in enumerate(scheds):
        P, Vw = wigner_samples(pos0, masses, freqs, modes, sc[0], 1, rng, s.wigner_cutoff_cm, s.wigner_soft_cm)
        X[r], V[r] = P[0], Vw[0]
    ev = _evaluator(s)
    inv_m = (ACC / masses)[None, :, None]
    c1 = np.exp(-s.md_friction_fs * dt)
    c2 = np.sqrt(1 - c1 ** 2)
    active = np.ones(R, bool)
    stop_reason = [None] * R
    frames = [[] for _ in range(R)]
    centre0, radius0 = pos0.mean(axis=0), np.linalg.norm(pos0 - pos0.mean(axis=0), axis=1).max()

    def forces(idx):
        res = ev([(symbols, X[r]) for r in idx])
        return {r: (e, f) for r, (e, f) in zip(idx, res)}

    t0 = time.time()
    cur = forces(list(range(R)))
    T_inst_sum = np.zeros(R)
    n_inst = np.zeros(R)
    for step in range(n_steps + 1):
        idx = [r for r in range(R) if active[r]]
        if not idx:
            break
        if step > 0:
            for r in idx:                      # B A O A
                V[r] += 0.5 * dt * cur[r][1] * inv_m[0]
                X[r] += 0.5 * dt * V[r]
                kT = KB_EV * _schedule_T(scheds[r], step / n_steps)
                sig = np.sqrt(kT * ACC / masses)[:, None]
                V[r] = c1 * V[r] + c2 * sig * rng.standard_normal((n_at, 3))
                X[r] += 0.5 * dt * V[r]
            cur.update(forces(idx))
            for r in idx:                      # B
                V[r] += 0.5 * dt * cur[r][1] * inv_m[0]
                p = (masses[:, None] * V[r]).sum(axis=0) / masses.sum()
                V[r] -= p                      # no centre-of-mass drift from the noise
        for r in idx:
            e, f = cur[r]
            fmax = float(np.sqrt((f ** 2).sum(axis=1)).max())
            far = float(np.linalg.norm(X[r] - centre0, axis=1).max())
            if not np.isfinite(e) or fmax > s.md_max_force or far > radius0 + 10.0:
                active[r] = False
                stop_reason[r] = (f"step {step}: max |F| {fmax:.1f} eV/A" if fmax > s.md_max_force
                                  else f"step {step}: atom {far - radius0:.1f} A outside the start")
                continue
            T_inst = float((masses[:, None] * V[r] ** 2).sum() / ACC / (3 * n_at - 3) / KB_EV)
            if step >= skip:
                T_inst_sum[r] += T_inst
                n_inst[r] += 1
            if step >= skip and (step - skip) % stride == 0:
                frames[r].append({"positions": X[r].copy(), "velocities": V[r].copy(), "forces": f.copy(),
                                  "energy": e, "info": {"time_fs": step * dt, "T_target_K":
                                                        _schedule_T(scheds[r], step / n_steps),
                                                        "T_inst_K": T_inst, "source": "md", "replica": names[r]}})
    seconds_md = time.time() - t0
    evd = _evaluator(s, descriptors=True)
    emb = dict(np.load(out_dir / "embeddings.npz")) if (out_dir / "embeddings.npz").is_file() else {}
    per = {}
    for r, name in enumerate(names):
        if frames[r]:
            _write_extxyz(out_dir / f"md_{name}.extxyz", symbols, frames[r])
            lab = _label(evd, symbols, [fr["positions"] for fr in frames[r]])
            emb[f"md_{name}"] = np.stack([_element_means(symbols, d)[1] for _e, _f, d in lab])
        rmsd = [float(np.sqrt(((fr["positions"] - pos0) ** 2).sum(axis=1).mean())) for fr in frames[r]]
        per[name] = {"schedule_K": scheds[r], "frames": len(frames[r]),
                     "T_inst_mean_K": float(T_inst_sum[r] / max(n_inst[r], 1)),
                     "rmsd_max_A": max(rmsd) if rmsd else None, "stopped": stop_reason[r]}
    emb.setdefault("elements", np.array(sorted(set(symbols))))
    np.savez_compressed(out_dir / "embeddings.npz", **emb)
    _update_manifest(out_dir, {"md": {"schedule_K": scheds, "names": names, "ps": s.md_ps, "timestep_fs": dt,
                                      "friction_fs": s.md_friction_fs, "stride_fs": s.md_stride_fs,
                                      "skip_ps": s.md_skip_ps, "seed": s.md_seed, "thermostat": "Langevin BAOAB"}})
    return {"summary": {"replicas": R, "frames": int(sum(len(f) for f in frames)), "steps": n_steps,
                        "timestep_fs": dt, "seconds_md": round(seconds_md, 1),
                        "ms_per_step": round(1e3 * seconds_md / max(n_steps, 1), 2),
                        "stopped": int(sum(x is not None for x in stop_reason))},
            "replicas": per}
