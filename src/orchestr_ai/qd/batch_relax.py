# src/orchestr_ai/qd/batch_relax.py
"""
Batched MACE relaxations: many structures, one model pass per optimisation step.

Every structure keeps its own L-BFGS history (the algorithm of ASE's LBFGS:
memory 100, initial inverse Hessian 1/70 Å²/eV, at most 0.2 Å per atom per
step); the energies and forces of all structures still relaxing come from one
batched MACE evaluation, built exactly as mace-torch's ASE calculator builds a
single structure (same cutoff, head and dtype), so a batched energy equals the
serial one.

A structure leaves the batch when it has converged:

  fmax      max |F| <= fmax (0.02 eV/Å, as the serial BFGS runs), or
  plateau   max |F| <= plateau_fmax (0.15 eV/Å) and its energy fell by less than
            plateau_de_atom x N over the last plateau_window steps: a flat
            surface (soft ligand rotations, a floppy shell) where the remaining
            force does almost no work, so more steps buy nothing;

The stop is decided on the energy, not on a looser force threshold: a Cd16Se13Cl6
desorption product had max |F| < 0.10 eV/Å at step 52 while still 0.49 eV above
its minimum (rearranging), 7.0 meV above at 0.05 eV/Å (step 96) and 1.1 meV at
0.02 eV/Å (step 113). The plateau test does not fire while the energy still falls.

or when it stops without converging:

  stalled   the energy has been flat for stall_windows windows while
            max |F| > plateau_fmax (oscillating or stuck),
  max_steps the step budget is spent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

FMAX = 0.02              # eV/Å, converged (as the serial BFGS runs)
PLATEAU_FMAX = 0.15      # eV/Å, flat surface accepted below this
PLATEAU_WINDOW = 20      # steps
PLATEAU_DE_ATOM = 1e-5   # eV per atom over the window (1.5 meV for a 149-atom dot)
STALL_WINDOWS = 3        # flat windows with larger forces before giving up
MAX_STEPS = 1000
MAXSTEP = 0.2            # Å, largest displacement of any atom per step
MEMORY = 100             # L-BFGS history length
ALPHA = 70.0             # eV/Å², initial Hessian guess (inverse: 1/ALPHA)


@dataclass
class RelaxResult:
    positions: np.ndarray
    energy: float
    fmax: float
    steps: int
    reason: str           # fmax | plateau | stalled | max_steps

    @property
    def converged(self) -> bool:
        return self.reason in ("fmax", "plateau")


class BatchMACE:
    """Energies (eV) and forces (eV/Å) of many structures per MACE call, chunked by total atom count."""

    def __init__(self, calc, max_atoms: int = 20000):
        self.calc = calc                      # mace.calculators.MACECalculator (one model)
        self.model = calc.models[0]
        self.max_atoms = int(max_atoms)
        self.n_calls = 0
        self.n_oom = 0                        # chunks split after a CUDA out-of-memory error

    def _graph(self, symbols, positions):
        import ase
        from mace import data as mace_data
        from mace.tools import torch_tools
        c = self.calc
        c.arrays_keys.update({c.charges_key: "charges"})
        keyspec = mace_data.KeySpecification(info_keys=c.info_keys, arrays_keys=c.arrays_keys)
        atoms = ase.Atoms(list(symbols), positions=np.asarray(positions, float))
        with torch_tools.default_dtype(c.default_dtype):
            config = mace_data.config_from_atoms(atoms, key_specification=keyspec, head_name=c.head)
            return mace_data.AtomicData.from_config(config, z_table=c.z_table, cutoff=c.r_max,
                                                    heads=c.available_heads)

    def _chunk(self, structures):
        import torch
        from mace.tools import torch_geometric
        batch = torch_geometric.Batch.from_data_list([self._graph(s, p) for s, p in structures]).to(self.calc.device)
        dtype = next(self.model.parameters()).dtype
        for key in batch.keys:
            value = batch[key]
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(dtype=dtype)
        out = self.model(batch.to_dict(), compute_stress=False, training=False)
        self.n_calls += 1
        e = out["energy"].detach().cpu().numpy().astype(float) * self.calc.energy_units_to_eV
        f = out["forces"].detach().cpu().numpy().astype(float) * (
            self.calc.energy_units_to_eV / self.calc.length_units_to_A)
        sizes = [len(s) for s, _ in structures]
        return [(float(e[i]), f[a:b]) for i, (a, b) in
                enumerate(zip(np.cumsum([0] + sizes[:-1]), np.cumsum(sizes)))]

    def _safe_chunk(self, chunk):
        """One MACE call; on CUDA out-of-memory split the chunk in half, retry, and lower
        max_atoms for the rest of the run (MACE-MH-1 in float64 needs ~5 MB per atom)."""
        import torch
        try:
            return self._chunk(chunk)
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(chunk) // 2
            self.max_atoms = max(1, min(self.max_atoms, sum(len(s) for s, _ in chunk[:half])))
            self.n_oom += 1
            return self._safe_chunk(chunk[:half]) + self._safe_chunk(chunk[half:])

    def __call__(self, structures: Sequence[Tuple[Sequence[str], np.ndarray]]):
        out, chunk, n = [], [], 0
        for s, p in structures:
            if chunk and n + len(s) > self.max_atoms:
                out += self._safe_chunk(chunk)
                chunk, n = [], 0
            chunk.append((s, p))
            n += len(s)
        if chunk:
            out += self._safe_chunk(chunk)
        return out


class _LBFGS:
    """ASE's LBFGS step for one structure (positions as a flat array)."""

    def __init__(self, n_atoms: int):
        self.s, self.y, self.rho = [], [], []
        self.r0 = self.f0 = None
        self.n = n_atoms

    def step(self, r: np.ndarray, f: np.ndarray) -> np.ndarray:
        r, f = r.reshape(-1), f.reshape(-1)
        if self.r0 is not None:
            s0, y0 = r - self.r0, self.f0 - f
            sy = float(np.dot(y0, s0))
            if sy > 1e-12:                    # keep the update positive definite
                self.s.append(s0)
                self.y.append(y0)
                self.rho.append(1.0 / sy)
                if len(self.s) > MEMORY:
                    self.s.pop(0), self.y.pop(0), self.rho.pop(0)
        q = -f.copy()
        a = np.empty(len(self.s))
        for i in range(len(self.s) - 1, -1, -1):
            a[i] = self.rho[i] * np.dot(self.s[i], q)
            q -= a[i] * self.y[i]
        z = q / ALPHA
        for i in range(len(self.s)):
            b = self.rho[i] * np.dot(self.y[i], z)
            z += self.s[i] * (a[i] - b)
        dr = -z.reshape(-1, 3)
        longest = float(np.sqrt((dr ** 2).sum(axis=1)).max())
        if longest > MAXSTEP:
            dr *= MAXSTEP / longest
        self.r0, self.f0 = r.copy(), f.copy()
        return dr


def relax_many(structures: Sequence[Tuple[Sequence[str], np.ndarray]], evaluator, *,
               fmax: float = FMAX, plateau_fmax: float = PLATEAU_FMAX, plateau_window: int = PLATEAU_WINDOW,
               plateau_de_atom: float = PLATEAU_DE_ATOM, stall_windows: int = STALL_WINDOWS,
               max_steps: int = MAX_STEPS) -> List[RelaxResult]:
    """
    Relax all `structures` together; each leaves the batch as soon as it converges
    (fmax or plateau) or stops (stalled, max_steps). `evaluator` maps a list of
    (symbols, positions) to [(energy, forces), ...], e.g. a BatchMACE.
    """
    n = len(structures)
    syms = [list(s) for s, _ in structures]
    pos = [np.asarray(p, float).copy() for _, p in structures]
    opt = [_LBFGS(len(s)) for s in syms]
    hist: List[List[float]] = [[] for _ in range(n)]
    stalls = [0] * n
    done: List[RelaxResult] = [None] * n
    active = list(range(n))
    step = 0
    while active:
        results = evaluator([(syms[i], pos[i]) for i in active])
        still = []
        for i, (e, f) in zip(active, results):
            fm = float(np.sqrt((f ** 2).sum(axis=1)).max())
            h = hist[i]
            h.append(e)
            reason = None
            if fm <= fmax:
                reason = "fmax"
            elif len(h) > plateau_window and (h[-plateau_window - 1] - e) < plateau_de_atom * len(syms[i]):
                if fm <= plateau_fmax:
                    reason = "plateau"
                else:
                    stalls[i] += 1
                    if stalls[i] >= stall_windows:
                        reason = "stalled"
                    else:
                        h[:] = h[-1:]         # start a new window
            if reason is None and step >= max_steps:
                reason = "max_steps"
            if reason:
                done[i] = RelaxResult(pos[i].copy(), e, fm, step, reason)
                continue
            pos[i] = pos[i] + opt[i].step(pos[i], f)
            still.append(i)
        active = still
        step += 1
    return done


def fd_hessian(symbols: Sequence[str], pts: np.ndarray, evaluator, delta: float = 0.01,
               columns: Sequence[int] | None = None) -> np.ndarray:
    """
    Central finite-difference Hessian (eV/Å², shape (3N, 3N)) from batched force
    calls: the 6N displaced copies are independent, so they go through `evaluator`
    (a BatchMACE) many at a time, in blocks small enough that only one block of
    geometries is held at once. Same formula as the serial steps.hessian._fd_hessian.
    `columns` restricts the work to some Cartesian columns (the others stay zero),
    for timing a large structure on a sample.
    """
    pos0 = np.asarray(pts, float)
    n = len(symbols)
    cols = list(range(3 * n)) if columns is None else list(columns)
    h = np.zeros((3 * n, 3 * n))
    per_call = max(1, getattr(evaluator, "max_atoms", 20000) // (2 * n))   # columns per MACE call
    block = per_call * 8
    for b in range(0, len(cols), block):
        ks = cols[b:b + block]
        structures = []
        for k in ks:
            i, a = divmod(k, 3)
            for sgn in (1.0, -1.0):
                p = pos0.copy()
                p[i, a] += sgn * delta
                structures.append((symbols, p))
        res = evaluator(structures)
        for j, k in enumerate(ks):
            h[:, k] = -(res[2 * j][1].ravel() - res[2 * j + 1][1].ravel()) / (2.0 * delta)
    return h
