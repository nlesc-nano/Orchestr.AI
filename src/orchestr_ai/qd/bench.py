# src/orchestr_ai/qd/bench.py
"""
GPU size benchmark for the property pipeline (Phase 1 gate before tier 1).

For CdSe zinc-blende spheres of the requested sizes it times, with MACE-MH-1:

  force      one energy+forces call on one structure
  batched    energy+forces per structure when as many copies as fit in one call
             (batch_atoms) go together (the desorption and Wigner path)
  analytic   autograd Hessian (qdprops uses it up to 300 atoms); 'oom' when it does not fit
  fd         batched central finite-difference Hessian, 6N displaced copies; above
             fd_full_atoms timed on a sample of columns and scaled to all 3N

and fits t = a N^b to each. The spheres are bare cuts of the bulk lattice: the
cost of a MACE call depends on the atom and neighbour counts, not on relaxation.

    python -m orchestr_ai.qd.bench --model macemh1model --sizes 100,300,1000,2000,5000 -o bench.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .batch_relax import BatchMACE, fd_hessian
from .engines import mace_calculator, mace_hessian, resolve_device, sync_device

A_CDSE = 6.05     # Å, zinc-blende lattice constant


def cdse_sphere(n_atoms: int):
    """The n_atoms zinc-blende CdSe sites nearest to a Se atom."""
    m = int(np.ceil((n_atoms / 8.0) ** (1 / 3))) + 2
    fcc = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    cells = np.array([[i, j, k] for i in range(-m, m + 1) for j in range(-m, m + 1) for k in range(-m, m + 1)])
    se = (cells[:, None, :] + fcc[None]).reshape(-1, 3) * A_CDSE
    cd = se + A_CDSE * 0.25
    pts = np.vstack([se, cd])
    sym = np.array(["Se"] * len(se) + ["Cd"] * len(cd))
    order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")[:n_atoms]
    return list(sym[order]), pts[order]


def _timed(fn, device, repeat=1):
    best, out = None, None
    for _ in range(repeat):
        sync_device(device)
        t0 = time.perf_counter()
        out = fn()
        sync_device(device)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, out


def _peak_reset(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_gb(device):
    if str(device).startswith("cuda"):
        import torch
        return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return None


def _is_oom(exc) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def bench(sizes, model, head="omat_pbe", device="auto", dtype="float64", batch_atoms=20000,
          analytic_max=2000, fd_full_atoms=300, fd_sample_columns=96, delta=0.01, log=print) -> dict:
    from ase import Atoms
    device = resolve_device(device)
    calc = mace_calculator(head, model, device, dtype)
    ev = BatchMACE(calc, batch_atoms)
    rows = []
    sym, pts = cdse_sphere(min(sizes))
    ev([(sym, pts)])                                            # warm-up (kernels, compilation)
    for n in sizes:
        sym, pts = cdse_sphere(n)
        row = {"n_atoms": n}
        _peak_reset(device)
        row["force_s"], _ = _timed(lambda: ev([(sym, pts)]), device, repeat=3)
        row["force_peak_gb"] = _peak_gb(device)
        k = max(1, batch_atoms // n)
        if k > 1:
            t, _ = _timed(lambda: ev([(sym, pts)] * k), device, repeat=2)
            row["batched_s_per_structure"], row["batch_size"] = t / k, k
        if n <= analytic_max:
            atoms = Atoms(sym, positions=pts)
            _peak_reset(device)
            try:
                t, h_an = _timed(lambda: mace_hessian(atoms, calc), device)
                row["analytic_s"], row["analytic_peak_gb"] = t, _peak_gb(device)
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                row["analytic_s"], h_an = "oom", None
                _peak_reset(device)
        else:
            h_an = None
        cols = None if n <= fd_full_atoms else list(np.linspace(0, 3 * n - 1, min(3 * n, fd_sample_columns)).astype(int))
        _peak_reset(device)
        t, h_fd = _timed(lambda: fd_hessian(sym, pts, ev, delta, columns=cols), device)
        row["fd_s"] = t if cols is None else t * 3 * n / len(cols)
        row["fd_sampled_columns"] = None if cols is None else len(cols)
        row["fd_peak_gb"] = _peak_gb(device)
        if cols is None and isinstance(h_an, np.ndarray):
            row["fd_vs_analytic_max_abs"] = float(np.abs(h_fd - h_an).max())     # eV/Å²
        rows.append(row)
        log(json.dumps(row))
    fits = {}
    for key in ("force_s", "batched_s_per_structure", "analytic_s", "fd_s"):
        xy = [(r["n_atoms"], r[key]) for r in rows if isinstance(r.get(key), float) and r[key] > 0]
        if len(xy) >= 2:
            b, loga = np.polyfit(np.log([x for x, _ in xy]), np.log([y for _, y in xy]), 1)
            fits[key] = {"a": float(np.exp(loga)), "b": float(b), "n_points": len(xy)}
    import torch
    gpu = torch.cuda.get_device_name(0) if str(device).startswith("cuda") else device
    return {"device": gpu, "dtype": dtype, "head": head, "model": model, "batch_atoms": batch_atoms,
            "fd_delta": delta, "rows": rows, "power_law_fits": fits}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default="omat_pbe")
    ap.add_argument("--sizes", default="100,300,1000,2000,5000")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--batch-atoms", type=int, default=20000)
    ap.add_argument("--analytic-max", type=int, default=2000)
    ap.add_argument("--fd-full-atoms", type=int, default=300)
    ap.add_argument("--fd-sample-columns", type=int, default=96)
    ap.add_argument("-o", "--output", default="bench.json")
    a = ap.parse_args(argv)
    res = bench([int(x) for x in a.sizes.split(",")], a.model, a.head, a.device, a.dtype, a.batch_atoms,
                a.analytic_max, a.fd_full_atoms, a.fd_sample_columns)
    open(a.output, "w").write(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["power_law_fits"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
