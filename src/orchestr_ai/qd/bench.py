# src/orchestr_ai/qd/bench.py
"""
GPU size benchmark for the property pipeline (Phase 1 gate before tier 1).

For CdSe zinc-blende spheres of the requested sizes it times, with MACE-MH-1:

  force      one energy+forces call on one structure
  batched    energy+forces per structure when as many copies as fit in one call
             (batch_atoms) go together (the desorption and Wigner path)
  analytic   autograd Hessian (qdprops uses it up to 300 atoms), timed on 48 sampled rows and
             scaled to 3N; 'oom' when the force graph does not fit
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


def analytic_rows(ev, sym, pts, rows):
    """Rows of the analytic (autograd) Hessian, eV/A^2: one forward pass with the force graph
    kept, one backward pass per row. Timing a sample of rows and scaling to 3N gives the cost
    of the full analytic Hessian without computing it."""
    import torch
    from mace.tools import torch_geometric
    batch = torch_geometric.Batch.from_data_list([ev._graph(sym, pts)]).to(ev.calc.device)
    dtype = next(ev.model.parameters()).dtype
    for key in batch.keys:
        if torch.is_tensor(batch[key]) and torch.is_floating_point(batch[key]):
            batch[key] = batch[key].to(dtype=dtype)
    d = batch.to_dict()
    out = ev.model(d, compute_force=True, compute_stress=False, training=True)
    F = out["forces"].reshape(-1)
    pos = d["positions"]
    H = []
    for j in rows:
        g, = torch.autograd.grad(-F[j], pos, retain_graph=True)
        H.append(g.reshape(-1).detach().cpu().numpy())
    return np.array(H)


def bench(sizes, model, head="omat_pbe", device="auto", dtype="float64", batch_atoms=8000,
          analytic_max=5000, analytic_rows_sampled=48, fd_full_atoms=300, fd_sample_columns=96, delta=0.01,
          log=print, output=None) -> dict:
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
        if row["force_peak_gb"]:
            row["force_mb_per_atom"] = round(1e3 * row["force_peak_gb"] / n, 3)
        k = max(1, ev.max_atoms // n)
        if k > 1:
            _peak_reset(device)
            t, _ = _timed(lambda: ev([(sym, pts)] * k), device, repeat=2)
            row["batched_s_per_structure"], row["batch_size"] = t / k, k
            row["batched_peak_gb"] = _peak_gb(device)
            row["max_atoms_per_call"] = ev.max_atoms        # lowered if the GPU ran out of memory
        h_an = None
        if n <= analytic_max:
            k_rows = list(np.linspace(0, 3 * n - 1, min(3 * n, analytic_rows_sampled)).astype(int))
            _peak_reset(device)
            try:
                t, _ = _timed(lambda: analytic_rows(ev, sym, pts, k_rows), device)
                row["analytic_s"] = t * 3 * n / len(k_rows)          # forward pass included once per sample
                row["analytic_sampled_rows"], row["analytic_peak_gb"] = len(k_rows), _peak_gb(device)
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                row["analytic_s"] = "oom"
            _peak_reset(device)
            if n == min(sizes) and n <= fd_full_atoms:              # one full check of analytic vs FD
                h_an = mace_hessian(Atoms(sym, positions=pts), calc)
        cols = None if n <= fd_full_atoms else list(np.linspace(0, 3 * n - 1, min(3 * n, fd_sample_columns)).astype(int))
        _peak_reset(device)
        t, h_fd = _timed(lambda: fd_hessian(sym, pts, ev, delta, columns=cols), device)
        row["fd_s"] = t if cols is None else t * 3 * n / len(cols)
        row["fd_sampled_columns"] = None if cols is None else len(cols)
        row["fd_peak_gb"] = _peak_gb(device)
        if h_an is not None and cols is None:
            row["fd_vs_analytic_max_abs"] = float(np.abs(h_fd - h_an).max())     # eV/Å²
        rows.append(row)
        log(json.dumps(row), flush=True)
        if output:                                             # keep what is done if the job is stopped
            open(output, "w").write(json.dumps({"rows": rows}, indent=1) + "\n")
    fits = {}
    for key in ("force_s", "batched_s_per_structure", "analytic_s", "fd_s"):
        xy = [(r["n_atoms"], r[key]) for r in rows if isinstance(r.get(key), float) and r[key] > 0]
        if len(xy) >= 2:
            b, loga = np.polyfit(np.log([x for x, _ in xy]), np.log([y for _, y in xy]), 1)
            fits[key] = {"a": float(np.exp(loga)), "b": float(b), "n_points": len(xy)}
    import torch
    gpu = torch.cuda.get_device_name(0) if str(device).startswith("cuda") else device
    return {"device": gpu, "dtype": dtype, "head": head, "model": model, "batch_atoms": batch_atoms,
            "max_atoms_per_call": ev.max_atoms, "oom_splits": ev.n_oom,
            "fd_delta": delta, "rows": rows, "power_law_fits": fits}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", default="omat_pbe")
    ap.add_argument("--sizes", default="100,300,1000,2000,5000")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--batch-atoms", type=int, default=8000)
    ap.add_argument("--analytic-max", type=int, default=5000)
    ap.add_argument("--fd-full-atoms", type=int, default=300)
    ap.add_argument("--fd-sample-columns", type=int, default=96)
    ap.add_argument("-o", "--output", default="bench.json")
    a = ap.parse_args(argv)
    res = bench([int(x) for x in a.sizes.split(",")], a.model, a.head, a.device, a.dtype, a.batch_atoms,
                a.analytic_max, fd_full_atoms=a.fd_full_atoms, fd_sample_columns=a.fd_sample_columns,
                output=a.output)
    open(a.output, "w").write(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["power_law_fits"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
