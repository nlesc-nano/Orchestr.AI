# src/orchestr_ai/qd/__main__.py
"""
Command line:

    python -m orchestr_ai.qd run <record_dir> [--steps relax,structure,hessian,electronic,stability,detachment,report] [--head omat_pbe]
    python -m orchestr_ai.qd batch <library tree> [--max-atoms N] [--steps ...]

A record directory holds record.json and start.xyz (library generator output
or the webapp's public tree); results go to <record_dir>/props/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import STEPS
from .run import Settings, run_record


def _settings(args) -> Settings:
    s = Settings(head=args.head, device=args.device, dtype=args.dtype, fmax=args.fmax,
                 hessian=args.hessian, xtb_ip_ea=not args.no_ip_ea, solvation_checks=args.solvation_checks,
                 sites_hessian_max_atoms=args.site_hessians, sites_solvation=args.site_solvation)
    if args.model:
        s.model = args.model
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qdprops", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "batch"):
        p = sub.add_parser(name)
        p.add_argument("path", help="record directory (run) or library tree (batch)")
        p.add_argument("--steps", default=",".join(STEPS))
        p.add_argument("--head", default=Settings.head)
        p.add_argument("--model", default=None, help="MACE model file (default: QDPROPS_MACE_MODEL or ~/.cache/mace/macemh1model)")
        p.add_argument("--device", default="auto", help="auto (CUDA GPU, else Apple GPU, else CPU), cuda, mps or cpu")
        p.add_argument("--dtype", default="float64")
        p.add_argument("--fmax", type=float, default=Settings.fmax)
        p.add_argument("--hessian", choices=("auto", "analytic", "fd"), default="auto")
        p.add_argument("--no-ip-ea", action="store_true", help="skip the charged xtb single points")
        p.add_argument("--solvation-checks", action="store_true",
                       help="also compute ddCOSMO (eps 2.4, 80) and ALPB solvation per structure, as checks")
        p.add_argument("--site-hessians", type=int, default=Settings.sites_hessian_max_atoms, metavar="N",
                       help="binding-site map: a MACE Hessian per site class for dots of up to N atoms "
                            "(default 0: reuse the first desorption step's thermal part)")
        p.add_argument("--site-solvation", choices=("scf", "frozen"), default=Settings.sites_solvation,
                       help="binding-site map solvation: a GFN2-xTB single point per site (scf) or the dot's charges")
        p.add_argument("--cif", default=None, help="bulk CIF (default: resolved from record.origin.cif)")
        p.add_argument("--force", action="store_true", help="ignore cached step results")
        if name == "batch":
            p.add_argument("--max-atoms", type=int, default=None)
    p = sub.add_parser("synthesis", help="multi-dot synthesis dashboard from records with props/solution.json")
    p.add_argument("records", nargs="+", help="record directories (run first, so props/solution.json exists)")
    p.add_argument("-o", "--out", default="synthesis.html")
    args = ap.parse_args(argv)
    if args.cmd == "synthesis":
        from .synthesis import write_synthesis
        write_synthesis([Path(r) for r in args.records], Path(args.out))
        return 0
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = set(steps) - set(STEPS)
    if bad:
        ap.error(f"unknown steps: {', '.join(sorted(bad))}")
    settings = _settings(args)

    if args.cmd == "run":
        summary = run_record(Path(args.path), steps, settings, cif=args.cif, force=args.force)
        print(json.dumps(summary["summary"], indent=1))
        return 0

    records = sorted(p.parent for p in Path(args.path).rglob("record.json"))
    failed = []
    for rd in records:
        n = json.loads((rd / "record.json").read_text()).get("n_atoms", 0)
        if args.max_atoms and n > args.max_atoms:
            continue
        try:
            run_record(rd, steps, settings, cif=args.cif, force=args.force)
        except Exception as exc:  # keep the batch going
            failed.append(rd.name)
            print(f"[qdprops] {rd.name}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"[qdprops] batch: {len(records)} records, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
