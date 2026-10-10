# src/orchestr_ai/qd/run.py
"""
Step runner for one library record.

Each step reads the record context (and the outputs of earlier steps), returns
a JSON-serialisable dict and may write arrays/files into `<id>/props/`.  Its
result is stored as `props/<step>.json` together with a hash of everything it
depends on (step settings, input geometry, upstream results); a later run
with the same hash reuses it.  `properties.json` collects the step summaries
and the provenance for the webapp.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .records import read_xyz_first_frame

from . import SCHEMA_VERSION, STEPS
from .engines import DEFAULT_HEAD, DEFAULT_MODEL

REPO = Path(__file__).resolve().parents[3]


def _cif_dirs() -> List[Path]:
    """Bulk CIFs: $QDPROPS_CIF_DIRS (os.pathsep-separated), then the QD_builder checkout's examples."""
    dirs = [Path(p) for p in os.environ.get("QDPROPS_CIF_DIRS", "").split(os.pathsep) if p]
    try:
        import builder
        root = Path(builder.__file__).resolve().parents[2]
        dirs += [root / "examples/library/cifs", root / "examples/cifs"]
    except ImportError:
        pass
    return dirs


CIF_DIRS = _cif_dirs()
# Steps whose results a step reads (their hashes enter its cache key).
DEPS = {
    "relax": [], "structure": ["relax"], "hessian": ["relax", "structure"],
    "vibspec": ["relax", "structure", "hessian"], "electronic": ["relax", "structure"],
    "stability": ["structure", "hessian"], "detachment": ["structure", "hessian"],
    "solvation": ["relax", "detachment"],
    "sites": ["relax", "structure", "hessian", "detachment", "solvation"],
    "report": ["relax", "structure", "hessian", "vibspec", "stability", "detachment", "solvation", "sites"],
    "wigner": ["relax", "structure", "hessian"], "md": ["relax", "hessian", "wigner"],
}
# Source files whose content enters a step's cache key (engine versions are in the provenance).
CODE_DEPS = {
    "relax": ["steps/relax.py"], "structure": ["steps/structure.py"], "hessian": ["steps/hessian.py"],
    "vibspec": ["steps/vibspec.py", "bulk.py"],
    "electronic": ["steps/electronic.py"], "stability": ["steps/stability.py", "references.py"],
    "detachment": ["steps/detachment.py", "references.py"], "solvation": ["steps/solvation.py"], "sites": ["steps/sites.py"],
    "report": ["steps/report.py", "steps/vibplots.py", "bulk.py", "solution.py", "dashboards.py"],
    "wigner": ["steps/ensemble.py"], "md": ["steps/ensemble.py"],
}
FORMAL_CHARGES = {
    "Cd": 2, "Zn": 2, "Pb": 2, "Hg": 2, "In": 3, "Ga": 3, "Al": 3, "Cs": 1, "Rb": 1,
    "S": -2, "Se": -2, "Te": -2, "O": -2, "P": -3, "As": -3, "Sb": -3,
    "F": -1, "Cl": -1, "Br": -1, "I": -1,
}


@dataclass
class Settings:
    head: str = DEFAULT_HEAD
    model: str = DEFAULT_MODEL
    device: str = "auto"            # auto: CUDA GPU, else Apple GPU (MPS, float32), else CPU
    dtype: str = "float64"
    fmax: float = 0.01               # eV/Å, relaxation convergence
    max_steps: int = 2000
    relax_bfgs_max_atoms: int = 500  # BFGS up to here, L-BFGS above (BFGS is O(N^3) per step on the CPU)
    hessian: str = "auto"            # auto | analytic | fd
    analytic_max_atoms: int = 2000   # autograd Hessian up to here (43 min on an A100 at 2000); FD above
    hessian_max_atoms: int = 2000    # larger dots: relax, structure and single points only (no Hessian or its dependents)
    fd_step: float = 0.01            # Å
    temperatures: List[float] = field(default_factory=lambda: [float(t) for t in range(50, 801, 25)])
    vdos_sigma: float = 5.0          # cm-1, Gaussian broadening
    vibspec_step: float = 0.1        # amu^1/2 Å, normal-coordinate displacement for the derivatives
    vibspec_field: float = 0.02      # V/Å, finite field for the polarisability
    vibspec_max_atoms: int = 300
    vibspec_workers: int = 0         # concurrent single-threaded g-xTB runs (0: cpu count - 2, at most 12)
    xtb_method: str = "auto"        # auto (g-xTB, GFN2 if g-xTB fails) | gxtb | gfn2; electronic, solvation, sites
    gxtb_max_atoms: int = 500       # "auto": GFN2 only above this size (g-xTB 2.0.1 diverges on larger dots)
    solvation_checks: bool = False   # also run ddCOSMO (eps 2.4, 80) and ALPB checks per structure
    xtb_ip_ea: bool = True
    xtb_gradient: bool = True
    detach_max_atoms: int = 600      # desorption search (and the site map built on it) up to this size
    detach_max_steps: int = 4        # stepwise MX_q removals
    detach_max_candidates: int = 12  # symmetry-unique sites relaxed per step
    detach_thermo_max_atoms: int = 300  # Hessians of the products up to this size
    detach_perturb_A: float = 0.02      # random displacement (sigma, A) of every candidate's start: leaves flat valleys
    detach_seed: int = 0
    relax_batch_atoms: int = 4000       # desorption candidates relaxed together, at most this many atoms per
                                        # MACE call (batched L-BFGS, see batch_relax); 0: one at a time (ASE BFGS).
                                        # MACE-MH-1 in float64 needs 11.4 MB per atom (A100 benchmark): 4,000 atoms (46 GB)
                                        # (larger chunks are split automatically when the GPU runs out of memory)
    sites_hessian_max_atoms: int = 0    # per-site Hessians for the binding-site map up to this size (0: never;
                                        # otherwise the first path step's thermal part is used for every site)
    sites_solvation: str = "scf"        # scf: a GFN2-xTB single point per site | frozen: the intact dot's charges
    mu_grid: List[float] = field(default_factory=lambda: [round(-3.0 + 0.02 * i, 4) for i in range(201)])
    # ensembles for DFT labelling (steps/ensemble.py)
    wigner_temperatures: List[float] = field(default_factory=lambda: [300.0, 400.0, 500.0, 600.0])
    wigner_samples: int = 100           # per temperature
    wigner_cutoff_cm: float = 20.0      # modes below are not sampled (translations, rotations, numerical noise)
    wigner_soft_cm: float = 50.0        # modes below are sampled with the width of a mode at this frequency
    wigner_seed: int = 0
    md_enabled: bool = False            # MD runs on a subset of the library: switch on per job
    md_schedule: List = field(default_factory=lambda: [[300.0], [400.0], [500.0], [600.0],
                                                       [300.0, 1000.0], [300.0, 1000.0]])  # [T] or ramp [T0, T1]
    md_ps: float = 5.0                  # length of every replica
    md_timestep_fs: float = 2.0         # fs; the library's dots are heavy elements only
    md_friction_fs: float = 0.01        # Langevin friction (1/fs; 100 fs relaxation time)
    md_stride_fs: float = 100.0         # one stored frame every stride
    md_skip_ps: float = 0.5             # not stored at the start
    md_max_force: float = 50.0          # eV/A: a replica above this is stopped (model out of its domain)
    md_seed: int = 0

    def relaxer(self) -> str:
        """How desorption products are relaxed (enters the step hashes and the relaxation cache)."""
        if not self.relax_batch_atoms:
            return "serial-bfgs-0.02"
        from . import batch_relax as b
        from .steps import detachment as d
        return (f"batched-lbfgs-{b.FMAX}-plateau-{b.PLATEAU_FMAX}-{b.PLATEAU_WINDOW}-{b.PLATEAU_DE_ATOM}"
                f"-stall-{b.STALL_WINDOWS}-{b.MAX_STEPS}-polish-{d.POLISH_FMAX}-{d.POLISH_WINDOW}")

    def for_step(self, step: str) -> dict:
        from .engines import resolve_device
        dev = resolve_device(self.device)
        # The device is provenance, not a cache key: a CPU job reuses the GPU job's MACE steps
        # (float64 on either; the Apple GPU runs float32, which the dtype records).
        mace = {"head": self.head, "model": Path(self.model).name,
                "dtype": "float32" if dev == "mps" else self.dtype}
        return {
            "relax": {**mace, "fmax": self.fmax, "max_steps": self.max_steps,
                      **({"bfgs_max_atoms": self.relax_bfgs_max_atoms} if self.relax_bfgs_max_atoms != 500 else {})},
            "structure": {},
            "hessian": {**mace, "hessian": self.hessian, "analytic_max_atoms": self.analytic_max_atoms,
                        "fd_step": self.fd_step, "temperatures": self.temperatures,
                        "vdos_sigma": self.vdos_sigma},
            "vibspec": {**mace, "method": "gxtb", "acc": "0.01", "step": self.vibspec_step,
                        "field": self.vibspec_field, "max_atoms": self.vibspec_max_atoms, "cif": "record"},
            "electronic": {"method": self.xtb_method, "gxtb_max_atoms": self.gxtb_max_atoms, "ip_ea": self.xtb_ip_ea, "gradient": self.xtb_gradient},
            "stability": {**mace, "temperatures": self.temperatures, "cif": "record"},
            "detachment": {**mace, "fmax": self.fmax, "max_steps": self.detach_max_steps, "relaxer": self.relaxer(),
                           "max_atoms": self.detach_max_atoms,
                           "max_candidates": self.detach_max_candidates,
                           **({"perturb_A": self.detach_perturb_A, "seed": self.detach_seed}
                              if self.detach_perturb_A else {}),
                           "thermo_max_atoms": self.detach_thermo_max_atoms, "mu_grid": self.mu_grid,
                           "temperatures": self.temperatures},
            "solvation": {"method": self.xtb_method, "gxtb_max_atoms": self.gxtb_max_atoms, "checks": self.solvation_checks},
            "sites": {**mace, "method": self.xtb_method, "gxtb_max_atoms": self.gxtb_max_atoms, "relaxer": self.relaxer(), "thermo_max_atoms": self.detach_thermo_max_atoms,
                      "hessian_max_atoms": self.sites_hessian_max_atoms, "solvation": self.sites_solvation},
            "report": {},
            "wigner": {**mace, "temperatures": self.wigner_temperatures, "samples": self.wigner_samples,
                       "cutoff_cm": self.wigner_cutoff_cm, "soft_cm": self.wigner_soft_cm, "seed": self.wigner_seed},
            "md": {**mace, "enabled": self.md_enabled, "schedule": self.md_schedule, "ps": self.md_ps,
                   "timestep_fs": self.md_timestep_fs, "friction_fs": self.md_friction_fs,
                   "stride_fs": self.md_stride_fs, "skip_ps": self.md_skip_ps, "max_force": self.md_max_force,
                   "seed": self.md_seed},
        }[step]


@dataclass
class Context:
    record_dir: Path
    record: dict
    symbols: List[str]
    start_pts: np.ndarray
    charges: Dict[str, int]
    native: List[str]
    cif: Optional[Path]
    settings: Settings
    results: Dict[str, dict] = field(default_factory=dict)

    @property
    def props(self) -> Path:
        p = self.record_dir / "props"
        p.mkdir(exist_ok=True)
        return p

    @property
    def ligands(self) -> List[str]:
        return sorted(set(self.symbols) - set(self.native))

    def relaxed(self):
        """Symbols and relaxed coordinates (props/relaxed.xyz)."""
        return read_xyz_first_frame(str(self.props / "relaxed.xyz"))


def load_context(record_dir: Path, settings: Settings, cif: Optional[str] = None) -> Context:
    record_dir = Path(record_dir).resolve()
    rec = json.loads((record_dir / "record.json").read_text())
    start = next((s["file"] for s in rec.get("stages", []) if s.get("stage") == "start"), "start.xyz")
    start_path = record_dir / Path(start).name
    symbols, pts = read_xyz_first_frame(str(start_path))
    recipe = (rec.get("origin") or {}).get("recipe") or {}
    charges = {k: int(v) for k, v in (recipe.get("charges") or {}).items()}
    for s in set(symbols):
        charges.setdefault(s, FORMAL_CHARGES.get(s, 0))
    native = [e for e in rec.get("core", {}) if e in set(symbols)]
    return Context(record_dir=record_dir, record=rec, symbols=list(symbols), start_pts=np.asarray(pts, float),
                   charges=charges, native=native, cif=_resolve_cif(rec, cif, record_dir), settings=settings)


def _resolve_cif(rec: dict, override: Optional[str], record_dir: Path) -> Optional[Path]:
    if override:
        return Path(override).resolve()
    name = (rec.get("origin") or {}).get("cif")
    if not name:
        return None
    dirs = list(CIF_DIRS)
    # webapp layout: <public>/<family>/<material>/builder*/<id>/ -> <public>/<family>/bulk_cifs
    if len(record_dir.parents) > 3:
        dirs.append(record_dir.parents[2] / "bulk_cifs")
    for d in dirs:
        if (d / name).is_file():
            return d / name
    return None


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _geometry_hash(path: Path) -> str:
    """Short sha256 of a file's bytes (geometry or source)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else "-"


def _step_inputs(ctx: Context, step: str) -> dict:
    deps = DEPS[step]
    start = ctx.record_dir / "start.xyz"
    here = Path(__file__).resolve().parent
    return {
        "schema": SCHEMA_VERSION,
        "code": "".join(_geometry_hash(here / f) for f in CODE_DEPS[step]),
        "settings": ctx.settings.for_step(step),
        "start": _geometry_hash(start),
        "upstream": {d: ctx.results.get(d, {}).get("_hash") for d in deps},
    }


def run_record(record_dir: Path, steps: Sequence[str] = STEPS, settings: Optional[Settings] = None,
               cif: Optional[str] = None, force: bool = False, log: Callable[[str], None] = print,
               upstream_cached: bool = False) -> dict:
    """
    Run (or load from cache) `steps` and the steps they depend on for one record.
    With `upstream_cached`, dependencies that were not requested must already be cached:
    the CPU stage (xTB steps) never redoes the GPU stage's MACE steps.
    """
    from .steps import (detachment, electronic, ensemble, hessian, relax, report, sites, solvation, stability,
                        structure, vibspec)
    impl = {"wigner": ensemble.run_wigner, "md": ensemble.run_md,
            "relax": relax.run, "structure": structure.run, "hessian": hessian.run, "vibspec": vibspec.run,
            "electronic": electronic.run, "stability": stability.run, "detachment": detachment.run,
            "solvation": solvation.run, "sites": sites.run, "report": report.run}
    settings = settings or Settings()
    ctx = load_context(record_dir, settings, cif)
    wanted = [s for s in STEPS if s in set(steps)]
    # Earlier steps a requested one depends on are run (or loaded) too.
    def closure(step):
        out = {step}
        for d in DEPS[step]:
            out |= closure(d)
        return out
    if len(ctx.symbols) > settings.hessian_max_atoms:
        dropped = [w for w in wanted if "hessian" in closure(w)]
        wanted = [w for w in wanted if w not in dropped]
        if dropped:
            log(f"[qdprops] {len(ctx.symbols)} atoms > hessian_max_atoms = {settings.hessian_max_atoms}: "
                f"skipping {', '.join(dropped)}")
    todo = [s for s in STEPS if any(s in closure(w) for w in wanted)]
    log(f"[qdprops] {ctx.record.get('id', ctx.record_dir.name)}: {len(ctx.symbols)} atoms; steps {', '.join(todo)}")
    for step in todo:
        inputs = _step_inputs(ctx, step)
        h = _hash(inputs)
        out = ctx.props / f"{step}.json"
        if not force and out.is_file():
            prev = json.loads(out.read_text())
            if prev.get("_hash") == h:
                ctx.results[step] = prev
                log(f"[qdprops]   {step}: cached")
                continue
        if upstream_cached and step not in wanted:
            raise RuntimeError(f"{step} is not cached (its inputs changed or the GPU stage did not finish it)")
        t0 = time.time()
        res = impl[step](ctx)
        res["_hash"] = h
        res["_inputs"] = inputs
        res["_seconds"] = round(time.time() - t0, 2)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=_json_default))
        os.replace(tmp, out)
        ctx.results[step] = res
        log(f"[qdprops]   {step}: done in {res['_seconds']:.1f} s")
    summary = _write_properties(ctx)
    return summary


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _code_revision() -> dict:
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src/orchestr_ai/qd"],
                                    capture_output=True, text=True).stdout.strip())
        return {"orchestr_ai_commit": commit, "orchestr_ai_dirty": dirty}
    except Exception:
        return {"orchestr_ai_commit": None, "orchestr_ai_dirty": None}


def _write_properties(ctx: Context) -> dict:
    """props/properties.json: per-step summaries plus provenance."""
    path = ctx.props / "properties.json"
    prev = json.loads(path.read_text()) if path.is_file() else {}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "id": ctx.record.get("id"),
        "fingerprint": ctx.record.get("fingerprint"),
        "n_atoms": len(ctx.symbols),
        "provenance": {**{k: v for k, v in prev.get("provenance", {}).items() if k not in ("gxtb",)},
                       **_code_revision()},
        "summary": {k: v for k, v in prev.get("summary", {}).items() if k in STEPS},
    }
    for step, res in ctx.results.items():
        summary["summary"][step] = res.get("summary", {})
        if res.get("provenance"):
            summary["provenance"][step] = res["provenance"]
    path.write_text(json.dumps(summary, indent=1, default=_json_default))
    return summary
