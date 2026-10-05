# src/orchestr_ai/qd/engines.py
"""
Energy engines: MACE-MH-1 (ASE calculator, in-process) and xtb (GFN2-xTB by
default, or g-xTB from the bleeding-edge build), run as a subprocess.

MACE-MH-1 is a multi-head model; every energy, force and Hessian of a run
uses one head (default `omat_pbe`), recorded in the provenance.  device="auto"
picks a CUDA GPU (float64), else the Apple GPU (MPS) in float32, 3-8x faster
than float64 on the CPU with energies within 0.001 meV/atom and harmonic
frequencies within 0.001 cm-1 for Cd16Se13Cl6, else the CPU in float64.

GFN2-xTB is xtb 6.7.1 (QDPROPS_XTB).  g-xTB is the binary from
grimme-lab/g-xtb (QDPROPS_GXTB; static on Linux); QDPROPS_XTB_LIBS adds
dynamic-library directories (':'-separated), needed only by the macOS build.
Without the variables the defaults are the layout on Ivan's Mac, else xtb on PATH.
"""
from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903

DEFAULT_MODEL = os.environ.get("QDPROPS_MACE_MODEL", str(Path.home() / ".cache/mace/macemh1model"))
DEFAULT_HEAD = "omat_pbe"
# Defaults: the environment variables, else Ivan's Mac layout when present, else xtb on PATH.
_GXTB_ROOT = Path.home() / "Documents/University/Programs/GXTB/xtb-bleed-macos-arm64"
_MAC_XTB = Path.home() / "miniforge3/envs/xtb/bin/xtb"
DEFAULT_GXTB = os.environ.get("QDPROPS_GXTB") or (
    str(_GXTB_ROOT / "bin/xtb") if (_GXTB_ROOT / "bin/xtb").is_file() else "xtb")
DEFAULT_XTB = os.environ.get("QDPROPS_XTB") or (str(_MAC_XTB) if _MAC_XTB.is_file() else (shutil.which("xtb") or "xtb"))
XTB_METHODS = {"gfn2": ["--gfn", "2"], "gxtb": ["--gxtb"]}
# Last-resort SCF annealing (electronic temperatures, K): converge hot, then cool. Cd14Se13Cl2
# desorption products failed at every fixed temperature up to 10000 K; 20000 K converges and
# the ladder below reaches the base temperature.
ANNEAL_ETEMPS = (20000, 10000, 6000, 3000, 1500)
ANNEAL_ITERATIONS = 1000
# Extra library directories for the macOS g-xTB build; the static Linux binary needs none.
DEFAULT_XTB_LIBS = os.environ.get(
    "QDPROPS_XTB_LIBS",
    ":".join(str(d) for d in (_MAC_XTB.parent.parent / "lib", _GXTB_ROOT / "lib") if d.is_dir()),
)


# --------------------------------------------------------------------------
# MACE-MH-1
# --------------------------------------------------------------------------

def _allow_mps_double() -> None:
    """Apple MPS has no float64, but mace-torch 0.3.16 calls .double() in the
    forward pass; make that cast a no-op for MPS tensors (CPU ones unchanged)."""
    import torch
    if getattr(torch.Tensor.double, "_qdprops_mps_safe", False):
        return
    orig = torch.Tensor.double

    def double(self, *args, **kwargs):
        if self.device.type == "mps":
            return self
        return orig(self, *args, **kwargs)

    double._qdprops_mps_safe = True
    torch.Tensor.double = double


@functools.lru_cache(maxsize=4)
def mace_calculator(head: str = DEFAULT_HEAD, model: str = DEFAULT_MODEL,
                    device: str = "cpu", dtype: str = "float64"):
    """
    ASE calculator for one MACE-MH-1 head (cached per settings).

    device="mps" runs on the Apple GPU in float32: the float64 checkpoint is
    loaded on the CPU, converted with .float() and handed to the calculator.
    """
    warnings.filterwarnings("ignore", module="mace")
    warnings.filterwarnings("ignore", module="e3nn")
    from mace.calculators import MACECalculator
    device = resolve_device(device)
    if device == "mps":
        import torch
        _allow_mps_double()
        net = torch.load(model, map_location="cpu", weights_only=False).float()
        return MACECalculator(models=[net], device="mps", default_dtype="float32", head=head)
    return MACECalculator(model_paths=model, device=device, default_dtype=dtype, head=head)


def resolve_device(device: str) -> str:
    """'auto': a CUDA GPU when available, else the Apple GPU (MPS), else the CPU."""
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sync_device(device: str) -> None:
    """Wait for queued GPU work (for timing)."""
    import torch
    if device == "mps":
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def mace_hessian(atoms, calc) -> np.ndarray:
    """Analytic (autograd) Cartesian Hessian d2E/dx2 in eV/Å², shape (3N, 3N)."""
    h = np.asarray(calc.get_hessian(atoms=atoms), float)
    n = len(atoms)
    return h.reshape(3 * n, 3 * n)


@functools.lru_cache(maxsize=4)
def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mace_provenance(head: str, model: str, device: str, dtype: str) -> dict:
    import mace
    import torch
    device = resolve_device(device)
    return {"engine": "MACE-MH-1", "head": head, "model": Path(model).name,
            "model_sha256": file_sha256(model), "mace": getattr(mace, "__version__", "?"),
            "torch": torch.__version__, "device": device, "dtype": "float32" if device == "mps" else dtype}


# --------------------------------------------------------------------------
# xtb (GFN2-xTB, g-xTB)
# --------------------------------------------------------------------------

@dataclass
class XtbResult:
    energy_eV: float
    homo_eV: Optional[float]
    lumo_eV: Optional[float]
    gap_eV: Optional[float]
    charges: List[float]
    dipole_au: Optional[List[float]]
    dipole_debye: Optional[float]
    gradient_eV_A: Optional[np.ndarray] = None
    raw: Dict[str, str] = field(default_factory=dict)


class XtbRunner:
    """Runs xtb (`method` gfn2 or gxtb) on a geometry in a scratch directory and parses the output."""

    def __init__(self, method: str = "gfn2", binary: Optional[str] = None, libs: str = DEFAULT_XTB_LIBS,
                 threads: Optional[int] = None):
        if method not in XTB_METHODS:
            raise ValueError(f"unknown xtb method {method!r}")
        self.method = method
        self.binary = binary or (DEFAULT_GXTB if method == "gxtb" else DEFAULT_XTB)
        self.env = dict(os.environ)
        if method == "gxtb":
            # Only the bleeding-edge g-xTB build needs the extra libraries; loading them into
            # the stock xtb makes it ~25x slower (wrong BLAS/LAPACK).
            self.env["DYLD_LIBRARY_PATH"] = ":".join(p for p in (libs, self.env.get("DYLD_LIBRARY_PATH", "")) if p)
            self.env["LD_LIBRARY_PATH"] = ":".join(p for p in (libs, self.env.get("LD_LIBRARY_PATH", "")) if p)
        # xtb runs single-threaded here unless told otherwise (12 s vs 0.5 s for a 149-atom dot).
        self.env["OMP_NUM_THREADS"] = str(threads or min(8, os.cpu_count() or 1))
        self.env.setdefault("OMP_STACKSIZE", "1G")

    def version(self) -> str:
        out = subprocess.run([self.binary, "--version"], env=self.env, capture_output=True, text=True).stdout
        m = re.search(r"xtb version\s+(\S+\s+\(\S+\))", out)
        return m.group(1) if m else "unknown"

    def provenance(self) -> dict:
        return {"engine": {"gfn2": "GFN2-xTB", "gxtb": "g-xTB"}[self.method], "binary": self.binary,
                "xtb": self.version()}

    def run(self, symbols: Sequence[str], pts: np.ndarray, *, charge: int = 0, uhf: int = 0,
            gradient: bool = False, solvation: Optional[Sequence[str]] = None, attempts: int = 3) -> XtbResult:
        """
        Single point; `solvation` adds flags, e.g. ["--cosmo", "9.0"] or ["--alpb", "toluene"].
        The SCC of small-gap structures occasionally fails at random (threaded runs are not
        bit-reproducible), so a failed run is repeated up to `attempts` times.
        """
        for i in range(attempts):
            try:
                return self._run_once(symbols, pts, charge=charge, uhf=uhf, gradient=gradient, solvation=solvation)
            except RuntimeError:
                if i == attempts - 1:
                    raise

    def _run_once(self, symbols, pts, *, charge, uhf, gradient, solvation) -> XtbResult:
        with tempfile.TemporaryDirectory(prefix="xtb_") as tmp:
            tmp = Path(tmp)
            xyz = tmp / "mol.xyz"
            lines = [str(len(symbols)), "qdprops"]
            lines += [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, np.asarray(pts, float))]
            xyz.write_text("\n".join(lines) + "\n")
            cmd = [self.binary, xyz.name, *XTB_METHODS[self.method], "--chrg", str(int(charge)),
                   "--uhf", str(int(uhf))]
            if gradient:
                cmd.append("--grad")
            if solvation:
                cmd.extend(str(x) for x in solvation)
            proc = subprocess.run(cmd, cwd=tmp, env=self.env, capture_output=True, text=True)
            out = proc.stdout + proc.stderr
            if not re.search(r"^\s*normal termination of xtb", out, re.M) or "abnormal termination" in out:
                tail = "\n".join(out.strip().splitlines()[-15:])
                raise RuntimeError(f"xtb {self.method} failed (charge {charge}, uhf {uhf}):\n{tail}")
            res = _parse_xtb(out)
            charges_file = tmp / "charges"
            if charges_file.exists():
                res.charges = [float(x) for x in charges_file.read_text().split()]
            engrad = tmp / "mol.engrad"
            if gradient and engrad.exists():
                res.gradient_eV_A = _read_engrad_gradient(engrad.read_text(), len(symbols))
            return res


    def run_series(self, symbols: Sequence[str], pts: np.ndarray, flag_sets: Sequence[Sequence[str]],
                   base: Sequence[str] = (), fallbacks: Sequence[Sequence[str]] = ()) -> tuple:
        """
        Gas-phase single point, then one run per flag set, each restarted from the
        converged gas-phase density (xtbrestart), in one directory.  Returns
        (gas XtbResult, [energy_eV or None per flag set], base flags used); a run that
        fails is None.  If the gas phase does not converge with `base`, each of
        `fallbacks` is tried in turn and the first that works is used for all runs.
        If none does, the SCF is annealed: converged at a high electronic
        temperature, then cooled through ANNEAL_ETEMPS, each run restarting from the
        previous density, and finished with `base` (the flags then end with
        "annealed-from-<T>K"). A 0.001 Å change of a 29-atom CdSe desorption product
        was enough to make every fixed-temperature option fail.
        """
        import shutil
        with tempfile.TemporaryDirectory(prefix="xtb_") as tmp:
            tmp = Path(tmp)
            lines = [str(len(symbols)), "qdprops"]
            lines += [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, np.asarray(pts, float))]
            (tmp / "mol.xyz").write_text("\n".join(lines) + "\n")

            used = list(base)

            def call(flags):
                proc = subprocess.run([self.binary, "mol.xyz", *XTB_METHODS[self.method], *used, *flags],
                                      cwd=tmp, env=self.env, capture_output=True, text=True)
                out = proc.stdout + proc.stderr
                ok = re.search(r"^\s*normal termination of xtb", out, re.M) and "abnormal termination" not in out
                return out if ok else None

            out = None
            for option in [list(base), *[list(f) for f in fallbacks]]:
                used[:] = option
                for _ in range(2):
                    out = call([])
                    if out:
                        break
                if out:
                    break
            annealed = None
            if out is None:
                (tmp / "xtbrestart").unlink(missing_ok=True)
                for etemp in ANNEAL_ETEMPS:
                    used[:] = ["--etemp", str(etemp), "--iterations", str(ANNEAL_ITERATIONS)]
                    if call([]) is None:
                        break
                else:
                    used[:] = list(base)
                    out = call([])
                    annealed = ANNEAL_ETEMPS[0]
            if out is None:
                raise RuntimeError("xtb gas-phase reference failed")
            gas = _parse_xtb(out)
            if (tmp / "charges").exists():
                gas.charges = [float(x) for x in (tmp / "charges").read_text().split()]
            shutil.copy(tmp / "xtbrestart", tmp / "gas.restart")
            energies = []
            for flags in flag_sets:
                shutil.copy(tmp / "gas.restart", tmp / "xtbrestart")
                o = call(list(flags))
                energies.append(_parse_xtb(o).energy_eV if o else None)
            return gas, energies, list(used) + ([f"annealed-from-{annealed}K"] if annealed else [])

    def vipea(self, symbols: Sequence[str], pts: np.ndarray) -> Dict[str, float]:
        """Vertical IP and EA (eV) from `xtb --vipea`: IPEA-xTB delta-SCC with its empirical shift."""
        with tempfile.TemporaryDirectory(prefix="xtb_") as tmp:
            tmp = Path(tmp)
            lines = [str(len(symbols)), "qdprops"]
            lines += [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in zip(symbols, np.asarray(pts, float))]
            (tmp / "mol.xyz").write_text("\n".join(lines) + "\n")
            proc = subprocess.run([self.binary, "mol.xyz", "--vipea"], cwd=tmp, env=self.env,
                                  capture_output=True, text=True)
            out = proc.stdout + proc.stderr
        ip = re.search(r"delta SCC IP \(eV\):\s+(-?\d+\.\d+)", out)
        if "abnormal termination" in out:
            raise RuntimeError("xtb --vipea failed")
        ea = re.search(r"delta SCC EA \(eV\):\s+(-?\d+\.\d+)", out)
        if not (ip and ea) or "normal termination" not in out:
            raise RuntimeError("xtb --vipea failed")
        return {"ip_eV": float(ip.group(1)), "ea_eV": float(ea.group(1))}


def _read_engrad_gradient(text: str, n_atoms: int) -> np.ndarray:
    """Gradient block of an ORCA-style .engrad file, converted to eV/Å."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if "gradient" in l.lower() and l.startswith("#"))
    vals: List[float] = []
    for l in lines[start + 1:]:
        if l.startswith("#"):
            if vals:
                break
            continue
        vals.append(float(l.split()[0]))
        if len(vals) == 3 * n_atoms:
            break
    return np.asarray(vals, float).reshape(n_atoms, 3) * HARTREE_EV / BOHR_A


def _parse_xtb(out: str) -> XtbResult:
    m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", out)
    if not m:
        raise RuntimeError("xtb output has no TOTAL ENERGY")
    energy = float(m.group(1)) * HARTREE_EV
    homo = lumo = None
    for line in out.splitlines():
        if "(HOMO)" in line:
            homo = float(line.split()[-2])
        elif "(LUMO)" in line:
            lumo = float(line.split()[-2])
    gap = None
    mg = re.search(r"HOMO-LUMO gap\s+(-?\d+\.\d+)\s+eV", out)
    if mg:
        gap = float(mg.group(1))
    elif homo is not None and lumo is not None:
        gap = lumo - homo
    dip_au = dip_d = None
    if "molecular dipole:" in out:          # GFN2-xTB: "full:  x y z  tot(Debye)" in a.u.
        block = out.split("molecular dipole:")[-1]
        mf = re.search(r"^\s*full:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", block, re.M)
        if mf:
            dip_au = [float(mf.group(k)) for k in (1, 2, 3)]
            dip_d = float(mf.group(4))
    elif "Atomic dipole moments" in out:    # g-xTB
        block = out.split("Atomic dipole moments")[-1]
        mt = re.search(r"^\s*total\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", block, re.M)
        if mt:
            dip_au = [float(mt.group(k)) for k in (1, 2, 3)]
        md = re.search(r"\|total\|\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+Debye", block)
        if md:
            dip_d = float(md.group(2))
    return XtbResult(energy_eV=energy, homo_eV=homo, lumo_eV=lumo, gap_eV=gap, charges=[],
                      dipole_au=dip_au, dipole_debye=dip_d)
