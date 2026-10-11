# src/orchestr_ai/qd/field.py
"""
Born charges, Raman tensors and atomic charges from CP2K single points in a uniform
electric field (the DFT chain runs them; this module writes their inputs and reads them).

The field E is an external potential phi(r) = -E.r on electrons and cores, so the force
on atom i responds as

    Z*_{i,ab}       = dF_ib / dE_a                    (Born charge, = dmu_a / dR_ib)
    dalpha_ac/dR_ib = d2F_ib / dE_a dE_c              (Raman tensor)
    alpha_ab        = dmu_a / dE_b                    (polarizability)

13 single points per dot, all restarted from the geo_opt wavefunction:
    f0                   zero field: reference forces, dipole, Hirshfeld charges
    f{x,y,z}{p,m}        +-E along one axis      -> Z*, the diagonal of dalpha/dR, alpha
    f{xy,xz,yz}{pp,mm}   (+E,+E) and (-E,-E)     -> the off-diagonal of dalpha/dR

    d2F/dEa dEc = [F(E,E) + F(-E,-E) - F(E,0) - F(-E,0) - F(0,E) - F(0,-E) + 2 F(0,0)] / 2E^2

which is second-order accurate like the four-point stencil (Cd16, Cd68: Raman spectra
identical to the 19-point set, cosine 1.00000, total activity within 0.1%). Two more single
points give the Hirshfeld charges of the MA and MX_q monomers (solvation references).

Inputs are derived from the chain's own `sample` single point (qdw cp2k prepare --step
sample: GAPW, MULTIPOLE, OT/DIIS, one SCF loop, forces to forces.xyz), so every setting
matches the geo_opt. Cd16/Cd68 tests: Born-charge sum rule 1e-4 e, polarizability within
0.01% of linear response, IR spectra cosine 0.993-0.996 against g-xTB on the same modes.

    python -m orchestr_ai.qd.field prepare <base cp2k.inp> <field dir> [--monomers monomers.xyz]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

FIELD_AU = 0.002            # a.u. (0.1 V/A); forces precise to ~1e-7 Ha/bohr at EPS_SCF 1e-7
EPS_SCF = 1e-7
AXES = "xyz"
PAIRS = ("xy", "xz", "yz")
BOHR_A = 0.529177210903
DEBYE_AU = 0.393430237      # 1 D in e bohr
MONOMER_CELL_A = 15.0


def run_names() -> List[str]:
    return (["f0"] + [f"f{a}{s}" for a in AXES for s in "pm"]
            + [f"f{p}{s}" for p in PAIRS for s in ("pp", "mm")])


def field_vector(name: str, e: float = FIELD_AU) -> np.ndarray:
    f = np.zeros(3)
    if name == "f0":
        return f
    body = name[1:]
    axes, signs = (body[:1], body[1:]) if len(body) == 2 else (body[:2], body[2:])
    for ax, sg in zip(axes, signs):
        f[AXES.index(ax)] = e if sg == "p" else -e
    return f


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

EXTERNAL = """    &EXTERNAL_POTENTIAL
      FUNCTION -(EX*X+EY*Y+EZ*Z)
      PARAMETERS EX EY EZ
      VALUES {:.6e} {:.6e} {:.6e}
    &END EXTERNAL_POTENTIAL
"""
PRINTS = """    &PRINT
      &MOMENTS
        PERIODIC F
      &END MOMENTS
{hirshfeld}    &END PRINT
"""
HIRSHFELD = """      &HIRSHFELD
        FILENAME =hirshfeld.dat
      &END HIRSHFELD
"""


def _set(text: str, pattern: str, repl: str, count: int = 1) -> str:
    new, n = re.subn(pattern, repl, text, count=count, flags=re.M)
    if n == 0:
        raise ValueError(f"base input has no match for {pattern!r}")
    return new


def field_input(base: str, name: str, e: float = FIELD_AU) -> str:
    """One field single point from the chain's `sample` input (restart from the geo_opt .wfn)."""
    t = _set(base, r"^(\s*PROJECT\s+)\S+", rf"\g<1>{name}")
    t = _set(t, r"^(\s*EPS_SCF\s+)\S+", rf"\g<1>{EPS_SCF:g}")
    f = field_vector(name, e)
    block = PRINTS.format(hirshfeld=HIRSHFELD if name == "f0" else "")
    if np.any(f):
        block += EXTERNAL.format(*f)
    return _set(t, r"^(\s*)&SCF\s*$", block + r"\g<1>&SCF")


def monomer_input(base: str, name: str, cell: float = MONOMER_CELL_A) -> str:
    """Zero-field single point of a monomer (Hirshfeld charges), from scratch in a small cell."""
    t = _set(base, r"^(\s*PROJECT\s+)\S+", rf"\g<1>{name}")
    t = _set(t, r"^(\s*RUN_TYPE\s+)\S+", r"\g<1>ENERGY")
    t = _set(t, r"^(\s*SCF_GUESS\s+)\S+", r"\g<1>atomic")
    t = _set(t, r"^(\s*ABC\s+).*$", rf"\g<1>{cell:.3f} {cell:.3f} {cell:.3f}")
    t = re.sub(r"^\s*WFN_RESTART_FILE_NAME.*\n", "", t, flags=re.M)     # (the forces print is idle at ENERGY)
    return _set(t, r"^(\s*)&SCF\s*$", PRINTS.format(hirshfeld=HIRSHFELD) + r"\g<1>&SCF")


def _read_frames(path: Path) -> List[tuple]:
    lines, out, i = Path(path).read_text().splitlines(), [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        rows = [l.split() for l in lines[i + 2:i + 2 + n]]
        out.append((lines[i + 1].strip().split()[0] if lines[i + 1].strip() else f"m{len(out)}",
                    [r[0] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows])))
        i += n + 2
    return out


def prepare(base_inp: str, out_dir: str, monomers: Optional[str] = None, e: float = FIELD_AU) -> List[str]:
    """Write <out_dir>/<run>/{cp2k.inp, geom.xyz} for the 13 field runs (and the monomers).
    The base is the chain's `sample` input; its WFN_RESTART_FILE_NAME must point two levels
    up (field/base/ -> ../../geo_opt/), which holds for field/<run>/ too."""
    base_path = Path(base_inp)
    base = base_path.read_text()
    out = Path(out_dir)
    geom = (base_path.parent / "geom.xyz").read_text()
    names = []
    for name in run_names():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "cp2k.inp").write_text(field_input(base, name, e))
        (d / "geom.xyz").write_text(geom)
        names.append(name)
    if monomers:
        for label, sym, pts in _read_frames(Path(monomers)):
            name = f"mono_{label}"
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "cp2k.inp").write_text(monomer_input(base, name))
            (d / "geom.xyz").write_text(f"{len(sym)}\n{label}\n" + "".join(
                f"{s} {x:.8f} {y:.8f} {z:.8f}\n" for s, (x, y, z) in zip(sym, pts)))
            names.append(name)
    (out / "runs.txt").write_text("\n".join(names) + "\n")
    (out / "field.json").write_text(json.dumps({"field_au": e, "eps_scf": EPS_SCF, "runs": names}, indent=1))
    return names


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

def read_forces(run_dir: Path) -> Optional[np.ndarray]:
    """Forces (Ha/bohr) from forces.xyz (CP2K &PRINT &FORCES), else from cp2k.out."""
    files = sorted(Path(run_dir).glob("*forces*.xyz")) + [Path(run_dir) / "cp2k.out"]
    for f in files:
        if not f.is_file():
            continue
        text = f.read_text(errors="ignore")
        rows = re.findall(r"^\s*FORCES\|\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)", text, re.M)
        if rows:
            n = int(re.findall(r"^\s*FORCES\|\s+(\d+)\s", text, re.M)[-1])
            return np.array(rows[-n:], float)
        if "ATOMIC FORCES in [a.u.]" in text:
            block = text.split("ATOMIC FORCES in [a.u.]")[-1].split("SUM OF ATOMIC FORCES")[0]
            rows = [l.split() for l in block.splitlines() if re.match(r"^\s*\d+\s+\d+\s+\w+\s+-?\d", l)]
            return np.array([[float(x) for x in r[3:6]] for r in rows])
    return None


def read_dipole(run_dir: Path) -> Optional[np.ndarray]:
    """Dipole moment (e bohr) printed by &MOMENTS."""
    out = Path(run_dir) / "cp2k.out"
    if not out.is_file():
        return None
    m = re.findall(r"Dipole moment \[Debye\]\s*\n\s*X=\s*(\S+)\s+Y=\s*(\S+)\s+Z=\s*(\S+)", out.read_text(errors="ignore"))
    return np.array([float(x) for x in m[-1]]) * DEBYE_AU if m else None


def read_hirshfeld(run_dir: Path) -> Optional[np.ndarray]:
    """Hirshfeld net charges (e), in atom order."""
    files = sorted(Path(run_dir).glob("*hirshfeld*.dat")) + [Path(run_dir) / "cp2k.out"]
    for f in files:
        if not f.is_file():
            continue
        text = f.read_text(errors="ignore")
        if "Hirshfeld Charges" not in text:
            continue
        block = text.split("Hirshfeld Charges")[-1].split("Total Charge")[0]
        # atom, element, kind, reference charge, population(s), net charge (last column)
        rows = re.findall(r"^\s*\d+\s+[A-Z][a-z]?\s+\d+\s+.*?(-?\d+\.\d+)\s*$", block, re.M)
        if rows:
            return np.array(rows, float)
    return None


def done(run_dir: Path) -> bool:
    out = Path(run_dir) / "cp2k.out"
    return out.is_file() and "PROGRAM ENDED" in out.read_text(errors="ignore")[-4000:]


def input_digest(field_dir: Path) -> Optional[str]:
    """Hash of every run's forces/dipole/charge output: the field step's external input."""
    d = Path(field_dir)
    if not (d / "runs.txt").is_file():
        return None
    h = hashlib.sha256()
    for name in (d / "runs.txt").read_text().split():
        for f in sorted((d / name).glob("*forces*.xyz")) + sorted((d / name).glob("*hirshfeld*.dat")) + [d / name / "cp2k.out"]:
            if f.is_file():
                h.update(f.name.encode())
                h.update(f.read_bytes()[-200000:])
    return h.hexdigest()[:16]


def analyse(field_dir: str, e: Optional[float] = None) -> Dict:
    """Born charges, Raman tensors, polarizability and charges from the field runs (atomic units)."""
    d = Path(field_dir)
    meta = json.loads((d / "field.json").read_text()) if (d / "field.json").is_file() else {}
    e = e or meta.get("field_au", FIELD_AU)
    missing = [n for n in run_names() if not done(d / n)]
    if missing:
        raise RuntimeError(f"field runs not finished: {', '.join(missing)}")
    F = {n: read_forces(d / n) for n in run_names()}
    bad = [n for n, f in F.items() if f is None]
    if bad:
        raise RuntimeError(f"no forces in {', '.join(bad)}")
    n = len(F["f0"])
    Z = np.zeros((3, n, 3))                    # [field axis, atom, force component]
    A = np.zeros((3, 3, n, 3))                 # [field axis, field axis, atom, force component]
    for a, ax in enumerate(AXES):
        p, m = F[f"f{ax}p"], F[f"f{ax}m"]
        Z[a] = (p - m) / (2 * e)
        A[a, a] = (p + m - 2 * F["f0"]) / e ** 2
    for pair in PAIRS:
        a, c = AXES.index(pair[0]), AXES.index(pair[1])
        A[a, c] = A[c, a] = (F[f"f{pair}pp"] + F[f"f{pair}mm"] - F[f"f{pair[0]}p"] - F[f"f{pair[0]}m"]
                             - F[f"f{pair[1]}p"] - F[f"f{pair[1]}m"] + 2 * F["f0"]) / (2 * e ** 2)
    mu = {name: read_dipole(d / name) for name in ("f0",) + tuple(f"f{ax}{s}" for ax in AXES for s in "pm")}
    alpha = None
    if all(v is not None for v in mu.values()):
        alpha = np.array([(mu[f"f{ax}p"] - mu[f"f{ax}m"]) / (2 * e) for ax in AXES]).T     # bohr^3
    charges = read_hirshfeld(d / "f0")
    monomers = {}
    for name in sorted(p.name for p in d.glob("mono_*") if p.is_dir()):
        q = read_hirshfeld(d / name)
        if q is not None:
            sym = [l.split()[0] for l in (d / name / "geom.xyz").read_text().splitlines()[2:] if l.strip()]
            monomers[name[5:]] = {"symbols": sym, "charges": q.tolist()}
    return {
        "field_au": e, "n_atoms": n,
        "born": np.einsum("aib->iab", Z),      # per atom: [field, displacement] (e)
        "raman_tensor": A,                     # d alpha_ac / dR_ib (bohr^2)
        "alpha": alpha,                        # bohr^3
        "dipole": mu["f0"],                    # e bohr
        "charges": charges,                    # Hirshfeld (e)
        "monomers": monomers,
        "sum_rule_born_e": float(np.abs(Z.sum(1)).max()),
        "sum_rule_raman_rel": float(np.abs(A.sum(2)).max() / max(np.abs(A).max(), 1e-12)),
    }


def mode_derivatives(res: Dict, modes_mass_weighted: np.ndarray, masses: np.ndarray):
    """dmu/dQ (D/(amu^1/2 A)) and dalpha/dQ (A^2/amu^1/2) along mass-weighted modes (3N, m)."""
    E_D = 4.80320471                           # e A in Debye
    n = res["n_atoms"]
    disp = np.asarray(modes_mass_weighted) / np.sqrt(np.repeat(np.asarray(masses, float), 3))[:, None]
    Z = np.einsum("iab->aib", res["born"]).reshape(3, 3 * n)
    dmu = (Z @ disp).T * E_D
    dal = np.einsum("acj,jk->kac", res["raman_tensor"].reshape(3, 3, 3 * n), disp) * BOHR_A ** 2
    return dmu, dal


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m orchestr_ai.qd.field")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare", help="write the field (and monomer) single-point inputs")
    q.add_argument("base", help="the chain's sample cp2k.inp (restarts from the geo_opt .wfn)")
    q.add_argument("out", help="field directory")
    q.add_argument("--monomers", help="xyz with the MA and MX_q monomers (props/monomers.xyz)")
    q.add_argument("--field", type=float, default=FIELD_AU)
    q = sub.add_parser("status", help="finished and missing runs")
    q.add_argument("out")
    a = p.parse_args(argv)
    if a.cmd == "prepare":
        names = prepare(a.base, a.out, a.monomers, a.field)
        print(f"{len(names)} single points in {a.out}")
    else:
        names = (Path(a.out) / "runs.txt").read_text().split()
        left = [x for x in names if not done(Path(a.out) / x)]
        print(" ".join(left) if left else "all done")


if __name__ == "__main__":
    main()
