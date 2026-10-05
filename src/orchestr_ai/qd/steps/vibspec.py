# src/orchestr_ai/qd/steps/vibspec.py
"""
IR and non-resonant Raman spectra: MACE-MH-1 normal modes, g-xTB intensities.

Peak positions are the harmonic MACE-MH-1 frequencies of the hessian step.
For each mode k the structure is displaced to x0 ± h e_k / sqrt(m) (h in
amu^1/2 Å) and g-xTB is run in a static field ±F along x, y and z (six runs
per geometry).  At each geometry

    mu      = mean of the ± field runs                (field-free dipole, O(F²))
    alpha_ij = [mu_i(+F e_j) − mu_i(−F e_j)] / 2F     (static polarisability)

and the derivatives dmu/dQ_k and dalpha/dQ_k follow by central difference.

    IR intensity       A_k = (N_A π / 3c) |dmu/dQ_k|²                  (km/mol)
    Raman invariants   a' = tr(alpha')/3,
                       γ'² = ½[(xx−yy)² + (yy−zz)² + (zz−xx)² + 6(xy² + yz² + zx²)]
    Raman activity     S_k = 45 a'² + 7 γ'²                            (Å⁴/amu)
    depolarisation     rho_k = 3 γ'² / (45 a'² + 4 γ'²)                 (0 … 3/4)

g-xTB is used only for these response derivatives (it is not at its own
minimum at the MACE geometry; the residual g-xTB gradient is recorded).  Its
`--efield` is in V/Å: the induced dipole and −½ alpha F² of the field-dependent
energy agree only with that unit.  GFN2-xTB has no field support in xtb.

Modes are symmetry-adapted within (near-)degenerate sets by projection
operators of the point group (pymatgen), with irreducible-representation
labels and formal IR/Raman selection rules for the groups tabulated below;
for any other group only the totally symmetric modes are identified.  Each
mode is characterised by its core / surface / ligand and element shares of
the mass-weighted amplitude, its radial share and its overlap with a uniform
breathing of the dot, and assigned a descriptive class (see `mode_class`).
The bulk Gamma-point optical frequency of the CIF (MACE, primitive-cell force
constants; no LO–TO splitting, which needs Born charges) is the reference.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..engines import XtbRunner

AU_FIELD_V_A = 51.42206747632590     # 1 a.u. of field in V/Å
BOHR_A = 0.529177210903
AU_DEBYE = 2.541746473
KMMOL = 42.2561                      # km/mol per (D/Å)² amu⁻¹
CACHE = "vibspec_cache.json"
ACC = "0.01"                         # xtb --acc (tight SCF; derivatives of small dipole changes)

# Character tables (real irreps; complex-conjugate pairs merged), classes keyed
# by `_op_class`.  ir / raman: irreps spanned by (x, y, z) / by the quadratic forms.
TABLES = {
    "C1": {"classes": ["E"], "order": {"E": 1}, "irreps": {"A": [1]}, "ir": ["A"], "raman": ["A"]},
    "Cs": {"classes": ["E", "sh"], "order": {"E": 1, "sh": 1},
           "irreps": {"A'": [1, 1], "A''": [1, -1]}, "ir": ["A'", "A''"], "raman": ["A'", "A''"]},
    "C2": {"classes": ["E", "C2"], "order": {"E": 1, "C2": 1},
           "irreps": {"A": [1, 1], "B": [1, -1]}, "ir": ["A", "B"], "raman": ["A", "B"]},
    "C2v": {"classes": ["E", "C2", "sv", "sv'"], "order": {"E": 1, "C2": 1, "sv": 1, "sv'": 1},
            "irreps": {"A1": [1, 1, 1, 1], "A2": [1, 1, -1, -1], "B1": [1, -1, 1, -1], "B2": [1, -1, -1, 1]},
            "ir": ["A1", "B1", "B2"], "raman": ["A1", "A2", "B1", "B2"]},
    "C3": {"classes": ["E", "C3"], "order": {"E": 1, "C3": 2},
           "irreps": {"A": [1, 1], "E": [2, -1]}, "ir": ["A", "E"], "raman": ["A", "E"]},
    "C3v": {"classes": ["E", "C3", "sv"], "order": {"E": 1, "C3": 2, "sv": 3},
            "irreps": {"A1": [1, 1, 1], "A2": [1, 1, -1], "E": [2, -1, 0]}, "ir": ["A1", "E"],
            "raman": ["A1", "E"]},
    "D2d": {"classes": ["E", "S4", "C2", "C2'", "sd"], "order": {"E": 1, "S4": 2, "C2": 1, "C2'": 2, "sd": 2},
            "irreps": {"A1": [1, 1, 1, 1, 1], "A2": [1, 1, 1, -1, -1], "B1": [1, -1, 1, 1, -1],
                       "B2": [1, -1, 1, -1, 1], "E": [2, 0, -2, 0, 0]},
            "ir": ["B2", "E"], "raman": ["A1", "B1", "B2", "E"]},
    "Td": {"classes": ["E", "C3", "C2", "S4", "sd"], "order": {"E": 1, "C3": 8, "C2": 3, "S4": 6, "sd": 6},
           "irreps": {"A1": [1, 1, 1, 1, 1], "A2": [1, 1, 1, -1, -1], "E": [2, -1, 2, 0, 0],
                      "T1": [3, 0, -1, 1, -1], "T2": [3, 0, -1, -1, 1]},
           "ir": ["T2"], "raman": ["A1", "E", "T2"]},
}


def _axis(R):
    """Rotation axis of a proper rotation R (or of -R for an improper one)."""
    P = R if np.linalg.det(R) > 0 else -R
    w, v = np.linalg.eig(P)
    return np.real(v[:, np.argmin(np.abs(w - 1))])


def _op_class(group, R, ref):
    """Class of operation R; `ref` holds the reference axes that split same-(det, trace) classes."""
    det, tr = int(round(np.linalg.det(R))), int(round(np.trace(R)))
    if det > 0 and tr == 3:
        return "E"
    if group == "Td":
        return {(1, 0): "C3", (1, -1): "C2", (-1, -1): "S4", (-1, 1): "sd"}.get((det, tr))
    if group in ("C3", "C3v"):
        return {(1, 0): "C3", (-1, 1): "sv"}.get((det, tr))
    if group == "Cs":
        return "sh" if (det, tr) == (-1, 1) else None
    if group == "C2":
        return "C2" if (det, tr) == (1, -1) else None
    if group == "C2v":
        if (det, tr) == (1, -1):
            return "C2"
        if (det, tr) == (-1, 1) and ref.get("sv") is not None:
            return "sv" if abs(_axis(R) @ ref["sv"]) > 0.99 else "sv'"
    if group == "D2d":
        if (det, tr) == (-1, -1):
            return "S4"
        if (det, tr) == (1, -1):
            return "C2" if abs(_axis(R) @ ref["principal"]) > 0.99 else "C2'"
        if (det, tr) == (-1, 1):
            return "sd"
    return None


def _references(group, ops, xc, masses, tol):
    """Orientation-independent reference axes: the S4 axis of D2d; the sv mirror of C2v."""
    ref = {}
    if group == "D2d":
        s4 = [R for R, _ in ops if int(round(np.linalg.det(R))) == -1 and int(round(np.trace(R))) == -1]
        ref["principal"] = _axis(s4[0]) if s4 else None
    if group == "C2v":
        # Mulliken's convention: the plane holding more atoms is sv'(yz), so sv is the other one
        # (water's antisymmetric stretch is then B2); tie -> more mass spread within sv'
        mirrors = [_axis(R) for R, _ in ops if int(round(np.linalg.det(R))) == -1 and int(round(np.trace(R))) == 1]
        if mirrors:
            key = lambda nrm: (int((np.abs(xc @ nrm) < tol).sum()), -float((masses * (xc @ nrm) ** 2).sum()))
            ref["sv"] = min(mirrors, key=key)
    return ref


def _align(blocks, size):
    """
    Orthonormal basis of a (near-)degenerate set from per-irrep subspaces, each column as
    close as possible to one of the original modes, so non-degenerate modes that merely
    fall within the tolerance are not mixed.  blocks: [(label, U (size x m))].
    Returns (B (size x size), labels by column).
    """
    from scipy.optimize import linear_sum_assignment
    slots = [(b, j) for b, (_l, U) in enumerate(blocks) for j in range(U.shape[1])]
    cost = np.array([[-(blocks[b][1][pos] ** 2).sum() for pos in range(size)] for b, _j in slots])
    rows, cols = linear_sum_assignment(cost)
    pos_of = {slots[r]: c for r, c in zip(rows, cols)}
    B = np.zeros((size, size))
    labels = [None] * size
    for b, (lab, U) in enumerate(blocks):
        rows_b = [pos_of[(b, j)] for j in range(U.shape[1])]
        A, _sv, Vt = np.linalg.svd(U[rows_b, :])          # maximise trace(E_rows^T U Y) over orthogonal Y
        C = U @ (Vt.T @ A.T)
        for j, pos in enumerate(rows_b):
            B[:, pos] = C[:, j]
            labels[pos] = lab
    u, _s, vt = np.linalg.svd(B)                           # re-orthonormalise (approximate symmetry)
    return u @ vt, labels


def symmetry(symbols, pts, masses, modes, freqs, tol=0.1, degen_cm1=0.5):
    """
    Point group and a block-diagonal orthogonal U (modes x modes) such that modes @ U are
    symmetry-adapted, with irrep labels and totally symmetric flags of the adapted modes.
    Sets are modes within `degen_cm1` of the set's lowest one (no chaining).
    """
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    n = len(symbols)
    xc = pts - (masses[:, None] * pts).sum(0) / masses.sum()
    try:
        pga = PointGroupAnalyzer(Molecule(list(symbols), xc), tolerance=tol)
        group, raw_ops = pga.sch_symbol, pga.get_symmetry_operations()
    except Exception:
        group, raw_ops = "C1", []
    ops = []
    for op in raw_ops:
        R = op.rotation_matrix
        y = xc @ R.T
        perm = np.array([int(np.argmin(np.linalg.norm(xc - y[i], axis=1))) for i in range(n)])
        if np.abs(y - xc[perm]).max() > 3 * tol or len(set(perm)) != n:
            continue
        ops.append((R, perm))
    if not ops:
        group, ops = "C1", [(np.eye(3), np.arange(n))]
    table = TABLES.get(group)
    ref = _references(group, ops, xc, masses, tol) if table else {}
    classes = [(_op_class(group, R, ref) if table else None) for R, _ in ops]
    if table and any(classes.count(c) != k for c, k in table["order"].items()):
        table = None                     # operations do not match the tabulated classes

    def apply(R, perm, v):
        v = v.reshape(n, 3)
        out = np.zeros_like(v)
        out[perm] = v @ R.T
        return out.ravel()

    groups, i = [], 0
    while i < len(freqs):
        j = i + 1
        while j < len(freqs) and freqs[j] - freqs[i] < degen_cm1:
            j += 1
        groups.append(list(range(i, j)))
        i = j
    U_all = np.eye(len(freqs))
    labels = [None] * len(freqs)
    for g in groups:
        V = modes[:, g]
        D = [V.T @ np.stack([apply(R, p, V[:, k]) for k in range(len(g))], 1) for R, p in ops]
        blocks = []
        if table:
            for ir, chars in table["irreps"].items():
                ch = dict(zip(table["classes"], chars))
                # (for the merged complex pair E of C3 the eigenvalue is 2, not 1; the threshold holds)
                P = ch["E"] / len(ops) * sum(ch[c] * Dm for c, Dm in zip(classes, D))
                w, u = np.linalg.eigh(0.5 * (P + P.T))
                if (w > 0.5).any():
                    blocks.append((ir, u[:, w > 0.5]))
        if sum(U.shape[1] for _l, U in blocks) != len(g):
            # no table, or a set the table cannot split: separate only the totally symmetric part
            P = sum(D) / len(ops)
            w, u = np.linalg.eigh(0.5 * (P + P.T))
            other = "?" if table else "–"
            blocks = [(lab, u[:, sel]) for lab, sel in (("A" if not table else list(table["irreps"])[0], w > 0.5),
                                                        (other, w <= 0.5)) if sel.any()]
        B, lab = _align(blocks, len(g))
        U_all[np.ix_(g, g)] = B
        for k, l in zip(g, lab):
            labels[k] = l
    first = list(table["irreps"])[0] if table else "A"
    tsym = np.array([l == first for l in labels])
    return {"group": group, "n_ops": len(ops), "tabulated": table is not None,
            "ir_irreps": table["ir"] if table else None, "raman_irreps": table["raman"] if table else None,
            "labels": labels, "totally_symmetric": tsym, "U": U_all}


class FieldDipoles:
    """g-xTB dipoles of displaced geometries in static fields, restarting from the reference wavefunction."""

    def __init__(self, symbols, charge=0, acc=ACC):
        self.symbols = list(symbols)
        self.runner = XtbRunner("gxtb", threads=1)
        self.acc = acc
        self.charge = int(charge)
        self.restart = None

    def dipole(self, pts, field_VA):
        with tempfile.TemporaryDirectory(prefix="vibspec_") as d:
            d = Path(d)
            (d / "m.xyz").write_text(f"{len(self.symbols)}\nqdprops\n" + "".join(
                f"{s} {x:.10f} {y:.10f} {z:.10f}\n" for s, (x, y, z) in zip(self.symbols, pts)))
            cmd = [self.runner.binary, "m.xyz", "--gxtb", "--acc", self.acc, "--chrg", str(self.charge)]
            if np.any(field_VA):
                cmd += ["--efield", ",".join(f"{v:.8f}" for v in field_VA)]   # (not --efield=…, ignored)
            for attempt in range(3):
                if self.restart is not None and attempt == 0:
                    (d / "xtbrestart").write_bytes(self.restart)
                else:
                    (d / "xtbrestart").unlink(missing_ok=True)
                p = subprocess.run(cmd, cwd=d, env=self.runner.env, capture_output=True, text=True)
                out = p.stdout + p.stderr
                if "normal termination of xtb" in out and "abnormal termination" not in out:
                    mu = _parse_dipole(p.stdout)
                    if mu is not None:
                        if self.restart is None and not np.any(field_VA):
                            self.restart = (d / "xtbrestart").read_bytes()
                            self.gradient_norm = _parse_float(p.stdout, "GRADIENT NORM")
                        return mu
            raise RuntimeError("g-xTB field run failed:\n" + "\n".join(out.strip().splitlines()[-12:]))

    def response(self, pts, F):
        mus = [self.dipole(pts, s * F * np.eye(3)[j]) for j in range(3) for s in (1.0, -1.0)]
        mu = np.mean(mus, axis=0)
        f_au = F / AU_FIELD_V_A
        alpha = np.stack([(mus[2 * j] - mus[2 * j + 1]) / (2 * f_au) for j in range(3)], axis=1)
        return mu, alpha


def _parse_dipole(text):
    # g-xTB prints the full molecular dipole (charges + atomic dipoles) as "total  x y z" in a.u.
    for line in text.splitlines():
        t = line.split()
        if len(t) == 4 and t[0] == "total" and t[1] != "energy":
            try:
                return np.array([float(v) for v in t[1:]])
            except ValueError:
                pass
    return None


def _parse_float(text, key):
    for line in text.splitlines():
        if key in line:
            try:
                return float(line.split()[-3] if line.strip().endswith("|") else line.split()[-2])
            except (ValueError, IndexError):
                return None
    return None


def raman_invariants(dalpha):
    """(activity Å⁴/amu, depolarisation ratio) from dalpha/dQ in Å²/amu^1/2, shape (k, 3, 3)."""
    da = 0.5 * (dalpha + np.transpose(dalpha, (0, 2, 1)))
    a = np.trace(da, axis1=1, axis2=2) / 3
    g2 = 0.5 * ((da[:, 0, 0] - da[:, 1, 1]) ** 2 + (da[:, 1, 1] - da[:, 2, 2]) ** 2 + (da[:, 2, 2] - da[:, 0, 0]) ** 2
                + 6 * (da[:, 0, 1] ** 2 + da[:, 1, 2] ** 2 + da[:, 0, 2] ** 2))
    S = 45 * a ** 2 + 7 * g2
    rho = np.where(S > 1e-6 * max(S.max(), 1e-30), 3 * g2 / np.maximum(45 * a ** 2 + 4 * g2, 1e-30), np.nan)
    return S, rho


def bulk_gamma(cif, settings):
    """MACE Gamma-point optical frequencies (cm⁻¹) of the bulk (qdprops.bulk, cached)."""
    from ..bulk import mace_bulk
    return mace_bulk(cif, settings)["optical_cm1"]


def mode_class(freq, core, surface, ligand, breathing, ir_rel, raman_rel, omega_to):
    """
    Descriptive class of one mode.  Thresholds: 'optical' means nu >= 0.75 nu_TO(bulk, Gamma);
    a share > 0.4 of the amplitude on core atoms makes a mode 'core', > 0.5 on ligands 'M–X'.
    """
    if breathing >= 0.3:
        return "breathing"
    if ligand > 0.5:
        return "M–X ligand"
    if omega_to and freq >= 0.75 * omega_to:
        return "core optical" if core >= 0.4 else "surface optical"
    return "core acoustic-like" if core >= 0.4 else "surface / torsional"


def run(ctx) -> dict:
    s = ctx.settings
    symbols, pts = ctx.relaxed()
    pts = np.asarray(pts, float)
    n = len(symbols)
    if n > s.vibspec_max_atoms:
        return {"summary": {"skipped": f"{n} atoms > vibspec_max_atoms = {s.vibspec_max_atoms}"}}
    m = np.load(ctx.props / "modes.npz")
    freqs, modes, masses = m["frequencies_cm1"], m["modes_mass_weighted"], m["masses"]
    h, F = s.vibspec_step, s.vibspec_field
    charge = sum(ctx.charges.get(x, 0) for x in symbols)
    fd = FieldDipoles(symbols, charge)

    # Derivatives along the MACE modes as they come from the hessian step; the symmetry
    # adaptation is a rotation within degenerate sets applied afterwards, so it can change
    # without invalidating the checkpoint.  One checkpoint entry per mode, valid for this
    # geometry, mode set, step, field, charge, SCF accuracy and g-xTB build.
    key = hashlib.sha256(pts.round(8).tobytes() + modes.round(8).tobytes()
                         + f"{h}|{F}|{charge}|{ACC}|{fd.runner.version()}".encode()).hexdigest()[:16]
    cpath = ctx.props / CACHE
    cache = json.loads(cpath.read_text()) if cpath.is_file() else {}
    if cache.get("key") != key:
        cache = {"key": key, "basis": "mace", "modes": {}}
    lock = threading.Lock()
    t0 = time.time()
    fd.dipole(pts, np.zeros(3))                    # field-free reference: its wavefunction seeds every later run
    mu0, a0 = fd.response(pts, F)
    sqm = np.sqrt(np.repeat(masses, 3))

    def one(k):
        if str(k) in cache["modes"]:
            return
        dx = (modes[:, k] / sqm).reshape(-1, 3) * h
        mp, ap = fd.response(pts + dx, F)
        mm, am = fd.response(pts - dx, F)
        with lock:
            cache["modes"][str(k)] = {"dmu": ((mp - mm) / (2 * h)).tolist(), "dalpha": ((ap - am) / (2 * h)).tolist()}
            tmp = cpath.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            os.replace(tmp, cpath)                 # atomic: a kill never leaves a truncated checkpoint

    n_cached = sum(1 for k in range(len(freqs)) if str(k) in cache["modes"])
    workers = s.vibspec_workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, range(len(freqs))))
    dmu = np.array([cache["modes"][str(k)]["dmu"] for k in range(len(freqs))]) * AU_DEBYE        # D/(amu^1/2 Å)
    dal = np.array([cache["modes"][str(k)]["dalpha"] for k in range(len(freqs))]) * BOHR_A ** 3  # Å²/amu^1/2
    sym = symmetry(symbols, pts, masses, modes, freqs)
    U = sym["U"]
    modes = modes @ U                               # symmetry-adapted modes; derivatives rotate with them
    dmu = U.T @ dmu
    dal = np.einsum("kl,kij->lij", U, dal)
    ir = KMMOL * (dmu ** 2).sum(1)
    S, rho = raman_invariants(dal)

    role = ctx.results.get("structure", {}).get("role") or ["surface"] * n
    amp = (modes.reshape(n, 3, -1) ** 2).sum(1)
    share = {r: amp[[i for i in range(n) if role[i] == r]].sum(0) for r in ("core", "surface", "ligand")}
    elem = {e: amp[[i for i in range(n) if symbols[i] == e]].sum(0) for e in sorted(set(symbols))}
    xc = pts - (masses[:, None] * pts).sum(0) / masses.sum()
    rhat = xc / np.maximum(np.linalg.norm(xc, axis=1), 1e-9)[:, None]
    radial = ((modes.reshape(n, 3, -1) * rhat[:, :, None]).sum(1) ** 2).sum(0)
    u = (xc * np.sqrt(masses)[:, None]).ravel()
    breathing = (u / np.linalg.norm(u)) @ modes
    breathing = breathing ** 2

    try:
        gamma = bulk_gamma(ctx.cif, s) if ctx.cif else []
    except Exception as exc:  # the bulk line is a guide only
        gamma = []
        print(f"[qdprops]   vibspec: bulk Gamma frequencies failed ({exc})")
    omega_to = max(gamma) if gamma else None
    irr, rr = ir / max(ir.max(), 1e-30), S / max(S.max(), 1e-30)
    classes = [mode_class(freqs[k], share["core"][k], share["surface"][k], share["ligand"][k], breathing[k],
                          irr[k], rr[k], omega_to) for k in range(len(freqs))]
    modes_out = [{"nu": float(freqs[k]), "irrep": sym["labels"][k], "totally_symmetric": bool(sym["totally_symmetric"][k]),
                  "ir_km_mol": float(ir[k]), "raman_A4_amu": float(S[k]),
                  "rho": None if np.isnan(rho[k]) else float(rho[k]),
                  "core": float(share["core"][k]), "surface": float(share["surface"][k]),
                  "ligand": float(share["ligand"][k]), "radial": float(radial[k]), "breathing": float(breathing[k]),
                  "elements": {e: float(v[k]) for e, v in elem.items()}, "class": classes[k]}
                 for k in range(len(freqs))]
    np.savez_compressed(ctx.props / "vibspec_modes.npz", frequencies_cm1=freqs, modes_symmetry_adapted=modes,
                        dmu_dQ_D=dmu, dalpha_dQ_A2=dal)

    def top(arr, n_=4):
        # partners of a degenerate (E, T) irrep within 0.5 cm-1 are summed into one peak
        peaks = []
        for k in np.argsort(freqs):
            if (peaks and sym["labels"][k][:1] in ("E", "T") and peaks[-1][1] == sym["labels"][k]
                    and abs(freqs[k] - peaks[-1][0]) < 0.5):
                peaks[-1][2] += float(arr[k])
            else:
                peaks.append([float(freqs[k]), sym["labels"][k], float(arr[k])])
        return [[round(p[0], 1), p[1], round(p[2], 2)] for p in sorted(peaks, key=lambda p: -p[2])[:n_]]
    alpha_A3 = a0 * BOHR_A ** 3
    return {
        "summary": {
            "point_group": sym["group"], "n_modes": int(len(freqs)),
            "ir_active_irreps": sym["ir_irreps"], "raman_active_irreps": sym["raman_irreps"],
            "n_totally_symmetric": int(sym["totally_symmetric"].sum()),
            "alpha_iso_A3": float(np.trace(alpha_A3) / 3), "dipole_D": float(np.linalg.norm(mu0) * AU_DEBYE),
            "top_ir": top(ir), "top_raman": top(S),
            "bulk_gamma_optical_cm1": gamma, "gxtb_gradient_norm_Eh_bohr": getattr(fd, "gradient_norm", None),
            "step_amu05_A": h, "field_V_A": F, "n_gxtb_runs": 12 * (len(freqs) - n_cached) + 7,
            "n_modes_from_checkpoint": n_cached,
            "seconds_gxtb": round(time.time() - t0, 1),
        },
        "alpha_A3": alpha_A3.tolist(),
        "modes": modes_out,
        "provenance": {**fd.runner.provenance(), "frequencies": "MACE-MH-1 (hessian step)"},
    }
