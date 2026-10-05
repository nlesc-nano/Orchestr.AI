# src/orchestr_ai/qd/steps/hessian.py
"""
Harmonic vibrations at the MACE-MH-1 minimum.

The Cartesian Hessian is MACE's analytic (autograd) one for small dots and a
central finite difference of forces otherwise.  It is mass-weighted and the
three translations and three rotations are removed exactly by diagonalising
it in the orthogonal complement of their (Eckart) vectors, which leaves the
3N - 6 vibrations.  From the frequencies: zero-point energy, harmonic
U_vib(T), S_vib(T), Cv(T), F_vib(T), and the vibrational density of states,
total and projected on elements and on core / surface / ligand atoms (role
from the structure step) by the squared mass-weighted mode amplitudes.
"""
from __future__ import annotations

import json

import numpy as np

from ..engines import mace_calculator, mace_hessian, mace_provenance

# sqrt(eV / (Å² amu)) in cm⁻¹ and hc in eV·cm
EIG_TO_CM1 = 521.4708983725064
CM1_TO_EV = 1.2398419843320026e-4
KB_EV = 8.617333262e-5


def _fd_hessian(atoms, delta: float) -> np.ndarray:
    pos0 = atoms.get_positions().copy()
    n = len(atoms)
    h = np.zeros((3 * n, 3 * n))
    for k in range(3 * n):
        i, a = divmod(k, 3)
        f = []
        for sgn in (1.0, -1.0):
            p = pos0.copy()
            p[i, a] += sgn * delta
            atoms.set_positions(p)
            f.append(atoms.get_forces().ravel())
        h[:, k] = -(f[0] - f[1]) / (2.0 * delta)
    atoms.set_positions(pos0)
    return h


def _tr_basis(pos: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Orthonormal mass-weighted translation/rotation vectors, shape (3N, k)."""
    n = len(masses)
    sm = np.sqrt(masses)
    r = pos - (masses[:, None] * pos).sum(axis=0) / masses.sum()
    vecs = []
    for a in range(3):
        v = np.zeros((n, 3))
        v[:, a] = sm
        vecs.append(v.ravel())
    for a in range(3):
        e = np.zeros(3)
        e[a] = 1.0
        vecs.append((np.cross(e, r) * sm[:, None]).ravel())
    # SVD, not unpivoted QR: for a linear molecule off the Cartesian axes two rotation
    # vectors are parallel and QR would drop the wrong direction.
    U, sv, _ = np.linalg.svd(np.stack(vecs, axis=1), full_matrices=False)
    return U[:, sv > 1e-8 * sv[0]]


def vibrations(h: np.ndarray, pos: np.ndarray, masses: np.ndarray):
    """(frequencies in cm⁻¹ (imaginary < 0), mass-weighted eigenvectors (3N, nvib), TR residuals in cm⁻¹)."""
    n = len(masses)
    h = 0.5 * (h + h.T)
    w = 1.0 / np.sqrt(np.repeat(masses, 3))
    hm = h * w[:, None] * w[None, :]
    d = _tr_basis(pos, masses)
    # Complement of the TR space: the remaining columns of a full QR.
    full, _ = np.linalg.qr(np.concatenate([d, np.eye(3 * n)], axis=1))
    comp = full[:, d.shape[1]:3 * n]
    lam, v = np.linalg.eigh(comp.T @ hm @ comp)
    modes = comp @ v
    freqs = np.sign(lam) * np.sqrt(np.abs(lam)) * EIG_TO_CM1
    lam_tr = np.linalg.eigvalsh(d.T @ hm @ d)
    tr = np.sign(lam_tr) * np.sqrt(np.abs(lam_tr)) * EIG_TO_CM1
    return freqs, modes, tr


SOFT_FLOOR_CM1 = 10.0    # |nu| below this is raised to it
QRRHO_W0_CM1 = 100.0     # Grimme's quasi-RRHO switching frequency
B_AV = 1.0e-44           # kg m², Grimme's average molecular moment of inertia
_H, _KB_SI, _C_CM = 6.62607015e-34, 1.380649e-23, 2.99792458e10


def thermo_modes(freqs_cm1) -> np.ndarray:
    """Frequencies used for thermochemistry: imaginary modes as |nu|, soft ones floored."""
    return np.maximum(np.abs(np.asarray(freqs_cm1, float)), SOFT_FLOOR_CM1)


def thermo(freqs_cm1: np.ndarray, temperatures) -> dict:
    """
    Vibrational thermochemistry per temperature: harmonic ZPE, U_vib and C_v;
    entropy by Grimme's quasi-RRHO (harmonic above ~100 cm-1, free rotor for
    soft modes, weight w = 1/(1 + (w0/nu)^4)), so soft modes do not produce a
    spurious entropy.  Imaginary modes are taken as |nu|, |nu| < 10 cm-1 as 10.
    The same treatment is used for every species (qdprops.references.ideal_gas_g).
    """
    nu = thermo_modes(freqs_cm1)
    e = nu * CM1_TO_EV
    zpe = 0.5 * e.sum()
    w = 1.0 / (1.0 + (QRRHO_W0_CM1 / nu) ** 4)
    mu = _H / (8 * np.pi ** 2 * nu * _C_CM)                  # free-rotor moment of inertia
    mu_eff = mu * B_AV / (mu + B_AV)
    rows = {"T": [], "U_vib_eV": [], "S_vib_meV_K": [], "Cv_meV_K": [], "F_vib_eV": [], "F_vib_harmonic_eV": []}
    for t in temperatures:
        x = e / (KB_EV * t)
        ex = np.expm1(x)
        u = zpe + (e / ex).sum()
        s_harm = KB_EV * (x / ex - np.log1p(-np.exp(-x)))
        rows["F_vib_harmonic_eV"].append(float(u - t * s_harm.sum()))
        s_rot = KB_EV * (0.5 + np.log(np.sqrt(8 * np.pi ** 3 * mu_eff * _KB_SI * t / _H ** 2)))
        s = (w * s_harm + (1 - w) * s_rot).sum()
        cv = KB_EV * (x * x * np.exp(-x) / (-np.expm1(-x)) ** 2).sum()
        rows["T"].append(float(t))
        rows["U_vib_eV"].append(float(u))
        rows["S_vib_meV_K"].append(1000.0 * float(s))
        rows["Cv_meV_K"].append(1000.0 * float(cv))
        rows["F_vib_eV"].append(float(u - t * s))
    return {"zpe_eV": float(zpe), **rows}


def vdos(freqs: np.ndarray, modes: np.ndarray, groups: dict, sigma: float) -> dict:
    """Gaussian-broadened VDOS on a 1 cm⁻¹ grid, normalised to one state per mode."""
    real = freqs > 0
    f = freqs[real]
    amp = (modes[:, real].reshape(-1, 3, real.sum()) ** 2).sum(axis=1)   # (N, nmodes), columns sum to 1
    grid = np.arange(0.0, float(f.max()) + 5 * sigma + 1.0, 1.0)
    g = np.exp(-0.5 * ((grid[:, None] - f[None, :]) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    out = {"cm1": grid.tolist(), "total": (g.sum(axis=1)).tolist(), "projected": {}}
    for kind, members in groups.items():
        out["projected"][kind] = {}
        for name, idx in members.items():
            wgt = amp[idx].sum(axis=0)
            out["projected"][kind][name] = (g @ wgt).tolist()
    return out


def run(ctx) -> dict:
    from ase import Atoms

    s = ctx.settings
    symbols, pts = ctx.relaxed()
    atoms = Atoms(list(symbols), positions=np.asarray(pts, float))
    atoms.calc = mace_calculator(s.head, s.model, s.device, s.dtype)
    n = len(atoms)
    method = s.hessian
    if method == "auto":
        method = "analytic" if n <= s.analytic_max_atoms else "fd"
    h = mace_hessian(atoms, atoms.calc) if method == "analytic" else _fd_hessian(atoms, s.fd_step)
    masses = atoms.get_masses()
    freqs, modes, tr = vibrations(h, atoms.get_positions(), masses)

    role = ctx.results.get("structure", {}).get("role") or ["atom"] * n
    groups = {
        "element": {e: [i for i, x in enumerate(symbols) if x == e] for e in sorted(set(symbols))},
        "role": {r: [i for i, x in enumerate(role) if x == r] for r in ("core", "surface", "ligand")
                 if r in role},
    }
    th = thermo(freqs, s.temperatures)                    # quasi-RRHO entropy, see thermo()
    dos = vdos(freqs, modes, groups, s.vdos_sigma)
    np.savez_compressed(ctx.props / "modes.npz", frequencies_cm1=freqs, modes_mass_weighted=modes,
                        masses=masses, symbols=np.array(symbols))
    spectra_path = ctx.props / "spectra.json"
    spectra = json.loads(spectra_path.read_text()) if spectra_path.is_file() else {}
    spectra.update({"vdos": dos, "thermo": th})
    spectra_path.write_text(json.dumps(spectra))

    imag = freqs[freqs < 0]
    i300 = th["T"].index(300.0) if 300.0 in th["T"] else None
    return {
        "summary": {
            "method": method,
            "n_modes": int(len(freqs)),
            "n_imaginary": int((freqs < -10.0).sum()),
            "n_imaginary_small": int(((freqs < 0) & (freqs >= -10.0)).sum()),
            "lowest_cm1": float(freqs.min()),
            "lowest_real_cm1": float(freqs[freqs > 0].min()),
            "highest_cm1": float(freqs.max()),
            "zpe_eV": th["zpe_eV"],
            "zpe_meV_atom": 1000.0 * th["zpe_eV"] / n,
            "tr_residual_max_cm1": float(np.abs(tr).max()),
            **({"S_vib_300K_meV_K": th["S_vib_meV_K"][i300], "Cv_300K_meV_K": th["Cv_meV_K"][i300],
                "F_vib_300K_eV": th["F_vib_eV"][i300]} if i300 is not None else {}),
        },
        "imaginary_cm1": imag.tolist(),
        "frequencies_cm1": freqs.tolist(),
        "provenance": mace_provenance(s.head, s.model, s.device, s.dtype),
    }
