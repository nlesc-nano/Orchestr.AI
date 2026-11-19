# src/orchestr_ai/qd/ligand_restore.py
"""
Ligand geometry of Wigner frames.

A Wigner frame displaces every atom along straight lines, x = x0 + sum_j L_j Q_j / sqrt(m).
For a rotation-like mode (a methyl torsion, a carboxylate libration) the true motion is an
arc, and the straight line lengthens the bonds it should only turn: at second order in the
amplitude, so a 20-400 cm-1 mode at its thermal width stretches C-H by up to 2-3 A
(Cd10Se6(OAc)8 with MACE-MH-1: frames 2.4x (300 K) to 3.4x (600 K) above the harmonic
energy, up to 127 eV).

The fix keeps the frame and moves only the atoms of covalent ligands, as little as possible,
until every ligand bond (1-2) and every 1-3 distance (the angles) has its first-order value:

    d_ij(target) = d_ij(0) + e_ij . (dx_j - dx_i)

So the real stretch and bend fluctuations (the first-order changes) stay in, the rotations
stay in (the torsion spread is unchanged), and only the spurious second-order lengthening
goes.  Metal atoms, core anions and halide ligands (no non-metal neighbour) are never moved,
and a dot without covalent ligands (Cl-capped) comes back unchanged at no cost.  With the
restore the mean MACE energy of the frames equals the harmonic <V> of the sampled modes.

Covalent bonds: non-metal pairs within ASE's natural cutoffs x 1.1 at the minimum.  The
metal-ligand contacts (Cd-O) are left free: restoring them as well changed neither their
spread nor the energies (it is what the soft modes themselves give, not the straight-line
artefact).  Velocities are not changed (they are first order already).

Clashes.  The restore keeps every ligand's own geometry but not the contacts between
ligands: at 600 K about 1 frame in 100 of Cd10Se6(OAc)8 had a methyl H 0.6 A from an O of the
next acetate (33 eV above the minimum).  `clashes(x)` lists the pairs with a ligand atom that
are not bonded (nor 1-3) at the minimum and are closer than CLASH_FRAC x the sum of their
covalent radii; the samplers redraw such frames (up to MAX_REDRAW times).  Leaving out these
rare frames trims the far tail of the distribution only.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# elements whose mutual contacts are covalent bonds; everything else is a metal for this purpose
NONMETALS = frozenset("H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe At Rn".split())
CUTOFF_MULT = 1.1
LAMBDA = 0.05          # weight of |x - x_linear| in the least squares: the smallest change that restores
CLASH_FRAC = 0.75      # a non-bonded pair closer than this x (r_cov,i + r_cov,j) is a clash
MAX_REDRAW = 50        # redraws of one clashing frame before it is kept as drawn


class LigandRestore:
    """Restores the 1-2 and 1-3 distances of covalent ligands in displaced frames; build() gives None when there are none."""

    def __init__(self, positions, cons: np.ndarray):
        pos0 = np.asarray(positions, float)
        self.pos0 = pos0
        self.cons = cons
        v = pos0[cons[:, 1]] - pos0[cons[:, 0]]
        self.d0 = np.linalg.norm(v, axis=1)
        self.e0 = v / self.d0[:, None]
        self.atoms = np.unique(cons)
        idx = np.full(len(pos0), -1)
        idx[self.atoms] = np.arange(len(self.atoms))
        self.ci, self.cj = idx[cons[:, 0]], idx[cons[:, 1]]
        self._rows = np.repeat(np.arange(len(cons)), 6)
        self._cols = np.stack([3 * self.ci, 3 * self.ci + 1, 3 * self.ci + 2,
                               3 * self.cj, 3 * self.cj + 1, 3 * self.cj + 2], axis=1).ravel()
        self._clash_atoms = None

    @classmethod
    def build(cls, symbols, positions) -> Optional["LigandRestore"]:
        cons = ligand_constraints(symbols, positions)
        if not len(cons):
            return None
        r = cls(positions, cons)
        r._setup_clashes(symbols)
        return r

    def _setup_clashes(self, symbols):
        """The pairs a clash check skips: bonded at the minimum (any elements) and the ligand 1-3 pairs."""
        from ase import Atoms
        from ase.neighborlist import natural_cutoffs, neighbor_list

        n = len(self.pos0)
        at = Atoms(list(symbols), positions=self.pos0)
        self._clash_atoms = at
        self._clash_cut = natural_cutoffs(at, mult=CLASH_FRAC)
        i, j = neighbor_list("ij", at, natural_cutoffs(at, mult=max(CUTOFF_MULT, CLASH_FRAC)))
        i, j = np.minimum(i, j), np.maximum(i, j)
        self._skip = set((i * n + j).tolist()) | set((self.cons[:, 0] * n + self.cons[:, 1]).tolist())
        self._ligand = np.zeros(n, bool)
        self._ligand[self.atoms] = True

    def clashes(self, x) -> np.ndarray:
        """(k, 2) pairs of frame x (n, 3) A with a ligand atom, not bonded at the minimum, closer than
        CLASH_FRAC x (r_cov,i + r_cov,j)."""
        from ase.neighborlist import neighbor_list

        if self._clash_atoms is None:
            return np.zeros((0, 2), dtype=int)
        at = self._clash_atoms
        at.set_positions(np.asarray(x, float))
        i, j = neighbor_list("ij", at, self._clash_cut)
        n = len(at)
        keep = (i < j) & (self._ligand[i] | self._ligand[j])
        i, j = i[keep], j[keep]
        new = np.array([k not in self._skip for k in (i * n + j).tolist()], bool)
        return np.stack([i[new], j[new]], axis=1).reshape(-1, 2)

    def summary(self) -> dict:
        return {"atoms": int(len(self.atoms)), "constraints": int(len(self.cons))}

    def __call__(self, x_lin) -> np.ndarray:
        """One frame (n, 3) A -> the same frame with the ligand distances restored."""
        from scipy.optimize import least_squares
        from scipy.sparse import coo_matrix, identity, vstack

        x_lin = np.asarray(x_lin, float)
        dx = x_lin - self.pos0
        target = self.d0 + np.einsum("ij,ij->i", self.e0, dx[self.cons[:, 1]] - dx[self.cons[:, 0]])
        z0 = x_lin[self.atoms].ravel()
        ci, cj = self.ci, self.cj
        reg = LAMBDA * identity(len(z0), format="csr")

        def res(z):
            X = z.reshape(-1, 3)
            return np.concatenate([np.linalg.norm(X[cj] - X[ci], axis=1) - target, LAMBDA * (z - z0)])

        def jac(z):
            X = z.reshape(-1, 3)
            u = X[cj] - X[ci]
            u /= np.linalg.norm(u, axis=1)[:, None]
            a = coo_matrix((np.concatenate([-u, u], axis=1).ravel(), (self._rows, self._cols)),
                           shape=(len(target), len(z)))
            return vstack([a, reg]).tocsr()

        z = least_squares(res, z0, jac=jac, method="trf", xtol=1e-10, ftol=1e-10).x
        out = x_lin.copy()
        out[self.atoms] = z.reshape(-1, 3)
        return out

    def frames(self, P) -> np.ndarray:
        return np.stack([self(p) for p in np.asarray(P, float)])


def ligand_constraints(symbols, positions) -> np.ndarray:
    """(k, 2) atom pairs: non-metal bonds and the 1-3 pairs around every non-metal atom of a covalent ligand."""
    from ase import Atoms
    from ase.neighborlist import natural_cutoffs, neighbor_list

    symbols = list(symbols)
    nm = np.array([s in NONMETALS for s in symbols])
    if nm.sum() < 2:
        return np.zeros((0, 2), dtype=int)
    atoms = Atoms(symbols, positions=np.asarray(positions, float))
    i, j = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=CUTOFF_MULT))
    keep = nm[i] & nm[j]
    i, j = i[keep], j[keep]
    nbr = {}
    for a, b in zip(i, j):
        nbr.setdefault(int(a), set()).add(int(b))
    cons = {(min(a, b), max(a, b)) for a, b in zip(i.tolist(), j.tolist())}
    for c, ns in nbr.items():
        ns = sorted(ns)
        cons.update((a, b) for k, a in enumerate(ns) for b in ns[k + 1:])
    return np.array(sorted(cons), dtype=int).reshape(-1, 2)
