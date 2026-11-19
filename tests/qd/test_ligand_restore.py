"""Ligand restore of Wigner frames: straight-line rotations of covalent groups lose their bond stretch, nothing else moves."""
import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("scipy")
from orchestr_ai.qd.ligand_restore import LigandRestore, ligand_constraints  # noqa: E402


def _cd_acetate():
    """Acetic acid with its OH hydrogen replaced by a Cd next to both O atoms (a bidentate Cd acetate)."""
    from ase.build import molecule
    m = molecule("CH3COOH")
    sym = m.get_chemical_symbols()
    pos = m.get_positions()
    o = [i for i, s in enumerate(sym) if s == "O"]
    c_carb = [i for i, s in enumerate(sym) if s == "C" and all(np.linalg.norm(pos[i] - pos[k]) < 1.5 for k in o)][0]
    h_acid = [i for i, s in enumerate(sym) if s == "H" and min(np.linalg.norm(pos[i] - pos[k]) for k in o) < 1.1][0]
    mid = pos[o].mean(axis=0)
    out = mid + 2.0 * (mid - pos[c_carb]) / np.linalg.norm(mid - pos[c_carb])
    keep = [i for i in range(len(sym)) if i != h_acid]
    return [sym[i] for i in keep] + ["Cd"], np.vstack([pos[keep], out])


def _methyl(sym, pos):
    c = [i for i, s in enumerate(sym) if s == "C"]
    hs = [i for i, s in enumerate(sym) if s == "H"]
    cm = [i for i in c if sum(np.linalg.norm(pos[i] - pos[h]) < 1.2 for h in hs) == 3][0]
    cc = [i for i in c if i != cm][0]
    return cm, cc, hs


def test_no_covalent_ligands_gives_none():
    from ase.build import bulk
    a = bulk("CdSe", "zincblende", a=6.05) * (2, 2, 2)
    sym, pos = a.get_chemical_symbols() + ["Cl"], np.vstack([a.get_positions(), [[-2.4, 0.0, 0.0]]])
    assert len(ligand_constraints(sym, pos)) == 0
    assert LigandRestore.build(sym, pos) is None


def test_straight_line_methyl_rotation_keeps_bonds_and_turns():
    sym, pos = _cd_acetate()
    r = LigandRestore.build(sym, pos)
    assert r is not None and sym.index("Cd") not in r.atoms
    cm, cc, hs = _methyl(sym, pos)
    axis = (pos[cm] - pos[cc]) / np.linalg.norm(pos[cm] - pos[cc])
    theta = np.radians(40.0)
    x_lin = pos.copy()
    for h in hs:                                            # the tangent of a 40 degree torsion: what a mode does
        x_lin[h] += theta * np.cross(axis, pos[h] - pos[cm])
    d = lambda x, i, j: np.linalg.norm(x[i] - x[j])
    assert max(d(x_lin, cm, h) - d(pos, cm, h) for h in hs) > 0.08     # straight line: C-H stretched
    x = r(x_lin)
    for h in hs:
        assert abs(d(x, cm, h) - d(pos, cm, h)) < 2e-3
    # the torsion stays: the H atoms turned about the C-C axis by about the sampled angle
    def ang(x, h):
        a = x[h] - x[cm]
        a -= a.dot(axis) * axis
        b = pos[h] - pos[cm]
        b -= b.dot(axis) * axis
        return np.degrees(np.arctan2(np.cross(b, a).dot(axis), a.dot(b)))
    assert all(25.0 < ang(x, h) < 45.0 for h in hs)
    assert np.allclose(x[sym.index("Cd")], x_lin[sym.index("Cd")])  # metals never move


def test_first_order_stretch_is_kept():
    sym, pos = _cd_acetate()
    r = LigandRestore.build(sym, pos)
    cm, cc, hs = _methyl(sym, pos)
    x_lin = pos.copy()
    u = (pos[hs[0]] - pos[cm]) / np.linalg.norm(pos[hs[0]] - pos[cm])
    x_lin[hs[0]] += 0.15 * u                                   # a pure C-H stretch: already exact
    x = r(x_lin)
    assert abs(np.linalg.norm(x[hs[0]] - x[cm]) - np.linalg.norm(x_lin[hs[0]] - x_lin[cm])) < 1e-3
    assert np.abs(x - x_lin).max() < 0.02


def test_wigner_samples_restore_hook():
    from orchestr_ai.qd.steps.ensemble import wigner_samples
    sym, pos = _cd_acetate()
    r = LigandRestore.build(sym, pos)
    n = len(sym)
    rng = np.random.default_rng(0)
    modes, _ = np.linalg.qr(rng.standard_normal((3 * n, 3 * n - 6)))
    freqs = np.linspace(30.0, 3000.0, 3 * n - 6)
    masses = np.ones(n) * 12.0
    P0, V0 = wigner_samples(pos, masses, freqs, modes, 300.0, 3, np.random.default_rng(1))
    P1, V1 = wigner_samples(pos, masses, freqs, modes, 300.0, 3, np.random.default_rng(1), restore=r)
    assert np.array_equal(V0, V1)
    assert np.allclose(P1[:, sym.index("Cd")], P0[:, sym.index("Cd")])
    assert np.allclose(P1, r.frames(P0))


def test_clash_between_ligands_is_found():
    sym, pos = _cd_acetate()
    pos2 = pos + np.array([8.0, 0.0, 0.0])                     # a second Cd acetate, far away
    sym2, P = sym + sym, np.vstack([pos, pos2])
    r = LigandRestore.build(sym2, P)
    assert len(r.clashes(P)) == 0                              # the minimum has no clash
    n = len(sym)
    o2 = n + [i for i, s in enumerate(sym) if s == "O"][0]
    h1 = [i for i, s in enumerate(sym) if s == "H"][0]
    x = P.copy()
    u = (P[o2] - P[h1]) / np.linalg.norm(P[o2] - P[h1])
    x[h1] = P[o2] - 0.6 * u                                    # a methyl H 0.6 A from the other acetate's O
    pairs = {tuple(p) for p in r.clashes(x).tolist()}
    assert (min(h1, o2), max(h1, o2)) in pairs
    assert all(a in r.atoms or b in r.atoms for a, b in pairs)


def test_wigner_samples_redraws_clashing_frames():
    from orchestr_ai.qd.steps.ensemble import wigner_samples
    sym, pos = _cd_acetate()
    r = LigandRestore.build(sym, pos)
    n = len(sym)
    rng = np.random.default_rng(0)
    modes, _ = np.linalg.qr(rng.standard_normal((3 * n, 3 * n - 6)))
    freqs = np.linspace(30.0, 3000.0, 3 * n - 6)
    masses = np.ones(n) * 12.0

    class FirstFrameClashes:                                   # the restore, with frame 0 of the first draw clashing
        def __init__(self, r):
            self.r, self.calls = r, 0
        def frames(self, P):
            return self.r.frames(P)
        def clashes(self, x):
            self.calls += 1
            return np.array([[0, 1]]) if self.calls == 1 else np.zeros((0, 2), int)

    P0, V0 = wigner_samples(pos, masses, freqs, modes, 300.0, 3, np.random.default_rng(1), restore=r)
    stats = {}
    P1, V1 = wigner_samples(pos, masses, freqs, modes, 300.0, 3, np.random.default_rng(1),
                            restore=FirstFrameClashes(r), stats=stats)
    assert stats == {"redrawn": 1, "clashing": 0}
    assert not np.allclose(P1[0], P0[0]) and not np.allclose(V1[0], V0[0])     # frame 0 drawn again
    assert np.array_equal(P1[1:], P0[1:]) and np.array_equal(V1[1:], V0[1:])   # the others unchanged
