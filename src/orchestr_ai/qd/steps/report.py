# src/orchestr_ai/qd/steps/report.py
"""
Summary figures for one dot: props/ground_state.html (interactive Plotly, the
webapp's Properties tab) and props/ground_state.png (static, matplotlib).

Panels in four groups (structure and relaxation, vibrations and
thermochemistry, vibrational spectra when the vibspec step ran, stability and
ligands), each with a caption that defines
what is plotted and how to read it, plus a table of key numbers.  The relaxed
geometry itself is not drawn (the webapp viewer shows it) and electronic
properties are left to the DFT panels.

Colours keep one meaning each: sites (core / surface / ligand) use the
reference categorical slots blue, orange, aqua; elements use violet, magenta,
green (both sets validated all-pairs for colour-vision deficiency); the
coverage map uses a one-hue sequential blue ramp.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
import textwrap

import numpy as np

from ..dashboards import CONTROLS_CSS
from . import vibplots

SITE = {"core": "#2a78d6", "surface": "#eb6834", "ligand": "#1baf7a"}
ELEMENT_SLOTS = ["#4a3aa7", "#e87ba4", "#008300"]
INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
START = "#8a8984"                 # builder start geometry (neutral)
KB_MEV = 8.617333262e-2           # meV/K


# --------------------------------------------------------------------------
# Text helpers: formulas with subscripts in HTML and matplotlib mathtext
# --------------------------------------------------------------------------

def formula_html(f: str) -> str:
    return re.sub(r"(\d+)", r"<sub>\1</sub>", html.escape(str(f)))


def formula_tex(f: str) -> str:
    return re.sub(r"(\d+)", r"$_{\1}$", str(f))


def _elements(ctx, syms):
    return sorted(set(syms), key=lambda e: (e not in ctx.native, ctx.native.index(e) if e in ctx.native else 0, e))


def _wrap(text: str, width: int) -> str:
    """Wrap text without breaking inside $...$ mathtext."""
    parts = re.split(r"(\$[^$]*\$)", text)
    protected = "".join(p.replace(" ", " ") if p.startswith("$") else p for p in parts)
    lines = []
    for para in protected.split("\n"):
        lines += textwrap.wrap(para, width, break_long_words=False, break_on_hyphens=False) or [""]
    return "\n".join(lines).replace(" ", " ")


# --------------------------------------------------------------------------
# Data shared by both renderers
# --------------------------------------------------------------------------

def _bond_rows(ctx):
    """Per-bond lengths (start, relaxed) with categories core / surface / ligand."""
    from .structure import _bonds, _cutoffs
    st = ctx.results["structure"]
    role = st["role"]
    bulk = st["summary"].get("bulk_bond_A")
    cut = _cutoffs(ctx.symbols, ctx.charges, ctx.native, bulk)
    _syms_r, pts_r = ctx.relaxed()
    nat = set(ctx.native)
    rows = {}
    for tag, pts in (("start", ctx.start_pts), ("relaxed", np.asarray(pts_r, float))):
        pts = np.asarray(pts, float)
        nb = _bonds(ctx.symbols, pts, cut)
        for i in range(len(ctx.symbols)):
            for j in nb[i]:
                if j <= i:
                    continue
                a, b = sorted((ctx.symbols[i], ctx.symbols[j]), key=lambda e: (ctx.charges.get(e, 0) < 0, e))
                if a in nat and b in nat:
                    cat = "core" if role[i] == role[j] == "core" else "surface"
                    label = f"{a}–{b}, {cat}"
                else:
                    cat = "ligand"
                    label = f"{a}–{b}"
                row = rows.setdefault((cat, label), {"start": [], "relaxed": [], "pairs": []})
                row[tag].append(float(np.linalg.norm(pts[i] - pts[j])))
                if tag == "relaxed":
                    row["pairs"].append(f"{ctx.symbols[i]}{i + 1}–{ctx.symbols[j]}{j + 1}")
    order = {"core": 0, "surface": 1, "ligand": 2}
    return sorted(rows.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])), bulk


def _data(ctx) -> dict:
    r = ctx.results
    syms, _ = ctx.relaxed()
    spectra = json.loads((ctx.props / "spectra.json").read_text())
    elements = _elements(ctx, syms)
    d = {
        "n": len(syms), "elements": elements,
        "ecol": {e: ELEMENT_SLOTS[i % 3] for i, e in enumerate(elements)},
        "relax": r["relax"]["summary"], "trace": np.asarray(r["relax"]["energy_trace_eV"], float),
        "structure": r["structure"], "vdos": spectra["vdos"], "thermo": spectra["thermo"],
        "hess": r["hessian"]["summary"], "stab": r.get("stability", {}), "det": r.get("detachment", {}),
        "rec": ctx.record,
        "vib": r.get("vibspec") if r.get("vibspec", {}).get("modes") else None,
        "sites": r.get("sites"),
    }
    d["bond_rows"], d["bulk_bond_cif"] = _bond_rows(ctx)
    d["bulk_mace"], d["bulk_exp"] = _bulk_bonds(ctx)
    core = [x for (cat, _l), v in d["bond_rows"] if cat == "core" for x in v["relaxed"]]
    ref = d["bulk_mace"] or d["bulk_bond_cif"]
    d["core_strain"] = 100 * (np.mean(core) / ref - 1) if core and ref else None
    u = d["stab"].get("summary", {}).get("units")
    if u:
        q = u["q"]
        d["unit_ma"] = f"{u['M']}{u['A']}"
        d["unit_mx"] = f"{u['M']}{u['X']}{q if q > 1 else ''}" if u["X"] else None
        d["n_ma"], d["m_mx"] = u["n_MA"], u["m_MXq"]
    return d


def _bulk_bonds(ctx):
    """(MACE-MH-1 bulk bond Å or None, (experimental bond Å, reference) or None)."""
    from ..bulk import experimental_bond, mace_bulk
    if ctx.cif is None:
        return None, None
    try:
        mace = mace_bulk(ctx.cif, ctx.settings)["bond_A"]
    except Exception as exc:  # reference line only
        print(f"[qdprops]   report: MACE bulk reference failed ({exc})")
        mace = None
    return mace, experimental_bond(Path(ctx.cif).stem)


def _half_coverage_300(det):
    """dmu (eV) at which half of the units are removed at 300 K, or None."""
    mp = det["map"]
    if 300.0 not in mp["T"]:
        return None
    row = np.asarray(mp["k_mean"][mp["T"].index(300.0)], float)
    half = 0.5 * det["summary"]["n_steps"]
    mu = np.asarray(mp["dmu_eV"], float)
    j = np.where((row[:-1] - half) * (row[1:] - half) <= 0)[0]
    if not j.size:
        return None
    i = int(j[0])
    f = (half - row[i]) / (row[i + 1] - row[i]) if row[i + 1] != row[i] else 0.5
    return float(mu[i] + f * (mu[i + 1] - mu[i]))


def _log10_conc(dmu_eV: float, t: float = 300.0) -> float:
    return dmu_eV / (KB_MEV / 1000.0 * t * np.log(10.0))


# --------------------------------------------------------------------------
# Captions (one source; HTML and mathtext variants)
# --------------------------------------------------------------------------

def _captions(d) -> dict:
    """{panel: (title_html, title_tex, caption_html, caption_tex)}."""
    rel, hs = d["relax"], d["hess"]
    ma, mx = d.get("unit_ma", "MA"), d.get("unit_mx") or "MX"
    n, m = d.get("n_ma", 0), d.get("m_mx", 0)
    mah, mat = formula_html(ma), formula_tex(ma)
    mxh, mxt = formula_html(mx), formula_tex(mx)
    dot_h = f"({mah})<sub>{n}</sub>({mxh})<sub>{m}</sub>"
    dot_t = f"({mat})$_{{{n}}}$({mxt})$_{{{m}}}$"
    nvib = hs["n_modes"]
    c = {
        "relax": ("Relaxation", "Relaxation",
                  f"Energy along the MACE-MH-1 BFGS relaxation of the builder geometry, relative to the final "
                  f"minimum, per atom. Relaxation energy {rel['relaxation_energy_meV_atom']:.0f} meV/atom, RMSD "
                  f"{rel['rmsd_A']:.2f} Å, converged at f<sub>max</sub> = {rel['fmax_eV_A']:.3f} eV/Å.",
                  f"Energy along the MACE-MH-1 BFGS relaxation of the builder geometry, relative to the final "
                  f"minimum, per atom. Relaxation energy {rel['relaxation_energy_meV_atom']:.0f} meV/atom, RMSD "
                  f"{rel['rmsd_A']:.2f} Å, converged at $f_\\mathrm{{max}}$ = {rel['fmax_eV_A']:.3f} eV/Å."),
        "bonds": ("Bond lengths", "Bond lengths",
                  "Every bond, by site: core (both atoms with bulk coordination), surface (at least one "
                  "under-coordinated atom) and ligand. Filled circles: relaxed; grey diamonds: builder start "
                  "(bulk positions). Dashed: bulk bond d<sub>0</sub> of MACE-MH-1 (cell-relaxed bulk, same model); "
                  "dotted: experimental bulk bond.",
                  "Every bond, by site: core (both atoms with bulk coordination), surface (at least one "
                  "under-coordinated atom) and ligand. Filled circles: relaxed; grey diamonds: builder start "
                  "(bulk positions). Dashed: bulk $d_0$ of MACE-MH-1 (cell-relaxed bulk); dotted: experiment."),
        "cn": ("Coordination numbers", "Coordination numbers",
               "Atoms per coordination number: opposite-charge neighbours closer than 1.2 d<sub>0</sub> (native "
               "pairs) or 1.25 × the covalent-radius sum (ligands), at the start (outlines) and after relaxation "
               "(filled, coloured by element). Bulk CN = 4.",
               "Atoms per coordination number: opposite-charge neighbours closer than 1.2 $d_0$ (native pairs) "
               "or 1.25 × the covalent-radius sum (ligands), at the start (outlines) and after relaxation "
               "(filled, coloured by element). Bulk CN = 4."),
        "vdos_element": ("Vibrational density of states by element", "Vibrational DOS by element",
                         f"g(ω) = Σ<sub>k</sub> δ(ω − ω<sub>k</sub>) over the {nvib} harmonic modes, broadened by "
                         "a 5 cm<sup>−1</sup> Gaussian. Each element curve weights mode k by Σ<sub>i∈X</sub> "
                         "|e<sub>ik</sub>|², the share of its mass-weighted amplitude on element X; the curves add "
                         "up to the total.",
                         f"$g(\\omega) = \\sum_k \\delta(\\omega - \\omega_k)$ over the {nvib} harmonic modes, "
                         "broadened by a 5 cm$^{-1}$ Gaussian. Each element curve weights mode k by "
                         "$\\sum_{i\\in X} |e_{ik}|^2$, the share of its mass-weighted amplitude on element X; the "
                         "curves add up to the total."),
        "vdos_role": ("Vibrational density of states by site", "Vibrational DOS by site",
                      "The same modes projected on core, surface and ligand atoms. Soft surface and ligand modes "
                      "dominate the low-frequency range; the highest band shows which sites are stiffest.",
                      "The same modes projected on core, surface and ligand atoms. Soft surface and ligand modes "
                      "dominate the low-frequency range; the highest band shows which sites are stiffest."),
        "thermo_e": ("Vibrational energies", "Vibrational energies",
                     "ZPE = ½ Σ<sub>k</sub> ħω<sub>k</sub>;  U<sub>vib</sub>(T) = ZPE + Σ<sub>k</sub> ħω<sub>k</sub> / "
                     "(e<sup>ħω<sub>k</sub>/k<sub>B</sub>T</sup> − 1);  F<sub>vib</sub> = U<sub>vib</sub> − "
                     "T S<sub>vib</sub>. Both start at ZPE (dotted) at T → 0; their gap is the entropic term "
                     "T S<sub>vib</sub>, which lowers free energies as T rises.",
                     "ZPE $= \\frac{1}{2}\\sum_k \\hbar\\omega_k$;  $U_\\mathrm{vib}(T) = \\mathrm{ZPE} + "
                     "\\sum_k \\hbar\\omega_k/(e^{\\hbar\\omega_k/k_BT} - 1)$;  $F_\\mathrm{vib} = U_\\mathrm{vib} - "
                     "TS_\\mathrm{vib}$. Both start at ZPE (dotted) at T → 0; their gap is the entropic term "
                     "$TS_\\mathrm{vib}$, which lowers free energies as T rises."),
        "thermo_s": ("Vibrational entropy and heat capacity", "Vibrational entropy and heat capacity",
                     "With x<sub>k</sub> = ħω<sub>k</sub>/k<sub>B</sub>T: S<sub>vib</sub> = k<sub>B</sub> "
                     "Σ<sub>k</sub> [x<sub>k</sub>/(e<sup>x<sub>k</sub></sup> − 1) − ln(1 − e<sup>−x<sub>k</sub></sup>)] "
                     "and C<sub>v</sub> = k<sub>B</sub> Σ<sub>k</sub> x<sub>k</sub>² e<sup>x<sub>k</sub></sup> / "
                     "(e<sup>x<sub>k</sub></sup> − 1)². C<sub>v</sub> saturates at the classical "
                     f"(3N − 6) k<sub>B</sub> = {nvib * KB_MEV:.2f} meV/K (dotted) once k<sub>B</sub>T exceeds the "
                     "highest modes; S<sub>vib</sub> keeps growing, mostly from the soft modes.",
                     "With $x_k = \\hbar\\omega_k/k_BT$: $S_\\mathrm{vib} = k_B\\sum_k [x_k/(e^{x_k}-1) - "
                     "\\ln(1-e^{-x_k})]$ and $C_v = k_B\\sum_k x_k^2 e^{x_k}/(e^{x_k}-1)^2$. $C_v$ saturates at "
                     f"the classical $(3N-6)k_B$ = {nvib * KB_MEV:.2f} meV/K (dotted) once $k_BT$ exceeds the "
                     "highest modes; $S_\\mathrm{vib}$ keeps growing, mostly from the soft modes."),
    }
    if "decomposition_dG_eV" in d["stab"]:
        c["stability"] = (
            "Decomposition free energy", "Decomposition free energy",
            f"ΔG<sub>dec</sub> = G[{dot_h}] − {n} G[{mah}, bulk] − {m} G[{mxh}, 1 M], per {mah} unit (solid; "
            f"dotted: the electronic ΔE<sub>dec</sub>). It is the excess (surface) free energy of the dot against "
            f"bulk {mah} with the ligands in solution: positive means metastable against ripening into the bulk, "
            "and it decreases as the dot grows.",
            f"$\\Delta G_\\mathrm{{dec}}$ = G[{dot_t}] − {n} G[{mat}, bulk] − {m} G[{mxt}, 1 M], per {mat} unit "
            f"(solid; dotted: electronic $\\Delta E_\\mathrm{{dec}}$). The excess (surface) free energy against "
            f"bulk {mat} with the ligands in solution: positive means metastable against ripening; it decreases "
            "as the dot grows.")
        c["binding"] = (
            "Binding free energy", "Binding free energy",
            f"ΔG<sub>bind</sub> = (G[{dot_h}] − {n} G[{mah}, 1 M] − {m} G[{mxh}, 1 M]) / {n + m}, per unit "
            "(solid; dotted: electronic ΔE<sub>bind</sub>): the free energy of assembling the dot from monomers. "
            "More negative is more strongly bound; it rises with T because the free monomers gain translational "
            "and rotational entropy.",
            f"$\\Delta G_\\mathrm{{bind}}$ = (G[{dot_t}] − {n} G[{mat}, 1 M] − {m} G[{mxt}, 1 M]) / {n + m}, per "
            "unit (solid; dotted: electronic $\\Delta E_\\mathrm{bind}$): the free energy of assembling the dot "
            "from monomers. It rises with T because the free monomers gain translational and rotational entropy.")
    det = d["det"]
    if det.get("steps"):
        ds = det["summary"]
        mae = ds.get("model_mae_eV")
        fit_h = (f" Beam search (width {ds.get('beam', 2)}): units re-enumerated on the relaxed surface after "
                 f"every removal; units within {ds.get('local_radius_A', 7):.0f} Å of the last removal relaxed, "
                 f"the others carried over and relaxed before being chosen ({ds['n_relaxations']} MACE relaxations).")
        c["detach"] = (
            f"Stepwise {mxh} desorption", f"Stepwise {mxt} desorption",
            f"ΔE<sub>k</sub> = E(dot<sub>k</sub>) + E({mxh}) − E(dot<sub>k−1</sub>), where dot<sub>k</sub> is "
            f"the k-th structure of the lowest path found (bars, MACE-relaxed). ΔG<sub>k</sub>(300 K) adds "
            f"the vibrational change and the translational and rotational entropy of the released {mxh} at 1 M "
            f"(circles). Grey dots at k = 1: every symmetry-distinct surface site. Positive: bound.{fit_h}",
            f"$\\Delta E_k = E(\\mathrm{{dot}}_k) + E$({mxt}) $- E(\\mathrm{{dot}}_{{k-1}})$, "
            f"$\\mathrm{{dot}}_k$ the k-th structure of the lowest path found (bars). $\\Delta G_k$(300 K) "
            f"adds vibrations and the entropy of the released {mxt} at 1 M (circles). Grey dots at k = 1: every "
            f"distinct surface site. Positive: bound.{fit_h}")
        half = _half_coverage_300(det)
        half_h = (f" At 300 K half of the units are lost at Δμ = {half:.2f} eV, i.e. c ≈ "
                  f"10<sup>{_log10_conc(half):.0f}</sup> M." if half is not None else "")
        half_t = (f" At 300 K half are lost at Δμ = {half:.2f} eV, c ≈ $10^{{{_log10_conc(half):.0f}}}$ M."
                  if half is not None else "")
        c["map"] = (
            "Equilibrium ligand shell", "Equilibrium ligand shell",
            f"Mean number ⟨k⟩ of {mxh} units lost in equilibrium with {mxh} in solution at "
            f"μ = μ°(T) + Δμ, Δμ = k<sub>B</sub>T ln(c / 1 M): a Boltzmann average over all configurations "
            f"evaluated by the search, each weighted by its symmetry, so it includes configurational entropy. Right: concentrated "
            f"solution, shell intact; left: dilute, units desorb.{half_h}",
            f"Mean number ⟨k⟩ of {mxt} units lost in equilibrium with {mxt} in solution at μ = μ°(T) + Δμ, "
            f"Δμ = $k_BT\\ln(c/1\\,\\mathrm{{M}})$: Boltzmann average over every configuration evaluated by the "
            f"search, configurational entropy included.{half_t}")
    if d.get("vib"):
        c.update(vibplots.captions(d["vib"]))
    if (d.get("sites") or {}).get("classes"):
        c["sites"] = ("Binding-site map", f"{mxt} binding-site map (vacuum ΔE)", "",
                      f"Surface atoms by direction from the centre (azimuth θ, latitude φ, [001] at the poles); field: "
                      f"Gaussian-weighted average (σ = 7°) of the single-unit removal energy "
                      f"$\\Delta E_\\mathrm{{site}}$ = E(dot − unit) + E({mxt}) − E(dot), MACE-MH-1, vacuum. White: "
                      "facet edges. Black dots: atoms of a desorbable unit. The interactive page adds a 3D view and "
                      "the free energy in solution.")
    return c


def _key_numbers(d) -> list:
    """Rows (label_html, label_tex, value_html, value_tex, definition_html, definition_tex)."""
    rel, st, hs = d["relax"], d["structure"]["summary"], d["hess"]
    th = d["thermo"]
    i300 = th["T"].index(300.0) if 300.0 in th["T"] else None
    ma, mx = d.get("unit_ma", "MA"), d.get("unit_mx") or "MX"
    mah, mat, mxh, mxt = formula_html(ma), formula_tex(ma), formula_html(mx), formula_tex(mx)
    n, m = d.get("n_ma", 0), d.get("m_mx", 0)
    same = lambda t: (t, t)
    rows = [
        (*same("Relaxation energy"), *same(f"{rel['relaxation_energy_meV_atom']:.0f} meV/atom"),
         "[E<sub>relaxed</sub> − E<sub>start</sub>] / N, MACE-MH-1",
         r"$(E_\mathrm{relaxed} - E_\mathrm{start})/N$"),
        (*same("RMSD start → relaxed"), *same(f"{rel['rmsd_A']:.2f} Å"),
         "after optimal superposition (Kabsch)", "after optimal superposition"),
        (*same("Bond topology"), *same("preserved" if st["topology_preserved"] else
                                       f"{st['bonds_broken']} broken / {st['bonds_formed']} formed"),
         "relaxed vs start bond graph", "relaxed vs start bond graph"),
    ]
    if d.get("core_strain") is not None:
        ref_h = "MACE-MH-1 bulk d<sub>0</sub>" if d.get("bulk_mace") else "CIF bulk d<sub>0</sub>"
        rows.append((*same("Core strain"), *same(f"{d['core_strain']:+.1f} %"),
                     f"⟨d<sub>core</sub>⟩ / d<sub>0</sub> − 1, against the {ref_h}",
                     r"$\langle d_\mathrm{core}\rangle/d_0 - 1$"))
    rows += [
        (*same("Imaginary modes"), *same(f"{hs['n_imaginary']}"), "none at a true minimum", "none at a true minimum"),
        (*same("ZPE"), *same(f"{hs['zpe_eV']:.3f} eV"), "ZPE = ½ Σ<sub>k</sub> ħω<sub>k</sub>",
         r"$\frac{1}{2}\Sigma_k\hbar\omega_k$"),
    ]
    if i300 is not None:
        rows.append(("S<sub>vib</sub> (300 K)", "$S_\\mathrm{vib}$ (300 K)", *same(f"{th['S_vib_meV_K'][i300]:.2f} meV/K"),
                     "vibrational entropy (quasi-RRHO)", "vibrational entropy (quasi-RRHO)"))
        rows.append(("F<sub>vib</sub> (300 K)", "$F_\\mathrm{vib}$ (300 K)", *same(f"{th['F_vib_eV'][i300]:.3f} eV"),
                     "F<sub>vib</sub> = U<sub>vib</sub> − T S<sub>vib</sub>",
                     r"$F_\mathrm{vib} = U_\mathrm{vib} - TS_\mathrm{vib}$"))
    sb = d["stab"].get("summary", {})
    if "decomposition_dG_300K_per_MA_eV" in sb:
        dot_h = f"({mah})<sub>{n}</sub>({mxh})<sub>{m}</sub>"
        rows.append(("ΔG<sub>dec</sub> (300 K)", "$\\Delta G_\\mathrm{dec}$ (300 K)",
                     f"{sb['decomposition_dG_300K_per_MA_eV']:+.3f} eV per {mah}",
                     f"{sb['decomposition_dG_300K_per_MA_eV']:+.3f} eV per {mat}",
                     f"[G({dot_h}) − {n} G({mah}, bulk) − {m} G({mxh}, 1 M)] / {n}: against bulk {mah} and "
                     f"free {mxh}, vacuum",
                     f"[G(dot) − {n} G({mat}, bulk) − {m} G({mxt}, 1 M)] / {n}"))
        rows.append(("ΔG<sub>bind</sub> (300 K)", "$\\Delta G_\\mathrm{bind}$ (300 K)",
                     *same(f"{sb['binding_dG_300K_per_unit_eV']:+.3f} eV per unit"),
                     f"[G({dot_h}) − {n} G({mah}, 1 M) − {m} G({mxh}, 1 M)] / {n + m}: against the "
                     f"{mah} and {mxh} monomers, vacuum",
                     f"[G(dot) − {n} G({mat}, 1 M) − {m} G({mxt}, 1 M)] / {n + m}"))
    ds = d["det"].get("summary", {})
    if ds.get("dE_eV"):
        g = ds["dG_300K_eV"][0]
        val = f"ΔE {ds['dE_eV'][0]:+.2f} eV" + (f", ΔG {g:+.2f} eV" if g is not None else "")
        rows.append((f"First {mxh} detachment", f"First {mxt} detachment", *same(val),
                     f"ΔE<sub>1</sub> = E(dot<sub>1</sub>) + E({mxh}) − E(dot<sub>0</sub>), cheapest unit; "
                     f"ΔG at 300 K, {mxh} at 1 M, vacuum",
                     r"$\Delta E_1$, cheapest unit; $\Delta G$ at 300 K, 1 M"))
    if d.get("vib"):
        rows += vibplots.key_rows(d["vib"])
    return rows


# --------------------------------------------------------------------------
# Plotly (HTML)
# --------------------------------------------------------------------------

def _bulk_lines(d, html_):
    """Reference lines of the bond plot: (x, label, dash, colour, y offset)."""
    out = []
    sub = (lambda t: f"<sub>{t}</sub>") if html_ else (lambda t: f"$_{{{t}}}$")
    if d.get("bulk_mace"):
        out.append((d["bulk_mace"], f"bulk d{sub('0')}, MACE-MH-1 = {d['bulk_mace']:.3f} Å", "dash", INK, 0.6))
    elif d.get("bulk_bond_cif"):
        out.append((d["bulk_bond_cif"], f"bulk d{sub('0')}, CIF = {d['bulk_bond_cif']:.3f} Å", "dash", INK, 0.6))
    if d.get("bulk_exp"):
        out.append((d["bulk_exp"][0], f"experiment = {d['bulk_exp'][0]:.3f} Å", "dot", INK2, 0.0))
    return out


def _layout(xt, yt, **kw):
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=INK3, ticks="outside", tickcolor=INK3,
                tickfont=dict(color=INK2))
    lay = dict(template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
               font=dict(family="Helvetica, Arial, sans-serif", color=INK2, size=12),
               margin=dict(l=64, r=16, t=36, b=52), height=330,
               xaxis=dict(title=dict(text=xt), **axis), yaxis=dict(title=dict(text=yt), **axis),
               legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, xanchor="left", font=dict(size=11)),
               hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=INK, size=12)),
               hovermode="closest")
    for k, v in kw.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict):
            lay[k].update(v)
        else:
            lay[k] = v
    return lay


def _plotly_figures(d) -> dict:
    import plotly.graph_objects as go
    figs = {}
    n = d["n"]

    tr = d["trace"]
    if tr.size:
        y = 1000 * (tr - tr[-1]) / n
        fig = go.Figure(go.Scatter(x=np.arange(1, tr.size + 1), y=y, mode="lines+markers",
                                   line=dict(color=SITE["core"], width=2), marker=dict(size=5),
                                   hovertemplate="BFGS step %{x}<br>E − E<sub>min</sub> = %{y:.1f} meV/atom<extra></extra>"))
        fig.update_layout(**_layout("BFGS step", "E − E<sub>min</sub> (meV/atom)", showlegend=False))
        figs["relax"] = fig

    fig = go.Figure()
    labels = []
    shown = set()
    for yi, ((cat, label), v) in enumerate(d["bond_rows"]):
        labels.append(label)
        rng = np.random.default_rng(yi)
        if v["start"]:
            fig.add_scatter(x=v["start"], y=yi + 0.2 + rng.uniform(-0.05, 0.05, len(v["start"])), mode="markers",
                            name="start (builder)", legendgroup="start", showlegend="start" not in shown,
                            marker=dict(symbol="diamond-open", size=8, color=START, line=dict(width=1.5)),
                            hovertemplate=f"{label}, start<br>%{{x:.3f}} Å<extra></extra>")
            shown.add("start")
        fig.add_scatter(x=v["relaxed"], y=yi - 0.12 + rng.uniform(-0.08, 0.08, len(v["relaxed"])), mode="markers",
                        name=f"{cat}, relaxed", legendgroup=cat, showlegend=cat not in shown, text=v["pairs"],
                        marker=dict(size=9, color=SITE[cat], line=dict(color=SURFACE, width=1.5)),
                        hovertemplate=f"{label}, relaxed<br>%{{text}}: %{{x:.3f}} Å<extra></extra>")
        shown.add(cat)
    fig.update_layout(**_layout("bond length (Å)", "", yaxis=dict(tickvals=list(range(len(labels))), ticktext=labels,
                                                                   range=[-0.6, len(labels) + 0.4])))
    for x0, text, dash, col, y0 in _bulk_lines(d, html_=True):
        fig.add_vline(x=x0, line=dict(color=col, width=1.5, dash=dash))
        fig.add_annotation(x=x0, y=len(labels) - 0.45 + y0, text=text, showarrow=False, xanchor="left", xshift=6,
                           font=dict(color=col, size=11))
    figs["bonds"] = fig

    cn = d["structure"]["cn_hist"]
    fig = go.Figure()
    xs, rel_y, st_y, cols = [], [], [], []
    for e in d["elements"]:
        rel = {int(k): v for k, v in cn["relaxed"].get(e, {}).items()}
        st = {int(k): v for k, v in cn["start"].get(e, {}).items()}
        for k in sorted(set(rel) | set(st)):
            xs.append(f"{e} · CN {k}")
            rel_y.append(rel.get(k, 0))
            st_y.append(st.get(k, 0))
            cols.append(d["ecol"][e])
    fig.add_bar(x=xs, y=st_y, name="start", marker=dict(color="rgba(0,0,0,0)", line=dict(color=START, width=1.5)),
                hovertemplate="%{x}<br>start: %{y} atoms<extra></extra>")
    fig.add_bar(x=xs, y=rel_y, name="relaxed (colour = element)", marker=dict(color=cols), text=rel_y,
                textposition="outside", textfont=dict(color=INK2), cliponaxis=False,
                hovertemplate="%{x}<br>relaxed: %{y} atoms<extra></extra>")
    fig.update_layout(**_layout("", "atoms", barmode="group", bargap=0.35, bargroupgap=0.08))
    figs["cn"] = fig

    dos = d["vdos"]
    for kind, names, cmap in (("element", d["elements"], d["ecol"]),
                              ("role", [x for x in ("core", "surface", "ligand") if x in dos["projected"]["role"]], SITE)):
        fig = go.Figure()
        fig.add_scatter(x=dos["cm1"], y=dos["total"], mode="lines", name="total", line=dict(color=INK3, width=1.5),
                        hovertemplate="%{y:.3f}<extra>total</extra>")
        for nme in names:
            fig.add_scatter(x=dos["cm1"], y=dos["projected"][kind][nme], mode="lines", name=nme,
                            line=dict(color=cmap[nme], width=2), hovertemplate=f"%{{y:.3f}}<extra>{nme}</extra>")
        fig.update_layout(**_layout("wavenumber (cm<sup>−1</sup>)", "states per cm<sup>−1</sup>", hovermode="x unified",
                                    xaxis=dict(unifiedhovertitle=dict(text="%{x:.0f} cm<sup>−1</sup>"))))
        figs[f"vdos_{kind}"] = fig

    th = d["thermo"]
    T = np.asarray(th["T"])
    U, F = np.asarray(th["U_vib_eV"]), np.asarray(th["F_vib_eV"])
    TS = T * np.asarray(th["S_vib_meV_K"]) / 1000.0
    zpe = th["zpe_eV"]
    fig = go.Figure()
    for y, name, col, tip in ((U, "U<sub>vib</sub>", SITE["core"], "ZPE + thermally excited vibrations"),
                              (F, "F<sub>vib</sub>", SITE["surface"], "U<sub>vib</sub> − T S<sub>vib</sub>"),
                              (TS, "T S<sub>vib</sub>", SITE["ligand"], "entropic term")):
        fig.add_scatter(x=T, y=y, mode="lines", name=name, line=dict(color=col, width=2.5),
                        hovertemplate=f"{name} = %{{y:.3f}} eV  <i>({tip})</i><extra></extra>")
    fig.add_hline(y=zpe, line=dict(color=INK2, width=1, dash="dot"))
    fig.add_annotation(x=T[0] + 0.5 * (T[-1] - T[0]), y=zpe, text=f"ZPE = {zpe:.3f} eV", showarrow=False,
                       xanchor="center", yanchor="top", yshift=-3, bgcolor=SURFACE, font=dict(color=INK2, size=11))
    fig.update_layout(**_layout("T (K)", "energy (eV)", hovermode="x unified", xaxis=dict(unifiedhovertitle=dict(text="T = %{x:.0f} K"))))
    figs["thermo_e"] = fig

    fig = go.Figure()
    cl = d["hess"]["n_modes"] * KB_MEV
    for key, name, col, tip in (("S_vib_meV_K", "S<sub>vib</sub>", SITE["core"], "vibrational entropy"),
                                ("Cv_meV_K", "C<sub>v</sub>", SITE["surface"], "heat capacity")):
        fig.add_scatter(x=T, y=th[key], mode="lines", name=name, line=dict(color=col, width=2.5),
                        hovertemplate=f"{name} = %{{y:.2f}} meV/K  <i>({tip})</i><extra></extra>")
    fig.add_hline(y=cl, line=dict(color=INK2, width=1, dash="dot"))
    fig.add_annotation(x=T[0] + 0.62 * (T[-1] - T[0]), y=cl, text="classical limit (3N−6) k<sub>B</sub>",
                       showarrow=False, xanchor="center", yanchor="bottom", yshift=3, font=dict(color=INK2, size=11))
    fig.update_layout(**_layout("T (K)", "meV/K", hovermode="x unified", xaxis=dict(unifiedhovertitle=dict(text="T = %{x:.0f} K"))))
    figs["thermo_s"] = fig

    stab = d["stab"]
    if "decomposition_dG_eV" in stab:
        Ts = np.asarray(stab["temperatures"])
        sb = stab["summary"]
        ma = formula_html(d["unit_ma"])
        for key, y, de, name, ylab in (
                ("stability", np.asarray(stab["decomposition_dG_eV"]) / d["n_ma"], sb["decomposition_dE_per_MA_eV"],
                 "ΔG<sub>dec</sub>", f"eV per {ma}"),
                ("binding", np.asarray(stab["binding_dG_per_unit_eV"]), sb["binding_dE_per_unit_eV"],
                 "ΔG<sub>bind</sub>", "eV per unit")):
            fig = go.Figure()
            fig.add_scatter(x=Ts, y=y, mode="lines", name=name, line=dict(color=SITE["core"], width=2.5),
                            hovertemplate=f"{name} = %{{y:.3f}} {ylab}<extra></extra>")
            fig.add_hline(y=de, line=dict(color=INK2, width=1, dash="dot"))
            fig.add_annotation(x=Ts[-1], y=de, text=f"ΔE = {de:+.3f}", showarrow=False, xanchor="right",
                               yanchor="bottom", font=dict(color=INK2, size=11))
            fig.update_layout(**_layout("T (K)", ylab, showlegend=False, hovermode="x unified", xaxis=dict(unifiedhovertitle=dict(text="T = %{x:.0f} K"))))
            figs[key] = fig

    det = d["det"]
    if det.get("steps"):
        mx = formula_html(d["unit_mx"])
        steps = det["steps"]
        ks = [s["k"] for s in steps]
        fig = go.Figure()
        fig.add_bar(x=ks, y=[s["dE_eV"] for s in steps], name="ΔE<sub>k</sub> (lowest configuration)", width=0.45,
                    marker_color=SITE["core"],
                    customdata=[[s["n_candidates"], s["n_exact"], "molecular" if s["molecular"] else "CdCl + closest Cl"]
                                for s in steps],
                    hovertemplate="k = %{x}<br>ΔE = %{y:.3f} eV (MACE)<br>%{customdata[2]} unit; "
                                  "%{customdata[0]} candidates, %{customdata[1]} relaxed<extra></extra>")
        if det["summary"]["thermo"]:
            fig.add_scatter(x=ks, y=[s["dG_300K_eV"] for s in steps], mode="markers", name="ΔG<sub>k</sub>(300 K)",
                            marker=dict(size=13, color=SITE["surface"], line=dict(color=SURFACE, width=2)),
                            hovertemplate="k = %{x}<br>ΔG(300 K) = %{y:.3f} eV<extra></extra>")
        sites = det.get("site_classes", [])
        fig.add_scatter(x=[1.32] * len(sites), y=[c["dE_eV"] for c in sites], mode="markers",
                        name="all distinct sites (k = 1)",
                        text=[f"{c.get('site', 'site')} {' / '.join(c.get('facets', []))}: "
                              f"{c['multiplicity']} equivalent; M native CN {c['cation_native_cn']}, "
                              f"{c['cation_ligands']} own ligand(s), ligands bound to {c['ligand_mu']} cations"
                              for c in sites],
                        marker=dict(size=8, color=INK3, line=dict(color=SURFACE, width=1)),
                        hovertemplate="%{text}<br>ΔE = %{y:.3f} eV<extra></extra>")
        fig.add_hline(y=0, line=dict(color=INK2, width=1))
        fig.update_layout(**_layout("units removed k", "eV", xaxis=dict(dtick=1)))
        figs["detach"] = fig

        mp = det["map"]
        z = np.asarray(mp["k_mean"], float)
        seq = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"]]
        fig = go.Figure(go.Heatmap(x=mp["dmu_eV"], y=mp["T"], z=z, zmin=0, zmax=det["summary"]["n_steps"],
                                   colorscale=seq, colorbar=dict(title=dict(text=f"⟨k⟩ {mx}<br>removed")),
                                   hovertemplate=f"T = %{{y:.0f}} K, Δμ = %{{x:.2f}} eV<br>⟨k⟩ = %{{z:.2f}} {mx} "
                                                 "removed<extra></extra>"))
        fig.update_layout(**_layout(f"Δμ({mx}) = k<sub>B</sub>T ln(c / 1 M)  (eV)", "T (K)"))
        figs["map"] = fig
    if d.get("vib"):
        figs.update(vibplots.plotly_figs(d["vib"], SITE, _layout, INK, INK3, d["elements"]))
    return figs


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title_text}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-{plotly_js}.min.js"></script>
<style>
:root {{ --ink: {ink}; --ink2: {ink2}; --grid: {grid}; --surface: {surface}; }}
* {{ box-sizing: border-box; }}
body {{ font-family: Helvetica, Arial, sans-serif; background: #f8f9fa; margin: 0; padding: 20px; color: var(--ink); }}
.wrap {{ max-width: 1500px; margin: 0 auto; background: var(--surface); padding: 24px 28px; border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,.05); }}
h1 {{ font-size: 21px; margin: 0 0 4px; }}
h2 {{ font-size: 16px; margin: 30px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--grid); }}
.sub {{ color: var(--ink2); font-size: 13px; margin-bottom: 16px; }}
table.keys {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
table.keys td {{ padding: 6px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }}
table.keys td.v {{ font-weight: 700; white-space: nowrap; }}
table.keys td.d {{ color: var(--ink2); }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(460px, 100%), 1fr)); gap: 16px; }}
.panel {{ border: 1px solid var(--grid); border-radius: 8px; padding: 12px 12px 10px; }}
.panel h3 {{ font-size: 14px; margin: 0 0 4px; color: var(--ink); }}
.panel .cap {{ font-size: 12px; color: var(--ink2); line-height: 1.55; margin-top: 6px; }}
.note {{ font-size: 12px; color: var(--ink2); margin-top: 8px; line-height: 1.5; }}
{ctl_css}
@media (max-width: 520px) {{ body {{ padding: 8px; }} .wrap {{ padding: 14px 14px; }} }}
</style></head><body><div class="wrap">
<h1>{title}</h1><div class="sub">{subtitle}</div>
<table class="keys">{keys}</table>
{sections}
{extra_html}
</div>
<script>
{extra_js}
const FIGS = {figs};
for (const [id, f] of Object.entries(FIGS)) {{
  Plotly.newPlot(id, f.data, f.layout, {{responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"]}});
}}
</script></body></html>
"""

SECTIONS = [
    ("Structure &amp; relaxation", ["relax", "bonds", "cn"], ""),
    ("Vibrations &amp; thermochemistry",
     ["vdos_element", "vdos_role", "ir", "raman", "modemap", "thermo_e", "thermo_s"],
     "Harmonic normal modes of the analytic MACE-MH-1 Hessian at the relaxed geometry, with translations and "
     "rotations projected out. IR and Raman intensities: g-xTB dipole and polarisability derivatives along those "
     "modes (finite differences, static finite field). Hover a curve or a peak for its value and definition."),
    ("Stability &amp; ligands in vacuum", ["stability", "binding", "detach"],
     "References computed with the same MACE head: bulk {ma} (cell-relaxed, phonons) and the {ma} and {mx} "
     "monomers as ideal gases at a 1 M standard state (harmonic vibrations, rigid-rotor rotation; the dot is treated "
     "the same way). No solvent: everything here is in vacuum; the next section adds solvation and finite "
     "concentrations, and the equilibrium ligand shell."),
]


def _write_html(ctx, d, figs, caps, keys, solution=None) -> None:
    import plotly
    from plotly.offline import get_plotlyjs_version
    body = []
    ma = formula_html(d.get("unit_ma", "MA"))
    mx = formula_html(d.get("unit_mx") or "MX")
    for title, ids, note in SECTIONS:
        ids = [i for i in ids if i in figs]
        if not ids:
            continue
        panels = "".join(f"<div class='panel'><h3>{caps[i][0]}</h3><div id='{i}'></div>"
                         f"<div class='cap'>{caps[i][2]}</div></div>" for i in ids)
        body.append(f"<h2>{title}</h2><div class='grid'>{panels}</div>"
                    + (f"<div class='note'>{note.format(ma=ma, mx=mx)}</div>" if note else ""))
    shown = {i for _t, ids, _n in SECTIONS for i in ids}
    payload = {k: json.loads(plotly.io.to_json(f)) for k, f in figs.items() if k in shown}
    rec = d["rec"]
    head = ctx.results["relax"].get("provenance", {}).get("head", "")
    keys_html = "".join(f"<tr><td>{k[0]}</td><td class='v'>{k[2]}</td><td class='d'>{k[4]}</td></tr>" for k in keys)
    page = PAGE.format(
        title_text=html.escape(f"{rec.get('formula', '')} ground state"),
        title=f"{formula_html(rec.get('formula', ''))} · {html.escape(rec.get('material', ''))} "
              f"{html.escape(rec.get('phase', ''))}",
        subtitle=html.escape(f"{rec.get('id', '')} — ground state from MACE-MH-1 ({head})"),
        plotly_js=get_plotlyjs_version(), ink=INK, ink2=INK2, grid=GRID, surface=SURFACE,
        keys=keys_html, sections="".join(body), figs=json.dumps(payload),
        extra_html=solution[0] if solution else "", extra_js=solution[1] if solution else "",
        ctl_css=CONTROLS_CSS)
    (ctx.props / "ground_state.html").write_text(page)


# --------------------------------------------------------------------------
# Matplotlib (PNG)
# --------------------------------------------------------------------------

def _sites_png(ax, sd, fig):
    """2D (theta, phi) map of the vacuum site energies with the facet edges (mirrors the dashboard)."""
    from matplotlib.colors import LinearSegmentedColormap
    sym, pos, role = sd["dot"]["symbols"], np.asarray(sd["dot"]["positions"]), sd["dot"]["role"]
    n = len(sym)
    val = [None] * n
    for c in sd["classes"]:
        for unit in c["orbit"]:
            for a in unit:
                val[a] = c["dE_eV"] if val[a] is None else min(val[a], c["dE_eV"])
    dirs = pos / np.linalg.norm(pos, axis=1)[:, None]
    ang = lambda d: (np.arctan2(d[..., 1], d[..., 0]), np.arcsin(np.clip(d[..., 2], -1, 1)))
    surf = [a for a in range(n) if role[a] != "core"]
    on = [a for a in surf if val[a] is not None]
    gt, gp = np.linspace(-np.pi, np.pi, 361), np.linspace(-np.pi / 2, np.pi / 2, 181)
    T, P = np.meshgrid(gt, gp)
    g = np.stack([np.cos(P) * np.cos(T), np.cos(P) * np.sin(T), np.sin(P)], -1)
    c = np.einsum("ijk,ak->ija", g, dirs[on])
    th = np.arccos(np.clip(c, -1, 1))
    w = np.exp(-0.5 * (th / 0.13) ** 2)
    z = (w * np.array([val[a] for a in on])).sum(-1) / w.sum(-1)
    z[c.max(-1) < np.cos(0.32)] = np.nan
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    if hi - lo < 0.1:                                     # (nearly) equal sites: centre them, no rounding noise
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.05, mid + 0.05
    ax.set_facecolor("#efeeea")                           # bare surface (no desorbable unit nearby)
    im = ax.pcolormesh(gt, gp, z, cmap=cmap, vmin=lo, vmax=hi, shading="gouraud", rasterized=True)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(r"$\Delta E_\mathrm{site}$ (eV)", color=INK2)
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(colors=INK2, labelsize=8.5)
    for f in sd["faces"]:
        V = np.asarray(f["vertices"])
        for e in range(len(V)):
            seg = V[e] + (V[(e + 1) % len(V)] - V[e]) * np.linspace(0, 1, 41)[:, None]
            t, p = ang(seg / np.linalg.norm(seg, axis=1)[:, None])
            cut = np.where(np.abs(np.diff(t)) > np.pi)[0] + 1
            for tt, pp in zip(np.split(t, cut), np.split(p, cut)):
                ax.plot(tt, pp, color="white", lw=1.4)
        t, p = ang(np.asarray(f["normal"]))
        i, j = int(round((t + np.pi) / (2 * np.pi) * 360)), int(round((p + np.pi / 2) / np.pi * 180))
        from matplotlib import patheffects
        ax.text(t, p, f["label"].strip("{}"), color=INK, fontsize=8.5, ha="center", va="center", fontweight="bold",
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])
    t, p = ang(dirs[surf])
    ax.scatter(t, p, s=[10 if sym[a] == "Cl" else 14 for a in surf],
               c=["#0b0b0b" if val[a] is not None else INK3 for a in surf],
               marker="o", edgecolors="white", linewidths=0.5, zorder=4)
    ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi / 2, np.pi / 2), xlabel="θ (azimuth, rad)", ylabel="φ (latitude, rad)")
    ax.grid(False)


def _cells(d):
    """PNG panel order (letters follow it): structure, vibrations and spectra, vacuum stability, solution, keys."""
    c = ["relax", "bonds", "cn", "vdos_element", "vdos_role"]
    if d.get("vib"):
        c += ["ir", "raman", "modemap"]
    c += ["thermo_e", "thermo_s", "stability", "binding", "detach"]
    if (d.get("sites") or {}).get("classes"):
        c += ["sites"]
    if d.get("solution"):
        c += ["sol_dec", "sol_bind", "sol_iso", "sol_map"]
    return c + ["keys"]


def _style_axes(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK3)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _legend_above(ax, ncol=4, handles=None, labels=None):
    kw = dict(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=ncol, frameon=False, fontsize=8.5,
              handlelength=1.6, columnspacing=1.2, borderaxespad=0.25)
    if handles is not None:
        return ax.legend(handles, labels, **kw)
    return ax.legend(**kw)


SOL_EPS = [None, 2.4, 80.0]          # gas, toluene-like, water-like
SOL_C = 1e-2                          # M, CdCl2 (and CdSe monomer) for panels m and o


def _solution_captions(d) -> dict:
    ex = d["solution"]
    ma, mx = ex["units"]["MA"], ex["units"]["MX"]
    mat, mxt = formula_tex(ma), formula_tex(mx)
    n, m = ex["units"]["n"], ex["units"]["m"]
    gb = "Generalized Born solvation of the xTB charges, $(1 - 1/\\varepsilon)\\,G_\\mathrm{GB}$"
    return {
        "sol_dec": ("", "Decomposition in solution",
                    "", f"$\\Delta G_\\mathrm{{dec}}$ = [$G^\\mathrm{{sol}}$(dot) − {n} G({mat}, bulk) − {m} μ({mxt})] / {n} "
                        f"with [{mxt}] = 10 mM, for the gas phase and two dielectric constants ({gb}). Solvation stabilises "
                        f"the dissolved {mxt} and the dot differently, shifting the excess free energy."),
        "sol_bind": ("", "Binding and dissolution temperature",
                     "", f"$\\Delta G_\\mathrm{{bind}}$ = [$G^\\mathrm{{sol}}$(dot) − {n} μ({mat}) − {m} μ({mxt})] / {n + m} at "
                         f"$\\varepsilon$ = 2.4 and equal monomer concentrations c. Markers: $T_\\mathrm{{diss}}$, where "
                         "$\\Delta G_\\mathrm{bind}$ = 0 and the dot dissolves into monomers; dilution lowers it through "
                         "$k_BT\\ln c$. Interactive version with sliders in ground_state.html."),
        "sol_iso": ("", f"{mxt} desorption isotherm",
                    "", f"Equilibrium number of {mxt} units lost, ⟨k⟩, against the {mxt} concentration in solution at "
                        f"$\\varepsilon$ = 2.4 (Boltzmann average over the evaluated configurations, each with its own "
                        "solvation and vibrational free energy along the removal path)."),
        "sol_map": ("", "Equilibrium ligand shell in solution",
                    "", f"⟨k⟩, the mean number of {mxt} units lost, against [{mxt}] and T at $\\varepsilon$ = 2.4: a "
                        "Boltzmann average over every configuration evaluated by the desorption search (symmetry-"
                        "weighted, so configurational entropy is included). Right: concentrated solution, shell intact; "
                        "left: dilute, units desorb."),
    }


def _solution_panels(axes, d) -> None:
    from matplotlib.lines import Line2D
    from ..solution import curves, mean_removed
    ex = d["solution"]
    T = np.asarray(ex["T"])
    mxt = formula_tex(ex["units"]["MX"])
    ax = axes["sol_dec"][0]
    for eps, col in zip(SOL_EPS, (INK3, SITE["core"], SITE["surface"])):
        c = curves(ex, eps, SOL_C, SOL_C)
        ax.plot(T, c["dec"], color=col, lw=2.2 if eps else 1.6, ls="-" if eps else ":",
                label="gas" if eps is None else f"$\\varepsilon$ = {eps:g}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(xlabel="T (K)", ylabel=f"eV per {formula_tex(ex['units']['MA'])}")
    _legend_above(ax, ncol=3)
    ax = axes["sol_bind"][0]
    for cval, col in zip((1e-2, 1e-4, 1e-6), (SITE["core"], SITE["surface"], SITE["ligand"])):
        c = curves(ex, 2.4, cval, cval)
        ax.plot(T, c["bind"], color=col, lw=2.2, label=f"c = $10^{{{int(np.log10(cval))}}}$ M")
        if c["t_diss"]:
            ax.plot([c["t_diss"]], [0], "o", ms=8, color=col, mec="white", mew=1.5, zorder=5)
            ax.annotate(f"{c['t_diss']:.0f} K", (c["t_diss"], 0), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(xlabel="T (K)", ylabel="eV per unit")
    _legend_above(ax, ncol=3)
    ax = axes["sol_iso"][0]
    lc = np.arange(-20, 0.01, 0.25)
    for tsel, col in ((300.0, SITE["core"]), (500.0, SITE["surface"])):
        ti = list(T).index(tsel)
        kk = [mean_removed(ex, 2.4, 10 ** v)[ti] for v in lc]
        ax.plot(lc, kk, color=col, lw=2.2, label=f"{tsel:.0f} K")
    ax.set(xlabel=f"log$_{{10}}$([{mxt}] / M)", ylabel=f"⟨k⟩ {mxt} removed", ylim=(-0.3, ex["units"]["m"] + 0.3))
    _legend_above(ax, ncol=2)
    if "sol_map" in axes:
        from matplotlib.colors import LinearSegmentedColormap
        ax = axes["sol_map"][0]
        lc = np.arange(-20, 0.01, 0.25)
        z = np.array([mean_removed(ex, 2.4, 10 ** v) for v in lc]).T          # (T, c)
        kmax = max(d_["k"] for d_ in ex["dots"])
        cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
        im = ax.pcolormesh(lc, T, z, cmap=cmap, vmin=0, vmax=kmax, shading="nearest", rasterized=True)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(rf"$\langle k\rangle$ {mxt} removed", color=INK2)
        cb.outline.set_edgecolor(GRID)
        cb.ax.tick_params(colors=INK2, labelsize=8.5)
        if kmax > 1:
            ax.contour(lc, T, z, levels=np.arange(0.5, kmax, 1.0), colors="white", linewidths=0.6, alpha=0.8)
        ax.set(xlabel=f"log$_{{10}}$([{mxt}] / M)", ylabel="T (K)", ylim=(T[0], 900))
        ax.grid(False)


def _write_png(ctx, d, caps, keys) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 9,
                         "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
                         "axes.labelsize": 9})
    sol = d.get("solution")
    vib = d.get("vib")
    nrow = -(-len(_cells(d)) // 3)
    fig = plt.figure(figsize=(17, 6 * nrow), facecolor="white")
    gs = GridSpec(2 * nrow, 3, figure=fig, height_ratios=[3, 1.1] * nrow, hspace=0.32, wspace=0.27,
                  left=0.05, right=0.975, top=1 - 0.065 * 4 / nrow, bottom=0.012)
    rec = d["rec"]
    fig.text(0.05, 1 - 0.36 / fig.get_figheight(), f"{formula_tex(rec.get('formula', ''))} · {rec.get('material', '')} {rec.get('phase', '')}",
             fontsize=16, fontweight="bold", color=INK, va="top")
    head = ctx.results["relax"].get("provenance", {}).get("head", "")
    fig.text(0.05, 1 - 0.70 / fig.get_figheight(), f"{rec.get('id', '')} — ground state from MACE-MH-1 ({head})", fontsize=10, color=INK2,
             va="top")

    cells = _cells(d)
    letters = "abcdefghijklmnopqrstuvwx"
    if sol:
        caps = dict(caps)
        caps.update(_solution_captions(d))
    axes = {}
    for idx, key in enumerate(cells):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[2 * row, col])
        cap_ax = fig.add_subplot(gs[2 * row + 1, col])
        cap_ax.axis("off")
        axes[key] = (ax, cap_ax)
        if key in caps:
            ax.set_title(f"{letters[idx]}   {caps[key][1]}", loc="left", pad=26)
            cap_ax.text(0, 1.0, _wrap(caps[key][3].replace("\\sum", "\\Sigma"), 80), va="top", ha="left", fontsize=8.3, color=INK2,
                        linespacing=1.5, transform=cap_ax.transAxes)
            # (mathtext \sum stacks its limits; captions use an inline \Sigma)
        elif key != "keys":
            ax.set_visible(False)

    n = d["n"]
    # a  relaxation
    ax = axes["relax"][0]
    tr = d["trace"]
    if tr.size:
        ax.plot(np.arange(1, tr.size + 1), 1000 * (tr - tr[-1]) / n, color=SITE["core"], lw=2, marker="o", ms=3)
    ax.set(xlabel="BFGS step", ylabel=r"$E - E_\mathrm{min}$ (meV/atom)")

    # b  bond strip plot
    ax = axes["bonds"][0]
    labels = []
    for yi, ((cat, label), v) in enumerate(d["bond_rows"]):
        labels.append(label)
        rng = np.random.default_rng(yi)
        if v["start"]:
            ax.scatter(v["start"], yi + 0.2 + rng.uniform(-0.05, 0.05, len(v["start"])), marker="D", s=26,
                       facecolors="none", edgecolors=START, linewidths=1.1, zorder=3)
        ax.scatter(v["relaxed"], yi - 0.12 + rng.uniform(-0.08, 0.08, len(v["relaxed"])), s=34, color=SITE[cat],
                   edgecolors="white", linewidths=0.8, zorder=4)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_ylim(-0.7, len(labels) + 0.5)
    for x0, text, dash, col, y0 in _bulk_lines(d, html_=False):
        ax.axvline(x0, color=col, lw=1.2, ls="--" if dash == "dash" else ":", zorder=2)
        ax.text(x0, len(labels) - 0.35 + y0, f"  {text}", color=col, fontsize=8.5, va="top", ha="left")
    ax.set_xlabel("bond length (Å)")
    _legend_above(ax, ncol=4, handles=[
        Line2D([], [], marker="o", ls="", color=SITE["core"]),
        Line2D([], [], marker="o", ls="", color=SITE["surface"]),
        Line2D([], [], marker="o", ls="", color=SITE["ligand"]),
        Line2D([], [], marker="D", ls="", mfc="none", mec=START)],
        labels=["core", "surface", "ligand", "start (builder)"])

    # c  coordination numbers
    ax = axes["cn"][0]
    cn = d["structure"]["cn_hist"]
    xl, rel_y, st_y, cols = [], [], [], []
    for e in d["elements"]:
        rel = {int(k): v for k, v in cn["relaxed"].get(e, {}).items()}
        st = {int(k): v for k, v in cn["start"].get(e, {}).items()}
        for k in sorted(set(rel) | set(st)):
            xl.append(f"{e}\nCN {k}")
            rel_y.append(rel.get(k, 0))
            st_y.append(st.get(k, 0))
            cols.append(d["ecol"][e])
    x = np.arange(len(xl))
    ax.bar(x - 0.19, st_y, width=0.36, facecolor="none", edgecolor=START, lw=1.2)
    bars = ax.bar(x + 0.19, rel_y, width=0.36, color=cols)
    for b_, v in zip(bars, rel_y):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.15, str(v), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, xl)
    ax.set_ylabel("atoms")
    ax.set_ylim(0, max(rel_y + st_y) * 1.18 + 0.5)
    _legend_above(ax, ncol=1 + len(d["elements"]),
                  handles=[Rectangle((0, 0), 1, 1, fc="none", ec=START, lw=1.2)]
                  + [Rectangle((0, 0), 1, 1, fc=d["ecol"][e]) for e in d["elements"]],
                  labels=["start"] + [f"{e}, relaxed" for e in d["elements"]])

    # d, e  VDOS
    dos = d["vdos"]
    for key, kind, names, cmap in (("vdos_element", "element", d["elements"], d["ecol"]),
                                   ("vdos_role", "role", [r for r in ("core", "surface", "ligand")
                                                          if r in dos["projected"]["role"]], SITE)):
        ax = axes[key][0]
        ax.plot(dos["cm1"], dos["total"], color=INK3, lw=1.3, label="total")
        for nme in names:
            ax.plot(dos["cm1"], dos["projected"][kind][nme], color=cmap[nme], lw=1.9, label=nme)
        ax.set(xlabel=r"wavenumber (cm$^{-1}$)", ylabel=r"states per cm$^{-1}$")
        ax.set_xlim(0, max(dos["cm1"]))
        ax.set_ylim(0, None)
        _legend_above(ax, ncol=4)

    # f  vibrational energies
    th = d["thermo"]
    T = np.asarray(th["T"])
    ax = axes["thermo_e"][0]
    U, F = np.asarray(th["U_vib_eV"]), np.asarray(th["F_vib_eV"])
    TS = T * np.asarray(th["S_vib_meV_K"]) / 1000.0
    ax.plot(T, U, color=SITE["core"], lw=2.2, label=r"$U_\mathrm{vib}$")
    ax.plot(T, F, color=SITE["surface"], lw=2.2, label=r"$F_\mathrm{vib}$")
    ax.plot(T, TS, color=SITE["ligand"], lw=2.2, label=r"$TS_\mathrm{vib}$")
    ax.axhline(th["zpe_eV"], color=INK2, lw=1, ls=":")
    ax.text(T[0] + 0.5 * (T[-1] - T[0]), th["zpe_eV"], f"ZPE = {th['zpe_eV']:.3f} eV", color=INK2, fontsize=8.5,
            va="top", ha="center", bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set(xlabel="T (K)", ylabel="energy (eV)")
    _legend_above(ax, ncol=3)

    # g  entropy and heat capacity
    ax = axes["thermo_s"][0]
    ax.plot(T, th["S_vib_meV_K"], color=SITE["core"], lw=2.2, label=r"$S_\mathrm{vib}$")
    ax.plot(T, th["Cv_meV_K"], color=SITE["surface"], lw=2.2, label=r"$C_v$")
    cl = d["hess"]["n_modes"] * KB_MEV
    ax.axhline(cl, color=INK2, lw=1, ls=":")
    ax.annotate(r"classical limit $(3N-6)k_B$", (T[0] + 0.62 * (T[-1] - T[0]), cl), xytext=(0, 5),
                textcoords="offset points", color=INK2, fontsize=8.5, va="bottom", ha="center")
    ax.set(xlabel="T (K)", ylabel="meV/K")
    _legend_above(ax, ncol=2)

    # h, i  stability
    stab = d["stab"]
    if "decomposition_dG_eV" in stab:
        Ts = np.asarray(stab["temperatures"])
        sb = stab["summary"]
        for key, y, de, ylab, lab in (
                ("stability", np.asarray(stab["decomposition_dG_eV"]) / d["n_ma"], sb["decomposition_dE_per_MA_eV"],
                 f"eV per {formula_tex(d['unit_ma'])}", r"$\Delta G_\mathrm{dec}$"),
                ("binding", np.asarray(stab["binding_dG_per_unit_eV"]), sb["binding_dE_per_unit_eV"],
                 "eV per unit", r"$\Delta G_\mathrm{bind}$")):
            ax = axes[key][0]
            ax.plot(Ts, y, color=SITE["core"], lw=2.2, label=lab)
            ax.axhline(de, color=INK2, lw=1, ls=":")
            ax.text(Ts[-1], de, f"ΔE = {de:+.3f} ", color=INK2, fontsize=8.5, ha="right", va="bottom")
            ax.set(xlabel="T (K)", ylabel=ylab)
            lo, hi = min(y.min(), de), max(y.max(), de)
            pad = 0.15 * (hi - lo or 1)
            ax.set_ylim(lo - pad, hi + pad)
            _legend_above(ax, ncol=1)

    # j, k  desorption ladder and equilibrium shell
    det = d["det"]
    if det.get("steps"):
        from matplotlib.colors import LinearSegmentedColormap
        mxt = formula_tex(d["unit_mx"])
        steps = det["steps"]
        ks = np.array([s["k"] for s in steps])
        ax = axes["detach"][0]
        ax.bar(ks, [s["dE_eV"] for s in steps], width=0.45, color=SITE["core"],
               label=r"$\Delta E_k$ (lowest configuration)")
        if det["summary"]["thermo"]:
            ax.plot(ks, [s["dG_300K_eV"] for s in steps], "o", ms=9, color=SITE["surface"], mec="white", mew=1.5,
                    label=r"$\Delta G_k$(300 K)", zorder=5)
        sites = det.get("site_classes", [])
        if sites:
            ax.plot([1.32] * len(sites), [c["dE_eV"] for c in sites], "o", ms=5, color=INK3, mec="white", mew=0.8,
                    label="distinct sites (k = 1)", zorder=4)
        ax.axhline(0, color=INK2, lw=0.9)
        ax.set_xticks(ks)
        ax.set(xlabel="units removed k", ylabel="eV")
        _legend_above(ax, ncol=3)

    if "sites" in axes:
        _sites_png(axes["sites"][0], d["sites"], fig)

    if det.get("steps") and "map" in axes:
        ax = axes["map"][0]
        mp = det["map"]
        z = np.asarray(mp["k_mean"], float)
        cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
        im = ax.pcolormesh(mp["dmu_eV"], mp["T"], z, cmap=cmap, vmin=0, vmax=det["summary"]["n_steps"],
                           shading="nearest", rasterized=True)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(rf"$\langle k\rangle$ {mxt} removed", color=INK2)
        cb.outline.set_edgecolor(GRID)
        cb.ax.tick_params(colors=INK2, labelsize=8.5)
        levels = [x for x in np.arange(0.5, det["summary"]["n_steps"], 1.0)]
        if levels and z.max() > levels[0]:
            cs = ax.contour(mp["dmu_eV"], mp["T"], z, levels=levels, colors="white", linewidths=0.6, alpha=0.8)
        ax.set(xlabel=rf"$\Delta\mu$({mxt}) $= k_BT\,\ln(c/1\,\mathrm{{M}})$ (eV)", ylabel="T (K)")
        mu = np.asarray(mp["dmu_eV"])
        ax.text(mu[-1] - 0.05 * (mu[-1] - mu[0]), mp["T"][len(mp["T"]) // 2], "shell intact", ha="right",
                va="center", fontsize=9.5, fontweight="bold", color=INK)
        ax.text(mu[0] + 0.05 * (mu[-1] - mu[0]), mp["T"][len(mp["T"]) // 2], "units desorbed", ha="left",
                va="center", fontsize=9.5, fontweight="bold", color="white")
        ax.grid(False)

    # l  key numbers
    ax = axes["keys"][0]
    ax.axis("off")
    ax.set_title(f"{letters[cells.index('keys')]}   Key numbers", loc="left", pad=26)
    y0 = 1.02
    step = min(0.085, 1.0 / max(len(keys), 1))
    for k in keys:
        ax.text(0.0, y0, k[1], fontsize=8.8, color=INK2, va="top", transform=ax.transAxes)
        ax.text(0.46, y0, k[3], fontsize=8.8, color=INK, fontweight="bold", va="top", transform=ax.transAxes)
        y0 -= step
    axes["keys"][1].text(0, 1.0, _wrap("Free energies at 300 K and a 1 M standard state, in vacuum; definitions in "
                                       "the interactive page and in the panel captions.", 80),
                         va="top", fontsize=8.3, color=INK2, transform=axes["keys"][1].transAxes)

    if sol:
        _solution_panels(axes, d)
    if vib:
        vibplots.png_panels(axes, vib, SITE, INK, INK3, _legend_above)

    for key, (ax, _c) in axes.items():
        if key != "keys" and ax.get_visible():
            _style_axes(ax)
            if key == "sites":
                ax.grid(False)
    fig.savefig(ctx.props / "ground_state.png", dpi=150)
    plt.close(fig)


def run(ctx) -> dict:
    from ..dashboards import JS_LIB, solution_section
    from ..solution import export
    d = _data(ctx)
    caps = _captions(d)
    keys = _key_numbers(d)
    figs = _plotly_figures(d)
    ex = export(ctx)
    solution = None
    if ex is not None:
        (ctx.props / "solution.json").write_text(json.dumps(ex))
        sec_html, sec_js = solution_section(ex, ctx.results.get("sites"))
        solution = (sec_html, JS_LIB + sec_js)
        d["solution"] = ex
    _write_html(ctx, d, figs, caps, keys, solution)
    _write_png(ctx, d, caps, keys)
    return {"summary": {"html": "ground_state.html", "png": "ground_state.png", "figures": sorted(figs)},
            "key_numbers": [[k[1], k[3], k[5]] for k in keys]}
