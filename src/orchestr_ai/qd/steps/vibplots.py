# src/orchestr_ai/qd/steps/vibplots.py
"""
Vibrational-spectra panels of the report (vibspec step): IR spectrum, Raman
spectrum and a mode map, as Plotly figures (HTML) and matplotlib axes (PNG).

Degenerate partners (same E or T irrep within 0.5 cm⁻¹) are merged into one peak with
summed intensity.  Sticks are coloured by the site that carries most of the
mode's mass-weighted amplitude (core / surface / ligand, the report's site
colours); activity is encoded by marker shape on the mode map.
"""
from __future__ import annotations

import numpy as np

FWHM = 6.0          # cm⁻¹, Lorentzian broadening of the envelopes
LASER_NM = 532.0
T_RAMAN = 300.0
ACTIVE = 0.01       # a mode is IR / Raman active above this fraction of the strongest one
C2_CM_K = 1.438776877  # hc/k_B in cm K


def _degenerate(irrep):
    """Only partners of a multi-dimensional irrep (E, T) are merged; accidental coincidences stay separate."""
    return bool(irrep) and irrep[0] in ("E", "T")


def peaks(v):
    """Merge degenerate partners; returns a list of dicts with summed intensities and mean shares."""
    out = []
    for m in sorted(v["modes"], key=lambda m: m["nu"]):
        if out and _degenerate(m["irrep"]) and out[-1]["irrep"] == m["irrep"] and abs(m["nu"] - out[-1]["nu"]) < 0.5:
            p = out[-1]
            p["deg"] += 1
            for k in ("ir_km_mol", "raman_A4_amu", "_perp"):
                p[k] += m[k] if k != "_perp" else _perp(m)
            for k in ("core", "surface", "ligand", "radial", "breathing"):
                p[k] += (m[k] - p[k]) / p["deg"]
            for e, x in m["elements"].items():
                p["elements"][e] += (x - p["elements"][e]) / p["deg"]
        else:
            out.append({**m, "elements": dict(m["elements"]), "deg": 1, "_perp": _perp(m)})
    ir_max = max((p["ir_km_mol"] for p in out), default=1.0) or 1.0
    s_max = max((p["raman_A4_amu"] for p in out), default=1.0) or 1.0
    for p in out:
        p["rho"] = p["_perp"] / max(p["raman_A4_amu"] - p["_perp"], 1e-30) if p["raman_A4_amu"] > 1e-6 * s_max else None
        p["site"] = max(("core", "surface", "ligand"), key=lambda r: p[r])
        ir_on, ra_on = p["ir_km_mol"] >= ACTIVE * ir_max, p["raman_A4_amu"] >= ACTIVE * s_max
        p["activity"] = "IR + Raman" if ir_on and ra_on else "IR" if ir_on else "Raman" if ra_on else "weak / silent"
        p["raman_I"] = raman_intensity(p["nu"], p["raman_A4_amu"])
        p["raman_I_perp"] = raman_intensity(p["nu"], p["_perp"])
    r_max = max((p["raman_I"] for p in out), default=1.0) or 1.0
    for p in out:
        p["raman_I"] /= r_max
        p["raman_I_perp"] /= r_max
    return out


def _perp(m):
    # I_perp / I_par = rho with I_par + I_perp = S:  I_perp = S rho / (1 + rho)
    rho = m["rho"] or 0.0
    return m["raman_A4_amu"] * rho / (1 + rho)


def raman_intensity(nu, s):
    """Stokes intensity (relative): (nu0 − nu)^4 / nu · S / (1 − exp(−hc nu / k_B T))."""
    if nu <= 1.0:
        return 0.0
    nu0 = 1e7 / LASER_NM
    return (nu0 - nu) ** 4 / nu * s / (-np.expm1(-C2_CM_K * nu / T_RAMAN))


def envelope(x, pk, key):
    g = (FWHM / 2) ** 2
    return sum(p[key] * g / ((x - p["nu"]) ** 2 + g) for p in pk) if pk else np.zeros_like(x)


HEADROOM = 1.5          # axis top / tallest feature: the legend sits above the data
RAMAN_SCALE_MIN = 50.0   # cm⁻¹: the Raman axis is scaled to the strongest feature above this


def env_maxima(x, env, pk, ykey, n=3, sep=15.0, xmin=0.0):
    """Highest local maxima of the envelope, >= sep apart: [(nu, height, irrep of the strongest stick nearby)]."""
    loc = np.where((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (x[1:-1] >= xmin))[0] + 1
    out = []
    for i in sorted(loc, key=lambda i: -env[i]):
        if all(abs(x[i] - o[0]) >= sep for o in out):
            near = [p for p in pk if abs(p["nu"] - x[i]) <= FWHM]
            best = max(near, key=lambda p: p[ykey]) if near else None
            out.append((float(x[i]), float(env[i]), best["irrep"] if best else ""))
        if len(out) == n:
            break
    return out


def y_top(x, env, raman):
    """Axis top: the whole envelope for IR; for Raman the strongest feature above RAMAN_SCALE_MIN."""
    hi = env[x >= RAMAN_SCALE_MIN].max() if raman and (x >= RAMAN_SCALE_MIN).any() else env.max()
    return HEADROOM * (hi or 1.0)


def _shares(p, elements):
    el = " · ".join(f"{e} {p['elements'][e]:.2f}" for e in elements if e in p["elements"])
    return (f"core {p['core']:.2f} · surface {p['surface']:.2f} · ligand {p['ligand']:.2f}<br>{el}"
            f"<br>radial share {p['radial']:.2f} · breathing overlap {p['breathing']:.2f}")


def _label(p):
    deg = f" (×{p['deg']})" if p["deg"] > 1 else ""
    return f"{p['nu']:.1f} cm<sup>−1</sup> · {_irrep_html(p['irrep'])}{deg}"


def _irrep_html(s):
    return s.replace("1", "<sub>1</sub>").replace("2", "<sub>2</sub>") if s and s[0] in "ABET" else (s or "")


def _irrep_tex(s):
    if not s or s[0] not in "ABET":
        return s or ""
    base = s.rstrip("'")
    primes = s[len(base):]
    return f"{base[0]}$_{{{base[1:]}}}${primes}" if len(base) > 1 else s


def captions(v):
    sm = v["summary"]
    pg = sm["point_group"]
    sel = ""
    if sm.get("ir_active_irreps"):
        sel = (f" In {pg}, IR-active: {', '.join(sm['ir_active_irreps'])}; Raman-active: "
               f"{', '.join(sm['raman_active_irreps'])}.")
    to = max(sm["bulk_gamma_optical_cm1"]) if sm.get("bulk_gamma_optical_cm1") else None
    to_h = (f" Dashed: bulk Γ-point optical frequency of the CIF ({to:.0f} cm<sup>−1</sup>, MACE, TO only — "
            "LO–TO splitting needs Born charges and is not computed)." if to else "")
    to_t = (f" Dashed: bulk Γ-point optical frequency ({to:.0f} cm$^{{-1}}$, MACE, TO only; no LO–TO splitting)."
            if to else "")
    grad = sm.get("gxtb_gradient_norm_Eh_bohr")
    g_h = f" The g-xTB gradient norm at the MACE minimum is {grad:.3f} E<sub>h</sub>/bohr." if grad else ""
    return {
        "ir": ("IR spectrum", "IR spectrum",
               "Peak positions: MACE-MH-1 harmonic modes. Intensities: A<sub>k</sub> = (N<sub>A</sub>π/3c) "
               "|∂μ/∂Q<sub>k</sub>|² (km/mol), with the g-xTB dipole differentiated along each normal coordinate "
               f"(central difference, ±{sm['step_amu05_A']} amu<sup>½</sup> Å). Sticks: degenerate partners summed, "
               f"coloured by the site carrying most of the amplitude; line: Lorentzians of {FWHM:.0f} cm<sup>−1</sup> "
               f"FWHM.{sel}{to_h}",
               "Peak positions: MACE-MH-1 modes. Intensities: $A_k = (N_A\\pi/3c)\\,|\\partial\\mu/\\partial Q_k|^2$ "
               "(km/mol), g-xTB dipole differentiated along each normal coordinate. Sticks coloured by the site "
               f"carrying most of the amplitude; line: {FWHM:.0f} cm$^{{-1}}$ Lorentzians.{sel}{to_t}"),
        "raman": ("Raman spectrum (non-resonant)", "Raman spectrum (non-resonant)",
                  "Activity S<sub>k</sub> = 45a′² + 7γ′² from the g-xTB polarisability derivative ∂α/∂Q<sub>k</sub> "
                  f"(finite field ±{sm['field_V_A']} V/Å along x, y, z), with a′ its isotropic part and γ′ its "
                  "anisotropy. Intensity I<sub>k</sub> ∝ (ν<sub>0</sub> − ν<sub>k</sub>)<sup>4</sup>/ν<sub>k</sub> · "
                  f"S<sub>k</sub> / (1 − e<sup>−hcν<sub>k</sub>/k<sub>B</sub>T</sup>), {LASER_NM:.0f} nm, "
                  f"{T_RAMAN:.0f} K, normalised to its maximum; the axis is scaled to the strongest feature above "
                  f"{RAMAN_SCALE_MIN:.0f} cm<sup>−1</sup> (1/ν and the Bose factor inflate the lowest modes). Dotted: depolarised part I<sub>⊥</sub> = ρ/(1 + ρ) I, with "
                  "ρ = 3γ′²/(45a′² + 4γ′²); totally symmetric modes are polarised (ρ < ¾) and nearly vanish in it. "
                  "Measured QD spectra are usually resonant (LO-enhanced): compare positions and symmetries, not "
                  f"relative intensities.{g_h}",
                  "Activity $S_k = 45a'^2 + 7\\gamma'^2$ from the g-xTB $\\partial\\alpha/\\partial Q_k$ (finite field "
                  f"±{sm['field_V_A']} V/Å). $I_k \\propto (\\nu_0-\\nu_k)^4/\\nu_k \\cdot S_k/(1-e^{{-hc\\nu_k/k_BT}})$, "
                  f"{LASER_NM:.0f} nm, {T_RAMAN:.0f} K. Dotted: depolarised part $\\rho/(1+\\rho)\\,I$, "
                  "$\\rho = 3\\gamma'^2/(45a'^2+4\\gamma'^2)$; totally symmetric modes nearly vanish in it. "
                  "Non-resonant: measured QD spectra are LO-enhanced."),
        "modemap": ("Mode character and activity", "Mode character and activity",
                    "Every mode (degenerate sets merged) by frequency and the share of its mass-weighted amplitude on "
                    "core atoms; marker area ∝ the larger of its relative IR and Raman intensity, shape = activity "
                    f"(above {ACTIVE:.0%} of the strongest peak), colour = dominant site. Hover for irrep, ρ, "
                    "element shares, radial share and the overlap with a uniform breathing of the dot. Classes: "
                    "breathing (overlap ≥ 0.3); M–X ligand (ligand share > 0.5); in the optical range "
                    "(ν ≥ 0.75 ν<sub>TO</sub>) core optical (core share ≥ 0.4) or surface optical.",
                    "Every mode by frequency and core share of its amplitude; area ∝ max(relative IR, relative Raman), "
                    f"shape = activity (above {ACTIVE:.0%} of the strongest), colour = dominant site. Classes: "
                    "breathing (overlap ≥ 0.3), M–X ligand (ligand share > 0.5), and above "
                    "$0.75\\,\\nu_\\mathrm{TO}$ core optical (core share ≥ 0.4) or surface optical."),
    }


def key_rows(v):
    """Rows (label_html, label_tex, value_html, value_tex, definition_html, definition_tex)."""
    sm = v["summary"]
    rows = [("Point group", "Point group", sm["point_group"], sm["point_group"], "symmetry of the relaxed dot",
             "symmetry of the relaxed dot"),
            ("Polarisability α<sub>iso</sub>", "Polarisability $\\alpha_\\mathrm{iso}$", f"{sm['alpha_iso_A3']:.0f} Å³",
             f"{sm['alpha_iso_A3']:.0f} Å³", "⅓ tr α, g-xTB, finite field", "g-xTB, finite field")]
    if sm.get("top_ir"):
        nu, irr, a = sm["top_ir"][0]
        rows.append(("Strongest IR peak", "Strongest IR peak", f"{nu:.0f} cm<sup>−1</sup> ({_irrep_html(irr)})",
                     f"{nu:.0f} cm$^{{-1}}$ ({_irrep_tex(irr)})", f"A = {a:.0f} km/mol", f"{a:.0f} km/mol"))
    if sm.get("top_raman"):
        nu, irr, s = sm["top_raman"][0]
        rows.append(("Strongest Raman activity", "Strongest Raman activity",
                     f"{nu:.0f} cm<sup>−1</sup> ({_irrep_html(irr)})", f"{nu:.0f} cm$^{{-1}}$ ({_irrep_tex(irr)})",
                     f"S = {s:.0f} Å<sup>4</sup>/amu, non-resonant", f"{s:.0f} Å$^4$/amu"))
    return rows


SHAPES = {"IR": ("circle", "o"), "Raman": ("diamond", "D"), "IR + Raman": ("star", "*"),
          "weak / silent": ("circle-open", "o")}


def _x_range(pk):
    return [0, max(p["nu"] for p in pk) + 25]


def plotly_figs(v, site, layout, ink, ink3, elements):
    import plotly.graph_objects as go
    pk = peaks(v)
    sm = v["summary"]
    to = max(sm["bulk_gamma_optical_cm1"]) if sm.get("bulk_gamma_optical_cm1") else None
    x = np.linspace(0, _x_range(pk)[1], 1500)
    figs = {}
    for key, ykey, perp, ytitle in (("ir", "ir_km_mol", None, "IR intensity (km/mol)"),
                                    ("raman", "raman_I", "raman_I_perp", "Raman intensity (normalised)")):
        fig = go.Figure()
        fig.add_scatter(x=x, y=envelope(x, pk, ykey), mode="lines", name=f"{FWHM:.0f} cm<sup>−1</sup> Lorentzians",
                        line=dict(color=ink, width=1.6), hoverinfo="skip")
        if perp:
            fig.add_scatter(x=x, y=envelope(x, pk, perp), mode="lines", name="depolarised (I<sub>⊥</sub>)",
                            line=dict(color=ink3, width=1.4, dash="dot"), hoverinfo="skip")
        for r in ("core", "surface", "ligand"):
            sel = [p for p in pk if p["site"] == r]
            if not sel:
                continue
            xs, ys = [], []
            for p in sel:
                xs += [p["nu"], p["nu"], None]
                ys += [0, p[ykey], None]
            fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=site[r], width=2.5), name=f"mostly {r}",
                            legendgroup=r, hoverinfo="skip")
            text = [f"<b>{_label(p)}</b><br>{p['class']}<br>IR {p['ir_km_mol']:.2f} km/mol · Raman S "
                    f"{p['raman_A4_amu']:.2f} Å<sup>4</sup>/amu" + (f" · ρ {p['rho']:.2f}" if p["rho"] is not None else "")
                    + f"<br>{_shares(p, elements)}" for p in sel]
            fig.add_scatter(x=[p["nu"] for p in sel], y=[p[ykey] for p in sel], mode="markers", legendgroup=r,
                            showlegend=False, marker=dict(color=site[r], size=7, line=dict(color="white", width=1)),
                            text=text, hovertemplate="%{text}<extra></extra>")
        if to:
            fig.add_vline(x=to, line=dict(color=ink3, width=1, dash="dash"))
            fig.add_annotation(x=to, y=1.0, yref="paper", text="bulk TO(Γ)", showarrow=False, yanchor="top",
                               xanchor="right", textangle=-90, font=dict(size=10, color=ink3))
        env = envelope(x, pk, ykey)
        top_y = y_top(x, env, perp is not None)
        for nu, hgt, _irr in env_maxima(x, env, pk, ykey, n=2):
            if hgt > top_y:
                fig.add_annotation(x=nu, y=top_y, text=f"{nu:.0f} cm<sup>−1</sup>: off scale ×{hgt / (top_y / HEADROOM):.1f}",
                                   showarrow=False, xanchor="left", yanchor="top", xshift=4,
                                   font=dict(size=10, color=ink3))
        fig.update_layout(**layout("wavenumber (cm<sup>−1</sup>)", ytitle, xaxis=dict(range=_x_range(pk)),
                                   yaxis=dict(range=[0, top_y])))
        figs[key] = fig

    fig = go.Figure()
    big = [max(p["ir_km_mol"] / (max(q["ir_km_mol"] for q in pk) or 1.0), p["raman_I"]) for p in pk]
    for act, (shape, _m) in SHAPES.items():
        for r in ("core", "surface", "ligand"):
            sel = [(p, b) for p, b in zip(pk, big) if p["activity"] == act and p["site"] == r]
            if not sel:
                continue
            fig.add_scatter(
                x=[p["nu"] for p, _ in sel], y=[p["core"] for p, _ in sel], mode="markers",
                name=f"{act}, mostly {r}", showlegend=False,
                marker=dict(symbol=shape, color=site[r], size=[7 + 18 * np.sqrt(b) for _, b in sel],
                            line=dict(color=site[r] if shape.endswith("open") else "white", width=1)),
                text=[f"<b>{_label(p)}</b> · {p['activity']}<br>{p['class']}<br>IR {p['ir_km_mol']:.2f} km/mol · "
                      f"Raman S {p['raman_A4_amu']:.2f} Å<sup>4</sup>/amu"
                      + (f" · ρ {p['rho']:.2f}" if p["rho"] is not None else "") + f"<br>{_shares(p, elements)}"
                      for p, _ in sel],
                hovertemplate="%{text}<extra></extra>")
    # compact legend: shapes (activity) in grey, colours (dominant site) as dots
    for act, (shape, _m) in SHAPES.items():
        fig.add_scatter(x=[None], y=[None], mode="markers", name=act,
                        marker=dict(symbol=shape, color=ink3, size=10, line=dict(color=ink3, width=1)))
    for r in ("core", "surface", "ligand"):
        fig.add_scatter(x=[None], y=[None], mode="markers", name=f"mostly {r}",
                        marker=dict(symbol="circle", color=site[r], size=10))
    if to:
        for xv, lab, dash in ((to, "bulk TO(Γ)", "dash"), (0.75 * to, "optical range", "dot")):
            fig.add_vline(x=xv, line=dict(color=ink3, width=1, dash=dash))
            fig.add_annotation(x=xv, y=1.0, yref="paper", text=lab, showarrow=False, yanchor="top",
                               xanchor="right", textangle=-90, font=dict(size=10, color=ink3))
    fig.update_layout(**layout("wavenumber (cm<sup>−1</sup>)", "core share of amplitude",
                               xaxis=dict(range=_x_range(pk)), yaxis=dict(range=[-0.04, 1.04]), height=380))
    figs["modemap"] = fig
    return figs


def png_panels(axes, v, site, ink, ink3, legend_above):
    from matplotlib.lines import Line2D
    pk = peaks(v)
    sm = v["summary"]
    to = max(sm["bulk_gamma_optical_cm1"]) if sm.get("bulk_gamma_optical_cm1") else None
    xr = _x_range(pk)
    x = np.linspace(0, xr[1], 1500)
    for key, ykey, perp, ylab in (("ir", "ir_km_mol", None, "IR intensity (km/mol)"),
                                  ("raman", "raman_I", "raman_I_perp", "Raman intensity (normalised)")):
        ax = axes[key][0]
        ax.plot(x, envelope(x, pk, ykey), color=ink, lw=1.4)
        if perp:
            ax.plot(x, envelope(x, pk, perp), color=ink3, lw=1.2, ls=":")
        for p in pk:
            ax.vlines(p["nu"], 0, p[ykey], color=site[p["site"]], lw=2.2, zorder=3)
        env = envelope(x, pk, ykey)
        top_y = y_top(x, env, perp is not None)
        ymax = top_y / HEADROOM
        for nu, hgt, irr in env_maxima(x, env, pk, ykey, xmin=RAMAN_SCALE_MIN if perp else 0.0):
            ax.annotate(f"{nu:.0f} {_irrep_tex(irr)}", (nu, hgt), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7.5, color=ink)
        for nu, hgt, _irr in env_maxima(x, env, pk, ykey, n=2):
            if hgt > top_y:          # low-frequency Raman peaks enhanced by 1/nu and the Bose factor
                ax.annotate(f"{nu:.0f} cm$^{{-1}}$, off scale ×{hgt / ymax:.1f}", (nu, top_y), xytext=(4, -10),
                            textcoords="offset points", ha="left", fontsize=7.5, color=ink3)
        if to:
            ax.axvline(to, color=ink3, ls="--", lw=0.9)
            ax.text(to, ymax * 1.17, " bulk TO(Γ)", ha="right", va="top", rotation=90, fontsize=7.5, color=ink3)
        ax.set(xlim=xr, ylim=(0, top_y), xlabel=r"wavenumber (cm$^{-1}$)", ylabel=ylab)
        handles = [Line2D([], [], color=site[r], lw=3) for r in ("core", "surface", "ligand")]
        labels = ["mostly core", "mostly surface", "mostly ligand"]
        handles.append(Line2D([], [], color=ink, lw=1.4))
        labels.append(f"{FWHM:.0f} cm$^{{-1}}$ Lorentzians")
        if perp:
            handles.append(Line2D([], [], color=ink3, lw=1.2, ls=":"))
            labels.append(r"depolarised $I_\perp$")
        # inside the axes, where it overlaps the curves and labels least
        ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=8, handlelength=1.6, ncol=2)

    ax = axes["modemap"][0]
    big = [max(p["ir_km_mol"] / (max(q["ir_km_mol"] for q in pk) or 1.0), p["raman_I"]) for p in pk]
    for p, b in zip(pk, big):
        _shape, mk = SHAPES[p["activity"]]
        open_ = p["activity"] == "weak / silent"
        ax.scatter(p["nu"], p["core"], marker=mk, s=(5 + 13 * np.sqrt(b)) ** 2 / 2,
                   facecolors="none" if open_ else site[p["site"]], edgecolors=site[p["site"]] if open_ else "white",
                   linewidths=0.9, zorder=3)
    if to:
        for xv, lab, ls in ((to, "bulk TO(Γ)", "--"), (0.75 * to, "optical range", ":")):
            ax.axvline(xv, color=ink3, ls=ls, lw=0.9)
            ax.text(xv, 1.02, f"{lab} ", ha="right", va="top", rotation=90, fontsize=7.5, color=ink3)
    # headroom above share = 1 holds the legend, so no marker is hidden
    ax.set(xlim=xr, ylim=(-0.04, 1.4), xlabel=r"wavenumber (cm$^{-1}$)", ylabel="core share of amplitude")
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    handles = [Line2D([], [], marker=SHAPES[a][1], ls="", color=ink3, mfc="none" if a == "weak / silent" else ink3,
                      ms=7) for a in SHAPES]
    handles += [Line2D([], [], marker="o", ls="", color=site[r], ms=7) for r in ("core", "surface", "ligand")]
    ax.legend(handles, list(SHAPES) + ["mostly core", "mostly surface", "mostly ligand"], loc="upper left",
              frameon=False, fontsize=8, ncol=4, columnspacing=1.0, handletextpad=0.3)
