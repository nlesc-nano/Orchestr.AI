# src/orchestr_ai/qd/dashboards.py
"""
Interactive solution-thermodynamics dashboards (Plotly + plain JS).

* solution_section(export): the "Stability & ligands in solution" block of a
  dot's ground_state.html: dG_dec(T), dG_bind(T) with T_diss, the stepwise
  CdCl2 ladder, the desorption isotherm <k>(c) and the <k>(c, T) map, each with
  its own controls (solvent model and eps, concentrations, precursor
  stabilisation, T) for only the quantities that enter it.
* synthesis_page(exports): several dots in mutual equilibrium with the
  monomers (mass balance on CdSe and CdCl2): yield of each dot family vs T,
  populations vs the [CdCl2]/[CdSe] ratio, free monomers and supersaturation,
  ligand coverage, optionally with bulk CdSe precipitation.

The formulas are those of qdprops.solution (the Python versions produce the
static PNGs and the tests); JS_LIB mirrors them.
"""
from __future__ import annotations

import html
import json
import re

from .solution import KB_EV

SITE = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"

JS_LIB = r"""
const KB = %(kb)r;
function solvAt(sp, epsGrid, solvent) {      // Generalized Born: (1 - 1/eps) G_GB; or ALPB; or gas
  if (solvent === null) return 0;
  if (typeof solvent === "string") return sp.alpb[solvent] === null ? NaN : sp.alpb[solvent];
  return (1 - 1 / solvent) * sp.solv_inf;
}
const gSol = (sp, eps, solvent, shift = 0) => { const s = solvAt(sp, eps, solvent) + shift; return sp.G.map(g => g + s); };
function lse(a) { let m = -Infinity; for (const v of a) if (v > m) m = v; let s = 0; for (const v of a) s += Math.exp(v - m); return m + Math.log(s); }

function curves(ex, solvent, cMA, cMX, shift) {
  const T = ex.T, n = ex.units.n, m = ex.units.m, eps = null;
  const g0 = gSol(ex.dots[0], eps, solvent), gMA = gSol(ex.MA, eps, solvent, shift), gMX = gSol(ex.MX, eps, solvent);
  const muMA = T.map((t, i) => gMA[i] + KB * t * Math.log(cMA)), muMX = T.map((t, i) => gMX[i] + KB * t * Math.log(cMX));
  const dec = T.map((t, i) => (g0[i] - n * ex.bulk[i] - m * muMX[i]) / n);
  const bind = T.map((t, i) => (g0[i] - n * muMA[i] - m * muMX[i]) / (n + m));
  let tDiss = null;
  for (let i = 0; i < T.length - 1; i++) if (bind[i] < 0 && bind[i + 1] >= 0) {
    tDiss = T[i] + (0 - bind[i]) * (T[i + 1] - T[i]) / (bind[i + 1] - bind[i]); break; }
  const gs = ex.dots.map(d => gSol(d, eps, solvent));
  const ladder = gs.slice(1).map((g, k) => T.map((t, i) => g[i] - gs[k][i] + muMX[i]));
  return {T, dec, bind, tDiss, ladder};
}

function meanRemovedAt(ex, solvent, cMX, ti) {
  const t = ex.T[ti], kT = KB * t, eps = null;
  const muMX = gSol(ex.MX, eps, solvent)[ti] + kT * Math.log(cMX);
  const gs = {}, sig = {}; for (const d of ex.dots) { gs[d.k] = gSol(d, eps, solvent)[ti]; sig[d.k] = d.sigma; }
  const confs = [[0, 0, sig[0]], ...ex.ensemble.filter(c => c[0] in gs)];
  // a configuration's G: the ladder dot_k's G with its own rotational symmetry number
  const lg = confs.map(([k, off, sc]) => -(gs[k] - kT * Math.log(sig[k]) + kT * Math.log(sc) + off + k * muMX) / kT);
  const z = lse(lg); let s = 0; confs.forEach((c, i) => { s += c[0] * Math.exp(lg[i] - z); });
  return s;
}

function speciesList(exports) {
  const out = []; for (const ex of exports) for (const d of ex.dots) out.push({family: ex.formula, ...d}); return out;
}
function bisect(f, lo, hi) {               // root of an increasing function
  for (let i = 0; i < 200 && f(lo) >= 0; i++) lo -= Math.max(50, Math.abs(lo));
  for (let i = 0; i < 200 && hi - lo > 1e-12; i++) { const mid = 0.5 * (lo + hi); if (f(mid) < 0) lo = mid; else hi = mid; }
  return 0.5 * (lo + hi);
}

function equilibrium(exports, sp, ti, solvent, CA, CX, shift, allowBulk) {
  // Nested bisection in (a, b) = (ln c_MA, ln c_MX): both mass balances are increasing log-sum-exps.
  const ex0 = exports[0], kT = KB * ex0.T[ti];
  const gMA = gSol(ex0.MA, null, solvent, shift)[ti], gMX = gSol(ex0.MX, null, solvent)[ti];
  const g0 = sp.map(s => (gSol(s, null, solvent)[ti] - s.n * gMA - s.m * gMX) / kT);
  const lnSat = -(gMA - ex0.bulk[ti]) / kT, lnA = Math.log(CA), lnX = Math.log(CX);
  const N = sp.map(s => s.n), M = sp.map(s => s.m), lnN = N.map(Math.log), lnM = M.map(v => v > 0 ? Math.log(v) : -Infinity);
  const buf = new Array(sp.length + 1);
  const lA = (a, b) => { buf[0] = a; for (let i = 0; i < N.length; i++) buf[i + 1] = lnN[i] - g0[i] + N[i] * a + M[i] * b; return lse(buf); };
  const lX = (a, b) => { buf[0] = b; for (let i = 0; i < N.length; i++) buf[i + 1] = lnM[i] - g0[i] + N[i] * a + M[i] * b; return lse(buf); };
  const aOf = b => bisect(a => lA(a, b) - lnA, lnA - 60, lnA);
  let b = bisect(bb => lX(aOf(bb), bb) - lnX, lnX - 60, lnX), a = aOf(b), bulk = 0, conserved = true;
  if (allowBulk && a > lnSat) {
    a = lnSat; b = bisect(bb => lX(a, bb) - lnX, lnX - 60, lnX);
    const excess = 1 - Math.exp(lA(a, b) - lnA); conserved = excess >= -1e-9; bulk = Math.max(0, excess);
  }
  const fam = {};
  sp.forEach((s, i) => { const c = Math.exp(-g0[i] + s.n * a + s.m * b);
    const f = fam[s.family] || (fam[s.family] = {frac: 0, conc: 0, ksum: 0});
    f.frac += s.n * c / CA; f.conc += c; f.ksum += s.k * c; });
  for (const f of Object.values(fam)) f.meanK = f.conc > 0 ? f.ksum / f.conc : null;
  return {a, b, monomer: Math.exp(a) / CA, bulk, fam, S: Math.exp(a - lnSat), conserved};
}

// Per-panel controls: only the parameters that change that panel.  specs[key] = {label (with {v}), min, max,
// step, value, log (slider is log10 of the value), digits, help, type: "range" | "check" | "solv"}.
function makeControls(host, keys, specs, alpb, onchange) {
  const root = document.getElementById(host), st = {}, parts = [];
  root.className = "ctl";
  for (const k of keys) {
    const sp = specs[k];
    if (sp.type === "solv") {
      parts.push(`<div><label>Solvent model <select data-k="model"><option value="gb" selected>Generalized Born, ε from the slider</option>` +
        alpb.map(([s, e]) => `<option value="alpb:${s}">ALPB ${s} (ε ${e})</option>`).join("") +
        `<option value="gas">gas phase</option></select></label></div>` +
        `<div><label>Dielectric constant ε = <output data-o="eps"></output>` +
        `<input type="range" data-k="eps" min="0" max="1.903" step="0.001" value="${Math.log10(sp.value)}"></label></div>`);
    } else if (sp.type === "check") {
      parts.push(`<div><label title="${sp.help || ""}"><input type="checkbox" data-k="${k}"> ${sp.label}</label></div>`);
    } else if (sp.type === "select") {
      parts.push(`<div><label title="${sp.help || ""}">${sp.label} <select data-k="${k}">` +
        sp.options.map(([v, t]) => `<option value="${v}"${v === sp.value ? " selected" : ""}>${t}</option>`).join("") +
        `</select></label></div>`);
    } else {
      parts.push(`<div><label title="${sp.help || ""}">${sp.label.replace("{v}", `<output data-o="${k}"></output>`)}` +
        `<input type="range" data-k="${k}" min="${sp.min}" max="${sp.max}" step="${sp.step}" value="${sp.value}"></label></div>`);
    }
  }
  root.innerHTML = parts.join("");
  const q = sel => root.querySelector(sel);
  function read() {
    for (const k of keys) {
      const sp = specs[k];
      if (sp.type === "solv") {
        const model = q('[data-k="model"]').value, e = Math.pow(10, +q('[data-k="eps"]').value);
        q('[data-o="eps"]').textContent = e.toFixed(2); q('[data-k="eps"]').disabled = model !== "gb";
        st[k] = model === "gas" ? null : model.startsWith("alpb:") ? model.slice(5) : e;
      } else if (sp.type === "check") st[k] = q(`[data-k="${k}"]`).checked;
      else if (sp.type === "select") st[k] = q(`[data-k="${k}"]`).value;
      else { const v = +q(`[data-k="${k}"]`).value; q(`[data-o="${k}"]`).textContent = v.toFixed(sp.digits ?? 1);
             st[k] = sp.log ? Math.pow(10, v) : v; st[k + "_raw"] = v; }
    }
  }
  root.addEventListener("input", () => { read(); onchange(st); });
  st._root = root;
  read(); onchange(st);
  return st;
}
const setDisabled = (st, keys, off) => keys.forEach(k => st._root.querySelectorAll(`[data-k="${k}"]`).forEach(e => { e.disabled = off; }));
""" % {"kb": KB_EV}


def _fh(f):
    return re.sub(r"(\d+)", r"<sub>\1</sub>", html.escape(str(f)))


CONTROLS_CSS = """
.ctl {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(260px, 100%), 1fr)); gap: 10px 20px;
        background: #f8f9fa; border: 1px solid {grid}; border-radius: 8px; padding: 12px 14px; margin: 6px 0 14px; }}
.ctl label {{ font-size: 12px; color: {ink2}; display: block; }}
.ctl output {{ font-weight: 700; color: {ink}; }}
.ctl input[type=range] {{ width: 100%; }}
.ctl select {{ font-size: 12px; padding: 3px 6px; border: 1px solid {grid}; border-radius: 6px; background: white; }}
.panel .ctl {{ grid-template-columns: repeat(auto-fit, minmax(min(190px, 100%), 1fr)); gap: 6px 14px; padding: 8px 10px;
               margin: 4px 0 6px; }}
.ctl input:disabled, .ctl select:disabled {{ opacity: .4; }}
.panel.wide {{ grid-column: 1 / -1; }}
""".format(grid=GRID, ink2=INK2, ink=INK)


def _specs(ma, mx, eps0=2.4):
    """Slider definitions shared by the solution section and the synthesis page."""
    shift_help = ("Constant free-energy offset of the " + re.sub("<[^>]+>", "", ma) + " monomer: its chemical potential is "
                  "mu = G(isolated molecule) + dG_solv + dmu_prec + kT ln c. A negative value mimics a stabilised molecular "
                  "precursor (e.g. a metal carboxylate with a phosphine chalcogenide) instead of the bare molecule.")
    return {
        "solv": {"type": "solv", "value": eps0},
        "cmx": {"label": f"[{mx}] = 10<sup>{{v}}</sup> M", "min": -20, "max": 0, "step": 0.1, "value": -2, "log": True},
        "cma": {"label": f"[{ma}] monomer = 10<sup>{{v}}</sup> M", "min": -30, "max": 0, "step": 0.1, "value": -2,
                "log": True, "help": "concentration of free monomer (enters as kT ln c)"},
        "shift": {"label": f"Precursor stabilisation Δμ<sub>prec</sub>({ma}) = {{v}} eV", "min": -3, "max": 1,
                  "step": 0.01, "value": 0, "digits": 2, "help": shift_help},
        "T": {"label": "T = {v} K", "min": 50, "max": 1500, "step": 10, "value": 300, "digits": 0},
        "Tsyn": {"label": "T = {v} K", "min": 50, "max": 1500, "step": 10, "value": 500, "digits": 0},
        "ca": {"label": f"Total [{ma}] (precursor) = 10<sup>{{v}}</sup> M", "min": -8, "max": 0, "step": 0.1,
               "value": -2, "log": True},
        "ratio": {"label": f"Ratio [{mx}] / [{ma}] = 10<sup>{{v}}</sup>", "min": -3, "max": 2, "step": 0.05,
                  "value": -0.6, "log": True, "digits": 2},
        "bulk": {"type": "check", "label": f"allow bulk {ma} to precipitate (thermodynamic sink)"},
        "view": {"type": "select", "label": "View", "value": "3d",
                 "options": [["3d", "3D, space-filling (orthographic)"], ["2d", "2D projection (θ, φ)"]]},
        "colour": {"type": "select", "label": "Colour by", "value": "dG",
                   "options": [["dG", "ΔG<sub>site</sub> in solution".replace("<sub>", "").replace("</sub>", "")],
                               ["dE", "ΔE<sub>site</sub>, vacuum, electronic".replace("<sub>", "").replace("</sub>", "")]]},
    }


SHIFT_NOTE = ("Δμ<sub>prec</sub> versus concentration: the monomer chemical potential is μ = G°(molecule) + "
              "ΔG<sub>solv</sub> + Δμ<sub>prec</sub> + k<sub>B</sub>T ln c. The concentration term is the ideal-dilution "
              "entropy of the free monomer and scales with T; Δμ<sub>prec</sub> is a constant offset for a monomer that is "
              "really bound in a molecular precursor, which lies well below the bare gas-phase molecule.")


SITES_JS = r"""
(function () {
const EX = __EX__, SD = __SITES__, SPECS = __SPECS__;
const el = id => document.getElementById(id);
const KBT = t => KB * t, TI = t => Math.max(0, SD.T.indexOf(t));
const R_VDW = {Cd: 1.58, Se: 1.90, S: 1.80, Te: 2.06, Cl: 1.75, Br: 1.85, I: 1.98, Zn: 1.39, In: 1.93, Ga: 1.87,
               P: 1.80, As: 1.85, Sb: 2.06, Pb: 2.02, Cs: 3.43, Al: 1.84};
const GREY = {Se: "#7d7c78", S: "#8d8c6c", Te: "#6d6c68", Cd: "#a3a29d"};   // darker than the lightest site colour
const SYM = SD.dot.symbols, POS = SD.dot.positions, ROLE = SD.dot.role, N = SYM.length;
const hasG = SD.classes.every(c => c.G_prod_eV);
// which class colours each atom (the cheapest unit it belongs to is chosen per state)
const MEMBER = Array.from({length: N}, () => []);
SD.classes.forEach((c, ci) => c.orbit.forEach(u => u.forEach(a => MEMBER[a].push(ci))));
function classValues(st) {
  const ti = TI(st.T), kT = KBT(st.T), f = st.solv === null ? 0 : 1 - 1 / st.solv;
  return SD.classes.map(c => {
    if (st.colour === "dE" || !hasG) return c.dE_eV;
    const muMX = EX.MX.G[ti] + f * EX.MX.solv_inf + kT * Math.log(st.cmx);
    return (c.G_prod_eV[ti] + f * c.solv_inf_eV) - (SD.dot.G_label_eV[ti] + f * SD.dot.solv_inf_eV) + muMX;
  });
}
function atomValues(cv) {
  return MEMBER.map(ms => ms.length ? ms.reduce((b, ci) => (b === null || cv[ci] < cv[b] ? ci : b), null) : null)
               .map(ci => ci === null ? null : {v: cv[ci], ci});
}
// colour scales: magnitude (dE) sequential; dG diverging around 0 (orange: desorbs, blue: bound)
const SEQ = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]];
const DIV = [[0, "#b84a12"], [0.25, "#eb8a5c"], [0.5, "#efeeea"], [0.75, "#6aa3e8"], [1, "#1c5cab"]];
function scale(st, vals) {
  const v = vals.filter(x => x !== null);
  if (st.colour === "dE" || !hasG) { let lo = Math.min(...v), hi = Math.max(...v);
    if (hi - lo < 0.1) { const m = 0.5 * (lo + hi); lo = m - 0.05; hi = m + 0.05; }    // equal sites: mid-scale
    return {cs: SEQ, lo, hi}; }
  // dG: white pinned at 0; blues for bound sites, oranges for sites that desorb; full range used
  let lo = Math.min(...v), hi = Math.max(...v);
  if (hi - lo < 0.1) { const m = 0.5 * (lo + hi); lo = m - 0.05; hi = m + 0.05; }
  if (lo >= 0) return {cs: SEQ, lo, hi, kind: "bound"};
  if (hi <= 0) return {cs: [[0, "#b84a12"], [0.5, "#eb8a5c"], [1, "#fbe3d4"]], lo, hi, kind: "desorb"};
  const z0 = -lo / (hi - lo);
  return {cs: [[0, "#b84a12"], [z0 / 2, "#eb8a5c"], [z0, "#efeeea"], [z0 + (1 - z0) / 2, "#6aa3e8"], [1, "#1c5cab"]], lo, hi, kind: "mixed"};
}
function label(a, av) {
  if (!av) return `${SYM[a]}${a + 1} · ${ROLE[a]}<br>not part of a desorbable ${EX.units.MX} unit`;
  const c = SD.classes[av.ci];
  return `${SYM[a]}${a + 1} · ${c.site}${c.facets && c.facets.length ? " " + c.facets.join(" ") : ""}<br>` +
         `${EX.units.MX} unit: Cd CN ${c.cation_native_cn}, ${c.cation_ligands} own Cl, Cl μ ${c.ligand_mu.join("/")}<br>` +
         `ΔE<sub>site</sub> (vacuum) = ${c.dE_eV.toFixed(3)} eV<br><b>shown: ${av.v.toFixed(3)} eV</b>`;
}
// icosphere (subdivision 2) for space-filling atoms
const ICO = (function () {
  const t = (1 + Math.sqrt(5)) / 2; let v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
    [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  let f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const norm = p => { const r = Math.hypot(...p); return p.map(x => x / r); };
  v = v.map(norm);
  for (let s = 0; s < 2; s++) { const mid = {}, nf = [];
    const m = (a, b) => { const k = a < b ? a + "_" + b : b + "_" + a; if (!(k in mid)) { mid[k] = v.length; v.push(norm(v[a].map((x, i) => (x + v[b][i]) / 2))); } return mid[k]; };
    for (const [a, b, c] of f) { const ab = m(a, b), bc = m(b, c), ca = m(c, a); nf.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]); }
    f = nf; }
  return {v, f};
})();
function meshFor(atoms, colourOf, text) {
  const x = [], y = [], z = [], i = [], j = [], k = [], c = [], tx = [];
  atoms.forEach((a, n) => { const r = R_VDW[SYM[a]] || 1.8, o = n * ICO.v.length;
    for (const p of ICO.v) { x.push(POS[a][0] + r * p[0]); y.push(POS[a][1] + r * p[1]); z.push(POS[a][2] + r * p[2]);
                             c.push(colourOf(a)); tx.push(text(a)); }
    for (const [p, q, w] of ICO.f) { i.push(o + p); j.push(o + q); k.push(o + w); } });
  return {x, y, z, i, j, k, c, tx};
}
const LIGHT = {ambient: 0.45, diffuse: 0.75, specular: 0.15, roughness: 0.6, fresnel: 0.1};
function draw3d(st, av, sc) {
  const on = [...Array(N).keys()].filter(a => av[a]), off = [...Array(N).keys()].filter(a => !av[a]);
  const m1 = meshFor(on, a => av[a].v, a => label(a, av[a])), m2 = meshFor(off, a => GREY[SYM[a]] || "#cccccc", a => label(a, null));
  const cb = st.colour === "dE" || !hasG ? "ΔE<sub>site</sub> (eV)" : "ΔG<sub>site</sub> (eV)";
  const traces = [
    {type: "mesh3d", x: m1.x, y: m1.y, z: m1.z, i: m1.i, j: m1.j, k: m1.k, intensity: m1.c, colorscale: sc.cs, cmin: sc.lo, cmax: sc.hi,
     colorbar: {title: {text: cb}, thickness: 12}, text: m1.tx, hovertemplate: "%{text}<extra></extra>", flatshading: false, lighting: LIGHT},
    {type: "mesh3d", x: m2.x, y: m2.y, z: m2.z, i: m2.i, j: m2.j, k: m2.k, vertexcolor: m2.c, text: m2.tx,
     hovertemplate: "%{text}<extra></extra>", lighting: LIGHT, showscale: false}];
  const ext = Math.max(...POS.map((p, a) => Math.hypot(...p) + (R_VDW[SYM[a]] || 1.8))) * 1.02;
  const ax = {visible: false, showspikes: false, range: [-ext, ext]};
  const prev = el("s_sites").layout && el("s_sites").layout.scene ? el("s_sites").layout.scene.camera : null;
  Plotly.react("s_sites", traces, {height: 560, margin: {l: 0, r: 0, t: 10, b: 0}, paper_bgcolor: "white",
    // orthographic: the camera distance does not zoom; the aspect ratio does
    scene: {xaxis: ax, yaxis: ax, zaxis: ax, aspectmode: "manual", aspectratio: {x: 1.75, y: 1.75, z: 1.75}, bgcolor: "white",
            camera: Object.assign({eye: {x: 1.25, y: 1.0, z: 0.8}}, prev || {}, {projection: {type: "orthographic"}})}},
    {responsive: true, displaylogo: false});
}
// 2D: direction from the centre as (theta, phi) = (azimuth, latitude), [001] at the poles
const DIR = POS.map(p => { const r = Math.hypot(...p) || 1; return [p[0] / r, p[1] / r, p[2] / r]; });
const ang = d => [Math.atan2(d[1], d[0]), Math.asin(Math.max(-1, Math.min(1, d[2])))];
const SURF = [...Array(N).keys()].filter(a => ROLE[a] !== "core");
const GT = [], GP = []; for (let i = 0; i <= 360; i++) GT.push(-Math.PI + i * 2 * Math.PI / 360); for (let j = 0; j <= 180; j++) GP.push(-Math.PI / 2 + j * Math.PI / 180);
const GDIR = GP.map(p => GT.map(t => [Math.cos(p) * Math.cos(t), Math.cos(p) * Math.sin(t), Math.sin(p)]));
function outlines() {
  const xs = [], ys = [], lab = [];
  for (const f of SD.faces) { const V = f.vertices;
    for (let e = 0; e < V.length; e++) { const a = V[e], b = V[(e + 1) % V.length]; let last = null;
      for (let s = 0; s <= 40; s++) { const p = a.map((x, i) => x + (b[i] - x) * s / 40), r = Math.hypot(...p), [t, ph] = ang(p.map(x => x / r));
        if (last !== null && Math.abs(t - last) > Math.PI) { xs.push(null); ys.push(null); }
        xs.push(t); ys.push(ph); last = t; }
      xs.push(null); ys.push(null); }
    const [t, ph] = ang(f.normal); lab.push({t, ph, text: f.label.replace(/[{}]/g, "")}); }
  return {xs, ys, lab};
}
const OUT = outlines();
function draw2d(st, av, sc) {
  const S = 0.13, on = SURF.filter(a => av[a]);
  const z = GDIR.map(row => row.map(g => { let w = 0, wv = 0, near = -1;
    for (const a of on) { const d = DIR[a], c = g[0] * d[0] + g[1] * d[1] + g[2] * d[2]; if (c > near) near = c;
      const th = Math.acos(Math.max(-1, Math.min(1, c))), ww = Math.exp(-0.5 * (th / S) ** 2); w += ww; wv += ww * av[a].v; }
    return near < Math.cos(0.32) ? null : wv / w; }));
  const pts = SURF.map(a => ang(DIR[a]));
  const cb = st.colour === "dE" || !hasG ? "ΔE<sub>site</sub> (eV)" : "ΔG<sub>site</sub> (eV)";
  Plotly.react("s_sites", [
    {type: "heatmap", x: GT, y: GP, z, colorscale: sc.cs, zmin: sc.lo, zmax: sc.hi, zsmooth: "best", hoverinfo: "skip",
     colorbar: {title: {text: cb}, thickness: 12}},
    {x: OUT.xs, y: OUT.ys, mode: "lines", line: {color: "white", width: 2}, hoverinfo: "skip", showlegend: false},
    {x: pts.map(p => p[0]), y: pts.map(p => p[1]), mode: "markers", showlegend: false,
     marker: {size: SURF.map(a => SYM[a] === "Cl" ? 6 : 8), color: SURF.map(a => av[a] ? "#0b0b0b" : "#8a8984"),
              symbol: SURF.map(a => SYM[a] === "Cl" ? "diamond" : "circle"), line: {color: "white", width: 1}},
     text: SURF.map(a => label(a, av[a])), hovertemplate: "%{text}<extra></extra>"},
    {x: OUT.lab.map(l => l.t), y: OUT.lab.map(l => l.ph), mode: "text", text: OUT.lab.map(l => l.text), showlegend: false,
     textfont: {size: 13, color: OUT.lab.map(l => {      // white on the field, dark on the bare (masked) background
       const i = Math.round((l.t + Math.PI) / (2 * Math.PI) * 360), j = Math.round((l.ph + Math.PI / 2) / Math.PI * 180);
       const v = z[j] ? z[j][i] : null;     // dark on the bare background and on light colours, white on dark ones
       return v === null || (v - sc.lo) / (sc.hi - sc.lo) < 0.55 ? "#0b0b0b" : "white"; })}, hoverinfo: "skip"}],
    {height: 520, margin: {l: 64, r: 16, t: 20, b: 52}, plot_bgcolor: "#efeeea", paper_bgcolor: "white",
     font: {family: "Helvetica, Arial, sans-serif", color: "#52514e", size: 12}, hovermode: "closest",
     xaxis: {title: {text: "θ (azimuth, rad)"}, range: [-Math.PI, Math.PI], zeroline: false, gridcolor: "rgba(0,0,0,0)"},
     yaxis: {title: {text: "φ (latitude, rad)"}, range: [-Math.PI / 2, Math.PI / 2], zeroline: false, gridcolor: "rgba(0,0,0,0)"}},
    {responsive: true, displaylogo: false});
}
let lastView = null;
makeControls("s_sites_ctl", ["view", "colour", "solv", "cmx", "T"], SPECS, [], st => {
  if (!hasG) st.colour = "dE";
  setDisabled(st, ["model", "eps", "cmx", "T"], st.colour === "dE");
  const cv = classValues(st), av = atomValues(cv), sc = scale(st, av.map(x => x && x.v));
  if (st.view !== lastView) { Plotly.purge("s_sites"); lastView = st.view; }
  (st.view === "3d" ? draw3d : draw2d)(st, av, sc);
});
})();
"""


def solution_section(ex: dict, sites: dict | None = None) -> tuple[str, str]:
    """(HTML, JS) of the per-dot 'Stability & ligands in solution' section, with controls per panel."""
    ma, mx = _fh(ex["units"]["MA"]), _fh(ex["units"]["MX"])
    n, m = ex["units"]["n"], ex["units"]["m"]

    def panel(pid, title, cap):
        return (f"<div class='panel'><h3>{title}</h3><div id='{pid}_ctl'></div><div id='{pid}'></div>"
                f"<div class='cap'>{cap}</div></div>")
    sites_panel = ""
    if sites and sites.get("classes"):
        sites_panel = (
            "<div class='panel wide'><h3>Binding-site map</h3><div id='s_sites_ctl'></div><div id='s_sites'></div>"
            f"<div class='cap'>Every Cd and Cl of a desorbable {mx} unit of the intact dot, coloured by the free energy of "
            f"removing that unit (the cheapest unit it belongs to): ΔG<sub>site</sub> = G<sup>sol</sup>(dot − unit) − "
            f"G<sup>sol</sup>(dot) + μ({mx}), each product relaxed with MACE-MH-1, with its own harmonic free energy and "
            f"Generalized Born solvation; or the electronic ΔE<sub>site</sub> = E(dot − unit) + E({mx}) − E(dot) in "
            f"vacuum. ΔG colours: blue, bound; orange, ΔG &lt; 0, the unit desorbs at these conditions (white is pinned at "
            f"0 when both occur). "
            f"Grey: Se and atoms of no desorbable unit. 3D: orthographic, van der Waals spheres. 2D: each surface atom "
            f"by its direction from the centre (azimuth θ, latitude φ, [001] at the poles), the field a Gaussian-weighted "
            f"average of the site values (σ = 7°), with the facet edges in white; circles Cd/Se, diamonds Cl "
            f"(black: part of a unit). Sites are labelled positions, so symmetry numbers are left out; their "
            f"multiplicity is what the map shows. The {len(sites['classes'])} symmetry-distinct units come from the "
            f"first step of the desorption search.</div></div>")
    html_ = f"""
<h2>Stability &amp; ligands in solution</h2>
<div class='note'>The vacuum free energies of the previous section plus an implicit solvent and finite concentrations:
μ<sub>i</sub> = G°<sub>i</sub>(T) + ΔG<sub>solv,i</sub>(ε) + k<sub>B</sub>T ln(c<sub>i</sub> / 1 M), bulk {ma} as a solid.
ΔG<sub>solv</sub>(ε) = −½ (1 − 1/ε) Σ<sub>ij</sub> q<sub>i</sub>q<sub>j</sub>/f<sub>GB</sub>(r<sub>ij</sub>): Generalized Born
electrostatics of the GFN2-xTB charges, one single point at each MACE-MH-1 gas-phase geometry (no re-optimisation in
solvent; non-electrostatic terms not included; ALPB offered for the named solvents where it converged for every species).
Each panel has its own controls, only for the quantities that enter it.</div>
<div class='grid'>
{panel("s_dec", "Decomposition free energy in solution",
       f"ΔG<sub>dec</sub> = [G<sup>sol</sup>(dot) − {n} G({ma}, bulk) − {m} μ({mx})] / {n}, per {ma}, against T. "
       f"{ma} is the bulk solid, so only the solvent and [{mx}] enter. Positive: the dot is metastable against "
       f"ripening into bulk {ma} with its ligands dissolved; more {mx} in solution stabilises it. Dotted: vacuum, 1 M.")}
{panel("s_bind", "Binding free energy and dissolution temperature",
       f"ΔG<sub>bind</sub> = [G<sup>sol</sup>(dot) − {n} μ({ma}) − {m} μ({mx})] / {n + m}, per unit, against T. The dot "
       f"assembles where it is negative and dissolves into monomers above T<sub>diss</sub> (ΔG<sub>bind</sub> = 0), where "
       f"the monomers' k<sub>B</sub>T ln c entropy overcomes the bonding; dilution lowers T<sub>diss</sub>. {SHIFT_NOTE}")}
{panel("s_ladder", f"Stepwise {mx} desorption at T",
       f"ΔG<sub>k</sub> = G<sup>sol</sup>(dot<sub>k</sub>) − G<sup>sol</sup>(dot<sub>k−1</sub>) + μ({mx}) along the lowest "
       f"removal path. Negative bars: that unit desorbs spontaneously at this T and [{mx}].")}
{sites_panel}
{panel("s_iso", f"{mx} desorption isotherm at T",
       f"Equilibrium number of {mx} units lost, ⟨k⟩, against [{mx}] in solution at the chosen T: a Boltzmann average "
       "over every configuration evaluated by the desorption search, each with its own solvation and vibrational free "
       "energy and its symmetry weight.")}
{panel("s_map", "Equilibrium ligand shell in solution",
       f"The same ⟨k⟩ over [{mx}] and T at once. Right: concentrated solution, shell intact; left: dilute, units desorb; "
       "contours at half-integer ⟨k⟩. The vacuum section has no such map: ligand exchange is an equilibrium with the "
       "solution.")}
</div>"""
    js = """
(function () {
const EX = %(ex)s, SPECS = %(specs)s, ALPB = %(alpb)s;
const AX = {gridcolor: "%(grid)s", zeroline: false, linecolor: "%(ink3)s", ticks: "outside", tickcolor: "%(ink3)s"};
const LAY = (xt, yt, extra = {}) => Object.assign({template: "plotly_white", height: 330, margin: {l: 64, r: 16, t: 36, b: 52},
  font: {family: "Helvetica, Arial, sans-serif", color: "%(ink2)s", size: 12}, hovermode: "x unified",
  legend: {orientation: "h", y: 1.02, yanchor: "bottom", x: 0}, xaxis: Object.assign({title: {text: xt}}, AX),
  yaxis: Object.assign({title: {text: yt}}, AX)}, extra);
const CFG = {responsive: true, displaylogo: false};
const ZERO = {type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: {color: "%(ink2)s", width: 1}};
const gas = curves(EX, null, 1, 1, 0);
const tIndex = t => Math.max(0, EX.T.indexOf(t));

makeControls("s_dec_ctl", ["solv", "cmx"], SPECS, ALPB, st => {
  const c = curves(EX, st.solv, 1, st.cmx, 0);
  Plotly.react("s_dec", [
    {x: c.T, y: c.dec, mode: "lines", name: "ΔG<sub>dec</sub>, this solution", line: {color: "%(c0)s", width: 2.5},
     hovertemplate: "%%{y:.3f} eV per %(ma)s<extra>solution</extra>"},
    {x: gas.T, y: gas.dec, mode: "lines", name: "vacuum, 1 M", line: {color: "%(ink3)s", width: 1.5, dash: "dot"},
     hovertemplate: "%%{y:.3f} eV<extra>vacuum, 1 M</extra>"}], LAY("T (K)", "eV per %(ma)s", {shapes: [ZERO]}), CFG);
});
makeControls("s_bind_ctl", ["solv", "cma", "cmx", "shift"], SPECS, ALPB, st => {
  const c = curves(EX, st.solv, st.cma, st.cmx, st.shift), shapes = [ZERO], ann = [];
  if (c.tDiss) { shapes.push({type: "line", x0: c.tDiss, x1: c.tDiss, yref: "paper", y0: 0, y1: 1, line: {color: "%(c1)s", width: 1.5, dash: "dash"}});
    ann.push({x: c.tDiss, yref: "paper", y: 1, text: "T<sub>diss</sub> = " + c.tDiss.toFixed(0) + " K", showarrow: false,
              xanchor: "left", yanchor: "top", xshift: 4, font: {color: "%(c1)s"}}); }
  Plotly.react("s_bind", [
    {x: c.T, y: c.bind, mode: "lines", name: "ΔG<sub>bind</sub>, this solution", line: {color: "%(c0)s", width: 2.5},
     hovertemplate: "%%{y:.3f} eV per unit<extra>solution</extra>"},
    {x: gas.T, y: gas.bind, mode: "lines", name: "vacuum, 1 M", line: {color: "%(ink3)s", width: 1.5, dash: "dot"},
     hovertemplate: "%%{y:.3f} eV<extra>vacuum, 1 M</extra>"}], LAY("T (K)", "eV per unit", {shapes, annotations: ann}), CFG);
});
makeControls("s_ladder_ctl", ["solv", "cmx", "T"], SPECS, ALPB, st => {
  const c = curves(EX, st.solv, 1, st.cmx, 0), ti = tIndex(st.T);
  const ks = c.ladder.map((_, k) => k + 1), lv = c.ladder.map(row => row[ti]);
  Plotly.react("s_ladder", [{x: ks, y: lv, type: "bar", name: "ΔG<sub>k</sub>", width: 0.5,
     marker: {color: lv.map(v => v < 0 ? "%(c1)s" : "%(c0)s")}, hovertemplate: "k = %%{x}<br>ΔG = %%{y:.3f} eV<extra></extra>"}],
    LAY("units removed k", "ΔG<sub>k</sub> (eV) at " + st.T + " K", {hovermode: "closest",
      xaxis: Object.assign({title: {text: "units removed k"}, dtick: 1}, AX), shapes: [ZERO]}), CFG);
});
const LC = []; for (let v = -20; v <= 0.001; v += 0.25) LC.push(v);
makeControls("s_iso_ctl", ["solv", "T"], SPECS, ALPB, st => {
  const ti = tIndex(st.T), kk = LC.map(v => meanRemovedAt(EX, st.solv, Math.pow(10, v), ti));
  Plotly.react("s_iso", [{x: LC, y: kk, mode: "lines", name: "⟨k⟩", line: {color: "%(c0)s", width: 2.5},
     hovertemplate: "c = 10<sup>%%{x:.2f}</sup> M: ⟨k⟩ = %%{y:.2f}<extra></extra>"}],
    LAY("log<sub>10</sub>([%(mx)s] / M)", "⟨k⟩ removed at " + st.T + " K", {hovermode: "closest",
      yaxis: Object.assign({title: {text: "⟨k⟩ removed at " + st.T + " K"}, range: [-0.3, EX.units.m + 0.3]}, AX)}), CFG);
});
const TM = EX.T.filter(t => t <= 900), KMAX = Math.max(...EX.dots.map(d => d.k));
makeControls("s_map_ctl", ["solv"], SPECS, ALPB, st => {
  const z = TM.map((t, i) => LC.map(v => meanRemovedAt(EX, st.solv, Math.pow(10, v), i)));
  Plotly.react("s_map", [{type: "heatmap", x: LC, y: TM, z, zmin: 0, zmax: KMAX,
     colorscale: [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]],
     colorbar: {title: {text: "⟨k⟩"}, thickness: 12},
     hovertemplate: "[%(mx)s] = 10<sup>%%{x:.2f}</sup> M, T = %%{y} K<br>⟨k⟩ = %%{z:.2f}<extra></extra>"},
    {type: "contour", x: LC, y: TM, z, showscale: false, contours: {coloring: "none", start: 0.5, end: KMAX, size: 1},
     line: {color: "white", width: 1}, hoverinfo: "skip"}],
    LAY("log<sub>10</sub>([%(mx)s] / M)", "T (K)", {hovermode: "closest", height: 360}), CFG);
});
})();
""" % {"ex": json.dumps(ex), "specs": json.dumps(_specs(ma, mx)), "alpb": json.dumps(list(ex["alpb_solvents"].items())),
       "grid": GRID, "ink3": INK3, "ink2": INK2, "c0": SITE[0], "c1": SITE[1],
       "ma": ex["units"]["MA"], "mx": ex["units"]["MX"]}
    if sites_panel:
        keep = {k: sites[k] for k in ("T", "dot", "classes", "faces")}
        keep["classes"] = [{k: c[k] for k in ("dE_eV", "G_prod_eV", "solv_inf_eV", "orbit", "site", "facets",
                                             "cation_native_cn", "cation_ligands", "ligand_mu") if k in c}
                           for c in sites["classes"]]
        js += (SITES_JS.replace("__EX__", json.dumps({"MX": ex["MX"], "units": ex["units"]}))
               .replace("__SITES__", json.dumps(keep)).replace("__SPECS__", json.dumps(_specs(ma, mx))))
    return html_, js


# --------------------------------------------------------------------------
# Synthesis page (several dots)
# --------------------------------------------------------------------------

SYN_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Synthesis thermodynamics</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-{plotly_js}.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{ font-family: Helvetica, Arial, sans-serif; background: #f8f9fa; margin: 0; padding: 20px; color: {ink}; }}
.wrap {{ max-width: 1500px; margin: 0 auto; background: white; padding: 24px 28px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,.05); }}
h1 {{ font-size: 21px; margin: 0 0 4px; }} h2 {{ font-size: 16px; margin: 24px 0 8px; padding-bottom: 6px; border-bottom: 1px solid {grid}; }}
.sub, .note {{ color: {ink2}; font-size: 13px; line-height: 1.55; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(460px, 100%), 1fr)); gap: 16px; margin-top: 14px; }}
.panel {{ border: 1px solid {grid}; border-radius: 8px; padding: 12px; }} .panel h3 {{ font-size: 14px; margin: 0 0 4px; }}
.cap {{ font-size: 12px; color: {ink2}; line-height: 1.55; margin-top: 6px; }}
{ctl_css}
@media (max-width: 520px) {{ body {{ padding: 8px; }} .wrap {{ padding: 14px; }} }}
</style></head><body><div class="wrap">
<h1>Synthesis thermodynamics: {title}</h1>
<div class="sub">Dots {species} in equilibrium with {ma} and {mx} monomers in solution, including every ligand-stripped
state found by the desorption search. Free energies: MACE-MH-1 (harmonic vibrations, rigid-rotor rotation, ideal solutes at
1 M) plus Generalized Born solvation of the GFN2-xTB charges. For each temperature the coupled equilibria
c<sub>i</sub> = exp(−ΔG°<sub>i</sub>/k<sub>B</sub>T) c<sub>{ma}</sub><sup>n<sub>i</sub></sup> c<sub>{mx}</sub><sup>m<sub>i</sub></sup>,
with mass balances C<sub>{ma}</sub> = c<sub>{ma}</sub> + Σ n<sub>i</sub> c<sub>i</sub> and
C<sub>{mx}</sub> = c<sub>{mx}</sub> + Σ m<sub>i</sub> c<sub>i</sub>, are solved for the free monomers (nested bisection in
log space). Each panel has its own controls, only for the quantities that enter it.</div>
<div class="grid">
<div class="panel"><h3>Yield against temperature</h3><div id="y_yield_ctl"></div><div id="y_yield"></div><div class="cap">
Fraction of all {ma} held in each dot family (summed over its ligand-stripped states), as free monomer and, if enabled,
as bulk. Dashed: the dissolution onset T<sub>diss</sub>, where the dots hold half of the {ma}, at these concentrations.</div></div>
<div class="panel"><h3>Populations against the [{mx}]/[{ma}] ratio</h3><div id="y_ratio_ctl"></div><div id="y_ratio"></div>
<div class="cap">At one temperature and total [{ma}]: how the ligand-to-monomer stoichiometry selects between the dot
families and the free monomers, i.e. whether growth stops at a small, atomically precise cluster or proceeds to the
larger dot.</div></div>
<div class="panel"><h3>Free monomers and supersaturation</h3><div id="y_mono_ctl"></div><div id="y_mono"></div>
<div class="cap">Free [{ma}] and [{mx}] in equilibrium, and the supersaturation S = c<sub>{ma}</sub> / c<sub>sat</sub>
against bulk {ma} (c<sub>sat</sub> = exp[−(G<sup>sol</sup><sub>{ma}</sub> − G<sub>bulk</sub>)/k<sub>B</sub>T]). S &gt; 1:
supersaturated with respect to bulk growth, the driving force of nucleation and ripening.</div></div>
<div class="panel"><h3>Ligand shell against temperature</h3><div id="y_cov_ctl"></div><div id="y_cov"></div>
<div class="cap">Mean number of {mx} units bound per dot of each family in the equilibrium population: how the shell
desorbs as T rises at these concentrations.</div></div>
</div>
<div class="note" style="margin-top:12px">{shift_note}</div>
<div class="note" style="margin-top:8px">Caveats: the monomers are the bare {ma} and {mx} molecules unless the precursor
stabilisation is set; implicit solvation omits specific coordination by L-type ligands. Only the dot sizes and stripped
states computed here are present, so the populations are relative to this set of species.</div>
</div>
<script>
{js_lib}
(function () {{
const EXPORTS = {exports};
const SP = speciesList(EXPORTS), FAMS = EXPORTS.map(e => e.formula), T = EXPORTS[0].T;
const COLORS = {colors};
const AX = {{gridcolor: "{grid}", zeroline: false, linecolor: "{ink3}", ticks: "outside", tickcolor: "{ink3}"}};
const LAY = (xt, yt, extra = {{}}) => Object.assign({{template: "plotly_white", height: 340, margin: {{l: 64, r: 16, t: 36, b: 52}},
  font: {{family: "Helvetica, Arial, sans-serif", color: "{ink2}", size: 12}}, hovermode: "x unified",
  legend: {{orientation: "h", y: 1.02, yanchor: "bottom", x: 0}}, xaxis: Object.assign({{title: {{text: xt}}}}, AX),
  yaxis: Object.assign({{title: {{text: yt}}}}, AX)}}, extra);
const CFG = {{responsive: true, displaylogo: false}};
const SPECS = {specs}, ALPB = {alpb};
const sub = s => s.replace(/(\d+)/g, "<sub>$1</sub>");
function scanT(sv, CA, CX, shift, bulk) {{
  const out = new Array(T.length);
  for (let i = T.length - 1; i >= 0; i--) {{ out[i] = equilibrium(EXPORTS, SP, i, sv, CA, CX, shift, bulk); }}
  return out;
}}
const FAMK = ["solv", "ca", "ratio", "shift", "bulk"];
makeControls("y_yield_ctl", FAMK, SPECS, ALPB, st => {{
  const res = scanT(st.solv, st.ca, st.ca * st.ratio, st.shift, st.bulk);
  const tr = FAMS.map((f, j) => ({{x: T, y: res.map(r => r.fam[f].frac), mode: "lines", name: sub(f),
     line: {{color: COLORS[j % COLORS.length], width: 2.5}}, hovertemplate: "%{{y:.3f}}<extra>" + sub(f) + "</extra>"}}));
  tr.push({{x: T, y: res.map(r => r.monomer), mode: "lines", name: "free {ma}", line: {{color: "{ink3}", width: 2, dash: "dot"}},
     hovertemplate: "%{{y:.3f}}<extra>free monomer</extra>"}});
  if (st.bulk) tr.push({{x: T, y: res.map(r => r.bulk), mode: "lines", name: "bulk {ma}", line: {{color: "{ink}", width: 2, dash: "dash"}},
     hovertemplate: "%{{y:.3f}}<extra>bulk</extra>"}});
  const dot = res.map(r => FAMS.reduce((s, f) => s + r.fam[f].frac, 0)), shapes = [], ann = [];
  for (let i = 0; i < T.length - 1; i++) if (dot[i] >= 0.5 && dot[i + 1] < 0.5) {{
    const td = T[i] + (dot[i] - 0.5) * (T[i + 1] - T[i]) / (dot[i] - dot[i + 1]);
    shapes.push({{type: "line", x0: td, x1: td, yref: "paper", y0: 0, y1: 1, line: {{color: "{ink2}", width: 1.2, dash: "dash"}}}});
    ann.push({{x: td, yref: "paper", y: 1, text: "T<sub>diss</sub> ≈ " + td.toFixed(0) + " K", showarrow: false, xanchor: "left", xshift: 4,
               yanchor: "top", font: {{color: "{ink2}"}}}}); break; }}
  Plotly.react("y_yield", tr, LAY("T (K)", "fraction of {ma}", {{shapes, annotations: ann,
     yaxis: Object.assign({{title: {{text: "fraction of {ma}"}}, range: [-0.02, 1.02]}}, AX)}}), CFG);
}});
makeControls("y_ratio_ctl", ["solv", "ca", "shift", "Tsyn", "bulk"], SPECS, ALPB, st => {{
  const ti = Math.max(0, T.indexOf(st.Tsyn)), rr = []; for (let v = -3; v <= 2.0001; v += 0.05) rr.push(v);
  const rs = rr.map(v => equilibrium(EXPORTS, SP, ti, st.solv, st.ca, st.ca * Math.pow(10, v), st.shift, st.bulk));
  const tr = FAMS.map((f, j) => ({{x: rr, y: rs.map(r => r.fam[f].frac), mode: "lines", name: sub(f),
     line: {{color: COLORS[j % COLORS.length], width: 2.5}}, hovertemplate: "%{{y:.3f}}<extra>" + sub(f) + "</extra>"}}));
  tr.push({{x: rr, y: rs.map(r => r.monomer), mode: "lines", name: "free {ma}", line: {{color: "{ink3}", width: 2, dash: "dot"}},
     hovertemplate: "%{{y:.3f}}<extra>free monomer</extra>"}});
  Plotly.react("y_ratio", tr, LAY("log<sub>10</sub>([{mx}] / [{ma}])", "fraction of {ma} at " + st.Tsyn + " K", {{
     yaxis: Object.assign({{title: {{text: "fraction of {ma} at " + st.Tsyn + " K"}}, range: [-0.02, 1.02]}}, AX)}}), CFG);
}});
makeControls("y_mono_ctl", FAMK, SPECS, ALPB, st => {{
  const res = scanT(st.solv, st.ca, st.ca * st.ratio, st.shift, st.bulk);
  Plotly.react("y_mono", [
    {{x: T, y: res.map(r => r.a / Math.LN10), mode: "lines", name: "log<sub>10</sub> c({ma})", line: {{color: COLORS[0], width: 2.5}}}},
    {{x: T, y: res.map(r => r.b / Math.LN10), mode: "lines", name: "log<sub>10</sub> c({mx})", line: {{color: COLORS[1], width: 2.5}}}},
    {{x: T, y: res.map(r => Math.log10(r.S)), mode: "lines", name: "log<sub>10</sub> S", line: {{color: "{ink}", width: 1.8, dash: "dash"}}}}],
    LAY("T (K)", "log<sub>10</sub> (c / M)  or  log<sub>10</sub> S"), CFG);
}});
makeControls("y_cov_ctl", FAMK, SPECS, ALPB, st => {{
  const res = scanT(st.solv, st.ca, st.ca * st.ratio, st.shift, st.bulk);
  Plotly.react("y_cov", FAMS.map((f, j) => {{ const m = EXPORTS[j].units.m;
     return {{x: T, y: res.map(r => r.fam[f].meanK === null ? null : m - r.fam[f].meanK), mode: "lines", name: sub(f),
              line: {{color: COLORS[j % COLORS.length], width: 2.5}}, hovertemplate: "%{{y:.2f}} of " + m + "<extra>" + sub(f) + "</extra>"}}; }}),
    LAY("T (K)", "{mx} bound per dot"), CFG);
}});
}})();
</script></body></html>
"""


def synthesis_page(exports: list) -> str:
    from plotly.offline import get_plotlyjs_version
    ex0 = exports[0]
    ma, mx = ex0["units"]["MA"], ex0["units"]["MX"]
    return SYN_PAGE.format(
        plotly_js=get_plotlyjs_version(), ink=INK, ink2=INK2, ink3=INK3, grid=GRID, ctl_css=CONTROLS_CSS,
        title=", ".join(_fh(e["formula"]) for e in exports), species=", ".join(_fh(e["formula"]) for e in exports),
        ma=_fh(ma), mx=_fh(mx), js_lib=JS_LIB, exports=json.dumps(exports), colors=json.dumps(SITE),
        specs=json.dumps(_specs(_fh(ma), _fh(mx))), alpb=json.dumps(list(ex0["alpb_solvents"].items())),
        shift_note=SHIFT_NOTE)
