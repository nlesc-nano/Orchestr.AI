"""Standalone plotting utilities for pool active-learning diagnostics CSV files."""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from orchestr_ai.postprocessing.metrics import _split_atom_vectors


def _to_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool_array(values):
    return np.array([str(v).strip().lower() in {"1", "true", "yes"} for v in values], dtype=bool)


def read_al_diagnostics_csv(path):
    """Read an AL diagnostics CSV with leading '# key = value' metadata comments."""
    metadata = {"thresholds": defaultdict(dict), "global_thresholds": {}}
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        data_lines = []
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("#"):
                text = stripped.lstrip("#").strip()
                if "=" not in text:
                    continue
                key, value = [part.strip() for part in text.rsplit("=", 1)]
                if key.startswith("threshold[state=") and "]." in key:
                    state = key.split("threshold[state=", 1)[1].split("]", 1)[0]
                    threshold_key = key.split("].", 1)[1]
                    metadata["thresholds"][state][threshold_key] = _to_float(value)
                elif key.startswith(("thr_", "hard_", "train_")):
                    metadata["global_thresholds"][key] = _to_float(value)
                else:
                    metadata[key] = value
                continue
            if stripped:
                data_lines.append(line)
        if data_lines:
            header_line = data_lines[0].strip()
            if "," in header_line:
                rows = list(csv.DictReader(data_lines))
            else:
                headers = header_line.split()
                for line in data_lines[1:]:
                    parts = line.strip().split()
                    if len(parts) >= len(headers):
                        rows.append(dict(zip(headers, parts)))
    metadata["thresholds"] = dict(metadata["thresholds"])
    return rows, metadata


def _parse_per_atom_uncertainties(path):
    """Return (elements_per_frame, sFnorm_per_frame_meV, sigma_E_per_frame_meV, muFnorm_per_frame_meV)."""
    with open(path) as f:
        lines = f.readlines()

    elements_all = []
    sFnorm_all = []
    sigma_E_list = []
    muFnorm_all = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue

        header = lines[i + 1].strip()
        sigma_E = 0.0
        for part in header.split():
            if part.startswith("sigma_E="):
                try:
                    sigma_E = float(part.split("=")[1])
                except ValueError:
                    pass

        elements = []
        norms = []
        muF_norms = []
        for j in range(n_atoms):
            parts = lines[i + 2 + j].split()
            if len(parts) >= 8:
                elements.append(parts[0])
                norms.append(float(parts[7]))
                if len(parts) >= 9:
                    muF_norms.append(float(parts[8]))
        if muF_norms and len(muF_norms) == len(norms):
            muFnorm_all.append(muF_norms)
        else:
            muFnorm_all.append(None)

        elements_all.append(elements)
        # Convert eV -> meV (×1000)
        sFnorm_all.append([v * 1000.0 for v in norms])
        sigma_E_list.append(sigma_E * 1000.0)

        i += 2 + n_atoms

    has_muF = any(x is not None for x in muFnorm_all)
    if not has_muF:
        muFnorm_all = None
    elif all(x is not None for x in muFnorm_all):
        muFnorm_all = [[v * 1000.0 for v in x] for x in muFnorm_all]
    else:
        muFnorm_all = None

    return elements_all, sFnorm_all, sigma_E_list, muFnorm_all


def _thresholds_for_state(metadata, state):
    thresholds = dict(metadata.get("global_thresholds", {}))
    # Normalize state-specific merges (e.g. map triplet_reconstructed to triplet if needed)
    state_thr = metadata.get("thresholds", {}).get(state, {})
    if not state_thr and "reconstructed" in str(state):
        alt_state = str(state).replace("_reconstructed", "")
        state_thr = metadata.get("thresholds", {}).get(alt_state, {})
    for k, v in state_thr.items():
        if np.isfinite(v):
            thresholds[k] = v
    return thresholds


def _get_mapped(row, key):
    mapping = {
        "sigma_E_atom": ["sigma_E_atom", "σE_atom", "sE_atom", "sigmaE_atom"],
        "sigma_F_max": ["sigma_F_max", "σF_max", "sF_max", "sigmaF_max"],
        "sigma_F_mean": ["sigma_F_mean", "σF_mean", "sF_mean", "sigmaF_mean"],
        "gamma_gate": ["gamma_gate", "γ_gate"],
        "geom_ok": ["geom_ok", "rdf_ok"],
        "caps_ok": ["caps_ok", "pass_caps"],
        "cal_ok": ["cal_ok", "cal_support"],
    }
    candidates = mapping.get(key, [key])
    for c in candidates:
        if c in row:
            return row[c]
    return row.get(key)


def _state_arrays(rows):
    keys_float = [
        "idx", "pool_row", "n_atoms", "gamma0", "dM", "Dgain", "raw_score",
        "E_pred", "E_pred_atom", "sigma_E", "sigma_E_atom", "sigma_F_max",
        "sigma_F_mean", "Eabs_exp", "Fabs_mean", "Fabs_max", "Fmax", "Fmean",
        "env_margin", "rel_unc", "rel_weight",
    ]
    keys_bool = ["geom_ok", "caps_ok", "force_inf", "gamma_gate", "cal_ok", "ood", "selected", "shortlist"]
    out = {key: np.array([_to_float(_get_mapped(row, key)) for row in rows], dtype=float) for key in keys_float}
    for key in keys_bool:
        out[key] = _to_bool_array([_get_mapped(row, key) or "0" for row in rows])
    order = np.argsort(out["idx"])
    for key, values in out.items():
        out[key] = values[order]
    return out


def _rolling_mean(x, y, window=50, drop_first=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if drop_first and np.any(mask):
        mask &= x != np.nanmin(x[mask])
    x_valid = x[mask]
    y_valid = y[mask]
    if y_valid.size == 0:
        return x_valid, y_valid
    half = max(1, int(window)) // 2
    smooth = np.empty_like(y_valid, dtype=float)
    for i in range(y_valid.size):
        smooth[i] = np.nanmean(y_valid[max(0, i - half): min(y_valid.size, i + half + 1)])
    return x_valid, smooth


def _energy_reference(arrays):
    energy = np.asarray(arrays["E_pred"], dtype=float)
    stable = arrays["geom_ok"] & arrays["caps_ok"] & np.isfinite(energy)
    idx = np.where(stable)[0][:10]
    if idx.size:
        return float(np.nanmedian(energy[idx])), "median first 10 stable frames"
    finite = np.where(np.isfinite(energy))[0]
    if finite.size:
        return float(energy[finite[0]]), "first finite frame"
    return 0.0, "0"


def _realistic_mask(arrays, y):
    finite = np.isfinite(y)
    if finite.shape != arrays["geom_ok"].shape:
        return finite
    return arrays["geom_ok"] & arrays["caps_ok"] & finite


def _robust_limits(y, arrays, *, floor_zero=False, threshold_values=(), pad=0.12):
    y = np.asarray(y, dtype=float)
    mask = _realistic_mask(arrays, y)
    vals = y[mask] if mask.shape == y.shape else y[np.isfinite(y)]
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        vals = y[np.isfinite(y)]
    if vals.size == 0:
        return None
    lo, hi = np.nanpercentile(vals, [1.0, 99.0])
    finite_thr = [float(v) for v in threshold_values if np.isfinite(v)]
    if finite_thr:
        lo = min(lo, min(finite_thr))
        hi = max(hi, max(finite_thr))
    if floor_zero:
        lo = 0.0
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        delta = max(abs(hi), 1.0) * 0.1
        lo -= delta
        hi += delta
    span = hi - lo
    return lo - span * pad, hi + span * pad


def _apply_robust_ylim(ax, y, arrays, *, floor_zero=False, threshold_values=()):
    limits = _robust_limits(y, arrays, floor_zero=floor_zero, threshold_values=threshold_values)
    if limits is None:
        return None
    ax.set_ylim(*limits)
    finite = np.isfinite(y)
    clipped = int(np.sum(finite & ((y < limits[0]) | (y > limits[1]))))
    if clipped:
        ax.text(
            0.995, 0.94, f"{clipped} outlier(s) outside y-range",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
        )
    return limits


def _realistic_trace_limits(y, arrays, *, floor_zero=False, pad=0.05, threshold_values=()):
    y = np.asarray(y, dtype=float)
    mask = _realistic_mask(arrays, y)
    vals = y[mask] if mask.shape == y.shape else y[np.isfinite(y)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = y[np.isfinite(y)]
    if vals.size == 0:
        return None
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    for v in threshold_values:
        if np.isfinite(v):
            hi = max(hi, float(v))
    if floor_zero:
        lo = 0.0
        hi_limit = hi * 1.05 if hi > 0.0 else 0.05
        return 0.0, hi_limit
    else:
        if hi > 0.0:
            hi_limit = hi * 1.05
        elif hi < 0.0:
            hi_limit = hi * 0.95
        else:
            hi_limit = 0.05
        span = hi_limit - lo
        if span <= 0:
            span = 1.0
        lo_limit = lo - span * pad
        return lo_limit, hi_limit


def _apply_trace_ylim(ax, y, arrays, *, floor_zero=False, threshold_values=()):
    limits = _realistic_trace_limits(y, arrays, floor_zero=floor_zero, threshold_values=threshold_values)
    if limits is None:
        return None
    ax.set_ylim(*limits)
    finite = np.isfinite(y)
    clipped = int(np.sum(finite & ((y < limits[0]) | (y > limits[1]))))
    if clipped:
        ax.text(
            0.995, 0.94, f"{clipped} outlier(s) outside y-range",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
        )
    return limits


def _plotly_trace_range(fig, row, y, arrays, *, floor_zero=False, threshold_values=()):
    limits = _realistic_trace_limits(y, arrays, floor_zero=floor_zero, threshold_values=threshold_values)
    if limits is not None:
        fig.update_yaxes(range=list(limits), row=row, col=1)


def _apply_cap_ylim(ax, y, cap, *, pad=0.05):
    if not np.isfinite(cap) or cap <= 0:
        return None
    upper = cap * (1.0 + pad)
    ax.set_ylim(0.0, upper)
    finite = np.isfinite(y)
    clipped = int(np.sum(finite & (y > upper)))
    if clipped:
        ax.text(
            0.995, 0.94, f"{clipped} above upper cap",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
        )
    return 0.0, upper


def _in_cap_mask(arrays, thresholds):
    mask = arrays["geom_ok"] & arrays["caps_ok"]
    checks = (
        ("sigma_E_atom", "thr_sigma_E_hi_eff", 1.0),
        ("sigma_F_max", "thr_sigma_F_hi_eff", 1.0),
        ("sigma_F_mean", "thr_sigma_Fmean_hi_eff", 1.0),
        ("Fmax", "train_Fmax_hard_cap", 1.0),
    )
    for arr_key, thr_key, scale in checks:
        cap = thresholds.get(thr_key, np.nan)
        vals = arrays[arr_key]
        if np.isfinite(cap):
            mask &= np.isfinite(vals) & (vals <= cap * scale)
    return mask


def _apply_score_ylim(ax, score, arrays, thresholds):
    mask = _in_cap_mask(arrays, thresholds) & np.isfinite(score)
    vals = score[mask]
    if vals.size == 0:
        mask_realistic = arrays["geom_ok"] & arrays["caps_ok"] & np.isfinite(score)
        vals = score[mask_realistic]
    if vals.size == 0:
        vals = score[np.isfinite(score)]
    ymax = float(np.nanmax(vals)) if vals.size else 1.0
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    limit = ymax * 1.05
    ax.set_ylim(0.0, limit)
    clipped = int(np.sum(np.isfinite(score) & (score > limit)))
    if clipped:
        ax.text(
            0.995, 0.94, f"{clipped} outlier(s) outside y-range",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
        )
    return 0.0, limit


def _threshold_values(thresholds, keys, scale=1.0):
    return [thresholds.get(key, np.nan) * scale for key in keys]


def _threshold_lines(ax, thresholds, keys, scale=1.0):
    for key, color, label in keys:
        value = thresholds.get(key, np.nan)
        if np.isfinite(value):
            ax.axhline(value * scale, ls="--", lw=1.0, color=color, alpha=0.85, label=label)


def _mark_events(ax, x, y, arrays, *, show_label=False):
    geom_bad = ~arrays["geom_ok"]
    caps_bad = arrays["geom_ok"] & ~arrays["caps_ok"]
    shortlist = arrays["shortlist"]
    ood = arrays["ood"]
    uncertain = arrays["force_inf"]
    if np.any(uncertain):
        ax.scatter(x[uncertain], y[uncertain], marker="s", s=14, color="#d95f02", alpha=0.45, label="Uncertain" if show_label else None)
    if np.any(ood):
        ax.scatter(x[ood], y[ood], marker="^", s=18, color="#7b3294", alpha=0.55, label="OOD" if show_label else None)
    if np.any(caps_bad):
        ax.scatter(x[caps_bad], y[caps_bad], marker="x", s=24, color="#e66101", alpha=0.8, label="Failed caps" if show_label else None)
    if np.any(geom_bad):
        ax.scatter(x[geom_bad], y[geom_bad], marker="x", s=20, color="#b2182b", alpha=0.45, label="Failed geom" if show_label else None)
    if np.any(shortlist):
        ax.scatter(x[shortlist], y[shortlist], marker="o", s=28, color="black", alpha=0.9, label="Shortlist" if show_label else None, zorder=5)


def _plot_status_track(ax, x, arrays):
    events = [
        ("shortlist", arrays["shortlist"], 5, "black"),
        ("uncertain", arrays["force_inf"], 4, "#d95f02"),
        ("OOD", arrays["ood"], 3, "#7b3294"),
        ("failed caps", arrays["geom_ok"] & ~arrays["caps_ok"], 2, "#e66101"),
        ("failed geom", ~arrays["geom_ok"], 1, "#b2182b"),
    ]
    for label, mask, ypos, color in events:
        if np.any(mask):
            ax.scatter(x[mask], np.full(np.sum(mask), ypos), s=30, color=color, label=label)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["failed geom", "failed caps", "OOD", "uncertain", "shortlist"])
    ax.set_ylim(0.4, 5.6)
    ax.legend(loc="upper right", fontsize=8, ncol=5)


def _plotly_imports():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except ImportError:
        print("[AL Plot] Plotly is not installed; skipping interactive HTML plots.")
        return None, None


def _add_plotly_thresholds(fig, row, x, thresholds, keys, scale=1.0):
    if not len(x):
        return
    x0 = float(np.nanmin(x))
    x1 = float(np.nanmax(x))
    for key, color, label in keys:
        value = thresholds.get(key, np.nan)
        if np.isfinite(value):
            fig.add_scatter(
                x=[x0, x1], y=[value * scale, value * scale], mode="lines",
                line=dict(color=color, dash="dash", width=1),
                name=f"{label}: {value * scale:.4g}", row=row, col=1,
            )


def _add_plotly_events(fig, row, x, y, arrays, go):
    for label, mask, symbol, color in (
        ("OOD", arrays["ood"], "triangle-up", "#7b3294"),
        ("Failed caps", arrays["geom_ok"] & ~arrays["caps_ok"], "x", "#e66101"),
        ("Failed geom", ~arrays["geom_ok"], "x", "#b2182b"),
        ("Uncertain", arrays["force_inf"], "square", "#d95f02"),
        ("Shortlist", arrays["shortlist"], "circle", "black"),
    ):
        if np.any(mask):
            fig.add_trace(
                go.Scatter(
                    x=x[mask], y=y[mask], mode="markers", name=label,
                    marker=dict(color=color, symbol=symbol, size=8),
                    hovertemplate="frame=%{x}<br>value=%{y:.5g}<extra>" + label + "</extra>",
                ), row=row, col=1,
            )


def _plotly_range(fig, row, y, arrays, *, floor_zero=False, threshold_values=()):
    limits = _robust_limits(y, arrays, floor_zero=floor_zero, threshold_values=threshold_values)
    if limits is not None:
        fig.update_yaxes(range=list(limits), row=row, col=1)


def _plotly_cap_range(fig, row, cap, *, pad=0.05):
    if np.isfinite(cap) and cap > 0:
        fig.update_yaxes(range=[0.0, cap * (1.0 + pad)], row=row, col=1)


def _plotly_score_range(fig, row, score, arrays, thresholds):
    mask = _in_cap_mask(arrays, thresholds) & np.isfinite(score)
    vals = score[mask]
    if vals.size == 0:
        mask_realistic = arrays["geom_ok"] & arrays["caps_ok"] & np.isfinite(score)
        vals = score[mask_realistic]
    if vals.size == 0:
        vals = score[np.isfinite(score)]
    ymax = float(np.nanmax(vals)) if vals.size else 1.0
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    fig.update_yaxes(range=[0.0, ymax * 1.05], row=row, col=1)


def plot_al_trace_state(rows, metadata, state, out_dir="al_plots", window=50, drop_first=True, dpi=300):
    arrays = _state_arrays(rows)
    thr = _thresholds_for_state(metadata, state)
    x = arrays["idx"]
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(11.0, 10.5), sharex=True, constrained_layout=True)
    fig.suptitle(f"Pool AL physical trace: {state}", fontsize=14, fontweight="bold")

    e_ref, e_ref_label = _energy_reference(arrays)
    dE = arrays["E_pred"] - e_ref
    sE = arrays["sigma_E"]
    xs, dE_smooth = _rolling_mean(x, dE, window=window, drop_first=drop_first)
    _, sE_smooth = _rolling_mean(x, sE, window=window, drop_first=drop_first)
    axes[0].plot(x, dE, lw=0.7, color="#4d4d4d", alpha=0.28, label="raw")
    axes[0].plot(xs, dE_smooth, lw=1.9, color="#2166ac", label=f"{window}-frame running mean")
    band = 2.0 * sE_smooth
    if band.size == dE_smooth.size:
        axes[0].fill_between(xs, dE_smooth - band, dE_smooth + band, color="#67a9cf", alpha=0.45, label="± 2σE")
        axes[0].plot(xs, dE_smooth + band, lw=0.8, color="#67a9cf", alpha=0.75)
        axes[0].plot(xs, dE_smooth - band, lw=0.8, color="#67a9cf", alpha=0.75)
        dE_limits = np.concatenate([dE, dE_smooth + band, dE_smooth - band])
    else:
        dE_limits = dE
    _mark_events(axes[0], x, dE, arrays, show_label=True)
    _apply_trace_ylim(axes[0], dE, arrays, floor_zero=False)
    axes[0].set_ylabel("ΔE (eV)")
    axes[0].set_title(f"Energy relative to {e_ref_label}: E - {e_ref:.6g} eV", fontsize=10)
    axes[0].legend(loc="best", fontsize=8, ncol=4)

    axes[1].plot(x, arrays["Fmax"], lw=1.0, color="#1b7837")
    _mark_events(axes[1], x, arrays["Fmax"], arrays)
    _apply_trace_ylim(axes[1], arrays["Fmax"], arrays, floor_zero=True)
    axes[1].set_ylabel("Fmax (eV/Å)")

    axes[2].plot(x, arrays["Fmean"], lw=1.0, color="#5aae61")
    _mark_events(axes[2], x, arrays["Fmean"], arrays)
    _apply_trace_ylim(axes[2], arrays["Fmean"], arrays, floor_zero=True)
    axes[2].set_ylabel("Fmean (eV/Å)")

    _plot_status_track(axes[3], x, arrays)
    axes[3].set_ylabel("AL status")
    axes[3].set_xlabel("Pool frame index")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    out_path = os.path.join(out_dir, f"al_trace_{state}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[AL Plot] Saved {out_path}")
    return out_path


def plot_al_uncertainty_state(rows, metadata, state, out_dir="al_plots", dpi=300):
    arrays = _state_arrays(rows)
    thr = _thresholds_for_state(metadata, state)
    x = arrays["idx"]
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(6, 1, figsize=(11.5, 15.0), sharex=True, constrained_layout=True)
    fig.suptitle(f"Pool AL uncertainty/acquisition: {state}", fontsize=14, fontweight="bold")

    alpha = thr.get("envelope_alpha", np.nan)

    sigma_e_mev = arrays["sigma_E_atom"] * 1000.0
    e_keys = [("thr_sigma_E_low", "#636363", "low"), ("thr_sigma_E_hi_eff", "#969696", "upper cap"), ("hard_sigma_E_atom_min", "#b2182b", "hard floor")]
    axes[0].plot(x, sigma_e_mev, lw=1.0, color="#2166ac")
    _threshold_lines(axes[0], thr, e_keys, scale=1000.0)
    _mark_events(axes[0], x, sigma_e_mev, arrays)
    _apply_cap_ylim(axes[0], sigma_e_mev, thr.get("thr_sigma_E_hi_eff", np.nan) * 1000.0)
    axes[0].set_ylabel("σE/atom (meV)")

    sigma_fmax_mev = arrays["sigma_F_max"] * 1000.0
    fmax_keys = [("thr_sigma_F", "#636363", "low"), ("thr_sigma_F_hi_eff", "#969696", "upper cap"), ("hard_sigma_F_max_min", "#b2182b", "hard floor")]
    axes[1].plot(x, sigma_fmax_mev, lw=1.0, color="#762a83")
    _threshold_lines(axes[1], thr, fmax_keys, scale=1000.0)
    _mark_events(axes[1], x, sigma_fmax_mev, arrays)
    _apply_cap_ylim(axes[1], sigma_fmax_mev, thr.get("thr_sigma_F_hi_eff", np.nan) * 1000.0)
    axes[1].set_ylabel(r"$\sigma F_{\max}$ (meV/Å)")

    sigma_fmean_mev = arrays["sigma_F_mean"] * 1000.0
    fmean_keys = [("thr_sigma_Fmean", "#636363", "low"), ("thr_sigma_Fmean_hi_eff", "#969696", "upper cap"), ("hard_sigma_F_mean_min", "#b2182b", "hard floor")]
    axes[2].plot(x, sigma_fmean_mev, lw=1.0, color="#af8dc3")
    _threshold_lines(axes[2], thr, fmean_keys, scale=1000.0)
    _mark_events(axes[2], x, sigma_fmean_mev, arrays)
    _apply_cap_ylim(axes[2], sigma_fmean_mev, thr.get("thr_sigma_Fmean_hi_eff", np.nan) * 1000.0)
    axes[2].set_ylabel(r"$\sigma F_{\mathrm{mean}}$ (meV/Å)")

    rel = np.asarray(arrays["rel_unc"], dtype=float)
    axes[3].plot(x, rel, lw=1.0, color="#8e0152")
    if np.isfinite(alpha):
        axes[3].axhline(alpha, ls="--", lw=1.2, color="#c51b7d", alpha=0.9, label=f"α = {alpha:.3f}")
    _mark_events(axes[3], x, rel, arrays)
    _apply_trace_ylim(axes[3], rel, arrays, floor_zero=True,
                         threshold_values=[alpha] if np.isfinite(alpha) else [])
    axes[3].set_ylabel("Relative force uncertainty\nγ = σF / (‖F‖ + ε)")
    # axes[3].set_title("Relative force uncertainty (ε = Atol/α; ranking weight = clip(γ/α, 0, 1))", fontsize=10)
    # if np.isfinite(alpha):
    #     axes[3].legend(loc="best", fontsize=8)

    axes[4].plot(x, arrays["raw_score"], lw=1.0, color="#8c510a")
    _mark_events(axes[4], x, arrays["raw_score"], arrays)
    _apply_score_ylim(axes[4], arrays["raw_score"], arrays, thr)
    axes[4].set_ylabel("acquisition score")
    axes[4].set_title("Acquisition score = leverage/novelty ranking (× relative-uncertainty weight)", fontsize=10)

    _plot_status_track(axes[5], x, arrays)
    axes[5].set_ylabel("AL status")
    axes[5].set_xlabel("Pool frame index")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    out_path = os.path.join(out_dir, f"al_uncertainty_{state}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[AL Plot] Saved {out_path}")
    return out_path


def plot_al_trace_state_interactive(rows, metadata, state, out_dir="al_plots", window=50, drop_first=True):
    go, make_subplots = _plotly_imports()
    if go is None:
        return None
    arrays = _state_arrays(rows)
    thr = _thresholds_for_state(metadata, state)
    x = arrays["idx"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055, subplot_titles=("Relative energy", "Maximum force", "Mean force", "AL status"))
    e_ref, e_ref_label = _energy_reference(arrays)
    dE = arrays["E_pred"] - e_ref
    sE = arrays["sigma_E"]
    xs, dE_smooth = _rolling_mean(x, dE, window=window, drop_first=drop_first)
    _, sE_smooth = _rolling_mean(x, sE, window=window, drop_first=drop_first)
    fig.add_trace(go.Scatter(x=x, y=dE, mode="lines", name="ΔE raw", line=dict(color="#777", width=1), opacity=0.45), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=dE_smooth, mode="lines", name=f"ΔE {window}-frame mean", line=dict(color="#2166ac", width=2)), row=1, col=1)
    band = 2.0 * sE_smooth
    if band.size == dE_smooth.size:
        fig.add_trace(go.Scatter(x=xs, y=dE_smooth + band, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=dE_smooth - band, mode="lines", fill="tonexty", fillcolor="rgba(103,169,207,0.45)", line=dict(width=0), name="± 2σE", hoverinfo="skip"), row=1, col=1)
        dE_limits = np.concatenate([dE, dE_smooth + band, dE_smooth - band])
    else:
        dE_limits = dE
    _add_plotly_events(fig, 1, x, dE, arrays, go)
    _plotly_trace_range(fig, 1, dE, arrays, floor_zero=False)

    fig.add_trace(go.Scatter(x=x, y=arrays["Fmax"], mode="lines", name="Fmax", line=dict(color="#1b7837")), row=2, col=1)
    _add_plotly_events(fig, 2, x, arrays["Fmax"], arrays, go)
    _plotly_trace_range(fig, 2, arrays["Fmax"], arrays, floor_zero=True)

    fig.add_trace(go.Scatter(x=x, y=arrays["Fmean"], mode="lines", name="Fmean", line=dict(color="#5aae61")), row=3, col=1)
    _add_plotly_events(fig, 3, x, arrays["Fmean"], arrays, go)
    _plotly_trace_range(fig, 3, arrays["Fmean"], arrays, floor_zero=True)

    for label, mask, ypos, color in (("Shortlist", arrays["shortlist"], 5, "black"), ("Uncertain", arrays["force_inf"], 4, "#d95f02"), ("OOD", arrays["ood"], 3, "#7b3294"), ("Failed caps", arrays["geom_ok"] & ~arrays["caps_ok"], 2, "#e66101"), ("Failed geom", ~arrays["geom_ok"], 1, "#b2182b")):
        if np.any(mask):
            fig.add_trace(go.Scatter(x=x[mask], y=np.full(np.sum(mask), ypos), mode="markers", name=label, marker=dict(color=color, size=8)), row=4, col=1)
    fig.update_yaxes(title_text="ΔE (eV)", row=1, col=1)
    fig.update_yaxes(title_text="Fmax (eV/Å)", row=2, col=1)
    fig.update_yaxes(title_text="Fmean (eV/Å)", row=3, col=1)
    fig.update_yaxes(title_text="AL status", tickmode="array", tickvals=[1, 2, 3, 4, 5], ticktext=["failed geom", "failed caps", "OOD", "uncertain", "shortlist"], row=4, col=1)
    fig.update_xaxes(title_text="Pool frame index", row=4, col=1)
    fig.update_layout(title=f"Pool AL physical trace: {state}<br><sup>Energy reference: {e_ref_label}, E_ref={e_ref:.6g} eV</sup>", height=900, width=1150, hovermode="x unified", template="plotly_white")
    out_path = os.path.join(out_dir, f"al_trace_{state}.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[AL Plot] Saved {out_path}")
    return out_path


def plot_al_uncertainty_state_interactive(rows, metadata, state, out_dir="al_plots"):
    go, make_subplots = _plotly_imports()
    if go is None:
        return None
    arrays = _state_arrays(rows)
    thr = _thresholds_for_state(metadata, state)
    x = arrays["idx"]
    alpha = thr.get("envelope_alpha", np.nan)
    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.038,
        subplot_titles=(
            "Energy uncertainty", "Maximum force uncertainty", "Mean force uncertainty",
            "Relative force uncertainty", "Acquisition score", "AL status",
        ),
    )
    panels = [
        (1, arrays["sigma_E_atom"] * 1000.0, "σE/atom", "#2166ac", [("thr_sigma_E_low", "#636363", "σE low"), ("thr_sigma_E_hi_eff", "#969696", "σE upper cap"), ("hard_sigma_E_atom_min", "#b2182b", "σE hard floor")], 1000.0, "σE/atom (meV)", "thr_sigma_E_hi_eff"),
        (2, arrays["sigma_F_max"] * 1000.0, "σF<sub>max</sub>", "#762a83", [("thr_sigma_F", "#636363", "σFmax low"), ("thr_sigma_F_hi_eff", "#969696", "σFmax upper cap"), ("hard_sigma_F_max_min", "#b2182b", "σFmax hard floor")], 1000.0, "σF<sub>max</sub> (meV/Å)", "thr_sigma_F_hi_eff"),
        (3, arrays["sigma_F_mean"] * 1000.0, "σF<sub>mean</sub>", "#af8dc3", [("thr_sigma_Fmean", "#636363", "σFmean low"), ("thr_sigma_Fmean_hi_eff", "#969696", "σFmean upper cap"), ("hard_sigma_F_mean_min", "#b2182b", "σFmean hard floor")], 1000.0, "σF<sub>mean</sub> (meV/Å)", "thr_sigma_Fmean_hi_eff"),
    ]
    for row, y, name, color, keys, scale, ylabel, cap_key in panels:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color)), row=row, col=1)
        _add_plotly_thresholds(fig, row, x, thr, keys, scale=scale)
        _add_plotly_events(fig, row, x, y, arrays, go)
        _plotly_cap_range(fig, row, thr.get(cap_key, np.nan) * scale)
        fig.update_yaxes(title_text=ylabel, row=row, col=1)

    rel = np.asarray(arrays["rel_unc"], dtype=float)
    fig.add_trace(go.Scatter(x=x, y=rel, mode="lines", name="γ = σF/(‖F‖+ε)", line=dict(color="#8e0152")), row=4, col=1)
    _add_plotly_thresholds(fig, 4, x, thr, [("envelope_alpha", "#c51b7d", "α")], scale=1.0)
    _add_plotly_events(fig, 4, x, rel, arrays, go)
    _plotly_trace_range(fig, 4, rel, arrays, floor_zero=True, threshold_values=[alpha] if np.isfinite(alpha) else [])
    fig.update_yaxes(title_text="γ = σF / (‖F‖ + ε)", row=4, col=1)

    fig.add_trace(go.Scatter(x=x, y=arrays["raw_score"], mode="lines", name="acquisition score", line=dict(color="#8c510a")), row=5, col=1)
    _add_plotly_events(fig, 5, x, arrays["raw_score"], arrays, go)
    _plotly_score_range(fig, 5, arrays["raw_score"], arrays, thr)
    fig.update_yaxes(title_text="acquisition score", row=5, col=1)

    for label, mask, ypos, color in (("Shortlist", arrays["shortlist"], 5, "black"), ("Uncertain", arrays["force_inf"], 4, "#d95f02"), ("OOD", arrays["ood"], 3, "#7b3294"), ("Failed caps", arrays["geom_ok"] & ~arrays["caps_ok"], 2, "#e66101"), ("Failed geom", ~arrays["geom_ok"], 1, "#b2182b")):
        if np.any(mask):
            fig.add_trace(go.Scatter(x=x[mask], y=np.full(np.sum(mask), ypos), mode="markers", name=label, marker=dict(color=color, size=8)), row=6, col=1)
    fig.update_yaxes(title_text="AL status", tickmode="array", tickvals=[1, 2, 3, 4, 5], ticktext=["failed geom", "failed caps", "OOD", "uncertain", "shortlist"], row=6, col=1)
    fig.update_xaxes(title_text="Pool frame index", row=6, col=1)
    fig.update_layout(title=f"Pool AL uncertainty/acquisition: {state}", height=1350, width=1150, hovermode="x unified", template="plotly_white")
    out_path = os.path.join(out_dir, f"al_uncertainty_{state}.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[AL Plot] Saved {out_path}")
    return out_path


def plot_al_multihead_comparison(rows_by_state, metadata, out_dir="al_plots", dpi=300):
    if len(rows_by_state) < 2:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.0), sharex=True, constrained_layout=True)
    
    caps_E = [_thresholds_for_state(metadata, state).get("thr_sigma_E_hi_eff", np.nan) for state in rows_by_state]
    caps_F = [_thresholds_for_state(metadata, state).get("thr_sigma_F_hi_eff", np.nan) for state in rows_by_state]
    caps_Fmean = [_thresholds_for_state(metadata, state).get("thr_sigma_Fmean_hi_eff", np.nan) for state in rows_by_state]
    
    finite_caps_E = [c for c in caps_E if np.isfinite(c) and c > 0]
    finite_caps_F = [c for c in caps_F if np.isfinite(c) and c > 0]
    finite_caps_Fmean = [c for c in caps_Fmean if np.isfinite(c) and c > 0]
    
    cap_E = max(finite_caps_E) if finite_caps_E else np.nan
    cap_F = max(finite_caps_F) if finite_caps_F else np.nan
    cap_Fmean = max(finite_caps_Fmean) if finite_caps_Fmean else np.nan

    colors = ["#2166ac", "#b2182b", "#1b7837", "#762a83"]
    for color, (state, rows) in zip(colors, rows_by_state.items()):
        arrays = _state_arrays(rows)
        x = arrays["idx"]
        axes[0].plot(x, arrays["sigma_E_atom"] * 1000.0, lw=1.0, color=color, label=state)
        axes[1].plot(x, arrays["sigma_F_max"] * 1000.0, lw=1.0, color=color, label=state)
        axes[2].plot(x, arrays["sigma_F_mean"] * 1000.0, lw=1.0, color=color, label=state)
        if np.any(arrays["shortlist"]):
            axes[3].scatter(x[arrays["shortlist"]], np.full(np.sum(arrays["shortlist"]), state), s=24, color=color, edgecolor="black", linewidth=0.3, label=state)
    axes[0].set_ylabel("σE/atom (meV)")
    axes[1].set_ylabel("σFmax (meV/Å)")
    axes[2].set_ylabel("σFmean (meV/Å)")
    axes[3].set_ylabel("Shortlist")
    axes[3].set_xlabel("Pool frame index")
    
    if np.isfinite(cap_E):
        axes[0].set_ylim(0.0, cap_E * 1000.0 * 1.05)
    if np.isfinite(cap_F):
        axes[1].set_ylim(0.0, cap_F * 1000.0 * 1.05)
    if np.isfinite(cap_Fmean):
        axes[2].set_ylim(0.0, cap_Fmean * 1000.0 * 1.05)
    for ax in axes[:3]:
        ax.legend(loc="best", fontsize=8)
    if axes[3].collections:
        axes[3].legend(loc="best", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Multihead active-learning comparison", fontsize=14, fontweight="bold")
    out_path = os.path.join(out_dir, "al_uncertainty_multihead_comparison.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[AL Plot] Saved {out_path}")
    return out_path


def plot_al_multihead_comparison_interactive(rows_by_state, metadata, out_dir="al_plots"):
    if len(rows_by_state) < 2:
        return None
    go, make_subplots = _plotly_imports()
    if go is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045, subplot_titles=("Energy uncertainty", "Maximum force uncertainty", "Mean force uncertainty", "Shortlisted frames"))
    
    caps_E = [_thresholds_for_state(metadata, state).get("thr_sigma_E_hi_eff", np.nan) for state in rows_by_state]
    caps_F = [_thresholds_for_state(metadata, state).get("thr_sigma_F_hi_eff", np.nan) for state in rows_by_state]
    caps_Fmean = [_thresholds_for_state(metadata, state).get("thr_sigma_Fmean_hi_eff", np.nan) for state in rows_by_state]
    
    finite_caps_E = [c for c in caps_E if np.isfinite(c) and c > 0]
    finite_caps_F = [c for c in caps_F if np.isfinite(c) and c > 0]
    finite_caps_Fmean = [c for c in caps_Fmean if np.isfinite(c) and c > 0]
    
    cap_E = max(finite_caps_E) if finite_caps_E else np.nan
    cap_F = max(finite_caps_F) if finite_caps_F else np.nan
    cap_Fmean = max(finite_caps_Fmean) if finite_caps_Fmean else np.nan

    colors = ["#2166ac", "#b2182b", "#1b7837", "#762a83"]
    for color, (state, rows) in zip(colors, rows_by_state.items()):
        arrays = _state_arrays(rows)
        x = arrays["idx"]
        fig.add_trace(go.Scatter(x=x, y=arrays["sigma_E_atom"] * 1000.0, mode="lines", name=state, line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=arrays["sigma_F_max"] * 1000.0, mode="lines", name=state, line=dict(color=color), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=arrays["sigma_F_mean"] * 1000.0, mode="lines", name=state, line=dict(color=color), showlegend=False), row=3, col=1)
        if np.any(arrays["shortlist"]):
            fig.add_trace(go.Scatter(x=x[arrays["shortlist"]], y=np.full(np.sum(arrays["shortlist"]), state), mode="markers", name=f"{state} shortlist", marker=dict(color=color, size=8)), row=4, col=1)
            
    if np.isfinite(cap_E):
        fig.update_yaxes(range=[0.0, cap_E * 1000.0 * 1.05], row=1, col=1)
    if np.isfinite(cap_F):
        fig.update_yaxes(range=[0.0, cap_F * 1000.0 * 1.05], row=2, col=1)
    if np.isfinite(cap_Fmean):
        fig.update_yaxes(range=[0.0, cap_Fmean * 1000.0 * 1.05], row=3, col=1)

    fig.update_yaxes(title_text="σE/atom (meV)", row=1, col=1)
    fig.update_yaxes(title_text="σFmax (meV/Å)", row=2, col=1)
    fig.update_yaxes(title_text="σFmean (meV/Å)", row=3, col=1)
    fig.update_yaxes(title_text="Shortlist", row=4, col=1)
    fig.update_xaxes(title_text="Pool frame index", row=4, col=1)
    fig.update_layout(title="Multihead AL uncertainty comparison", height=900, width=1150, hovermode="x unified", template="plotly_white")
    out_path = os.path.join(out_dir, "al_uncertainty_multihead_comparison.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[AL Plot] Saved {out_path}")
    return out_path


def generate_al_diagnostic_plots(csv_path, out_dir="al_plots", window=50, drop_first=True, dpi=300, state=None):
    rows, metadata = read_al_diagnostics_csv(csv_path)
    if not rows:
        print(f"[AL Plot] No rows found in {csv_path}.")
        return []
    rows_by_state = defaultdict(list)
    for row in rows:
        rows_by_state[row.get("state", "unknown")].append(row)
    if state is not None:
        rows_by_state = {state: rows_by_state.get(state, [])}
    outputs = []
    for state_name, state_rows in rows_by_state.items():
        if not state_rows:
            continue
        outputs.append(plot_al_trace_state(state_rows, metadata, state_name, out_dir=out_dir, window=window, drop_first=drop_first, dpi=dpi))
        outputs.append(plot_al_uncertainty_state(state_rows, metadata, state_name, out_dir=out_dir, dpi=dpi))
        for html in (
            plot_al_trace_state_interactive(state_rows, metadata, state_name, out_dir=out_dir, window=window, drop_first=drop_first),
            plot_al_uncertainty_state_interactive(state_rows, metadata, state_name, out_dir=out_dir),
        ):
            if html:
                outputs.append(html)
    if state is None:
        comparison = plot_al_multihead_comparison(rows_by_state, metadata, out_dir=out_dir, dpi=dpi)
        if comparison:
            outputs.append(comparison)
        comparison_html = plot_al_multihead_comparison_interactive(rows_by_state, metadata, out_dir=out_dir)
        if comparison_html:
            outputs.append(comparison_html)
    return outputs


def generate_per_atom_uncertainty_plots(
    xyz_path: str,
    out_dir: str = "al_plots",
    dpi: int = 300,
    diag_csv_path: str = None,
) -> str:
    """Generate per-element sigma_F plots from per_atom_uncertainty.xyz.

    Parameters
    ----------
    xyz_path : str
        Path to the per-atom uncertainty XYZ file.
    out_dir : str
        Output directory for the plot.
    dpi : int
        Resolution of the saved PNG.
    diag_csv_path : str, optional
        Path to al_pool_diagnostics.csv to read AL thresholds from.

    Returns
    -------
    str
        Path to the saved plot (per_element_sigmaF.png).
    """
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Parse XYZ
    # ------------------------------------------------------------------
    elements_all, sFnorm_all, _sigma_E, muFnorm_all = _parse_per_atom_uncertainties(xyz_path)
    n_frames = len(elements_all)
    if n_frames == 0:
        return ""

    # ------------------------------------------------------------------
    # Parse AL thresholds from diagnostics CSV
    # ------------------------------------------------------------------
    thresholds = {}
    if diag_csv_path and os.path.exists(diag_csv_path):
        try:
            _, metadata = read_al_diagnostics_csv(diag_csv_path)
            thresholds = _thresholds_for_state(metadata, "default")
        except Exception:
            pass

    thr_sfmean = thresholds.get("thr_sigma_Fmean", np.nan)
    thr_sfmean_hi = thresholds.get("thr_sigma_Fmean_hi_eff", np.nan)
    thr_sfmean_hard = thresholds.get("hard_sigma_F_mean_min", np.nan)
    thr_sfmax = thresholds.get("thr_sigma_F", np.nan)
    thr_sfmax_hi = thresholds.get("thr_sigma_F_hi_eff", np.nan)
    thr_sfmax_hard = thresholds.get("hard_sigma_F_max_min", np.nan)
    env_alpha = thresholds.get("envelope_alpha", np.nan)
    env_atol = thresholds.get("envelope_floor", np.nan)
    if not np.isfinite(env_alpha):
        env_alpha = 0.20
    if not np.isfinite(env_atol):
        env_atol = 0.05

    def _thr_line(ax, value, color, scale=1000.0):
        val = float(value) * scale if np.isfinite(float(value)) else np.nan
        if not np.isnan(val):
            ax.axhline(val, ls="--", lw=1.0, color=color, alpha=0.85)

    def _gamma_ylim(ax, series_list):
        vals = [v for series in series_list for v in series if np.isfinite(v)]
        if not vals:
            return
        top = max(max(vals), env_alpha if np.isfinite(env_alpha) else 0.0)
        ax.set_ylim(0.0, top * 1.1)

    # ------------------------------------------------------------------
    # Per-frame stats
    # ------------------------------------------------------------------
    sf_max = np.array([max(nrm) if nrm else np.nan for nrm in sFnorm_all], dtype=float)
    sf_mean = np.array([np.mean(nrm) if nrm else np.nan for nrm in sFnorm_all], dtype=float)

    # Relative force uncertainty gamma = sigmaF / (||F|| + eps), eps = Atol/alpha
    # (meV-based ratio is unitless; eps in meV)
    eps_mev = env_atol * 1000.0 / env_alpha if env_alpha > 0 else np.nan
    has_gamma = muFnorm_all is not None and np.isfinite(eps_mev)
    gamma_atoms_all = []
    if has_gamma:
        for sf, mf in zip(sFnorm_all, muFnorm_all):
            gamma_atoms_all.append([s / (mu + eps_mev) for s, mu in zip(sf, mf)])
    else:
        gamma_atoms_all = [None] * n_frames
    gamma_per_frame = [
        (max(g) if g else np.nan) if g is not None else np.nan
        for g in gamma_atoms_all
    ]

    # Per-element per-frame stats
    elem_sfmax = defaultdict(list)
    elem_sfmean = defaultdict(list)
    elem_gammax = defaultdict(list)
    elem_gammamean = defaultdict(list)
    for els, nrm in zip(elements_all, sFnorm_all):
        by_elem = defaultdict(list)
        for e, v in zip(els, nrm):
            by_elem[e].append(v)
        for e, vals in by_elem.items():
            elem_sfmax[e].append(max(vals))
            elem_sfmean[e].append(np.mean(vals))
    if has_gamma:
        for els, gammas in zip(elements_all, gamma_atoms_all):
            by_elem = defaultdict(list)
            for e, g in zip(els, gammas):
                by_elem[e].append(g)
            for e, vals in by_elem.items():
                elem_gammax[e].append(max(vals))
                elem_gammamean[e].append(np.mean(vals))

    elements_sorted = sorted(elem_sfmax.keys())

    # ------------------------------------------------------------------
    # Plot: 3 x 3 grid
    #   col 1: sigma_F_mean  (violin / evolution / total)
    #   col 2: sigma_F_max   (violin / per-element evolution / total)
    #   col 3: relative force gamma (violin / per-element evolution / total)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(37, 16),
                             gridspec_kw={"height_ratios": [1, 1.5, 1]})
    (ax_vln_mean, ax_vln_max, ax_vln_gamma), (ax_evo_mean, ax_evo_max, ax_evo_gamma), (ax_total_mean, ax_total_max, ax_total_gamma) = axes

    colors = plt.cm.tab20(np.linspace(0, 1, len(elements_sorted)))
    elem_color = {e: colors[i] for i, e in enumerate(elements_sorted)}

    positions = list(range(1, len(elements_sorted) + 1))

    def _add_fmean_lines(ax):
        _thr_line(ax, thr_sfmean, "#636363")
        _thr_line(ax, thr_sfmean_hi, "#969696")
        _thr_line(ax, thr_sfmean_hard, "#b2182b")

    def _add_fmax_lines(ax):
        _thr_line(ax, thr_sfmax, "#636363")
        _thr_line(ax, thr_sfmax_hi, "#969696")
        _thr_line(ax, thr_sfmax_hard, "#b2182b")

    # -- Row 1 (col 1): violin sigma_F_mean --
    ax_vln_mean.set_axisbelow(True)
    vln_data_mean = [elem_sfmean[e] for e in elements_sorted]
    vp = ax_vln_mean.violinplot(vln_data_mean, positions=positions,
                                showmeans=True, showmedians=False, showextrema=False)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(1)
        body.set_edgecolor("#202020")
        body.set_zorder(2)
    vp["cmeans"].set_color("black")
    ax_vln_mean.set_xticks(positions)
    ax_vln_mean.set_xticklabels(elements_sorted)
    _add_fmean_lines(ax_vln_mean)
    ax_vln_mean.set_ylabel(r"$\sigma F_{\mathrm{mean}}$ (meV/$\AA$)")
    ax_vln_mean.set_title(r"Per-element $\sigma F_{\mathrm{mean}}$ distribution", fontsize=14, fontweight="bold")
    ax_vln_mean.grid(axis="y", alpha=0.3)

    # -- Row 2 (col 1): evolution sigma_F_mean --
    for e in elements_sorted:
        ax_evo_mean.plot(elem_sfmean[e], linewidth=0.6, alpha=1,
                         label=e, color=elem_color[e])
    _add_fmean_lines(ax_evo_mean)
    ax_evo_mean.set_ylabel(r"$\sigma F_{\mathrm{mean}}$ (meV/$\AA$)")
    # ax_evo_mean.set_title(r"Per-element $\sigma F_{\mathrm{mean}}$ evolution")
    ax_evo_mean.legend(fontsize=12, ncol=len(elements_sorted) + 1)
    ax_evo_mean.grid(alpha=0.3)

    # -- Row 1 (col 2): violin sigma_F_max --
    ax_vln_max.set_axisbelow(True)
    vln_data_max = [elem_sfmax[e] for e in elements_sorted]
    vp2 = ax_vln_max.violinplot(vln_data_max, positions=positions,
                                showmeans=True, showmedians=False, showextrema=False)
    for i, body in enumerate(vp2["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(1)
        body.set_edgecolor("#202020")
        body.set_zorder(2)
    vp2["cmeans"].set_color("black")
    ax_vln_max.set_xticks(positions)
    ax_vln_max.set_xticklabels(elements_sorted)
    _add_fmax_lines(ax_vln_max)
    ax_vln_max.set_ylabel(r"$\sigma F_{\max}$ (meV/$\AA$)")
    ax_vln_max.set_title(r"Per-element $\sigma F_{\max}$ distribution", fontsize=14, fontweight="bold")
    ax_vln_max.grid(axis="y", alpha=0.3)

    # -- Row 2 (col 2): evolution sigma_F_max --
    for e in elements_sorted:
        ax_evo_max.plot(elem_sfmax[e], linewidth=0.6, alpha=1,
                        label=e, color=elem_color[e])
    _add_fmax_lines(ax_evo_max)
    ax_evo_max.set_ylabel(r"$\sigma F_{\max}$ (meV/$\AA$)")
    # ax_evo_max.set_title(r"Per-element $\sigma F_{\max}$ evolution")
    ax_evo_max.legend(fontsize=12, ncol=len(elements_sorted))
    ax_evo_max.grid(alpha=0.3)

    # -- Row 1 (col 3): violin relative force gamma --
    if has_gamma:
        ax_vln_gamma.set_axisbelow(True)
        vln_data_gamma = [elem_gammax[e] for e in elements_sorted]
        vp3 = ax_vln_gamma.violinplot(vln_data_gamma, positions=positions,
                                      showmeans=True, showmedians=False, showextrema=False)
        for i, body in enumerate(vp3["bodies"]):
            body.set_facecolor(colors[i])
            body.set_alpha(1)
            body.set_edgecolor("#202020")
            body.set_zorder(2)
        vp3["cmeans"].set_color("black")
        ax_vln_gamma.set_xticks(positions)
        ax_vln_gamma.set_xticklabels(elements_sorted)
        _thr_line(ax_vln_gamma, env_alpha, "#c51b7d", scale=1.0)
        _gamma_ylim(ax_vln_gamma, vln_data_gamma)
        ax_vln_gamma.set_ylabel(r"$\gamma$ = $\sigma F$/($\|F\|$+$\varepsilon$)")
        ax_vln_gamma.set_title(r"Per-element relative force $\gamma$ distribution", fontsize=14, fontweight="bold")
        ax_vln_gamma.grid(axis="y", alpha=0.3)
    else:
        ax_vln_gamma.text(0.5, 0.5, "relative force not available\n(no muF_norm column in XYZ)",
                          ha="center", va="center", fontsize=12, color="#888888")
        ax_vln_gamma.set_title(r"Per-element relative force $\gamma$ distribution")

    # -- Row 2 (col 3): evolution relative force gamma --
    if has_gamma:
        for e in elements_sorted:
            ax_evo_gamma.plot(elem_gammax[e], linewidth=0.6, alpha=1,
                              label=e, color=elem_color[e])
        _thr_line(ax_evo_gamma, env_alpha, "#c51b7d", scale=1.0)
        _gamma_ylim(ax_evo_gamma, [elem_gammax[e] for e in elements_sorted])
        ax_evo_gamma.set_ylabel(r"$\gamma$ = $\sigma F$/($\|F\|$+$\varepsilon$)")
        # ax_evo_gamma.set_title(r"Per-element relative force $\gamma$ evolution")
        ax_evo_gamma.legend(fontsize=12, ncol=len(elements_sorted))
        ax_evo_gamma.grid(alpha=0.3)

    # -- Row 3 (col 3): total relative force gamma evolution --
    if has_gamma:
        ax_total_gamma.plot(gamma_per_frame, linewidth=1, color="black", alpha=1, label="Total")
        _thr_line(ax_total_gamma, env_alpha, "#c51b7d", scale=1.0)
        _gamma_ylim(ax_total_gamma, [gamma_per_frame])
        ax_total_gamma.set_xlabel("Frame index")
        ax_total_gamma.set_ylabel(r"$\gamma$ = $\sigma F$/($\|F\|$+$\varepsilon$)")
        # ax_total_gamma.set_title(r"Total $\gamma$ evolution (frame max over atoms)")
        ax_total_gamma.legend(fontsize=12)
        ax_total_gamma.grid(alpha=0.3)

    # -- Row 3 (col 1): total sigma_F_mean evolution --
    ax_total_mean.plot(sf_mean, linewidth=1, color="black", alpha=1, label="Total")
    _add_fmean_lines(ax_total_mean)
    ax_total_mean.set_xlabel("Frame index")
    ax_total_mean.set_ylabel(r"$\sigma F_{\mathrm{mean}}$ (meV/$\AA$)")
    # ax_total_mean.set_title(r"Total $\sigma F_{\mathrm{mean}}$ evolution")
    ax_total_mean.legend(fontsize=12)
    ax_total_mean.grid(alpha=0.3)

    # -- Row 3 (col 2): total sigma_F_max evolution --
    ax_total_max.plot(sf_max, linewidth=1, color="black", alpha=1, label="Total")
    _add_fmax_lines(ax_total_max)
    ax_total_max.set_xlabel("Frame index")
    ax_total_max.set_ylabel(r"$\sigma F_{\max}$ (meV/$\AA$)")
    # ax_total_max.set_title(r"Total $\sigma F_{\max}$ evolution")
    ax_total_max.legend(fontsize=12)
    ax_total_max.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "per_element_sigmaF.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[AL Plot] Saved {out_path}")

    return out_path


def generate_force_envelope_scatter(
    pool_frames,
    mu_F_pool_flat,
    sigma_F_pool_flat,
    rdf_ok_mask,
    sel_rel_indices,
    *,
    alpha=0.20,
    atol=0.05,
    out_dir="uq_plots",
):
    """Scatter σF vs ‖F‖ per atom with the envelope trigger boundary.

    Answers: do high-σF atoms also carry high *relative* force error?
    Points above the line σF = Atol + α·‖F‖ are envelope-triggered.
    """
    if mu_F_pool_flat is None or sigma_F_pool_flat is None:
        print("[EnvelopeScatter] Per-atom force arrays unavailable; skipping.")
        return
    muF = np.asarray(mu_F_pool_flat, dtype=float)
    sigF = np.asarray(sigma_F_pool_flat, dtype=float)
    frames = _split_atom_vectors(muF, pool_frames)
    sig_frames = _split_atom_vectors(sigF, pool_frames)
    if len(frames) != len(pool_frames) or len(sig_frames) != len(pool_frames):
        print("[EnvelopeScatter] Frame/array mismatch; skipping.")
        return

    force_mag = []
    sigma_mag = []
    frame_of = []
    for i, (mf, sf) in enumerate(zip(frames, sig_frames)):
        fm = np.linalg.norm(np.asarray(mf, dtype=float).reshape(-1, 3), axis=1)
        sm = np.linalg.norm(np.asarray(sf, dtype=float).reshape(-1, 3), axis=1)
        force_mag.append(fm)
        sigma_mag.append(sm)
        frame_of.extend([i] * len(fm))
    force_mag = np.concatenate(force_mag)
    sigma_mag = np.concatenate(sigma_mag)
    frame_of = np.asarray(frame_of, dtype=int)

    rdf_ok = np.repeat(
        np.asarray(rdf_ok_mask, dtype=bool) if rdf_ok_mask is not None else np.ones(len(pool_frames), dtype=bool),
        [len(f) for f in frames],
    )
    sel_set = set(int(i) for i in (sel_rel_indices or []))
    is_sel = np.isin(frame_of, list(sel_set))

    fmax = max(force_mag.max(), 0.1)
    line_f = np.linspace(0, fmax, 200)
    line_s = atol + alpha * line_f

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(force_mag[~rdf_ok], sigma_mag[~rdf_ok], s=5, color="#b2182b", alpha=0.35, label="failed RDF")
    ax.scatter(force_mag[rdf_ok & ~is_sel], sigma_mag[rdf_ok & ~is_sel], s=5, color="#4d4d4d", alpha=0.30, label="pool")
    ax.scatter(force_mag[rdf_ok & is_sel], sigma_mag[rdf_ok & is_sel], s=16, color="#d95f02", alpha=0.85, label="selected")
    ax.plot(line_f, line_s, ls="--", lw=1.5, color="#276419", label=f"Atol + α·‖F‖ (α={alpha:.2f}, Atol={atol:.4g})")
    ax.fill_between(line_f, line_s, atol, where=(line_s >= atol), color="#276419", alpha=0.10, label="envelope trigger region")
    ax.set_xlabel(r"$\|\mathbf{F}\|$ (eV/Å)")
    ax.set_ylabel(r"$\sigma_F$ (eV/Å)")
    ax.set_title("Per-atom force uncertainty vs force magnitude")
    ax.legend(loc="best", fontsize=8)
    ax.set_xlim(0, fmax * 1.02)
    ax.set_ylim(0, max(sigma_mag.max(), atol * 1.1) * 1.05)
    ax.grid(True, alpha=0.25)
    out_path = os.path.join(out_dir, "force_envelope_scatter.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[EnvelopeScatter] Saved {out_path}")
    return out_path
