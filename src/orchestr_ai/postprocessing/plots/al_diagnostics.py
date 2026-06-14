"""Standalone plotting utilities for pool active-learning diagnostics CSV files."""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


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
        return _apply_robust_ylim(ax, score, arrays, floor_zero=True)
    ymax = float(np.nanpercentile(vals, 99.0))
    shortlist_vals = score[mask & arrays["shortlist"]]
    if shortlist_vals.size:
        ymax = max(ymax, float(np.nanmax(shortlist_vals)))
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    ax.set_ylim(0.0, ymax * 1.08)
    clipped = int(np.sum(np.isfinite(score) & (score > ymax * 1.08)))
    if clipped:
        ax.text(
            0.995, 0.94, f"{clipped} outlier(s) outside y-range",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.8),
        )
    return 0.0, ymax * 1.08


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
        return _plotly_range(fig, row, score, arrays, floor_zero=True)
    ymax = float(np.nanpercentile(vals, 99.0))
    shortlist_vals = score[mask & arrays["shortlist"]]
    if shortlist_vals.size:
        ymax = max(ymax, float(np.nanmax(shortlist_vals)))
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    fig.update_yaxes(range=[0.0, ymax * 1.08], row=row, col=1)


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
    _apply_robust_ylim(axes[0], dE, arrays)
    axes[0].set_ylabel("ΔE (eV)")
    axes[0].set_title(f"Energy relative to {e_ref_label}: E - {e_ref:.6g} eV", fontsize=10)
    axes[0].legend(loc="best", fontsize=8, ncol=4)

    axes[1].plot(x, arrays["Fmax"], lw=1.0, color="#1b7837")
    fmax_keys = [("thr_Fmag", "#636363", "low"), ("thr_Fmag_hi_eff", "#969696", "upper cap"), ("train_Fmax_hard_cap", "#b2182b", "hard cap")]
    _threshold_lines(axes[1], thr, fmax_keys)
    _mark_events(axes[1], x, arrays["Fmax"], arrays)
    _apply_robust_ylim(axes[1], arrays["Fmax"], arrays, floor_zero=True, threshold_values=_threshold_values(thr, [k[0] for k in fmax_keys]))
    axes[1].set_ylabel("Fmax (eV/Å)")

    axes[2].plot(x, arrays["Fmean"], lw=1.0, color="#5aae61")
    _mark_events(axes[2], x, arrays["Fmean"], arrays)
    _apply_robust_ylim(axes[2], arrays["Fmean"], arrays, floor_zero=True)
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
    fig, axes = plt.subplots(5, 1, figsize=(11.0, 12.5), sharex=True, constrained_layout=True)
    fig.suptitle(f"Pool AL uncertainty/acquisition: {state}", fontsize=14, fontweight="bold")

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

    axes[3].plot(x, arrays["raw_score"], lw=1.0, color="#8c510a")
    _mark_events(axes[3], x, arrays["raw_score"], arrays)
    _apply_score_ylim(axes[3], arrays["raw_score"], arrays, thr)
    axes[3].set_ylabel("acquisition score")
    axes[3].set_title("Acquisition score = leverage/novelty ranking used for final diverse selection", fontsize=10)

    _plot_status_track(axes[4], x, arrays)
    axes[4].set_ylabel("AL status")
    axes[4].set_xlabel("Pool frame index")
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
    _plotly_range(fig, 1, dE, arrays)

    fmax_keys = [("thr_Fmag", "#636363", "Fmax low"), ("thr_Fmag_hi_eff", "#969696", "Fmax upper cap"), ("train_Fmax_hard_cap", "#b2182b", "Fmax hard cap")]
    fig.add_trace(go.Scatter(x=x, y=arrays["Fmax"], mode="lines", name="Fmax", line=dict(color="#1b7837")), row=2, col=1)
    _add_plotly_thresholds(fig, 2, x, thr, fmax_keys)
    _add_plotly_events(fig, 2, x, arrays["Fmax"], arrays, go)
    _plotly_range(fig, 2, arrays["Fmax"], arrays, floor_zero=True, threshold_values=_threshold_values(thr, [k[0] for k in fmax_keys]))

    fig.add_trace(go.Scatter(x=x, y=arrays["Fmean"], mode="lines", name="Fmean", line=dict(color="#5aae61")), row=3, col=1)
    _add_plotly_events(fig, 3, x, arrays["Fmean"], arrays, go)
    _plotly_range(fig, 3, arrays["Fmean"], arrays, floor_zero=True)

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
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.045, subplot_titles=("Energy uncertainty", "Maximum force uncertainty", "Mean force uncertainty", "Acquisition score", "AL status"))
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
    fig.add_trace(go.Scatter(x=x, y=arrays["raw_score"], mode="lines", name="acquisition score", line=dict(color="#8c510a")), row=4, col=1)
    _add_plotly_events(fig, 4, x, arrays["raw_score"], arrays, go)
    _plotly_score_range(fig, 4, arrays["raw_score"], arrays, thr)
    fig.update_yaxes(title_text="acquisition score", row=4, col=1)
    for label, mask, ypos, color in (("Shortlist", arrays["shortlist"], 5, "black"), ("Uncertain", arrays["force_inf"], 4, "#d95f02"), ("OOD", arrays["ood"], 3, "#7b3294"), ("Failed caps", arrays["geom_ok"] & ~arrays["caps_ok"], 2, "#e66101"), ("Failed geom", ~arrays["geom_ok"], 1, "#b2182b")):
        if np.any(mask):
            fig.add_trace(go.Scatter(x=x[mask], y=np.full(np.sum(mask), ypos), mode="markers", name=label, marker=dict(color=color, size=8)), row=5, col=1)
    fig.update_yaxes(title_text="AL status", tickmode="array", tickvals=[1, 2, 3, 4, 5], ticktext=["failed geom", "failed caps", "OOD", "uncertain", "shortlist"], row=5, col=1)
    fig.update_xaxes(title_text="Pool frame index", row=5, col=1)
    fig.update_layout(title=f"Pool AL uncertainty/acquisition: {state}", height=1050, width=1150, hovermode="x unified", template="plotly_white")
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
