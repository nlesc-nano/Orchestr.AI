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
    metadata = {"thresholds": defaultdict(dict)}
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        data_lines = []
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("#"):
                text = stripped.lstrip("#").strip()
                if "=" not in text:
                    continue
                key, value = [part.strip() for part in text.split("=", 1)]
                if key.startswith("threshold[state=") and "]." in key:
                    state = key.split("threshold[state=", 1)[1].split("]", 1)[0]
                    threshold_key = key.split("].", 1)[1]
                    metadata["thresholds"][state][threshold_key] = _to_float(value)
                else:
                    metadata[key] = value
                continue
            if stripped:
                data_lines.append(line)
        if data_lines:
            rows = list(csv.DictReader(data_lines))
    metadata["thresholds"] = dict(metadata["thresholds"])
    return rows, metadata


def _state_arrays(rows):
    keys_float = [
        "idx", "pool_row", "n_atoms", "gamma0", "dM", "Dgain", "raw_score",
        "E_pred", "E_pred_atom", "sigma_E", "sigma_E_atom", "sigma_F_max",
        "sigma_F_mean", "Eabs_exp", "Fabs_mean", "Fabs_max", "Fmax",
    ]
    keys_bool = ["geom_ok", "caps_ok", "force_inf", "gamma_gate", "cal_ok", "ood", "selected", "shortlist"]
    out = {key: np.array([_to_float(row.get(key)) for row in rows], dtype=float) for key in keys_float}
    for key in keys_bool:
        out[key] = _to_bool_array([row.get(key, "0") for row in rows])
    order = np.argsort(out["idx"])
    for key, values in out.items():
        out[key] = values[order]
    return out


def _rolling_mean(x, y, window=50, drop_first=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if drop_first and mask.size:
        first = np.nanmin(x[mask]) if np.any(mask) else np.nan
        mask &= x != first
    x_valid = x[mask]
    y_valid = y[mask]
    if y_valid.size == 0:
        return x_valid, y_valid
    window = max(1, int(window))
    half = window // 2
    smooth = np.empty_like(y_valid, dtype=float)
    for i in range(y_valid.size):
        lo = max(0, i - half)
        hi = min(y_valid.size, i + half + 1)
        smooth[i] = np.nanmean(y_valid[lo:hi])
    return x_valid, smooth


def _draw_thresholds(ax, thresholds, keys, scale=1.0):
    for key, color in keys:
        value = thresholds.get(key, np.nan)
        if np.isfinite(value):
            ax.axhline(value * scale, ls="--", lw=1.0, color=color, alpha=0.75)


def _mark_events(ax, x, y, arrays, *, scale=1.0, show_label=False):
    geom_bad = ~arrays["geom_ok"]
    caps_bad = arrays["geom_ok"] & ~arrays["caps_ok"]
    shortlist = arrays["shortlist"]
    ood = arrays["ood"]
    if np.any(ood):
        ax.scatter(x[ood], y[ood] * scale, marker="^", s=18, color="#7b3294", alpha=0.55, label="OOD" if show_label else None)
    if np.any(caps_bad):
        ax.scatter(x[caps_bad], y[caps_bad] * scale, marker="x", s=24, color="#e66101", alpha=0.8, label="Failed caps" if show_label else None)
    if np.any(geom_bad):
        ax.scatter(x[geom_bad], y[geom_bad] * scale, marker="x", s=20, color="#b2182b", alpha=0.45, label="Failed geom" if show_label else None)
    if np.any(shortlist):
        ax.scatter(x[shortlist], y[shortlist] * scale, marker="o", s=28, color="black", alpha=0.9, label="Shortlist" if show_label else None, zorder=5)


def plot_al_state(rows, metadata, state, out_dir="al_plots", window=50, drop_first=True, dpi=300):
    arrays = _state_arrays(rows)
    thr = metadata.get("thresholds", {}).get(state, {})
    x = arrays["idx"]
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(10.5, 13.0), sharex=True, constrained_layout=True)
    fig.suptitle(f"Pool active-learning diagnostics: {state}", fontsize=14, fontweight="bold")

    e_mev = arrays["E_pred_atom"] * 1000.0
    se_mev = arrays["sigma_E_atom"] * 1000.0
    xs, e_smooth = _rolling_mean(x, e_mev, window=window, drop_first=drop_first)
    _, se_smooth = _rolling_mean(x, se_mev, window=window, drop_first=drop_first)
    axes[0].plot(x, e_mev, lw=0.7, color="#4d4d4d", alpha=0.25, label="raw")
    axes[0].plot(xs, e_smooth, lw=1.8, color="#2166ac", label=f"{window}-frame running mean")
    if se_smooth.size == e_smooth.size:
        axes[0].fill_between(xs, e_smooth - se_smooth, e_smooth + se_smooth, color="#67a9cf", alpha=0.25, label="± sigma_E")
    _mark_events(axes[0], x, e_mev, arrays, show_label=True)
    axes[0].set_ylabel("E_pred/atom (meV)")
    axes[0].legend(loc="best", fontsize=8, ncol=4)

    axes[1].plot(x, se_mev, lw=1.0, color="#2166ac")
    _draw_thresholds(axes[1], thr, [("thr_sigma_E_low", "#636363"), ("thr_sigma_E_hi_eff", "#636363"), ("hard_sigma_E_atom_min", "#b2182b")], scale=1000.0)
    _mark_events(axes[1], x, se_mev, arrays)
    axes[1].set_ylabel("sigma_E/atom (meV)")

    axes[2].plot(x, arrays["Fmax"], lw=1.0, color="#1b7837")
    _draw_thresholds(axes[2], thr, [("thr_Fmag", "#636363"), ("thr_Fmag_hi_eff", "#636363"), ("train_Fmax_hard_cap", "#b2182b")])
    _mark_events(axes[2], x, arrays["Fmax"], arrays)
    axes[2].set_ylabel("Fmax (eV/A)")

    axes[3].plot(x, arrays["sigma_F_max"], lw=1.0, color="#762a83", label="sigma_F_max")
    axes[3].plot(x, arrays["sigma_F_mean"], lw=1.0, color="#af8dc3", label="sigma_F_mean")
    _draw_thresholds(axes[3], thr, [("thr_sigma_F", "#636363"), ("thr_sigma_F_hi_eff", "#636363"), ("hard_sigma_F_max_min", "#b2182b"), ("hard_sigma_F_mean_min", "#b2182b")])
    _mark_events(axes[3], x, arrays["sigma_F_max"], arrays)
    axes[3].set_ylabel("Force uncertainty (eV/A)")
    axes[3].legend(loc="best", fontsize=8)

    axes[4].plot(x, arrays["raw_score"], lw=1.0, color="#8c510a", label="AL score")
    _mark_events(axes[4], x, arrays["raw_score"], arrays)
    axes[4].set_ylabel("raw score")
    axes[4].set_xlabel("Pool frame index")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    out_path = os.path.join(out_dir, f"al_trace_{state}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[AL Plot] Saved {out_path}")
    return out_path


def plot_al_multihead_comparison(rows_by_state, out_dir="al_plots", dpi=300):
    if len(rows_by_state) < 2:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True, constrained_layout=True)
    colors = ["#2166ac", "#b2182b", "#1b7837", "#762a83"]
    for color, (state, rows) in zip(colors, rows_by_state.items()):
        arrays = _state_arrays(rows)
        x = arrays["idx"]
        axes[0].plot(x, arrays["sigma_E_atom"] * 1000.0, lw=1.0, color=color, label=state)
        axes[1].plot(x, arrays["sigma_F_max"], lw=1.0, color=color, label=state)
        axes[2].plot(x, arrays["raw_score"], lw=0.9, color=color, alpha=0.8, label=state)
        if np.any(arrays["shortlist"]):
            axes[2].scatter(x[arrays["shortlist"]], arrays["raw_score"][arrays["shortlist"]], s=22, color=color, edgecolor="black", linewidth=0.3)
    axes[0].set_ylabel("sigma_E/atom (meV)")
    axes[1].set_ylabel("sigma_F_max (eV/A)")
    axes[2].set_ylabel("AL score")
    axes[2].set_xlabel("Pool frame index")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Multihead active-learning comparison", fontsize=14, fontweight="bold")
    out_path = os.path.join(out_dir, "al_trace_multihead_comparison.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
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
        if state_rows:
            outputs.append(plot_al_state(state_rows, metadata, state_name, out_dir=out_dir, window=window, drop_first=drop_first, dpi=dpi))
    if state is None:
        comparison = plot_al_multihead_comparison(rows_by_state, out_dir=out_dir, dpi=dpi)
        if comparison:
            outputs.append(comparison)
    return outputs
