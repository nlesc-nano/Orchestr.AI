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
    if drop_first and np.any(mask):
        mask &= x != np.nanmin(x[mask])
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


def _threshold_lines(ax, thresholds, keys, scale=1.0):
    for key, color, label in keys:
        value = thresholds.get(key, np.nan)
        if np.isfinite(value):
            ax.axhline(value * scale, ls="--", lw=1.0, color=color, alpha=0.8, label=label)


def _mark_events(ax, x, y, arrays, *, show_label=False):
    geom_bad = ~arrays["geom_ok"]
    caps_bad = arrays["geom_ok"] & ~arrays["caps_ok"]
    shortlist = arrays["shortlist"]
    ood = arrays["ood"]
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
        ("shortlist", arrays["shortlist"], 4, "black"),
        ("OOD", arrays["ood"], 3, "#7b3294"),
        ("failed caps", arrays["geom_ok"] & ~arrays["caps_ok"], 2, "#e66101"),
        ("failed geom", ~arrays["geom_ok"], 1, "#b2182b"),
    ]
    for label, mask, ypos, color in events:
        if np.any(mask):
            ax.scatter(x[mask], np.full(np.sum(mask), ypos), s=30, color=color, label=label)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["failed geom", "failed caps", "OOD", "shortlist"])
    ax.set_ylim(0.4, 4.6)
    ax.legend(loc="upper right", fontsize=8, ncol=4)


def plot_al_state(rows, metadata, state, out_dir="al_plots", window=50, drop_first=True, dpi=300):
    arrays = _state_arrays(rows)
    thr = metadata.get("thresholds", {}).get(state, {})
    x = arrays["idx"]
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(6, 1, figsize=(11.0, 15.0), sharex=True, constrained_layout=True)
    fig.suptitle(f"Pool active-learning diagnostics: {state}", fontsize=14, fontweight="bold")

    e_ref, e_ref_label = _energy_reference(arrays)
    dE = arrays["E_pred"] - e_ref
    sE = arrays["sigma_E"]
    xs, dE_smooth = _rolling_mean(x, dE, window=window, drop_first=drop_first)
    _, sE_smooth = _rolling_mean(x, sE, window=window, drop_first=drop_first)
    axes[0].plot(x, dE, lw=0.7, color="#4d4d4d", alpha=0.28, label="raw")
    axes[0].plot(xs, dE_smooth, lw=1.8, color="#2166ac", label=f"{window}-frame running mean")
    if sE_smooth.size == dE_smooth.size:
        axes[0].fill_between(xs, dE_smooth - sE_smooth, dE_smooth + sE_smooth, color="#67a9cf", alpha=0.25, label="± sigma_E")
    _mark_events(axes[0], x, dE, arrays, show_label=True)
    axes[0].set_ylabel("ΔE (eV)")
    axes[0].set_title(f"Energy relative to {e_ref_label}: E - {e_ref:.6g} eV", fontsize=10)
    axes[0].legend(loc="best", fontsize=8, ncol=4)

    sigma_e_mev = arrays["sigma_E_atom"] * 1000.0
    axes[1].plot(x, sigma_e_mev, lw=1.0, color="#2166ac")
    _threshold_lines(axes[1], thr, [
        ("thr_sigma_E_low", "#636363", "low"),
        ("thr_sigma_E_hi_eff", "#969696", "upper cap"),
        ("hard_sigma_E_atom_min", "#b2182b", "hard floor"),
    ], scale=1000.0)
    _mark_events(axes[1], x, sigma_e_mev, arrays)
    axes[1].set_ylabel("σE/atom (meV)")

    axes[2].plot(x, arrays["Fmax"], lw=1.0, color="#1b7837")
    _threshold_lines(axes[2], thr, [
        ("thr_Fmag", "#636363", "low"),
        ("thr_Fmag_hi_eff", "#969696", "upper cap"),
        ("train_Fmax_hard_cap", "#b2182b", "hard cap"),
    ])
    _mark_events(axes[2], x, arrays["Fmax"], arrays)
    axes[2].set_ylabel("Fmax (eV/Å)")

    sigma_fmax_mev = arrays["sigma_F_max"] * 1000.0
    axes[3].plot(x, sigma_fmax_mev, lw=1.0, color="#762a83")
    _threshold_lines(axes[3], thr, [
        ("thr_sigma_F", "#636363", "low"),
        ("thr_sigma_F_hi_eff", "#969696", "upper cap"),
        ("hard_sigma_F_max_min", "#b2182b", "hard floor"),
    ], scale=1000.0)
    _mark_events(axes[3], x, sigma_fmax_mev, arrays)
    axes[3].set_ylabel("σFmax (meV/Å)")

    sigma_fmean_mev = arrays["sigma_F_mean"] * 1000.0
    axes[4].plot(x, sigma_fmean_mev, lw=1.0, color="#af8dc3")
    _threshold_lines(axes[4], thr, [
        ("thr_sigma_Fmean", "#636363", "low"),
        ("thr_sigma_Fmean_hi_eff", "#969696", "upper cap"),
        ("hard_sigma_F_mean_min", "#b2182b", "hard floor"),
    ], scale=1000.0)
    _mark_events(axes[4], x, sigma_fmean_mev, arrays)
    axes[4].set_ylabel("σFmean (meV/Å)")

    _plot_status_track(axes[5], x, arrays)
    axes[5].set_ylabel("AL status")
    axes[5].set_xlabel("Pool frame index")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    out_path = os.path.join(out_dir, f"al_trace_{state}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[AL Plot] Saved {out_path}")
    return out_path


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
                x=[x0, x1], y=[value * scale, value * scale],
                mode="lines", line=dict(color=color, dash="dash", width=1),
                name=f"{label}: {value * scale:.4g}", row=row, col=1,
            )


def _add_plotly_events(fig, row, x, y, arrays, go):
    events = [
        ("OOD", arrays["ood"], "triangle-up", "#7b3294"),
        ("Failed caps", arrays["geom_ok"] & ~arrays["caps_ok"], "x", "#e66101"),
        ("Failed geom", ~arrays["geom_ok"], "x", "#b2182b"),
        ("Shortlist", arrays["shortlist"], "circle", "black"),
    ]
    for label, mask, symbol, color in events:
        if np.any(mask):
            fig.add_trace(
                go.Scatter(
                    x=x[mask], y=y[mask], mode="markers", name=label,
                    marker=dict(color=color, symbol=symbol, size=8),
                    hovertemplate="frame=%{x}<br>value=%{y:.5g}<extra>" + label + "</extra>",
                ), row=row, col=1,
            )


def plot_al_state_interactive(rows, metadata, state, out_dir="al_plots", window=50, drop_first=True):
    go, make_subplots = _plotly_imports()
    if go is None:
        return None
    arrays = _state_arrays(rows)
    thr = metadata.get("thresholds", {}).get(state, {})
    x = arrays["idx"]
    os.makedirs(out_dir, exist_ok=True)

    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        subplot_titles=("Relative energy", "Energy uncertainty", "Maximum force", "Maximum force uncertainty", "Mean force uncertainty", "AL status"),
    )
    e_ref, e_ref_label = _energy_reference(arrays)
    dE = arrays["E_pred"] - e_ref
    sE = arrays["sigma_E"]
    xs, dE_smooth = _rolling_mean(x, dE, window=window, drop_first=drop_first)
    _, sE_smooth = _rolling_mean(x, sE, window=window, drop_first=drop_first)
    fig.add_trace(go.Scatter(x=x, y=dE, mode="lines", name="ΔE raw", line=dict(color="#777", width=1), opacity=0.45), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=dE_smooth, mode="lines", name=f"ΔE {window}-frame mean", line=dict(color="#2166ac", width=2)), row=1, col=1)
    if sE_smooth.size == dE_smooth.size:
        fig.add_trace(go.Scatter(x=xs, y=dE_smooth + sE_smooth, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=dE_smooth - sE_smooth, mode="lines", fill="tonexty", fillcolor="rgba(103,169,207,0.25)", line=dict(width=0), name="± sigma_E", hoverinfo="skip"), row=1, col=1)
    _add_plotly_events(fig, 1, x, dE, arrays, go)

    sigma_e_mev = arrays["sigma_E_atom"] * 1000.0
    fig.add_trace(go.Scatter(x=x, y=sigma_e_mev, mode="lines", name="σE/atom", line=dict(color="#2166ac")), row=2, col=1)
    _add_plotly_thresholds(fig, 2, x, thr, [("thr_sigma_E_low", "#636363", "σE low"), ("thr_sigma_E_hi_eff", "#969696", "σE upper cap"), ("hard_sigma_E_atom_min", "#b2182b", "σE hard floor")], scale=1000.0)
    _add_plotly_events(fig, 2, x, sigma_e_mev, arrays, go)

    fig.add_trace(go.Scatter(x=x, y=arrays["Fmax"], mode="lines", name="Fmax", line=dict(color="#1b7837")), row=3, col=1)
    _add_plotly_thresholds(fig, 3, x, thr, [("thr_Fmag", "#636363", "Fmax low"), ("thr_Fmag_hi_eff", "#969696", "Fmax upper cap"), ("train_Fmax_hard_cap", "#b2182b", "Fmax hard cap")])
    _add_plotly_events(fig, 3, x, arrays["Fmax"], arrays, go)

    sigma_fmax_mev = arrays["sigma_F_max"] * 1000.0
    fig.add_trace(go.Scatter(x=x, y=sigma_fmax_mev, mode="lines", name="σFmax", line=dict(color="#762a83")), row=4, col=1)
    _add_plotly_thresholds(fig, 4, x, thr, [("thr_sigma_F", "#636363", "σFmax low"), ("thr_sigma_F_hi_eff", "#969696", "σFmax upper cap"), ("hard_sigma_F_max_min", "#b2182b", "σFmax hard floor")], scale=1000.0)
    _add_plotly_events(fig, 4, x, sigma_fmax_mev, arrays, go)

    sigma_fmean_mev = arrays["sigma_F_mean"] * 1000.0
    fig.add_trace(go.Scatter(x=x, y=sigma_fmean_mev, mode="lines", name="σFmean", line=dict(color="#af8dc3")), row=5, col=1)
    _add_plotly_thresholds(fig, 5, x, thr, [("thr_sigma_Fmean", "#636363", "σFmean low"), ("thr_sigma_Fmean_hi_eff", "#969696", "σFmean upper cap"), ("hard_sigma_F_mean_min", "#b2182b", "σFmean hard floor")], scale=1000.0)
    _add_plotly_events(fig, 5, x, sigma_fmean_mev, arrays, go)

    status_events = [("Shortlist", arrays["shortlist"], 4, "black"), ("OOD", arrays["ood"], 3, "#7b3294"), ("Failed caps", arrays["geom_ok"] & ~arrays["caps_ok"], 2, "#e66101"), ("Failed geom", ~arrays["geom_ok"], 1, "#b2182b")]
    for label, mask, ypos, color in status_events:
        if np.any(mask):
            fig.add_trace(go.Scatter(x=x[mask], y=np.full(np.sum(mask), ypos), mode="markers", name=label, marker=dict(color=color, size=8), hovertemplate="frame=%{x}<extra>" + label + "</extra>"), row=6, col=1)

    fig.update_yaxes(title_text="ΔE (eV)", row=1, col=1)
    fig.update_yaxes(title_text="σE/atom (meV)", row=2, col=1)
    fig.update_yaxes(title_text="Fmax (eV/Å)", row=3, col=1)
    fig.update_yaxes(title_text="σFmax (meV/Å)", row=4, col=1)
    fig.update_yaxes(title_text="σFmean (meV/Å)", row=5, col=1)
    fig.update_yaxes(title_text="AL status", tickmode="array", tickvals=[1, 2, 3, 4], ticktext=["failed geom", "failed caps", "OOD", "shortlist"], row=6, col=1)
    fig.update_xaxes(title_text="Pool frame index", row=6, col=1)
    fig.update_layout(
        title=f"Pool active-learning diagnostics: {state}<br><sup>Energy reference: {e_ref_label}, E_ref={e_ref:.6g} eV</sup>",
        height=1200, width=1150, hovermode="x unified", template="plotly_white",
    )
    out_path = os.path.join(out_dir, f"al_trace_{state}.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[AL Plot] Saved {out_path}")
    return out_path


def plot_al_multihead_comparison(rows_by_state, out_dir="al_plots", dpi=300):
    if len(rows_by_state) < 2:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 11.0), sharex=True, constrained_layout=True)
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
    for ax in axes[:3]:
        ax.legend(loc="best", fontsize=8)
    if axes[3].collections:
        axes[3].legend(loc="best", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Multihead active-learning comparison", fontsize=14, fontweight="bold")
    out_path = os.path.join(out_dir, "al_trace_multihead_comparison.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[AL Plot] Saved {out_path}")
    return out_path


def plot_al_multihead_comparison_interactive(rows_by_state, out_dir="al_plots"):
    if len(rows_by_state) < 2:
        return None
    go, make_subplots = _plotly_imports()
    if go is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045, subplot_titles=("Energy uncertainty", "Maximum force uncertainty", "Mean force uncertainty", "Shortlisted frames"))
    colors = ["#2166ac", "#b2182b", "#1b7837", "#762a83"]
    for color, (state, rows) in zip(colors, rows_by_state.items()):
        arrays = _state_arrays(rows)
        x = arrays["idx"]
        fig.add_trace(go.Scatter(x=x, y=arrays["sigma_E_atom"] * 1000.0, mode="lines", name=state, line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=arrays["sigma_F_max"] * 1000.0, mode="lines", name=state, line=dict(color=color), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=arrays["sigma_F_mean"] * 1000.0, mode="lines", name=state, line=dict(color=color), showlegend=False), row=3, col=1)
        if np.any(arrays["shortlist"]):
            fig.add_trace(go.Scatter(x=x[arrays["shortlist"]], y=np.full(np.sum(arrays["shortlist"]), state), mode="markers", name=f"{state} shortlist", marker=dict(color=color, size=8)), row=4, col=1)
    fig.update_yaxes(title_text="σE/atom (meV)", row=1, col=1)
    fig.update_yaxes(title_text="σFmax (meV/Å)", row=2, col=1)
    fig.update_yaxes(title_text="σFmean (meV/Å)", row=3, col=1)
    fig.update_yaxes(title_text="Shortlist", row=4, col=1)
    fig.update_xaxes(title_text="Pool frame index", row=4, col=1)
    fig.update_layout(title="Multihead active-learning comparison", height=900, width=1150, hovermode="x unified", template="plotly_white")
    out_path = os.path.join(out_dir, "al_trace_multihead_comparison.html")
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
        if state_rows:
            outputs.append(plot_al_state(state_rows, metadata, state_name, out_dir=out_dir, window=window, drop_first=drop_first, dpi=dpi))
            html = plot_al_state_interactive(state_rows, metadata, state_name, out_dir=out_dir, window=window, drop_first=drop_first)
            if html:
                outputs.append(html)
    if state is None:
        comparison = plot_al_multihead_comparison(rows_by_state, out_dir=out_dir, dpi=dpi)
        if comparison:
            outputs.append(comparison)
        comparison_html = plot_al_multihead_comparison_interactive(rows_by_state, out_dir=out_dir)
        if comparison_html:
            outputs.append(comparison_html)
    return outputs
