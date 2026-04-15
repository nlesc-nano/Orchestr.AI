import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)


def _plot_grouped_metric_panel(
    ax,
    methods,
    size_to_values,
    title,
    ylabel,
    ylim=None,
    max_sizes=5,
):
    all_sizes = sorted(size_to_values.keys())
    sizes = all_sizes[:max_sizes]

    if not sizes:
        logger.warning("[coverage_report] No sizes available for grouped panel.")
        return

    n_methods = len(methods)
    n_sizes = len(sizes)

    x = np.arange(n_methods) * 1.25
    total_width = 0.56
    slot_width = total_width / max(n_sizes, 1)
    bar_width = slot_width * 0.78

    cmap = plt.get_cmap("Set2")
    colors = [cmap(i) for i in range(n_sizes)]

    for i, size in enumerate(sizes):
        values = np.asarray(size_to_values[size], dtype=float)
        offsets = x - total_width / 2 + (i + 0.5) * slot_width

        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            label=str(size),
            color=colors[i],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.95,
        )

        local_max = np.nanmax(values) if np.any(~np.isnan(values)) else 1.0
        offset = 0.015 * local_max

        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{val:.1f}" if abs(val) >= 10 else f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    current_vals = []
    for size_vals in size_to_values.values():
        current_vals.extend(size_vals)

    current_vals = np.asarray(current_vals, dtype=float)
    current_vals = current_vals[~np.isnan(current_vals)]

    if ylim is not None:
        ax.set_ylim(*ylim)
    elif len(current_vals) > 0:
        vmin = 0.0
        vmax = np.max(current_vals)
        margin = 0.12 * vmax
        ax.set_ylim(vmin, vmax + margin)


def plot_coverage_summary_report(
    compact_csv,
    output_prefix=None,
    max_sizes=5,
):
    """
    Read compact coverage summary CSV and generate comparison plots by size.

    Parameters
    ----------
    compact_csv : str or Path
        Path to *_coverage_summary_compact.csv
    output_prefix : str or None
        Prefix for output plot names.
        If None, derived from compact_csv by removing '_coverage_summary_compact.csv'
    """
    compact_csv = Path(compact_csv)

    if not compact_csv.exists():
        raise FileNotFoundError(f"Compact coverage CSV not found: {compact_csv}")

    if output_prefix is None:
        name = compact_csv.name
        suffix = "_coverage_summary_compact.csv"
        if name.endswith(suffix):
            output_prefix = str(compact_csv.with_name(name[:-len(suffix)]))
        else:
            output_prefix = str(compact_csv.with_suffix(""))

    rows = []
    with open(compact_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "size": int(row["size"]),
                "method": row["method"],
                "mean_min_dist_avg": float(row["mean_min_dist_avg"]),
                "p95_min_dist_avg": float(row["p95_min_dist_avg"]),
                "max_min_dist_avg": float(row["max_min_dist_avg"]),
                "entropy_avg": float(row["entropy_avg"]) if row["entropy_avg"] else np.nan,
                "normalized_entropy_avg": float(row["normalized_entropy_avg"]) if row["normalized_entropy_avg"] else np.nan,
                "occupied_clusters_avg": float(row["occupied_clusters_avg"]) if row["occupied_clusters_avg"] else np.nan,
                "n_repeats": int(row["n_repeats"]),
            })

    if not rows:
        logger.warning("[coverage_report] No rows found in compact CSV. Skipping plots.")
        return

    sizes_present = sorted(set(r["size"] for r in rows))
    preferred_order = ["fps", "kmeans", "random"]

    if not sizes_present:
        logger.warning("[coverage_report] No sizes found in compact CSV. Skipping plots.")
        return

    if len(sizes_present) > max_sizes:
        logger.info(
            f"[coverage_report] More than {max_sizes} sizes found; using first {max_sizes}: {sizes_present[:max_sizes]}"
            )

    methods_present = sorted(set(r["method"] for r in rows))
    methods = [m for m in preferred_order if m in methods_present]
    methods += sorted([m for m in methods_present if m not in preferred_order])

    mean_by_size = {}
    p95_by_size = {}
    max_by_size = {}
    entropy_by_size = {}

    for size in sizes_present[:max_sizes]:
        rows_size = [r for r in rows if r["size"] == size]
        row_map = {r["method"]: r for r in rows_size}

        mean_by_size[size] = [row_map[m]["mean_min_dist_avg"] if m in row_map else np.nan for m in methods]
        p95_by_size[size] = [row_map[m]["p95_min_dist_avg"] if m in row_map else np.nan for m in methods]
        max_by_size[size] = [row_map[m]["max_min_dist_avg"] if m in row_map else np.nan for m in methods]
        entropy_by_size[size] = [row_map[m]["normalized_entropy_avg"] if m in row_map else np.nan for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    _plot_grouped_metric_panel(
        axes[0],
        methods,
        mean_by_size,
        title="Mean Coverage Distance",
        ylabel="Mean nearest-selected distance",
        max_sizes=max_sizes,
    )

    _plot_grouped_metric_panel(
        axes[1],
        methods,
        p95_by_size,
        title="P95 Coverage Distance",
        ylabel="P95 nearest-selected distance",
        max_sizes=max_sizes,
    )

    _plot_grouped_metric_panel(
        axes[2],
        methods,
        max_by_size,
        title="Max Coverage Distance",
        ylabel="Max nearest-selected distance",
        max_sizes=max_sizes,
    )

    _plot_grouped_metric_panel(
        axes[3],
        methods,
        entropy_by_size,
        title="Normalized Entropy",
        ylabel="Entropy",
        ylim=(0.0, 1.05),
        max_sizes=max_sizes,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="Subset size",
            loc="upper center",
            ncol=min(len(labels), 5),
            bbox_to_anchor=(0.5, 0.98),
        )

    fig.suptitle("Coverage Summary Report Across Sizes", fontsize=15, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out_fn = f"{output_prefix}_coverage_report_all_sizes.png"
    fig.savefig(out_fn, dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[coverage_report] Saved: {out_fn}")
