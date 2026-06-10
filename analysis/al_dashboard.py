#!/usr/bin/env python3
"""Plot diagnostics from pool active-learning output."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_energy_trace(npz_path="pool_energy_trace.npz"):
    if not os.path.exists(npz_path):
        print(f"[Warning] {npz_path} not found. Skipping energy trace.")
        return

    data = np.load(npz_path)
    steps = data["steps"]
    mu = data["mu"]
    sigma = data["sigma"]
    bad = data["bad"]

    lower, upper = mu - sigma, mu + sigma

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, mu, label="50-pt MA of mu_E")
    ax.fill_between(steps, lower, upper, alpha=0.3, label="sigma (smoothed)")
    ax.scatter(
        steps[bad],
        mu[bad],
        marker="x",
        color="r",
        alpha=0.5,
        label="Failed geometry",
    )

    ax.set_xlabel("Pool frame index")
    ax.set_ylabel("Predicted energy")
    ax.set_title("Pool energy +/- uncertainty")
    ax.legend()
    plt.tight_layout()
    plt.savefig("al_dashboard_energy_trace.png", dpi=200)
    print("[INFO] Saved al_dashboard_energy_trace.png")
    plt.close()


def _to_bool(value):
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return False


def parse_diagnostics_file(path):
    pool_rows = []
    train_rows = []
    thresholds = {}
    in_pool = False
    in_train = False
    colnames = None

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()

            if stripped.startswith("# TRAIN DATASET"):
                in_train, in_pool = True, False
                continue

            if not in_pool and not in_train:
                if stripped.startswith("#") and "=" in stripped:
                    left, right = stripped.lstrip("#").split("=", 1)
                    try:
                        thresholds[left.strip()] = float(right.strip().split()[0])
                    except ValueError:
                        pass
                elif stripped.startswith("idx") and "shortlist" in stripped:
                    in_pool, colnames = True, stripped.split()
                continue

            if in_pool and not in_train:
                if stripped.startswith("#") or stripped.startswith("-") or not stripped:
                    continue
                parts = stripped.split()
                if colnames is None or len(parts) < len(colnames):
                    continue
                row = dict(zip(colnames, parts))
                try:
                    pool_rows.append(
                        {
                            "idx": int(row["idx"]),
                            "sE": float(row.get("sigmaE_atom", row.get("sE_atom", row.get("σE_atom")))),
                            "sFmax": float(row.get("sigmaF_max", row.get("sF_max", row.get("σF_max")))),
                            "sFmean": float(row.get("sigmaF_mean", row.get("sF_mean", row.get("σF_mean")))),
                            "Eabs_exp": float(row.get("Eabs_exp", np.nan)),
                            "Fabs_mean": float(row.get("Fabs_mean", np.nan)),
                            "Fmax": float(row.get("Fmax", row.get("F_mag"))),
                            "shortlist": _to_bool(row.get("shortlist", 0)),
                            "geom_ok": _to_bool(row.get("geom_ok", row.get("rdf_ok", 0))),
                            "caps_ok": _to_bool(row.get("caps_ok", row.get("pass_caps", 0))),
                            "cal_ok": _to_bool(row.get("cal_ok", 1)),
                            "ood": _to_bool(row.get("ood", 0)),
                        }
                    )
                except (ValueError, KeyError):
                    pass

            if in_train:
                if stripped.startswith("#") or stripped.startswith("-") or not stripped:
                    continue
                parts = stripped.split()
                if len(parts) >= 5:
                    try:
                        train_rows.append(
                            {
                                "idx": int(parts[0]),
                                "sE": float(parts[1]),
                                "sFmax": float(parts[2]),
                                "sFmean": float(parts[3]),
                                "Fmax": float(parts[4]),
                            }
                        )
                    except ValueError:
                        pass

    return pool_rows, train_rows, thresholds


def _array(rows, key, dtype=float):
    return np.array([row[key] for row in rows], dtype=dtype)


def _draw_lines(ax, low, hi, hard):
    if np.isfinite(low):
        ax.axhline(low, ls="--", lw=1.0, color="gray")
    if np.isfinite(hi):
        ax.axhline(hi, ls="--", lw=1.0, color="gray")
    if np.isfinite(hard):
        ax.axhline(hard, ls="--", lw=1.0, color="red")


def plot_uncertainty_panels(diag_file="al_pool_per_frame_diagnostics.txt"):
    if not os.path.exists(diag_file):
        print(f"[Warning] {diag_file} not found. Skipping panels.")
        return

    pool_rows, train_rows, thr = parse_diagnostics_file(diag_file)
    if not pool_rows:
        print(f"[Warning] No pool rows parsed from {diag_file}.")
        return

    pool_idx = _array(pool_rows, "idx", int)
    pool_sel = _array(pool_rows, "shortlist", bool)
    pool_geom_ok = _array(pool_rows, "geom_ok", bool)
    pool_caps_ok = _array(pool_rows, "caps_ok", bool)
    pool_ood = _array(pool_rows, "ood", bool)
    pool_cal_ok = _array(pool_rows, "cal_ok", bool)

    train_idx = _array(train_rows, "idx", int) if train_rows else np.array([])
    sel_x = pool_idx[pool_sel]
    failed_geom_idx = pool_idx[~pool_geom_ok]
    failed_caps_idx = pool_idx[pool_geom_ok & ~pool_caps_ok]
    ood_idx = pool_idx[pool_ood]

    fig, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True, constrained_layout=True)

    def plot_panel(ax, metric_key, train_key, ylabel, thr_low, thr_hi, thr_hard, mult=1.0):
        metric_pool = _array(pool_rows, metric_key)
        metric_train = _array(train_rows, train_key) if train_rows else np.array([])
        ax.plot(pool_idx, metric_pool * mult, lw=1, color="blue", alpha=0.45, label="Pool")
        ax.scatter(
            failed_caps_idx,
            metric_pool[pool_geom_ok & ~pool_caps_ok] * mult,
            color="orange",
            marker="x",
            alpha=0.7,
            s=18,
            label="Failed caps",
        )
        ax.scatter(
            failed_geom_idx,
            metric_pool[~pool_geom_ok] * mult,
            color="red",
            marker="x",
            alpha=0.35,
            s=18,
            label="Failed geom",
        )
        ax.scatter(
            ood_idx,
            metric_pool[pool_ood] * mult,
            color="purple",
            marker="^",
            alpha=0.75,
            s=28,
            label="OOD risk",
        )
        ax.scatter(
            sel_x,
            metric_pool[pool_sel] * mult,
            color="k",
            marker="o",
            s=36,
            label="Shortlist",
            zorder=5,
        )
        if train_idx.size > 0:
            ax.plot(train_idx, metric_train * mult, lw=1, color="green", alpha=0.75, label="Train")
        _draw_lines(
            ax,
            thr.get(thr_low, np.nan) * mult,
            thr.get(thr_hi, np.nan) * mult,
            thr.get(thr_hard, np.nan) * mult,
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plot_panel(
        axes[0],
        "sE",
        "sE",
        "sigma_E/atom (meV)",
        "thr_sigma_E_low",
        "thr_sigma_E_hi_eff",
        "hard_sigma_E_atom_min",
        mult=1000,
    )
    axes[0].legend(loc="upper right", fontsize=8, ncol=3)

    plot_panel(
        axes[1],
        "sFmax",
        "sFmax",
        "sigma F_max (eV/A)",
        "thr_sigma_F",
        "thr_sigma_F_hi_eff",
        "hard_sigma_F_max_min",
    )
    plot_panel(
        axes[2],
        "sFmean",
        "sFmean",
        "sigma F_mean (eV/A)",
        "thr_sigma_Fmean",
        "thr_sigma_Fmean_hi_eff",
        "hard_sigma_F_mean_min",
    )
    plot_panel(
        axes[3],
        "Fmax",
        "Fmax",
        "|F|_max (eV/A)",
        "thr_Fmag",
        "thr_Fmag_hi_eff",
        "train_Fmax_hard_cap",
    )
    plot_panel(axes[4], "Eabs_exp", "sE", "E[|dE|]/atom (meV)", "", "", "", mult=1000)
    plot_panel(axes[5], "Fabs_mean", "sFmean", "E[|dF|]_mean (eV/A)", "", "", "")
    axes[5].set_xlabel("Frame index")

    plt.savefig("al_dashboard_panels.png", dpi=300)
    print("[INFO] Saved al_dashboard_panels.png")
    print(
        "[INFO] Parsed AL diagnostics: "
        f"pool={len(pool_rows)}, shortlist={int(pool_sel.sum())}, "
        f"ood_risk={int(pool_ood.sum())}, calibration_in_support={int(pool_cal_ok.sum())}."
    )
    plt.close()


def plot_ood_summary(diag_file="al_pool_per_frame_diagnostics.txt"):
    if not os.path.exists(diag_file):
        return
    pool_rows, _, _ = parse_diagnostics_file(diag_file)
    if not pool_rows:
        return

    pool_idx = _array(pool_rows, "idx", int)
    pool_ood = _array(pool_rows, "ood", bool)
    pool_sel = _array(pool_rows, "shortlist", bool)
    eabs = _array(pool_rows, "Eabs_exp") * 1000.0
    fabs = _array(pool_rows, "Fabs_mean")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    for ax, values, ylabel in (
        (axes[0], eabs, "E[|dE|]/atom (meV)"),
        (axes[1], fabs, "E[|dF|]_mean (eV/A)"),
    ):
        ax.scatter(pool_idx[~pool_ood], values[~pool_ood], s=10, alpha=0.4, label="In support")
        ax.scatter(pool_idx[pool_ood], values[pool_ood], s=18, alpha=0.8, color="purple", label="OOD risk")
        ax.scatter(pool_idx[pool_sel], values[pool_sel], s=35, color="black", marker="o", label="Shortlist")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("Frame index")
    plt.savefig("al_dashboard_ood_expected_error.png", dpi=250)
    print("[INFO] Saved al_dashboard_ood_expected_error.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AL Dashboard Plotter")
    parser.add_argument("--npz", default="pool_energy_trace.npz")
    parser.add_argument("--txt", default="al_pool_per_frame_diagnostics.txt")
    args = parser.parse_args()

    plot_energy_trace(args.npz)
    plot_uncertainty_panels(args.txt)
    plot_ood_summary(args.txt)
