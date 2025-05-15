# uq_metrics_calculator.py
"""Uncertainty‑quantification metrics **calculator – 2025 refactor**

This file exposes **one public entry‑point** – :func:`run_uq_metrics` – that
covers everything the former 2 000‑line implementation did while being
readable and unit‑testable.  Legacy code that still calls
``calculate_uq_metrics`` keeps working thanks to a shim at the bottom.

--------------------------------------------------------------------------
API
---
.. autofunction:: run_uq_metrics
.. autofunction:: calculate_uq_metrics  (compat)
--------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import logging

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm, spearmanr, ks_2samp

# ---------------------------------------------------------------------------
# Logging – configured *once* when the module is imported
# ---------------------------------------------------------------------------
_LOGGER = logging.getLogger("uq_metrics")
if not _LOGGER.handlers:  # avoid duplicate handlers under re‑import
    _LOGGER.setLevel(logging.INFO)
    _h = logging.FileHandler("metrics.log", mode="a", encoding="utf‑8")
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    _LOGGER.addHandler(_h)


# ---------------------------------------------------------------------------
# Dataclasses & helpers
# ---------------------------------------------------------------------------
@dataclass
class MetricResult:
    name: str
    value: float
    group: str  # e.g. "probabilistic", "calibration", "diagnostic"


# ---- Calibration -----------------------------------------------------------
class VarianceScalingCalibrator:
    """One‑parameter variance scaling (a.k.a. temperature scaling).

    Finds *s* so that NLL is minimised ⇒ closed‑form solution.
    """

    def __init__(self) -> None:
        self.s: float = 1.0

    # pylint: disable=invalid-name
    @staticmethod
    def _closed_form_s(delta: np.ndarray, sigma: np.ndarray) -> float:
        return float(np.sqrt(np.nanmean((delta ** 2) / (sigma ** 2))))

    def fit(self, delta: np.ndarray, sigma: np.ndarray) -> "VarianceScalingCalibrator":
        self.s = self._closed_form_s(delta, sigma)
        return self

    def transform(self, sigma: np.ndarray) -> np.ndarray:
        return self.s * sigma

    # convenience ----------------------------------------------------------
    def fit_transform(self, delta: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        self.fit(delta, sigma)
        return self.transform(sigma)


# ---- Metric primitives -----------------------------------------------------
_EPS = 1e-12

def _safe_sigma(sigma: np.ndarray) -> np.ndarray:
    """Clamp too‑small σ to avoid /0 and log(0)."""
    return np.where(sigma < _EPS, _EPS, sigma)


def nll(delta: np.ndarray, sigma: np.ndarray) -> float:
    """Mean Negative‑Log‑Likelihood (Gaussian)."""
    sigma = _safe_sigma(sigma)
    return float(np.nanmean(0.5 * np.log(2 * np.pi * sigma ** 2) + (delta ** 2) / (2 * sigma ** 2)))

def rll(delta: np.ndarray, sigma: np.ndarray) -> float:
    """Relative Log‑Likelihood (%) compared to baseline and oracle."""
    sigma = _safe_sigma(sigma)
    nll_vals = 0.5 * np.log(2 * np.pi * sigma ** 2) + (delta ** 2) / (2 * sigma ** 2)
    sum_nll = np.nansum(nll_vals)

    # baseline = single global std
    sigma_base = max(np.nanstd(delta), _EPS)
    nll_base = 0.5 * np.log(2 * np.pi * sigma_base ** 2) + (delta ** 2) / (2 * sigma_base ** 2)
    sum_base = np.nansum(nll_base)

    # oracle = |delta| as sigma
    sigma_oracle = np.maximum(np.abs(delta), _EPS)
    nll_oracle = 0.5 * np.log(2 * np.pi * sigma_oracle ** 2) + (delta ** 2) / (2 * sigma_oracle ** 2)
    sum_oracle = np.nansum(nll_oracle)

    denom = sum_oracle - sum_base
    if abs(denom) < _EPS:
        return float("nan")
    return float((sum_nll - sum_base) / denom * 100.0)

def crps_gaussian(delta: np.ndarray, sigma: np.ndarray) -> float:
    """Closed‑form CRPS for a Gaussian predictive distribution."""
    from scipy.stats import norm  # local import; avoids heavy dependency at import time

    sigma = _safe_sigma(sigma)
    z = delta / sigma
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return float(np.nanmean(crps))


def ence(delta: np.ndarray, sigma: np.ndarray, *, bins: int = 10) -> float:
    """Expected *Normalised* Calibration Error (force/energy flavours)."""
    if len(delta) < bins:
        return float("nan")
    idx = np.argsort(sigma)
    sigma_sorted = sigma[idx]
    delta_sorted = delta[idx]
    split = np.array_split(np.arange(len(delta)), bins)
    rmv, rmse = [], []
    for s in split:
        if len(s) == 0:
            continue
        rmv.append(np.sqrt(np.mean(sigma_sorted[s] ** 2)))
        rmse.append(np.sqrt(np.mean(delta_sorted[s] ** 2)))
    rmv, rmse = np.asarray(rmv), np.asarray(rmse)
    return float(np.mean(np.abs(rmv - rmse) / rmv))


def picp(delta: np.ndarray, sigma: np.ndarray, *, alpha: float = 0.05) -> float:
    """Prediction‑interval coverage probability for (1‑α) central interval."""
    sigma = _safe_sigma(sigma)
    z = np.abs(delta) / sigma
    z_alpha = norm.ppf(1 - alpha / 2)
    return float(np.nanmean(z <= z_alpha))


def sharpness(sigma: np.ndarray) -> float:
    return float(np.nanmean(sigma ** 2))


def cv_sigma(sigma: np.ndarray) -> float:
    m = np.nanmean(sigma)
    return float(np.nanstd(sigma) / m) if m > _EPS else float("nan")


def spearman_corr(delta: np.ndarray, sigma: np.ndarray) -> float:
    if np.allclose(sigma, sigma[0]):  # constant array → undefined corr
        return float("nan")
    mask = ~np.isnan(delta) & ~np.isnan(sigma)
    if mask.sum() < 2:
        return float("nan")
    rho, _ = spearmanr(np.abs(delta[mask]), sigma[mask])
    return float(rho)


def ks_z(delta: np.ndarray, sigma: np.ndarray) -> float:
    """KS statistic between z‑scores and 𝓝(0,1)."""
    z = delta / _safe_sigma(sigma)
    z = z[np.isfinite(z)]
    if len(z) < 2:
        return float("nan")
    ks_stat, _ = ks_2samp(z, np.random.normal(size=len(z)))
    return float(ks_stat)


# ---------------------------------------------------------------------------
#  Main orchestrator
# ---------------------------------------------------------------------------

def run_uq_metrics(
    *,
    stats: "MLFFStats",  # quoted for forward‑decl.
    sigma_comp: np.ndarray,
    sigma_atom: np.ndarray,
    sigma_energy: Optional[np.ndarray] = None,
    split: str = "Eval",  # "Train" | "Eval"
    tag: str = "ensemble",
    log_path: str | Path = "metrics.log",
    ensemble_size: Optional[int] = None,
) -> Dict:
    """Compute, calibrate & log UQ metrics.

    Parameters
    ----------
    stats
        Pre‑computed force/energy residuals & masks.
    sigma_comp / sigma_atom / sigma_energy
        Uncalibrated predictive std at component/atom/frame levels.
    split
        Name of dataset (for log grouping).
    tag
        Short label (e.g. ``ensemble``, ``error_model`` …).
    log_path
        Destination file is automatically opened in *append* mode.
    ensemble_size
        Only used for labelling.

    Returns
    -------
    dict
        {"metrics": {name: value, …}, "npz_path": str}
    """

    # ---------------- residuals & shapes --------------------------------
    delta_comp = stats.all_force_residuals.reshape(-1)
    delta_atom = stats.force_rmse_per_atom
    delta_energy = stats.delta_E_frame if sigma_energy is not None else None

    # select mask ---------------------------------------------------------
    frame_mask = stats.train_mask if split.lower() == "train" else stats.eval_mask
    atom_mask = stats._get_atom_mask(frame_mask)
    comp_mask = np.repeat(atom_mask, 3)

    delta_c = delta_comp[comp_mask]
    sigma_c = sigma_comp[comp_mask]

    delta_a = delta_atom[atom_mask]
    sigma_a = sigma_atom[atom_mask]

    if delta_energy is not None:
        delta_e = delta_energy[frame_mask]
        sigma_e = sigma_energy[frame_mask]
    else:
        delta_e = sigma_e = None

    # ---------------- calibration ---------------------------------------
    calib = VarianceScalingCalibrator().fit(delta_c, sigma_c)
    sigma_c_cal = calib.transform(sigma_c)
    sigma_a_cal = calib.transform(sigma_a)
    if sigma_e is not None:
        calib_E = VarianceScalingCalibrator().fit(delta_e, sigma_e)
        sigma_e_cal = calib_E.transform(sigma_e)
    else:
        sigma_e_cal = None

    # ---------------- metrics -------------------------------------------
    metrics: list[MetricResult] = [
        MetricResult("NLL", nll(delta_c, sigma_c), "probabilistic"),
        MetricResult("NLL_cal", nll(delta_c, sigma_c_cal), "probabilistic"),
        MetricResult("RLL", rll(delta_c, sigma_c), "probabilistic"),
        MetricResult("RLL_cal", rll(delta_c, sigma_c_cal), "probabilistic"),
        MetricResult("CRPS", crps_gaussian(delta_c, sigma_c), "probabilistic"),
        MetricResult("ENCE", ence(delta_c, sigma_c), "calibration"),
        MetricResult("ENCE_cal", ence(delta_c, sigma_c_cal), "calibration"),
        MetricResult("PICP95", picp(delta_c, sigma_c, alpha=0.05), "calibration"),
        MetricResult("Sharpness", sharpness(sigma_c), "dispersion"),
        MetricResult("CV", cv_sigma(sigma_c), "dispersion"),
        MetricResult("ρ_spearman", spearman_corr(delta_a, sigma_a), "discrimination"),
        MetricResult("KS_z", ks_z(delta_c, sigma_c), "diagnostic"),
    ]
    if delta_e is not None:
        metrics.extend([
            MetricResult("NLL_E", nll(delta_e, sigma_e), "energy"),
            MetricResult("NLL_E_cal", nll(delta_e, sigma_e_cal), "energy"),
            MetricResult("RLL_E", rll(delta_e, sigma_e), "probabilistic"),
            MetricResult("RLL_E_cal", rll(delta_e, sigma_e_cal), "probabilistic"),
            MetricResult("ENCE_E", ence(delta_e, sigma_e), "energy"),
            MetricResult("ENCE_E_cal", ence(delta_e, sigma_e_cal), "energy"),
        ])

    # ---------------- logging -------------------------------------------
    ens_str = f" [ens={ensemble_size}]" if ensemble_size else ""
    _LOGGER.info("%s | %s%s", split.upper(), tag, ens_str)
    for m in metrics:
        _LOGGER.info("  %-15s : %.6f", m.name, m.value)

    # ---------------- save npz (for downstream plotting) -----------------
    npz_path = None
    try:
        # 1) build scalar‐metrics dict
        scalar_metrics = {m.name: m.value for m in metrics}

        # 2) coverage curves at a grid of nominal probabilities
        p_thresholds = np.linspace(0.0, 1.0, 21)
        coverage_uncal = np.array([
            picp(delta_c, sigma_c, alpha=1-p) for p in p_thresholds
        ])
        coverage_cal   = np.array([
            picp(delta_c, sigma_c_cal, alpha=1-p) for p in p_thresholds
        ])
        # same for energy, if present
        if delta_e is not None:
            coverage_uncal_e = np.array([
                picp(delta_e, sigma_e, alpha=1-p) for p in p_thresholds
            ])
            coverage_cal_e   = np.array([
                picp(delta_e, sigma_e_cal, alpha=1-p) for p in p_thresholds
            ])
        else:
            coverage_uncal_e = np.array([])
            coverage_cal_e   = np.array([])

        # Prepare filename
        base = f"uq_plot_data_{split.lower()}_{tag.lower()}"
        if ensemble_size:
            base += f"_ens{ensemble_size}"
        npz_path = Path("uq_plots") / f"{base}.npz"
        npz_path.parent.mkdir(exist_ok=True)

        # 3) save everything
        np.savez_compressed(
            npz_path,
            # force arrays
            delta_comp=delta_c,
            sigma_comp_uncal=sigma_c,
            sigma_comp_cal=sigma_c_cal,
            # energy arrays
            delta_energy=delta_e,
            sigma_energy_uncal=sigma_e,
            sigma_energy_cal=sigma_e_cal,
            # new UQ‐scalar metrics & coverage
            scalar_metrics=scalar_metrics,
            p_thresholds=p_thresholds,
            coverage_uncal=coverage_uncal,
            coverage_cal=coverage_cal,
            coverage_uncal_e=coverage_uncal_e,
            coverage_cal_e=coverage_cal_e,
        )
        _LOGGER.info("  saved ➜ %s", npz_path)
    except Exception as exc:  # pylint: disable=broad-except
        _LOGGER.warning("could not save plot data: %s", exc)

    # final ---------------------------------------------------------------
    return {
        "metrics": {m.name: m.value for m in metrics},
        "npz_path": str(npz_path) if npz_path else None,
    }


# ---------------------------------------------------------------------------
# Back‑compat function signature
# ---------------------------------------------------------------------------

def calculate_uq_metrics(  # noqa: C901  (complexity ignored – thin wrapper)
    stats: "MLFFStats",
    sigma_comp_all: np.ndarray,
    sigma_atom_all: np.ndarray,
    sigma_energy_all: Optional[np.ndarray] = None,
    set_name: str = "Eval",
    set_uq: str = "ensemble",
    log_file: str = "metrics.log",
    ensemble_size: Optional[int] = None,
):
    """Thin wrapper that forwards to :func:`run_uq_metrics`.

    This lets legacy code (*evaluate.py*, notebooks, …) keep working until
    everything migrates to the new explicit API.
    """

    return run_uq_metrics(
        stats=stats,
        sigma_comp=sigma_comp_all,
        sigma_atom=sigma_atom_all,
        sigma_energy=sigma_energy_all,
        split=set_name,
        tag=set_uq,
        log_path=log_file,
        ensemble_size=ensemble_size,
    )

