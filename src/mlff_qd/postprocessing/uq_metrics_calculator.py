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
from sklearn.isotonic import IsotonicRegression
from scipy.integrate import trapezoid
from scipy.stats import norm, spearmanr, ks_2samp, normaltest

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

class IsotonicCalibrator:
    """Non-parametric monotonic calibration σ' = f(σ).

    Learns a monotone mapping so that the *z*-scores follow 𝓝(0,1)
    (i.e. E[|δ|] ≈ σ').  Works even when the ensemble residuals are
    non-Gaussian or heteroscedastic.

    Notes
    -----
    • The target we fit is |δ| (absolute error).
    • We force the mapping through the origin and keep it non-decreasing.
    """

    def __init__(self) -> None:
        self._iso = IsotonicRegression(y_min=0.0, increasing=True, out_of_bounds="clip")

    def fit(self, delta: np.ndarray, sigma: np.ndarray) -> "IsotonicCalibrator":
        abs_delta = np.abs(delta)
        sigma = _safe_sigma(sigma)           # reuse helper from your module
        # sort by sigma to satisfy isotonic-regression requirements
        idx = np.argsort(sigma)
        self._iso.fit(sigma[idx], abs_delta[idx])
        return self

    def transform(self, sigma: np.ndarray) -> np.ndarray:
        return self._iso.predict(_safe_sigma(sigma))

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


def normality_pvalue(z: np.ndarray) -> float:
    """Return p-value of D’Agostino K² normality test (nan if <8 samples)."""
    z = z[np.isfinite(z)]
    if len(z) < 8:
        return float("nan")
    _, p = normaltest(z)
    return float(p)

from scipy.special import gammaln

def student_t_nll(y, mu, sigma, nu=3.0):
    # Handles 1D arrays: y (targets), mu (predictions), sigma (uncertainties)
    z = (y - mu) / sigma
    return (
        0.5 * np.log(np.pi * nu * sigma**2)
        + gammaln(0.5 * (nu + 1)) - gammaln(0.5 * nu)
        + 0.5 * (nu + 1) * np.log(1 + (z**2) / nu)
    )
def mean_student_t_nll(y, mu, sigma, nu=3.0):
    return float(np.nanmean(student_t_nll(y, mu, sigma, nu)))

from textwrap import indent

def verdict_short(v):
    return {'good': 'G', 'average': 'A', 'poor': 'P'}.get(v, '-')

def pretty_table_full(row_names, col_names, table, verdicts):
    col_w = max(8, max(len(c) for c in col_names))
    header = "│".join([f"{'Metric':<{col_w}}"] + [f"{c:^{col_w+7}}" for c in col_names])
    lines = [header, "─" * (col_w + (col_w+8) * len(col_names))]
    for i, name in enumerate(row_names):
        vals = [
            f"{table[i][j]:10.4f} [{verdict_short(verdicts[i][j])}]"
            for j in range(len(col_names))
        ]
        lines.append("│".join([f"{name:<{col_w}}"] + vals))
    return indent("\n".join(lines), "   ")

# Thresholds and verdict logic
_THRESHOLDS = {
    # RLL: +100 oracle, 0 baseline, <0 worse (larger is better)
    "RLL_raw": (0.20, 0.00), "RLL_calVAR": (0.20, 0.00), "RLL_calISO": (0.20, 0.00),
    "RLL_raw_E": (0.20, 0.00), "RLL_calVAR_E": (0.20, 0.00), "RLL_calISO_E": (0.20, 0.00),
    "CRPS_raw": (0.05, 0.10), "CRPS_calVAR": (0.05, 0.10), "CRPS_calISO": (0.05, 0.10),
    "CRPS_raw_E": (0.05, 0.10), "CRPS_calVAR_E": (0.05, 0.10), "CRPS_calISO_E": (0.05, 0.10),
    "ENCE_raw": (0.10, 0.20), "ENCE_calVAR": (0.10, 0.20), "ENCE_calISO": (0.10, 0.20),
    "ENCE_raw_E": (0.10, 0.20), "ENCE_calVAR_E": (0.10, 0.20), "ENCE_calISO_E": (0.10, 0.20),
    "PICP95_raw": (0.97, 1.03), "PICP95_calVAR": (0.97, 1.03), "PICP95_calISO": (0.97, 1.03),
    "PICP95_raw_E": (0.97, 1.03), "PICP95_calVAR_E": (0.97, 1.03), "PICP95_calISO_E": (0.97, 1.03),
    "Sharpness_raw": (float('inf'), float('inf')), "Sharpness_calVAR": (float('inf'), float('inf')), "Sharpness_calISO": (float('inf'), float('inf')),
    "Sharpness_raw_E": (float('inf'), float('inf')), "Sharpness_calVAR_E": (float('inf'), float('inf')), "Sharpness_calISO_E": (float('inf'), float('inf')),
    "CV_raw": (0.30, 0.60), "CV_calVAR": (0.30, 0.60), "CV_calISO": (0.30, 0.60),
    "CV_raw_E": (0.30, 0.60), "CV_calVAR_E": (0.30, 0.60), "CV_calISO_E": (0.30, 0.60),
    "Spearman_raw": (-float('inf'), 0.60), "Spearman_calVAR": (-float('inf'), 0.60), "Spearman_calISO": (-float('inf'), 0.60),
    "Spearman_raw_E": (-float('inf'), 0.60), "Spearman_calVAR_E": (-float('inf'), 0.60), "Spearman_calISO_E": (-float('inf'), 0.60),
    "KS_z_raw": (0.05, 0.10), "KS_z_calVAR": (0.05, 0.10), "KS_z_calISO": (0.05, 0.10),
    "KS_z_raw_E": (0.05, 0.10), "KS_z_calVAR_E": (0.05, 0.10), "KS_z_calISO_E": (0.05, 0.10),
    "Normal_p_raw": (float('inf'), 0.05), "Normal_p_calVAR": (float('inf'), 0.05), "Normal_p_calISO": (float('inf'), 0.05),
    "Normal_p_raw_E": (float('inf'), 0.05), "Normal_p_calVAR_E": (float('inf'), 0.05), "Normal_p_calISO_E": (float('inf'), 0.05),
}
_LARGER_IS_BETTER = {
    "RLL_raw", "RLL_calVAR", "RLL_calISO",
    "RLL_raw_E", "RLL_calVAR_E", "RLL_calISO_E",
    "Spearman_raw", "Spearman_calVAR", "Spearman_calISO",
    "Spearman_raw_E", "Spearman_calVAR_E", "Spearman_calISO_E",
    "Normal_p_raw", "Normal_p_calVAR", "Normal_p_calISO",
    "Normal_p_raw_E", "Normal_p_calVAR_E", "Normal_p_calISO_E"
}

def qualitative_label(val: float, name: str, ctx: dict | None = None) -> str:
    if np.isnan(val): return "n/a"
    # NLL or t-NLL is dataset dependent, so compare with baseline
    if (name.startswith("NLL") or name.startswith("t-NLL")) and ctx:
        base = ctx["NLL_base_E"] if "E" in name else ctx["NLL_base"]
        if val <= base - 0.05: return "good"
        if val < base: return "average"
        return "poor"
    # Everything else
    if name not in _THRESHOLDS: return "-"
    good, avg = _THRESHOLDS[name]
    if name in _LARGER_IS_BETTER:
        if val >= good: return "good"
        if val >= avg: return "average"
        return "poor"
    if val <= good: return "good"
    if val <= avg: return "average"
    return "poor"

# ---------------------------------------------------------------------------
#  Main orchestrator
# ---------------------------------------------------------------------------

def run_uq_metrics(
    *,
    stats: "MLFFStats",
    sigma_comp: np.ndarray,
    sigma_atom: np.ndarray,
    sigma_energy: Optional[np.ndarray] = None,
    split: str = "Eval",
    tag: str = "ensemble",
    log_path: str | Path = "metrics.log",
    ensemble_size: Optional[int] = None,
) -> Dict:
    """
    Compute, calibrate (variance + isotonic) & log UQ metrics.
    """

    # ---------------- Residuals and Masks ----------------------------
    delta_comp = stats.all_force_residuals.reshape(-1)
    delta_atom = stats.force_rmse_per_atom
    delta_energy = stats.delta_E_frame if sigma_energy is not None else None

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

    print(delta_e.mean(), np.median(delta_e))
    # --------- Calibration: Variance scaling and Isotonic Regression -----------
    cal_var_F = VarianceScalingCalibrator().fit(delta_c, sigma_c)
    sigma_c_cal_var = cal_var_F.transform(sigma_c)
    sigma_a_cal_var = cal_var_F.transform(sigma_a)
    # Isotonic
    try:
        from sklearn.isotonic import IsotonicRegression
        class _Iso:
            def __init__(self):
                self._iso = IsotonicRegression(y_min=0.0, increasing=True, out_of_bounds="clip")
            def fit(self, d, s):
                idx = np.argsort(s)
                self._iso.fit(s[idx], np.abs(d[idx]))
                return self
            def transform(self, s):
                return self._iso.predict(_safe_sigma(s))
        cal_iso_F = _Iso().fit(delta_c, sigma_c)
        sigma_c_cal_iso = cal_iso_F.transform(sigma_c)
        sigma_a_cal_iso = cal_iso_F.transform(sigma_a)
    except Exception:
        cal_iso_F = None
        sigma_c_cal_iso = sigma_c
        sigma_a_cal_iso = sigma_a

    # Energies
    if delta_e is not None:
        cal_var_E = VarianceScalingCalibrator().fit(delta_e, sigma_e)
        sigma_e_cal_var = cal_var_E.transform(sigma_e)
        try:
            cal_iso_E = _Iso().fit(delta_e, sigma_e)
            sigma_e_cal_iso = cal_iso_E.transform(sigma_e)
        except Exception:
            sigma_e_cal_iso = sigma_e
    else:
        sigma_e_cal_var = sigma_e_cal_iso = None

    # ----------- Normality checks (p-values) --------------------------
    z_comp      = delta_c / _safe_sigma(sigma_c)
    z_comp_cal  = delta_c / _safe_sigma(sigma_c_cal_var)
    z_comp_iso  = delta_c / _safe_sigma(sigma_c_cal_iso)
    z_energy    = delta_e / _safe_sigma(sigma_e) if delta_e is not None else None
    z_energy_cal= delta_e / _safe_sigma(sigma_e_cal_var) if delta_e is not None else None
    z_energy_iso= delta_e / _safe_sigma(sigma_e_cal_iso) if delta_e is not None else None

    p_comp      = normality_pvalue(z_comp)
    p_comp_cal  = normality_pvalue(z_comp_cal)
    p_comp_iso  = normality_pvalue(z_comp_iso)
    p_energy    = normality_pvalue(z_energy) if delta_e is not None else float("nan")
    p_energy_cal= normality_pvalue(z_energy_cal) if delta_e is not None else float("nan")
    p_energy_iso= normality_pvalue(z_energy_iso) if delta_e is not None else float("nan")

    # --- Report normality for forces/energy
    def normality_report(z, p, label, delta=None, sigma=None):
        verdict = "Gaussian enough ✅" if p >= 0.05 else "NOT Gaussian ⚠️"
        print(f"Normality test [{label}]: p = {p:.3g} → {verdict}")
        if p < 0.05:
            print(f"  [!] WARNING: Residuals are NOT Gaussian. Probabilistic metrics (NLL, RLL, CRPS) may be unreliable for {label}.")
            if delta is not None and sigma is not None:
                st_nll = mean_student_t_nll(delta, np.zeros_like(delta), _safe_sigma(sigma))
                print(f"      Student-t NLL for {label}: {st_nll:.4f} (use as robust alternative to NLL)")
                return st_nll
        else:
            print(f"  Residuals look Gaussian for {label}, probabilistic metrics can be trusted.")
        return None

    student_t_nlls = {}  # Store results for metrics list (optional)
    
    student_t_nlls['forces_raw'] = normality_report(z_comp, p_comp, "Forces (raw)", delta_c, sigma_c)
    student_t_nlls['forces_var'] = normality_report(z_comp_cal, p_comp_cal, "Forces (var)", delta_c, sigma_c_cal_var)
    student_t_nlls['forces_iso'] = normality_report(z_comp_iso, p_comp_iso, "Forces (iso)", delta_c, sigma_c_cal_iso)
    
    if delta_e is not None:
        student_t_nlls['energy_raw'] = normality_report(z_energy, p_energy, "Energies (raw)", delta_e, sigma_e)
        student_t_nlls['energy_var'] = normality_report(z_energy_cal, p_energy_cal, "Energies (var)", delta_e, sigma_e_cal_var)
        student_t_nlls['energy_iso'] = normality_report(z_energy_iso, p_energy_iso, "Energies (iso)", delta_e, sigma_e_cal_iso)
    print()
    
    # --- Legend for Verdicts
    print("Legend for verdicts in table:")
    print("   G = Good   |   A = Average   |   P = Poor\n")
    
    # --- Legend for Metric Types
    print("Metric Types:")
    print("   Probabilistic: NLL, RLL, CRPS")
    print("   Calibration  : ENCE, PICP95")
    print("   Dispersion   : Sharpness, CV")
    print("   Discrimination: Spearman")
    print("   Diagnostic   : KS_z, Normal_p")
    print()
    
    # ------------ Baseline NLL calculation ----------------------------
    sigma_base_c = max(np.nanstd(delta_c), _EPS)
    nll_base_c = 0.5 * np.log(2 * np.pi * sigma_base_c ** 2) + (delta_c ** 2) / (2 * sigma_base_c ** 2)
    mean_nll_base_c = float(np.nanmean(nll_base_c))
    print(f"NLL baseline is: {mean_nll_base_c}")
    if delta_e is not None:
        sigma_base_e = max(np.nanstd(delta_e), _EPS)
        nll_base_e = 0.5 * np.log(2 * np.pi * sigma_base_e ** 2) + (delta_e ** 2) / (2 * sigma_base_e ** 2)
        mean_nll_base_e = float(np.nanmean(nll_base_e))
        print(f"NLL_E baseline is: {mean_nll_base_e}")
    else:
        mean_nll_base_e = float("nan")

    # --------------- Helper: Block of metrics -------------------------
    def _metric_block(d, s, z, p, label):
        """Metrics that need *matching* shapes: d and s."""
        return {
            f"NLL_{label}": nll(d, s),
            f"RLL_{label}": rll(d, s),
            f"CRPS_{label}": crps_gaussian(d, s),
            f"ENCE_{label}": ence(d, s),
            f"PICP95_{label}": picp(d, s, alpha=0.05),
            f"Sharpness_{label}": sharpness(s),
            f"CV_{label}": cv_sigma(s),
            f"KS_z_{label}": ks_z(d, s),
            f"Normal_p_{label}": p,
        }

    # Forces: raw, variance, isotonic
    m_raw_F = _metric_block(delta_c, sigma_c,           z_comp,      p_comp,      "raw")
    m_var_F = _metric_block(delta_c, sigma_c_cal_var,   z_comp_cal,  p_comp_cal,  "calVAR")
    m_iso_F = _metric_block(delta_c, sigma_c_cal_iso,   z_comp_iso,  p_comp_iso,  "calISO")
    # Energies if present
    if delta_e is not None:
        m_raw_E = _metric_block(delta_e, sigma_e,              z_energy,      p_energy,      "raw_E")
        m_var_E = _metric_block(delta_e, sigma_e_cal_var,      z_energy_cal,  p_energy_cal,  "calVAR_E")
        m_iso_E = _metric_block(delta_e, sigma_e_cal_iso,      z_energy_iso,  p_energy_iso,  "calISO_E")
    else:
        m_raw_E = m_var_E = m_iso_E = {}

    # Forces: Student-t NLL for each calibration
    tNLL_raw      = mean_student_t_nll(delta_c, np.zeros_like(delta_c), _safe_sigma(sigma_c))
    tNLL_calVAR   = mean_student_t_nll(delta_c, np.zeros_like(delta_c), _safe_sigma(sigma_c_cal_var))
    tNLL_calISO   = mean_student_t_nll(delta_c, np.zeros_like(delta_c), _safe_sigma(sigma_c_cal_iso))
    
    # Energies (if present)
    if delta_e is not None:
        tNLL_raw_E    = mean_student_t_nll(delta_e, np.zeros_like(delta_e), _safe_sigma(sigma_e))
        tNLL_calVAR_E = mean_student_t_nll(delta_e, np.zeros_like(delta_e), _safe_sigma(sigma_e_cal_var))
        tNLL_calISO_E = mean_student_t_nll(delta_e, np.zeros_like(delta_e), _safe_sigma(sigma_e_cal_iso))

    # Define metrics and verdicts for table
    row_names = ["NLL", "t-NLL", "RLL", "CRPS", "ENCE", "PICP95", "Sharpness", "CV", "KS_z", "Normal_p"]
    col_names = ["RAW", "VAR", "ISO"]

    force_table = [
        [m_raw_F["NLL_raw"],      m_var_F["NLL_calVAR"],      m_iso_F["NLL_calISO"]],
        [tNLL_raw,                tNLL_calVAR,                tNLL_calISO],
        [m_raw_F["RLL_raw"],      m_var_F["RLL_calVAR"],      m_iso_F["RLL_calISO"]],
        [m_raw_F["CRPS_raw"],     m_var_F["CRPS_calVAR"],     m_iso_F["CRPS_calISO"]],
        [m_raw_F["ENCE_raw"],     m_var_F["ENCE_calVAR"],     m_iso_F["ENCE_calISO"]],
        [m_raw_F["PICP95_raw"],   m_var_F["PICP95_calVAR"],   m_iso_F["PICP95_calISO"]],
        [m_raw_F["Sharpness_raw"],m_var_F["Sharpness_calVAR"],m_iso_F["Sharpness_calISO"]],
        [m_raw_F["CV_raw"],       m_var_F["CV_calVAR"],       m_iso_F["CV_calISO"]],
        [m_raw_F["KS_z_raw"],     m_var_F["KS_z_calVAR"],     m_iso_F["KS_z_calISO"]],
        [m_raw_F["Normal_p_raw"], m_var_F["Normal_p_calVAR"], m_iso_F["Normal_p_calISO"]],
    ]
    if delta_e is not None:
        energy_table = [
            [m_raw_E["NLL_raw_E"],      m_var_E["NLL_calVAR_E"],      m_iso_E["NLL_calISO_E"]],
            [tNLL_raw_E,                tNLL_calVAR_E,                tNLL_calISO_E],
            [m_raw_E["RLL_raw_E"],      m_var_E["RLL_calVAR_E"],      m_iso_E["RLL_calISO_E"]],
            [m_raw_E["CRPS_raw_E"],     m_var_E["CRPS_calVAR_E"],     m_iso_E["CRPS_calISO_E"]],
            [m_raw_E["ENCE_raw_E"],     m_var_E["ENCE_calVAR_E"],     m_iso_E["ENCE_calISO_E"]],
            [m_raw_E["PICP95_raw_E"],   m_var_E["PICP95_calVAR_E"],   m_iso_E["PICP95_calISO_E"]],
            [m_raw_E["Sharpness_raw_E"],m_var_E["Sharpness_calVAR_E"],m_iso_E["Sharpness_calISO_E"]],
            [m_raw_E["CV_raw_E"],       m_var_E["CV_calVAR_E"],       m_iso_E["CV_calISO_E"]],
            [m_raw_E["KS_z_raw_E"],     m_var_E["KS_z_calVAR_E"],     m_iso_E["KS_z_calISO_E"]],
            [m_raw_E["Normal_p_raw_E"], m_var_E["Normal_p_calVAR_E"], m_iso_E["Normal_p_calISO_E"]],
        ]

    # Qualitative verdicts for table
    context = {"NLL_base": mean_nll_base_c, "NLL_base_E": mean_nll_base_e}
    force_verdicts = [
        [qualitative_label(force_table[i][j], f"{row_names[i]}_{['raw','calVAR','calISO'][j]}", context)
         for j in range(3)]
        for i in range(len(row_names))
    ]
    if delta_e is not None:
        energy_verdicts = [
            [qualitative_label(energy_table[i][j], f"{row_names[i]}_{['raw_E','calVAR_E','calISO_E'][j]}", context)
             for j in range(3)]
            for i in range(len(row_names))
        ]

    # ------------ Best ENCE verdict -----------------------------------
    best_F_idx = int(np.argmin([force_table[4][i] for i in range(3)]))  # ENCE row
    best_E_idx = int(np.argmin([energy_table[4][i] for i in range(3)])) if delta_e is not None else 0
    calib_labels = ["RAW", "VAR", "ISO"]

    # ------------- Compose Output ------------------------------------
    banner = (
        f"\n=== {split.upper()} | {tag} ===\n"
        f"Best calibration for FORCES:   {calib_labels[best_F_idx]} (ENCE={force_table[3][best_F_idx]:.3f})\n"
    )
    if delta_e is not None:
        banner += f"Best calibration for ENERGIES: {calib_labels[best_E_idx]} (ENCE={energy_table[3][best_E_idx]:.3f})\n"
    banner += "─" * 50

    print(banner)
    print("Forces:")
    print(pretty_table_full(row_names, col_names, force_table, force_verdicts))
    if delta_e is not None:
        print("\nEnergies:")
        print(pretty_table_full(row_names, col_names, energy_table, energy_verdicts))
    print("─" * 50)

    _LOGGER.info(banner)
    _LOGGER.info("\nForces:\n" + pretty_table_full(row_names, col_names, force_table, force_verdicts))
    if delta_e is not None:
        _LOGGER.info("\nEnergies:\n" + pretty_table_full(row_names, col_names, energy_table, energy_verdicts))

    # ------------ SUGGESTION SUMMARY -------------------------------
    # Verdict for 'raw' (uncalibrated)
    forces_raw_verdict = qualitative_label(m_raw_F['ENCE_raw'], "ENCE_raw")
    if delta_e is not None:
        energy_raw_verdict = qualitative_label(m_raw_E['ENCE_raw_E'], "ENCE_raw_E")
    else:
        energy_raw_verdict = "n/a"

    # ------------ Best ENCE verdict -----------------------------------
    best_F_idx = int(np.argmin([force_table[3][i] for i in range(3)]))  # ENCE row
    best_F_label = calib_labels[best_F_idx]
    best_F_val = force_table[3][best_F_idx]
    if delta_e is not None:
        best_E_idx = int(np.argmin([energy_table[3][i] for i in range(3)]))
        best_E_label = calib_labels[best_E_idx]
        best_E_val = energy_table[3][best_E_idx]
    else:
        best_E_label = None
        best_E_val = None

    # ------------ SUGGESTION SUMMARY -------------------------------
    forces_raw_verdict = qualitative_label(m_raw_F['ENCE_raw'], "ENCE_raw")
    if delta_e is not None:
        energy_raw_verdict = qualitative_label(m_raw_E['ENCE_raw_E'], "ENCE_raw_E")
    else:
        energy_raw_verdict = "n/a"

    lines = []
    lines.append("=== UQ CALIBRATION SUGGESTION ===")

    if forces_raw_verdict == "good":
        lines.append("• Calibration is NOT needed for forces: uncalibrated uncertainties are already well calibrated (ENCE_raw = {:.3f})".format(m_raw_F['ENCE_raw']))
    else:
        lines.append("• Best calibration for forces: {} (ENCE = {:.3f})".format(best_F_label, best_F_val))

    if delta_e is not None:
        if energy_raw_verdict == "good":
            lines.append("• Calibration is NOT needed for energies: uncalibrated uncertainties are already well calibrated (ENCE_raw_E = {:.3f})".format(m_raw_E['ENCE_raw_E']))
        else:
            lines.append("• Best calibration for energies: {} (ENCE = {:.3f})".format(best_E_label, best_E_val))

    suggestion_block = "\n".join(lines)
    print(suggestion_block)
    _LOGGER.info("\n" + suggestion_block)

    # --------- MetricResult list, including all variants --------------
    metrics: list[MetricResult] = [
        MetricResult("NLL_base", mean_nll_base_c, "probabilistic"),
        *[MetricResult(k, v, "probabilistic") for k, v in m_raw_F.items()],
        *[MetricResult(k, v, "probabilistic") for k, v in m_var_F.items()],
        *[MetricResult(k, v, "probabilistic") for k, v in m_iso_F.items()],
    ]
    if delta_e is not None:
        metrics += [
            MetricResult("NLL_base_E", mean_nll_base_e, "energy"),
            *[MetricResult(k, v, "energy") for k, v in m_raw_E.items()],
            *[MetricResult(k, v, "energy") for k, v in m_var_E.items()],
            *[MetricResult(k, v, "energy") for k, v in m_iso_E.items()],
        ]
    # Spearman for forces (atom-level)
    spearman_raw_F  = spearman_corr(delta_a, sigma_a)
    spearman_var_F  = spearman_corr(delta_a, sigma_a_cal_var)
    spearman_iso_F  = spearman_corr(delta_a, sigma_a_cal_iso)
    metrics += [
        MetricResult("Spearman_raw",     spearman_raw_F, "discrimination"),
        MetricResult("Spearman_calVAR",  spearman_var_F, "discrimination"),
        MetricResult("Spearman_calISO",  spearman_iso_F, "discrimination"),
    ]
    if delta_e is not None:
        spearman_raw_E = spearman_corr(delta_e, sigma_e)
        spearman_var_E = spearman_corr(delta_e, sigma_e_cal_var)
        spearman_iso_E = spearman_corr(delta_e, sigma_e_cal_iso)
        metrics += [
            MetricResult("Spearman_raw_E",     spearman_raw_E, "discrimination"),
            MetricResult("Spearman_calVAR_E",  spearman_var_E, "discrimination"),
            MetricResult("Spearman_calISO_E",  spearman_iso_E, "discrimination"),
        ]

    for k, v in student_t_nlls.items():
        if v is not None:
            metrics.append(MetricResult(f"Student_t_NLL_{k}", v, "probabilistic"))
    
    # --- Save npz for downstream plotting ------------------------------
    npz_path = None
    try:
        scalar_metrics = {m.name: m.value for m in metrics}
        p_thresholds = np.linspace(0.0, 1.0, 21)

        rmse_force = float(np.sqrt(np.mean(delta_c**2)))
        rmv_force  = float(np.sqrt(np.mean(sigma_c**2)))
        scalar_metrics.update({
            "rmse_force": rmse_force,
            "rmv_force":  rmv_force,
        })
        if delta_e is not None:
            rmse_energy = float(np.sqrt(np.mean(delta_e**2)))
            rmv_energy  = float(np.sqrt(np.mean(sigma_e**2)))
            scalar_metrics.update({
                "rmse_energy": rmse_energy,
                "rmv_energy":  rmv_energy,
            })

        # Forces
        coverage_uncal     = np.array([picp(delta_c, sigma_c, alpha=1-p) for p in p_thresholds])
        coverage_cal_var   = np.array([picp(delta_c, sigma_c_cal_var, alpha=1-p) for p in p_thresholds])
        coverage_cal_iso   = np.array([picp(delta_c, sigma_c_cal_iso, alpha=1-p) for p in p_thresholds])

        # Energies (if present)
        if delta_e is not None:
            coverage_uncal_e   = np.array([picp(delta_e, sigma_e, alpha=1-p) for p in p_thresholds])
            coverage_cal_var_e = np.array([picp(delta_e, sigma_e_cal_var, alpha=1-p) for p in p_thresholds])
            coverage_cal_iso_e = np.array([picp(delta_e, sigma_e_cal_iso, alpha=1-p) for p in p_thresholds])
        else:
            coverage_uncal_e   = np.array([])
            coverage_cal_var_e = np.array([])
            coverage_cal_iso_e = np.array([])

        base = f"uq_plot_data_{split.lower()}_{tag.lower()}"
        if ensemble_size:
            base += f"_ens{ensemble_size}"
        npz_path = Path("uq_plots") / f"{base}.npz"
        npz_path.parent.mkdir(exist_ok=True)

        np.savez_compressed(
            npz_path,
            # Forces
            delta_comp=delta_c,
            sigma_comp_uncal=sigma_c,
            sigma_comp_cal_var=sigma_c_cal_var,
            sigma_comp_cal_iso=sigma_c_cal_iso,
            # Energies
            delta_energy=delta_e,
            sigma_energy_uncal=sigma_e,
            sigma_energy_cal_var=sigma_e_cal_var,
            sigma_energy_cal_iso=sigma_e_cal_iso,
            # Scalar metrics & coverage
            scalar_metrics=scalar_metrics,
            p_thresholds=p_thresholds,
            coverage_uncal=coverage_uncal,
            coverage_cal_var=coverage_cal_var,
            coverage_cal_iso=coverage_cal_iso,
            coverage_uncal_e=coverage_uncal_e,
            coverage_cal_var_e=coverage_cal_var_e,
            coverage_cal_iso_e=coverage_cal_iso_e,
        )
        _LOGGER.info("  saved ➜ %s", npz_path)
    except Exception as exc:
        _LOGGER.warning("could not save plot data: %s", exc)

    # --- Return dict for further processing -----------------------------
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

