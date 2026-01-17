"""LC-only false-alarm checks (V13, V15).

These checks target common TESS false-alarm morphologies that often survive
basic astrophysical-false-positive screening:
- missing cadence / gap-edge events (V13)
- ramp/step-like asymmetry around transit (V15)

All checks return legacy `VetterCheckResult` objects in metrics-only mode
(`passed=None`). Host applications apply policy/guardrails externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bittr_tess_vetter.domain.detection import VetterCheckResult
from bittr_tess_vetter.validation.base import (
    get_in_transit_mask,
    get_out_of_transit_mask,
    phase_fold,
)

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.lightcurve import LightCurveData


_EPS = 1e-12


@dataclass
class DataGapConfig:
    """Configuration for V13 data-gap near-transit check."""

    window_mult: float = 2.0
    max_epoch_rows: int = 10
    min_expected_points: int = 5
    max_expected_points: int = 20000


def _median_dt_days(time_sorted: np.ndarray) -> float | None:
    if time_sorted.size < 3:
        return None
    dt = np.diff(time_sorted)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    return float(np.median(dt))


def check_data_gaps(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    duration_hours: float,
    config: DataGapConfig | None = None,
) -> VetterCheckResult:
    """V13: Data gap / missing cadence near transit windows.

    Computes a per-epoch missing-cadence fraction in a fixed window around each
    transit center. High missingness is associated with gap-edge artifacts and
    scattered-light ramps that can trigger transit searches.
    """
    if config is None:
        config = DataGapConfig()

    time = lightcurve.time[lightcurve.valid_mask]
    time = time[np.isfinite(time)]
    warnings: list[str] = []

    duration_days = float(duration_hours) / 24.0
    if time.size < 10 or period <= 0 or duration_days <= 0:
        return VetterCheckResult(
            id="V13",
            name="data_gaps",
            passed=None,
            confidence=0.2,
            details={
                "missing_frac_max": None,
                "missing_frac_median": None,
                "n_epochs_evaluated": 0,
                "warnings": ["insufficient_time_points"],
                "_metrics_only": True,
            },
        )

    time_sorted = np.sort(time)
    global_dt = _median_dt_days(time_sorted)
    if global_dt is None or not np.isfinite(global_dt) or global_dt <= 0:
        global_dt = float(np.median(np.diff(time_sorted[: min(time_sorted.size, 50)])))
    if not np.isfinite(global_dt) or global_dt <= 0:
        global_dt = 0.0014  # fallback ~2 min cadence
        warnings.append("cadence_fallback_default")

    # Epoch indexing consistent with V04: epoch 0 contains transit at t0.
    epoch = np.floor((time - float(t0) + float(period) / 2) / float(period)).astype(int)
    unique_epochs = np.unique(epoch)

    rows: list[dict[str, float | int]] = []
    for ep in unique_epochs:
        t_center = float(t0) + float(ep) * float(period)
        half_window_days = float(config.window_mult) * duration_days
        window_mask = (time >= t_center - half_window_days) & (time <= t_center + half_window_days)
        n_observed = int(np.sum(window_mask))

        local_dt = global_dt
        if n_observed >= 3:
            dt_local = _median_dt_days(np.sort(time[window_mask]))
            if dt_local is not None and np.isfinite(dt_local) and dt_local > 0:
                local_dt = dt_local

        expected = int(np.floor((2.0 * half_window_days) / max(_EPS, local_dt)) + 1)
        expected = max(int(config.min_expected_points), min(int(config.max_expected_points), expected))

        missing_frac = 1.0 - (float(n_observed) / float(expected))
        missing_frac = float(np.clip(missing_frac, 0.0, 1.0))

        rows.append(
            {
                "epoch_index": int(ep),
                "t_center_btjd": float(t_center),
                "missing_frac": float(missing_frac),
                "n_observed": int(n_observed),
                "n_expected": int(expected),
            }
        )

    if not rows:
        return VetterCheckResult(
            id="V13",
            name="data_gaps",
            passed=None,
            confidence=0.3,
            details={
                "missing_frac_max": None,
                "missing_frac_median": None,
                "n_epochs_evaluated": 0,
                "warnings": ["no_epochs_evaluated"],
                "_metrics_only": True,
            },
        )

    missing = np.array([r["missing_frac"] for r in rows], dtype=np.float64)
    missing_frac_max = float(np.max(missing))
    missing_frac_median = float(np.median(missing))
    n_epochs_evaluated = int(len(rows))
    n_epochs_missing_ge_0p25 = int(np.sum(missing >= 0.25))

    # Coverage-aware summaries: exclude epochs whose window contains no observed cadences.
    # This avoids confusing cases where the ephemeris predicts transits during large
    # inter-sector gaps (missing_frac=1.0 but no data exists to evaluate that epoch).
    rows_in_coverage = [r for r in rows if int(r.get("n_observed", 0)) > 0]
    if rows_in_coverage:
        missing_in_cov = np.array([r["missing_frac"] for r in rows_in_coverage], dtype=np.float64)
        missing_frac_max_in_coverage = float(np.max(missing_in_cov))
        missing_frac_median_in_coverage = float(np.median(missing_in_cov))
        n_epochs_evaluated_in_coverage = int(len(rows_in_coverage))
        n_epochs_missing_ge_0p25_in_coverage = int(np.sum(missing_in_cov >= 0.25))
    else:
        missing_frac_max_in_coverage = None
        missing_frac_median_in_coverage = None
        n_epochs_evaluated_in_coverage = 0
        n_epochs_missing_ge_0p25_in_coverage = 0

    n_epochs_excluded_no_coverage = int(n_epochs_evaluated - n_epochs_evaluated_in_coverage)

    # Confidence: mostly about how many epochs we could evaluate.
    if n_epochs_evaluated < 2:
        confidence = 0.35
    elif n_epochs_evaluated < 5:
        confidence = 0.6
    else:
        confidence = 0.75
    if warnings:
        confidence *= 0.95
    confidence = float(np.clip(confidence, 0.2, 0.95))

    worst = sorted(rows, key=lambda r: float(r["missing_frac"]), reverse=True)[: int(config.max_epoch_rows)]

    return VetterCheckResult(
        id="V13",
        name="data_gaps",
        passed=None,
        confidence=round(confidence, 3),
        details={
            "missing_frac_max": round(missing_frac_max, 3),
            "missing_frac_median": round(missing_frac_median, 3),
            "n_epochs_evaluated": n_epochs_evaluated,
            "n_epochs_missing_ge_0p25": n_epochs_missing_ge_0p25,
            "missing_frac_max_in_coverage": (
                round(float(missing_frac_max_in_coverage), 3)
                if missing_frac_max_in_coverage is not None
                else None
            ),
            "missing_frac_median_in_coverage": (
                round(float(missing_frac_median_in_coverage), 3)
                if missing_frac_median_in_coverage is not None
                else None
            ),
            "n_epochs_evaluated_in_coverage": int(n_epochs_evaluated_in_coverage),
            "n_epochs_missing_ge_0p25_in_coverage": int(n_epochs_missing_ge_0p25_in_coverage),
            "n_epochs_excluded_no_coverage": int(n_epochs_excluded_no_coverage),
            "window_mult": float(config.window_mult),
            "worst_epochs": worst,
            "warnings": warnings,
            "_metrics_only": True,
        },
    )


@dataclass
class AsymmetryConfig:
    """Configuration for V15 transit asymmetry check."""

    window_mult: float = 2.0
    n_bins_half: int = 12
    min_points_per_half: int = 20


def _binned_means(
    phase: np.ndarray,
    values: np.ndarray,
    *,
    lo: float,
    hi: float,
    n_bins: int,
) -> np.ndarray:
    if n_bins < 1 or hi <= lo:
        return np.array([], dtype=np.float64)
    edges = np.linspace(lo, hi, n_bins + 1)
    means: list[float] = []
    for i in range(n_bins):
        if i == 0:
            m = (phase >= edges[i]) & (phase <= edges[i + 1])
        else:
            m = (phase > edges[i]) & (phase <= edges[i + 1])
        if np.any(m):
            means.append(float(np.nanmean(values[m])))
        else:
            means.append(float("nan"))
    return np.array(means, dtype=np.float64)


def check_transit_asymmetry(
    lightcurve: LightCurveData,
    period: float,
    t0: float,
    duration_hours: float,
    config: AsymmetryConfig | None = None,
) -> VetterCheckResult:
    """V15: Transit-window asymmetry (ramp/step proxy)."""
    if config is None:
        config = AsymmetryConfig()

    mask = lightcurve.valid_mask & np.isfinite(lightcurve.time) & np.isfinite(lightcurve.flux)
    time = lightcurve.time[mask]
    flux = lightcurve.flux[mask]

    warnings: list[str] = []
    duration_days = float(duration_hours) / 24.0
    if time.size < 50 or period <= 0 or duration_days <= 0:
        warnings.append("insufficient_time_points")
        return VetterCheckResult(
            id="V15",
            name="transit_asymmetry",
            passed=None,
            confidence=0.2,
            details={"warnings": warnings, "_metrics_only": True},
        )

    phase, _ = phase_fold(time, flux, float(period), float(t0))
    half_window_phase = float(config.window_mult) * duration_days / float(period)
    half_window_phase = float(np.clip(half_window_phase, 0.0, 0.5))
    if half_window_phase <= 0:
        warnings.append("invalid_window")
        return VetterCheckResult(
            id="V15",
            name="transit_asymmetry",
            passed=None,
            confidence=0.2,
            details={"warnings": warnings, "_metrics_only": True},
        )

    window_mask = np.abs(phase) <= half_window_phase
    if int(np.sum(window_mask)) < 50:
        warnings.append("insufficient_points_in_window")

    in_transit = get_in_transit_mask(time, float(period), float(t0), float(duration_hours), buffer_factor=1.0)
    oot_local = window_mask & (~in_transit)
    if int(np.sum(oot_local)) < 10:
        # Global fallback (exclude a wider margin to avoid transit leakage).
        oot_global = get_out_of_transit_mask(time, float(period), float(t0), float(duration_hours), buffer_factor=3.0)
        if int(np.sum(oot_global)) >= 10:
            oot_local = oot_global
            warnings.append("baseline_fallback_global_oot")
        else:
            warnings.append("baseline_unavailable")

    baseline = float(np.nanmedian(flux[oot_local])) if int(np.sum(oot_local)) > 0 else 1.0
    flux_norm = flux - baseline

    left_mask = window_mask & (phase < 0)
    right_mask = window_mask & (phase > 0)
    n_left = int(np.sum(left_mask))
    n_right = int(np.sum(right_mask))
    if n_left < config.min_points_per_half or n_right < config.min_points_per_half:
        warnings.append("insufficient_points_per_half")

    left_bins = _binned_means(
        phase[left_mask],
        flux_norm[left_mask],
        lo=-half_window_phase,
        hi=0.0,
        n_bins=int(config.n_bins_half),
    )
    right_bins = _binned_means(
        phase[right_mask],
        flux_norm[right_mask],
        lo=0.0,
        hi=half_window_phase,
        n_bins=int(config.n_bins_half),
    )

    left_valid = left_bins[np.isfinite(left_bins)]
    right_valid = right_bins[np.isfinite(right_bins)]
    n_left_bins = int(left_valid.size)
    n_right_bins = int(right_valid.size)

    if n_left_bins < 3 or n_right_bins < 3:
        warnings.append("insufficient_bins")
        return VetterCheckResult(
            id="V15",
            name="transit_asymmetry",
            passed=None,
            confidence=0.3,
            details={
                "asymmetry_sigma": None,
                "window_mult": float(config.window_mult),
                "n_bins_half": int(config.n_bins_half),
                "n_left_bins": n_left_bins,
                "n_right_bins": n_right_bins,
                "warnings": warnings,
                "_metrics_only": True,
            },
        )

    mu_left = float(np.mean(left_valid))
    mu_right = float(np.mean(right_valid))
    se_left = float(np.std(left_valid) / np.sqrt(max(1, n_left_bins)))
    se_right = float(np.std(right_valid) / np.sqrt(max(1, n_right_bins)))
    denom = np.sqrt(se_left**2 + se_right**2) + _EPS
    asym_sigma = float(np.abs(mu_left - mu_right) / denom)

    # Confidence: degrade if baseline fallback or weak binning.
    confidence = 0.75
    if "insufficient_bins" in warnings or "insufficient_points_in_window" in warnings:
        confidence *= 0.85
    if any(w.startswith("baseline_") for w in warnings):
        confidence *= 0.9
    confidence = float(np.clip(confidence, 0.2, 0.95))

    return VetterCheckResult(
        id="V15",
        name="transit_asymmetry",
        passed=None,
        confidence=round(confidence, 3),
        details={
            "asymmetry_sigma": round(asym_sigma, 3),
            "mu_left": round(mu_left, 6),
            "mu_right": round(mu_right, 6),
            "window_mult": float(config.window_mult),
            "n_bins_half": int(config.n_bins_half),
            "n_left_bins": n_left_bins,
            "n_right_bins": n_right_bins,
            "n_left_points": n_left,
            "n_right_points": n_right,
            "baseline": round(baseline, 6),
            "warnings": warnings,
            "_metrics_only": True,
        },
    )


__all__ = [
    "DataGapConfig",
    "AsymmetryConfig",
    "check_data_gaps",
    "check_transit_asymmetry",
]
