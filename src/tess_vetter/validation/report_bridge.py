"""Deterministic bridge helpers shared by report and API composition.

This module centralizes report-facing computations while depending only on
low-level validation/transit/domain primitives (no API wrapper calls).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from tess_vetter.domain.detection import VetterCheckResult
from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.domain.target import StellarParameters
from tess_vetter.transit.result import TransitTimingSeries
from tess_vetter.transit.timing import build_timing_series, measure_all_transit_times
from tess_vetter.validation.alias_diagnostics import (
    HarmonicPowerSummary,
    HarmonicScore,
    classify_alias,
    compute_secondary_significance,
    detect_phase_shift_events,
    summarize_harmonic_power,
)
from tess_vetter.validation.lc_checks import (
    DepthStabilityConfig,
    OddEvenConfig,
    SecondaryEclipseConfig,
    VShapeConfig,
    check_depth_stability,
    check_duration_consistency,
    check_odd_even_depth,
    check_secondary_eclipse,
    check_v_shape,
)
from tess_vetter.validation.lc_false_alarm_checks import (
    AsymmetryConfig,
    DataGapConfig,
    check_data_gaps,
    check_transit_asymmetry,
)
from tess_vetter.validation.result_schema import CheckResult, ok_result

_LC_CHECK_ORDER: tuple[str, ...] = ("V01", "V02", "V03", "V04", "V05", "V13", "V15")
_REPORT_DEFAULT_ENABLED: set[str] = {"V01", "V02", "V04", "V05", "V13", "V15"}


@dataclass(frozen=True, slots=True)
class AliasDiagnosticsConfig:
    """Single source of deterministic alias diagnostic thresholds."""

    n_phase_bins: int = 10
    phase_shift_significance_threshold: float = 3.0
    alias_ratio_threshold_strong: float = 1.5
    alias_ratio_threshold_weak: float = 1.1


@dataclass(frozen=True, slots=True)
class AliasDiagnosticsResult:
    """Canonical alias diagnostics payload (harmonic + scalar + metadata)."""

    base_period: float
    base_t0: float
    duration_hours: float
    harmonic_labels: list[str]
    periods: list[float]
    scores: list[float]
    harmonic_depth_ppm: list[float]
    harmonic_duration_hours: list[float | None]
    best_harmonic: str
    best_ratio_over_p: float
    classification: str
    phase_shift_event_count: int
    phase_shift_peak_sigma: float | None
    secondary_significance: float
    n_phase_bins: int
    phase_shift_significance_threshold: float
    alias_ratio_threshold_strong: float
    alias_ratio_threshold_weak: float


def _masked_lc_arrays(
    lightcurve: LightCurveData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(lightcurve.valid_mask, dtype=np.bool_)
    return (
        np.asarray(lightcurve.time, dtype=np.float64)[mask],
        np.asarray(lightcurve.flux, dtype=np.float64)[mask],
        np.asarray(lightcurve.flux_err, dtype=np.float64)[mask],
    )


def _summary_from_alias_diagnostics(diagnostics: AliasDiagnosticsResult) -> HarmonicPowerSummary:
    harmonics = [
        HarmonicScore(
            harmonic=label,
            period=float(period),
            score=float(score),
            depth_ppm=float(depth_ppm),
            duration_hours=duration_hours,
        )
        for label, period, score, depth_ppm, duration_hours in zip(
            diagnostics.harmonic_labels,
            diagnostics.periods,
            diagnostics.scores,
            diagnostics.harmonic_depth_ppm,
            diagnostics.harmonic_duration_hours,
            strict=False,
        )
    ]
    return HarmonicPowerSummary(
        base_period=float(diagnostics.base_period),
        base_t0=float(diagnostics.base_t0),
        duration_hours=float(diagnostics.duration_hours),
        harmonics=harmonics,
        best_harmonic=str(diagnostics.best_harmonic),
        best_ratio_over_p=float(diagnostics.best_ratio_over_p),
    )


def _scalar_signals_from_alias_diagnostics(
    diagnostics: AliasDiagnosticsResult,
) -> dict[str, float | int | str | None]:
    return {
        "classification": str(diagnostics.classification),
        "phase_shift_event_count": int(diagnostics.phase_shift_event_count),
        "phase_shift_peak_sigma": diagnostics.phase_shift_peak_sigma,
        "secondary_significance": float(diagnostics.secondary_significance),
    }


def _convert_legacy_result(result: VetterCheckResult) -> CheckResult:
    """Convert legacy ``VetterCheckResult`` into canonical ``CheckResult``."""
    details = dict(result.details)

    metrics: dict[str, float | int | str | bool | None] = {}
    raw: dict[str, Any] = {}
    for key, value in details.items():
        if isinstance(value, (float, int, str, bool, type(None))):
            metrics[key] = value
        else:
            raw[key] = value

    return ok_result(
        id=result.id,
        name=result.name,
        metrics=metrics,
        confidence=result.confidence,
        raw=raw or None,
    )


def run_lc_checks(
    lightcurve: LightCurveData,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    stellar: StellarParameters | None = None,
    enabled: set[str] | None = None,
    config: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[CheckResult]:
    """Run LC checks in report order and return canonical results.

    Defaults match current report behavior (V03 excluded unless explicitly
    enabled by the caller).
    """
    checks_to_run = [
        check_id
        for check_id in _LC_CHECK_ORDER
        if check_id in (enabled if enabled is not None else _REPORT_DEFAULT_ENABLED)
    ]
    cfg = config or {}

    results: list[CheckResult] = []
    for check_id in checks_to_run:
        if check_id == "V01":
            legacy = check_odd_even_depth(
                lightcurve=lightcurve,
                period=period_days,
                t0=t0_btjd,
                duration_hours=duration_hours,
                config=OddEvenConfig(**dict(cfg["V01"])) if "V01" in cfg else None,
            )
        elif check_id == "V02":
            legacy = check_secondary_eclipse(
                lightcurve=lightcurve,
                period=period_days,
                t0=t0_btjd,
                config=SecondaryEclipseConfig(**dict(cfg["V02"])) if "V02" in cfg else None,
            )
        elif check_id == "V03":
            legacy = check_duration_consistency(
                period=period_days,
                duration_hours=duration_hours,
                stellar=stellar,
            )
        elif check_id == "V04":
            legacy = check_depth_stability(
                lightcurve=lightcurve,
                period=period_days,
                t0=t0_btjd,
                duration_hours=duration_hours,
                config=DepthStabilityConfig(**dict(cfg["V04"])) if "V04" in cfg else None,
            )
        elif check_id == "V05":
            legacy = check_v_shape(
                lightcurve=lightcurve,
                period=period_days,
                t0=t0_btjd,
                duration_hours=duration_hours,
                config=VShapeConfig(**dict(cfg["V05"])) if "V05" in cfg else None,
            )
        elif check_id == "V13":
            legacy = check_data_gaps(
                lightcurve=lightcurve,
                period=period_days,
                t0=t0_btjd,
                duration_hours=duration_hours,
                config=DataGapConfig(**dict(cfg["V13"])) if "V13" in cfg else None,
            )
        elif check_id == "V15":
            legacy = check_transit_asymmetry(
                lightcurve=lightcurve,
                period=period_days,
                t0=t0_btjd,
                duration_hours=duration_hours,
                config=AsymmetryConfig(**dict(cfg["V15"])) if "V15" in cfg else None,
            )
        else:
            continue
        results.append(_convert_legacy_result(legacy))

    return results


def compute_timing_series(
    lightcurve: LightCurveData,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    min_snr: float = 2.0,
) -> TransitTimingSeries:
    """Measure and summarize transit timing diagnostics from internal LC data."""
    mask = np.asarray(lightcurve.valid_mask, dtype=np.bool_)
    transit_times = measure_all_transit_times(
        time=np.asarray(lightcurve.time, dtype=np.float64)[mask],
        flux=np.asarray(lightcurve.flux, dtype=np.float64)[mask],
        flux_err=np.asarray(lightcurve.flux_err, dtype=np.float64)[mask],
        period=period_days,
        t0=t0_btjd,
        duration_hours=duration_hours,
        min_snr=min_snr,
    )
    return build_timing_series(
        transit_times=transit_times,
        period=period_days,
        t0=t0_btjd,
    )


def compute_alias_summary(
    lightcurve: LightCurveData,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    diagnostics: AliasDiagnosticsResult | None = None,
    config: AliasDiagnosticsConfig | None = None,
) -> HarmonicPowerSummary:
    """Compute compact harmonic alias summary at P, P/2, and 2P."""
    alias_diagnostics = diagnostics or compute_alias_diagnostics(
        lightcurve,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        config=config or AliasDiagnosticsConfig(),
    )
    return _summary_from_alias_diagnostics(alias_diagnostics)


def compute_alias_diagnostics(
    lightcurve: LightCurveData,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    config: AliasDiagnosticsConfig = AliasDiagnosticsConfig(),  # noqa: B008
) -> AliasDiagnosticsResult:
    """Compute canonical harmonic + scalar alias diagnostics."""
    time, flux, flux_err = _masked_lc_arrays(lightcurve)

    summary = summarize_harmonic_power(
        time=time,
        flux=flux,
        flux_err=flux_err,
        base_period=period_days,
        base_t0=t0_btjd,
        duration_hours=duration_hours,
    )
    harmonics = list(summary.harmonics)
    base_score = next((h.score for h in harmonics if h.harmonic == "P"), 0.0)
    classification, best_harmonic, best_ratio_over_p = classify_alias(
        harmonics,
        base_score=base_score,
        strong_ratio_threshold=config.alias_ratio_threshold_strong,
        weak_ratio_threshold=config.alias_ratio_threshold_weak,
    )
    events = detect_phase_shift_events(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=period_days,
        t0=t0_btjd,
        n_phase_bins=config.n_phase_bins,
        significance_threshold=config.phase_shift_significance_threshold,
    )
    peak_sigma = max((float(e.significance) for e in events), default=None)
    secondary_significance = compute_secondary_significance(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=period_days,
        t0=t0_btjd,
        duration_hours=duration_hours,
    )

    return AliasDiagnosticsResult(
        base_period=float(summary.base_period),
        base_t0=float(summary.base_t0),
        duration_hours=float(summary.duration_hours),
        harmonic_labels=[str(h.harmonic) for h in harmonics],
        periods=[float(h.period) for h in harmonics],
        scores=[float(h.score) for h in harmonics],
        harmonic_depth_ppm=[float(h.depth_ppm) for h in harmonics],
        harmonic_duration_hours=[h.duration_hours for h in harmonics],
        best_harmonic=str(best_harmonic),
        best_ratio_over_p=float(best_ratio_over_p),
        classification=str(classification),
        phase_shift_event_count=int(len(events)),
        phase_shift_peak_sigma=peak_sigma,
        secondary_significance=float(secondary_significance),
        n_phase_bins=int(config.n_phase_bins),
        phase_shift_significance_threshold=float(config.phase_shift_significance_threshold),
        alias_ratio_threshold_strong=float(config.alias_ratio_threshold_strong),
        alias_ratio_threshold_weak=float(config.alias_ratio_threshold_weak),
    )


def compute_alias_scalar_signals(
    lightcurve: LightCurveData,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    harmonic_summary: HarmonicPowerSummary | None = None,
    n_phase_bins: int = 10,
    significance_threshold: float = 3.0,
    diagnostics: AliasDiagnosticsResult | None = None,
    config: AliasDiagnosticsConfig | None = None,
) -> dict[str, float | int | str | None]:
    """Compute scalar alias diagnostics from deterministic base assumptions."""
    effective_config = config or AliasDiagnosticsConfig(
        n_phase_bins=n_phase_bins,
        phase_shift_significance_threshold=significance_threshold,
    )
    if diagnostics is not None:
        return _scalar_signals_from_alias_diagnostics(diagnostics)

    # Preserve backward compatibility: when a harmonic summary is supplied,
    # derive alias classification from that summary rather than discarding it.
    if harmonic_summary is not None:
        time, flux, flux_err = _masked_lc_arrays(lightcurve)
        harmonics = list(harmonic_summary.harmonics)
        base_score = next((h.score for h in harmonics if h.harmonic == "P"), 0.0)
        classification, _best_harmonic, _ratio = classify_alias(
            harmonics,
            base_score=base_score,
            strong_ratio_threshold=effective_config.alias_ratio_threshold_strong,
            weak_ratio_threshold=effective_config.alias_ratio_threshold_weak,
        )
        events = detect_phase_shift_events(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=period_days,
            t0=t0_btjd,
            n_phase_bins=effective_config.n_phase_bins,
            significance_threshold=effective_config.phase_shift_significance_threshold,
        )
        peak_sigma = max((float(e.significance) for e in events), default=None)
        secondary_significance = compute_secondary_significance(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=period_days,
            t0=t0_btjd,
            duration_hours=duration_hours,
        )
        return {
            "classification": str(classification),
            "phase_shift_event_count": int(len(events)),
            "phase_shift_peak_sigma": peak_sigma,
            "secondary_significance": float(secondary_significance),
        }

    alias_diagnostics = compute_alias_diagnostics(
        lightcurve,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        config=effective_config,
    )
    return _scalar_signals_from_alias_diagnostics(alias_diagnostics)


__all__ = [
    "AliasDiagnosticsConfig",
    "AliasDiagnosticsResult",
    "compute_alias_diagnostics",
    "compute_alias_scalar_signals",
    "compute_alias_summary",
    "compute_timing_series",
    "run_lc_checks",
]
