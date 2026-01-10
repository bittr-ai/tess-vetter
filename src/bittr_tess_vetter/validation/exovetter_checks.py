"""Exovetter-based computations (metrics-only).

Wraps selected `exovetter` vetters and returns metrics as `VetterCheckResult`
objects with `passed=None` and `details["_metrics_only"]=True`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from bittr_tess_vetter.domain.detection import TransitCandidate, VetterCheckResult
from bittr_tess_vetter.domain.lightcurve import LightCurveData

logger = logging.getLogger(__name__)


def _metrics_result(
    *,
    check_id: str,
    name: str,
    confidence: float,
    details: dict[str, Any],
) -> VetterCheckResult:
    details = dict(details)
    details["_metrics_only"] = True
    return VetterCheckResult(
        id=check_id,
        name=name,
        passed=None,
        confidence=float(max(0.0, min(1.0, confidence))),
        details=details,
    )


def _coerce_scalar(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _coerce_metrics_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _coerce_metrics_dict(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [_coerce_scalar(x) for x in v]
        else:
            out[k] = _coerce_scalar(v)
    return out


def _is_likely_folded(time: np.ndarray, period_days: float) -> bool:
    if len(time) < 2:
        return False
    baseline = float(np.nanmax(time) - np.nanmin(time))
    if not np.isfinite(baseline):
        return False
    if baseline < 1.5 * period_days:
        return True
    return bool(np.nanmin(time) >= 0 and np.nanmax(time) <= period_days * 1.1)


def _inputs_summary(lightcurve: LightCurveData, candidate: TransitCandidate) -> dict[str, Any]:
    mask = lightcurve.valid_mask
    time = lightcurve.time[mask]
    if time.size < 2:
        return {
            "n_points": int(time.size),
            "baseline_days": 0.0,
            "n_transits_expected": 0,
            "cadence_median_min": 0.0,
            "flux_err_available": lightcurve.flux_err is not None,
        }
    baseline_days = float(time.max() - time.min())
    n_transits = int(baseline_days / float(candidate.period)) if candidate.period > 0 else 0
    cadence = float(np.median(np.diff(time))) if time.size > 2 else 0.0
    return {
        "n_points": int(np.sum(mask)),
        "baseline_days": round(baseline_days, 2),
        "n_transits_expected": n_transits,
        "cadence_median_min": round(cadence * 24.0 * 60.0, 2),
        "flux_err_available": lightcurve.flux_err is not None,
    }


class _LightkurveLike:
    """Minimal lightkurve-like object for `exovetter` compatibility."""

    def __init__(self, time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray | None):
        self.time = time
        self.flux = flux
        self.flux_err = flux_err if flux_err is not None else np.ones_like(flux) * 0.001
        self.time_format = "btjd"


def _create_lightkurve_like(lightcurve: LightCurveData) -> _LightkurveLike:
    mask = lightcurve.valid_mask
    time = lightcurve.time[mask]
    flux = lightcurve.flux[mask]
    flux_err = lightcurve.flux_err[mask] if lightcurve.flux_err is not None else None
    return _LightkurveLike(time=time, flux=flux, flux_err=flux_err)


def run_modshift(
    *,
    candidate: TransitCandidate,
    lightcurve: LightCurveData,
) -> VetterCheckResult:
    """V11: ModShift metrics.

    Returns (when available): pri/sec/ter/pos, Fred, false_alarm_threshold,
    and any additional metrics returned by `exovetter`.
    """
    inputs_summary = _inputs_summary(lightcurve, candidate)
    time = lightcurve.time[lightcurve.valid_mask]
    if _is_likely_folded(time, float(candidate.period)):
        return _metrics_result(
            check_id="V11",
            name="modshift",
            confidence=0.0,
            details={
                "status": "invalid",
                "reason": "folded_input_detected",
                "warnings": ["FOLDED_INPUT_DETECTED"],
                "inputs_summary": inputs_summary,
            },
        )

    try:
        import astropy.units as u
        from exovetter import const as exo_const
        from exovetter.tce import Tce
        from exovetter.vetters import ModShift
    except Exception as e:
        return _metrics_result(
            check_id="V11",
            name="modshift",
            confidence=0.0,
            details={
                "status": "error",
                "reason": "exovetter_import_failed",
                "error": str(e),
                "warnings": ["EXOVETTER_IMPORT_ERROR"],
                "inputs_summary": inputs_summary,
            },
        )

    try:
        lk_obj = _create_lightkurve_like(lightcurve)
        tce = Tce(
            period=float(candidate.period) * u.day,
            epoch=float(candidate.t0) * u.day,
            epoch_offset=exo_const.btjd,
            depth=float(candidate.depth) * 1e6 * exo_const.ppm,
            duration=float(candidate.duration_hours) * u.hour,
        )
        vetter = ModShift(lc_name="flux")
        metrics = vetter.run(tce, lk_obj, plot=False)
    except Exception as e:
        logger.warning("ModShift failed: %s", e)
        return _metrics_result(
            check_id="V11",
            name="modshift",
            confidence=0.0,
            details={
                "status": "error",
                "reason": "modshift_execution_failed",
                "error": str(e),
                "warnings": ["MODSHIFT_EXECUTION_ERROR"],
                "inputs_summary": inputs_summary,
            },
        )

    metrics = _coerce_metrics_dict(dict(metrics))
    pri = float(metrics.get("pri", 0.0) or 0.0)
    sec = float(metrics.get("sec", 0.0) or 0.0)
    ter = float(metrics.get("ter", 0.0) or 0.0)
    fred = float(metrics.get("Fred", 0.0) or 0.0)
    fa = float(metrics.get("false_alarm_threshold", 0.0) or 0.0)

    sec_pri = (sec / pri) if pri > 0 else 0.0
    ter_pri = (ter / pri) if pri > 0 else 0.0

    n_transits = int(inputs_summary.get("n_transits_expected", 0))
    confidence = min(1.0, (n_transits / 5.0)) if n_transits > 0 else 0.5

    return _metrics_result(
        check_id="V11",
        name="modshift",
        confidence=confidence,
        details={
            "inputs_summary": inputs_summary,
            "primary_signal": round(pri, 6),
            "secondary_signal": round(sec, 6),
            "tertiary_signal": round(ter, 6),
            "fred": round(fred, 6),
            "false_alarm_threshold": round(fa, 6),
            "secondary_primary_ratio": round(sec_pri, 6),
            "tertiary_primary_ratio": round(ter_pri, 6),
            "raw_metrics": metrics,
        },
    )


def run_sweet(
    *,
    candidate: TransitCandidate,
    lightcurve: LightCurveData,
) -> VetterCheckResult:
    """V12: SWEET metrics."""
    inputs_summary = _inputs_summary(lightcurve, candidate)
    try:
        import astropy.units as u
        from exovetter import const as exo_const
        from exovetter.tce import Tce
        from exovetter.vetters import Sweet
    except Exception as e:
        return _metrics_result(
            check_id="V12",
            name="sweet",
            confidence=0.0,
            details={
                "status": "error",
                "reason": "exovetter_import_failed",
                "error": str(e),
                "warnings": ["EXOVETTER_IMPORT_ERROR"],
                "inputs_summary": inputs_summary,
            },
        )

    try:
        lk_obj = _create_lightkurve_like(lightcurve)
        tce = Tce(
            period=float(candidate.period) * u.day,
            epoch=float(candidate.t0) * u.day,
            epoch_offset=exo_const.btjd,
            depth=float(candidate.depth) * 1e6 * exo_const.ppm,
            duration=float(candidate.duration_hours) * u.hour,
        )
        vetter = Sweet(lc_name="flux")
        metrics = vetter.run(tce, lk_obj, plot=False)
    except Exception as e:
        logger.warning("SWEET failed: %s", e)
        return _metrics_result(
            check_id="V12",
            name="sweet",
            confidence=0.0,
            details={
                "status": "error",
                "reason": "sweet_execution_failed",
                "error": str(e),
                "warnings": ["SWEET_EXECUTION_ERROR"],
                "inputs_summary": inputs_summary,
            },
        )

    metrics = _coerce_metrics_dict(dict(metrics))
    n_points = int(inputs_summary.get("n_points", 0))
    confidence = min(1.0, n_points / 200.0) if n_points > 0 else 0.5

    return _metrics_result(
        check_id="V12",
        name="sweet",
        confidence=confidence,
        details={
            "inputs_summary": inputs_summary,
            "raw_metrics": metrics,
        },
    )


__all__ = ["run_modshift", "run_sweet"]

