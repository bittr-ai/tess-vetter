"""Exovetter-based computations (metrics-only).

Wraps selected `exovetter` vetters and returns metrics as `VetterCheckResult`
objects with `passed=None` and `details["_metrics_only"]=True`.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from tess_vetter.domain.detection import TransitCandidate, VetterCheckResult
from tess_vetter.domain.lightcurve import LightCurveData

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
        elif isinstance(v, np.ndarray):
            # Convert numpy arrays to nested lists with Python scalars
            out[k] = _ndarray_to_list(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [_coerce_scalar(x) for x in v]
        else:
            out[k] = _coerce_scalar(v)
    return out


def _ndarray_to_list(arr: np.ndarray) -> list[Any]:
    """Convert numpy array to nested list of Python scalars."""
    if arr.ndim == 0:
        return _coerce_scalar(arr.item())  # type: ignore[return-value]
    return [_ndarray_to_list(x) if isinstance(x, np.ndarray) else _coerce_scalar(x) for x in arr]


def _as_jsonable_metrics(metrics: Any) -> dict[str, Any]:
    """Coerce exovetter metrics to a JSON-serializable dict.

    exovetter vetters may return dict-like objects and may include NumPy scalar
    types; we normalize these into plain Python scalars/lists/dicts.
    """
    return _coerce_metrics_dict(dict(metrics))


def _is_likely_folded(time: np.ndarray, period_days: float) -> bool:
    if len(time) < 2:
        return False
    baseline = float(np.nanmax(time) - np.nanmin(time))
    if not np.isfinite(baseline):
        return False
    # Do not treat "baseline shorter than the period" as folded: long-period
    # candidates observed for a single sector can have baseline < P but are still
    # provided in absolute BTJD time. The reliable folded-input indicator is that
    # the time axis itself lives in a [0, ~P] range.
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

    metrics = _as_jsonable_metrics(metrics)
    fred = float(metrics.get("Fred", 0.0) or 0.0)
    fa = float(metrics.get("false_alarm_threshold", 0.0) or 0.0)

    # exovetter ModShift returns both:
    # - `pri/sec/ter/pos`: often large integers (phase-bin indices) and NOT reliable as signal strengths
    # - `sigma_pri/sigma_sec/sigma_ter/sigma_pos`: the signal significances we want for ratios
    #
    # Use sigma_* when available; fall back to pri/sec/... only if sigma_* missing.
    def _f(key: str) -> float | None:
        v = metrics.get(key)
        if v is None:
            return None
        try:
            x = float(v)
        except Exception:
            return None
        return x if np.isfinite(x) else None

    pri_sig = _f("sigma_pri")
    sec_sig = _f("sigma_sec")
    ter_sig = _f("sigma_ter")
    pos_sig = _f("sigma_pos")

    if pri_sig is None:
        pri_sig = _f("pri") or 0.0
    if sec_sig is None:
        sec_sig = _f("sec") or 0.0
    if ter_sig is None:
        ter_sig = _f("ter") or 0.0
    if pos_sig is None:
        pos_sig = _f("pos") or 0.0

    # ModShift signals can be negative depending on sign conventions.
    # Ratios should be computed against the magnitude of the primary signal.
    pri_abs = abs(float(pri_sig))
    sec_abs = abs(float(sec_sig))
    ter_abs = abs(float(ter_sig))
    pos_abs = abs(float(pos_sig))

    sec_pri = (sec_abs / pri_abs) if pri_abs > 0 else 0.0
    ter_pri = (ter_abs / pri_abs) if pri_abs > 0 else 0.0
    pos_pri = (pos_abs / pri_abs) if pri_abs > 0 else 0.0

    n_transits = int(inputs_summary.get("n_transits_expected", 0))
    confidence = min(1.0, (n_transits / 5.0)) if n_transits > 0 else 0.5

    # Plot data (lightweight): exovetter does not expose the full ModShift
    # periodogram, so we synthesize a simple “bump” representation centered on
    # the reported peak phases to enable stable plotting in notebooks/docs.
    def _phase(v: object | None) -> float | None:
        if v is None:
            return None
        try:
            x = float(v)
        except Exception:
            return None
        if not np.isfinite(x):
            return None
        # exovetter can emit phases outside [0,1); normalize defensively.
        return float(x % 1.0)

    phase_pri = _phase(metrics.get("phase_pri")) or 0.0
    phase_sec_f = _phase(metrics.get("phase_sec"))

    phase_bins = np.linspace(0.0, 1.0, 200, dtype=np.float64)
    periodogram = np.zeros_like(phase_bins)

    def _circ_dist(x: np.ndarray, p: float) -> np.ndarray:
        d = np.abs(x - p)
        return np.minimum(d, 1.0 - d)

    def _add_bump(p: float | None, amp: float) -> None:
        if p is None or not np.isfinite(p):
            return
        sigma = 0.015
        periodogram[:] += amp * np.exp(-0.5 * (_circ_dist(phase_bins, float(p)) / sigma) ** 2)

    _add_bump(phase_pri, pri_abs)
    _add_bump(phase_sec_f, sec_abs)
    _add_bump(_phase(metrics.get("phase_ter")) or 0.0, ter_abs)
    _add_bump(_phase(metrics.get("phase_pos")) or 0.0, pos_abs)

    plot_data: dict[str, Any] = {
        "version": 1,
        "phase_bins": phase_bins.tolist(),
        "periodogram": periodogram.tolist(),
        "primary_phase": phase_pri,
        "primary_signal": pri_abs,
        "secondary_phase": phase_sec_f,
        "secondary_signal": sec_abs if phase_sec_f is not None else None,
    }

    return _metrics_result(
        check_id="V11",
        name="modshift",
        confidence=confidence,
        details={
            "inputs_summary": inputs_summary,
            # Backward-compatible keys (now magnitude-based for interpretability)
            "primary_signal": round(pri_abs, 6),
            "secondary_signal": round(sec_abs, 6),
            "tertiary_signal": round(ter_abs, 6),
            "positive_signal": round(pos_abs, 6),
            # Signed values for debugging / traceability
            "primary_signal_signed": round(float(pri_sig), 6),
            "secondary_signal_signed": round(float(sec_sig), 6),
            "tertiary_signal_signed": round(float(ter_sig), 6),
            "positive_signal_signed": round(float(pos_sig), 6),
            "fred": round(fred, 6),
            "false_alarm_threshold": round(fa, 6),
            "secondary_primary_ratio": round(sec_pri, 6),
            "tertiary_primary_ratio": round(ter_pri, 6),
            "positive_primary_ratio": round(pos_pri, 6),
            "raw_metrics": metrics,
            "plot_data": plot_data,
        },
    )


def run_sweet(
    *,
    candidate: TransitCandidate,
    lightcurve: LightCurveData,
) -> VetterCheckResult:
    """V12: SWEET metrics.

    SWEET (Sine Wave Evaluation for Ephemeris Transits) fits sinusoids at
    P/2, P, and 2P to out-of-transit flux to detect stellar variability.

    Returns (when available):
    - snr_half_period: SNR of sinusoid fit at P/2 (even harmonics)
    - snr_at_period: SNR of sinusoid fit at P (direct variability)
    - snr_double_period: SNR of sinusoid fit at 2P (subharmonics)
    - msg: Human-readable pass/fail message (legacy, less useful)

    The amp array from exovetter has shape (3, 3):
    - Row 0 = P/2, Row 1 = P, Row 2 = 2P
    - Columns = [amplitude, uncertainty, SNR]
    """
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

    metrics = _as_jsonable_metrics(metrics)
    n_points = int(inputs_summary.get("n_points", 0))
    confidence = min(1.0, n_points / 200.0) if n_points > 0 else 0.5

    # Extract SNR values from the amp array
    # amp has shape (3, 3): [[amp, unc, SNR], ...] for P/2, P, 2P
    snr_half_period: float | None = None
    snr_at_period: float | None = None
    snr_double_period: float | None = None

    amp = metrics.get("amp")
    if amp is not None and isinstance(amp, list) and len(amp) >= 3:
        try:
            # Row 0 = P/2, Row 1 = P, Row 2 = 2P; column 2 = SNR
            if isinstance(amp[0], list) and len(amp[0]) >= 3:
                snr_half_period = float(amp[0][2])
            if isinstance(amp[1], list) and len(amp[1]) >= 3:
                snr_at_period = float(amp[1][2])
            if isinstance(amp[2], list) and len(amp[2]) >= 3:
                snr_double_period = float(amp[2][2])
        except (IndexError, TypeError, ValueError) as e:
            logger.debug("Failed to extract SWEET SNR values: %s", e)

    # Plot data (lightweight): build a deterministic phase-folded view and simple
    # linear least-squares sinusoid fits at P/2, P, and 2P. This enables plotting
    # without relying on exovetter internals to expose fit arrays.
    mask = lightcurve.valid_mask & np.isfinite(lightcurve.time) & np.isfinite(lightcurve.flux)
    time = lightcurve.time[mask]
    flux = lightcurve.flux[mask]

    if time.size > 0 and float(candidate.period) > 0:
        phase = ((time - float(candidate.t0)) / float(candidate.period)) % 1.0
        flux_norm = flux / (np.nanmedian(flux) if np.isfinite(np.nanmedian(flux)) else 1.0)

        # Subsample for plot stability
        max_points = 600
        if phase.size > max_points:
            idx = np.linspace(0, phase.size - 1, max_points).astype(int)
            phase = phase[idx]
            flux_norm = flux_norm[idx]

        def _fit_sinusoid(k: float) -> np.ndarray:
            # Model: a*sin(2πkφ) + b*cos(2πkφ) + c
            w = 2.0 * np.pi * k * phase
            x = np.column_stack([np.sin(w), np.cos(w), np.ones_like(w)])
            y = flux_norm
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
            return (x @ beta).astype(np.float64)

        half_fit = _fit_sinusoid(2.0)  # P/2 => 2 cycles per phase
        at_fit = _fit_sinusoid(1.0)  # P
        double_fit = _fit_sinusoid(0.5)  # 2P => 0.5 cycles per phase

        plot_data = {
            "version": 1,
            "phase": phase.astype(np.float64).tolist(),
            "flux": flux_norm.astype(np.float64).tolist(),
            "half_period_fit": half_fit.tolist(),
            "at_period_fit": at_fit.tolist(),
            "double_period_fit": double_fit.tolist(),
            "snr_half_period": snr_half_period,
            "snr_at_period": snr_at_period,
            "snr_double_period": snr_double_period,
        }
    else:
        plot_data = {
            "version": 1,
            "phase": [],
            "flux": [],
            "half_period_fit": None,
            "at_period_fit": None,
            "double_period_fit": None,
            "snr_half_period": snr_half_period,
            "snr_at_period": snr_at_period,
            "snr_double_period": snr_double_period,
        }

    return _metrics_result(
        check_id="V12",
        name="sweet",
        confidence=confidence,
        details={
            "inputs_summary": inputs_summary,
            "snr_half_period": snr_half_period,
            "snr_at_period": snr_at_period,
            "snr_double_period": snr_double_period,
            "raw_metrics": metrics,
            "plot_data": plot_data,
        },
    )


__all__ = ["run_modshift", "run_sweet"]
