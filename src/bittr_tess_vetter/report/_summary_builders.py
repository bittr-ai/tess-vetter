"""Pure summary builder helpers for report summary payload blocks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from bittr_tess_vetter.activity.rotation_context import build_rotation_context
from bittr_tess_vetter.report._serialization_utils import _coerce_finite_float, _coerce_int
from bittr_tess_vetter.validation.result_schema import CheckResult

if TYPE_CHECKING:
    from bittr_tess_vetter.report._data import (
        AliasHarmonicSummaryData,
        LCRobustnessData,
        LCSummary,
        SecondaryScanPlotData,
        TransitTimingPlotData,
    )

_SUMMARY_RELEVANT_CHECK_IDS: tuple[str, ...] = ("V01", "V02", "V04", "V05", "V13", "V15")
_REQUIRED_METRIC_KEYS_BY_CHECK: dict[str, tuple[str, ...]] = {
    "V01": ("delta_sigma", "depth_even_ppm", "depth_odd_ppm"),
    "V02": ("secondary_depth_sigma",),
    "V04": ("chi2_reduced",),
    "V05": ("tflat_ttotal_ratio",),
    "V13": (
        "missing_frac_max_in_coverage",
        "missing_frac_median_in_coverage",
        "n_epochs_evaluated_in_coverage",
        "n_epochs_excluded_no_coverage",
        "n_epochs_missing_ge_0p25_in_coverage",
    ),
    "V15": ("asymmetry_sigma",),
}


def _model_dump_like(value: Any) -> Any:
    """Serialize model-like objects without requiring pydantic at call sites."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _build_odd_even_summary(checks: dict[str, CheckResult]) -> dict[str, Any]:
    """Build deterministic odd/even summary from V01 metrics."""
    metrics = checks.get("V01").metrics if checks.get("V01") is not None else {}
    odd_depth_ppm = _coerce_finite_float(metrics.get("depth_odd_ppm"))
    if odd_depth_ppm is None:
        odd_depth_ppm = _coerce_finite_float(metrics.get("odd_depth"), scale=1e6)
    even_depth_ppm = _coerce_finite_float(metrics.get("depth_even_ppm"))
    if even_depth_ppm is None:
        even_depth_ppm = _coerce_finite_float(metrics.get("even_depth"), scale=1e6)
    depth_diff_ppm = _coerce_finite_float(metrics.get("delta_ppm"))
    if depth_diff_ppm is None and odd_depth_ppm is not None and even_depth_ppm is not None:
        depth_diff_ppm = odd_depth_ppm - even_depth_ppm
    depth_diff_sigma = _coerce_finite_float(metrics.get("depth_diff_sigma"))
    if depth_diff_sigma is None:
        depth_diff_sigma = _coerce_finite_float(metrics.get("delta_sigma"))
    is_significant = abs(depth_diff_sigma) >= 3.0 if depth_diff_sigma is not None else None
    flags: list[str] = []
    if is_significant is True:
        flags.append("ODD_EVEN_MISMATCH")
    return {
        "odd_depth_ppm": odd_depth_ppm,
        "even_depth_ppm": even_depth_ppm,
        "depth_diff_ppm": depth_diff_ppm,
        "depth_diff_sigma": depth_diff_sigma,
        "is_significant": is_significant,
        "flags": flags,
    }


def _build_noise_summary(
    lc_summary: LCSummary | None,
    lc_robustness: LCRobustnessData | None,
) -> dict[str, Any]:
    """Build deterministic noise summary from existing report metrics."""
    beta_30m = lc_robustness.red_noise.beta_30m if lc_robustness is not None else None
    beta_60m = lc_robustness.red_noise.beta_60m if lc_robustness is not None else None
    beta_duration = lc_robustness.red_noise.beta_duration if lc_robustness is not None else None
    slopes_per_day: list[float] = []
    if lc_robustness is not None:
        for epoch in lc_robustness.per_epoch:
            if epoch.baseline_slope_per_day is None:
                continue
            slope = _coerce_finite_float(epoch.baseline_slope_per_day)
            if slope is not None:
                slopes_per_day.append(abs(slope))
    trend_value = float(np.median(slopes_per_day)) if slopes_per_day else None
    flags: list[str] = []
    if beta_duration is not None and beta_duration > 1.2:
        flags.append("RED_NOISE_ELEVATED")
    if trend_value is not None and trend_value > 1e-3:
        flags.append("BASELINE_TREND_ELEVATED")
    return {
        "white_noise_ppm": lc_summary.flux_mad_ppm if lc_summary is not None else None,
        "red_noise_beta_30m": beta_30m,
        "red_noise_beta_60m": beta_60m,
        "red_noise_beta_duration": beta_duration,
        "trend_stat": trend_value,
        "trend_stat_unit": "relative_flux_per_day",
        "flags": flags,
        "semantics": {
            "white_noise_source": "lc_summary.flux_mad_ppm",
            "trend_source": "median(abs(lc_robustness.per_epoch.baseline_slope_per_day))",
        },
    }


def _build_variability_summary(
    lc_summary: LCSummary | None,
    timing_series: TransitTimingPlotData | None,
    alias_summary: AliasHarmonicSummaryData | None = None,
    stellar: Any | None = None,
) -> dict[str, Any]:
    """Build deterministic variability summary block."""
    variability_index: float | None = None
    if (
        lc_summary is not None
        and lc_summary.flux_mad_ppm > 0
        and math.isfinite(lc_summary.flux_std_ppm)
        and math.isfinite(lc_summary.flux_mad_ppm)
    ):
        variability_index = float(lc_summary.flux_std_ppm / lc_summary.flux_mad_ppm)
    periodicity_score = timing_series.periodicity_score if timing_series is not None else None
    alias_classification = (
        str(alias_summary.classification).upper()
        if alias_summary is not None and alias_summary.classification is not None
        else None
    )
    phase_shift_peak_sigma = (
        _coerce_finite_float(alias_summary.phase_shift_peak_sigma)
        if alias_summary is not None
        else None
    )
    phase_shift_event_count = (
        int(alias_summary.phase_shift_event_count)
        if alias_summary is not None and alias_summary.phase_shift_event_count is not None
        else None
    )
    secondary_significance = (
        _coerce_finite_float(alias_summary.secondary_significance)
        if alias_summary is not None
        else None
    )
    periodic_signal_present = bool(
        (periodicity_score is not None and periodicity_score >= 3.0)
        or alias_classification in {"ALIAS_WEAK", "ALIAS_STRONG"}
        or (phase_shift_peak_sigma is not None and phase_shift_peak_sigma >= 3.0)
        or (phase_shift_event_count is not None and phase_shift_event_count > 0)
        or (secondary_significance is not None and secondary_significance >= 3.0)
    )

    classification = "unknown"
    if variability_index is not None or periodicity_score is not None:
        var_level = variability_index if variability_index is not None else 1.0
        per_level = periodicity_score if periodicity_score is not None else 0.0
        if var_level >= 1.5 or per_level >= 3.0 or alias_classification == "ALIAS_STRONG":
            classification = "high_variability"
        elif (
            var_level >= 1.2
            or per_level >= 1.5
            or periodic_signal_present
            or alias_classification == "ALIAS_WEAK"
        ):
            classification = "moderate_variability"
        else:
            classification = "low_variability"

    flags: list[str] = []
    if variability_index is not None and variability_index >= 1.5:
        flags.append("ELEVATED_SCATTER")
    if periodicity_score is not None and periodicity_score >= 3.0:
        flags.append("PERIODIC_SIGNAL")
    if periodic_signal_present:
        flags.append("PERIODIC_SIGNAL_PRESENT")

    stellar_radius_rsun = None
    if stellar is not None:
        if hasattr(stellar, "radius"):
            stellar_radius_rsun = _coerce_finite_float(getattr(stellar, "radius"))
        elif isinstance(stellar, dict):
            stellar_radius_rsun = _coerce_finite_float(stellar.get("radius"))
    rotation_context = build_rotation_context(
        rotation_period_days=None,
        stellar_radius_rsun=stellar_radius_rsun,
        rotation_period_source_path=None,
        stellar_radius_source_path="summary.stellar.radius" if stellar_radius_rsun is not None else None,
        rotation_period_source_authority=None,
        stellar_radius_source_authority="report_seed_or_target_stellar" if stellar_radius_rsun is not None else None,
    )

    return {
        "variability_index": variability_index,
        "periodicity_score": periodicity_score,
        "flare_rate_per_day": None,
        "classification": classification,
        "flags": flags,
        "rotation_context": rotation_context,
        "semantics": {
            "variability_index_source": "lc_summary.flux_std_ppm/flux_mad_ppm",
            "periodicity_source": "timing_series.periodicity_score",
            "periodic_signal_present": periodic_signal_present,
            "alias_classification_source": "alias_summary.classification",
            "rotation_context_note": (
                "rotation_period_days not computed in report summary; "
                "combine with activity output to derive v_eq_est_kms."
            ),
        },
    }


def _build_alias_scalar_summary(
    alias_summary: AliasHarmonicSummaryData | None,
) -> dict[str, Any]:
    """Build scalar-only alias summary from compact harmonic diagnostics."""
    if alias_summary is None:
        return {
            "best_harmonic": None,
            "best_ratio_over_p": None,
            "score_p": None,
            "score_p_over_2": None,
            "score_2p": None,
            "depth_ppm_peak": None,
            "classification": None,
            "phase_shift_event_count": None,
            "phase_shift_peak_sigma": None,
            "secondary_significance": None,
            "alias_interpretation": None,
        }

    score_by_label: dict[str, float] = {}
    for label, score in zip(
        alias_summary.harmonic_labels,
        alias_summary.scores,
        strict=False,
    ):
        value = _coerce_finite_float(score)
        if value is not None:
            score_by_label[str(label)] = value

    depths = [
        d
        for d in (_coerce_finite_float(v) for v in alias_summary.harmonic_depth_ppm)
        if d is not None
    ]
    # Do not backfill this with harmonic scores; keep ppm semantics strict.
    depth_ppm_peak: float | None = float(max(depths)) if depths else None

    alias_classification = (
        str(alias_summary.classification).upper() if alias_summary.classification is not None else None
    )
    phase_shift_peak_sigma = _coerce_finite_float(alias_summary.phase_shift_peak_sigma)
    secondary_significance = _coerce_finite_float(alias_summary.secondary_significance)
    alias_interpretation = "no_alias_evidence"
    if alias_classification == "ALIAS_STRONG":
        alias_interpretation = "strong_alias_preferred"
    elif alias_classification == "ALIAS_WEAK":
        alias_interpretation = "weak_alias_candidate"
    elif (
        (phase_shift_peak_sigma is not None and phase_shift_peak_sigma >= 3.0)
        or (secondary_significance is not None and secondary_significance >= 3.0)
    ):
        alias_interpretation = "phase_shift_or_secondary_caution"

    return {
        "best_harmonic": str(alias_summary.best_harmonic) if alias_summary.best_harmonic else None,
        "best_ratio_over_p": _coerce_finite_float(alias_summary.best_ratio_over_p),
        "score_p": score_by_label.get("P"),
        "score_p_over_2": score_by_label.get("P/2"),
        "score_2p": score_by_label.get("2P"),
        "depth_ppm_peak": depth_ppm_peak,
        "classification": alias_summary.classification,
        "phase_shift_event_count": (
            int(alias_summary.phase_shift_event_count)
            if alias_summary.phase_shift_event_count is not None
            else None
        ),
        "phase_shift_peak_sigma": phase_shift_peak_sigma,
        "secondary_significance": secondary_significance,
        "alias_interpretation": alias_interpretation,
    }


def _build_timing_summary(
    timing_series: TransitTimingPlotData | None,
    checks: dict[str, CheckResult] | None = None,
) -> dict[str, Any]:
    """Build scalar timing rollup from existing per-epoch timing series."""
    v04_metrics = {}
    if checks is not None and checks.get("V04") is not None:
        v04_metrics = checks["V04"].metrics
    n_transits_measured = _coerce_int(v04_metrics.get("n_transits_measured"))
    depth_scatter_ppm = _coerce_finite_float(v04_metrics.get("depth_scatter_ppm"))
    chi2_reduced = _coerce_finite_float(v04_metrics.get("chi2_reduced"))

    if timing_series is None:
        return {
            "n_epochs_measured": 0,
            "rms_seconds": None,
            "periodicity_score": None,
            "linear_trend_sec_per_epoch": None,
            "max_abs_oc_seconds": None,
            "max_snr": None,
            "snr_median": None,
            "oc_median": None,
            "outlier_count": 0,
            "outlier_fraction": None,
            "deepest_epoch": None,
            "n_transits_measured": n_transits_measured,
            "depth_scatter_ppm": depth_scatter_ppm,
            "chi2_reduced": chi2_reduced,
        }

    epochs = [int(e) for e in timing_series.epochs]
    oc_seconds = [_coerce_finite_float(v) for v in timing_series.oc_seconds]
    snr_values = [_coerce_finite_float(v) for v in timing_series.snr]
    n_epochs_measured = len(epochs)

    finite_oc = [abs(v) for v in oc_seconds if v is not None]
    max_abs_oc_seconds = float(max(finite_oc)) if finite_oc else None

    finite_snr = [v for v in snr_values if v is not None]
    max_snr = float(max(finite_snr)) if finite_snr else None
    snr_median = float(np.median(finite_snr)) if finite_snr else None
    finite_oc_signed = [v for v in oc_seconds if v is not None]
    oc_median = float(np.median(finite_oc_signed)) if finite_oc_signed else None

    # Outlier policy is deterministic and derived from existing O-C + RMS only.
    rms_seconds = _coerce_finite_float(timing_series.rms_seconds)
    outlier_count = 0
    if rms_seconds is not None and rms_seconds > 0:
        threshold = 3.0 * rms_seconds
        outlier_count = sum(
            1
            for value in oc_seconds
            if value is not None and abs(value) > threshold
        )

    outlier_fraction = (
        float(outlier_count) / float(n_epochs_measured)
        if n_epochs_measured > 0
        else None
    )

    deepest_epoch: int | None = None
    best_snr: float | None = None
    for epoch, snr in zip(epochs, snr_values, strict=False):
        if snr is None:
            continue
        if best_snr is None or snr > best_snr or (snr == best_snr and epoch < deepest_epoch):
            best_snr = snr
            deepest_epoch = int(epoch)

    return {
        "n_epochs_measured": n_epochs_measured,
        "rms_seconds": rms_seconds,
        "periodicity_score": _coerce_finite_float(timing_series.periodicity_score),
        "linear_trend_sec_per_epoch": _coerce_finite_float(
            timing_series.linear_trend_sec_per_epoch
        ),
        "max_abs_oc_seconds": max_abs_oc_seconds,
        "max_snr": max_snr,
        "snr_median": snr_median,
        "oc_median": oc_median,
        "outlier_count": int(outlier_count),
        "outlier_fraction": outlier_fraction,
        "deepest_epoch": deepest_epoch,
        "n_transits_measured": n_transits_measured,
        "depth_scatter_ppm": depth_scatter_ppm,
        "chi2_reduced": chi2_reduced,
    }


def _build_secondary_scan_summary(
    secondary_scan: SecondaryScanPlotData | None,
) -> dict[str, Any]:
    """Build scalar secondary-scan quality and dip rollup."""
    if secondary_scan is None:
        return {
            "n_raw_points": None,
            "n_bins": None,
            "phase_coverage_fraction": None,
            "largest_phase_gap": None,
            "n_bins_with_error": None,
            "strongest_dip_phase": None,
            "strongest_dip_depth_ppm": None,
            "is_degraded": None,
            "quality_flag_count": 0,
        }

    quality = secondary_scan.quality
    strongest_dip_depth_ppm = None
    strongest_dip_flux = _coerce_finite_float(secondary_scan.strongest_dip_flux)
    if strongest_dip_flux is not None:
        strongest_dip_depth_ppm = float((1.0 - strongest_dip_flux) * 1e6)

    quality_flag_count = 0
    if quality is not None and quality.flags is not None:
        quality_flag_count = int(len(quality.flags))

    return {
        "n_raw_points": (
            int(quality.n_raw_points) if quality is not None else int(len(secondary_scan.phase))
        ),
        "n_bins": int(quality.n_bins) if quality is not None else int(len(secondary_scan.bin_centers)),
        "phase_coverage_fraction": (
            _coerce_finite_float(quality.phase_coverage_fraction) if quality is not None else None
        ),
        "largest_phase_gap": (
            _coerce_finite_float(quality.largest_phase_gap) if quality is not None else None
        ),
        "n_bins_with_error": int(quality.n_bins_with_error) if quality is not None else None,
        "strongest_dip_phase": _coerce_finite_float(secondary_scan.strongest_dip_phase),
        "strongest_dip_depth_ppm": strongest_dip_depth_ppm,
        "is_degraded": bool(quality.is_degraded) if quality is not None else None,
        "quality_flag_count": quality_flag_count,
    }


def _build_data_gap_summary(checks: dict[str, CheckResult]) -> dict[str, Any]:
    """Build scalar V13 in-coverage data-gap summary."""
    metrics = checks.get("V13").metrics if checks.get("V13") is not None else {}
    return {
        "missing_frac_max_in_coverage": _coerce_finite_float(
            metrics.get("missing_frac_max_in_coverage")
        ),
        "missing_frac_median_in_coverage": _coerce_finite_float(
            metrics.get("missing_frac_median_in_coverage")
        ),
        "n_epochs_missing_ge_0p25_in_coverage": _coerce_int(
            metrics.get("n_epochs_missing_ge_0p25_in_coverage")
        ),
        "n_epochs_excluded_no_coverage": _coerce_int(
            metrics.get("n_epochs_excluded_no_coverage")
        ),
        "n_epochs_evaluated_in_coverage": _coerce_int(
            metrics.get("n_epochs_evaluated_in_coverage")
        ),
    }


def _build_lc_robustness_summary(lc_robustness: Any | None) -> dict[str, Any] | None:
    """Build scalar LC robustness summary from compact robustness payload."""
    if lc_robustness is None:
        return None
    rb = lc_robustness.robustness
    rn = lc_robustness.red_noise
    fp = lc_robustness.fp_signals
    return {
        "version": lc_robustness.version,
        "n_epochs_stored": len(lc_robustness.per_epoch),
        "n_epochs_measured": rb.n_epochs_measured,
        "dominance_index": rb.dominance_index,
        "loto_snr_min": rb.loto_snr_min,
        "loto_snr_mean": rb.loto_snr_mean,
        "loto_snr_max": rb.loto_snr_max,
        "loto_depth_ppm_min": rb.loto_depth_ppm_min,
        "loto_depth_ppm_max": rb.loto_depth_ppm_max,
        "loto_depth_shift_ppm_max": rb.loto_depth_shift_ppm_max,
        "beta_30m": rn.beta_30m,
        "beta_60m": rn.beta_60m,
        "beta_duration": rn.beta_duration,
        "odd_even_depth_diff_sigma": fp.odd_even_depth_diff_sigma,
        "secondary_depth_sigma": fp.secondary_depth_sigma,
        "phase_0p5_bin_depth_ppm": fp.phase_0p5_bin_depth_ppm,
        "v_shape_metric": fp.v_shape_metric,
        "asymmetry_sigma": fp.asymmetry_sigma,
    }


def _build_check_metric_contract_meta(checks: dict[str, CheckResult]) -> dict[str, Any]:
    """Build deterministic metric-contract introspection metadata."""
    required_metrics_by_check: dict[str, list[str]] = {}
    missing_required_metrics_by_check: dict[str, list[str]] = {}
    metric_keys_by_check: dict[str, list[str]] = {}

    for check_id in sorted(_SUMMARY_RELEVANT_CHECK_IDS):
        required_keys = sorted(_REQUIRED_METRIC_KEYS_BY_CHECK.get(check_id, ()))
        required_metrics_by_check[check_id] = list(required_keys)

        check = checks.get(check_id)
        raw_metrics = check.metrics if check is not None else {}
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        metric_keys = sorted(str(key) for key in metrics)
        metric_keys_by_check[check_id] = metric_keys

        if check is None:
            continue

        missing_keys = [key for key in required_keys if key not in metrics]
        if missing_keys:
            missing_required_metrics_by_check[check_id] = missing_keys

    return {
        "contract_version": "1",
        "required_metrics_by_check": required_metrics_by_check,
        "missing_required_metrics_by_check": missing_required_metrics_by_check,
        "metric_keys_by_check": metric_keys_by_check,
        "has_missing_required_metrics": bool(missing_required_metrics_by_check),
    }
