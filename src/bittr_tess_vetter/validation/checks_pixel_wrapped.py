"""Pixel-level vetting check wrappers implementing VettingCheck protocol.

This module wraps the existing pixel-level check implementations (V08-V10)
to conform to the new VettingCheck protocol and CheckResult schema.

Check Summary:
- V08 CentroidShiftCheck: Compare in-transit vs out-of-transit centroid positions
- V09 DifferenceImageCheck: Analyze pixel-level light curves to locate transit source
- V10 ApertureDependenceCheck: Measure transit depth vs aperture size

Novelty: standard (wrapping existing implementations)

References:
    [1] Bryson et al. 2013, PASP 125, 889 - Kepler pixel-level diagnostics
    [2] Twicken et al. 2018, PASP 130, 064502 - Kepler Data Validation
    [3] Torres et al. 2011, ApJ 727, 24 - Background blend detection
    [4] Guerrero et al. 2021, ApJS 254, 39 - TESS TOI catalog vetting
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.validation.checks_pixel import (
    check_aperture_dependence_with_tpf,
    check_centroid_shift_with_tpf,
    check_pixel_level_lc_with_tpf,
)
from bittr_tess_vetter.validation.registry import (
    CheckConfig,
    CheckInputs,
    CheckRegistry,
    CheckRequirements,
    CheckTier,
)
from bittr_tess_vetter.validation.result_schema import (
    CheckResult,
    error_result,
    ok_result,
    skipped_result,
)


def _extract_tpf_arrays(
    tpf: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract time and flux arrays from TPF data.

    Args:
        tpf: TPFStamp or compatible object with time and flux attributes.

    Returns:
        Tuple of (time_array, flux_array) as float64 arrays.
    """
    time_arr = np.asarray(tpf.time, dtype=np.float64)
    flux_arr = np.asarray(tpf.flux, dtype=np.float64)
    return time_arr, flux_arr


def _lc_tpf_time_overlap_metrics(
    inputs: CheckInputs,
    *,
    tpf_time: np.ndarray,
) -> tuple[dict[str, float | int | str | bool | None], list[str], list[str]]:
    """Compute objective LC vs TPF timebase overlap metrics.

    Pixel checks run on the TPF timebase, but many hosts also pass an LC for
    other checks. When LC and TPF are from different sectors or targets, the
    pixel diagnostics can be misleading. This helper surfaces a purely
    time-range-based guardrail.

    Returns:
        (metrics, flags, notes)
    """
    flags: list[str] = []
    notes: list[str] = []

    lc_time = np.asarray(inputs.lc.time, dtype=np.float64)
    lc_valid = np.asarray(inputs.lc.valid_mask, dtype=bool)
    lc_time = lc_time[lc_valid]
    lc_time = lc_time[np.isfinite(lc_time)]

    tpf_time = np.asarray(tpf_time, dtype=np.float64)
    tpf_time = tpf_time[np.isfinite(tpf_time)]

    if lc_time.size == 0 or tpf_time.size == 0:
        flags.append("LC_TPF_TIME_INVALID")
        notes.append("Could not compute LC/TPF time overlap (empty or non-finite time arrays).")
        return (
            {
                "lc_time_min_btjd": None,
                "lc_time_max_btjd": None,
                "tpf_time_min_btjd": None,
                "tpf_time_max_btjd": None,
                "lc_tpf_time_overlap_days": None,
                "lc_tpf_time_overlap_fraction_union": None,
                "tpf_fraction_within_lc_timerange": None,
            },
            flags,
            notes,
        )

    lc_min = float(np.min(lc_time))
    lc_max = float(np.max(lc_time))
    tpf_min = float(np.min(tpf_time))
    tpf_max = float(np.max(tpf_time))

    overlap_days = float(max(0.0, min(lc_max, tpf_max) - max(lc_min, tpf_min)))
    union_days = float(max(lc_max, tpf_max) - min(lc_min, tpf_min))
    overlap_frac = float(overlap_days / union_days) if union_days > 0 else 1.0

    within = (tpf_time >= lc_min) & (tpf_time <= lc_max)
    tpf_fraction_within = float(np.mean(within)) if tpf_time.size else float("nan")

    if overlap_days == 0.0:
        flags.append("LC_TPF_TIME_DISJOINT")
        notes.append("LC and TPF time ranges are disjoint; pixel diagnostics may be meaningless.")

    return (
        {
            "lc_time_min_btjd": lc_min,
            "lc_time_max_btjd": lc_max,
            "tpf_time_min_btjd": tpf_min,
            "tpf_time_max_btjd": tpf_max,
            "lc_tpf_time_overlap_days": overlap_days,
            "lc_tpf_time_overlap_fraction_union": overlap_frac,
            "tpf_fraction_within_lc_timerange": tpf_fraction_within,
        },
        flags,
        notes,
    )


def _candidate_to_internal(candidate: Any) -> TransitCandidate:
    """Convert API Candidate or internal TransitCandidate to TransitCandidate.

    Args:
        candidate: Either an API Candidate with ephemeris attribute,
            or an internal TransitCandidate with flat fields.

    Returns:
        Internal TransitCandidate for vetting checks.
    """
    # If already a TransitCandidate, use it directly
    if isinstance(candidate, TransitCandidate):
        return candidate

    # Handle API Candidate with nested ephemeris
    if hasattr(candidate, "ephemeris"):
        return TransitCandidate(
            period=candidate.ephemeris.period_days,
            t0=candidate.ephemeris.t0_btjd,
            duration_hours=candidate.ephemeris.duration_hours,
            depth=candidate.depth or 0.001,
            snr=0.0,  # Placeholder - not used by pixel checks
        )

    # Handle object with flat fields (like TransitCandidate but not instance)
    return TransitCandidate(
        period=candidate.period,
        t0=candidate.t0,
        duration_hours=candidate.duration_hours,
        depth=getattr(candidate, "depth", None) or 0.001,
        snr=getattr(candidate, "snr", 0.0),
    )


def _coerce_scalar(v: Any) -> float | int | str | bool | None:
    """Coerce numpy scalars to Python primitives."""
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (float, int, str, bool, type(None))):
        return v
    return str(v)


def _flatten_metrics(
    details: dict[str, Any],
    prefix: str = "",
) -> dict[str, float | int | str | bool | None]:
    """Flatten nested dict to flat metrics with JSON-serializable values.

    Args:
        details: Possibly nested dict from legacy check result.
        prefix: Key prefix for nested values.

    Returns:
        Flat dict with only JSON-primitive values.
    """
    metrics: dict[str, float | int | str | bool | None] = {}
    for k, v in details.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            metrics.update(_flatten_metrics(v, f"{key}_"))
        elif isinstance(v, (list, tuple)):
            # Store list length and first few values if numeric
            metrics[f"{key}_count"] = len(v)
            if len(v) > 0 and isinstance(v[0], (int, float, np.floating, np.integer)):
                for i, item in enumerate(v[:5]):  # First 5 elements max
                    metrics[f"{key}_{i}"] = _coerce_scalar(item)
        else:
            metrics[key] = _coerce_scalar(v)
    return metrics


class CentroidShiftCheck:
    """V08: Centroid shift analysis during transit.

    Compares the flux-weighted centroid position during transit versus
    out-of-transit. A significant shift indicates the transit source is
    not the target star, but a nearby or background eclipsing binary.

    References:
        [1] Bryson et al. 2013, PASP 125, 889 - Section 3.1: Centroid offset test
        [2] Twicken et al. 2018, PASP 130, 064502 - Section 4.1: Difference image centroids
    """

    _id = "V08"
    _name = "Centroid Shift"
    _tier = CheckTier.PIXEL
    _requirements = CheckRequirements(needs_tpf=True)
    _citations = [
        "Bryson et al. 2013, PASP 125, 889",
        "Twicken et al. 2018, PASP 130, 064502",
    ]

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> CheckTier:
        return self._tier

    @property
    def requirements(self) -> CheckRequirements:
        return self._requirements

    @property
    def citations(self) -> list[str]:
        return self._citations

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute centroid shift check.

        Args:
            inputs: Check inputs containing TPF and candidate.
            config: Check configuration.

        Returns:
            CheckResult with centroid shift metrics.
        """
        if inputs.tpf is None:
            return skipped_result(
                self.id,
                self.name,
                reason_flag="NO_TPF",
                notes=["TPF data not provided; cannot perform centroid analysis"],
            )

        try:
            time_arr, flux_arr = _extract_tpf_arrays(inputs.tpf)
            internal_candidate = _candidate_to_internal(inputs.candidate)

            # Extract config parameters
            extra = config.extra_params
            overlap_metrics, overlap_flags, overlap_notes = _lc_tpf_time_overlap_metrics(
                inputs, tpf_time=time_arr
            )
            result = check_centroid_shift_with_tpf(
                tpf_data=flux_arr,
                time=time_arr,
                candidate=internal_candidate,
                centroid_method=extra.get("centroid_method", "median"),
                significance_method=extra.get("significance_method", "bootstrap"),
                n_bootstrap=int(extra.get("n_bootstrap", 1000)),
                bootstrap_seed=extra.get("bootstrap_seed") or config.random_seed,
                outlier_sigma=float(extra.get("outlier_sigma", 3.0)),
                window_policy_version=extra.get("window_policy_version", "v1"),
            )

            # Extract key metrics from legacy result
            details = dict(result.details)
            flags: list[str] = []
            notes: list[str] = []

            # Collect warnings as flags
            warnings = details.get("warnings", [])
            for w in warnings:
                if isinstance(w, str):
                    flags.append(w)

            if details.get("saturation_risk"):
                flags.append("SATURATION_RISK")

            # Build structured metrics
            metrics: dict[str, float | int | str | bool | None] = {
                **overlap_metrics,
                "centroid_shift_pixels": details.get("centroid_shift_pixels"),
                "centroid_shift_arcsec": details.get("centroid_shift_arcsec"),
                "significance_sigma": details.get("significance_sigma"),
                "shift_uncertainty_pixels": details.get("shift_uncertainty_pixels"),
                "shift_ci_lower_pixels": details.get("shift_ci_lower_pixels"),
                "shift_ci_upper_pixels": details.get("shift_ci_upper_pixels"),
                "n_in_transit_cadences": details.get("n_in_transit_cadences"),
                "n_out_of_transit_cadences": details.get("n_out_of_transit_cadences"),
                "centroid_method": details.get("centroid_method"),
                "significance_method": details.get("significance_method"),
                "n_bootstrap": details.get("n_bootstrap"),
                "saturation_risk": details.get("saturation_risk"),
                "max_flux_fraction": details.get("max_flux_fraction"),
                "n_outliers_rejected": details.get("n_outliers_rejected"),
            }

            # Store centroid coordinates (flatten tuples)
            in_centroid = details.get("in_transit_centroid")
            if in_centroid and len(in_centroid) == 2:
                metrics["in_transit_centroid_x"] = float(in_centroid[0])
                metrics["in_transit_centroid_y"] = float(in_centroid[1])

            out_centroid = details.get("out_of_transit_centroid")
            if out_centroid and len(out_centroid) == 2:
                metrics["out_of_transit_centroid_x"] = float(out_centroid[0])
                metrics["out_of_transit_centroid_y"] = float(out_centroid[1])

            flags.extend(overlap_flags)
            notes.extend(overlap_notes)

            return ok_result(
                self.id,
                self.name,
                metrics=metrics,
                confidence=result.confidence,
                flags=flags,
                notes=notes,
                raw=details,
            )

        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )


class DifferenceImageCheck:
    """V09: Pixel-level depth map analysis.

    Extracts light curves from individual pixels, measures transit depth
    in each, and determines if the signal is on-target. This is a proxy
    for difference image localization when full WCS is not available.

    References:
        [1] Bryson et al. 2013, PASP 125, 889 - Section 3.2: Difference image analysis
        [2] Torres et al. 2011, ApJ 727, 24 - Section 4: Blend scenarios
        [3] Twicken et al. 2018, PASP 130, 064502 - Section 4.2: Difference image PRF
    """

    _id = "V09"
    _name = "Difference Image"
    _tier = CheckTier.PIXEL
    _requirements = CheckRequirements(needs_tpf=True)
    _citations = [
        "Bryson et al. 2013, PASP 125, 889",
        "Torres et al. 2011, ApJ 727, 24",
        "Twicken et al. 2018, PASP 130, 064502",
    ]

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> CheckTier:
        return self._tier

    @property
    def requirements(self) -> CheckRequirements:
        return self._requirements

    @property
    def citations(self) -> list[str]:
        return self._citations

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute difference image localization check.

        Args:
            inputs: Check inputs containing TPF and candidate.
            config: Check configuration.

        Returns:
            CheckResult with pixel-level localization metrics.
        """
        if inputs.tpf is None:
            return skipped_result(
                self.id,
                self.name,
                reason_flag="NO_TPF",
                notes=["TPF data not provided; cannot perform pixel-level analysis"],
            )

        try:
            time_arr, flux_arr = _extract_tpf_arrays(inputs.tpf)
            internal_candidate = _candidate_to_internal(inputs.candidate)

            # Extract target pixel from config
            extra = config.extra_params
            overlap_metrics, overlap_flags, overlap_notes = _lc_tpf_time_overlap_metrics(
                inputs, tpf_time=time_arr
            )
            target_pixel: tuple[int, int] | None = None
            target_rc = extra.get("target_rc")
            if target_rc is not None:
                target_pixel = (int(target_rc[0]), int(target_rc[1]))

            result = check_pixel_level_lc_with_tpf(
                tpf_data=flux_arr,
                time=time_arr,
                candidate=internal_candidate,
                target_pixel=target_pixel,
            )

            details = dict(result.details)
            status = details.get("status", "ok")

            # Handle error/insufficient data cases
            if status == "insufficient_data":
                return skipped_result(
                    self.id,
                    self.name,
                    reason_flag="INSUFFICIENT_DATA",
                    notes=[details.get("error", "Insufficient cadences for analysis")],
                    raw=details,
                )
            if status == "error":
                return error_result(
                    self.id,
                    self.name,
                    error=details.get("error", "Unknown error"),
                    raw=details,
                )

            # Build metrics
            flags: list[str] = []
            notes: list[str] = []
            warnings = details.get("warnings", [])
            for w in warnings:
                if isinstance(w, str):
                    flags.append(w)

            # Extract max_depth_pixel tuple
            max_pixel = details.get("max_depth_pixel")
            target_pixel_out = details.get("target_pixel")

            metrics: dict[str, float | int | str | bool | None] = {
                **overlap_metrics,
                "max_depth_ppm": details.get("max_depth_ppm"),
                "target_depth_ppm": details.get("target_depth_ppm"),
                "concentration_ratio": details.get("concentration_ratio"),
                "distance_to_target_pixels": details.get("distance_to_target_pixels"),
                "n_cadences": details.get("n_cadences"),
            }

            if max_pixel and len(max_pixel) == 2:
                metrics["max_depth_pixel_row"] = int(max_pixel[0])
                metrics["max_depth_pixel_col"] = int(max_pixel[1])

            if target_pixel_out and len(target_pixel_out) == 2:
                metrics["target_pixel_row"] = int(target_pixel_out[0])
                metrics["target_pixel_col"] = int(target_pixel_out[1])

            flags.extend(overlap_flags)
            notes.extend(overlap_notes)

            return ok_result(
                self.id,
                self.name,
                metrics=metrics,
                confidence=result.confidence,
                flags=flags,
                notes=notes,
                raw=details,
            )

        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )


class ApertureDependenceCheck:
    """V10: Aperture dependence analysis.

    Measures transit depth at multiple aperture radii centered on the target.
    True planetary transits should show consistent depth regardless of aperture
    size, while contamination from nearby sources causes depth to vary.

    References:
        [1] Bryson et al. 2013, PASP 125, 889 - Section 3.3: Contamination assessment
        [2] Guerrero et al. 2021, ApJS 254, 39 - Section 3.4: TESS aperture analysis
        [3] Mullally et al. 2015, ApJS 217, 31 - Kepler vetting diagnostics
    """

    _id = "V10"
    _name = "Aperture Dependence"
    _tier = CheckTier.PIXEL
    _requirements = CheckRequirements(needs_tpf=True)
    _citations = [
        "Bryson et al. 2013, PASP 125, 889",
        "Guerrero et al. 2021, ApJS 254, 39",
        "Mullally et al. 2015, ApJS 217, 31",
    ]

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def tier(self) -> CheckTier:
        return self._tier

    @property
    def requirements(self) -> CheckRequirements:
        return self._requirements

    @property
    def citations(self) -> list[str]:
        return self._citations

    def run(self, inputs: CheckInputs, config: CheckConfig) -> CheckResult:
        """Execute aperture dependence check.

        Args:
            inputs: Check inputs containing TPF and candidate.
            config: Check configuration.

        Returns:
            CheckResult with aperture dependence metrics.
        """
        if inputs.tpf is None:
            return skipped_result(
                self.id,
                self.name,
                reason_flag="NO_TPF",
                notes=["TPF data not provided; cannot perform aperture analysis"],
            )

        try:
            time_arr, flux_arr = _extract_tpf_arrays(inputs.tpf)
            internal_candidate = _candidate_to_internal(inputs.candidate)

            # Extract config parameters
            extra = config.extra_params
            overlap_metrics, overlap_flags, overlap_notes = _lc_tpf_time_overlap_metrics(
                inputs, tpf_time=time_arr
            )
            radii_px = extra.get("radii_px")
            center_row_col = extra.get("center_row_col")

            result = check_aperture_dependence_with_tpf(
                tpf_data=flux_arr,
                time=time_arr,
                candidate=internal_candidate,
                aperture_radii_px=radii_px,
                center_row_col=center_row_col,
            )

            details = dict(result.details)
            status = details.get("status", "ok")

            # Handle error/insufficient data cases
            if status == "insufficient_data":
                return skipped_result(
                    self.id,
                    self.name,
                    reason_flag="INSUFFICIENT_DATA",
                    notes=[details.get("error", "Insufficient cadences for analysis")],
                    raw=details,
                )
            if status == "error":
                return error_result(
                    self.id,
                    self.name,
                    error=details.get("error", "Unknown error"),
                    raw=details,
                )

            # Build metrics
            flags: list[str] = []
            notes: list[str] = []
            detail_flags = details.get("flags", [])
            for f in detail_flags:
                if isinstance(f, str):
                    flags.append(f)

            # Flatten depths_by_aperture to individual metrics
            depths = details.get("depths_by_aperture_ppm", {})
            metrics: dict[str, float | int | str | bool | None] = {
                **overlap_metrics,
                "stability_metric": details.get("stability_metric"),
                "depth_variance_ppm2": details.get("depth_variance_ppm2"),
                "recommended_aperture_pixels": details.get("recommended_aperture_pixels"),
                "n_in_transit_cadences": details.get("n_in_transit_cadences"),
                "n_out_of_transit_cadences": details.get("n_out_of_transit_cadences"),
                "n_transit_epochs": details.get("n_transit_epochs"),
                "baseline_mode": details.get("baseline_mode"),
                "local_baseline_window_days": details.get("local_baseline_window_days"),
                "background_mode": details.get("background_mode"),
                "n_background_pixels": details.get("n_background_pixels"),
                "drift_fraction_recommended": details.get("drift_fraction_recommended"),
            }

            # Add individual aperture depths
            for aperture_str, depth in depths.items():
                key = f"depth_ppm_aperture_{aperture_str.replace('.', 'p')}"
                metrics[key] = _coerce_scalar(depth)

            flags.extend(overlap_flags)
            notes.extend(overlap_notes)

            return ok_result(
                self.id,
                self.name,
                metrics=metrics,
                confidence=result.confidence,
                flags=flags,
                notes=notes,
                raw=details,
            )

        except Exception as e:
            return error_result(
                self.id,
                self.name,
                error=type(e).__name__,
                notes=[str(e)],
            )


def register_pixel_checks(registry: CheckRegistry) -> None:
    """Register pixel-level checks V08-V10 with the registry.

    Args:
        registry: CheckRegistry to register checks with.
    """
    registry.register(CentroidShiftCheck())
    registry.register(DifferenceImageCheck())
    registry.register(ApertureDependenceCheck())


__all__ = [
    "CentroidShiftCheck",
    "DifferenceImageCheck",
    "ApertureDependenceCheck",
    "register_pixel_checks",
]
