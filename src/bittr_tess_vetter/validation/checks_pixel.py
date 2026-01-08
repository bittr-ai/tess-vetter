"""Pixel-level vetting checks for transit candidate validation (V08-V10).

This module implements pixel-level checks that require TPF (Target Pixel File) data:
- V08: Centroid Shift - Compare in-transit vs out-of-transit centroid positions
- V09: Pixel-Level LC - Analyze light curves from individual pixels
- V10: Aperture Dependence - Measure transit depth vs aperture size

These checks are critical for detecting background eclipsing binaries that
contaminate the photometric aperture.

V08 and V10 wrap existing implementations in bittr_tess_vetter.pixel:
- V08: Uses pixel.centroid.compute_centroid_shift()
- V10: Uses pixel.aperture.compute_aperture_dependence()

Thresholds (from spec):
- V08: FAIL if centroid_shift >= 1.0 pixel AND significance >= 5.0 sigma
       WARN if centroid_shift >= 0.5 pixel OR significance >= 3.0 sigma
- V10: FAIL if stability_metric < 0.5 (depth varies >20% across apertures)
       WARN if stability_metric < 0.7
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.domain.detection import VetterCheckResult
from bittr_tess_vetter.pixel.aperture import (
    ApertureDependenceResult,
    compute_aperture_dependence,
)
from bittr_tess_vetter.pixel.aperture import (
    TransitParams as ApertureTransitParams,
)
from bittr_tess_vetter.pixel.centroid import (
    CentroidResult,
    compute_centroid_shift,
)
from bittr_tess_vetter.pixel.centroid import (
    TransitParams as CentroidTransitParams,
)
from bittr_tess_vetter.validation.base import (
    CheckConfig,
    VetterCheck,
    get_in_transit_mask,
    get_out_of_transit_mask,
    make_result,
)

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.detection import TransitCandidate
    from bittr_tess_vetter.domain.lightcurve import LightCurveData
    from bittr_tess_vetter.domain.target import StellarParameters

logger = logging.getLogger(__name__)


# =============================================================================
# V08: Centroid Shift Check (wrapper around pixel.centroid)
# =============================================================================


class CentroidShiftCheck(VetterCheck):
    """V08: Detect centroid motion during transit.

    A significant centroid shift between in-transit and out-of-transit
    cadences indicates that the signal source is not the target star,
    but rather a background eclipsing binary or nearby contaminating source.

    TESS pixel scale: 21 arcsec/pixel

    Thresholds (from spec):
    - FAIL: shift >= 1.0 pixel AND significance >= 5.0 sigma
    - WARN: shift >= 0.5 pixel OR significance >= 3.0 sigma
    - PASS: otherwise

    This check wraps `bittr_tess_vetter.pixel.centroid.compute_centroid_shift()`.
    """

    id: ClassVar[str] = "V08"
    name: ClassVar[str] = "centroid_shift"

    def __init__(
        self,
        config: CheckConfig | None = None,
        tpf_data: NDArray[np.floating[Any]] | None = None,
        time: NDArray[np.floating[Any]] | None = None,
    ) -> None:
        """Initialize CentroidShiftCheck.

        Args:
            config: Check configuration (optional).
            tpf_data: TPF flux data with shape (time, rows, cols).
            time: Time array in BTJD.
        """
        super().__init__(config)
        self.tpf_data = tpf_data
        self.time = time

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Return default configuration for V08 check."""
        return CheckConfig(
            enabled=True,
            threshold=1.0,  # fail_shift_threshold in pixels
            additional={
                "fail_sigma_threshold": 5.0,
                "warn_shift_threshold": 0.5,
                "warn_sigma_threshold": 3.0,
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Execute the V08 centroid shift check.

        Args:
            candidate: Transit candidate with period, t0, duration_hours.
            lightcurve: Light curve data (not used, TPF required).
            stellar: Stellar parameters (not used).

        Returns:
            VetterCheckResult with centroid shift analysis.
        """
        # Check if TPF data is available
        if self.tpf_data is None or self.time is None:
            logger.warning("V08 (Centroid Shift) requires TPF data. Returning low-confidence pass.")
            return make_result(
                "V08",
                passed=True,
                confidence=0.1,
                details={
                    "note": "TPF data not available for centroid shift analysis",
                    "deferred": True,
                },
            )

        # Validate TPF data shape
        if self.tpf_data.ndim != 3:
            return make_result(
                "V08",
                passed=True,
                confidence=0.0,
                details={
                    "error": f"Invalid TPF shape: expected 3D, got {self.tpf_data.ndim}D",
                },
            )

        if len(self.time) != self.tpf_data.shape[0]:
            return make_result(
                "V08",
                passed=True,
                confidence=0.0,
                details={
                    "error": (
                        f"Time/TPF mismatch: {len(self.time)} times, "
                        f"{self.tpf_data.shape[0]} frames"
                    ),
                },
            )

        # Create TransitParams for centroid module
        transit_params = CentroidTransitParams(
            period=candidate.period,
            t0=candidate.t0,
            duration=candidate.duration_hours,  # hours
        )

        # Run centroid shift computation
        try:
            result: CentroidResult = compute_centroid_shift(
                tpf_data=self.tpf_data,
                time=self.time,
                transit_params=transit_params,
                significance_method="analytic",
            )
        except ValueError as e:
            return make_result(
                "V08",
                passed=True,
                confidence=0.1,
                details={
                    "reason": "computation_error",
                    "message": str(e),
                },
            )

        # Handle NaN results
        if np.isnan(result.centroid_shift_pixels) or np.isnan(result.significance_sigma):
            return make_result(
                "V08",
                passed=True,
                confidence=0.2,
                details={
                    "reason": "insufficient_data",
                    "message": "Could not compute centroid - insufficient in/out transit data",
                    "n_in_transit": result.n_in_transit_cadences,
                    "n_out_transit": result.n_out_transit_cadences,
                },
            )

        # Get thresholds from config
        fail_shift = self.config.threshold if self.config.threshold is not None else 1.0
        fail_sigma = self.config.additional.get("fail_sigma_threshold", 5.0)
        warn_shift = self.config.additional.get("warn_shift_threshold", 0.5)
        warn_sigma = self.config.additional.get("warn_sigma_threshold", 3.0)

        # Evaluate pass/fail based on thresholds
        shift = result.centroid_shift_pixels
        sigma = result.significance_sigma

        # FAIL condition: both shift AND significance exceed thresholds
        is_fail = (shift >= fail_shift) and (sigma >= fail_sigma)

        # WARN condition: either shift OR significance exceeds warning threshold
        is_warn = (shift >= warn_shift) or (sigma >= warn_sigma)

        if is_fail:
            passed = False
            confidence = 0.95  # High confidence in failure
        elif is_warn:
            passed = True  # Pass but with warning
            confidence = 0.6  # Lower confidence
        else:
            passed = True
            # Confidence based on number of cadences
            base_confidence = 0.85
            if result.n_in_transit_cadences >= 20 and result.n_out_transit_cadences >= 100:
                base_confidence = 0.95
            elif result.n_in_transit_cadences < 5 or result.n_out_transit_cadences < 20:
                base_confidence = 0.6
            confidence = base_confidence

        # Build details dict
        details: dict[str, Any] = {
            "centroid_shift_pixels": round(shift, 4),
            "significance_sigma": round(sigma, 2),
            "in_transit_centroid": (
                round(result.in_transit_centroid[0], 3),
                round(result.in_transit_centroid[1], 3),
            ),
            "out_of_transit_centroid": (
                round(result.out_of_transit_centroid[0], 3),
                round(result.out_of_transit_centroid[1], 3),
            ),
            "n_in_transit_cadences": result.n_in_transit_cadences,
            "n_out_transit_cadences": result.n_out_transit_cadences,
            "shift_arcsec": round(shift * 21.0, 2),  # TESS pixel scale
            "thresholds": {
                "fail_shift": fail_shift,
                "fail_sigma": fail_sigma,
                "warn_shift": warn_shift,
                "warn_sigma": warn_sigma,
            },
        }

        if is_warn and not is_fail:
            details["warning"] = "Marginal centroid shift detected - recommend follow-up"

        return make_result(
            "V08",
            passed=passed,
            confidence=round(confidence, 3),
            details=details,
        )


def check_centroid_shift_with_tpf(
    tpf_data: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    period: float,
    t0: float,
    duration_hours: float,
    fail_shift_threshold: float = 1.0,
    fail_sigma_threshold: float = 5.0,
    warn_shift_threshold: float = 0.5,
    warn_sigma_threshold: float = 3.0,
) -> VetterCheckResult:
    """V08: Centroid shift check (functional interface).

    Convenience function for running V08 without creating a check instance.

    Args:
        tpf_data: TPF flux data with shape (time, rows, cols).
        time: Time array in BTJD.
        period: Orbital period in days.
        t0: Reference transit epoch in BTJD.
        duration_hours: Transit duration in hours.
        fail_shift_threshold: Shift threshold for failure (pixels).
        fail_sigma_threshold: Significance threshold for failure (sigma).
        warn_shift_threshold: Shift threshold for warning (pixels).
        warn_sigma_threshold: Significance threshold for warning (sigma).

    Returns:
        VetterCheckResult with centroid shift analysis.
    """
    # Validate inputs
    if tpf_data.ndim != 3:
        return make_result(
            "V08",
            passed=True,
            confidence=0.0,
            details={"error": f"Expected 3D TPF data, got shape {tpf_data.shape}"},
        )

    if len(time) != tpf_data.shape[0]:
        return make_result(
            "V08",
            passed=True,
            confidence=0.0,
            details={"error": f"Time length {len(time)} != TPF cadences {tpf_data.shape[0]}"},
        )

    # Create TransitParams for centroid module
    transit_params = CentroidTransitParams(
        period=period,
        t0=t0,
        duration=duration_hours,  # hours
    )

    # Run centroid shift computation
    try:
        result: CentroidResult = compute_centroid_shift(
            tpf_data=tpf_data,
            time=time,
            transit_params=transit_params,
            significance_method="analytic",
        )
    except ValueError as e:
        return make_result(
            "V08",
            passed=True,
            confidence=0.1,
            details={"reason": "computation_error", "message": str(e)},
        )

    # Handle NaN results
    if np.isnan(result.centroid_shift_pixels) or np.isnan(result.significance_sigma):
        return make_result(
            "V08",
            passed=True,
            confidence=0.2,
            details={
                "reason": "insufficient_data",
                "message": "Could not compute centroid",
                "n_in_transit": result.n_in_transit_cadences,
                "n_out_transit": result.n_out_transit_cadences,
            },
        )

    shift = result.centroid_shift_pixels
    sigma = result.significance_sigma

    # FAIL condition: both shift AND significance exceed thresholds
    is_fail = (shift >= fail_shift_threshold) and (sigma >= fail_sigma_threshold)
    is_warn = (shift >= warn_shift_threshold) or (sigma >= warn_sigma_threshold)

    if is_fail:
        passed = False
        confidence = 0.95
    elif is_warn:
        passed = True
        confidence = 0.6
    else:
        passed = True
        confidence = 0.85 if result.n_in_transit_cadences >= 10 else 0.6

    details: dict[str, Any] = {
        "centroid_shift_pixels": round(shift, 4),
        "significance_sigma": round(sigma, 2),
        "in_transit_centroid": (
            round(result.in_transit_centroid[0], 3),
            round(result.in_transit_centroid[1], 3),
        ),
        "out_of_transit_centroid": (
            round(result.out_of_transit_centroid[0], 3),
            round(result.out_of_transit_centroid[1], 3),
        ),
        "n_in_transit_cadences": result.n_in_transit_cadences,
        "n_out_transit_cadences": result.n_out_transit_cadences,
        "shift_arcsec": round(shift * 21.0, 2),
    }

    if is_warn and not is_fail:
        details["warning"] = "Marginal centroid shift detected"

    return make_result("V08", passed=passed, confidence=round(confidence, 3), details=details)


# =============================================================================
# V09: Pixel-Level Light Curve Check
# =============================================================================


@dataclass(frozen=True)
class PixelLevelLCResult:
    """Result of pixel-level light curve analysis.

    Attributes:
        passed: Whether the check passed (transit appears on target).
        confidence: Confidence in the result (0-1).
        depth_map_ppm: 2D array of transit depths per pixel in ppm.
        max_depth_pixel: (row, col) of pixel with maximum transit depth.
        max_depth_ppm: Maximum transit depth in ppm.
        target_depth_ppm: Transit depth at target pixel in ppm.
        concentration_ratio: target_depth / max_depth (1.0 = on target).
        transit_on_target: True if max depth pixel is near target location.
    """

    passed: bool
    confidence: float
    depth_map_ppm: NDArray[np.floating[Any]]
    max_depth_pixel: tuple[int, int]
    max_depth_ppm: float
    target_depth_ppm: float
    concentration_ratio: float
    transit_on_target: bool


def compute_pixel_level_depths(
    tpf_data: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    period: float,
    t0: float,
    duration_hours: float,
) -> NDArray[np.floating[Any]]:
    """Compute transit depth for each pixel in the TPF.

    For each pixel, extracts its light curve and measures the transit depth
    by comparing in-transit vs out-of-transit flux.

    Args:
        tpf_data: Target pixel file flux data with shape (time, rows, cols).
        time: Time array in BTJD.
        period: Orbital period in days.
        t0: Reference transit epoch in BTJD.
        duration_hours: Transit duration in hours.

    Returns:
        2D array of transit depths in ppm with shape (rows, cols).
        Positive values indicate flux decrease (dip).
    """
    n_time, n_rows, n_cols = tpf_data.shape

    # Get transit masks
    in_transit_mask = get_in_transit_mask(time, period, t0, duration_hours)
    out_of_transit_mask = get_out_of_transit_mask(
        time, period, t0, duration_hours, buffer_factor=2.0
    )

    depth_map = np.zeros((n_rows, n_cols), dtype=np.float64)

    for i in range(n_rows):
        for j in range(n_cols):
            # Extract single-pixel light curve
            pixel_flux = tpf_data[:, i, j]

            # Get valid (finite) data
            valid_in = in_transit_mask & np.isfinite(pixel_flux)
            valid_out = out_of_transit_mask & np.isfinite(pixel_flux)

            if np.sum(valid_in) < 3 or np.sum(valid_out) < 3:
                # Insufficient data for this pixel
                depth_map[i, j] = np.nan
                continue

            in_flux = pixel_flux[valid_in]
            out_flux = pixel_flux[valid_out]

            # Compute depth: (out - in) / out
            # Positive depth = flux decrease during transit
            out_median = np.nanmedian(out_flux)
            in_median = np.nanmedian(in_flux)

            if out_median <= 0 or not np.isfinite(out_median):
                depth_map[i, j] = np.nan
                continue

            depth = (out_median - in_median) / out_median
            depth_map[i, j] = depth * 1e6  # Convert to ppm

    return depth_map


def find_target_pixel(
    depth_map: NDArray[np.floating[Any]],
    target_pixel: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Find the expected target pixel location.

    If target_pixel is provided, returns it. Otherwise, assumes target is
    at the center of the TPF (standard TESS convention).

    Args:
        depth_map: 2D array of depths (used for shape).
        target_pixel: Optional explicit target pixel (row, col).

    Returns:
        Target pixel location as (row, col).
    """
    if target_pixel is not None:
        return target_pixel

    # Default: center of TPF
    n_rows, n_cols = depth_map.shape
    return (n_rows // 2, n_cols // 2)


def compute_pixel_level_lc_check(
    tpf_data: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    period: float,
    t0: float,
    duration_hours: float,
    target_pixel: tuple[int, int] | None = None,
    concentration_threshold: float = 0.7,
    proximity_radius: int = 1,
) -> PixelLevelLCResult:
    """Analyze pixel-level light curves to locate transit source.

    Extracts light curves from individual pixels, measures transit depth
    in each, and determines if the signal is on-target.

    Args:
        tpf_data: TPF flux data with shape (time, rows, cols).
        time: Time array in BTJD.
        period: Orbital period in days.
        t0: Reference transit epoch in BTJD.
        duration_hours: Transit duration in hours.
        target_pixel: Expected target location (row, col). If None, uses center.
        concentration_threshold: Minimum concentration ratio to pass (default 0.7).
        proximity_radius: Max distance from target for "on-target" (default 1 pixel).

    Returns:
        PixelLevelLCResult with depth map and pass/fail determination.
    """
    # Compute depth map
    depth_map = compute_pixel_level_depths(tpf_data, time, period, t0, duration_hours)

    # Find target pixel (default: center of TPF)
    target_pix = find_target_pixel(depth_map, target_pixel)

    # Find pixel with maximum depth (ignoring NaN)
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)

    if not np.any(valid_mask):
        # No valid transit signal found in any pixel
        return PixelLevelLCResult(
            passed=False,
            confidence=0.3,
            depth_map_ppm=depth_map,
            max_depth_pixel=(0, 0),
            max_depth_ppm=0.0,
            target_depth_ppm=0.0,
            concentration_ratio=0.0,
            transit_on_target=False,
        )

    # Find max depth pixel
    masked_depths = np.where(valid_mask, depth_map, -np.inf)
    max_idx = np.unravel_index(np.argmax(masked_depths), depth_map.shape)
    max_depth_pixel = (int(max_idx[0]), int(max_idx[1]))
    max_depth_ppm = float(depth_map[max_depth_pixel])

    # Get target pixel depth
    target_row, target_col = target_pix
    if (
        0 <= target_row < depth_map.shape[0]
        and 0 <= target_col < depth_map.shape[1]
        and np.isfinite(depth_map[target_row, target_col])
    ):
        target_depth_ppm = float(depth_map[target_row, target_col])
    else:
        target_depth_ppm = 0.0

    # Compute concentration ratio
    concentration_ratio = target_depth_ppm / max_depth_ppm if max_depth_ppm > 0 else 0.0

    # Check if max depth pixel is near target
    distance = np.sqrt(
        (max_depth_pixel[0] - target_row) ** 2 + (max_depth_pixel[1] - target_col) ** 2
    )
    transit_on_target = distance <= proximity_radius

    # Determine pass/fail based on spec thresholds:
    # PASS: concentration_ratio >= 0.7 AND transit_on_target == True
    # WARN: concentration_ratio >= 0.5 AND transit_on_target == True
    # FAIL: transit_on_target == False OR concentration_ratio < 0.5
    passed = concentration_ratio >= concentration_threshold and transit_on_target

    # Confidence based on depth significance and data quality
    n_valid_pixels = int(np.sum(valid_mask))
    total_pixels = depth_map.size

    # Base confidence from valid pixel coverage
    coverage_factor = n_valid_pixels / total_pixels if total_pixels > 0 else 0.0

    # Higher confidence if depth is well-concentrated
    if concentration_ratio >= 0.8:
        concentration_factor = 0.95
    elif concentration_ratio >= 0.6:
        concentration_factor = 0.75
    else:
        concentration_factor = 0.5

    confidence = min(0.95, coverage_factor * 0.3 + concentration_factor * 0.7)

    # Lower confidence if transit is off-target
    if not transit_on_target:
        confidence *= 0.8

    return PixelLevelLCResult(
        passed=passed,
        confidence=round(confidence, 3),
        depth_map_ppm=depth_map,
        max_depth_pixel=max_depth_pixel,
        max_depth_ppm=round(max_depth_ppm, 2),
        target_depth_ppm=round(target_depth_ppm, 2),
        concentration_ratio=round(concentration_ratio, 3),
        transit_on_target=transit_on_target,
    )


class PixelLevelLCCheck(VetterCheck):
    """V09: Pixel-Level Light Curve vetting check.

    Analyzes light curves from individual pixels to locate the source of
    the transit signal. If the maximum depth pixel is not at the expected
    target location, the signal may be from a background source.

    Pass Criteria:
        - concentration_ratio >= 0.7 (target depth / max depth)
        - transit_on_target == True (max depth within 1 pixel of target)

    Fail Criteria:
        - transit_on_target == False (signal off-target)
        - concentration_ratio < 0.5 (signal not concentrated on target)

    Requires:
        - TPF data (tpf_data parameter)
        - Time array
        - Transit ephemeris (period, t0, duration)
    """

    id: ClassVar[str] = "V09"
    name: ClassVar[str] = "pixel_level_lc"

    def __init__(
        self,
        config: CheckConfig | None = None,
        tpf_data: NDArray[np.floating[Any]] | None = None,
        time: NDArray[np.floating[Any]] | None = None,
        target_pixel: tuple[int, int] | None = None,
    ) -> None:
        """Initialize PixelLevelLCCheck.

        Args:
            config: Check configuration (optional).
            tpf_data: TPF flux data with shape (time, rows, cols).
            time: Time array in BTJD.
            target_pixel: Expected target pixel location (row, col).
        """
        super().__init__(config)
        self.tpf_data = tpf_data
        self.time = time
        self.target_pixel = target_pixel

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Return default configuration for V09 check."""
        return CheckConfig(
            enabled=True,
            threshold=0.7,  # concentration_threshold
            additional={
                "proximity_radius": 1,  # pixels
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Execute the V09 pixel-level light curve check.

        Args:
            candidate: Transit candidate with period, t0, duration.
            lightcurve: Light curve data (not used, TPF required).
            stellar: Stellar parameters (not used).

        Returns:
            VetterCheckResult with pass/fail and depth map details.
        """
        # Check if TPF data is available
        if self.tpf_data is None or self.time is None:
            logger.warning("V09 (Pixel-Level LC) requires TPF data. Returning low-confidence pass.")
            return make_result(
                "V09",
                passed=True,
                confidence=0.1,
                details={
                    "note": "TPF data not available for pixel-level analysis",
                    "deferred": True,
                },
            )

        # Validate TPF data shape
        if self.tpf_data.ndim != 3:
            return make_result(
                "V09",
                passed=False,
                confidence=0.3,
                details={
                    "error": f"Invalid TPF shape: expected 3D, got {self.tpf_data.ndim}D",
                },
            )

        if len(self.time) != self.tpf_data.shape[0]:
            return make_result(
                "V09",
                passed=False,
                confidence=0.3,
                details={
                    "error": (
                        f"Time/TPF mismatch: {len(self.time)} times, "
                        f"{self.tpf_data.shape[0]} frames"
                    ),
                },
            )

        # Run pixel-level analysis
        concentration_threshold = (
            self.config.threshold if self.config.threshold is not None else 0.7
        )
        proximity_radius = self.config.additional.get("proximity_radius", 1)

        result = compute_pixel_level_lc_check(
            tpf_data=self.tpf_data,
            time=self.time,
            period=candidate.period,
            t0=candidate.t0,
            duration_hours=candidate.duration_hours,
            target_pixel=self.target_pixel,
            concentration_threshold=concentration_threshold,
            proximity_radius=proximity_radius,
        )

        # Build details dict (without large depth_map)
        details: dict[str, Any] = {
            "max_depth_pixel": result.max_depth_pixel,
            "max_depth_ppm": result.max_depth_ppm,
            "target_depth_ppm": result.target_depth_ppm,
            "concentration_ratio": result.concentration_ratio,
            "transit_on_target": result.transit_on_target,
            "tpf_shape": list(self.tpf_data.shape),
            "n_cadences": len(self.time),
        }

        # Add target pixel info
        if self.target_pixel is not None:
            details["target_pixel"] = self.target_pixel
        else:
            details["target_pixel"] = find_target_pixel(result.depth_map_ppm, None)
            details["target_pixel_source"] = "center"

        return make_result(
            "V09",
            passed=result.passed,
            confidence=result.confidence,
            details=details,
        )


def check_pixel_level_lc_with_tpf(
    tpf_data: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    period: float,
    t0: float,
    duration_hours: float,
    target_pixel: tuple[int, int] | None = None,
    concentration_threshold: float = 0.7,
    proximity_radius: int = 1,
) -> VetterCheckResult:
    """V09: Pixel-level light curve check (functional interface).

    Convenience function for running V09 without creating a check instance.

    Args:
        tpf_data: TPF flux data with shape (time, rows, cols).
        time: Time array in BTJD.
        period: Orbital period in days.
        t0: Reference transit epoch in BTJD.
        duration_hours: Transit duration in hours.
        target_pixel: Expected target pixel (row, col). Default: center.
        concentration_threshold: Min ratio for pass (default 0.7).
        proximity_radius: Max pixel distance for on-target (default 1).

    Returns:
        VetterCheckResult with pass/fail and depth analysis details.
    """
    result = compute_pixel_level_lc_check(
        tpf_data=tpf_data,
        time=time,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        target_pixel=target_pixel,
        concentration_threshold=concentration_threshold,
        proximity_radius=proximity_radius,
    )

    # Determine target pixel for details
    actual_target = target_pixel
    target_source = "provided"
    if actual_target is None:
        actual_target = find_target_pixel(result.depth_map_ppm, None)
        target_source = "center"

    details: dict[str, Any] = {
        "max_depth_pixel": result.max_depth_pixel,
        "max_depth_ppm": result.max_depth_ppm,
        "target_depth_ppm": result.target_depth_ppm,
        "concentration_ratio": result.concentration_ratio,
        "transit_on_target": result.transit_on_target,
        "target_pixel": actual_target,
        "target_pixel_source": target_source,
        "tpf_shape": list(tpf_data.shape),
        "n_cadences": len(time),
    }

    return make_result(
        "V09",
        passed=result.passed,
        confidence=result.confidence,
        details=details,
    )


# =============================================================================
# V10: Aperture Dependence Check (wrapper around pixel.aperture)
# =============================================================================


class ApertureDependenceCheck(VetterCheck):
    """V10: Check if transit depth varies with aperture size.

    If the transit depth changes significantly with aperture size,
    the signal may be from contaminating flux (background EB or
    nearby variable star) rather than the target.

    Interpretation:
    - Stable depth across apertures = on-target signal (PASS)
    - Depth increases with aperture = contamination dilution (FAIL)
    - Depth decreases with aperture = background source (FAIL)

    Thresholds (from spec, stability_metric 0-1 scale):
    - FAIL: stability_metric < 0.5
    - WARN: stability_metric < 0.7
    - PASS: stability_metric >= 0.7

    This check wraps `bittr_tess_vetter.pixel.aperture.compute_aperture_dependence()`.
    """

    id: ClassVar[str] = "V10"
    name: ClassVar[str] = "aperture_dependence"

    def __init__(
        self,
        config: CheckConfig | None = None,
        tpf_data: NDArray[np.floating[Any]] | None = None,
        time: NDArray[np.floating[Any]] | None = None,
        aperture_radii: list[float] | None = None,
    ) -> None:
        """Initialize ApertureDependenceCheck.

        Args:
            config: Check configuration (optional).
            tpf_data: TPF flux data with shape (time, rows, cols).
            time: Time array in BTJD.
            aperture_radii: List of aperture radii to test (pixels).
        """
        super().__init__(config)
        self.tpf_data = tpf_data
        self.time = time
        # Default starts at 1.5px to avoid fragile 1.0px behavior on small/synthetic stamps.
        self.aperture_radii = aperture_radii or [1.5, 2.0, 2.5, 3.0, 3.5]

    @classmethod
    def _default_config(cls) -> CheckConfig:
        """Return default configuration for V10 check."""
        return CheckConfig(
            enabled=True,
            threshold=0.5,  # fail_stability_threshold
            additional={
                "warn_stability_threshold": 0.7,
            },
        )

    def run(
        self,
        candidate: TransitCandidate,
        lightcurve: LightCurveData | None = None,
        stellar: StellarParameters | None = None,
    ) -> VetterCheckResult:
        """Execute the V10 aperture dependence check.

        Args:
            candidate: Transit candidate with period, t0, duration_hours, depth.
            lightcurve: Light curve data (not used, TPF required).
            stellar: Stellar parameters (not used).

        Returns:
            VetterCheckResult with aperture dependence analysis.
        """
        # Check if TPF data is available
        if self.tpf_data is None or self.time is None:
            logger.warning(
                "V10 (Aperture Dependence) requires TPF data. Returning low-confidence pass."
            )
            return make_result(
                "V10",
                passed=True,
                confidence=0.1,
                details={
                    "note": "TPF data not available for aperture dependence analysis",
                    "deferred": True,
                },
            )

        # Validate TPF data shape
        if self.tpf_data.ndim != 3:
            return make_result(
                "V10",
                passed=True,
                confidence=0.0,
                details={
                    "error": f"Invalid TPF shape: expected 3D, got {self.tpf_data.ndim}D",
                },
            )

        if len(self.time) != self.tpf_data.shape[0]:
            return make_result(
                "V10",
                passed=True,
                confidence=0.0,
                details={
                    "error": (
                        f"Time/TPF mismatch: {len(self.time)} times, "
                        f"{self.tpf_data.shape[0]} frames"
                    ),
                },
            )

        # Create TransitParams for aperture module
        # Note: aperture module expects duration in days, not hours
        duration_days = candidate.duration_hours / 24.0
        transit_params = ApertureTransitParams(
            period=candidate.period,
            t0=candidate.t0,
            duration=duration_days,
            depth=candidate.depth,
        )

        # Run aperture dependence computation
        try:
            result: ApertureDependenceResult = compute_aperture_dependence(
                tpf_data=self.tpf_data,
                time=self.time,
                transit_params=transit_params,
                aperture_radii=self.aperture_radii,
            )
        except ValueError as e:
            return make_result(
                "V10",
                passed=True,
                confidence=0.1,
                details={
                    "reason": "computation_error",
                    "message": str(e),
                },
            )

        # Get thresholds from config
        fail_stability = self.config.threshold if self.config.threshold is not None else 0.5
        warn_stability = self.config.additional.get("warn_stability_threshold", 0.7)

        # Evaluate pass/fail based on stability metric
        stability = result.stability_metric

        if stability < fail_stability:
            passed = False
            confidence = 0.90  # High confidence in failure
        elif stability < warn_stability:
            passed = True  # Pass but with warning
            confidence = 0.65  # Moderate confidence
        else:
            passed = True
            # High confidence when stability is good
            confidence = min(0.95, 0.7 + stability * 0.25)

        # Round depths for display
        depths_rounded = {
            radius: round(depth, 1) for radius, depth in result.depths_by_aperture.items()
        }

        # Calculate depth variation metrics
        depth_values = list(result.depths_by_aperture.values())
        depth_range_ppm = max(depth_values) - min(depth_values) if depth_values else 0.0
        mean_depth_ppm = float(np.mean(depth_values)) if depth_values else 0.0
        relative_variation = depth_range_ppm / mean_depth_ppm if mean_depth_ppm > 0 else 0.0

        # Build details dict
        details: dict[str, Any] = {
            "stability_metric": round(stability, 3),
            "depths_by_aperture_ppm": depths_rounded,
            "depth_variance_ppm2": round(result.depth_variance, 1),
            "recommended_aperture_pixels": result.recommended_aperture,
            "depth_range_ppm": round(depth_range_ppm, 1),
            "mean_depth_ppm": round(mean_depth_ppm, 1),
            "relative_variation": round(relative_variation, 3),
            "n_apertures_tested": len(result.depths_by_aperture),
            "n_in_transit_cadences": int(getattr(result, "n_in_transit_cadences", 0)),
            "n_out_of_transit_cadences": int(getattr(result, "n_out_of_transit_cadences", 0)),
            "n_transit_epochs": int(getattr(result, "n_transit_epochs", 0)),
            "baseline_mode": str(getattr(result, "baseline_mode", "unknown")),
            "local_baseline_window_days": float(getattr(result, "local_baseline_window_days", float("nan"))),
            "background_mode": str(getattr(result, "background_mode", "unknown")),
            "background_annulus_radii_pixels": list(getattr(result, "background_annulus_radii_pixels", []) or []),
            "n_background_pixels": int(getattr(result, "n_background_pixels", 0)),
            "drift_fraction_recommended": getattr(result, "drift_fraction_recommended", None),
            "flags": list(getattr(result, "flags", []) or []),
            "notes": getattr(result, "notes", {}),
            "thresholds": {
                "fail_stability": fail_stability,
                "warn_stability": warn_stability,
            },
        }

        if stability < warn_stability and passed:
            details["warning"] = "Marginal depth variation across apertures - recommend follow-up"

        if not passed and len(depth_values) >= 2:
            # Add interpretation for failed check
            radii = list(result.depths_by_aperture.keys())
            depths = list(result.depths_by_aperture.values())
            if len(radii) >= 2:
                correlation = float(np.corrcoef(radii, depths)[0, 1])
                if correlation > 0.5:
                    details["interpretation"] = (
                        "Depth increases with aperture - suggests contamination dilution"
                    )
                elif correlation < -0.5:
                    details["interpretation"] = (
                        "Depth decreases with aperture - suggests background source"
                    )
                else:
                    details["interpretation"] = (
                        "Irregular depth variation - possible complex contamination"
                    )
                details["depth_aperture_correlation"] = round(correlation, 3)

        return make_result(
            "V10",
            passed=passed,
            confidence=round(confidence, 3),
            details=details,
        )


def check_aperture_dependence_with_tpf(
    tpf_data: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    period: float,
    t0: float,
    duration_hours: float,
    depth: float = 0.01,
    aperture_radii: list[float] | None = None,
    fail_stability_threshold: float = 0.5,
    warn_stability_threshold: float = 0.7,
) -> VetterCheckResult:
    """V10: Aperture dependence check (functional interface).

    Convenience function for running V10 without creating a check instance.

    Args:
        tpf_data: TPF flux data with shape (time, rows, cols).
        time: Time array in BTJD.
        period: Orbital period in days.
        t0: Reference transit epoch in BTJD.
        duration_hours: Transit duration in hours.
        depth: Expected transit depth (fraction).
        aperture_radii: List of aperture radii to test (pixels).
        fail_stability_threshold: Below this stability, check fails.
        warn_stability_threshold: Below this (but above fail), check warns.

    Returns:
        VetterCheckResult with aperture dependence analysis.
    """
    # Validate inputs
    if tpf_data.ndim != 3:
        return make_result(
            "V10",
            passed=True,
            confidence=0.0,
            details={"error": f"Expected 3D TPF data, got shape {tpf_data.shape}"},
        )

    if len(time) != tpf_data.shape[0]:
        return make_result(
            "V10",
            passed=True,
            confidence=0.0,
            details={"error": f"Time length {len(time)} != TPF cadences {tpf_data.shape[0]}"},
        )

    if aperture_radii is None:
        aperture_radii = [1.5, 2.0, 2.5, 3.0, 3.5]

    # Create TransitParams for aperture module
    duration_days = duration_hours / 24.0
    transit_params = ApertureTransitParams(
        period=period,
        t0=t0,
        duration=duration_days,
        depth=depth,
    )

    # Run aperture dependence computation
    try:
        result: ApertureDependenceResult = compute_aperture_dependence(
            tpf_data=tpf_data,
            time=time,
            transit_params=transit_params,
            aperture_radii=aperture_radii,
        )
    except ValueError as e:
        return make_result(
            "V10",
            passed=True,
            confidence=0.1,
            details={"reason": "computation_error", "message": str(e)},
        )

    # Evaluate pass/fail
    stability = result.stability_metric

    if stability < fail_stability_threshold:
        passed = False
        confidence = 0.90
    elif stability < warn_stability_threshold:
        passed = True
        confidence = 0.65
    else:
        passed = True
        confidence = min(0.95, 0.7 + stability * 0.25)

    # Round depths for display
    depths_rounded = {radius: round(d, 1) for radius, d in result.depths_by_aperture.items()}

    depth_values = list(result.depths_by_aperture.values())
    depth_range_ppm = max(depth_values) - min(depth_values) if depth_values else 0.0
    mean_depth_ppm = float(np.mean(depth_values)) if depth_values else 0.0

    details: dict[str, Any] = {
        "stability_metric": round(stability, 3),
        "depths_by_aperture_ppm": depths_rounded,
        "depth_variance_ppm2": round(result.depth_variance, 1),
        "recommended_aperture_pixels": result.recommended_aperture,
        "depth_range_ppm": round(depth_range_ppm, 1),
        "mean_depth_ppm": round(mean_depth_ppm, 1),
        "n_apertures_tested": len(result.depths_by_aperture),
    }

    if stability < warn_stability_threshold and passed:
        details["warning"] = "Marginal depth variation across apertures"

    return make_result("V10", passed=passed, confidence=round(confidence, 3), details=details)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # V08: Centroid Shift
    "CentroidShiftCheck",
    "check_centroid_shift_with_tpf",
    # V09: Pixel-Level LC
    "PixelLevelLCCheck",
    "PixelLevelLCResult",
    "compute_pixel_level_lc_check",
    "compute_pixel_level_depths",
    "check_pixel_level_lc_with_tpf",
    # V10: Aperture Dependence
    "ApertureDependenceCheck",
    "check_aperture_dependence_with_tpf",
]
