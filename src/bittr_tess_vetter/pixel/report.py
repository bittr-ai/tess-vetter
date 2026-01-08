"""Pixel vetting report generator.

This module provides the PixelVetReport model and generate_pixel_vet_report
function for combining pixel-level analysis results into a comprehensive
vetting report with pass/fail determination.

The pixel vetting report combines results from:
- Centroid shift analysis (CentroidResult)
- Difference image localization (DifferenceImageResult)
- Aperture dependence analysis (ApertureDependenceResult)

And applies versioned thresholds to determine overall PIXEL_PASS status.

Threshold Versions:
    v1: Initial threshold set with conservative values
        - max_centroid_shift_pixels: 1.0
        - min_centroid_significance_sigma: 3.0
        - min_localization_score: 0.7
        - min_aperture_stability: 0.8
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bittr_tess_vetter.pixel.aperture import ApertureDependenceResult
from bittr_tess_vetter.pixel.centroid import CentroidResult
from bittr_tess_vetter.pixel.difference import DifferenceImageResult

# Versioned threshold configurations for pixel vetting
# Each version defines the thresholds used to determine pass/fail
THRESHOLD_VERSIONS: dict[str, dict[str, float]] = {
    "v1": {
        "max_centroid_shift_pixels": 1.0,
        "min_centroid_significance_sigma": 3.0,
        "min_localization_score": 0.7,
        "min_aperture_stability": 0.8,
    }
}


class PixelVetReport(BaseModel):
    """Combined pixel vetting report with pass/fail determination.

    The PixelVetReport aggregates results from all pixel-level analyses
    and applies versioned thresholds to determine whether the candidate
    passes pixel vetting.

    Attributes:
        pixel_pass: Overall pass/fail determination for pixel vetting.
            True if all individual tests pass.
        centroid_result: Results from centroid shift analysis.
        diff_image_result: Results from difference image localization.
        aperture_result: Results from aperture dependence analysis.
        individual_tests: Dictionary mapping test names to pass/fail boolean.
            Keys include:
            - "centroid_shift": True if shift < max threshold
            - "centroid_significance": True if significance < min threshold
              (i.e., shift is NOT statistically significant)
            - "localization": True if score >= min threshold
            - "aperture_stability": True if stability >= min threshold
        failure_reasons: List of human-readable reasons for any failures.
            Empty if pixel_pass is True.
        plots: Optional list of plot descriptors generated externally.
    """

    # Allow arbitrary types to support frozen dataclasses
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    pixel_pass: bool = Field(description="Overall pass/fail for pixel vetting")
    centroid_result: Any = Field(
        description="Results from centroid shift analysis (CentroidResult)"
    )
    diff_image_result: Any = Field(
        description="Results from difference image localization (DifferenceImageResult)"
    )
    aperture_result: Any = Field(
        description="Results from aperture dependence analysis (ApertureDependenceResult)"
    )
    individual_tests: dict[str, bool] = Field(description="Mapping of test name to pass/fail")
    failure_reasons: list[str] = Field(
        default_factory=list, description="Human-readable reasons for failures"
    )
    plots: list[dict[str, Any]] = Field(default_factory=list, description="Optional plot metadata")


def _evaluate_centroid_shift(
    centroid_result: CentroidResult,
    thresholds: dict[str, float],
) -> tuple[bool, str | None]:
    """Evaluate centroid shift test.

    The centroid shift test passes if the shift is less than the maximum
    allowed threshold (indicating the transit is on-target).

    Args:
        centroid_result: Results from centroid analysis.
        thresholds: Threshold configuration dict.

    Returns:
        Tuple of (passed, failure_reason). failure_reason is None if passed.
    """
    max_shift = thresholds["max_centroid_shift_pixels"]
    if centroid_result.centroid_shift_pixels >= max_shift:
        return (
            False,
            f"Centroid shift of {centroid_result.centroid_shift_pixels:.3f} pixels "
            f"exceeds threshold of {max_shift:.1f} pixels",
        )
    return (True, None)


def _evaluate_centroid_significance(
    centroid_result: CentroidResult,
    thresholds: dict[str, float],
) -> tuple[bool, str | None]:
    """Evaluate centroid significance test.

    The centroid significance test passes if the significance is BELOW
    the threshold, meaning the measured shift is NOT statistically
    significant (i.e., consistent with on-target transit).

    A high significance indicates a statistically significant shift,
    which would suggest an off-target or blended signal.

    Args:
        centroid_result: Results from centroid analysis.
        thresholds: Threshold configuration dict.

    Returns:
        Tuple of (passed, failure_reason). failure_reason is None if passed.
    """
    min_significance = thresholds["min_centroid_significance_sigma"]
    if centroid_result.significance_sigma >= min_significance:
        return (
            False,
            f"Centroid shift significance of {centroid_result.significance_sigma:.2f} sigma "
            f"exceeds threshold of {min_significance:.1f} sigma "
            f"(shift is statistically significant, suggesting off-target source)",
        )
    return (True, None)


def _evaluate_localization(
    diff_image_result: DifferenceImageResult,
    thresholds: dict[str, float],
) -> tuple[bool, str | None]:
    """Evaluate difference image localization test.

    The localization test passes if the localization score meets or
    exceeds the minimum threshold, indicating the transit signal
    localizes to the target position.

    Args:
        diff_image_result: Results from difference image analysis.
        thresholds: Threshold configuration dict.

    Returns:
        Tuple of (passed, failure_reason). failure_reason is None if passed.
    """
    min_score = thresholds["min_localization_score"]
    if diff_image_result.localization_score < min_score:
        return (
            False,
            f"Localization score of {diff_image_result.localization_score:.3f} "
            f"is below threshold of {min_score:.1f} "
            f"(transit may not originate from target)",
        )
    return (True, None)


def _evaluate_aperture_stability(
    aperture_result: ApertureDependenceResult,
    thresholds: dict[str, float],
) -> tuple[bool, str | None]:
    """Evaluate aperture stability test.

    The aperture stability test passes if the stability metric meets
    or exceeds the minimum threshold, indicating consistent transit
    depth across different aperture sizes.

    Args:
        aperture_result: Results from aperture dependence analysis.
        thresholds: Threshold configuration dict.

    Returns:
        Tuple of (passed, failure_reason). failure_reason is None if passed.
    """
    min_stability = thresholds["min_aperture_stability"]
    if aperture_result.stability_metric < min_stability:
        return (
            False,
            f"Aperture stability metric of {aperture_result.stability_metric:.3f} "
            f"is below threshold of {min_stability:.1f} "
            f"(depth varies significantly with aperture size)",
        )
    return (True, None)


def generate_pixel_vet_report(
    centroid_result: CentroidResult,
    diff_image_result: DifferenceImageResult,
    aperture_result: ApertureDependenceResult,
    threshold_version: str = "v1",
    plots: list[dict[str, Any]] | None = None,
) -> PixelVetReport:
    """Generate a pixel vetting report from analysis results.

    This function combines results from centroid shift analysis,
    difference image localization, and aperture dependence analysis
    into a comprehensive pixel vetting report. It applies versioned
    thresholds to determine pass/fail status for each individual test
    and computes the overall PIXEL_PASS determination.

    The overall pixel_pass is True only if ALL individual tests pass.

    Args:
        centroid_result: Results from centroid shift analysis.
        diff_image_result: Results from difference image localization.
        aperture_result: Results from aperture dependence analysis.
        threshold_version: Version identifier for the threshold configuration
            to use (default: "v1"). Must be a key in THRESHOLD_VERSIONS.
        plots: Optional plot metadata (format determined by caller).

    Returns:
        A PixelVetReport containing all results, individual test outcomes,
        and the overall pass/fail determination.

    Raises:
        ValueError: If threshold_version is not found in THRESHOLD_VERSIONS.

    Example:
        >>> centroid = CentroidResult(
        ...     in_transit_centroid=(10.5, 20.3),
        ...     out_transit_centroid=(10.4, 20.2),
        ...     centroid_shift_pixels=0.14,
        ...     significance_sigma=1.2,
        ... )
        >>> diff_image = DifferenceImageResult(
        ...     brightest_pixel_coords=(10.0, 20.0),
        ...     target_coords=(10.0, 20.0),
        ...     localization_score=0.95,
        ... )
        >>> aperture = ApertureDependenceResult(
        ...     depths_by_aperture={1.0: 1.5, 2.0: 1.52, 3.0: 1.48},
        ...     stability_metric=0.92,
        ...     recommended_aperture=2.0,
        ... )
        >>> report = generate_pixel_vet_report(centroid, diff_image, aperture)
        >>> report.pixel_pass
        True
    """
    if threshold_version not in THRESHOLD_VERSIONS:
        valid_versions = ", ".join(sorted(THRESHOLD_VERSIONS.keys()))
        raise ValueError(
            f"Unknown threshold_version '{threshold_version}'. Valid versions: {valid_versions}"
        )

    thresholds = THRESHOLD_VERSIONS[threshold_version]
    individual_tests: dict[str, bool] = {}
    failure_reasons: list[str] = []

    # Evaluate centroid shift test
    passed, reason = _evaluate_centroid_shift(centroid_result, thresholds)
    individual_tests["centroid_shift"] = passed
    if reason:
        failure_reasons.append(reason)

    # Evaluate centroid significance test
    passed, reason = _evaluate_centroid_significance(centroid_result, thresholds)
    individual_tests["centroid_significance"] = passed
    if reason:
        failure_reasons.append(reason)

    # Evaluate localization test
    passed, reason = _evaluate_localization(diff_image_result, thresholds)
    individual_tests["localization"] = passed
    if reason:
        failure_reasons.append(reason)

    # Evaluate aperture stability test
    passed, reason = _evaluate_aperture_stability(aperture_result, thresholds)
    individual_tests["aperture_stability"] = passed
    if reason:
        failure_reasons.append(reason)

    # Overall pass requires ALL individual tests to pass
    pixel_pass = all(individual_tests.values())

    return PixelVetReport(
        pixel_pass=pixel_pass,
        centroid_result=centroid_result,
        diff_image_result=diff_image_result,
        aperture_result=aperture_result,
        individual_tests=individual_tests,
        failure_reasons=failure_reasons,
        plots=plots or [],
    )


__all__ = [
    "THRESHOLD_VERSIONS",
    "PixelVetReport",
    "generate_pixel_vet_report",
]
