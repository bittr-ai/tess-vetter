"""Tolerance checking utilities for comparing astronomical parameter values.

Provides tolerance-based comparison for exoplanet transit parameters,
supporting different tolerance types:
- period_days: relative tolerance + harmonic allowances (1/2, 1/3, 2x, 3x)
- t0_btjd: phase tolerance relative to period
- depth: relative tolerance
- default: absolute tolerance

Example:
    >>> tolerances = {
    ...     "period_days": {"relative": 0.01, "harmonics": True},
    ...     "t0_btjd": {"phase_fraction": 0.1},
    ...     "depth": {"relative": 0.1},
    ...     "default": {"absolute": 0.001}
    ... }
    >>> result = check_tolerance("period_days", 2.0, 2.02, tolerances)
    >>> result.within_tolerance
    True
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToleranceResult(FrozenModel):
    """Result of a tolerance check between original and replayed values.

    Attributes:
        within_tolerance: Whether the replayed value is within tolerance.
        delta: Absolute difference between original and replayed values.
        tolerance_used: Description of the tolerance type and threshold used.
        relative_error: Relative error (|delta| / |original|), or None if
            original is zero or tolerance type doesn't use relative error.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    within_tolerance: bool
    delta: float
    tolerance_used: str
    relative_error: float | None


# Standard harmonic ratios to check for period comparisons
HARMONIC_RATIOS: tuple[float, ...] = (0.5, 1 / 3, 2.0, 3.0)


def _check_period_tolerance(
    original: float,
    replayed: float,
    config: dict[str, Any],
) -> ToleranceResult:
    """Check tolerance for period_days with optional harmonic allowances.

    For period values, we check:
    1. Direct relative tolerance (e.g., 1% difference)
    2. If harmonics enabled, also check if replayed is a harmonic of original
       (1/2, 1/3, 2x, 3x times the original period)

    Args:
        original: Original period value in days.
        replayed: Replayed period value in days.
        config: Tolerance config with 'relative' and optionally 'harmonics' keys.

    Returns:
        ToleranceResult with check outcome.
    """
    relative_tol = config.get("relative", 0.01)
    check_harmonics = config.get("harmonics", False)

    delta = abs(replayed - original)
    relative_error = delta / abs(original) if original != 0 else None

    # Check direct relative tolerance
    if original != 0 and relative_error is not None and relative_error <= relative_tol:
        return ToleranceResult(
            within_tolerance=True,
            delta=replayed - original,
            tolerance_used=f"period_days relative ({relative_tol * 100:.1f}%)",
            relative_error=relative_error,
        )

    # Check harmonics if enabled
    if check_harmonics and original != 0:
        for ratio in HARMONIC_RATIOS:
            harmonic_period = original * ratio
            harmonic_delta = abs(replayed - harmonic_period)
            harmonic_relative = harmonic_delta / abs(harmonic_period)

            if harmonic_relative <= relative_tol:
                ratio_str = _format_ratio(ratio)
                return ToleranceResult(
                    within_tolerance=True,
                    delta=replayed - original,
                    tolerance_used=f"period_days harmonic ({ratio_str}x, {relative_tol * 100:.1f}%)",
                    relative_error=relative_error,
                )

    # Not within tolerance
    tolerance_desc = f"period_days relative ({relative_tol * 100:.1f}%)"
    if check_harmonics:
        tolerance_desc += " + harmonics"

    return ToleranceResult(
        within_tolerance=False,
        delta=replayed - original,
        tolerance_used=tolerance_desc,
        relative_error=relative_error,
    )


def _format_ratio(ratio: float) -> str:
    """Format a harmonic ratio for display.

    Args:
        ratio: The harmonic ratio (e.g., 0.5, 0.333..., 2.0, 3.0)

    Returns:
        Formatted string like "1/2", "1/3", "2", "3"
    """
    if ratio == 0.5:
        return "1/2"
    elif abs(ratio - 1 / 3) < 0.001:
        return "1/3"
    elif ratio == 2.0:
        return "2"
    elif ratio == 3.0:
        return "3"
    else:
        return f"{ratio:.3f}"


def _check_t0_tolerance(
    original: float,
    replayed: float,
    config: dict[str, Any],
    tolerances: dict[str, Any],
) -> ToleranceResult:
    """Check tolerance for t0_btjd using phase-relative tolerance.

    T0 (epoch) differences should be evaluated relative to the period,
    since a phase shift of e.g., 10% of the period is meaningful regardless
    of the absolute T0 value.

    Args:
        original: Original T0 value in BTJD.
        replayed: Replayed T0 value in BTJD.
        config: Tolerance config with 'phase_fraction' key.
        tolerances: Full tolerances dict to get period if needed.

    Returns:
        ToleranceResult with check outcome.
    """
    phase_fraction = config.get("phase_fraction", 0.1)

    delta = replayed - original
    abs_delta = abs(delta)

    # Get period for phase calculation
    # Default to 1.0 if no period config available
    _ = tolerances.get("period_days", {})  # period_config unused
    # We need an actual period value, not tolerance config
    # Use a reference period if available, otherwise assume delta is meaningful
    reference_period = config.get("reference_period", 1.0)

    # Calculate phase difference
    phase_diff = abs_delta / reference_period if reference_period != 0 else abs_delta

    # Check if within phase tolerance
    within = phase_diff <= phase_fraction

    # Also handle case where t0 is wrapped by multiple periods
    if not within and reference_period != 0:
        # Check modulo period
        wrapped_delta = abs_delta % reference_period
        wrapped_phase = min(wrapped_delta, reference_period - wrapped_delta) / reference_period
        if wrapped_phase <= phase_fraction:
            within = True
            phase_diff = wrapped_phase

    return ToleranceResult(
        within_tolerance=within,
        delta=delta,
        tolerance_used=f"t0_btjd phase ({phase_fraction * 100:.1f}% of period)",
        relative_error=phase_diff if reference_period != 0 else None,
    )


def _check_depth_tolerance(
    original: float,
    replayed: float,
    config: dict[str, Any],
) -> ToleranceResult:
    """Check tolerance for depth using relative tolerance.

    Transit depth is typically a small fractional value, so relative
    tolerance is appropriate.

    Args:
        original: Original depth value (typically ppm or fraction).
        replayed: Replayed depth value.
        config: Tolerance config with 'relative' key.

    Returns:
        ToleranceResult with check outcome.
    """
    relative_tol = config.get("relative", 0.1)

    delta = replayed - original
    abs_delta = abs(delta)

    if original != 0:
        relative_error = abs_delta / abs(original)
        within = relative_error <= relative_tol
    else:
        # If original is zero, use absolute comparison
        relative_error = None
        within = abs_delta <= relative_tol

    return ToleranceResult(
        within_tolerance=within,
        delta=delta,
        tolerance_used=f"depth relative ({relative_tol * 100:.1f}%)",
        relative_error=relative_error,
    )


def _check_default_tolerance(
    original: float,
    replayed: float,
    config: dict[str, Any],
) -> ToleranceResult:
    """Check tolerance using absolute difference.

    Fallback for parameters without specific tolerance types.

    Args:
        original: Original value.
        replayed: Replayed value.
        config: Tolerance config with 'absolute' key.

    Returns:
        ToleranceResult with check outcome.
    """
    absolute_tol = config.get("absolute", 0.001)

    delta = replayed - original
    abs_delta = abs(delta)
    within = abs_delta <= absolute_tol

    relative_error = abs_delta / abs(original) if original != 0 else None

    return ToleranceResult(
        within_tolerance=within,
        delta=delta,
        tolerance_used=f"default absolute ({absolute_tol})",
        relative_error=relative_error,
    )


def check_tolerance(
    name: str,
    original: float,
    replayed: float,
    tolerances: dict[str, Any],
) -> ToleranceResult:
    """Check if a replayed value is within tolerance of the original.

    Supports different tolerance types based on parameter name:
    - period_days: relative tolerance + optional harmonic allowances
    - t0_btjd: phase tolerance relative to period
    - depth: relative tolerance
    - default: absolute tolerance (fallback)

    Args:
        name: Parameter name (e.g., "period_days", "t0_btjd", "depth").
        original: Original/reference value.
        replayed: Replayed/computed value to check.
        tolerances: Dict mapping parameter names to tolerance configs.
            Example:
            {
                "period_days": {"relative": 0.01, "harmonics": True},
                "t0_btjd": {"phase_fraction": 0.1, "reference_period": 2.5},
                "depth": {"relative": 0.1},
                "default": {"absolute": 0.001}
            }

    Returns:
        ToleranceResult with:
        - within_tolerance: True if replayed is within tolerance of original
        - delta: Signed difference (replayed - original)
        - tolerance_used: Description of tolerance type and threshold
        - relative_error: Relative error if applicable, None otherwise

    Example:
        >>> tolerances = {
        ...     "period_days": {"relative": 0.01, "harmonics": True},
        ...     "default": {"absolute": 0.001}
        ... }
        >>> result = check_tolerance("period_days", 2.0, 2.02, tolerances)
        >>> result.within_tolerance
        True
        >>> result.tolerance_used
        'period_days relative (1.0%)'
    """
    # Get config for this parameter, or default
    if name in tolerances:
        config = tolerances[name]
    elif "default" in tolerances:
        config = tolerances["default"]
        name = "default"  # Use default tolerance type
    else:
        # No tolerance config at all, use strict default
        config = {"absolute": 0.001}
        name = "default"

    # Dispatch to appropriate tolerance checker
    if name == "period_days":
        return _check_period_tolerance(original, replayed, config)
    elif name == "t0_btjd":
        return _check_t0_tolerance(original, replayed, config, tolerances)
    elif name == "depth":
        return _check_depth_tolerance(original, replayed, config)
    else:
        return _check_default_tolerance(original, replayed, config)
