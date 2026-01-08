"""Advanced vetting checks for transit candidate validation.

This module provides:
- VetterCheck: Abstract base class for all vetting checks
- V11 StellarDensityCheck: Compare transit-derived density with TIC stellar density
- V12 SingleTransitCheck: Flag if only one transit visible (period uncertain)
- V13 EphemerisStabilityCheck: Verify transit times are consistent with linear ephemeris

All checks return VetterCheckResult with standardized structure.

Note: These were originally V08-V10 but were renumbered to V11-V13 to avoid
collision with the pixel checks (V08-V10) defined in lc_checks.py.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.domain.detection import TransitCandidate, VetterCheckResult

if TYPE_CHECKING:
    from bittr_tess_vetter.domain.target import StellarParameters


class VetterCheck(ABC):
    """Abstract base class for transit vetting checks.

    Each check evaluates a specific aspect of a transit candidate
    and returns a VetterCheckResult indicating pass/fail status,
    confidence level, and detailed diagnostics.

    Subclasses must implement:
    - check_id: Class property with check ID (e.g., "V11")
    - check_name: Class property with human-readable name
    - run(): Method to execute the check and return result

    Example:
        class MyCheck(VetterCheck):
            check_id = "V99"
            check_name = "My Custom Check"

            def run(self, candidate, ...) -> VetterCheckResult:
                passed = some_test()
                return VetterCheckResult(
                    id=self.check_id,
                    name=self.check_name,
                    passed=passed,
                    confidence=0.9,
                    details={"metric": value}
                )
    """

    check_id: str
    check_name: str

    @abstractmethod
    def run(self, candidate: TransitCandidate, **kwargs: Any) -> VetterCheckResult:
        """Execute the vetting check.

        Args:
            candidate: TransitCandidate to evaluate.
            **kwargs: Check-specific parameters.

        Returns:
            VetterCheckResult with check outcome.
        """
        ...

    def _make_result(
        self,
        passed: bool,
        confidence: float,
        details: dict[str, Any] | None = None,
    ) -> VetterCheckResult:
        """Helper to create a VetterCheckResult with this check's ID and name.

        Args:
            passed: Whether the check passed.
            confidence: Confidence level (0-1).
            details: Optional details dictionary.

        Returns:
            VetterCheckResult instance.
        """
        return VetterCheckResult(
            id=self.check_id,
            name=self.check_name,
            passed=passed,
            confidence=confidence,
            details=details or {},
        )


class StellarDensityCheck(VetterCheck):
    """V11: Compare transit-derived stellar density with TIC catalog density.

    This check tests whether the stellar density implied by the transit
    parameters (period, duration) is consistent with the density from
    the TIC catalog (derived from mass and radius).

    For a circular orbit, the stellar density can be estimated from:
        rho_star = (3 * pi / G / P^2) * (a/R_star)^3

    where a/R_star can be derived from the transit duration and period
    assuming a central transit (impact parameter b=0).

    A significant discrepancy may indicate:
    - Blended eclipsing binary (diluted depth, wrong density)
    - Incorrect period (half or double the true period)
    - Grazing transit (underestimated a/R_star)
    - Stellar parameters in TIC are incorrect

    Note: This was originally V08 but was renumbered to V11 to avoid
    collision with the pixel check V08 (Centroid Shift) in lc_checks.py.

    Attributes:
        tolerance: Maximum allowed ratio difference (default 3.0x).
    """

    check_id = "V11"
    check_name = "Stellar Density Check"

    def __init__(self, tolerance: float = 3.0):
        """Initialize with density tolerance.

        Args:
            tolerance: Maximum allowed density ratio (transit/TIC).
                       Values outside 1/tolerance to tolerance fail.
        """
        self.tolerance = tolerance

    def run(
        self,
        candidate: TransitCandidate,
        stellar: StellarParameters | None = None,
        **kwargs: Any,
    ) -> VetterCheckResult:
        """Execute stellar density comparison check.

        Args:
            candidate: TransitCandidate with period and duration.
            stellar: StellarParameters from TIC catalog.

        Returns:
            VetterCheckResult with density comparison outcome.
        """
        # Handle missing stellar parameters
        if stellar is None:
            return self._make_result(
                passed=True,  # Cannot fail without data
                confidence=0.0,  # No confidence in result
                details={
                    "reason": "no_stellar_params",
                    "message": "No stellar parameters available for comparison",
                },
            )

        # Get TIC stellar density
        tic_density = stellar.stellar_density_solar()
        if tic_density is None:
            return self._make_result(
                passed=True,
                confidence=0.0,
                details={
                    "reason": "missing_mass_radius",
                    "message": "TIC mass or radius unavailable for density calculation",
                },
            )

        if tic_density <= 0:
            return self._make_result(
                passed=True,
                confidence=0.0,
                details={
                    "reason": "invalid_tic_density",
                    "message": f"TIC density is non-positive: {tic_density}",
                },
            )

        # Calculate transit-derived density
        # Using simplified formula assuming central transit (b=0)
        # a/R_star ~ (P / pi / T14) for circular orbit, central transit
        # where T14 is the full transit duration

        period_days = candidate.period
        duration_days = candidate.duration_hours / 24.0

        if duration_days <= 0 or duration_days >= period_days:
            return self._make_result(
                passed=False,
                confidence=0.8,
                details={
                    "reason": "invalid_duration",
                    "message": f"Invalid duration {duration_days:.4f}d for period {period_days:.4f}d",
                    "period_days": period_days,
                    "duration_days": duration_days,
                },
            )

        # Estimate a/R_star from duration and period
        # For central transit: T14/P = (1/pi) * (R_star/a) * sqrt(1 - b^2)
        # Assuming b=0: a/R_star = P / (pi * T14)
        a_over_rstar = period_days / (math.pi * duration_days)

        # Stellar density from Kepler's third law:
        # rho_star = (3*pi / G / P^2) * (a/R_star)^3
        # In solar units: rho_star/rho_sun = (a/R_star)^3 * (1 day / P)^2 * constant

        # The constant for solar units:
        # (3*pi / G) * (1 day)^2 / (M_sun/R_sun^3) = 0.01891 (approx)
        # So: rho_star [solar] = 0.01891 * (a/R_star)^3 / P^2 [days]

        density_const = 0.01891  # Solar density units with P in days
        transit_density = density_const * (a_over_rstar**3) / (period_days**2)

        # Calculate density ratio
        density_ratio = transit_density / tic_density

        # Check if within tolerance
        passed = (1.0 / self.tolerance) <= density_ratio <= self.tolerance

        # Confidence based on how far from tolerance boundary
        if passed:
            # Higher confidence when ratio is closer to 1.0
            log_ratio = abs(math.log10(density_ratio))
            max_log_ratio = math.log10(self.tolerance)
            confidence = max(0.5, 1.0 - (log_ratio / max_log_ratio) * 0.4)
        else:
            # Failed check - high confidence in failure
            confidence = 0.9

        return self._make_result(
            passed=passed,
            confidence=confidence,
            details={
                "transit_density_solar": round(transit_density, 4),
                "tic_density_solar": round(tic_density, 4),
                "density_ratio": round(density_ratio, 3),
                "a_over_rstar": round(a_over_rstar, 2),
                "tolerance": self.tolerance,
                "stellar_mass": stellar.mass,
                "stellar_radius": stellar.radius,
            },
        )


class SingleTransitCheck(VetterCheck):
    """V12: Flag candidates with only one observed transit.

    Single-transit events have highly uncertain periods since the
    period is only constrained by the data span, not multiple transits.
    This check counts the number of transits visible in the data and
    flags candidates where only one transit is observed.

    A single transit may indicate:
    - Long-period planet (real but period uncertain)
    - Systematic artifact or data anomaly
    - Stellar variability event
    - Eclipsing binary with very long period

    The check passes if at least 2 transits are expected in the data,
    based on data span and orbital period.

    Note: This was originally V09 but was renumbered to V12 to avoid
    collision with the pixel check V09 (Pixel-Level LC) in lc_checks.py.
    """

    check_id = "V12"
    check_name = "Single Transit Check"

    def __init__(self, min_transits: int = 2):
        """Initialize with minimum transit count.

        Args:
            min_transits: Minimum number of transits required to pass.
        """
        self.min_transits = min_transits

    def run(
        self,
        candidate: TransitCandidate,
        time: NDArray[np.float64] | None = None,
        data_span_days: float | None = None,
        **kwargs: Any,
    ) -> VetterCheckResult:
        """Execute single transit check.

        Args:
            candidate: TransitCandidate with period.
            time: Time array to calculate data span (optional).
            data_span_days: Data span in days (alternative to time array).

        Returns:
            VetterCheckResult flagging single-transit candidates.
        """
        # Determine data span
        if time is not None and len(time) > 1:
            span = float(time[-1] - time[0])
        elif data_span_days is not None:
            span = data_span_days
        else:
            return self._make_result(
                passed=True,  # Cannot evaluate without data span
                confidence=0.0,
                details={
                    "reason": "no_data_span",
                    "message": "Cannot determine data span for transit count",
                },
            )

        if span <= 0:
            return self._make_result(
                passed=True,
                confidence=0.0,
                details={
                    "reason": "invalid_data_span",
                    "message": f"Data span is non-positive: {span}",
                },
            )

        period = candidate.period

        # Calculate number of transits expected
        # Account for partial transits at edges
        n_transits_expected = span / period

        # Integer count (floor)
        n_transits_int = int(n_transits_expected)

        # Check if we have minimum transits
        passed = n_transits_int >= self.min_transits

        # Confidence depends on how clearly we pass or fail
        # More transits = higher confidence; single transit = high confidence in the flag
        confidence = (0.95 if n_transits_int >= 3 else 0.75) if passed else 0.9

        return self._make_result(
            passed=passed,
            confidence=confidence,
            details={
                "data_span_days": round(span, 2),
                "period_days": round(period, 4),
                "n_transits_expected": round(n_transits_expected, 2),
                "n_transits_int": n_transits_int,
                "min_transits_required": self.min_transits,
                "is_single_transit": n_transits_int < 2,
            },
        )


class EphemerisStabilityCheck(VetterCheck):
    """V13: Verify transit times are consistent with linear ephemeris.

    For a true planet, individual transit times should follow a linear
    ephemeris (T_n = T0 + n * P) with only small deviations due to
    measurement uncertainty and physical effects (TTVs).

    Large deviations from linear ephemeris may indicate:
    - Incorrect period identification
    - Blended eclipsing binaries with different periods
    - Stellar variability mimicking transits
    - Real transit timing variations (TTVs) from planet interactions

    This check computes the O-C (Observed minus Calculated) residuals
    for measured transit times and flags candidates with excessive scatter.

    Note: This was originally V10 but was renumbered to V13 to avoid
    collision with the pixel check V10 (Aperture Dependence) in lc_checks.py.

    Attributes:
        max_oc_sigma: Maximum allowed O-C scatter in units of expected uncertainty.
        max_oc_hours: Maximum allowed O-C scatter in hours.
    """

    check_id = "V13"
    check_name = "Ephemeris Stability Check"

    def __init__(self, max_oc_sigma: float = 5.0, max_oc_hours: float = 2.0):
        """Initialize with O-C tolerance thresholds.

        Args:
            max_oc_sigma: Maximum allowed O-C RMS in sigma units.
            max_oc_hours: Maximum allowed O-C RMS in hours.
        """
        self.max_oc_sigma = max_oc_sigma
        self.max_oc_hours = max_oc_hours

    def run(
        self,
        candidate: TransitCandidate,
        transit_times: NDArray[np.float64] | list[float] | None = None,
        transit_time_errors: NDArray[np.float64] | list[float] | None = None,
        time: NDArray[np.float64] | None = None,
        flux: NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> VetterCheckResult:
        """Execute ephemeris stability check.

        Args:
            candidate: TransitCandidate with period and t0.
            transit_times: Measured individual transit times (BTJD).
            transit_time_errors: Uncertainties on transit times (days).
            time: Full time array (used if transit_times not provided).
            flux: Full flux array (used if transit_times not provided).

        Returns:
            VetterCheckResult with ephemeris stability assessment.
        """
        # If transit times not provided, try to estimate from data
        if transit_times is None:
            if time is not None and len(time) > 1:
                # Estimate number of transits from data span
                span = float(time[-1] - time[0])
                n_expected = int(span / candidate.period)

                if n_expected < 2:
                    return self._make_result(
                        passed=True,
                        confidence=0.3,  # Low confidence - cannot verify
                        details={
                            "reason": "insufficient_transits",
                            "message": f"Only {n_expected} transit(s) expected, cannot verify ephemeris",
                            "n_transits_expected": n_expected,
                        },
                    )

                # Estimate transit times from ephemeris
                t0 = candidate.t0
                period = candidate.period

                # Find transit epochs within data range
                n_min = int(np.floor((time[0] - t0) / period))
                n_max = int(np.ceil((time[-1] - t0) / period))

                expected_times = []
                for n in range(n_min, n_max + 1):
                    t_transit = t0 + n * period
                    if time[0] <= t_transit <= time[-1]:
                        expected_times.append(t_transit)

                n_transits = len(expected_times)

                if n_transits < 2:
                    return self._make_result(
                        passed=True,
                        confidence=0.3,
                        details={
                            "reason": "insufficient_transits_in_data",
                            "message": f"Only {n_transits} transit(s) in data window",
                            "n_transits": n_transits,
                        },
                    )

                # Without measured times, we can only verify ephemeris coverage
                return self._make_result(
                    passed=True,
                    confidence=0.5,  # Moderate confidence - ephemeris is self-consistent
                    details={
                        "reason": "no_measured_transit_times",
                        "message": "Ephemeris coverage verified but individual times not measured",
                        "n_transits_in_window": n_transits,
                        "expected_times": [round(t, 4) for t in expected_times[:5]],
                    },
                )
            else:
                return self._make_result(
                    passed=True,
                    confidence=0.0,
                    details={
                        "reason": "no_transit_times",
                        "message": "No transit times provided for ephemeris check",
                    },
                )

        # Convert to numpy array
        transit_times = np.asarray(transit_times, dtype=np.float64)
        n_transits = len(transit_times)

        if n_transits < 2:
            return self._make_result(
                passed=True,
                confidence=0.3,
                details={
                    "reason": "insufficient_transit_times",
                    "message": f"Only {n_transits} transit time(s) provided",
                    "n_transits": n_transits,
                },
            )

        # Calculate expected transit times from linear ephemeris
        t0 = candidate.t0
        period = candidate.period

        # Find epoch numbers for each transit
        epochs = np.round((transit_times - t0) / period).astype(int)
        calculated_times: NDArray[np.float64] = t0 + epochs * period

        # Calculate O-C residuals
        oc_residuals = transit_times - calculated_times
        oc_residuals_hours = oc_residuals * 24.0

        # Calculate O-C statistics
        oc_rms_days = float(np.sqrt(np.mean(oc_residuals**2)))
        oc_rms_hours = oc_rms_days * 24.0
        oc_max_hours = float(np.max(np.abs(oc_residuals_hours)))

        # Check against thresholds
        passed_hours = oc_rms_hours <= self.max_oc_hours

        # Check against sigma threshold if errors provided
        if transit_time_errors is not None:
            transit_time_errors = np.asarray(transit_time_errors, dtype=np.float64)
            if len(transit_time_errors) == n_transits and np.all(transit_time_errors > 0):
                oc_sigma = oc_residuals / transit_time_errors
                oc_rms_sigma = float(np.sqrt(np.mean(oc_sigma**2)))
                passed_sigma = oc_rms_sigma <= self.max_oc_sigma
            else:
                oc_rms_sigma = None
                passed_sigma = True  # Cannot evaluate
        else:
            oc_rms_sigma = None
            passed_sigma = True  # Cannot evaluate

        passed = passed_hours and passed_sigma

        # Confidence based on number of transits and residual quality
        if passed:
            if n_transits >= 5:
                confidence = 0.95
            elif n_transits >= 3:
                confidence = 0.85
            else:
                confidence = 0.7
        else:
            confidence = 0.9  # High confidence in failure

        details = {
            "n_transits": n_transits,
            "oc_rms_hours": round(oc_rms_hours, 4),
            "oc_max_hours": round(oc_max_hours, 4),
            "max_oc_hours_threshold": self.max_oc_hours,
            "epochs": epochs.tolist(),
        }

        if oc_rms_sigma is not None:
            details["oc_rms_sigma"] = round(oc_rms_sigma, 2)
            details["max_oc_sigma_threshold"] = self.max_oc_sigma

        return self._make_result(
            passed=passed,
            confidence=confidence,
            details=details,
        )


# Export all check classes
__all__ = [
    "VetterCheck",
    "StellarDensityCheck",
    "SingleTransitCheck",
    "EphemerisStabilityCheck",
]
