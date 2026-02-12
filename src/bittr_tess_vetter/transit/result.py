"""Domain models for transit timing and vetting analysis.

This module provides dataclasses for representing transit analysis results:
- TransitTime: Single transit timing measurement
- TTVResult: Transit timing variation analysis summary
- OddEvenResult: Odd/even depth comparison for EB vetting
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitTime:
    """Single transit timing measurement.

    Represents the measured mid-transit time for one transit event,
    along with associated uncertainties and quality metrics.

    Attributes:
        epoch: Transit epoch number (integer, relative to t0)
        tc: Measured mid-transit time (BTJD)
        tc_err: Uncertainty on tc (days)
        depth_ppm: Measured transit depth in parts per million
        duration_hours: Measured transit duration in hours
        snr: Signal-to-noise ratio of this individual transit
        is_outlier: True if this transit is flagged as an outlier
        outlier_reason: Reason for outlier flag, or None if not an outlier
    """

    epoch: int
    tc: float
    tc_err: float
    depth_ppm: float
    duration_hours: float
    snr: float
    is_outlier: bool = False
    outlier_reason: str | None = None


@dataclass(frozen=True)
class TransitTimingPoint:
    """Per-epoch transit timing point for diagnostics.

    Bundles timing residual, per-transit significance, and quality flags
    in one row-oriented structure.
    """

    epoch: int
    tc_btjd: float
    tc_err_days: float
    oc_seconds: float
    snr: float
    depth_ppm: float
    duration_hours: float
    is_outlier: bool
    outlier_reason: str | None


@dataclass(frozen=True)
class TransitTimingSeries:
    """Transit timing diagnostics series for API/report consumption."""

    points: list[TransitTimingPoint]
    n_points: int
    rms_seconds: float | None
    periodicity_score: float | None
    linear_trend_sec_per_epoch: float | None

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "n_points": self.n_points,
            "rms_seconds": self.rms_seconds,
            # Heuristic periodicity score from LS power contrast (not a calibrated sigma).
            "periodicity_score": self.periodicity_score,
            # Backward-compat alias for older callers expecting this key.
            "periodicity_sigma": self.periodicity_score,
            "linear_trend_sec_per_epoch": self.linear_trend_sec_per_epoch,
            "points": [
                {
                    "epoch": p.epoch,
                    "tc_btjd": p.tc_btjd,
                    "tc_err_days": p.tc_err_days,
                    "oc_seconds": p.oc_seconds,
                    "snr": p.snr,
                    "depth_ppm": p.depth_ppm,
                    "duration_hours": p.duration_hours,
                    "is_outlier": p.is_outlier,
                    "outlier_reason": p.outlier_reason,
                }
                for p in self.points
            ],
        }


@dataclass(frozen=True)
class TTVResult:
    """Transit timing variation analysis summary.

    Contains the full TTV analysis results including measured transit times,
    O-C residuals, and timing statistics.

    Attributes:
        transit_times: List of individual transit measurements
        o_minus_c: O-C residuals in seconds (observed - calculated)
        rms_seconds: RMS of O-C residuals in seconds
        periodicity_sigma: Significance of any periodic TTV signal (sigma)
        n_transits: Number of transits successfully measured
        linear_trend: Linear trend in O-C (seconds per epoch), if detected
    """

    transit_times: list[TransitTime]
    o_minus_c: list[float]
    rms_seconds: float
    periodicity_sigma: float
    n_transits: int
    linear_trend: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "n_transits": self.n_transits,
            "rms_seconds": round(self.rms_seconds, 2),
            "periodicity_sigma": round(self.periodicity_sigma, 2),
            "linear_trend": (
                round(self.linear_trend, 4) if self.linear_trend is not None else None
            ),
            "o_minus_c": [round(x, 2) for x in self.o_minus_c],
            "transit_times": [
                {
                    "epoch": t.epoch,
                    "tc": round(t.tc, 6),
                    "tc_err": round(t.tc_err, 6),
                    "depth_ppm": round(t.depth_ppm, 1),
                    "duration_hours": round(t.duration_hours, 2),
                    "snr": round(t.snr, 2),
                    "is_outlier": t.is_outlier,
                    "outlier_reason": t.outlier_reason,
                }
                for t in self.transit_times
            ],
        }


@dataclass(frozen=True)
class OddEvenResult:
    """Odd/even depth comparison result for eclipsing binary vetting.

    Compares the transit depth between odd and even epochs to detect
    diluted eclipsing binaries. A significant depth difference suggests
    the signal may not be a planet.

    Attributes:
        depth_odd_ppm: Mean transit depth for odd epochs (ppm)
        depth_even_ppm: Mean transit depth for even epochs (ppm)
        depth_diff_ppm: Absolute difference ``abs(odd - even)`` (ppm)
        relative_depth_diff_percent: Relative difference as percentage of mean depth
        significance_sigma: Statistical significance of the difference (sigma)
        is_suspicious: True if relative depth difference exceeds threshold
        interpretation: Coarse classification of the result
        n_odd: Number of odd-epoch transits measured
        n_even: Number of even-epoch transits measured
    """

    depth_odd_ppm: float
    depth_even_ppm: float
    depth_diff_ppm: float
    relative_depth_diff_percent: float
    significance_sigma: float
    is_suspicious: bool = False
    interpretation: str = "INSUFFICIENT_DATA"
    n_odd: int = 0
    n_even: int = 0

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "depth_odd_ppm": round(self.depth_odd_ppm, 1),
            "depth_even_ppm": round(self.depth_even_ppm, 1),
            "depth_diff_ppm": round(self.depth_diff_ppm, 1),
            "relative_depth_diff_percent": round(self.relative_depth_diff_percent, 2),
            "significance_sigma": round(self.significance_sigma, 2),
            "is_suspicious": bool(self.is_suspicious),
            "interpretation": self.interpretation,
            "n_odd": self.n_odd,
            "n_even": self.n_even,
        }
