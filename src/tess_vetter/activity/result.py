"""Domain models for stellar activity characterization.

This module provides dataclasses for representing activity analysis results:
- Flare: Individual flare detection
- ActivityResult: Comprehensive activity characterization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Flare:
    """Single flare event detection.

    Represents a detected stellar flare with timing and amplitude information.

    Attributes:
        start_time: Flare start time in days (e.g., BTJD)
        end_time: Flare end time in days
        peak_time: Time of peak brightness in days
        amplitude: Peak fractional flux increase above baseline
        duration_minutes: Total flare duration in minutes
        energy_estimate: Estimated bolometric energy in ergs (rough)
    """

    start_time: float
    end_time: float
    peak_time: float
    amplitude: float
    duration_minutes: float
    energy_estimate: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation with rounded values.
        """
        return {
            "start_time": round(self.start_time, 6),
            "end_time": round(self.end_time, 6),
            "peak_time": round(self.peak_time, 6),
            "amplitude": round(self.amplitude, 6),
            "duration_minutes": round(self.duration_minutes, 2),
            "energy_estimate": self.energy_estimate,
        }


@dataclass(frozen=True)
class ActivityResult:
    """Comprehensive stellar activity characterization.

    Contains rotation period measurement, variability classification,
    flare statistics, and recommendations for transit detection.

    Attributes:
        rotation_period: Measured rotation period in days
        rotation_err: Uncertainty on rotation period in days
        rotation_snr: Signal-to-noise ratio of rotation detection
        variability_ppm: RMS variability amplitude in parts per million
        variability_class: Classification (spotted_rotator, pulsator, etc.)
        flares: List of detected flares
        flare_rate: Flare rate in flares per day
        activity_index: Photometric activity proxy (0 to 1 scale)
        recommendation: Guidance for transit detection
        suggested_params: Machine-actionable parameters for recover_transit
        sectors_used: TESS sectors included in analysis
        tic_id: TESS Input Catalog identifier
    """

    rotation_period: float
    rotation_err: float
    rotation_snr: float
    variability_ppm: float
    variability_class: str
    flares: list[Flare]
    flare_rate: float
    activity_index: float
    recommendation: str
    suggested_params: dict[str, Any] = field(default_factory=dict)
    sectors_used: list[int] = field(default_factory=list)
    tic_id: int = 0

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "rotation_period": round(self.rotation_period, 4),
            "rotation_period_err": round(self.rotation_err, 4),
            "rotation_snr": round(self.rotation_snr, 2),
            "variability_amplitude_ppm": round(self.variability_ppm, 0),
            "variability_class": self.variability_class,
            "n_flares": len(self.flares),
            "flare_rate_per_day": round(self.flare_rate, 3),
            "mean_flare_energy_erg": (
                sum(f.energy_estimate for f in self.flares) / len(self.flares)
                if self.flares
                else 0.0
            ),
            "activity_index": round(self.activity_index, 3),
            "flare_catalog": [f.to_dict() for f in self.flares],
            "recommendation": self.recommendation,
            "suggested_recover_params": self.suggested_params,
            "sectors_used": self.sectors_used,
            "tic_id": self.tic_id,
        }
