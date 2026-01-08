"""Public API types for bittr-tess-vetter.

This module provides user-facing types for the API facade:
- Ephemeris: Transit ephemeris (period, t0, duration)
- LightCurve: Simplified light curve container with dtype normalization
- StellarParams: Alias for internal StellarParameters
- CheckResult: Vetting check result (id, name, passed, confidence, details)
- Candidate: Transit candidate container (v2)
- TPFStamp: Target Pixel File data container (v2)
- VettingBundleResult: Orchestrator output with provenance (v2)

v3 type re-exports:
- TransitFitResult: Physical transit model fit result
- TransitTime: Single transit timing measurement
- TTVResult: Transit timing variation analysis summary
- OddEvenResult: Odd/even depth comparison for EB vetting
- ActivityResult: Stellar activity characterization
- Flare: Individual flare detection
- StackedTransit: Stacked transit light curve data
- TrapezoidFit: Trapezoid model fit parameters
- RecoveryResult: Transit recovery result from active star
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

# v3 type re-exports from internal modules
from bittr_tess_vetter.activity.result import ActivityResult, Flare
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.domain.target import StellarParameters
from bittr_tess_vetter.recovery.result import StackedTransit, TrapezoidFit
from bittr_tess_vetter.transit.result import OddEvenResult, TransitTime, TTVResult

# Re-export v3 types for public API
__all__ = [
    "ActivityResult",
    "Candidate",
    "CheckResult",
    "Ephemeris",
    "Flare",
    "LightCurve",
    "OddEvenResult",
    "StackedTransit",
    "StellarParams",
    "TPFStamp",
    "TransitTime",
    "TrapezoidFit",
    "TTVResult",
    "VettingBundleResult",
]

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Re-export StellarParameters as StellarParams for API consistency
StellarParams = StellarParameters


@dataclass(frozen=True)
class Ephemeris:
    """Transit ephemeris parameters.

    Attributes:
        period_days: Orbital period in days
        t0_btjd: Reference transit epoch in BTJD (Barycentric TESS Julian Date)
        duration_hours: Transit duration in hours
    """

    period_days: float
    t0_btjd: float
    duration_hours: float

    def __post_init__(self) -> None:
        """Validate ephemeris parameters."""
        if self.period_days <= 0:
            raise ValueError(f"period_days must be positive, got {self.period_days}")
        if self.duration_hours <= 0:
            raise ValueError(f"duration_hours must be positive, got {self.duration_hours}")


@dataclass
class LightCurve:
    """Simplified light curve container for API use.

    This is the public-facing light curve type. It accepts numpy arrays
    of any compatible dtype and normalizes them via to_internal().

    Attributes:
        time: Time array (BTJD timestamps)
        flux: Normalized flux values (median ~1.0)
        flux_err: Flux uncertainties (optional, defaults to zeros)
        quality: TESS quality flags (optional, defaults to zeros)
        valid_mask: Boolean mask for valid data points (optional, defaults to all True)
    """

    time: NDArray[Any]
    flux: NDArray[Any]
    flux_err: NDArray[Any] | None = None
    quality: NDArray[Any] | None = None
    valid_mask: NDArray[Any] | None = None

    def to_internal(
        self,
        *,
        tic_id: int = 0,
        sector: int = 0,
        cadence_seconds: float = 120.0,
    ) -> LightCurveData:
        """Convert to internal LightCurveData with dtype normalization.

        Normalizes dtypes to match internal constraints:
        - time -> float64
        - flux -> float64
        - flux_err -> float64
        - quality -> int32 (default to zeros if None)
        - valid_mask -> bool_ (default to all-True if None)

        Args:
            tic_id: TIC identifier (default 0 for anonymous data)
            sector: TESS sector number (default 0 for anonymous data)
            cadence_seconds: Observation cadence in seconds (default 120.0)

        Returns:
            LightCurveData instance with normalized dtypes
        """
        n = len(self.time)

        # Normalize time to float64
        time_arr = np.asarray(self.time, dtype=np.float64)

        # Normalize flux to float64
        flux_arr = np.asarray(self.flux, dtype=np.float64)

        # Normalize flux_err to float64, default to zeros
        if self.flux_err is not None:
            flux_err_arr = np.asarray(self.flux_err, dtype=np.float64)
        else:
            flux_err_arr = np.zeros(n, dtype=np.float64)

        # Normalize quality to int32, default to zeros
        if self.quality is not None:
            quality_arr = np.asarray(self.quality, dtype=np.int32)
        else:
            quality_arr = np.zeros(n, dtype=np.int32)

        # Normalize valid_mask to bool_, default to all True
        if self.valid_mask is not None:
            valid_mask_arr = np.asarray(self.valid_mask, dtype=np.bool_)
        else:
            valid_mask_arr = np.ones(n, dtype=np.bool_)

        return LightCurveData(
            time=time_arr,
            flux=flux_arr,
            flux_err=flux_err_arr,
            quality=quality_arr,
            valid_mask=valid_mask_arr,
            tic_id=tic_id,
            sector=sector,
            cadence_seconds=cadence_seconds,
        )

    @classmethod
    def from_internal(cls, data: LightCurveData) -> LightCurve:
        """Create LightCurve from internal LightCurveData.

        Args:
            data: Internal LightCurveData instance

        Returns:
            LightCurve instance wrapping the internal arrays
        """
        # Make writable copies to avoid issues with immutable internal arrays
        return cls(
            time=np.array(data.time),
            flux=np.array(data.flux),
            flux_err=np.array(data.flux_err),
            quality=np.array(data.quality),
            valid_mask=np.array(data.valid_mask),
        )


@dataclass(frozen=True)
class CheckResult:
    """Result of a single vetting check.

    Attributes:
        id: Check identifier (e.g., "V01", "V02")
        name: Human-readable check name (e.g., "odd_even_depth")
        passed: Whether the check passed
        confidence: Confidence in the result (0.0 to 1.0)
        details: Check-specific details dictionary
    """

    id: str
    name: str
    passed: bool
    confidence: float
    details: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate check result fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class Candidate:
    """Transit candidate container for vetting.

    Bundles ephemeris with optional depth information to avoid repeated
    argument lists across check functions.

    Attributes:
        ephemeris: Transit ephemeris (period, t0, duration)
        depth_ppm: Transit depth in parts per million (optional)
        depth_fraction: Transit depth as fraction 0-1 (optional)

    Note:
        If both depth_ppm and depth_fraction are provided, they must agree
        within 1% tolerance (depth_ppm == depth_fraction * 1e6).
    """

    ephemeris: Ephemeris
    depth_ppm: float | None = None
    depth_fraction: float | None = None

    def __post_init__(self) -> None:
        """Validate depth consistency if both provided."""
        if self.depth_ppm is not None and self.depth_fraction is not None:
            expected_ppm = self.depth_fraction * 1e6
            relative_diff = abs(self.depth_ppm - expected_ppm) / max(self.depth_ppm, 1e-10)
            if relative_diff > 0.01:
                raise ValueError(
                    f"depth_ppm ({self.depth_ppm}) and depth_fraction ({self.depth_fraction}) "
                    f"disagree by {relative_diff * 100:.1f}% (>1% tolerance)"
                )

    @property
    def depth(self) -> float | None:
        """Return depth as fraction, preferring depth_fraction if set."""
        if self.depth_fraction is not None:
            return self.depth_fraction
        if self.depth_ppm is not None:
            return self.depth_ppm / 1e6
        return None


@dataclass
class TPFStamp:
    """Target Pixel File data container (array-only, no FITS I/O).

    Holds TPF flux cube and associated metadata for pixel-level vetting.
    This is the public API type; internal code may use TPFFitsData for
    full FITS/WCS support.

    Attributes:
        time: Time array in BTJD, shape (n_cadences,)
        flux: Flux cube, shape (n_cadences, n_rows, n_cols)
        flux_err: Flux error cube (optional), same shape as flux
        wcs: WCS object for coordinate transforms (optional, passthrough)
        aperture_mask: Pipeline aperture mask, shape (n_rows, n_cols)
        quality: Quality flags, shape (n_cadences,)
    """

    time: NDArray[Any]
    flux: NDArray[Any]  # (n_cadences, n_rows, n_cols)
    flux_err: NDArray[Any] | None = None
    wcs: Any | None = None
    aperture_mask: NDArray[Any] | None = None
    quality: NDArray[Any] | None = None

    def __post_init__(self) -> None:
        """Validate array shapes."""
        if self.flux.ndim != 3:
            raise ValueError(f"flux must be 3D (n_cadences, n_rows, n_cols), got {self.flux.ndim}D")
        n_cadences = self.flux.shape[0]
        if len(self.time) != n_cadences:
            raise ValueError(
                f"time length ({len(self.time)}) must match flux cadences ({n_cadences})"
            )
        if self.flux_err is not None and self.flux_err.shape != self.flux.shape:
            raise ValueError(
                f"flux_err shape {self.flux_err.shape} must match flux shape {self.flux.shape}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return (n_cadences, n_rows, n_cols)."""
        return (self.flux.shape[0], self.flux.shape[1], self.flux.shape[2])

    @property
    def n_cadences(self) -> int:
        """Number of time cadences."""
        return self.flux.shape[0]

    @property
    def stamp_shape(self) -> tuple[int, int]:
        """Return (n_rows, n_cols)."""
        return (self.flux.shape[1], self.flux.shape[2])


@dataclass
class VettingBundleResult:
    """Structured output from the vetting orchestrator.

    Wraps check results with provenance metadata for reproducibility
    and audit trails.

    Attributes:
        results: List of CheckResult from all enabled checks
        provenance: Metadata dict with versions, thresholds, citations
        warnings: List of warning messages from checks
    """

    results: list[CheckResult]
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_passed(self) -> int:
        """Count of passed checks."""
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        """Count of failed checks."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def all_passed(self) -> bool:
        """True if all checks passed."""
        return all(r.passed for r in self.results)

    @property
    def failed_check_ids(self) -> list[str]:
        """List of IDs for failed checks."""
        return [r.id for r in self.results if not r.passed]

    def get_result(self, check_id: str) -> CheckResult | None:
        """Get result by check ID."""
        for r in self.results:
            if r.id == check_id:
                return r
        return None
