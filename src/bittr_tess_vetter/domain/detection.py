"""Detection and validation domain models.

This module provides:
- PeriodogramPeak: Individual peak from periodogram analysis
- TransitCandidate: Candidate transit parameters for vetting
- VetterCheckResult: Result of a single vetting check
- Detection: Wrapper combining candidate with metadata
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# Type aliases with validation
SNR = Annotated[float, Field(ge=0, description="Signal-to-noise ratio")]
FAP = Annotated[float, Field(ge=0, le=1, description="False alarm probability")]
PeriodDays = Annotated[float, Field(gt=0, description="Period, in days")]
DurationHours = Annotated[float, Field(gt=0, description="Duration, in hours")]


class PeriodogramPeak(FrozenModel):
    """Individual peak from periodogram analysis.

    Represents a single significant peak from BLS or Lomb-Scargle periodogram.
    """

    period: PeriodDays
    power: float = Field(ge=0, description="Periodogram power at peak")
    t0: float = Field(description="Reference epoch, in BTJD (days)")
    duration_hours: DurationHours | None = None  # BLS only
    depth_ppm: float | None = Field(
        default=None,
        ge=0,
        description="Transit depth in ppm (available for transit-search methods like TLS/BLS)",
    )
    snr: SNR | None = None
    fap: FAP | None = None


class TransitCandidate(FrozenModel):
    """Candidate transit parameters for vetting.

    This is the input to the validation pipeline.
    """

    period: PeriodDays
    t0: float = Field(description="Reference epoch, in BTJD (days)")
    duration_hours: DurationHours
    depth: float = Field(gt=0, le=1, description="Transit depth (fractional)")
    snr: SNR

    @property
    def duration_days(self) -> float:
        """Duration in days for calculations."""
        return self.duration_hours / 24.0


class VetterCheckResult(FrozenModel):
    """Result of a single vetting check.

    Each check produces one of these with metrics-only status,
    confidence, and check-specific details.

    Check IDs:
    - V01-V10: Canonical vetting checks (see lc_checks.py)
    - PF01-PF99: Pre-filter checks (optional, run before V01-V10)

    Passed field:
    - Always None (metrics-only). Host applications may apply policy/guardrails.
    """

    id: str = Field(
        pattern=r"^(V|PF)\d{2}$",
        description="Check ID (V01-V10 for canonical, PF01-PF99 for pre-filters)",
    )
    name: str = Field(description="Human-readable check name")
    passed: bool | None = Field(
        description="Metrics-only mode: always None. Host applications may apply policy/guardrails."
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in result")
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        """Check if result has high confidence (>0.8)."""
        return self.confidence > 0.8


class Detection(FrozenModel):
    """Complete detection wrapper combining candidate with metadata.

    This is the output of the detection pipeline before host-side interpretation.
    """

    candidate: TransitCandidate
    data_ref: str = Field(description="Reference to source light curve")
    method: Literal["bls", "ls", "auto"] = "bls"
    rank: int = Field(ge=1, description="Rank among candidates (1 = best)")


class PeriodogramResult(FrozenModel):
    """Complete periodogram analysis result.

    This is the response from astro_periodogram tool.
    """

    data_ref: str
    method: Literal["bls", "tls", "ls"]
    signal_type: Literal["transit", "sinusoidal"] = Field(
        description="Semantic type of the detected signal; LS is for sinusoidal rotation/variability, TLS is for transits."
    )
    peaks: list[PeriodogramPeak]
    best_period: PeriodDays
    best_t0: float
    best_duration_hours: DurationHours | None = None
    snr: SNR | None = None
    fap: FAP | None = None
    n_periods_searched: int = Field(ge=0)  # 0 for TLS (auto-generates grid)
    period_range: tuple[float, float]

    @model_validator(mode="after")
    def _validate_n_periods_searched(self) -> PeriodogramResult:
        # For BLS/LS we always search an explicit period grid, so 0 is nonsensical.
        # TLS can generate its own adaptive grid and may not expose the effective count.
        if self.method != "tls" and self.n_periods_searched < 1:
            raise ValueError("n_periods_searched must be >= 1 for non-TLS methods")
        return self

    @property
    def top_peak(self) -> PeriodogramPeak | None:
        """Get the highest-power peak."""
        if not self.peaks:
            return None
        return self.peaks[0]
