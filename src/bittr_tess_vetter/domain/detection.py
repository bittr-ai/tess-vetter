"""Detection and validation domain models.

This module provides:
- PeriodogramPeak: Individual peak from periodogram analysis
- TransitCandidate: Candidate transit parameters for vetting
- VetterCheckResult: Result of a single vetting check
- ValidationResult: Complete validation output with verdict
- Detection: Wrapper combining candidate with metadata
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Verdict(str, Enum):
    """Aggregated vetting outcome."""

    PASS = "PASS"  # All checks passed
    WARN = "WARN"  # Some checks marginal
    REJECT = "REJECT"  # One or more checks failed


class Disposition(str, Enum):
    """Final classification for a candidate."""

    PLANET = "PLANET"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    UNCERTAIN = "UNCERTAIN"


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

    @property
    def is_significant(self) -> bool:
        """Check if peak meets basic significance threshold."""
        if self.fap is not None:
            return self.fap < 0.01
        if self.snr is not None:
            return self.snr > 7.0
        return self.power > 0  # Fallback


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

    Each check produces one of these with pass/fail status,
    confidence, and check-specific details.

    Check IDs:
    - V01-V10: Canonical vetting checks (see lc_checks.py)
    - PF01-PF99: Pre-filter checks (optional, run before V01-V10)

    Passed field:
    - True: Check passed (candidate is consistent with planet)
    - False: Check failed (candidate shows signs of false positive)
    - None: Metrics-only mode (policy decision deferred to caller)
      When passed=None, the check returns raw metrics and the caller
      (e.g., astro-arc-tess guardrails) makes the policy decision.
    """

    id: str = Field(
        pattern=r"^(V|PF)\d{2}$",
        description="Check ID (V01-V10 for canonical, PF01-PF99 for pre-filters)",
    )
    name: str = Field(description="Human-readable check name")
    passed: bool | None = Field(description="Pass/fail status. None indicates metrics-only mode.")
    confidence: float = Field(ge=0, le=1, description="Confidence in result")
    details: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_high_confidence(self) -> bool:
        """Check if result has high confidence (>0.8)."""
        return self.confidence > 0.8


class ValidationResult(FrozenModel):
    """Complete validation output with aggregated verdict.

    Contains results from all vetting checks and final disposition.
    """

    disposition: Disposition
    verdict: Verdict
    checks: list[VetterCheckResult]
    summary: str = Field(description="Human-readable summary")

    @property
    def n_passed(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.passed is True)

    @property
    def n_failed(self) -> int:
        """Number of checks that failed."""
        return sum(1 for c in self.checks if c.passed is False)

    @property
    def failed_checks(self) -> list[str]:
        """IDs of failed checks."""
        return [c.id for c in self.checks if c.passed is False]

    @property
    def n_unknown(self) -> int:
        """Number of checks that returned metrics-only results (passed=None)."""
        return sum(1 for c in self.checks if c.passed is None)

    @property
    def unknown_checks(self) -> list[str]:
        """IDs of checks that returned metrics-only results (passed=None)."""
        return [c.id for c in self.checks if c.passed is None]


class Detection(FrozenModel):
    """Complete detection wrapper combining candidate with metadata.

    This is the output of the detection pipeline before validation.
    """

    candidate: TransitCandidate
    data_ref: str = Field(description="Reference to source light curve")
    method: Literal["bls", "ls", "auto"] = "bls"
    rank: int = Field(ge=1, description="Rank among candidates (1 = best)")
    validation: ValidationResult | None = None

    @property
    def is_validated(self) -> bool:
        """Check if detection has been validated."""
        return self.validation is not None

    @property
    def is_planet_candidate(self) -> bool:
        """Check if validation passed as planet candidate."""
        if self.validation is None:
            return False
        return self.validation.disposition == Disposition.PLANET


class PeriodogramResult(FrozenModel):
    """Complete periodogram analysis result.

    This is the response from astro_periodogram tool.
    """

    data_ref: str
    method: Literal["bls", "tls", "ls"]
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
