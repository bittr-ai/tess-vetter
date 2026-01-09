"""Detection/periodogram domain models (public API).

These models are part of the stable contract between the bittr-tess-vetter
library and host applications (e.g., MCP servers).
"""

from __future__ import annotations

from bittr_tess_vetter.domain.detection import (  # noqa: F401
    Detection,
    Disposition,
    PeriodogramPeak,
    PeriodogramResult,
    TransitCandidate,
    ValidationResult,
    Verdict,
    VetterCheckResult,
)

__all__ = [
    "Detection",
    "Disposition",
    "PeriodogramPeak",
    "PeriodogramResult",
    "TransitCandidate",
    "ValidationResult",
    "Verdict",
    "VetterCheckResult",
]

