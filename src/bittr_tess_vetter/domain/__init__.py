"""Domain models for bittr-tess-vetter.

This package is domain-only. It intentionally excludes platform-layer concepts
like manifests, stores, tool frameworks, etc.
"""

from bittr_tess_vetter.domain.detection import (
    Detection,
    PeriodogramPeak,
    PeriodogramResult,
    TransitCandidate,
    VetterCheckResult,
)
from bittr_tess_vetter.domain.lightcurve import LightCurveData, make_data_ref
from bittr_tess_vetter.domain.target import StellarParameters, Target

__all__ = [
    "LightCurveData",
    "make_data_ref",
    "StellarParameters",
    "Target",
    "Detection",
    "PeriodogramPeak",
    "PeriodogramResult",
    "TransitCandidate",
    "VetterCheckResult",
]
