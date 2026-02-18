"""Reusable contrast-curve parsing, normalization, and summary primitives."""

from __future__ import annotations

from bittr_tess_vetter.contrast_curves.combine import combine_normalized_curves
from bittr_tess_vetter.contrast_curves.exceptions import ContrastCurveParseError
from bittr_tess_vetter.contrast_curves.models import NormalizedContrastCurve, RulingSummary
from bittr_tess_vetter.contrast_curves.normalize import (
    MAX_RELEVANT_SEPARATION_ARCSEC,
    normalize_contrast_curve,
)
from bittr_tess_vetter.contrast_curves.parsing import (
    parse_contrast_curve_file,
    parse_contrast_curve_with_provenance,
)
from bittr_tess_vetter.contrast_curves.summary import (
    build_ruling_summary,
    derive_contrast_verdict,
)

__all__ = [
    "ContrastCurveParseError",
    "MAX_RELEVANT_SEPARATION_ARCSEC",
    "NormalizedContrastCurve",
    "RulingSummary",
    "build_ruling_summary",
    "combine_normalized_curves",
    "derive_contrast_verdict",
    "normalize_contrast_curve",
    "parse_contrast_curve_file",
    "parse_contrast_curve_with_provenance",
]
