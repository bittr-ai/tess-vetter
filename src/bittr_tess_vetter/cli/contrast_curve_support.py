"""Backward-compatible CLI shim for reusable contrast-curve primitives."""

from __future__ import annotations

from bittr_tess_vetter.contrast_curves import (
    ContrastCurveParseError,
    build_ruling_summary,
    combine_normalized_curves,
    derive_contrast_verdict,
    normalize_contrast_curve,
    parse_contrast_curve_file,
    parse_contrast_curve_with_provenance,
)

__all__ = [
    "ContrastCurveParseError",
    "build_ruling_summary",
    "combine_normalized_curves",
    "derive_contrast_verdict",
    "normalize_contrast_curve",
    "parse_contrast_curve_file",
    "parse_contrast_curve_with_provenance",
]
