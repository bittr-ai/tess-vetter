"""Aperture-family depth curve utilities for the public API.

Delegates to `tess_vetter.pixel.aperture_family`.
"""

from __future__ import annotations

from tess_vetter.api.references import BRYSON_2013, TORRES_2011, TWICKEN_2018, cite, cites
from tess_vetter.pixel.aperture_family import (  # noqa: F401
    DEFAULT_RADII_PX,
    SLOPE_SIGNIFICANCE_THRESHOLD,
    ApertureFamilyResult,
    compute_aperture_family_depth_curve,
)

compute_aperture_family_depth_curve = cites(
    cite(TWICKEN_2018, "DV-style pixel-level contamination diagnostics context"),
    cite(BRYSON_2013, "Pixel-level background false positive diagnostics"),
    cite(TORRES_2011, "Blend scenario interpretation"),
)(compute_aperture_family_depth_curve)

__all__ = [
    "ApertureFamilyResult",
    "DEFAULT_RADII_PX",
    "SLOPE_SIGNIFICANCE_THRESHOLD",
    "compute_aperture_family_depth_curve",
]
