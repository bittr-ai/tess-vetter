"""Aperture-dependence utilities for the public API.

Delegates to `bittr_tess_vetter.pixel.aperture`.
"""

from __future__ import annotations

from bittr_tess_vetter.api.references import BRYSON_2013, TORRES_2011, cite, cites
from bittr_tess_vetter.pixel.aperture import (
    DEFAULT_APERTURE_RADII,
    ApertureDependenceResult,
    TransitParams,
    _compute_depth_from_lightcurve,
    _compute_stability_metric,
    _compute_transit_mask,
    _create_circular_aperture_mask,
    _select_recommended_aperture,
    compute_aperture_dependence,
)

compute_aperture_dependence = cites(
    cite(BRYSON_2013, "Aperture-dependent behavior as a contamination diagnostic"),
    cite(TORRES_2011, "Blend scenario interpretation and rejection"),
)(compute_aperture_dependence)

__all__ = [
    "DEFAULT_APERTURE_RADII",
    "ApertureDependenceResult",
    "TransitParams",
    "_compute_depth_from_lightcurve",
    "_compute_stability_metric",
    "_compute_transit_mask",
    "_create_circular_aperture_mask",
    "_select_recommended_aperture",
    "compute_aperture_dependence",
]
