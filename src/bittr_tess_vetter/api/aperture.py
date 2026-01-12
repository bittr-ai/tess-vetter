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


def create_circular_aperture_mask(
    shape: tuple[int, int],
    *,
    center_row: float,
    center_col: float,
    radius_px: float,
):
    """Create a circular aperture mask.

    This is a stable, JSON-friendly public wrapper around the internal
    `_create_circular_aperture_mask(shape, radius, center)` helper.
    """
    return _create_circular_aperture_mask(
        shape=shape,
        radius=float(radius_px),
        center=(float(center_row), float(center_col)),
    )


compute_aperture_dependence = cites(
    cite(BRYSON_2013, "Aperture-dependent behavior as a contamination diagnostic"),
    cite(TORRES_2011, "Blend scenario interpretation and rejection"),
)(compute_aperture_dependence)

__all__ = [
    "DEFAULT_APERTURE_RADII",
    "ApertureDependenceResult",
    "TransitParams",
    "create_circular_aperture_mask",
    "_compute_depth_from_lightcurve",
    "_compute_stability_metric",
    "_compute_transit_mask",
    "_create_circular_aperture_mask",
    "_select_recommended_aperture",
    "compute_aperture_dependence",
]
