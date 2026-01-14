"""Centroid-shift utilities for the public API.

Delegates to `bittr_tess_vetter.pixel.centroid`.
"""

from __future__ import annotations

from bittr_tess_vetter.api.references import (
    BRYSON_2010,
    BRYSON_2013,
    HIGGINS_BELL_2022,
    cite,
    cites,
)
from bittr_tess_vetter.pixel.centroid import (
    WINDOW_POLICIES,
    CentroidResult,
    TransitParams,
    _compute_flux_weighted_centroid,
    _get_transit_masks,
    compute_centroid_shift,
)

_compute_flux_weighted_centroid = cites(
    cite(BRYSON_2010, "Flux-weighted centroiding / PRF lineage"),
    cite(BRYSON_2013, "Centroid-based background false positive diagnostics"),
)(  # type: ignore[assignment]
    _compute_flux_weighted_centroid
)

compute_centroid_shift = cites(
    cite(BRYSON_2013, "Difference in/out-of-transit centroids for localization"),
    cite(HIGGINS_BELL_2022, "TESS-specific localization in crowded fields"),
)(compute_centroid_shift)

__all__ = [
    "WINDOW_POLICIES",
    "CentroidResult",
    "TransitParams",
    "_compute_flux_weighted_centroid",
    "_get_transit_masks",
    "compute_centroid_shift",
]
