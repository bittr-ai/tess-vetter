"""Ghost/scattered-light feature extraction for the public API.

Re-exports metrics-only pixel feature computations from
`bittr_tess_vetter.validation.ghost_features`.
"""

from __future__ import annotations

from bittr_tess_vetter.validation.ghost_features import (  # noqa: F401
    GhostFeatures,
    compute_aperture_contrast,
    compute_difference_image,
    compute_edge_gradient,
    compute_ghost_features,
    compute_prf_likeness,
    compute_spatial_uniformity,
)

__all__ = [
    "GhostFeatures",
    "compute_aperture_contrast",
    "compute_difference_image",
    "compute_edge_gradient",
    "compute_ghost_features",
    "compute_prf_likeness",
    "compute_spatial_uniformity",
]

