"""Aperture-family depth curve utilities for the public API.

Delegates to `bittr_tess_vetter.pixel.aperture_family`.
"""

from __future__ import annotations

from bittr_tess_vetter.pixel.aperture_family import (  # noqa: F401
    DEFAULT_RADII_PX,
    compute_aperture_family_depth_curve,
)

__all__ = ["DEFAULT_RADII_PX", "compute_aperture_family_depth_curve"]

