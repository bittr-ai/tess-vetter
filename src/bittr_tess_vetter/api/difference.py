"""Difference-imaging utilities for the public API.

Delegates to `bittr_tess_vetter.pixel.difference`.
"""

from __future__ import annotations

from bittr_tess_vetter.pixel.difference import (
    DifferenceImageResult,
    TransitParams,
    compute_difference_image,
)

__all__ = ["DifferenceImageResult", "TransitParams", "compute_difference_image"]

