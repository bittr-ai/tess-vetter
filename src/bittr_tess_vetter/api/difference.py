"""Difference-imaging utilities for the public API.

Delegates to `bittr_tess_vetter.pixel.difference`.
"""

from __future__ import annotations

from bittr_tess_vetter.api.references import BRYSON_2013, TWICKEN_2018, cite, cites
from bittr_tess_vetter.pixel.difference import (
    DifferenceImageResult,
    TransitParams,
    compute_difference_image,
)

compute_difference_image = cites(
    cite(BRYSON_2013, "Difference imaging for transit source localization"),
    cite(TWICKEN_2018, "Kepler DV difference-image diagnostics context"),
)(compute_difference_image)

__all__ = ["DifferenceImageResult", "TransitParams", "compute_difference_image"]
