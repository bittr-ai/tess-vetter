"""Pixel vetting report utilities for the public API.

Delegates to `bittr_tess_vetter.pixel.report`.
"""

from __future__ import annotations

from bittr_tess_vetter.pixel.report import (
    THRESHOLD_VERSIONS,
    PixelVetReport,
    generate_pixel_vet_report,
)
from bittr_tess_vetter.api.references import BRYSON_2013, HIGGINS_BELL_2022, TWICKEN_2018, cite, cites

# Attach citations to the API surface callable (no wrapping; adds __references__ metadata).
generate_pixel_vet_report = cites(
    cite(BRYSON_2013, "Pixel-level diagnostics (difference images/centroids) for background false positives"),
    cite(TWICKEN_2018, "Kepler DV pixel-level diagnostic products and interpretation"),
    cite(HIGGINS_BELL_2022, "TESS-specific localization considerations in crowded photometry"),
)(generate_pixel_vet_report)

__all__ = ["THRESHOLD_VERSIONS", "PixelVetReport", "generate_pixel_vet_report"]
