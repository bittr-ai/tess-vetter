"""WCS utility helpers for the public API.

Delegates to `bittr_tess_vetter.pixel.wcs_utils`.

References:
    - Greisen & Calabretta 2002 (2002A&A...395.1061G): FITS WCS framework (Paper I)
    - Calabretta & Greisen 2002 (2002A&A...395.1077C): celestial WCS conventions (Paper II)
    - Astropy Collaboration 2013 (2013A&A...558A..33A): astropy.wcs implementation
"""

from __future__ import annotations

from bittr_tess_vetter.api.references import (
    ASTROPY_COLLAB_2013,
    CALABRETTA_GREISEN_2002,
    GREISEN_CALABRETTA_2002,
)
from bittr_tess_vetter.pixel.wcs_utils import (
    compute_angular_distance,
    compute_pixel_scale,
    compute_source_distances,
    compute_wcs_checksum,
    extract_wcs_from_header,
    get_reference_source_pixel_positions,
    get_stamp_center,
    get_stamp_center_world,
    get_target_pixel_position,
    pixel_to_world,
    pixel_to_world_batch,
    verify_wcs_checksum,
    wcs_sanity_check,
    world_to_pixel,
    world_to_pixel_batch,
)

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [
    ref.to_dict()
    for ref in [
        GREISEN_CALABRETTA_2002,
        CALABRETTA_GREISEN_2002,
        ASTROPY_COLLAB_2013,
    ]
]

__all__ = [
    "compute_angular_distance",
    "compute_pixel_scale",
    "compute_source_distances",
    "compute_wcs_checksum",
    "extract_wcs_from_header",
    "get_reference_source_pixel_positions",
    "get_stamp_center",
    "get_stamp_center_world",
    "get_target_pixel_position",
    "pixel_to_world",
    "pixel_to_world_batch",
    "verify_wcs_checksum",
    "wcs_sanity_check",
    "world_to_pixel",
    "world_to_pixel_batch",
]
