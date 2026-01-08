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
    cite,
    cites,
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

# Attach citations to the core WCS entrypoints (no wrapping; adds __references__ metadata).
extract_wcs_from_header = cites(
    cite(GREISEN_CALABRETTA_2002, "FITS WCS framework"),
    cite(CALABRETTA_GREISEN_2002, "Celestial coordinate WCS conventions"),
    cite(ASTROPY_COLLAB_2013, "astropy.wcs implementation"),
)(extract_wcs_from_header)

world_to_pixel = cites(
    cite(GREISEN_CALABRETTA_2002, "World→pixel transform"),
    cite(CALABRETTA_GREISEN_2002, "Celestial WCS projections"),
    cite(ASTROPY_COLLAB_2013, "astropy.wcs implementation"),
)(world_to_pixel)

pixel_to_world = cites(
    cite(GREISEN_CALABRETTA_2002, "Pixel→world transform"),
    cite(CALABRETTA_GREISEN_2002, "Celestial WCS projections"),
    cite(ASTROPY_COLLAB_2013, "astropy.wcs implementation"),
)(pixel_to_world)

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
