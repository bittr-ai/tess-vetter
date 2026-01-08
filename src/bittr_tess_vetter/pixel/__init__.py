"""Pixel-level data handling for TESS Target Pixel Files (TPF).

This module provides utilities for managing TPF data including:
- TPFRef: Reference format for addressing TPF data by TIC ID, sector, camera, and CCD.
- TPFCache: Session-scoped cache for TPF data arrays.
- TPFHandler: Abstract handler for TPF acquisition.
- CachedTPFHandler: Handler that wraps another handler with caching.
- TPFNotFoundError: Error raised when TPF data is not available.

FITS-preserving TPF handling with WCS support:
- TPFFitsRef: Reference format for FITS-cached TPF with WCS preserved.
- TPFFitsData: Full TPF data including WCS, aperture mask, quality flags.
- TPFFitsCache: Disk cache with FITS files and JSON sidecar metadata.
- TPFFitsNotFoundError: Error raised when FITS TPF data is not available.

WCS utilities:
- world_to_pixel: Convert RA/Dec to pixel coordinates.
- pixel_to_world: Convert pixel coordinates to RA/Dec.
- compute_wcs_checksum: Compute SHA-256 checksum for WCS validation.
- compute_angular_distance: Compute angular distance between positions.
- compute_source_distances: Compute distances from centroid to reference sources.

Aperture analysis tools:
- ApertureDependenceResult: Results from aperture dependence analysis.
- TransitParams: Parameters describing a transit signal.
- compute_aperture_dependence: Analyze transit depth vs aperture size.

Centroid analysis tools:
- CentroidResult: Results from centroid shift analysis.
- compute_centroid_shift: Compute centroid shift between in/out of transit.

Difference imaging tools:
- DifferenceImageResult: Results from difference image analysis.
- compute_difference_image: Compute difference image for transit localization.

Pixel vetting report:
- PixelVetReport: Combined pixel vetting report with pass/fail determination.
- generate_pixel_vet_report: Generate pixel vetting report from analysis results.
- THRESHOLD_VERSIONS: Versioned threshold configurations.

TPF Reference Formats:
    tpf:<tic_id>:<sector>:<camera>:<ccd>         (npz cache)
    tpf_fits:<tic_id>:<sector>:<author>          (FITS cache with WCS)

Example:
    from bittr_tess_vetter.pixel import TPFRef, TPFCache, TPFFitsRef, TPFFitsCache

    # Parse a reference (npz format)
    ref = TPFRef.from_string("tpf:123456789:15:2:3")
    print(ref.tic_id)  # 123456789

    # Parse a FITS reference (WCS preserved)
    fits_ref = TPFFitsRef.from_string("tpf_fits:123456789:15:spoc")
    fits_cache = TPFFitsCache(cache_dir=Path("/tmp/tpf_fits_cache"))
    if fits_cache.has(fits_ref):
        data = fits_cache.get(fits_ref)
        print(data.wcs)  # astropy WCS object
"""

from __future__ import annotations

from bittr_tess_vetter.pixel.aperture import (
    ApertureDependenceResult,
    TransitParams,
    compute_aperture_dependence,
)
from bittr_tess_vetter.pixel.centroid import (
    CentroidResult,
    compute_centroid_shift,
)
from bittr_tess_vetter.pixel.difference import (
    DifferenceImageResult,
    compute_difference_image,
)
from bittr_tess_vetter.pixel.report import (
    THRESHOLD_VERSIONS,
    PixelVetReport,
    generate_pixel_vet_report,
)
from bittr_tess_vetter.pixel.tpf import (
    CachedTPFHandler,
    TPFCache,
    TPFData,
    TPFHandler,
    TPFNotFoundError,
    TPFRef,
)
from bittr_tess_vetter.pixel.tpf_fits import (
    VALID_AUTHORS,
    TPFFitsCache,
    TPFFitsData,
    TPFFitsNotFoundError,
    TPFFitsRef,
)
from bittr_tess_vetter.pixel.wcs_localization import (
    LocalizationResult,
    LocalizationVerdict,
    bootstrap_centroid_uncertainty,
    compute_difference_image_centroid,
    compute_reference_source_distances,
    localize_transit_source,
)
from bittr_tess_vetter.pixel.wcs_utils import (
    compute_angular_distance,
    compute_pixel_scale,
    compute_source_distances,
    compute_wcs_checksum,
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

__all__ = [
    # Aperture analysis
    "ApertureDependenceResult",
    "TransitParams",
    "compute_aperture_dependence",
    # Centroid analysis
    "CentroidResult",
    "compute_centroid_shift",
    # Difference imaging
    "DifferenceImageResult",
    "compute_difference_image",
    # Pixel vetting report
    "PixelVetReport",
    "THRESHOLD_VERSIONS",
    "generate_pixel_vet_report",
    # TPF handling (npz cache)
    "CachedTPFHandler",
    "TPFCache",
    "TPFData",
    "TPFHandler",
    "TPFNotFoundError",
    "TPFRef",
    # TPF FITS handling (WCS preserved)
    "TPFFitsCache",
    "TPFFitsData",
    "TPFFitsNotFoundError",
    "TPFFitsRef",
    "VALID_AUTHORS",
    # WCS-aware localization
    "LocalizationResult",
    "LocalizationVerdict",
    "bootstrap_centroid_uncertainty",
    "compute_difference_image_centroid",
    "compute_reference_source_distances",
    "localize_transit_source",
    # WCS utilities
    "compute_angular_distance",
    "compute_pixel_scale",
    "compute_source_distances",
    "compute_wcs_checksum",
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
