"""WCS utilities for pixel-level analysis.

This module provides utilities for working with World Coordinate System (WCS)
transformations in TESS Target Pixel Files:

- WCS extraction from FITS headers
- WCS checksum computation for validation
- World (RA/Dec) to pixel coordinate transforms
- Pixel to world coordinate transforms
- Target pixel position lookup

Usage:
    from bittr_tess_vetter.pixel.wcs_utils import (
        extract_wcs_from_header,
        compute_wcs_checksum,
        world_to_pixel,
        pixel_to_world,
        get_target_pixel_position,
    )

    # Get pixel position for a sky coordinate
    row, col = world_to_pixel(wcs, ra_deg=120.5, dec_deg=-50.2)

    # Get sky coordinates for a pixel position
    ra, dec = pixel_to_world(wcs, row=5.5, col=5.5)

References:
    - Greisen & Calabretta 2002 (2002A&A...395.1061G): FITS WCS framework (Paper I)
    - Calabretta & Greisen 2002 (2002A&A...395.1077C): celestial WCS conventions (Paper II)
    - Astropy Collaboration 2013 (2013A&A...558A..33A): astropy.wcs implementation
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


def extract_wcs_from_header(header: fits.Header) -> WCS:
    """Extract WCS from a FITS header.

    Args:
        header: FITS header containing WCS keywords.

    Returns:
        Astropy WCS object.

    Raises:
        ValueError: If the header does not contain valid WCS information.
    """
    try:
        wcs = WCS(header)
        # Validate that we have a usable WCS
        if not wcs.has_celestial:
            raise ValueError("Header does not contain celestial WCS information")
        return wcs
    except Exception as e:
        raise ValueError(f"Failed to extract WCS from header: {e}") from e


def compute_wcs_checksum(wcs: WCS) -> str:
    """Compute a SHA-256 checksum for WCS validation.

    The checksum is computed from the WCS header string representation
    to detect any changes to the WCS transformation.

    Args:
        wcs: Astropy WCS object.

    Returns:
        SHA-256 checksum string prefixed with 'sha256:'.
    """
    try:
        header_str = wcs.to_header_string()
        checksum = hashlib.sha256(header_str.encode("utf-8")).hexdigest()
        return f"sha256:{checksum}"
    except Exception as e:
        logger.warning("Failed to compute WCS checksum: %s", e)
        return "sha256:unknown"


def verify_wcs_checksum(wcs: WCS, expected_checksum: str) -> bool:
    """Verify that a WCS matches an expected checksum.

    Args:
        wcs: Astropy WCS object to verify.
        expected_checksum: Expected checksum string.

    Returns:
        True if the checksum matches, False otherwise.
    """
    actual = compute_wcs_checksum(wcs)
    return actual == expected_checksum


def world_to_pixel(
    wcs: WCS,
    ra_deg: float,
    dec_deg: float,
    origin: int = 0,
) -> tuple[float, float]:
    """Convert world coordinates (RA/Dec) to pixel coordinates.

    Args:
        wcs: Astropy WCS object.
        ra_deg: Right Ascension in degrees.
        dec_deg: Declination in degrees.
        origin: Pixel coordinate origin (0 for 0-indexed, 1 for 1-indexed).

    Returns:
        Tuple of (row, col) pixel coordinates (float).

    Note:
        Returns (row, col) which corresponds to (y, x) in array indexing.
        The WCS standard uses (x, y) = (col, row) ordering.
    """
    # WCS uses (x, y) = (col, row) ordering
    # all_world2pix returns (x, y) arrays
    result = wcs.all_world2pix([[ra_deg, dec_deg]], origin)
    col, row = float(result[0, 0]), float(result[0, 1])
    return (row, col)


def world_to_pixel_batch(
    wcs: WCS,
    ra_deg: np.ndarray[Any, np.dtype[np.floating[Any]]],
    dec_deg: np.ndarray[Any, np.dtype[np.floating[Any]]],
    origin: int = 0,
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating[Any]]], np.ndarray[Any, np.dtype[np.floating[Any]]]
]:
    """Convert arrays of world coordinates to pixel coordinates.

    Args:
        wcs: Astropy WCS object.
        ra_deg: Array of Right Ascension values in degrees.
        dec_deg: Array of Declination values in degrees.
        origin: Pixel coordinate origin (0 for 0-indexed, 1 for 1-indexed).

    Returns:
        Tuple of (rows, cols) arrays of pixel coordinates.
    """
    coords = np.column_stack([ra_deg, dec_deg])
    result = wcs.all_world2pix(coords, origin)
    cols = result[:, 0]
    rows = result[:, 1]
    return (rows.astype(np.float64), cols.astype(np.float64))


def pixel_to_world(
    wcs: WCS,
    row: float,
    col: float,
    origin: int = 0,
) -> tuple[float, float]:
    """Convert pixel coordinates to world coordinates (RA/Dec).

    Args:
        wcs: Astropy WCS object.
        row: Row (y) pixel coordinate.
        col: Column (x) pixel coordinate.
        origin: Pixel coordinate origin (0 for 0-indexed, 1 for 1-indexed).

    Returns:
        Tuple of (ra_deg, dec_deg) sky coordinates in degrees.
    """
    # WCS uses (x, y) = (col, row) ordering
    result = wcs.all_pix2world([[col, row]], origin)
    ra_deg, dec_deg = float(result[0, 0]), float(result[0, 1])
    return (ra_deg, dec_deg)


def pixel_to_world_batch(
    wcs: WCS,
    rows: np.ndarray[Any, np.dtype[np.floating[Any]]],
    cols: np.ndarray[Any, np.dtype[np.floating[Any]]],
    origin: int = 0,
) -> tuple[
    np.ndarray[Any, np.dtype[np.floating[Any]]], np.ndarray[Any, np.dtype[np.floating[Any]]]
]:
    """Convert arrays of pixel coordinates to world coordinates.

    Args:
        wcs: Astropy WCS object.
        rows: Array of row (y) pixel coordinates.
        cols: Array of column (x) pixel coordinates.
        origin: Pixel coordinate origin (0 for 0-indexed, 1 for 1-indexed).

    Returns:
        Tuple of (ra_deg, dec_deg) arrays of sky coordinates in degrees.
    """
    # WCS uses (x, y) = (col, row) ordering
    coords = np.column_stack([cols, rows])
    result = wcs.all_pix2world(coords, origin)
    ra_deg = result[:, 0]
    dec_deg = result[:, 1]
    return (ra_deg.astype(np.float64), dec_deg.astype(np.float64))


def get_target_pixel_position(
    wcs: WCS,
    target_ra_deg: float,
    target_dec_deg: float,
    stamp_shape: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Get the pixel position of the target within a TPF stamp.

    Args:
        wcs: Astropy WCS object from the TPF.
        target_ra_deg: Target Right Ascension in degrees.
        target_dec_deg: Target Declination in degrees.
        stamp_shape: Optional (n_rows, n_cols) to validate position is within stamp.

    Returns:
        Tuple of (row, col) pixel coordinates (0-indexed).

    Raises:
        ValueError: If the target position is outside the stamp (when stamp_shape provided).
    """
    row, col = world_to_pixel(wcs, target_ra_deg, target_dec_deg, origin=0)

    if stamp_shape is not None:
        n_rows, n_cols = stamp_shape
        if not (0 <= row < n_rows and 0 <= col < n_cols):
            raise ValueError(
                f"Target position ({row:.2f}, {col:.2f}) is outside stamp "
                f"boundaries (0-{n_rows - 1}, 0-{n_cols - 1})"
            )

    return (row, col)


def compute_angular_distance(
    ra1_deg: float,
    dec1_deg: float,
    ra2_deg: float,
    dec2_deg: float,
) -> float:
    """Compute the angular distance between two sky positions.

    Uses the Haversine formula for accurate spherical distance.

    Args:
        ra1_deg: Right Ascension of first position in degrees.
        dec1_deg: Declination of first position in degrees.
        ra2_deg: Right Ascension of second position in degrees.
        dec2_deg: Declination of second position in degrees.

    Returns:
        Angular distance in arcseconds.
    """
    # Convert to radians
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)

    # Haversine formula
    dra = ra2 - ra1
    ddec = dec2 - dec1

    a = np.sin(ddec / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2) ** 2
    # Guard against small floating-point drift pushing `a` marginally outside [0, 1].
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))

    # Convert radians to arcseconds
    distance_arcsec = float(np.degrees(c) * 3600)

    return distance_arcsec


def compute_pixel_scale(wcs: WCS) -> float:
    """Compute the pixel scale from WCS.

    Args:
        wcs: Astropy WCS object.

    Returns:
        Pixel scale in arcseconds per pixel.
    """
    # Get the pixel scale from the WCS
    # Use the CD matrix if available, otherwise CDELT
    try:
        # proj_plane_pixel_scales returns Quantity objects in degrees/pixel
        scales = wcs.proj_plane_pixel_scales()
        # Extract numeric values (handle both Quantity and plain float)
        scale_values = []
        for s in scales:
            if hasattr(s, "value"):
                scale_values.append(float(s.value))
            else:
                scale_values.append(float(s))
        # Average the two scales and convert to arcsec
        pixel_scale_arcsec = float(np.mean(scale_values) * 3600)
        return pixel_scale_arcsec
    except Exception:
        # Fallback: TESS nominal pixel scale
        return 21.0  # arcseconds per pixel


def get_stamp_center(stamp_shape: tuple[int, int]) -> tuple[float, float]:
    """Get the center pixel coordinates of a stamp.

    Args:
        stamp_shape: (n_rows, n_cols) shape of the stamp.

    Returns:
        Tuple of (row, col) center coordinates (0-indexed, can be fractional).
    """
    n_rows, n_cols = stamp_shape
    center_row = (n_rows - 1) / 2.0
    center_col = (n_cols - 1) / 2.0
    return (center_row, center_col)


def get_stamp_center_world(wcs: WCS, stamp_shape: tuple[int, int]) -> tuple[float, float]:
    """Get the world coordinates (RA/Dec) of the stamp center.

    Args:
        wcs: Astropy WCS object.
        stamp_shape: (n_rows, n_cols) shape of the stamp.

    Returns:
        Tuple of (ra_deg, dec_deg) center coordinates.
    """
    center_row, center_col = get_stamp_center(stamp_shape)
    return pixel_to_world(wcs, center_row, center_col, origin=0)


def compute_source_distances(
    centroid_ra_deg: float,
    centroid_dec_deg: float,
    reference_sources: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute angular distances from a centroid to reference sources.

    Args:
        centroid_ra_deg: Centroid Right Ascension in degrees.
        centroid_dec_deg: Centroid Declination in degrees.
        reference_sources: List of dicts with 'name', 'ra', 'dec' keys.
            RA and Dec should be in degrees.

    Returns:
        Dictionary mapping source names to distances in arcseconds.
    """
    distances: dict[str, float] = {}
    for source in reference_sources:
        name = source.get("name", "unknown")
        ra = source.get("ra")
        dec = source.get("dec")
        if ra is not None and dec is not None:
            dist = compute_angular_distance(
                centroid_ra_deg, centroid_dec_deg, float(ra), float(dec)
            )
            distances[name] = dist
        else:
            logger.warning("Source %s missing ra/dec coordinates", name)
    return distances


def get_reference_source_pixel_positions(
    wcs: WCS,
    reference_sources: list[dict[str, Any]],
    origin: int = 0,
) -> dict[str, tuple[float, float]]:
    """Get pixel positions of reference sources.

    Args:
        wcs: Astropy WCS object.
        reference_sources: List of dicts with 'name', 'ra', 'dec' keys.
        origin: Pixel coordinate origin (0 for 0-indexed).

    Returns:
        Dictionary mapping source names to (row, col) pixel positions.
    """
    positions: dict[str, tuple[float, float]] = {}
    for source in reference_sources:
        name = source.get("name", "unknown")
        ra = source.get("ra")
        dec = source.get("dec")
        if ra is not None and dec is not None:
            row, col = world_to_pixel(wcs, float(ra), float(dec), origin=origin)
            positions[name] = (row, col)
        else:
            logger.warning("Source %s missing ra/dec coordinates", name)
    return positions


def wcs_sanity_check(
    wcs: WCS,
    expected_ra_deg: float | None = None,
    expected_dec_deg: float | None = None,
    stamp_shape: tuple[int, int] | None = None,
    tolerance_arcsec: float = 60.0,
) -> tuple[bool, list[str]]:
    """Perform sanity checks on a WCS transformation.

    Args:
        wcs: Astropy WCS object to check.
        expected_ra_deg: Expected RA at stamp center (optional).
        expected_dec_deg: Expected Dec at stamp center (optional).
        stamp_shape: Shape of the stamp for center calculation.
        tolerance_arcsec: Maximum allowed offset in arcseconds.

    Returns:
        Tuple of (is_valid, warnings_list).
    """
    warnings: list[str] = []
    is_valid = True

    # Check if WCS has celestial component
    if not wcs.has_celestial:
        warnings.append("WCS lacks celestial coordinates")
        is_valid = False
        return (is_valid, warnings)

    # Check pixel scale is reasonable for TESS (should be ~21 arcsec/pixel)
    try:
        pixel_scale = compute_pixel_scale(wcs)
        if not (15 < pixel_scale < 30):  # Reasonable range for TESS
            warnings.append(
                f"Unusual pixel scale: {pixel_scale:.1f} arcsec/pixel (expected ~21 for TESS)"
            )
    except Exception as e:
        warnings.append(f"Could not compute pixel scale: {e}")

    # Check center position if expected coordinates provided
    if expected_ra_deg is not None and expected_dec_deg is not None and stamp_shape is not None:
        try:
            center_ra, center_dec = get_stamp_center_world(wcs, stamp_shape)
            offset = compute_angular_distance(
                center_ra, center_dec, expected_ra_deg, expected_dec_deg
            )
            if offset > tolerance_arcsec:
                warnings.append(
                    f"WCS center offset {offset:.1f} arcsec exceeds tolerance "
                    f"({tolerance_arcsec:.1f} arcsec)"
                )
                is_valid = False
        except Exception as e:
            warnings.append(f"Could not verify center position: {e}")

    return (is_valid, warnings)
