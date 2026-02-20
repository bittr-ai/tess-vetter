"""Synthetic TPF cube generators for testing WCS-aware localization.

This module provides functions to generate synthetic Target Pixel File (TPF) data
with known star positions, transits, and WCS transformations. These are used
to test localization algorithms with ground truth.

Key generators:
- make_synthetic_tpf_fits: General-purpose synthetic TPF with configurable stars
- make_blended_binary_tpf: Two-star blend scenario for testing localization
- make_crowded_field_tpf: Multiple stars for crowded field scenarios
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.wcs import WCS

from tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef


@dataclass(frozen=True)
class StarSpec:
    """Specification for a synthetic star.

    Attributes:
        row: Row position in pixels (0-indexed).
        col: Column position in pixels (0-indexed).
        flux: Total flux of the star (counts/cadence).
        sigma: PSF sigma in pixels (Gaussian width).
    """

    row: float
    col: float
    flux: float
    sigma: float = 1.5


@dataclass(frozen=True)
class TransitSpec:
    """Specification for a synthetic transit signal.

    Attributes:
        star_idx: Index of the star to apply transit to.
        depth_frac: Transit depth as fraction of stellar flux.
        period: Orbital period in days.
        t0: Transit epoch (time of first transit) in BTJD.
        duration_days: Full transit duration in days.
    """

    star_idx: int
    depth_frac: float
    period: float
    t0: float
    duration_days: float


def _make_test_wcs(
    crval: tuple[float, float] = (120.0, -50.0),
    pixel_scale_deg: float = 21.0 / 3600.0,
    shape: tuple[int, int] = (11, 11),
) -> WCS:
    """Create a test WCS object for TESS-like data.

    Args:
        crval: (RA, Dec) reference coordinates in degrees.
        pixel_scale_deg: Pixel scale in degrees per pixel.
        shape: (n_rows, n_cols) stamp shape.

    Returns:
        Configured WCS object with TAN projection.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(shape[1] + 1) / 2, (shape[0] + 1) / 2]
    wcs.wcs.crval = list(crval)
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _gaussian_psf(
    shape: tuple[int, int],
    center_row: float,
    center_col: float,
    sigma: float,
    total_flux: float,
) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
    """Generate a 2D Gaussian PSF.

    Args:
        shape: (n_rows, n_cols) output array shape.
        center_row: Row center of the PSF.
        center_col: Column center of the PSF.
        sigma: Standard deviation of the Gaussian.
        total_flux: Total integrated flux of the PSF.

    Returns:
        2D array with the PSF normalized to total_flux.
    """
    n_rows, n_cols = shape
    rows = np.arange(n_rows, dtype=np.float64)
    cols = np.arange(n_cols, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Gaussian PSF
    rsq = (row_grid - center_row) ** 2 + (col_grid - center_col) ** 2
    psf = np.exp(-rsq / (2 * sigma**2))

    # Normalize to total flux
    psf_sum = np.sum(psf)
    if psf_sum > 0:
        psf = psf * (total_flux / psf_sum)

    return psf.astype(np.float64)


def _compute_transit_mask(
    time: np.ndarray[Any, np.dtype[np.floating[Any]]],
    transit_spec: TransitSpec,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Compute boolean mask indicating in-transit times.

    Args:
        time: Array of observation times.
        transit_spec: Transit specification.

    Returns:
        Boolean mask where True indicates in-transit cadences.
    """
    phase = ((time - transit_spec.t0) % transit_spec.period) / transit_spec.period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    half_duration_phase = (transit_spec.duration_days / 2) / transit_spec.period
    mask: np.ndarray[Any, np.dtype[np.bool_]] = np.abs(phase) <= half_duration_phase
    return mask


def make_synthetic_tpf_fits(
    *,
    shape: tuple[int, int, int] = (1000, 11, 11),
    star_positions: list[tuple[float, float, float]] | None = None,
    stars: list[StarSpec] | None = None,
    transit_params: dict[str, Any] | None = None,
    transit_spec: TransitSpec | None = None,
    noise_level: float = 100.0,
    wcs_crval: tuple[float, float] = (120.0, -50.0),
    pixel_scale_arcsec: float = 21.0,
    seed: int = 42,
    tic_id: int = 999999999,
    sector: int = 99,
    author: str = "spoc",
    time_start_btjd: float = 2458000.0,
    time_span_days: float = 27.0,
    saturation_threshold: float | None = None,
) -> TPFFitsData:
    """Generate synthetic TPF with known star positions and optional transit.

    This creates a synthetic TPF data cube with configurable stars, noise,
    and transit signals. The WCS is set up so that pixel coordinates can be
    accurately converted to sky coordinates.

    Args:
        shape: (n_cadences, n_rows, n_cols) shape of the flux cube.
        star_positions: Legacy format [(row, col, flux), ...]. Use `stars` instead.
        stars: List of StarSpec objects defining star positions and properties.
        transit_params: Legacy format dict. Use `transit_spec` instead.
        transit_spec: Transit specification for one of the stars.
        noise_level: Standard deviation of Gaussian noise (per pixel per cadence).
        wcs_crval: (RA, Dec) of the stamp center in degrees.
        pixel_scale_arcsec: Pixel scale in arcseconds per pixel.
        seed: Random seed for reproducibility.
        tic_id: TIC ID for the reference.
        sector: Sector number for the reference.
        author: Pipeline author for the reference.
        time_start_btjd: Start time in BTJD.
        time_span_days: Duration of observations in days.
        saturation_threshold: Optional threshold for saturation (flux values above
            this will be clipped, useful for testing saturation detection).

    Returns:
        TPFFitsData with synthetic flux cube and properly configured WCS.

    Example:
        >>> stars = [
        ...     StarSpec(row=5.5, col=5.5, flux=10000),  # Central target
        ...     StarSpec(row=7.0, col=7.0, flux=5000),   # Neighbor
        ... ]
        >>> transit = TransitSpec(
        ...     star_idx=0, depth_frac=0.01, period=5.0,
        ...     t0=2458001.0, duration_days=0.2
        ... )
        >>> tpf = make_synthetic_tpf_fits(stars=stars, transit_spec=transit)
    """
    rng = np.random.default_rng(seed)
    n_cadences, n_rows, n_cols = shape

    # Convert legacy star_positions format to StarSpec
    if stars is None:
        if star_positions is not None:
            stars = [StarSpec(row=pos[0], col=pos[1], flux=pos[2]) for pos in star_positions]
        else:
            # Default: single star at center
            stars = [StarSpec(row=(n_rows - 1) / 2.0, col=(n_cols - 1) / 2.0, flux=10000.0)]

    # Convert legacy transit_params format to TransitSpec
    if transit_spec is None and transit_params is not None:
        transit_spec = TransitSpec(
            star_idx=transit_params.get("star_idx", 0),
            depth_frac=transit_params.get("depth_frac", 0.01),
            period=transit_params.get("period", 5.0),
            t0=transit_params.get("t0", time_start_btjd + 1.0),
            duration_days=transit_params.get("duration_days", 0.2),
        )

    # Create time array
    time = np.linspace(time_start_btjd, time_start_btjd + time_span_days, n_cadences).astype(
        np.float64
    )

    # Initialize flux cube with noise
    flux = rng.normal(0, noise_level, (n_cadences, n_rows, n_cols)).astype(np.float64)

    # Add stars
    for i, star in enumerate(stars):
        star_psf = _gaussian_psf(
            shape=(n_rows, n_cols),
            center_row=star.row,
            center_col=star.col,
            sigma=star.sigma,
            total_flux=star.flux,
        )

        # Apply transit if this star has one
        if transit_spec is not None and transit_spec.star_idx == i:
            transit_mask = _compute_transit_mask(time, transit_spec)
            for t in range(n_cadences):
                if transit_mask[t]:
                    flux[t] += star_psf * (1.0 - transit_spec.depth_frac)
                else:
                    flux[t] += star_psf
        else:
            # No transit - add full flux at each cadence
            for t in range(n_cadences):
                flux[t] += star_psf

    # Apply saturation threshold if specified
    if saturation_threshold is not None:
        flux = np.clip(flux, None, saturation_threshold)

    # Create flux error (proportional to sqrt of flux)
    flux_err = np.sqrt(np.maximum(flux, noise_level)).astype(np.float64)

    # Create WCS
    pixel_scale_deg = pixel_scale_arcsec / 3600.0
    wcs = _make_test_wcs(
        crval=wcs_crval,
        pixel_scale_deg=pixel_scale_deg,
        shape=(n_rows, n_cols),
    )

    # Create aperture mask (all pixels in mask by default)
    aperture_mask = np.ones((n_rows, n_cols), dtype=np.int32)

    # Create quality flags (all good)
    quality = np.zeros(n_cadences, dtype=np.int32)

    # Create reference
    ref = TPFFitsRef(tic_id=tic_id, sector=sector, author=author)

    # Build metadata
    meta: dict[str, Any] = {
        "RA_OBJ": wcs_crval[0],
        "DEC_OBJ": wcs_crval[1],
        "TSTART": time_start_btjd,
        "TSTOP": time_start_btjd + time_span_days,
    }

    return TPFFitsData(
        ref=ref,
        time=time,
        flux=flux,
        flux_err=flux_err,
        wcs=wcs,
        aperture_mask=aperture_mask,
        quality=quality,
        camera=1,
        ccd=1,
        meta=meta,
    )


def make_blended_binary_tpf(
    *,
    separation_arcsec: float = 10.0,
    flux_ratio: float = 0.5,
    transit_on_secondary: bool = True,
    transit_depth_frac: float = 0.01,
    primary_flux: float = 10000.0,
    shape: tuple[int, int, int] = (1000, 11, 11),
    noise_level: float = 100.0,
    wcs_crval: tuple[float, float] = (120.0, -50.0),
    pixel_scale_arcsec: float = 21.0,
    seed: int = 42,
    tic_id: int = 888888888,
    sector: int = 99,
    period: float = 5.0,
    t0_offset_days: float = 1.0,
    duration_days: float = 0.2,
) -> TPFFitsData:
    """Generate synthetic blended binary TPF for localization testing.

    Creates a TPF with two stars (primary and secondary) where the transit
    can be placed on either star. This is the canonical test case for
    checking that difference image localization can detect off-target sources.

    The primary star is placed at the stamp center, and the secondary is
    offset by separation_arcsec in the +row direction (approximately +Dec).

    Args:
        separation_arcsec: Angular separation between stars in arcseconds.
        flux_ratio: Secondary flux / primary flux (0 < ratio <= 1).
        transit_on_secondary: If True, transit is on secondary; else on primary.
        transit_depth_frac: Transit depth as fraction of stellar flux.
        primary_flux: Flux of the primary star.
        shape: (n_cadences, n_rows, n_cols) shape of the flux cube.
        noise_level: Standard deviation of Gaussian noise.
        wcs_crval: (RA, Dec) of the stamp center in degrees.
        pixel_scale_arcsec: Pixel scale in arcseconds per pixel.
        seed: Random seed for reproducibility.
        tic_id: TIC ID for the reference.
        sector: Sector number for the reference.
        period: Transit period in days.
        t0_offset_days: Offset from time_start for first transit.
        duration_days: Transit duration in days.

    Returns:
        TPFFitsData with blended binary configuration.
    """
    n_cadences, n_rows, n_cols = shape

    # Compute pixel separation
    separation_pixels = separation_arcsec / pixel_scale_arcsec

    # Place primary at center
    primary_row = (n_rows - 1) / 2.0
    primary_col = (n_cols - 1) / 2.0

    # Place secondary offset in row direction (approximately +Dec)
    secondary_row = primary_row + separation_pixels
    secondary_col = primary_col

    # Compute secondary flux
    secondary_flux = primary_flux * flux_ratio

    # Create star specs
    stars = [
        StarSpec(row=primary_row, col=primary_col, flux=primary_flux),
        StarSpec(row=secondary_row, col=secondary_col, flux=secondary_flux),
    ]

    # Create transit spec on the appropriate star
    time_start_btjd = 2458000.0
    transit_star_idx = 1 if transit_on_secondary else 0

    transit_spec = TransitSpec(
        star_idx=transit_star_idx,
        depth_frac=transit_depth_frac,
        period=period,
        t0=time_start_btjd + t0_offset_days,
        duration_days=duration_days,
    )

    return make_synthetic_tpf_fits(
        shape=shape,
        stars=stars,
        transit_spec=transit_spec,
        noise_level=noise_level,
        wcs_crval=wcs_crval,
        pixel_scale_arcsec=pixel_scale_arcsec,
        seed=seed,
        tic_id=tic_id,
        sector=sector,
        time_start_btjd=time_start_btjd,
    )


def make_crowded_field_tpf(
    *,
    n_stars: int = 5,
    transit_star_idx: int = 0,
    transit_depth_frac: float = 0.01,
    shape: tuple[int, int, int] = (1000, 11, 11),
    noise_level: float = 100.0,
    wcs_crval: tuple[float, float] = (120.0, -50.0),
    pixel_scale_arcsec: float = 21.0,
    seed: int = 42,
    tic_id: int = 777777777,
    sector: int = 99,
    period: float = 5.0,
    t0_offset_days: float = 1.0,
    duration_days: float = 0.2,
) -> TPFFitsData:
    """Generate synthetic crowded field TPF for localization testing.

    Creates a TPF with multiple stars placed randomly around the stamp.
    One star (specified by transit_star_idx) has a transit signal.
    The first star (idx=0) is always placed at the stamp center as the
    nominal target.

    Args:
        n_stars: Total number of stars to place (including target at center).
        transit_star_idx: Index of the star to apply transit to.
        transit_depth_frac: Transit depth as fraction of stellar flux.
        shape: (n_cadences, n_rows, n_cols) shape of the flux cube.
        noise_level: Standard deviation of Gaussian noise.
        wcs_crval: (RA, Dec) of the stamp center in degrees.
        pixel_scale_arcsec: Pixel scale in arcseconds per pixel.
        seed: Random seed for reproducibility.
        tic_id: TIC ID for the reference.
        sector: Sector number for the reference.
        period: Transit period in days.
        t0_offset_days: Offset from time_start for first transit.
        duration_days: Transit duration in days.

    Returns:
        TPFFitsData with crowded field configuration.

    Raises:
        ValueError: If transit_star_idx >= n_stars.
    """
    if transit_star_idx >= n_stars:
        raise ValueError(f"transit_star_idx ({transit_star_idx}) must be < n_stars ({n_stars})")

    n_cadences, n_rows, n_cols = shape
    rng = np.random.default_rng(seed)

    # Create star list
    stars: list[StarSpec] = []

    # First star at center (the nominal target)
    center_row = (n_rows - 1) / 2.0
    center_col = (n_cols - 1) / 2.0
    stars.append(StarSpec(row=center_row, col=center_col, flux=10000.0))

    # Add remaining stars at random positions
    for _ in range(1, n_stars):
        # Random position within stamp (with 1-pixel margin)
        row = rng.uniform(1.5, n_rows - 1.5)
        col = rng.uniform(1.5, n_cols - 1.5)
        # Random flux (0.2 to 1.0 times target flux)
        flux = 10000.0 * rng.uniform(0.2, 1.0)
        stars.append(StarSpec(row=row, col=col, flux=flux))

    # Create transit spec
    time_start_btjd = 2458000.0
    transit_spec = TransitSpec(
        star_idx=transit_star_idx,
        depth_frac=transit_depth_frac,
        period=period,
        t0=time_start_btjd + t0_offset_days,
        duration_days=duration_days,
    )

    return make_synthetic_tpf_fits(
        shape=shape,
        stars=stars,
        transit_spec=transit_spec,
        noise_level=noise_level,
        wcs_crval=wcs_crval,
        pixel_scale_arcsec=pixel_scale_arcsec,
        seed=seed + 1000,  # Different seed from star positions
        tic_id=tic_id,
        sector=sector,
        time_start_btjd=time_start_btjd,
    )


def make_saturated_tpf(
    *,
    shape: tuple[int, int, int] = (1000, 11, 11),
    star_flux: float = 100000.0,
    saturation_threshold: float = 50000.0,
    transit_depth_frac: float = 0.01,
    noise_level: float = 100.0,
    wcs_crval: tuple[float, float] = (120.0, -50.0),
    pixel_scale_arcsec: float = 21.0,
    seed: int = 42,
    tic_id: int = 666666666,
    sector: int = 99,
) -> TPFFitsData:
    """Generate synthetic TPF with a saturated star.

    Creates a TPF where the central star exceeds the saturation threshold,
    causing clipping of the central pixels. This is useful for testing
    saturation detection in localization algorithms.

    Args:
        shape: (n_cadences, n_rows, n_cols) shape of the flux cube.
        star_flux: Total flux of the star (should exceed saturation).
        saturation_threshold: Maximum pixel value before saturation.
        transit_depth_frac: Transit depth as fraction of stellar flux.
        noise_level: Standard deviation of Gaussian noise.
        wcs_crval: (RA, Dec) of the stamp center in degrees.
        pixel_scale_arcsec: Pixel scale in arcseconds per pixel.
        seed: Random seed for reproducibility.
        tic_id: TIC ID for the reference.
        sector: Sector number for the reference.

    Returns:
        TPFFitsData with saturated star.
    """
    n_cadences, n_rows, n_cols = shape
    center_row = (n_rows - 1) / 2.0
    center_col = (n_cols - 1) / 2.0

    stars = [StarSpec(row=center_row, col=center_col, flux=star_flux)]

    time_start_btjd = 2458000.0
    transit_spec = TransitSpec(
        star_idx=0,
        depth_frac=transit_depth_frac,
        period=5.0,
        t0=time_start_btjd + 1.0,
        duration_days=0.2,
    )

    return make_synthetic_tpf_fits(
        shape=shape,
        stars=stars,
        transit_spec=transit_spec,
        noise_level=noise_level,
        wcs_crval=wcs_crval,
        pixel_scale_arcsec=pixel_scale_arcsec,
        seed=seed,
        tic_id=tic_id,
        sector=sector,
        time_start_btjd=time_start_btjd,
        saturation_threshold=saturation_threshold,
    )


__all__ = [
    "StarSpec",
    "TransitSpec",
    "make_blended_binary_tpf",
    "make_crowded_field_tpf",
    "make_saturated_tpf",
    "make_synthetic_tpf_fits",
]
