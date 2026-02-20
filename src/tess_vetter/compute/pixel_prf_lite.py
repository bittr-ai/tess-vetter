"""PRF-lite model construction and aperture weight evaluation.

This module provides simplified Point Response Function (PRF) modeling
for TESS pixel-level host identification. It uses a 2D Gaussian as a
computationally efficient approximation of the TESS PRF.

CRITICAL: This module must remain pure compute:
- NO file I/O (open, Path, etc.)
- NO network access
- NO imports of astropy, lightkurve
- ONLY numpy dependencies

These functions are designed to work with pre-validated numpy arrays.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

BoolArray: TypeAlias = NDArray[np.bool_]

__all__ = [
    "build_prf_model",
    "evaluate_prf_weights",
    "build_prf_model_at_pixels",
]


def build_prf_model(
    center_row: float,
    center_col: float,
    shape: tuple[int, int],
    sigma: float = 1.5,
) -> NDArray[np.float64]:
    """Build a 2D Gaussian PSF model centered at given pixel coordinates.

    Creates a normalized Gaussian approximation of the TESS PRF. The Gaussian
    is centered at (center_row, center_col) and evaluated on a grid of the
    given shape.

    Parameters
    ----------
    center_row : float
        Row coordinate of the PSF center (0-indexed, can be fractional).
    center_col : float
        Column coordinate of the PSF center (0-indexed, can be fractional).
    shape : tuple[int, int]
        Shape of the output array as (n_rows, n_cols).
    sigma : float, optional
        Standard deviation of the Gaussian in pixels. Default is 1.5,
        which approximates the TESS pixel response function.
        TESS pixels are 21 arcsec, and typical PSF FWHM is ~42 arcsec,
        so sigma ~ 1.5 pixels is a reasonable approximation.

    Returns
    -------
    NDArray[np.float64]
        2D array of shape `shape` containing the normalized Gaussian PSF model.
        The model is normalized such that the sum equals 1.0.

    Raises
    ------
    ValueError
        If sigma is not positive or shape has non-positive dimensions.

    Examples
    --------
    >>> model = build_prf_model(5.0, 5.0, (11, 11), sigma=1.5)
    >>> model.shape
    (11, 11)
    >>> np.isclose(model.sum(), 1.0)
    True
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"shape dimensions must be positive, got {shape}")

    n_rows, n_cols = shape

    # Create coordinate grids (0-indexed pixel centers)
    row_coords, col_coords = np.mgrid[0:n_rows, 0:n_cols]

    # Compute squared distance from center
    dist_sq = (row_coords - center_row) ** 2 + (col_coords - center_col) ** 2

    # Gaussian: exp(-r^2 / (2 * sigma^2))
    model = np.exp(-dist_sq / (2.0 * sigma**2))

    # Normalize to sum to 1
    model_sum = model.sum()
    if model_sum > 0:
        model = model / model_sum

    result: NDArray[np.float64] = model.astype(np.float64)
    return result


def build_prf_model_at_pixels(
    center_row: float,
    center_col: float,
    pixel_rows: NDArray[np.intp],
    pixel_cols: NDArray[np.intp],
    sigma: float = 1.5,
) -> NDArray[np.float64]:
    """Build a 2D Gaussian PSF model evaluated at specific pixel coordinates.

    Similar to build_prf_model but evaluates only at specified pixel locations.
    Useful when working with irregular aperture masks.

    Parameters
    ----------
    center_row : float
        Row coordinate of the PSF center (0-indexed, can be fractional).
    center_col : float
        Column coordinate of the PSF center (0-indexed, can be fractional).
    pixel_rows : NDArray[np.intp]
        1D array of row indices where the model should be evaluated.
    pixel_cols : NDArray[np.intp]
        1D array of column indices where the model should be evaluated.
    sigma : float, optional
        Standard deviation of the Gaussian in pixels. Default is 1.5.

    Returns
    -------
    NDArray[np.float64]
        1D array of model values at each (row, col) pixel location.
        NOT normalized (caller should normalize if needed).

    Raises
    ------
    ValueError
        If sigma is not positive or arrays have mismatched lengths.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if len(pixel_rows) != len(pixel_cols):
        raise ValueError(
            f"pixel_rows and pixel_cols must have same length: "
            f"{len(pixel_rows)} vs {len(pixel_cols)}"
        )

    if len(pixel_rows) == 0:
        return np.array([], dtype=np.float64)

    # Compute squared distance from center for each pixel
    dist_sq = (pixel_rows - center_row) ** 2 + (pixel_cols - center_col) ** 2

    # Gaussian values (unnormalized)
    model = np.exp(-dist_sq / (2.0 * sigma**2))

    return model.astype(np.float64)


def evaluate_prf_weights(
    model: NDArray[np.float64],
    aperture_mask: BoolArray,
) -> float:
    """Evaluate the flux fraction captured within an aperture mask.

    Computes the fraction of the PRF model flux that falls within the
    specified aperture. This is used to predict the observed transit
    depth for a given source hypothesis.

    Parameters
    ----------
    model : NDArray[np.float64]
        2D array containing the normalized PRF model (should sum to ~1.0).
    aperture_mask : NDArray[bool]
        2D boolean array of the same shape as model. True indicates pixels
        included in the aperture.

    Returns
    -------
    float
        Fraction of model flux within the aperture (0.0 to 1.0).
        Returns 0.0 if the aperture is empty or model has zero flux.

    Raises
    ------
    ValueError
        If model and aperture_mask have different shapes.

    Notes
    -----
    The returned weight represents what fraction of a source's flux
    would be captured by the aperture. If a transit occurs on a source
    and the source's PRF weight is W, the observed transit depth would be:

        depth_observed = depth_true * W

    where depth_true is the geometric transit depth on that source.

    Examples
    --------
    >>> model = build_prf_model(5.0, 5.0, (11, 11))
    >>> mask = np.zeros((11, 11), dtype=bool)
    >>> mask[4:7, 4:7] = True  # 3x3 aperture centered on source
    >>> weight = evaluate_prf_weights(model, mask)
    >>> 0.5 < weight < 1.0  # Most flux captured in central aperture
    True
    """
    if model.shape != aperture_mask.shape:
        raise ValueError(
            f"model and aperture_mask must have same shape: {model.shape} vs {aperture_mask.shape}"
        )

    # Sum of model flux within aperture
    flux_in_aperture = np.sum(model[aperture_mask])

    # Total model flux (should be ~1.0 if normalized, but compute anyway)
    total_flux = np.sum(model)

    if total_flux <= 0:
        return 0.0

    return float(flux_in_aperture / total_flux)
