"""Difference imaging for transit localization.

This module provides tools for computing difference images from Target Pixel File
(TPF) data to verify that the transit signal originates from the target star.

The algorithm:
1. Compute median in-transit image (during transits)
2. Compute median out-of-transit image (outside transits)
3. Difference = out_of_transit - in_transit (transit makes star dimmer)
4. Find brightest pixel in difference image
5. Compute localization score based on distance to target

A high localization score (close to 1.0) indicates the transit signal is
localized to the target star, while a low score suggests a background
eclipsing binary or other contaminating source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tess_vetter.pixel.aperture import TransitParams
from tess_vetter.pixel.cadence_mask import default_cadence_mask

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class DifferenceImageResult:
    """Result of difference image computation.

    Attributes:
        difference_image: 2D array showing out_of_transit - in_transit
        localization_score: 0-1, higher = better localization to target
        brightest_pixel_coords: (row, col) of brightest pixel in difference image
        target_coords: (row, col) center of TPF (assumed target location)
        distance_to_target: Distance in pixels from brightest pixel to target
    """

    difference_image: NDArray[np.floating]  # 2D array
    localization_score: float  # 0-1, higher = better localization to target
    brightest_pixel_coords: tuple[int, int]  # (row, col)
    target_coords: tuple[int, int]  # (row, col) - center of TPF
    distance_to_target: float  # pixels


def _compute_transit_mask(
    time: NDArray[np.floating],
    transit_params: TransitParams,
) -> NDArray[np.bool_]:
    """Compute boolean mask indicating which times are in-transit.

    Args:
        time: 1D array of observation times
        transit_params: Transit parameters (period, t0, duration)

    Returns:
        Boolean array where True indicates in-transit times
    """
    # Phase fold the times
    phase = ((time - transit_params.t0) % transit_params.period) / transit_params.period

    # Handle wrap-around near phase 0
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    # Transit is centered at phase 0, duration covers [-half_dur, +half_dur]
    half_duration_phase = (transit_params.duration / 2) / transit_params.period

    result: NDArray[np.bool_] = np.abs(phase) <= half_duration_phase
    return result


def compute_difference_image(
    tpf_data: NDArray[np.floating],
    time: NDArray[np.floating],
    transit_params: TransitParams,
) -> DifferenceImageResult:
    """Compute difference image for transit localization.

    The difference image shows where flux decreases during transit. For a
    genuine transit on the target star, the brightest pixel in the difference
    image should be at or near the target (center of TPF).

    Algorithm:
        1. Compute median in-transit image
        2. Compute median out-of-transit image
        3. Difference = out_of_transit - in_transit (transit makes it dimmer)
        4. Find brightest pixel in difference image
        5. Compute localization score: 1.0 - (distance_to_target / max_distance)

    Args:
        tpf_data: 3D array of shape (time, rows, cols) containing pixel flux values
        time: 1D array of observation times matching tpf_data first dimension
        transit_params: Transit parameters defining the transit event

    Returns:
        DifferenceImageResult with difference image and localization metrics

    Raises:
        ValueError: If input arrays have invalid shapes or no in/out-of-transit data
    """
    # Validate inputs
    if tpf_data.ndim != 3:
        raise ValueError(f"tpf_data must be 3D, got shape {tpf_data.shape}")

    n_times, n_rows, n_cols = tpf_data.shape

    if time.ndim != 1:
        raise ValueError(f"time must be 1D, got shape {time.shape}")

    if len(time) != n_times:
        raise ValueError(
            f"time length ({len(time)}) must match tpf_data first dimension ({n_times})"
        )

    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"tpf_data spatial dimensions must be non-zero, got {tpf_data.shape}")

    cadence_mask = default_cadence_mask(
        time=time,
        flux=tpf_data,
        quality=np.zeros(int(time.shape[0]), dtype=np.int32),
        require_finite_pixels=True,
    )
    tpf_data = tpf_data[cadence_mask]
    time = time[cadence_mask]

    n_times = tpf_data.shape[0]
    if n_times < 3:
        raise ValueError("Insufficient valid cadences after masking.")

    # Compute transit mask
    in_transit_mask = _compute_transit_mask(time, transit_params)
    out_of_transit_mask = ~in_transit_mask

    # Check we have data in both regions
    n_in_transit = np.sum(in_transit_mask)
    n_out_of_transit = np.sum(out_of_transit_mask)

    if n_in_transit == 0:
        raise ValueError(
            "No in-transit data points found. Check transit_params against time array."
        )

    if n_out_of_transit == 0:
        raise ValueError("No out-of-transit data points found. Transit duration may be too long.")

    # Compute median images
    in_transit_image = np.nanmedian(tpf_data[in_transit_mask], axis=0)
    out_of_transit_image = np.nanmedian(tpf_data[out_of_transit_mask], axis=0)

    # Difference: out - in (transit causes dimming, so diff is positive where transit occurs)
    difference_image = out_of_transit_image - in_transit_image

    # Find brightest pixel (maximum flux decrease = strongest transit signal)
    brightest_flat_idx = np.argmax(difference_image)
    brightest_row, brightest_col = np.unravel_index(brightest_flat_idx, difference_image.shape)
    brightest_pixel_coords = (int(brightest_row), int(brightest_col))

    # Target is assumed to be at center of TPF
    target_row = n_rows // 2
    target_col = n_cols // 2
    target_coords = (target_row, target_col)

    # Compute distance from brightest pixel to target
    distance_to_target = np.sqrt(
        (brightest_row - target_row) ** 2 + (brightest_col - target_col) ** 2
    )

    # Compute maximum possible distance (corner to center)
    max_distance = np.sqrt((n_rows / 2) ** 2 + (n_cols / 2) ** 2)

    # Localization score: 1.0 = perfect (brightest at target), 0.0 = worst (brightest at corner)
    # Clamp to [0, 1] in case brightest is exactly at target edge
    if max_distance > 0:
        localization_score = max(0.0, min(1.0, 1.0 - (distance_to_target / max_distance)))
    else:
        # Degenerate case: 1x1 TPF
        localization_score = 1.0

    return DifferenceImageResult(
        difference_image=difference_image,
        localization_score=localization_score,
        brightest_pixel_coords=brightest_pixel_coords,
        target_coords=target_coords,
        distance_to_target=float(distance_to_target),
    )


__all__ = [
    "DifferenceImageResult",
    "TransitParams",
    "compute_difference_image",
]
