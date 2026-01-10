"""Aperture dependence analysis for pixel-level transit vetting.

This module provides tools to analyze how transit depth measurements vary
across different photometric aperture sizes. Stable depth measurements across
apertures indicate on-target signals, while variable depths suggest
contamination or off-target sources.

Algorithm:
1. For each aperture radius, create circular aperture mask
2. Sum flux within aperture for each cadence
3. Compute in/out transit flux ratio to estimate depth
4. Compare depths across apertures
5. Stable = low variance across apertures (on-target signal)
6. Unstable = high variance (contamination or off-target)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from bittr_tess_vetter.pixel.cadence_mask import default_cadence_mask


# Default aperture radii to test (in pixels)
DEFAULT_APERTURE_RADII: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0)


@dataclass(frozen=True)
class TransitParams:
    """Parameters describing a transit signal.

    Attributes:
        period: Orbital period in days.
        t0: Transit epoch (time of first transit) in same units as time array.
        duration: Transit duration in days.
        depth: Expected transit depth (fraction, e.g., 0.01 for 1%).
    """

    period: float
    t0: float
    duration: float
    depth: float = 0.01

    def __post_init__(self) -> None:
        """Validate transit parameters."""
        if self.period <= 0:
            raise ValueError(f"period must be positive, got {self.period}")
        if self.duration <= 0:
            raise ValueError(f"duration must be positive, got {self.duration}")
        if self.duration >= self.period:
            raise ValueError(f"duration ({self.duration}) must be less than period ({self.period})")


@dataclass(frozen=True)
class ApertureDependenceResult:
    """Results from aperture dependence analysis.

    Attributes:
        depths_by_aperture: Mapping from aperture radius (pixels) to measured
            transit depth (in ppm, parts per million).
        stability_metric: Score from 0 to 1 indicating depth consistency across
            apertures. Higher values indicate more stable (on-target) signals.
        recommended_aperture: Aperture radius (pixels) with best signal-to-noise
            or most reliable depth measurement.
        depth_variance: Variance of measured depths across apertures (ppm^2).
    """

    depths_by_aperture: dict[float, float]  # radius_pixels -> depth_ppm
    stability_metric: float  # 0-1, higher = more stable across apertures
    recommended_aperture: float  # radius in pixels
    depth_variance: float
    # Context (for interpretation / triage)
    n_in_transit_cadences: int = 0
    n_out_of_transit_cadences: int = 0
    n_transit_epochs: int = 0
    baseline_mode: str = "local_window"
    local_baseline_window_days: float = 0.5
    background_mode: str = "annulus_median"
    background_annulus_radii_pixels: tuple[float, float] | None = None
    n_background_pixels: int = 0
    drift_fraction_recommended: float | None = None
    flags: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


def _create_circular_aperture_mask(
    shape: tuple[int, int],
    radius: float,
    center: tuple[float, float],
) -> NDArray[np.bool_]:
    """Create a circular aperture mask.

    Args:
        shape: Shape of the output mask (rows, cols).
        radius: Radius of the circular aperture in pixels.
        center: Center of the aperture (row, col) in pixel coordinates.

    Returns:
        Boolean mask where True indicates pixels inside the aperture.
    """
    rows, cols = shape
    y_center, x_center = center

    # Create coordinate grids
    y_indices, x_indices = np.ogrid[:rows, :cols]

    # Compute distance from center
    distance_squared = (y_indices - y_center) ** 2 + (x_indices - x_center) ** 2

    # Create mask
    return distance_squared <= radius**2


def _create_background_annulus_mask(
    shape: tuple[int, int],
    inner_radius: float,
    outer_radius: float,
    center: tuple[float, float],
) -> NDArray[np.bool_]:
    """Create an annular mask for background estimation.

    Args:
        shape: Shape of the output mask (rows, cols).
        inner_radius: Inner radius of the annulus in pixels.
        outer_radius: Outer radius of the annulus in pixels.
        center: Center of the annulus (row, col) in pixel coordinates.

    Returns:
        Boolean mask where True indicates pixels inside the annulus.
    """
    rows, cols = shape
    y_center, x_center = center

    # Create coordinate grids
    y_indices, x_indices = np.ogrid[:rows, :cols]

    # Compute distance from center
    distance_squared = (y_indices - y_center) ** 2 + (x_indices - x_center) ** 2

    # Create annulus mask (inner_radius < distance <= outer_radius)
    return (distance_squared > inner_radius**2) & (distance_squared <= outer_radius**2)


def _compute_transit_mask(
    time: NDArray[np.floating],
    transit_params: TransitParams,
) -> NDArray[np.bool_]:
    """Identify cadences that fall within transits.

    Args:
        time: Array of observation times.
        transit_params: Transit parameters including period, t0, and duration.

    Returns:
        Boolean mask where True indicates in-transit cadences.
    """
    # Compute phase-folded time
    phase = (time - transit_params.t0) % transit_params.period

    # Handle phase wrapping for transits near phase 0
    phase = np.where(phase > transit_params.period / 2, phase - transit_params.period, phase)

    # Mark cadences within half duration of transit center
    half_duration = transit_params.duration / 2
    result: NDArray[np.bool_] = np.abs(phase) <= half_duration
    return result


def _compute_depth_from_lightcurve(
    flux: NDArray[np.floating],
    in_transit_mask: NDArray[np.bool_],
) -> float:
    """Compute transit depth from aperture photometry.

    Uses the simple in/out flux ratio method:
    depth = 1 - (mean_in_transit / mean_out_of_transit)

    Args:
        flux: 1D array of flux values from aperture photometry.
        in_transit_mask: Boolean mask indicating in-transit cadences.

    Returns:
        Transit depth in ppm (parts per million).

    Raises:
        ValueError: If insufficient data for depth calculation.
    """
    out_of_transit_mask = ~in_transit_mask

    # Need at least some data in both states
    if not np.any(in_transit_mask) or not np.any(out_of_transit_mask):
        raise ValueError("Insufficient in-transit or out-of-transit data")

    mean_in_transit = np.nanmean(flux[in_transit_mask])
    mean_out_of_transit = np.nanmean(flux[out_of_transit_mask])

    if mean_out_of_transit == 0:
        raise ValueError("Out-of-transit flux is zero")

    depth_fraction = 1.0 - (mean_in_transit / mean_out_of_transit)

    # Convert to ppm
    return float(depth_fraction * 1_000_000)


def _compute_stability_metric(
    depths: list[float],
    expected_depth_ppm: float | None = None,
) -> float:
    """Compute stability metric from depth measurements.

    The stability metric quantifies how consistent the measured depths are
    across different aperture sizes. A value close to 1 indicates stable
    measurements (consistent with an on-target signal), while values close
    to 0 indicate high variability (suggesting contamination).

    The metric uses the coefficient of variation (CV = std/mean) transformed
    to a 0-1 scale where lower CV maps to higher stability.

    Args:
        depths: List of depth measurements in ppm.
        expected_depth_ppm: Optional expected depth for reference (not used
            in current implementation but reserved for future enhancements).

    Returns:
        Stability metric between 0 and 1.
    """
    if len(depths) < 2:
        return 1.0  # Single measurement is trivially stable

    depths_array = np.array(depths)

    # Remove any invalid values
    valid_depths = depths_array[np.isfinite(depths_array)]
    if len(valid_depths) < 2:
        return 1.0

    mean_depth = np.mean(valid_depths)
    std_depth = np.std(valid_depths)

    if mean_depth == 0:
        # If mean is zero, check if all values are approximately zero
        if std_depth < 1e-10:
            return 1.0
        return 0.0

    # Coefficient of variation
    cv = std_depth / abs(mean_depth)

    # Transform CV to stability metric (0-1 range)
    # Using exponential decay: stability = exp(-cv * scale_factor)
    # Scale factor chosen so that CV=0.5 gives stability ~0.5
    scale_factor = 1.4  # ln(2) / 0.5
    stability = np.exp(-cv * scale_factor)

    return float(np.clip(stability, 0.0, 1.0))


def _select_recommended_aperture(
    depths_by_aperture: dict[float, float],
    stability_metric: float,
) -> float:
    """Select the recommended aperture based on depth measurements.

    Strategy:
    - If depths are stable (stability > 0.7), recommend median aperture
    - If depths are unstable, recommend the aperture with depth closest
      to the median depth (most "typical" measurement)

    Args:
        depths_by_aperture: Mapping from aperture radius to depth in ppm.
        stability_metric: Computed stability metric (0-1).

    Returns:
        Recommended aperture radius in pixels.
    """
    apertures = list(depths_by_aperture.keys())
    depths = list(depths_by_aperture.values())

    if len(apertures) == 0:
        return 2.0  # Fallback default

    if len(apertures) == 1:
        return apertures[0]

    if stability_metric > 0.7:
        # Depths are stable - recommend median-sized aperture
        sorted_apertures = sorted(apertures)
        median_idx = len(sorted_apertures) // 2
        return sorted_apertures[median_idx]

    # Depths are unstable - find aperture with median depth
    depths_array = np.array(depths)
    median_depth = np.median(depths_array)
    closest_idx = int(np.argmin(np.abs(depths_array - median_depth)))
    return apertures[closest_idx]


def compute_aperture_dependence(
    tpf_data: NDArray[np.floating],  # shape (time, rows, cols)
    time: NDArray[np.floating],
    transit_params: TransitParams,
    aperture_radii: list[float] | None = None,  # default: [1.0, 1.5, 2.0, 2.5, 3.0]
    center: tuple[float, float] | None = None,  # default: center of TPF
) -> ApertureDependenceResult:
    """Analyze transit depth dependence on photometric aperture size.

    This function measures transit depths using circular apertures of varying
    radii and assesses the stability of these measurements. Stable depths
    across apertures suggest an on-target transit signal, while variable
    depths indicate potential contamination or off-target sources.

    Args:
        tpf_data: 3D array of TPF flux values with shape (n_times, n_rows, n_cols).
        time: 1D array of observation times, same length as tpf_data's first axis.
        transit_params: Transit parameters (period, t0, duration, depth).
        aperture_radii: List of aperture radii to test (in pixels). If None,
            defaults to [1.0, 1.5, 2.0, 2.5, 3.0].
        center: Center position for apertures as (row, col). If None, defaults
            to the center of the TPF.

    Returns:
        ApertureDependenceResult containing:
        - depths_by_aperture: Measured depths for each aperture size
        - stability_metric: Score indicating measurement consistency
        - recommended_aperture: Suggested optimal aperture size
        - depth_variance: Variance across aperture depths

    Raises:
        ValueError: If input arrays have incompatible shapes or insufficient data.

    Example:
        >>> import numpy as np
        >>> # Create synthetic TPF data (10 time points, 11x11 pixels)
        >>> tpf = np.ones((100, 11, 11)) * 1000.0
        >>> time = np.linspace(0, 10, 100)
        >>> params = TransitParams(period=2.0, t0=1.0, duration=0.2)
        >>> result = compute_aperture_dependence(tpf, time, params)
        >>> result.stability_metric > 0.5
        True
    """
    # Validate inputs
    if tpf_data.ndim != 3:
        raise ValueError(f"tpf_data must be 3D, got shape {tpf_data.shape}")

    n_times, n_rows, n_cols = tpf_data.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"tpf_data spatial dimensions must be non-zero, got shape {tpf_data.shape}")

    if len(time) != n_times:
        raise ValueError(
            f"time array length ({len(time)}) must match tpf_data first axis ({n_times})"
        )

    cadence_mask = default_cadence_mask(
        time=time,
        flux=tpf_data,
        quality=np.zeros(int(time.shape[0]), dtype=np.int32),
        require_finite_pixels=True,
    )
    n_total = int(time.shape[0])
    n_used = int(np.sum(cadence_mask))
    n_dropped = n_total - n_used

    tpf_data = tpf_data[cadence_mask]
    time = time[cadence_mask]
    n_times = int(time.size)

    if n_times < 10:
        raise ValueError(f"Need at least 10 valid time points, got {n_times}")

    # Set defaults
    if aperture_radii is None:
        aperture_radii = list(DEFAULT_APERTURE_RADII)

    if center is None:
        center = ((n_rows - 1) / 2.0, (n_cols - 1) / 2.0)

    # Compute transit mask once
    in_transit_mask = _compute_transit_mask(time, transit_params)

    # Verify we have both in and out of transit data
    n_in_transit = np.sum(in_transit_mask)
    n_out_of_transit = np.sum(~in_transit_mask)

    if n_in_transit < 3:
        raise ValueError(
            f"Insufficient in-transit data points ({n_in_transit}). "
            "Check transit_params or time coverage."
        )

    if n_out_of_transit < 3:
        raise ValueError(
            f"Insufficient out-of-transit data points ({n_out_of_transit}). "
            "Check transit_params or time coverage."
        )

    # Identify distinct transit epochs (for interpretability / quality gating).
    transit_epochs: list[tuple[float, NDArray[np.int_]]] = []
    in_transit_indices = np.where(in_transit_mask)[0]
    if in_transit_indices.size > 0:
        # Group consecutive in-transit indices into epochs.
        # Threshold 10 indices ~= a few hours at 2-minute cadence, but can be larger
        # at 30-minute cadence; this is an intentionally loose grouping heuristic.
        breaks = np.where(np.diff(in_transit_indices) > 10)[0] + 1
        epoch_groups = np.split(in_transit_indices, breaks)
        for group in epoch_groups:
            if group.size == 0:
                continue
            mid_idx = int(group[group.size // 2])
            transit_epochs.append((float(time[mid_idx]), group.astype(int)))

    local_baseline_window = 0.5  # days (symmetric around each epoch midpoint)

    # Create background annulus mask for background subtraction
    # Use annulus starting beyond the largest aperture
    max_aperture = max(aperture_radii)
    bg_inner_radius = max(max_aperture + 1.0, 4.0)
    bg_outer_radius = bg_inner_radius + 2.0

    # Ensure annulus fits within TPF
    max_radius_possible = min(
        center[0],
        center[1],
        n_rows - 1 - center[0],
        n_cols - 1 - center[1],
    )

    # Adjust annulus radii if needed to fit within TPF
    if bg_outer_radius > max_radius_possible:
        bg_outer_radius = max_radius_possible
        bg_inner_radius = max(bg_outer_radius - 2.0, max_aperture + 0.5)

    bg_annulus_mask = _create_background_annulus_mask(
        shape=(n_rows, n_cols),
        inner_radius=bg_inner_radius,
        outer_radius=bg_outer_radius,
        center=center,
    )

    # Count pixels in background annulus
    n_bg_pixels = int(np.sum(bg_annulus_mask))

    # Compute per-cadence background as median of annulus pixels
    # If annulus has too few pixels, fall back to no background subtraction
    if n_bg_pixels >= 3:
        bg_per_cadence = np.array(
            [np.nanmedian(tpf_data[t][bg_annulus_mask]) for t in range(n_times)]
        )
    else:
        bg_per_cadence = np.zeros(n_times)

    # Measure depth for each aperture
    depths_by_aperture: dict[float, float] = {}
    drift_by_aperture: dict[float, float] = {}
    n_apertures_local_used = 0
    n_apertures_sector_fallback = 0

    for radius in aperture_radii:
        # Create aperture mask
        aperture_mask = _create_circular_aperture_mask(
            shape=(n_rows, n_cols),
            radius=radius,
            center=center,
        )

        n_aperture_pixels = int(np.sum(aperture_mask))

        # Compute aperture photometry with background subtraction
        raw_aperture_flux = np.nansum(tpf_data[:, aperture_mask], axis=1)
        aperture_flux = raw_aperture_flux - (bg_per_cadence * n_aperture_pixels)

        transit_depths = []

        # Measure depth for each transit using local baseline
        for epoch_time, transit_indices in transit_epochs:
            # Define local window around this transit
            local_mask = (np.abs(time - epoch_time) <= local_baseline_window) & ~in_transit_mask

            if np.sum(local_mask) < 5:
                # Not enough local baseline points, skip this transit
                continue

            # Use background-subtracted flux if positive, else raw
            if np.nanmedian(aperture_flux[local_mask]) > 0:
                flux_to_use = aperture_flux
            else:
                flux_to_use = raw_aperture_flux

            local_out_median = np.nanmedian(flux_to_use[local_mask])
            in_transit_median = np.nanmedian(flux_to_use[transit_indices])

            if local_out_median > 0:
                depth = (1.0 - in_transit_median / local_out_median) * 1_000_000
                transit_depths.append(depth)

        # Average depth across transits for this aperture
        if len(transit_depths) > 0:
            n_apertures_local_used += 1
            depths_by_aperture[radius] = float(np.mean(transit_depths))
        else:
            # Fallback to sector-wide if no local depths computed
            n_apertures_sector_fallback += 1
            out_of_transit_median = np.nanmedian(aperture_flux[~in_transit_mask])
            if out_of_transit_median > 0:
                normalized_flux = aperture_flux / out_of_transit_median
            else:
                raw_out_median = np.nanmedian(raw_aperture_flux[~in_transit_mask])
                if raw_out_median > 0:
                    normalized_flux = raw_aperture_flux / raw_out_median
                else:
                    continue
            try:
                depth_ppm = _compute_depth_from_lightcurve(normalized_flux, in_transit_mask)
                depths_by_aperture[radius] = depth_ppm
            except ValueError:
                continue

        # Drift diagnostic (out-of-transit median early vs late in the sector).
        try:
            mid = float(np.nanmedian(time))
            oot = ~in_transit_mask
            early = float(np.nanmedian(aperture_flux[oot & (time < mid)]))
            late = float(np.nanmedian(aperture_flux[oot & (time >= mid)]))
            drift_by_aperture[radius] = float((late - early) / early) if early != 0 else float("nan")
        except Exception:
            drift_by_aperture[radius] = float("nan")

    if len(depths_by_aperture) < 2:
        raise ValueError(
            "Could not compute depths for at least 2 apertures. Check input data and parameters."
        )

    # Compute statistics
    depth_values = list(depths_by_aperture.values())
    stability_metric = _compute_stability_metric(
        depth_values,
        expected_depth_ppm=transit_params.depth * 1_000_000,
    )

    depth_variance = float(np.var(depth_values))

    recommended_aperture = _select_recommended_aperture(
        depths_by_aperture,
        stability_metric,
    )

    flags: list[str] = []
    if n_dropped > 0:
        flags.append("dropped_invalid_cadences")
    if int(len(transit_epochs)) < 2:
        flags.append("low_n_transit_epochs")
    if int(n_in_transit) < 20:
        flags.append("low_n_in_transit_cadences")
    if n_bg_pixels < 3:
        flags.append("no_background_subtraction")
    elif n_bg_pixels < 10:
        flags.append("background_annulus_small")
    if any(float(v) < 0 for v in depths_by_aperture.values()):
        flags.append("negative_depths_present")
    if n_apertures_sector_fallback > 0:
        flags.append("sector_fallback_used")

    drift_rec = drift_by_aperture.get(float(recommended_aperture))
    if drift_rec is not None and np.isfinite(drift_rec) and abs(float(drift_rec)) >= 0.01:
        flags.append("high_out_of_transit_drift_recommended_aperture")

    return ApertureDependenceResult(
        depths_by_aperture=depths_by_aperture,
        stability_metric=stability_metric,
        recommended_aperture=recommended_aperture,
        depth_variance=depth_variance,
        n_in_transit_cadences=int(n_in_transit),
        n_out_of_transit_cadences=int(n_out_of_transit),
        n_transit_epochs=int(len(transit_epochs)),
        baseline_mode="local_window_per_epoch",
        local_baseline_window_days=float(local_baseline_window),
        background_mode="annulus_median_per_cadence" if n_bg_pixels >= 3 else "none",
        background_annulus_radii_pixels=(float(bg_inner_radius), float(bg_outer_radius)),
        n_background_pixels=int(n_bg_pixels),
        drift_fraction_recommended=float(drift_rec) if drift_rec is not None and np.isfinite(drift_rec) else None,
        flags=flags,
        notes={
            "center_policy": "tpf_center",
            "n_cadences_total": int(n_total),
            "n_cadences_used": int(n_used),
            "n_cadences_dropped": int(n_dropped),
            "n_apertures_local_used": int(n_apertures_local_used),
            "n_apertures_sector_fallback": int(n_apertures_sector_fallback),
            "drift_fraction_by_aperture": {str(k): float(v) for k, v in drift_by_aperture.items()},
        },
    )
