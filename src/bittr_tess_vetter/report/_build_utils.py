"""Reusable helper utilities for report build assembly."""

from __future__ import annotations

from typing import Any

import numpy as np

from bittr_tess_vetter.domain.lightcurve import LightCurveData


def _downsample_preserving_transits(
    time: np.ndarray,
    flux: np.ndarray,
    transit_mask: np.ndarray,
    max_points: int,
) -> tuple[list[float], list[float], list[bool]]:
    """Downsample LC to at most *max_points*, prioritizing in-transit points.

    Hard cap: output never exceeds *max_points*.  If in-transit points
    alone exceed the budget, they are uniformly thinned to fit.
    """
    in_transit_idx = np.where(transit_mask)[0]
    oot_idx = np.where(~transit_mask)[0]

    if len(time) <= max_points:
        return time.tolist(), flux.tolist(), transit_mask.tolist()

    if len(in_transit_idx) >= max_points:
        # In-transit alone exceeds budget — uniformly thin them
        pick = np.round(np.linspace(0, len(in_transit_idx) - 1, max_points)).astype(int)
        keep = np.sort(in_transit_idx[pick])
        return time[keep].tolist(), flux[keep].tolist(), transit_mask[keep].tolist()

    # Keep all in-transit; evenly sample OOT to fill remaining budget.
    # Use linspace to pick evenly-spaced indices across the full OOT
    # array, avoiding the start-biased truncation of [::step][:budget].
    n_oot_budget = max_points - len(in_transit_idx)
    pick = np.round(np.linspace(0, len(oot_idx) - 1, n_oot_budget)).astype(int)
    sampled_oot = oot_idx[pick]

    keep = np.sort(np.concatenate([in_transit_idx, sampled_oot]))
    return time[keep].tolist(), flux[keep].tolist(), transit_mask[keep].tolist()


def _downsample_phase_preserving_transit(
    phase: np.ndarray,
    flux: np.ndarray,
    max_points: int,
    near_transit_half_phase: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample phase-folded data to at most *max_points*.

    Near-transit points (within ±near_transit_half_phase) are prioritized.
    Hard cap: output never exceeds *max_points*.  If near-transit points
    alone exceed the budget, they are uniformly thinned to fit.

    Args:
        phase: Phase values (centered on 0).
        flux: Flux values.
        max_points: Maximum number of output points.
        near_transit_half_phase: Half-width of the near-transit
            preservation window in phase units.  Defaults to 0.1 for
            backward compatibility, but callers should pass a
            duration-based value (e.g. 3 * transit_duration_phase).
    """
    if len(phase) <= max_points:
        return phase, flux

    near_transit = np.abs(phase) < near_transit_half_phase
    near_idx = np.where(near_transit)[0]
    far_idx = np.where(~near_transit)[0]

    if len(near_idx) >= max_points:
        # Near-transit alone exceeds budget — uniformly thin them
        pick = np.round(np.linspace(0, len(near_idx) - 1, max_points)).astype(int)
        keep = np.sort(near_idx[pick])
        return phase[keep], flux[keep]

    # Keep all near-transit; evenly sample far points to fill budget.
    # Use linspace to pick evenly-spaced indices across the full far
    # array.  The old [::step][:budget] approach was biased toward the
    # start of the sorted-by-phase array, which systematically clipped
    # positive-phase (right-side) points and caused visible asymmetry
    # in the phase-folded plot.
    n_far_budget = max_points - len(near_idx)
    if len(far_idx) <= n_far_budget:
        return phase, flux

    pick = np.round(np.linspace(0, len(far_idx) - 1, n_far_budget)).astype(int)
    sampled_far = far_idx[pick]

    keep = np.sort(np.concatenate([near_idx, sampled_far]))
    return phase[keep], flux[keep]


def _suggest_flux_y_range(
    flux: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    padding_fraction: float = 0.1,
) -> tuple[float, float] | None:
    """Suggest a robust y-axis range for phase-folded display.

    Uses percentile clipping plus symmetric padding to keep shallow transits
    visible in the presence of outliers. This is display metadata only.
    """
    finite = flux[np.isfinite(flux)]
    if len(finite) < 3:
        return None

    lo, hi = np.percentile(finite, [lower_percentile, upper_percentile])
    lo_f = float(lo)
    hi_f = float(hi)
    if not np.isfinite(lo_f) or not np.isfinite(hi_f):
        return None

    span = hi_f - lo_f
    if span <= 0.0:
        # Degenerate case: nearly constant flux; add tiny guard band.
        pad = max(abs(lo_f), 1.0) * 1e-6
        return (lo_f - pad, hi_f + pad)

    pad = span * max(padding_fraction, 0.0)
    return (lo_f - pad, hi_f + pad)


def _depth_ppm_to_flux(depth_ppm: float | None) -> float | None:
    """Convert positive depth in ppm to normalized flux reference level."""
    if depth_ppm is None or not np.isfinite(depth_ppm) or depth_ppm <= 0.0:
        return None
    return float(1.0 - (depth_ppm / 1e6))


def _estimate_flux_err_fallback(flux: np.ndarray) -> float:
    """Estimate deterministic positive flux uncertainty from robust scatter."""
    finite_flux = np.asarray(flux[np.isfinite(flux)], dtype=np.float64)
    if len(finite_flux) == 0:
        return 1e-6

    # Prefer point-to-point scatter to suppress low-frequency variability.
    sigma_pp: float | None = None
    if len(finite_flux) >= 3:
        diffs = np.diff(finite_flux)
        mad_diff = float(np.median(np.abs(diffs - np.median(diffs))))
        if np.isfinite(mad_diff) and mad_diff > 0.0:
            sigma_pp = (1.4826 * mad_diff) / np.sqrt(2.0)

    mad_flux = float(np.median(np.abs(finite_flux - np.median(finite_flux))))
    sigma_flux = (1.4826 * mad_flux) if np.isfinite(mad_flux) and mad_flux > 0.0 else 0.0

    sigma = sigma_pp if sigma_pp is not None else sigma_flux
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = sigma_flux
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(finite_flux))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1e-6
    return float(max(sigma, 1e-12))


def _to_internal_lightcurve(lc: Any) -> LightCurveData:
    """Convert a report input light curve to internal immutable LightCurveData."""
    to_internal = getattr(lc, "to_internal", None)
    if callable(to_internal):
        internal = to_internal()
        if isinstance(internal, LightCurveData):
            if getattr(lc, "flux_err", None) is not None:
                return internal
            internal_flux = np.asarray(internal.flux, dtype=np.float64)
            internal_valid = np.asarray(internal.valid_mask, dtype=np.bool_)
            flux_for_estimate = internal_flux[internal_valid & np.isfinite(internal_flux)]
            if flux_for_estimate.size == 0:
                flux_for_estimate = internal_flux[np.isfinite(internal_flux)]
            fallback_sigma = _estimate_flux_err_fallback(flux_for_estimate)
            flux_err = np.full(len(internal.time), fallback_sigma, dtype=np.float64)
            return LightCurveData(
                time=np.asarray(internal.time, dtype=np.float64),
                flux=np.asarray(internal.flux, dtype=np.float64),
                flux_err=flux_err,
                quality=np.asarray(internal.quality, dtype=np.int32),
                valid_mask=np.asarray(internal.valid_mask, dtype=np.bool_),
                tic_id=int(internal.tic_id),
                sector=int(internal.sector),
                cadence_seconds=float(internal.cadence_seconds),
                provenance=internal.provenance,
            )

    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    if time.shape != flux.shape:
        raise ValueError(
            f"time and flux must have the same length, got {len(time)} and {len(flux)}"
        )
    n_points = len(time)

    flux_err_in = getattr(lc, "flux_err", None)
    if flux_err_in is not None:
        flux_err = np.asarray(flux_err_in, dtype=np.float64)
        if flux_err.shape != time.shape:
            raise ValueError(
                "flux_err must have the same length as time/flux, "
                f"got {len(flux_err)} vs {len(time)}"
            )
    else:
        valid_mask_in = getattr(lc, "valid_mask", None)
        if valid_mask_in is not None:
            provisional_valid = np.asarray(valid_mask_in, dtype=np.bool_)
            if provisional_valid.shape != time.shape:
                raise ValueError(
                    "valid_mask must have the same length as time/flux, "
                    f"got {len(provisional_valid)} vs {len(time)}"
                )
        else:
            provisional_valid = np.ones(n_points, dtype=np.bool_)
        flux_for_estimate = flux[provisional_valid & np.isfinite(flux)]
        if flux_for_estimate.size == 0:
            flux_for_estimate = flux[np.isfinite(flux)]
        fallback_sigma = _estimate_flux_err_fallback(flux_for_estimate)
        flux_err = np.full(n_points, fallback_sigma, dtype=np.float64)

    quality_in = getattr(lc, "quality", None)
    if quality_in is not None:
        quality = np.asarray(quality_in, dtype=np.int32)
        if quality.shape != time.shape:
            raise ValueError(
                "quality must have the same length as time/flux, "
                f"got {len(quality)} vs {len(time)}"
            )
    else:
        quality = np.zeros(n_points, dtype=np.int32)

    valid_mask_in = getattr(lc, "valid_mask", None)
    if valid_mask_in is not None:
        valid_mask = np.asarray(valid_mask_in, dtype=np.bool_)
        if valid_mask.shape != time.shape:
            raise ValueError(
                "valid_mask must have the same length as time/flux, "
                f"got {len(valid_mask)} vs {len(time)}"
            )
    else:
        valid_mask = np.ones(n_points, dtype=np.bool_)

    finite_mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    valid_mask = valid_mask & finite_mask

    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=0,
        sector=0,
        cadence_seconds=120.0,
    )


def _get_valid_time_flux(lc: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, mask-filtered time/flux arrays."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(flux)
    if lc.valid_mask is not None:
        valid = valid & np.asarray(lc.valid_mask, dtype=np.bool_)
    return time[valid], flux[valid]


def _thin_evenly(arr: np.ndarray, max_points: int) -> np.ndarray:
    """Evenly thin a 1D array to at most max_points elements."""
    if len(arr) <= max_points:
        return arr
    pick = np.round(np.linspace(0, len(arr) - 1, max_points)).astype(int)
    return arr[pick]


def _red_noise_beta(
    residuals: np.ndarray,
    times: np.ndarray,
    *,
    bin_size_days: float,
) -> float | None:
    if len(residuals) < 10 or bin_size_days <= 0:
        return None
    sort_idx = np.argsort(times)
    r = residuals[sort_idx]
    t = times[sort_idx]
    t_min = float(t[0])
    t_max = float(t[-1])
    if t_max - t_min < 2.0 * bin_size_days:
        return None
    n_bins = max(3, int((t_max - t_min) / bin_size_days))
    edges = np.linspace(t_min, t_max, n_bins + 1)
    means = []
    counts = []
    for i in range(n_bins):
        m = (t >= edges[i]) & (t < edges[i + 1])
        if int(np.sum(m)) >= 3:
            means.append(float(np.mean(r[m])))
            counts.append(int(np.sum(m)))
    if len(means) < 3:
        return None
    observed = float(np.std(np.asarray(means, dtype=np.float64), ddof=1))
    point = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    avg_per_bin = float(np.mean(counts)) if len(counts) > 0 else 0.0
    if point <= 0 or avg_per_bin <= 0:
        return None
    expected = point / np.sqrt(avg_per_bin)
    if expected <= 0:
        return None
    return float(max(observed / expected, 1.0))


def _get_valid_time_flux_quality(
    lc: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return finite, mask-filtered time/flux/(optional)quality arrays."""
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64)
    valid = np.isfinite(time) & np.isfinite(flux)
    if lc.valid_mask is not None:
        valid = valid & np.asarray(lc.valid_mask, dtype=np.bool_)
    quality = None
    if lc.quality is not None:
        q = np.asarray(lc.quality)
        if q.shape == time.shape:
            quality = q[valid]
    return time[valid], flux[valid], quality


def _bin_phase_data(
    phase: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    bin_minutes: float,
    phase_range: tuple[float, float] | None = None,
) -> tuple[list[float], list[float], list[float | None]]:
    """Bin phase-folded data by phase.

    Uses vectorized np.digitize + np.bincount for O(n_points + n_bins)
    performance instead of per-bin masking.

    Args:
        phase: Phase values (centered on 0).
        flux: Flux values.
        period_days: Orbital period in days.
        bin_minutes: Bin size in minutes.
        phase_range: If provided, only bin data within this phase window.
            Bins are placed to cover the window; data outside is ignored.
            If None, bins span the full range of the input data.

    Returns:
        Tuple of (bin_centers, bin_flux, bin_err) lists.
        bin_err entries are None for single-point bins.
    """
    if len(phase) == 0:
        return [], [], []

    # Convert bin size from minutes to phase units
    bin_phase = (bin_minutes / 60.0 / 24.0) / period_days

    # Determine bin edges — either from phase_range or data extent
    if phase_range is not None:
        phase_min, phase_max = phase_range
        # Filter data to the window for binning
        in_window = (phase >= phase_min) & (phase <= phase_max)
        phase = phase[in_window]
        flux = flux[in_window]
        if len(phase) == 0:
            return [], [], []
    else:
        phase_min = float(np.min(phase))
        phase_max = float(np.max(phase))

    n_bins = max(1, int(np.ceil((phase_max - phase_min) / bin_phase)))
    bin_edges = np.linspace(phase_min, phase_max, n_bins + 1)

    # Vectorized binning via digitize (returns 1-based bin indices)
    idx = np.digitize(phase, bin_edges) - 1
    # Clamp right-edge points into last bin (digitize puts them at n_bins)
    idx = np.clip(idx, 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins)[:n_bins]
    sums = np.bincount(idx, weights=flux, minlength=n_bins)[:n_bins]
    sumsq = np.bincount(idx, weights=flux * flux, minlength=n_bins)[:n_bins]

    # Only emit bins that have data
    occupied = counts > 0
    occ_idx = np.where(occupied)[0]

    bin_centers_arr = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    centers: list[float] = bin_centers_arr[occ_idx].tolist()
    mean = sums[occupied] / counts[occupied]
    fluxes: list[float] = mean.tolist()

    # SEM: sqrt(sample_variance / n) where sample_variance uses ddof=1
    errors: list[float | None] = []
    occ_counts = counts[occupied]
    occ_sumsq = sumsq[occupied]
    occ_sums = sums[occupied]
    for i in range(len(occ_counts)):
        n = int(occ_counts[i])
        if n > 1:
            var = (occ_sumsq[i] - occ_sums[i] ** 2 / n) / (n - 1)
            errors.append(float(np.sqrt(max(var, 0.0) / n)))
        else:
            errors.append(None)

    return centers, fluxes, errors
