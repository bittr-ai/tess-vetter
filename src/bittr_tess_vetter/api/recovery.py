"""Transit recovery API for active stars.

This module provides public API functions for recovering transit signals from
active stars with significant stellar variability. The workflow involves:

1. Detrending stellar variability (using harmonic subtraction or wotan)
2. Stacking multiple transits for improved SNR
3. Fitting a trapezoid model to recover depth and duration

Novelty: standard (implements well-established detrending and stacking techniques)

References:
    [1] Hippke et al. 2019, AJ 158, 143 (2019AJ....158..143H)
        Wotan detrending methods benchmark and recommendations
    [2] Hippke & Heller 2019, A&A 623, A39 (2019A&A...623A..39H)
        Transit Least Squares and phase-folding methodology
    [3] Barros et al. 2020, A&A 634, A75 (2020A&A...634A..75B)
        GP-based stellar variability modeling for transit recovery
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from bittr_tess_vetter.api.references import (
    AIGRAIN_2016,
    BARROS_2020,
    HIPPKE_2019_WOTAN,
    HIPPKE_HELLER_2019_TLS,
    KOVACS_2002,
    LUGER_2016,
    MORVAN_2020,
    PETIGURA_2012,
    cite,
    cites,
)
from bittr_tess_vetter.api.types import Candidate, LightCurve
from bittr_tess_vetter.compute.detrend import WOTAN_AVAILABLE
from bittr_tess_vetter.recovery.primitives import (
    compute_detection_snr,
    detrend_for_recovery,
    fit_trapezoid,
)
from bittr_tess_vetter.recovery.primitives import (
    stack_transits as _stack_transits,
)
from bittr_tess_vetter.recovery.pipeline import recover_transit_timeseries

if TYPE_CHECKING:
    from bittr_tess_vetter.activity.result import ActivityResult

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [
    ref.to_dict()
    for ref in [
        HIPPKE_2019_WOTAN,
        HIPPKE_HELLER_2019_TLS,
        BARROS_2020,
        AIGRAIN_2016,
        PETIGURA_2012,
        LUGER_2016,
        KOVACS_2002,
        MORVAN_2020,
    ]
]


@dataclass(frozen=True)
class RecoveryResult:
    """Transit recovery result from active star.

    Contains the detrended and stacked transit data along with
    fitted depth and detection significance metrics.

    Attributes:
        detrend_method: Method used for detrending ("harmonic", "wotan_biweight", etc.)
        rotation_period_used: Stellar rotation period used for detrending (if any)
        n_transits_stacked: Number of distinct transits stacked
        phase: Stacked phase curve bin centers (0-1, transit at 0.5)
        flux: Stacked and binned flux values
        flux_err: Propagated flux uncertainties
        fitted_depth_ppm: Fitted transit depth in parts per million
        fitted_depth_err_ppm: Uncertainty on fitted depth in ppm
        fitted_duration_hours: Fitted transit duration in hours
        detection_snr: Detection signal-to-noise ratio (depth / depth_err)
        converged: Whether the trapezoid fit converged successfully
    """

    detrend_method: str
    rotation_period_used: float | None
    n_transits_stacked: int
    phase: list[float]
    flux: list[float]
    flux_err: list[float]
    fitted_depth_ppm: float
    fitted_depth_err_ppm: float
    fitted_duration_hours: float
    detection_snr: float
    converged: bool


@cites(
    cite(HIPPKE_2019_WOTAN, "§2-3 detrending methods benchmark"),
    cite(HIPPKE_HELLER_2019_TLS, "§2,4 TLS for active stars"),
    cite(KOVACS_2002, "Phase-folding and stacking transit signals (BLS lineage)"),
)
def recover_transit(
    lc: LightCurve,
    candidate: Candidate,
    *,
    activity: ActivityResult | None = None,
    detrend_method: Literal["harmonic", "wotan_biweight", "wotan_gp"] = "harmonic",
    rotation_period: float | None = None,
    n_harmonics: int = 3,
    phase_bins: int = 100,
) -> RecoveryResult:
    """Recover transit signal from active star.

    Removes stellar variability while preserving transit signal,
    then stacks all transits for improved SNR. This is the main
    entry point for transit recovery from active stars.

    The workflow is:
    1. Create a transit mask to exclude in-transit points from detrending
    2. Detrend the light curve using the specified method
    3. Phase-fold and stack all transits
    4. Fit a trapezoid model to estimate depth and duration

    Args:
        lc: Light curve data (ideally with flares masked beforehand)
        candidate: Transit candidate with ephemeris (period, t0, duration)
        activity: Activity result from characterize_activity(). If provided
            and rotation_period is not specified, uses activity.rotation_period.
        detrend_method: Detrending algorithm to use:
            - "harmonic": Fourier series subtraction (requires rotation_period)
            - "wotan_biweight": Robust biweight filter (requires wotan package)
            - "wotan_gp": Gaussian process detrending (requires wotan package)
        rotation_period: Stellar rotation period in days. Required for "harmonic"
            method. If None and activity is provided, uses activity.rotation_period.
        n_harmonics: Number of Fourier harmonics for "harmonic" method (1-5).
        phase_bins: Number of phase bins for stacking (default 100).

    Returns:
        RecoveryResult with stacked transit and fitted depth.

    Raises:
        ValueError: If detrend_method is "harmonic" and no rotation_period
            is available (neither directly nor from activity).
        ImportError: If wotan method requested but wotan is not installed.

    Example:
        >>> from bittr_tess_vetter.api import LightCurve, Candidate, Ephemeris
        >>> from bittr_tess_vetter.api.recovery import recover_transit
        >>> lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        >>> eph = Ephemeris(period_days=8.46, t0_btjd=1330.4, duration_hours=3.5)
        >>> candidate = Candidate(ephemeris=eph)
        >>> result = recover_transit(lc, candidate, rotation_period=4.86)
        >>> print(f"Depth: {result.fitted_depth_ppm:.0f} +/- {result.fitted_depth_err_ppm:.0f} ppm")

    Novelty: standard

    References:
        [1] Hippke et al. 2019, AJ 158, 143 (2019AJ....158..143H)
            Section 2: Methods comparison; Section 3: Benchmark results
        [2] Hippke & Heller 2019, A&A 623, A39 (2019A&A...623A..39H)
            Section 2: Algorithm; Section 4: Application to active stars
    """
    # Resolve rotation period from activity if not provided directly
    rotation_period_used = rotation_period
    if rotation_period_used is None and activity is not None:
        rotation_period_used = activity.rotation_period

    # Validate method requirements
    if detrend_method == "harmonic" and rotation_period_used is None:
        raise ValueError(
            "rotation_period is required for harmonic detrending method. "
            "Provide rotation_period directly or pass an ActivityResult."
        )

    if detrend_method.startswith("wotan_") and not WOTAN_AVAILABLE:
        raise ImportError(
            f"wotan is required for {detrend_method} detrending. Install with: pip install wotan"
        )

    # Convert LightCurve to internal format
    internal_lc = lc.to_internal()
    time = internal_lc.time
    flux = internal_lc.flux
    flux_err = internal_lc.flux_err

    # Extract ephemeris
    period = candidate.ephemeris.period_days
    t0 = candidate.ephemeris.t0_btjd
    duration_hours = candidate.ephemeris.duration_hours
    duration_days = duration_hours / 24.0

    # Create transit mask (True = in-transit points to exclude from detrending)
    phase = ((time - t0) / period) % 1.0
    # Transit is at phase 0, mask points within +/- 1.5 * transit duration
    transit_phase_width = 1.5 * duration_days / period
    transit_mask = (phase < transit_phase_width) | (phase > (1.0 - transit_phase_width))

    # Detrend the light curve
    detrended_flux = detrend_for_recovery(
        time=time,
        flux=flux,
        transit_mask=transit_mask,
        method=detrend_method,
        rotation_period=rotation_period_used,
        n_harmonics=n_harmonics,
    )

    # Stack transits
    stacked = _stack_transits(
        time=time,
        flux=detrended_flux,
        flux_err=flux_err,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        phase_bins=phase_bins,
    )

    # Fit trapezoid model
    # Initial depth estimate from candidate if available, otherwise from stacked data
    initial_depth = 0.001  # Default 1000 ppm
    if candidate.depth is not None:
        initial_depth = candidate.depth

    # Initial duration in phase units
    initial_duration_phase = duration_days / period

    trap_fit = fit_trapezoid(
        phase=stacked.phase,
        flux=stacked.flux,
        flux_err=stacked.flux_err,
        initial_depth=initial_depth,
        initial_duration_phase=initial_duration_phase,
    )

    # Convert depth to ppm
    fitted_depth_ppm = trap_fit.depth * 1e6
    fitted_depth_err_ppm = trap_fit.depth_err * 1e6

    # Convert duration from phase to hours
    fitted_duration_hours = trap_fit.duration_phase * period * 24.0

    # Compute detection SNR
    detection_snr = compute_detection_snr(trap_fit.depth, trap_fit.depth_err)

    return RecoveryResult(
        detrend_method=detrend_method,
        rotation_period_used=rotation_period_used,
        n_transits_stacked=stacked.n_transits,
        phase=stacked.phase.tolist(),
        flux=stacked.flux.tolist(),
        flux_err=stacked.flux_err.tolist(),
        fitted_depth_ppm=fitted_depth_ppm,
        fitted_depth_err_ppm=fitted_depth_err_ppm,
        fitted_duration_hours=fitted_duration_hours,
        detection_snr=detection_snr,
        converged=trap_fit.converged,
    )


@cites(
    cite(HIPPKE_2019_WOTAN, "wotan methods comparison"),
)
def detrend(
    lc: LightCurve,
    candidate: Candidate,
    *,
    method: Literal["harmonic", "wotan_biweight", "wotan_gp"] = "wotan_biweight",
    rotation_period: float | None = None,
    window_length: float = 0.5,
    n_harmonics: int = 3,
) -> LightCurve:
    """Detrend light curve while preserving transits.

    Removes stellar variability from the light curve while masking
    transit windows to avoid distorting the transit signal. This is
    a standalone detrending function for use in custom workflows.

    Args:
        lc: Light curve data
        candidate: Transit candidate (used to mask transits during detrending)
        method: Detrending algorithm:
            - "harmonic": Fourier series (requires rotation_period)
            - "wotan_biweight": Robust biweight filter (default, requires wotan)
            - "wotan_gp": Gaussian process (requires wotan)
        rotation_period: Stellar rotation period in days. Required for "harmonic".
        window_length: Window size in days for wotan methods. Default 0.5 days.
        n_harmonics: Number of harmonics for "harmonic" method.

    Returns:
        Detrended LightCurve with same time/flux_err arrays.

    Raises:
        ValueError: If method is "harmonic" and rotation_period is None.
        ImportError: If wotan method requested but wotan is not installed.

    Example:
        >>> from bittr_tess_vetter.api import LightCurve, Candidate, Ephemeris
        >>> from bittr_tess_vetter.api.recovery import detrend
        >>> lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        >>> eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
        >>> candidate = Candidate(ephemeris=eph)
        >>> flat_lc = detrend(lc, candidate, method="wotan_biweight")

    Novelty: standard

    References:
        [1] Hippke et al. 2019, AJ 158, 143 (2019AJ....158..143H)
            Wotan methods comparison and recommendations
    """
    # Validate method requirements
    if method == "harmonic" and rotation_period is None:
        raise ValueError("rotation_period is required for harmonic detrending method")

    if method.startswith("wotan_") and not WOTAN_AVAILABLE:
        raise ImportError(
            f"wotan is required for {method} detrending. Install with: pip install wotan"
        )

    # Convert LightCurve to internal format
    internal_lc = lc.to_internal()
    time = internal_lc.time
    flux = internal_lc.flux

    # Extract ephemeris
    period = candidate.ephemeris.period_days
    t0 = candidate.ephemeris.t0_btjd
    duration_hours = candidate.ephemeris.duration_hours
    duration_days = duration_hours / 24.0

    # Create transit mask
    phase = ((time - t0) / period) % 1.0
    transit_phase_width = 1.5 * duration_days / period
    transit_mask = (phase < transit_phase_width) | (phase > (1.0 - transit_phase_width))

    # Detrend
    detrended_flux = detrend_for_recovery(
        time=time,
        flux=flux,
        transit_mask=transit_mask,
        method=method,
        rotation_period=rotation_period,
        n_harmonics=n_harmonics,
        window_length=window_length,
    )

    # Return new LightCurve with detrended flux
    return LightCurve(
        time=np.array(time),
        flux=detrended_flux,
        flux_err=np.array(internal_lc.flux_err) if lc.flux_err is not None else None,
        quality=np.array(internal_lc.quality) if lc.quality is not None else None,
        valid_mask=np.array(internal_lc.valid_mask) if lc.valid_mask is not None else None,
    )


def stack_transits(
    lc: LightCurve,
    candidate: Candidate,
    *,
    phase_bins: int = 100,
) -> tuple[list[float], list[float], list[float], int]:
    """Phase-fold and stack all transits.

    Uses inverse-variance weighting for optimal binning of the
    phase-folded light curve. This is useful for visualization
    and custom analysis workflows.

    Args:
        lc: Detrended light curve (use detrend() first for active stars)
        candidate: Transit candidate with ephemeris
        phase_bins: Number of phase bins for stacking. Default 100.

    Returns:
        Tuple of (phase, flux, flux_err, n_transits) where:
            - phase: List of phase bin centers (0-1, transit at 0.5)
            - flux: List of binned flux values
            - flux_err: List of propagated uncertainties
            - n_transits: Number of distinct transits stacked

    Example:
        >>> from bittr_tess_vetter.api import LightCurve, Candidate, Ephemeris
        >>> from bittr_tess_vetter.api.recovery import stack_transits
        >>> lc = LightCurve(time=time, flux=detrended_flux, flux_err=flux_err)
        >>> eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
        >>> candidate = Candidate(ephemeris=eph)
        >>> phase, flux, flux_err, n_transits = stack_transits(lc, candidate)
        >>> print(f"Stacked {n_transits} transits")

    Novelty: standard
    """
    # Convert LightCurve to internal format
    internal_lc = lc.to_internal()
    time = internal_lc.time
    flux = internal_lc.flux
    flux_err = internal_lc.flux_err

    # Extract ephemeris
    period = candidate.ephemeris.period_days
    t0 = candidate.ephemeris.t0_btjd
    duration_hours = candidate.ephemeris.duration_hours

    # Stack transits using internal primitive
    stacked = _stack_transits(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        phase_bins=phase_bins,
    )

    return (
        stacked.phase.tolist(),
        stacked.flux.tolist(),
        stacked.flux_err.tolist(),
        stacked.n_transits,
    )
