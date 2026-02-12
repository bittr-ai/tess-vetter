"""Transit timing measurement and TTV analysis for the public API.

This module provides wrappers around internal transit timing primitives:
- measure_transit_times: Measure mid-times for all transits in a light curve
- analyze_ttvs: Compute O-C residuals and TTV statistics

Transit Timing Variations (TTVs) arise from gravitational perturbations by
additional bodies in the system, primarily near mean-motion resonances.

Novelty: standard

References:
    [1] Holman & Murray 2005 (2005Sci...307.1288H) - TTV detection concept
    [2] Agol et al. 2005 (2005MNRAS.359..567A) - TTV sensitivity analysis
    [3] Lithwick et al. 2012 (2012ApJ...761..122L) - Analytic TTV formulae
    [4] Ivshina & Winn 2022 (2022ApJS..259...62I) - TESS transit timing methods
"""

from __future__ import annotations

from bittr_tess_vetter.api.references import (
    AGOL_2005,
    FABRYCKY_2012,
    FORD_2012,
    HADDEN_2019,
    HADDEN_LITHWICK_2016,
    HOLMAN_MURRAY_2005,
    IVSHINA_WINN_2022,
    LITHWICK_2012,
    RAGOZZINE_HOLMAN_2019,
    STEFFEN_AGOL_2006,
    cite,
    cites,
)
from bittr_tess_vetter.api.types import Candidate, LightCurve
from bittr_tess_vetter.transit.result import (
    TransitTime,
    TransitTimingSeries,
    TTVResult,
)
from bittr_tess_vetter.transit.timing import (
    build_timing_series,
    compute_ttv_statistics,
    measure_all_transit_times,
)

# Re-export types for convenience
__all__ = [
    "REFERENCES",
    "TransitTime",
    "TransitTimingSeries",
    "TTVResult",
    "analyze_ttvs",
    "measure_transit_times",
    "timing_series",
]

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [
    ref.to_dict()
    for ref in [
        HOLMAN_MURRAY_2005,
        AGOL_2005,
        LITHWICK_2012,
        HADDEN_LITHWICK_2016,
        HADDEN_2019,
        IVSHINA_WINN_2022,
        STEFFEN_AGOL_2006,
        FORD_2012,
        FABRYCKY_2012,
        RAGOZZINE_HOLMAN_2019,
    ]
]


@cites(
    cite(IVSHINA_WINN_2022, "§3 transit time template fitting"),
    cite(FORD_2012, "Kepler TTV detection methodology"),
    cite(HOLMAN_MURRAY_2005, "foundational TTV theory"),
    cite(AGOL_2005, "TTV detection methods"),
)
def measure_transit_times(
    lc: LightCurve,
    candidate: Candidate,
    *,
    min_snr: float = 2.0,
) -> list[TransitTime]:
    """Measure mid-transit times for all observable transits.

    Fits a trapezoid model to each transit window independently to measure
    the actual mid-transit time. Filters results by SNR threshold and
    convergence.

    Args:
        lc: Light curve data
        candidate: Transit candidate with ephemeris
        min_snr: Minimum SNR to include a transit (default 2.0)

    Returns:
        List of TransitTime measurements for successfully measured transits

    Example:
        >>> from bittr_tess_vetter.api import LightCurve, Candidate, Ephemeris
        >>> from bittr_tess_vetter.api.timing import measure_transit_times
        >>> lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        >>> eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
        >>> candidate = Candidate(ephemeris=eph)
        >>> times = measure_transit_times(lc, candidate, min_snr=3.0)
        >>> for t in times:
        ...     print(f"Epoch {t.epoch}: tc={t.tc:.6f} +/- {t.tc_err:.6f} days")

    Novelty: standard

    References:
        [1] Ivshina & Winn 2022 (2022ApJS..259...62I)
            Section 3: Transit time measurement via template fitting
        [2] Ford et al. 2012 (2012ApJ...750..113F)
            Kepler TTV detection methodology
        [3] Holman & Murray 2005 (2005Sci...307.1288H)
            TTV theory - foundational paper
        [4] Agol et al. 2005 (2005MNRAS.359..567A)
            TTV detection methods
    """
    # Convert to internal representation
    internal_lc = lc.to_internal()
    mask = internal_lc.valid_mask
    time = internal_lc.time[mask]
    flux = internal_lc.flux[mask]
    flux_err = internal_lc.flux_err[mask]

    # Extract ephemeris from candidate
    eph = candidate.ephemeris

    # Call internal implementation
    return measure_all_transit_times(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=eph.period_days,
        t0=eph.t0_btjd,
        duration_hours=eph.duration_hours,
        min_snr=min_snr,
    )


@cites(
    cite(LITHWICK_2012, "§2 analytic TTV formulae Eq.1-10"),
    cite(HADDEN_LITHWICK_2016, "§2-3 practical TTV analysis"),
    cite(HADDEN_2019, "§2 TESS TTV detection thresholds"),
)
def analyze_ttvs(
    transit_times: list[TransitTime],
    period_days: float,
    t0_btjd: float,
) -> TTVResult:
    """Compute O-C residuals and TTV statistics.

    Calculates observed minus calculated (O-C) residuals from measured
    transit times and computes summary statistics including RMS,
    periodicity significance, and linear trends. Also flags outliers
    based on O-C and duration criteria.

    Args:
        transit_times: Measured transit times from measure_transit_times()
        period_days: Orbital period in days
        t0_btjd: Reference transit epoch in BTJD

    Returns:
        TTVResult containing O-C residuals, TTV statistics, and outlier flags

    Example:
        >>> from bittr_tess_vetter.api.timing import analyze_ttvs
        >>> # After obtaining transit_times from measure_transit_times()
        >>> ttv_result = analyze_ttvs(transit_times, period_days=3.5, t0_btjd=1850.0)
        >>> print(f"RMS: {ttv_result.rms_seconds:.1f} seconds")
        >>> print(f"Periodicity: {ttv_result.periodicity_sigma:.1f} sigma")
        >>> if ttv_result.linear_trend is not None:
        ...     print(f"Linear trend: {ttv_result.linear_trend:.2f} sec/epoch")

    Novelty: standard

    References:
        [1] Lithwick et al. 2012 (2012ApJ...761..122L)
            Section 2: Analytic TTV formulae, Equations 1-10
        [2] Hadden & Lithwick 2017 (2017AJ....154....5H)
            Section 2-3: Practical TTV analysis methods
        [3] Hadden et al. 2019 (2019AJ....158..146H)
            Section 2: TESS TTV detection thresholds
    """
    # Use median duration from transit times for outlier detection
    expected_duration = None
    if transit_times:
        durations = [t.duration_hours for t in transit_times]
        expected_duration = float(sum(durations) / len(durations))

    return compute_ttv_statistics(
        transit_times=transit_times,
        period=period_days,
        t0=t0_btjd,
        expected_duration_hours=expected_duration,
    )


@cites(
    cite(IVSHINA_WINN_2022, "§3 transit time template fitting"),
    cite(FORD_2012, "Kepler TTV detection methodology"),
)
def timing_series(
    lc: LightCurve,
    candidate: Candidate,
    *,
    min_snr: float = 2.0,
) -> TransitTimingSeries:
    """Measure and summarize per-transit O-C + significance series."""
    transit_times = measure_transit_times(lc, candidate, min_snr=min_snr)
    eph = candidate.ephemeris
    return build_timing_series(
        transit_times=transit_times,
        period=eph.period_days,
        t0=eph.t0_btjd,
    )
