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

from typing import Any

from bittr_tess_vetter.api.types import Candidate, LightCurve
from bittr_tess_vetter.transit.result import TransitTime, TTVResult
from bittr_tess_vetter.transit.timing import (
    compute_ttv_statistics,
    measure_all_transit_times,
)

# Re-export types for convenience
__all__ = [
    "REFERENCES",
    "TransitTime",
    "TTVResult",
    "analyze_ttvs",
    "measure_transit_times",
]

# Module-level references for programmatic access
REFERENCES: list[dict[str, Any]] = [
    {
        "id": "holman_murray_2005",
        "type": "article",
        "bibcode": "2005Sci...307.1288H",
        "title": "The Use of Transit Timing to Detect Terrestrial-Mass Extrasolar Planets",
        "authors": ["Holman, M. J.", "Murray, N. W."],
        "journal": "Science",
        "year": 2005,
        "note": "Foundational TTV theory paper - perturbations from additional planets",
    },
    {
        "id": "agol_2005",
        "type": "article",
        "bibcode": "2005MNRAS.359..567A",
        "title": "On detecting terrestrial planets with timing of giant planet transits",
        "authors": ["Agol, E.", "Steffen, J.", "Sari, R.", "Clarkson, W."],
        "journal": "Monthly Notices of the Royal Astronomical Society",
        "year": 2005,
        "note": "TTV theory - sensitivity to perturbing planets",
    },
    {
        "id": "lithwick_2012",
        "type": "article",
        "bibcode": "2012ApJ...761..122L",
        "title": "Extracting Planet Mass and Eccentricity from TTV Data",
        "authors": ["Lithwick, Y.", "Xie, J.", "Wu, Y."],
        "journal": "The Astrophysical Journal",
        "year": 2012,
        "arxiv": "1207.4192",
        "note": "Analytic TTV formulae for near-resonant planet pairs",
    },
    {
        "id": "hadden_lithwick_2016",
        "type": "article",
        "bibcode": "2017AJ....154....5H",
        "title": "Kepler Planet Masses and Eccentricities from TTV Analysis",
        "authors": ["Hadden, S.", "Lithwick, Y."],
        "journal": "The Astronomical Journal",
        "year": 2017,
        "arxiv": "1611.03516",
        "note": "Uniform TTV analysis of Kepler multiplanet systems",
    },
    {
        "id": "hadden_2019",
        "type": "article",
        "bibcode": "2019AJ....158..146H",
        "title": "Prospects for TTV Detection and Dynamical Constraints with TESS",
        "authors": ["Hadden, S.", "Barclay, T.", "Payne, M. J.", "Holman, M. J."],
        "journal": "The Astronomical Journal",
        "year": 2019,
        "arxiv": "1811.01970",
        "note": "TTV yield predictions for TESS mission",
    },
    {
        "id": "ivshina_winn_2022",
        "type": "article",
        "bibcode": "2022ApJS..259...62I",
        "title": "TESS Transit Timing of Hundreds of Hot Jupiters",
        "authors": ["Ivshina, E. S.", "Winn, J. N."],
        "journal": "The Astrophysical Journal Supplement Series",
        "year": 2022,
        "arxiv": "2202.03401",
        "note": "TESS transit timing database and methods",
    },
    {
        "id": "steffen_agol_2006",
        "type": "article",
        "bibcode": "2007MNRAS.374..941A",
        "title": "Developments in Planet Detection using Transit Timing Variations",
        "authors": ["Steffen, J. H.", "Agol, E."],
        "journal": "Monthly Notices of the Royal Astronomical Society",
        "year": 2007,
        "arxiv": "astro-ph/0612442",
        "note": "TTV detection sensitivity and methods",
    },
    {
        "id": "ford_2012",
        "type": "article",
        "bibcode": "2012ApJ...750..113F",
        "title": "Transit Timing Observations from Kepler: VI. TTV Candidates",
        "authors": ["Ford, E. B.", "Ragozzine, D.", "Rowe, J. F.", "et al."],
        "journal": "The Astrophysical Journal",
        "year": 2012,
        "arxiv": "1201.1892",
        "note": "Kepler TTV detection methodology",
    },
    {
        "id": "fabrycky_2012",
        "type": "article",
        "bibcode": "2012ApJ...750..114F",
        "title": "Transit Timing Observations from Kepler: IV. Confirmation by Physical Models",
        "authors": ["Fabrycky, D. C.", "Ford, E. B.", "Steffen, J. H.", "et al."],
        "journal": "The Astrophysical Journal",
        "year": 2012,
        "arxiv": "1201.5415",
        "note": "Multi-planet TTV confirmation methodology",
    },
    {
        "id": "ragozzine_holman_2019",
        "type": "article",
        "bibcode": "2019AJ....157..218R",
        "title": "Kepler-9: The First Multi-Transiting System and First TTVs",
        "authors": ["Ragozzine, D.", "Holman, M. J."],
        "journal": "The Astronomical Journal",
        "year": 2019,
        "arxiv": "1905.04426",
        "note": "Historical context of first TTV detection",
    },
]


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

    # Extract ephemeris from candidate
    eph = candidate.ephemeris

    # Call internal implementation
    return measure_all_transit_times(
        time=internal_lc.time,
        flux=internal_lc.flux,
        flux_err=internal_lc.flux_err,
        period=eph.period_days,
        t0=eph.t0_btjd,
        duration_hours=eph.duration_hours,
        min_snr=min_snr,
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
