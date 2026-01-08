"""Transit vetting primitives for the public API.

This module provides thin wrappers around internal transit vetting functions,
converting between facade types and internal types.

Novelty: adapted (relative threshold variant of standard odd/even test)

References:
    [1] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
        Section 4.2: Original odd/even depth test in Kepler Robovetter
    [2] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
        Section 3.3.1: DR25 odd/even transit depth comparison
    [3] Prsa et al. 2011, AJ 141, 83 (2011AJ....141...83P)
        Section 3: Eclipsing binary depth ratio characteristics
"""

from __future__ import annotations

from bittr_tess_vetter.api.types import Ephemeris, LightCurve
from bittr_tess_vetter.transit.result import OddEvenResult
from bittr_tess_vetter.transit.vetting import (
    compute_odd_even_result as _compute_odd_even_result,
)

REFERENCES: list[dict[str, str | int | list[str]]] = [
    {
        "id": "coughlin_2016",
        "type": "article",
        "bibcode": "2016ApJS..224...12C",
        "title": (
            "Planetary Candidates Observed by Kepler. VII. The First Fully Uniform "
            "Catalog Based on the Entire 48-month Data Set (Q1-Q17 DR24)"
        ),
        "authors": ["Coughlin, J.L.", "Mullally, F.", "Thompson, S.E."],
        "journal": "The Astrophysical Journal Supplement Series",
        "year": 2016,
        "note": "Section 4.2: Original odd/even depth test in Kepler Robovetter",
    },
    {
        "id": "thompson_2018",
        "type": "article",
        "bibcode": "2018ApJS..235...38T",
        "title": (
            "Planetary Candidates Observed by Kepler. VIII. A Fully Automated "
            "Catalog with Measured Completeness and Reliability Based on Data "
            "Release 25"
        ),
        "authors": ["Thompson, S.E.", "Coughlin, J.L.", "Hoffman, K."],
        "journal": "The Astrophysical Journal Supplement Series",
        "year": 2018,
        "note": "Section 3.3.1: DR25 odd/even transit depth comparison",
    },
    {
        "id": "prsa_2011",
        "type": "article",
        "bibcode": "2011AJ....141...83P",
        "title": "Kepler Eclipsing Binary Stars. I. Catalog and Principal Characterization of 1879 Eclipsing Binaries in the First Data Release",
        "authors": ["Prsa, A.", "Batalha, N.", "Slawson, R.W."],
        "journal": "The Astronomical Journal",
        "year": 2011,
        "note": "Section 3: Eclipsing binary depth ratio characteristics (50-100% different)",
    },
]


def odd_even_result(
    lc: LightCurve,
    ephemeris: Ephemeris,
    *,
    relative_threshold_percent: float = 10.0,
) -> OddEvenResult:
    """Compute odd/even depth comparison for eclipsing binary detection.

    Splits transits by epoch parity (odd vs even), measures depths, and computes
    the significance of any difference. Uses relative depth difference to
    determine if the signal is suspicious (potential eclipsing binary).

    Real eclipsing binaries show 50-100% relative depth differences between
    primary and secondary eclipses. Confirmed planets typically show <5%
    relative difference. We flag as suspicious only when the relative
    depth difference exceeds the threshold (default 10%).

    Args:
        lc: Light curve data
        ephemeris: Transit ephemeris (period, t0, duration)
        relative_threshold_percent: Relative depth difference threshold in percent
            for flagging as suspicious (default 10.0%). Real EBs show 50-100%,
            planets show <5%.

    Returns:
        OddEvenResult with depth comparison and significance

    Novelty: adapted

    Adaptation: Uses 10% relative depth difference threshold instead of the
        3-sigma absolute threshold from the original Kepler Robovetter method.
        This reduces false positives on shallow transits where photometric noise
        dominates the absolute depth uncertainty. The relative threshold is
        motivated by the empirical observation that real EBs show 50-100%
        depth differences (Prsa et al. 2011), while planets show <5%.

    References:
        [1] Coughlin et al. 2016, ApJS 224, 12 (2016ApJS..224...12C)
            Section 4.2: Original odd/even depth test in Kepler Robovetter
        [2] Thompson et al. 2018, ApJS 235, 38 (2018ApJS..235...38T)
            Section 3.3.1: DR25 odd/even transit depth comparison
        [3] Prsa et al. 2011, AJ 141, 83 (2011AJ....141...83P)
            Section 3: EB depth ratios typically 50-100% different
    """
    # Convert facade types to internal types
    internal_lc = lc.to_internal()

    # Extract arrays for internal function
    # Use valid_mask to filter data
    time = internal_lc.time[internal_lc.valid_mask]
    flux = internal_lc.flux[internal_lc.valid_mask]
    flux_err = internal_lc.flux_err[internal_lc.valid_mask]

    # Call internal implementation
    return _compute_odd_even_result(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=ephemeris.period_days,
        t0=ephemeris.t0_btjd,
        duration_hours=ephemeris.duration_hours,
        relative_threshold_percent=relative_threshold_percent,
    )
