"""Stellar activity characterization for the public API.

This module provides activity analysis functions for TESS light curves:
- characterize_activity: Full stellar activity characterization (rotation, flares, classification)
- mask_flares: Remove flare events from light curves

All functions accept LightCurve from api.types and return public result types.

Novelty: standard (implements established techniques from literature)

References:
    [1] McQuillan et al. 2014, ApJS 211, 24 (2014ApJS..211...24M) - Rotation periods
    [2] Davenport 2016, ApJ 829, 23 (2016ApJ...829...23D) - Flare detection
    [3] Basri et al. 2013, ApJ 769, 37 (2013ApJ...769...37B) - Variability indices
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bittr_tess_vetter.activity.primitives import (
    classify_variability,
    compute_activity_index,
    compute_phase_amplitude,
    generate_recommendation,
    measure_rotation_period,
)
from bittr_tess_vetter.activity.primitives import (
    detect_flares as _detect_flares,
)
from bittr_tess_vetter.activity.primitives import (
    mask_flares as _mask_flares,
)
from bittr_tess_vetter.activity.result import ActivityResult, Flare
from bittr_tess_vetter.api.types import LightCurve

# Re-export result types for public API
__all__ = [
    "ActivityResult",
    "Flare",
    "REFERENCES",
    "characterize_activity",
    "mask_flares",
]

# Module-level references for programmatic access
REFERENCES: list[dict[str, Any]] = [
    {
        "id": "mcquillan_2014",
        "type": "article",
        "bibcode": "2014ApJS..211...24M",
        "title": "Rotation Periods of 34,030 Kepler Main-Sequence Stars: "
        "The Full Autocorrelation Sample",
        "authors": ["McQuillan, A.", "Mazeh, T.", "Aigrain, S."],
        "journal": "The Astrophysical Journal Supplement Series",
        "year": 2014,
        "arxiv": "1402.5694",
        "note": "Autocorrelation-based rotation period measurement methodology",
    },
    {
        "id": "mcquillan_2013",
        "type": "article",
        "bibcode": "2013MNRAS.432.1203M",
        "title": "Measuring the rotation period distribution of field M dwarfs with Kepler",
        "authors": ["McQuillan, A.", "Aigrain, S.", "Mazeh, T."],
        "journal": "Monthly Notices of the Royal Astronomical Society",
        "year": 2013,
        "arxiv": "1303.6787",
        "note": "ACF method development for rotation periods",
    },
    {
        "id": "davenport_2016",
        "type": "article",
        "bibcode": "2016ApJ...829...23D",
        "title": "The Kepler Catalog of Stellar Flares",
        "authors": ["Davenport, J. R. A."],
        "journal": "The Astrophysical Journal",
        "year": 2016,
        "arxiv": "1607.03494",
        "note": "Comprehensive flare detection methodology for Kepler",
    },
    {
        "id": "davenport_2014",
        "type": "article",
        "bibcode": "2014ApJ...797..122D",
        "title": "The Shape of M Dwarf Flares in Kepler Light Curves",
        "authors": ["Davenport, J. R. A."],
        "journal": "The Astrophysical Journal",
        "year": 2014,
        "arxiv": "1510.05695",
        "note": "Empirical flare template and morphology",
    },
    {
        "id": "basri_2013",
        "type": "article",
        "bibcode": "2013ApJ...769...37B",
        "title": "Photometric Variability in Kepler Target Stars III: Comparison with the Sun",
        "authors": ["Basri, G.", "Walkowicz, L.", "Reiners, A."],
        "journal": "The Astrophysical Journal",
        "year": 2013,
        "arxiv": "1304.0136",
        "note": "Stellar variability metrics and solar comparison",
    },
    {
        "id": "nielsen_2013",
        "type": "article",
        "bibcode": "2013A&A...557L..10N",
        "title": "Rotation periods of 12,000 main-sequence Kepler stars",
        "authors": ["Nielsen, M. B.", "Gizon, L.", "Schunker, H.", "Karoff, C."],
        "journal": "Astronomy & Astrophysics",
        "year": 2013,
        "arxiv": "1305.5721",
        "note": "Alternative rotation period measurement approach",
    },
    {
        "id": "reinhold_2020",
        "type": "article",
        "bibcode": "2020Sci...368..518R",
        "title": "The Sun is less active than other solar-like stars",
        "authors": ["Reinhold, T.", "Shapiro, A. I.", "Solanki, S. K.", "et al."],
        "journal": "Science",
        "year": 2020,
        "note": "Solar activity in context of stellar variability",
    },
    {
        "id": "davenport_2019",
        "type": "article",
        "bibcode": "2019ApJ...871..241D",
        "title": "The Evolution of Flare Activity with Stellar Age",
        "authors": ["Davenport, J. R. A.", "Covey, K. R.", "Clarke, R. W.", "et al."],
        "journal": "The Astrophysical Journal",
        "year": 2019,
        "arxiv": "1901.00890",
        "note": "Flare activity vs Rossby number and stellar age",
    },
    {
        "id": "tovar_mendoza_2022",
        "type": "article",
        "bibcode": "2022ApJ...927...31T",
        "title": "Llamaradas Estelares: Modeling the Morphology of White-Light Flares",
        "authors": ["Tovar Mendoza, G.", "Davenport, J. R. A.", "Agol, E.", "et al."],
        "journal": "The Astrophysical Journal",
        "year": 2022,
        "arxiv": "2205.05706",
        "note": "Improved analytic flare model",
    },
]


def characterize_activity(
    lc: LightCurve,
    *,
    detect_flares: bool = True,
    flare_sigma: float = 5.0,
    rotation_min_period: float = 0.5,
    rotation_max_period: float = 30.0,
) -> ActivityResult:
    """Characterize stellar activity from light curve.

    Measures rotation period, detects flares, classifies variability type,
    and generates recommendations for transit recovery.

    Args:
        lc: Light curve data
        detect_flares: Whether to run flare detection
        flare_sigma: Sigma threshold for flare detection
        rotation_min_period: Minimum rotation period to search (days)
        rotation_max_period: Maximum rotation period to search (days)

    Returns:
        ActivityResult with rotation, flares, classification, recommendations

    Novelty: standard

    References:
        [1] McQuillan et al. 2014 (2014ApJS..211...24M) - Rotation periods
        [2] Davenport 2016 (2016ApJ...829...23D) - Flare detection
        [3] Basri et al. 2013 (2013ApJ...769...37B) - Activity indices
    """
    # Convert to internal arrays (float64)
    internal_lc = lc.to_internal()
    time = internal_lc.time
    flux = internal_lc.flux
    flux_err = internal_lc.flux_err

    # Measure rotation period
    rotation_period, rotation_err, rotation_snr = measure_rotation_period(
        time=time,
        flux=flux,
        min_period=rotation_min_period,
        max_period=rotation_max_period,
    )

    # Detect flares
    flares: list[Flare] = []
    if detect_flares:
        flares = _detect_flares(
            time=time,
            flux=flux,
            flux_err=flux_err,
            sigma_threshold=flare_sigma,
        )

    # Compute observation baseline
    baseline_days = float(time[-1] - time[0]) if len(time) > 1 else 1.0

    # Compute flare rate
    flare_rate = len(flares) / baseline_days if baseline_days > 0 else 0.0

    # Compute phase amplitude (peak-to-peak variability at rotation period)
    phase_amplitude = compute_phase_amplitude(
        time=time,
        flux=flux,
        period=rotation_period,
    )

    # Compute RMS variability in ppm
    variability_ppm = float(np.std(flux) * 1e6)

    # Classify variability type
    variability_class = classify_variability(
        periodogram_power=rotation_snr,
        phase_amplitude=phase_amplitude,
        flare_count=len(flares),
        baseline_days=baseline_days,
    )

    # Compute activity index (0-1 scale)
    activity_index = compute_activity_index(
        variability_ppm=variability_ppm,
        rotation_period=rotation_period,
        flare_rate=flare_rate,
    )

    # Generate recommendation and suggested parameters
    recommendation, suggested_params = generate_recommendation(
        variability_class=variability_class,
        variability_ppm=variability_ppm,
        rotation_period=rotation_period,
        flare_rate=flare_rate,
        activity_index=activity_index,
    )

    return ActivityResult(
        rotation_period=rotation_period,
        rotation_err=rotation_err,
        rotation_snr=rotation_snr,
        variability_ppm=variability_ppm,
        variability_class=variability_class,
        flares=flares,
        flare_rate=flare_rate,
        activity_index=activity_index,
        recommendation=recommendation,
        suggested_params=suggested_params,
    )


def mask_flares(
    lc: LightCurve,
    flares: list[Flare],
    *,
    buffer_minutes: float = 5.0,
) -> LightCurve:
    """Replace flare regions with interpolated baseline.

    Args:
        lc: Light curve data
        flares: Detected flares from characterize_activity
        buffer_minutes: Buffer zone around each flare

    Returns:
        LightCurve with flares masked/interpolated

    Novelty: standard
    """
    # Convert to internal arrays
    internal_lc = lc.to_internal()
    time = internal_lc.time
    flux = internal_lc.flux

    # Apply flare masking
    masked_flux = _mask_flares(
        time=time,
        flux=flux,
        flares=flares,
        buffer_minutes=buffer_minutes,
    )

    # Return new LightCurve with masked flux
    return LightCurve(
        time=np.array(time),  # Copy to ensure writable
        flux=masked_flux,
        flux_err=np.array(internal_lc.flux_err) if lc.flux_err is not None else None,
        quality=np.array(internal_lc.quality) if lc.quality is not None else None,
        valid_mask=np.array(internal_lc.valid_mask) if lc.valid_mask is not None else None,
    )
