"""Pixel-level vetting checks for the public API.

This module provides thin wrappers around the pixel-level vetting checks (V08-V10),
converting between facade types and internal types.

Check Summary:
- V08 centroid_shift: Compare in-transit vs out-of-transit centroid positions
- V09 difference_image_localization: Analyze pixel-level light curves to locate transit source
- V10 aperture_dependence: Measure transit depth vs aperture size for contamination detection

Novelty: standard (all checks implement well-established techniques from literature)

References:
    [1] Bryson et al. 2013, PASP 125, 889 (2013PASP..125..889B) - Kepler pixel-level diagnostics
    [2] Twicken et al. 2018, PASP 130, 064502 (2018PASP..130f4502T) - Kepler Data Validation
    [3] Torres et al. 2011, ApJ 727, 24 (2011ApJ...727...24T) - Background blend detection
    [4] Guerrero et al. 2021, ApJS 254, 39 (2021ApJS..254...39G) - TESS TOI catalog vetting
    [5] Mullally et al. 2015, ApJS 217, 31 (2015ApJS..217...31M) - Kepler planet candidate vetting
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bittr_tess_vetter.api.types import Candidate, CheckResult, TPFStamp
from bittr_tess_vetter.validation.checks_pixel import (
    ApertureDependenceCheck,
    CentroidShiftCheck,
    PixelLevelLCCheck,
)

if TYPE_CHECKING:
    pass

# Module-level references for programmatic access
REFERENCES: list[dict[str, str | int | list[str]]] = [
    {
        "id": "bryson_2013",
        "type": "article",
        "bibcode": "2013PASP..125..889B",
        "title": "Identification of Background False Positives from Kepler Data",
        "authors": ["Bryson, S.T.", "Jenkins, J.M.", "Gilliland, R.L."],
        "journal": "PASP 125, 889",
        "year": 2013,
        "note": "Pixel-level diagnostics for identifying background false positives",
    },
    {
        "id": "twicken_2018",
        "type": "article",
        "bibcode": "2018PASP..130f4502T",
        "title": "Kepler Data Validation I -- Architecture, Diagnostic Tests, and "
        "Data Products for Vetting Transiting Planet Candidates",
        "authors": ["Twicken, J.D.", "Catanzarite, J.H.", "Clarke, B.D."],
        "journal": "PASP 130, 064502",
        "year": 2018,
        "note": "Kepler DV pipeline: centroid offset, difference imaging, ghost diagnostics",
    },
    {
        "id": "batalha_2010",
        "type": "article",
        "bibcode": "2010ApJ...713L.109B",
        "title": "Selection, Prioritization, and Characteristics of Kepler Target Stars",
        "authors": ["Batalha, N.M.", "Borucki, W.J.", "Koch, D.G."],
        "journal": "ApJ 713, L109",
        "year": 2010,
        "note": "Kepler target star selection and stellar classification methodology",
    },
    {
        "id": "torres_2011",
        "type": "article",
        "bibcode": "2011ApJ...727...24T",
        "title": "Modeling Kepler Transit Light Curves as False Positives: "
        "Rejection of Blend Scenarios for Kepler-9, and Validation of Kepler-9 d, "
        "a Super-Earth-size Planet in a Multiple System",
        "authors": ["Torres, G.", "Fressin, F.", "Batalha, N.M."],
        "journal": "ApJ 727, 24",
        "year": 2011,
        "note": "Background blend detection and rejection methodology",
    },
    {
        "id": "guerrero_2021",
        "type": "article",
        "bibcode": "2021ApJS..254...39G",
        "title": "The TESS Objects of Interest Catalog from the TESS Prime Mission",
        "authors": ["Guerrero, N.M.", "Seager, S.", "Huang, C.X."],
        "journal": "ApJS 254, 39",
        "year": 2021,
        "note": "TESS TOI catalog pixel-level vetting procedures",
    },
    {
        "id": "mullally_2015",
        "type": "article",
        "bibcode": "2015ApJS..217...31M",
        "title": "Planetary Candidates Observed by Kepler VI: Planet Sample from Q1-Q16 (47 Months)",
        "authors": ["Mullally, F.", "Coughlin, J.L.", "Thompson, S.E."],
        "journal": "ApJS 217, 31",
        "year": 2015,
        "note": "Kepler planet candidate catalog with vetting diagnostics and false alarm identification",
    },
]


def _convert_result(result: object) -> CheckResult:
    """Convert internal VetterCheckResult to facade CheckResult.

    Args:
        result: Internal VetterCheckResult (pydantic model)

    Returns:
        Facade CheckResult dataclass
    """
    # VetterCheckResult is a pydantic model with these attributes
    return CheckResult(
        id=result.id,  # type: ignore[attr-defined]
        name=result.name,  # type: ignore[attr-defined]
        passed=result.passed,  # type: ignore[attr-defined]
        confidence=result.confidence,  # type: ignore[attr-defined]
        details=dict(result.details),  # type: ignore[attr-defined]
    )


def _extract_arrays_from_tpf(
    tpf: TPFStamp,
) -> tuple[Any, Any]:
    """Extract time and flux arrays from TPFStamp for internal checks.

    Args:
        tpf: TPFStamp from API

    Returns:
        Tuple of (time_array, flux_array) suitable for internal checks
    """
    import numpy as np

    # Ensure float64 dtype for consistency with internal checks
    time_arr = np.asarray(tpf.time, dtype=np.float64)
    flux_arr = np.asarray(tpf.flux, dtype=np.float64)

    return time_arr, flux_arr


def centroid_shift(
    tpf: TPFStamp,
    candidate: Candidate,
    *,
    config: dict[str, Any] | None = None,
) -> CheckResult:
    """V08: Detect centroid motion during transit.

    Compares the flux-weighted centroid position during transit versus
    out-of-transit. A significant shift indicates the transit source is
    not the target star, but a nearby or background eclipsing binary.

    TESS pixel scale: 21 arcsec/pixel

    Thresholds (configurable):
    - FAIL: shift >= 1.0 pixel AND significance >= 5.0 sigma
    - WARN: shift >= 0.5 pixel OR significance >= 3.0 sigma
    - PASS: otherwise

    Args:
        tpf: Target Pixel File data (TPFStamp)
        candidate: Transit candidate with ephemeris
        config: Optional configuration overrides:
            - fail_shift_threshold: pixels (default 1.0)
            - fail_sigma_threshold: sigma (default 5.0)
            - warn_shift_threshold: pixels (default 0.5)
            - warn_sigma_threshold: sigma (default 3.0)

    Returns:
        CheckResult with centroid shift analysis including:
        - centroid_shift_pixels: measured shift in pixels
        - significance_sigma: statistical significance
        - shift_arcsec: shift converted to arcseconds
        - in_transit_centroid: (x, y) centroid during transit
        - out_of_transit_centroid: (x, y) centroid out of transit

    Novelty: standard

    References:
        [1] Bryson et al. 2013, PASP 125, 889 (2013PASP..125..889B)
            Section 3.1: Centroid offset test for background false positives
        [2] Twicken et al. 2018, PASP 130, 064502 (2018PASP..130f4502T)
            Section 4.1: Difference image centroid offsets in Kepler DV
    """
    time_arr, flux_arr = _extract_arrays_from_tpf(tpf)

    # Build check config from user-provided overrides
    check_config = None
    if config:
        from bittr_tess_vetter.validation.base import CheckConfig

        check_config = CheckConfig(
            enabled=True,
            threshold=config.get("fail_shift_threshold", 1.0),
            additional={
                "fail_sigma_threshold": config.get("fail_sigma_threshold", 5.0),
                "warn_shift_threshold": config.get("warn_shift_threshold", 0.5),
                "warn_sigma_threshold": config.get("warn_sigma_threshold", 3.0),
            },
        )

    # Create internal check instance
    check = CentroidShiftCheck(
        config=check_config,
        tpf_data=flux_arr,
        time=time_arr,
    )

    # Create a minimal TransitCandidate for the internal check
    from bittr_tess_vetter.domain.detection import TransitCandidate

    internal_candidate = TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=candidate.depth or 0.001,
    )

    result = check.run(internal_candidate)
    return _convert_result(result)


def difference_image_localization(
    tpf: TPFStamp,
    candidate: Candidate,
    *,
    target_rc: tuple[float, float] | None = None,
    wcs_sources: list[dict[str, Any]] | None = None,  # Reserved for WCS-aware localization
    config: dict[str, Any] | None = None,
) -> CheckResult:
    """V09: Locate transit source via pixel-level light curve analysis.

    Extracts light curves from individual pixels, measures transit depth
    in each, and determines if the signal is on-target. This is a proxy
    for difference image localization when full WCS is not available.

    Pass Criteria:
    - concentration_ratio >= 0.7 (target depth / max depth)
    - transit_on_target == True (max depth within proximity_radius of target)

    Fail Criteria:
    - transit_on_target == False (signal off-target)
    - concentration_ratio < 0.5 (signal not concentrated on target)

    Args:
        tpf: Target Pixel File data (TPFStamp)
        candidate: Transit candidate with ephemeris
        target_rc: Expected target pixel (row, col). If None, uses TPF center.
        wcs_sources: List of reference sources with WCS coords (reserved for future use)
        config: Optional configuration overrides:
            - concentration_threshold: minimum ratio for pass (default 0.7)
            - proximity_radius: max pixel distance for on-target (default 1)

    Returns:
        CheckResult with pixel-level localization including:
        - max_depth_pixel: (row, col) of pixel with maximum depth
        - max_depth_ppm: maximum transit depth in ppm
        - target_depth_ppm: transit depth at target pixel
        - concentration_ratio: target_depth / max_depth
        - transit_on_target: whether max depth is near target

    Novelty: standard

    References:
        [1] Bryson et al. 2013, PASP 125, 889 (2013PASP..125..889B)
            Section 3.2: Difference image analysis for source localization
        [2] Torres et al. 2011, ApJ 727, 24 (2011ApJ...727...24T)
            Section 4: Blend scenarios and centroid analysis
        [3] Twicken et al. 2018, PASP 130, 064502 (2018PASP..130f4502T)
            Section 4.2: Difference image pixel response function fitting
    """
    # wcs_sources reserved for future WCS-aware localization
    del wcs_sources

    time_arr, flux_arr = _extract_arrays_from_tpf(tpf)

    # Build check config from user-provided overrides
    check_config = None
    if config:
        from bittr_tess_vetter.validation.base import CheckConfig

        check_config = CheckConfig(
            enabled=True,
            threshold=config.get("concentration_threshold", 0.7),
            additional={
                "proximity_radius": config.get("proximity_radius", 1),
            },
        )

    # Convert target_rc to (row, col) tuple if provided
    target_pixel: tuple[int, int] | None = None
    if target_rc is not None:
        target_pixel = (int(target_rc[0]), int(target_rc[1]))

    # Create internal check instance
    check = PixelLevelLCCheck(
        config=check_config,
        tpf_data=flux_arr,
        time=time_arr,
        target_pixel=target_pixel,
    )

    # Create a minimal TransitCandidate for the internal check
    from bittr_tess_vetter.domain.detection import TransitCandidate

    internal_candidate = TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=candidate.depth or 0.001,
    )

    result = check.run(internal_candidate)
    return _convert_result(result)


def aperture_dependence(
    tpf: TPFStamp,
    candidate: Candidate,
    *,
    radii_px: list[float] | None = None,
    config: dict[str, Any] | None = None,
) -> CheckResult:
    """V10: Check if transit depth varies with aperture size.

    Measures transit depth at multiple aperture radii centered on the target.
    Significant variation indicates contamination:
    - Stable depth = on-target signal (PASS)
    - Depth increases with aperture = contamination dilution (FAIL)
    - Depth decreases with aperture = background source (FAIL)

    Thresholds (stability_metric 0-1 scale):
    - FAIL: stability_metric < 0.5
    - WARN: stability_metric < 0.7
    - PASS: stability_metric >= 0.7

    Args:
        tpf: Target Pixel File data (TPFStamp)
        candidate: Transit candidate with ephemeris and depth
        radii_px: List of aperture radii to test (pixels).
            Default: [1.5, 2.0, 2.5, 3.0, 3.5] per v2 spec.
        config: Optional configuration overrides:
            - fail_stability_threshold: stability for failure (default 0.5)
            - warn_stability_threshold: stability for warning (default 0.7)

    Returns:
        CheckResult with aperture dependence analysis including:
        - stability_metric: 0-1 metric of depth consistency
        - depths_by_aperture_ppm: {radius: depth} mapping
        - depth_variance_ppm2: variance of depths across apertures
        - recommended_aperture_pixels: optimal aperture from analysis
        - relative_variation: depth range / mean depth

    Novelty: standard

    References:
        [1] Bryson et al. 2013, PASP 125, 889 (2013PASP..125..889B)
            Section 3.3: Contamination assessment via aperture photometry
        [2] Guerrero et al. 2021, ApJS 254, 39 (2021ApJS..254...39G)
            Section 3.4: TESS aperture family analysis
        [3] Mullally et al. 2015, ApJS 217, 31 (2015ApJS..217...31M)
            Kepler planet candidate vetting diagnostics
    """
    time_arr, flux_arr = _extract_arrays_from_tpf(tpf)

    # Build check config from user-provided overrides
    check_config = None
    if config:
        from bittr_tess_vetter.validation.base import CheckConfig

        check_config = CheckConfig(
            enabled=True,
            threshold=config.get("fail_stability_threshold", 0.5),
            additional={
                "warn_stability_threshold": config.get("warn_stability_threshold", 0.7),
            },
        )

    # Default radii start at 1.5px per v2 spec (not 1.0px which is fragile)
    aperture_radii = radii_px if radii_px is not None else [1.5, 2.0, 2.5, 3.0, 3.5]

    # Create internal check instance
    check = ApertureDependenceCheck(
        config=check_config,
        tpf_data=flux_arr,
        time=time_arr,
        aperture_radii=aperture_radii,
    )

    # Create a minimal TransitCandidate for the internal check
    from bittr_tess_vetter.domain.detection import TransitCandidate

    internal_candidate = TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=candidate.depth or 0.001,
    )

    result = check.run(internal_candidate)
    return _convert_result(result)


# Define the default enabled checks and their order
_DEFAULT_PIXEL_CHECKS = ["V08", "V09", "V10"]


def vet_pixel(
    tpf: TPFStamp,
    candidate: Candidate,
    *,
    enabled: set[str] | None = None,
    config: dict[str, dict[str, Any]] | None = None,
) -> list[CheckResult]:
    """Run all pixel-level vetting checks (V08-V10).

    This is the orchestrator function for pixel-level vetting.
    Runs checks in order V08-V10, optionally filtered by the enabled set.

    Args:
        tpf: Target Pixel File data (TPFStamp)
        candidate: Transit candidate with ephemeris and depth
        enabled: Set of check IDs to run (e.g., {"V08", "V10"}).
            If None, runs all checks.
        config: Per-check configuration dict, keyed by check ID.
            E.g., {"V08": {"fail_shift_threshold": 0.8}}

    Returns:
        List of CheckResult objects for each enabled check

    Example:
        >>> from bittr_tess_vetter.api import TPFStamp, Candidate, Ephemeris, vet_pixel
        >>> tpf = TPFStamp(time=time, flux=flux_cube)
        >>> eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
        >>> cand = Candidate(ephemeris=eph, depth_ppm=1000)
        >>> results = vet_pixel(tpf, cand)
        >>> for r in results:
        ...     print(f"{r.id} {r.name}: {'PASS' if r.passed else 'FAIL'}")

    Novelty: standard

    References:
        See individual check functions (V08-V10) for specific citations.
        General methodology follows the Kepler Data Validation approach:
        [1] Twicken et al. 2018, PASP 130, 064502 (2018PASP..130f4502T)
        [2] Bryson et al. 2013, PASP 125, 889 (2013PASP..125..889B)
    """
    # Determine which checks to run
    checks_to_run = (
        _DEFAULT_PIXEL_CHECKS
        if enabled is None
        else [c for c in _DEFAULT_PIXEL_CHECKS if c in enabled]
    )

    config = config or {}
    results: list[CheckResult] = []

    for check_id in checks_to_run:
        check_config = config.get(check_id)

        if check_id == "V08":
            results.append(centroid_shift(tpf, candidate, config=check_config))
        elif check_id == "V09":
            results.append(difference_image_localization(tpf, candidate, config=check_config))
        elif check_id == "V10":
            results.append(aperture_dependence(tpf, candidate, config=check_config))

    return results


__all__ = [
    # Individual checks
    "centroid_shift",
    "difference_image_localization",
    "aperture_dependence",
    # Orchestrator
    "vet_pixel",
    # References
    "REFERENCES",
]
