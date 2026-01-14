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

from bittr_tess_vetter.api.references import (
    BATALHA_2010,
    BRYSON_2013,
    GUERRERO_2021,
    MULLALLY_2015,
    TORRES_2011,
    TWICKEN_2018,
    cite,
    cites,
)
from bittr_tess_vetter.api.types import Candidate, CheckResult, TPFStamp
from bittr_tess_vetter.validation.checks_pixel import (
    check_aperture_dependence_with_tpf,
    check_centroid_shift_with_tpf,
    check_pixel_level_lc_with_tpf,
)

if TYPE_CHECKING:
    pass

# Module-level references for programmatic access (generated from central registry)
REFERENCES = [
    ref.to_dict()
    for ref in [
        BRYSON_2013,
        TWICKEN_2018,
        BATALHA_2010,
        TORRES_2011,
        GUERRERO_2021,
        MULLALLY_2015,
    ]
]


def _convert_result(result: object) -> CheckResult:
    """Convert internal VetterCheckResult to facade CheckResult.

    Args:
        result: Internal VetterCheckResult (pydantic model)

    Returns:
        Facade CheckResult dataclass
    """
    # VetterCheckResult is a pydantic model with these attributes
    details = dict(result.details)  # type: ignore[attr-defined]
    details["_metrics_only"] = True
    return CheckResult(
        id=result.id,  # type: ignore[attr-defined]
        name=result.name,  # type: ignore[attr-defined]
        passed=None,
        confidence=result.confidence,  # type: ignore[attr-defined]
        details=details,
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


@cites(
    cite(BRYSON_2013, "§3.1 centroid offset test"),
    cite(TWICKEN_2018, "§4.1 difference image centroid offsets"),
)
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

    Args:
        tpf: Target Pixel File data (TPFStamp)
        candidate: Transit candidate with ephemeris
        config: Optional algorithm configuration overrides:
            - centroid_method: {"mean","median","huber"} (default "median")
            - significance_method: {"analytic","bootstrap","permutation"} (default "bootstrap")
            - n_bootstrap: int (default 1000)
            - bootstrap_seed: int | None
            - outlier_sigma: float (default 3.0)
            - window_policy_version: str (default "v1")

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

    from bittr_tess_vetter.domain.detection import TransitCandidate

    internal_candidate = TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=candidate.depth or 0.001,
        snr=0.0,  # Placeholder - not used by pixel checks
    )

    config = config or {}
    result = check_centroid_shift_with_tpf(
        tpf_data=flux_arr,
        time=time_arr,
        candidate=internal_candidate,
        centroid_method=config.get("centroid_method", "median"),
        significance_method=config.get("significance_method", "bootstrap"),
        n_bootstrap=int(config.get("n_bootstrap", 1000)),
        bootstrap_seed=config.get("bootstrap_seed"),
        outlier_sigma=float(config.get("outlier_sigma", 3.0)),
        window_policy_version=config.get("window_policy_version", "v1"),
    )
    return _convert_result(result)


@cites(
    cite(BRYSON_2013, "§3.2 difference image analysis"),
    cite(TORRES_2011, "§4 blend scenarios and centroid"),
    cite(TWICKEN_2018, "§4.2 difference image PRF fitting"),
)
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

    Args:
        tpf: Target Pixel File data (TPFStamp)
        candidate: Transit candidate with ephemeris
        target_rc: Expected target pixel (row, col). If None, uses TPF center.
        wcs_sources: List of reference sources with WCS coords (reserved for future use)
        config: Optional algorithm configuration overrides (currently unused).

    Returns:
        CheckResult with pixel-level localization including:
        - max_depth_pixel: (row, col) of pixel with maximum depth
        - max_depth_ppm: maximum transit depth in ppm
        - target_depth_ppm: transit depth at target pixel
        - concentration_ratio: target_depth / max_depth
        - distance_to_target_pixels: distance between max-depth pixel and target pixel

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

    # Convert target_rc to (row, col) tuple if provided
    target_pixel: tuple[int, int] | None = None
    if target_rc is not None:
        target_pixel = (int(target_rc[0]), int(target_rc[1]))

    # Create a minimal TransitCandidate for the internal check
    from bittr_tess_vetter.domain.detection import TransitCandidate

    internal_candidate = TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=candidate.depth or 0.001,
        snr=0.0,  # Placeholder - not used by pixel checks
    )

    result = check_pixel_level_lc_with_tpf(
        tpf_data=flux_arr,
        time=time_arr,
        candidate=internal_candidate,
        target_pixel=target_pixel,
    )
    return _convert_result(result)


@cites(
    cite(BRYSON_2013, "§3.3 contamination via aperture photometry"),
    cite(GUERRERO_2021, "§3.4 TESS aperture family analysis"),
    cite(MULLALLY_2015, "Kepler vetting diagnostics"),
)
def aperture_dependence(
    tpf: TPFStamp,
    candidate: Candidate,
    *,
    radii_px: list[float] | None = None,
    config: dict[str, Any] | None = None,
) -> CheckResult:
    """V10: Check if transit depth varies with aperture size.

    Measures transit depth at multiple aperture radii centered on the target.

    Args:
        tpf: Target Pixel File data (TPFStamp)
        candidate: Transit candidate with ephemeris and depth
        radii_px: List of aperture radii to test (pixels).
            Default: [1.5, 2.0, 2.5, 3.0, 3.5] per v2 spec.
        config: Optional algorithm configuration overrides (currently unused).

    Returns:
        CheckResult with aperture dependence analysis. The returned metrics
        include depth-vs-aperture behavior and a coarse stability assessment.

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

    # Default radii start at 1.5px per v2 spec (not 1.0px which is fragile)
    aperture_radii = radii_px if radii_px is not None else [1.5, 2.0, 2.5, 3.0, 3.5]

    # Create a minimal TransitCandidate for the internal check
    from bittr_tess_vetter.domain.detection import TransitCandidate

    internal_candidate = TransitCandidate(
        period=candidate.ephemeris.period_days,
        t0=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        depth=candidate.depth or 0.001,
        snr=0.0,  # Placeholder - not used by pixel checks
    )

    del config
    result = check_aperture_dependence_with_tpf(
        tpf_data=flux_arr,
        time=time_arr,
        candidate=internal_candidate,
        aperture_radii_px=aperture_radii,
        center_row_col=None,
    )
    return _convert_result(result)


# Define the default enabled checks and their order
_DEFAULT_PIXEL_CHECKS = ["V08", "V09", "V10"]


@cites(
    cite(BRYSON_2013, "Kepler pixel-level diagnostics"),
    cite(TWICKEN_2018, "Kepler Data Validation difference-image centroiding"),
    cite(GUERRERO_2021, "TESS TOI vetting procedures context"),
)
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
        ...     print(f\"{r.id} {r.name}: confidence={r.confidence:.2f}\")

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
