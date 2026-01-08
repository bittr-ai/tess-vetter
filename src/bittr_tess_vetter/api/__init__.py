"""Public API for bittr-tess-vetter.

This module provides the user-facing API for transit candidate vetting.

Types (v2):
- Ephemeris: Transit ephemeris (period, t0, duration)
- LightCurve: Simplified light curve container
- StellarParams: Stellar parameters from TIC
- CheckResult: Vetting check result
- Candidate: Transit candidate container (NEW in v2)
- TPFStamp: Target Pixel File data container (NEW in v2)
- VettingBundleResult: Orchestrator output with provenance (NEW in v2)

Main Entry Point (v2):
- vet_candidate: Run complete tiered vetting pipeline

Transit Primitives:
- odd_even_result: Odd/even depth comparison for EB detection

LC-Only Checks (V01-V05):
- odd_even_depth: V01 - Compare depth of odd vs even transits
- secondary_eclipse: V02 - Search for secondary eclipse
- duration_consistency: V03 - Check duration vs stellar density
- depth_stability: V04 - Check depth consistency across transits
- v_shape: V05 - Distinguish U-shaped vs V-shaped transits
- vet_lc_only: Orchestrator for all LC-only checks

Catalog Checks (V06-V07):
- nearby_eb_search: V06 - Search for nearby eclipsing binaries
- exofop_disposition: V07 - Check ExoFOP TOI dispositions
- vet_catalog: Orchestrator for catalog checks

Pixel Checks (V08-V10):
- centroid_shift: V08 - Detect centroid motion during transit
- difference_image_localization: V09 - Locate transit source
- aperture_dependence: V10 - Check depth vs aperture size
- vet_pixel: Orchestrator for pixel checks

Exovetter Checks (V11-V12):
- modshift: V11 - ModShift test for secondary eclipse detection
- sweet: V12 - SWEET test for stellar variability
- vet_exovetter: Orchestrator for exovetter checks

Example:
    >>> import numpy as np
    >>> from bittr_tess_vetter.api import (
    ...     LightCurve, Ephemeris, Candidate, vet_candidate
    ... )
    >>>
    >>> # Create light curve from your data
    >>> lc = LightCurve(time=time_array, flux=flux_array, flux_err=flux_err_array)
    >>>
    >>> # Define transit candidate
    >>> eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
    >>> candidate = Candidate(ephemeris=eph, depth_ppm=500)
    >>>
    >>> # Run complete vetting pipeline
    >>> result = vet_candidate(lc, candidate)
    >>> print(f"Passed: {result.n_passed}/{len(result.results)}")
    >>> for r in result.results:
    ...     status = "PASS" if r.passed else "FAIL"
    ...     print(f"{r.id} {r.name}: {status} (confidence={r.confidence:.2f})")
"""

# Types
from bittr_tess_vetter.api.types import (
    Candidate,
    CheckResult,
    Ephemeris,
    LightCurve,
    StellarParams,
    TPFStamp,
    VettingBundleResult,
)

# Main orchestrator
from bittr_tess_vetter.api.vet import vet_candidate

# LC-only checks (V01-V05)
from bittr_tess_vetter.api.lc_only import (
    depth_stability,
    duration_consistency,
    odd_even_depth,
    secondary_eclipse,
    v_shape,
    vet_lc_only,
)

# Catalog checks (V06-V07)
from bittr_tess_vetter.api.catalog import (
    exofop_disposition,
    nearby_eb_search,
    vet_catalog,
)

# Pixel checks (V08-V10)
from bittr_tess_vetter.api.pixel import (
    aperture_dependence,
    centroid_shift,
    difference_image_localization,
    vet_pixel,
)

# Exovetter checks (V11-V12)
from bittr_tess_vetter.api.exovetter import (
    modshift,
    sweet,
    vet_exovetter,
)

# Transit primitives
from bittr_tess_vetter.api.transit_primitives import odd_even_result

# Re-export OddEvenResult for completeness
from bittr_tess_vetter.transit.result import OddEvenResult

__all__ = [
    # Types (v2)
    "Ephemeris",
    "LightCurve",
    "StellarParams",
    "CheckResult",
    "Candidate",
    "TPFStamp",
    "VettingBundleResult",
    "OddEvenResult",
    # Main orchestrator (v2)
    "vet_candidate",
    # Transit primitives
    "odd_even_result",
    # LC-only checks (V01-V05)
    "odd_even_depth",
    "secondary_eclipse",
    "duration_consistency",
    "depth_stability",
    "v_shape",
    "vet_lc_only",
    # Catalog checks (V06-V07)
    "nearby_eb_search",
    "exofop_disposition",
    "vet_catalog",
    # Pixel checks (V08-V10)
    "centroid_shift",
    "difference_image_localization",
    "aperture_dependence",
    "vet_pixel",
    # Exovetter checks (V11-V12)
    "modshift",
    "sweet",
    "vet_exovetter",
]
