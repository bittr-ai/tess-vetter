"""Physical transit model fitting API.

This module provides a public API for fitting physical transit models using
batman for light curve generation and optional MCMC sampling with emcee.

The main entry points are:
- fit_transit: Fit physical transit model with batman
- quick_estimate: Fast analytic parameter estimation

Dependencies:
- batman: Required for transit model computation
- ldtk: Optional, for limb darkening from stellar parameters
- emcee: Optional, for MCMC posterior sampling
- arviz: Optional, for MCMC convergence diagnostics

References:
    [1] Mandel & Agol 2002, ApJ 580, L171 (2002ApJ...580L.171M)
        Analytic light curve formulae for planetary transit searches
    [2] Kreidberg 2015, PASP 127, 1161 (2015PASP..127.1161K)
        batman: BAsic Transit Model cAlculatioN in Python
    [3] Foreman-Mackey et al. 2013, PASP 125, 306 (2013PASP..125..306F)
        emcee: The MCMC Hammer
    [4] Claret 2018, A&A 618, A20 (2018A&A...618A..20C)
        Limb darkening coefficients for TESS
    [5] Parviainen & Aigrain 2015, MNRAS 453, 3821 (2015MNRAS.453.3821P)
        LDTk: Limb Darkening Toolkit
    [6] Seager & Mallen-Ornelas 2003, ApJ 585, 1038 (2003ApJ...585.1038S)
        Transit shape and duration relations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

from tess_vetter.api.contracts import callable_input_schema_from_signature, opaque_object_schema
from tess_vetter.api.references import (
    CLARET_2018,
    CLARET_SOUTHWORTH_2022,
    ESPINOZA_JORDAN_2015,
    ESPINOZA_JORDAN_2016,
    FOREMAN_MACKEY_2013,
    KREIDBERG_2015,
    MANDEL_AGOL_2002,
    PARVIAINEN_2015,
    SEAGER_MALLEN_ORNELAS_2003,
    SING_2010,
    cite,
    cites,
)
from tess_vetter.api.types import Candidate, LightCurve, StellarParams

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# =============================================================================
# API boundary contracts
# =============================================================================

TRANSIT_FIT_BOUNDARY_SCHEMA_VERSION = 1
TRANSIT_FIT_MIN_USABLE_POINTS = 20

TransitFitMethod: TypeAlias = Literal["optimize", "mcmc"]
TransitFitStatus: TypeAlias = Literal["success", "failed", "error"]

TRANSIT_FIT_METHODS: tuple[TransitFitMethod, ...] = ("optimize", "mcmc")
TRANSIT_FIT_STATUSES: tuple[TransitFitStatus, ...] = ("success", "failed", "error")
TRANSIT_FIT_MCMC_FALLBACK_METHOD: TransitFitMethod = "optimize"


class TransitFitBoundaryContract(TypedDict):
    schema_version: int
    methods: tuple[TransitFitMethod, ...]
    statuses: tuple[TransitFitStatus, ...]
    default_method: TransitFitMethod
    mcmc_fallback_method: TransitFitMethod
    min_usable_points: int


TRANSIT_FIT_BOUNDARY_CONTRACT = TransitFitBoundaryContract(
    schema_version=TRANSIT_FIT_BOUNDARY_SCHEMA_VERSION,
    methods=TRANSIT_FIT_METHODS,
    statuses=TRANSIT_FIT_STATUSES,
    default_method="optimize",
    mcmc_fallback_method=TRANSIT_FIT_MCMC_FALLBACK_METHOD,
    min_usable_points=TRANSIT_FIT_MIN_USABLE_POINTS,
)

# =============================================================================
# Module-level references for programmatic access (generated from central registry)
# =============================================================================

REFERENCES = [
    ref.to_dict()
    for ref in [
        MANDEL_AGOL_2002,
        KREIDBERG_2015,
        FOREMAN_MACKEY_2013,
        CLARET_2018,
        PARVIAINEN_2015,
        ESPINOZA_JORDAN_2015,
        ESPINOZA_JORDAN_2016,
        SING_2010,
        CLARET_SOUTHWORTH_2022,
        SEAGER_MALLEN_ORNELAS_2003,
    ]
]


# =============================================================================
# Result Types
# =============================================================================


@dataclass(frozen=True)
class TransitFitResult:
    """Result of physical transit model fit.

    Attributes:
        fit_method: Method used ("optimize" or "mcmc")
        rp_rs: Planet-to-star radius ratio
        rp_rs_err: Uncertainty on Rp/Rs
        a_rs: Scaled semi-major axis (a/Rs)
        a_rs_err: Uncertainty on a/Rs
        inclination_deg: Orbital inclination in degrees
        inclination_err: Uncertainty on inclination
        t0_offset: Mid-transit time offset from input t0 (days)
        t0_offset_err: Uncertainty on t0 offset
        u1: First limb darkening coefficient (quadratic law)
        u2: Second limb darkening coefficient (quadratic law)
        transit_depth_ppm: Derived transit depth in ppm
        duration_hours: Derived transit duration in hours
        impact_parameter: Derived impact parameter b
        stellar_density_gcc: Derived stellar density in g/cm^3
        chi_squared: Reduced chi-squared of fit
        bic: Bayesian Information Criterion
        converged: Whether the fit converged successfully
        phase: Phase array for model light curve (for plotting)
        flux_model: Model flux values at phase points
        flux_data: Observed flux values (phase-folded)
        mcmc_diagnostics: MCMC-specific info (Gelman-Rubin, acceptance rate, etc.)
        status: "success", "error", or "skipped"
        error_message: Error message if status is "error"
    """

    fit_method: str
    rp_rs: float
    rp_rs_err: float
    a_rs: float
    a_rs_err: float
    inclination_deg: float
    inclination_err: float
    t0_offset: float
    t0_offset_err: float
    u1: float
    u2: float
    transit_depth_ppm: float
    duration_hours: float
    impact_parameter: float
    stellar_density_gcc: float
    chi_squared: float
    bic: float
    converged: bool
    phase: list[float] = field(default_factory=list)
    flux_model: list[float] = field(default_factory=list)
    flux_data: list[float] = field(default_factory=list)
    mcmc_diagnostics: dict[str, Any] | None = None
    status: str = "success"
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "status": self.status,
            "fit_method": self.fit_method,
            "parameters": {
                "rp_rs": {"value": self.rp_rs, "uncertainty": self.rp_rs_err},
                "a_rs": {"value": self.a_rs, "uncertainty": self.a_rs_err},
                "inclination_deg": {
                    "value": self.inclination_deg,
                    "uncertainty": self.inclination_err,
                },
                "t0_offset": {"value": self.t0_offset, "uncertainty": self.t0_offset_err},
                "u1": self.u1,
                "u2": self.u2,
            },
            "derived": {
                "transit_depth_ppm": round(self.transit_depth_ppm, 1),
                "duration_hours": round(self.duration_hours, 3),
                "impact_parameter": round(self.impact_parameter, 3),
                "stellar_density_gcc": round(self.stellar_density_gcc, 3),
            },
            "goodness_of_fit": {
                "chi_squared": round(self.chi_squared, 3),
                "bic": round(self.bic, 1),
            },
            "converged": self.converged,
            "mcmc_diagnostics": self.mcmc_diagnostics,
            "error_message": self.error_message,
        }


def _make_error_result(error_message: str) -> TransitFitResult:
    """Create an error result when dependencies are missing or fit fails."""
    return TransitFitResult(
        fit_method="none",
        rp_rs=0.0,
        rp_rs_err=0.0,
        a_rs=0.0,
        a_rs_err=0.0,
        inclination_deg=0.0,
        inclination_err=0.0,
        t0_offset=0.0,
        t0_offset_err=0.0,
        u1=0.0,
        u2=0.0,
        transit_depth_ppm=0.0,
        duration_hours=0.0,
        impact_parameter=0.0,
        stellar_density_gcc=0.0,
        chi_squared=0.0,
        bic=0.0,
        converged=False,
        status="error",
        error_message=error_message,
    )


# =============================================================================
# Public API Functions
# =============================================================================


@cites(
    cite(SEAGER_MALLEN_ORNELAS_2003, "Eq.3,9,19 transit shape relations"),
)
def quick_estimate(
    depth_ppm: float,
    duration_hours: float,
    period_days: float,
    stellar_density_gcc: float = 1.41,
) -> dict[str, float]:
    """Analytic initial guesses for transit parameters.

    Provides fast analytic estimates of transit parameters without fitting,
    useful as initial guesses for optimization or for quick assessments.

    Based on Seager & Mallen-Ornelas 2003 relations connecting transit
    observables to physical parameters.

    Args:
        depth_ppm: Transit depth in parts per million
        duration_hours: Total transit duration (T14) in hours
        period_days: Orbital period in days
        stellar_density_gcc: Stellar density in g/cm^3 (default: solar = 1.41)

    Returns:
        Dictionary with initial guesses:
        - rp_rs: Planet-to-star radius ratio
        - a_rs: Scaled semi-major axis
        - inc_deg: Orbital inclination in degrees

    Novelty: standard

    References:
        [1] Seager & Mallen-Ornelas 2003, ApJ 585, 1038 (2003ApJ...585.1038S)
            Equations 3, 9, 19: Transit shape and duration relations
    """
    # Import from internal module which has the implementation
    from tess_vetter.transit.batman_model import (
        quick_estimate as _quick_estimate,
    )

    result = _quick_estimate(depth_ppm, duration_hours, period_days, stellar_density_gcc)

    # Convert from internal format to API format
    return {
        "rp_rs": result["rp_rs"],
        "a_rs": result["a_rs"],
        "inc_deg": result["inc"],
    }


QUICK_ESTIMATE_CALL_SCHEMA = callable_input_schema_from_signature(quick_estimate)
QUICK_ESTIMATE_OUTPUT_SCHEMA = opaque_object_schema()


@cites(
    cite(MANDEL_AGOL_2002, "§2-3 analytic transit model with LD"),
    cite(KREIDBERG_2015, "§2 batman algorithm, §3 performance"),
    cite(CLARET_2018, "Tables 1-5 TESS LD coefficients"),
)
def fit_transit(
    lc: LightCurve,
    candidate: Candidate,
    stellar: StellarParams,
    *,
    method: TransitFitMethod = "optimize",
    fit_limb_darkening: bool = False,
    mcmc_samples: int = 2000,
    mcmc_burn: int = 500,
) -> TransitFitResult:
    """Fit physical transit model using batman.

    Derives planet-to-star radius ratio (Rp/Rs), scaled semi-major axis (a/Rs),
    and orbital inclination from the light curve. Automatically computes
    limb darkening coefficients from stellar parameters via ldtk (if available).

    This is a high-level wrapper around the internal batman model fitting
    routines, providing a clean interface for the public API.

    Args:
        lc: Light curve data (time, flux, flux_err arrays)
        candidate: Transit candidate with ephemeris (period, t0, duration)
        stellar: Stellar parameters (teff, logg required for limb darkening)
        method: "optimize" for fast L-BFGS-B, "mcmc" for full Bayesian posterior
        fit_limb_darkening: If True, fit LD coefficients (MCMC only).
            If False, use ldtk-derived or empirical values.
        mcmc_samples: Number of MCMC samples after burn-in (if method="mcmc")
        mcmc_burn: Burn-in samples to discard (if method="mcmc")

    Returns:
        TransitFitResult with fitted parameters and derived quantities.
        If batman is not installed, returns error result with status="error".
        If MCMC is requested but emcee is not installed, falls back to optimize.

    Novelty: standard

    References:
        [1] Mandel & Agol 2002, ApJ 580, L171 (2002ApJ...580L.171M)
            Section 2-3: Analytic transit model formulae with limb darkening
        [2] Kreidberg 2015, PASP 127, 1161 (2015PASP..127.1161K)
            Section 2: batman algorithm, Section 3: Performance
        [3] Claret 2018, A&A 618, A20 (2018A&A...618A..20C)
            Tables 1-5: TESS limb darkening coefficients
    """
    # Normalize + apply valid_mask/finite filtering.
    internal_lc = lc.to_internal()
    time = internal_lc.time[internal_lc.valid_mask]
    flux = internal_lc.flux[internal_lc.valid_mask]
    flux_err = internal_lc.flux_err[internal_lc.valid_mask]

    if len(time) < TRANSIT_FIT_MIN_USABLE_POINTS:
        return _make_error_result(
            "Insufficient usable points for transit fit "
            f"(need >={TRANSIT_FIT_MIN_USABLE_POINTS}, got {len(time)})"
        )

    # Check for batman dependency (only after validating inputs)
    try:
        import batman  # noqa: F401
    except ImportError:
        logger.error("batman package not installed")
        return _make_error_result(
            "batman not installed - required for transit fitting. "
            "Install with: pip install batman-package"
        )

    # Check for emcee if MCMC requested
    actual_method = method
    if method == "mcmc":
        try:
            import emcee  # noqa: F401
        except ImportError:
            logger.warning(
                "emcee not installed, falling back to optimize method. "
                "Install with: pip install 'tess-vetter[fit]'"
            )
            actual_method = TRANSIT_FIT_MCMC_FALLBACK_METHOD

    # Extract ephemeris
    period = candidate.ephemeris.period_days
    t0 = candidate.ephemeris.t0_btjd
    duration = candidate.ephemeris.duration_hours

    # Build stellar params dict for internal function
    # StellarParams uses 'metallicity' field for [Fe/H]
    stellar_dict: dict[str, float] = {
        "teff": stellar.teff if stellar.teff is not None else 5800.0,
        "logg": stellar.logg if stellar.logg is not None else 4.44,
        "feh": stellar.metallicity if stellar.metallicity is not None else 0.0,
    }
    # Compute stellar density from mass/radius if available
    # stellar_density_solar returns density in solar units; convert to g/cm^3
    # Solar density = 1.41 g/cm^3
    density_solar = stellar.stellar_density_solar()
    if density_solar is not None:
        stellar_dict["stellar_density_gcc"] = density_solar * 1.41

    # Import internal fitting function
    from tess_vetter.transit.batman_model import fit_transit_model

    try:
        internal_result = fit_transit_model(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=period,
            t0=t0,
            stellar_params=stellar_dict,
            duration=duration,
            fit_limb_darkening=fit_limb_darkening,
            method=actual_method,  # type: ignore[arg-type]
            mcmc_samples=mcmc_samples,
            mcmc_burn=mcmc_burn,
        )
    except Exception as e:
        logger.exception("Transit fit failed")
        return _make_error_result(f"Transit fit failed: {e}")

    # Convert internal result to API result type
    return TransitFitResult(
        fit_method=internal_result.fit_method,
        rp_rs=internal_result.rp_rs.value,
        rp_rs_err=internal_result.rp_rs.uncertainty,
        a_rs=internal_result.a_rs.value,
        a_rs_err=internal_result.a_rs.uncertainty,
        inclination_deg=internal_result.inc.value,
        inclination_err=internal_result.inc.uncertainty,
        t0_offset=internal_result.t0.value - t0,
        t0_offset_err=internal_result.t0.uncertainty,
        u1=internal_result.u1.value,
        u2=internal_result.u2.value,
        transit_depth_ppm=internal_result.transit_depth_ppm,
        duration_hours=internal_result.duration_hours,
        impact_parameter=internal_result.impact_parameter,
        stellar_density_gcc=internal_result.stellar_density_gcc,
        chi_squared=internal_result.chi_squared,
        bic=internal_result.bic,
        converged=internal_result.converged,
        phase=internal_result.phase,
        flux_model=internal_result.flux_model,
        flux_data=internal_result.flux_data,
        mcmc_diagnostics=internal_result.mcmc_diagnostics,
        status="success" if internal_result.converged else "failed",
        error_message=None,
    )


FIT_TRANSIT_CALL_SCHEMA = callable_input_schema_from_signature(fit_transit)
FIT_TRANSIT_OUTPUT_SCHEMA = opaque_object_schema()
