"""Batman physical transit model fitting with limb darkening.

This module provides physical transit modeling using batman for light curve
generation and optional MCMC sampling with emcee for full posterior estimation.

Features:
- Physical transit light curves with quadratic limb darkening
- Limb darkening coefficients from ldtk using stellar parameters
- Two fitting methods: "optimize" (fast) and "mcmc" (full posteriors)
- Automatic exposure time detection from data cadence
- Gelman-Rubin convergence diagnostics for MCMC

References:
- batman: Kreidberg 2015, PASP, 127, 1161
- ldtk: Parviainen & Aigrain 2015, MNRAS, 453, 3821
- Quick estimates: Seager & Mallen-Ornelas 2003, ApJ, 585, 1038
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass(frozen=True)
class ParameterEstimate:
    """Single parameter estimate with uncertainty.

    Attributes:
        value: Best-fit or median value
        uncertainty: 1-sigma uncertainty (half of 68% CI width)
        credible_interval_68: 16th and 84th percentile bounds (for MCMC)
    """

    value: float
    uncertainty: float
    credible_interval_68: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "value": round(self.value, 6),
            "uncertainty": round(self.uncertainty, 6),
        }
        if self.credible_interval_68 is not None:
            result["credible_interval_68"] = [
                round(self.credible_interval_68[0], 6),
                round(self.credible_interval_68[1], 6),
            ]
        return result


@dataclass(frozen=True)
class TransitFitResult:
    """Result from physical transit model fit.

    Contains fitted parameters, derived quantities, and goodness-of-fit metrics.

    Attributes:
        fit_method: "optimize" or "mcmc"
        stellar_params: Input stellar parameters (Teff, logg, feh)
        rp_rs: Planet-to-star radius ratio
        a_rs: Scaled semi-major axis (a/Rs)
        inc: Orbital inclination in degrees
        t0: Mid-transit time (BTJD)
        u1: First quadratic LD coefficient
        u2: Second quadratic LD coefficient
        transit_depth_ppm: Derived transit depth in ppm
        duration_hours: Derived transit duration in hours
        impact_parameter: Derived impact parameter b
        stellar_density_gcc: Derived mean stellar density in g/cm^3
        chi_squared: Reduced chi-squared of fit
        bic: Bayesian Information Criterion
        rms_ppm: RMS of residuals in ppm
        phase: Phase array for model light curve
        flux_model: Model flux values
        flux_data: Observed flux values (phase-folded)
        flux_err: Flux uncertainties
        mcmc_diagnostics: MCMC-specific info (only if method="mcmc")
        converged: Whether the fit converged successfully
    """

    fit_method: str
    stellar_params: dict[str, float]
    rp_rs: ParameterEstimate
    a_rs: ParameterEstimate
    inc: ParameterEstimate
    t0: ParameterEstimate
    u1: ParameterEstimate
    u2: ParameterEstimate
    transit_depth_ppm: float
    duration_hours: float
    impact_parameter: float
    stellar_density_gcc: float
    chi_squared: float
    bic: float
    rms_ppm: float
    phase: list[float]
    flux_model: list[float]
    flux_data: list[float]
    flux_err: list[float]
    mcmc_diagnostics: dict[str, Any] | None
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "status": "success" if self.converged else "failed",
            "fit_method": self.fit_method,
            "stellar_params": self.stellar_params,
            "parameters": {
                "rp_rs": self.rp_rs.to_dict(),
                "a_rs": self.a_rs.to_dict(),
                "inc": self.inc.to_dict(),
                "t0": self.t0.to_dict(),
                "u1": self.u1.to_dict(),
                "u2": self.u2.to_dict(),
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
                "rms_ppm": round(self.rms_ppm, 1),
            },
            "model_lc": {
                "phase": self.phase,
                "flux_model": self.flux_model,
                "flux_data": self.flux_data,
                "flux_err": self.flux_err,
            },
            "mcmc_diagnostics": self.mcmc_diagnostics,
        }


# =============================================================================
# Utility Functions
# =============================================================================


def detect_exposure_time(time: NDArray[np.float64]) -> float:
    """Detect exposure time from time array cadence.

    TESS has multiple cadence modes:
    - 20-second (FFI in some sectors)
    - 2-minute (standard short cadence)
    - 10-minute (standard FFI)

    Args:
        time: Time array in days (BTJD)

    Returns:
        Exposure time in days (for batman).
    """
    time_arr = np.asarray(time, dtype=np.float64)
    finite = np.isfinite(time_arr)
    if np.sum(finite) < 2:
        # Default to 2-minute cadence
        return 2.0 / 60.0 / 24.0  # 2 minutes in days

    t = np.sort(time_arr[finite])
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 2.0 / 60.0 / 24.0

    # Robustly reject large gaps (e.g., between sectors) using an IQR filter.
    q25 = float(np.percentile(dt, 25))
    q75 = float(np.percentile(dt, 75))
    iqr = q75 - q25
    if iqr > 0:
        lo = q25 - 1.5 * iqr
        hi = q75 + 1.5 * iqr
        dt = dt[(dt >= lo) & (dt <= hi)]
        if len(dt) == 0:
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]

    cadence_days = float(np.median(dt))
    return cadence_days


def quick_estimate(
    depth_ppm: float,
    duration_hours: float,
    period_days: float,
    stellar_density_gcc: float = 1.41,
) -> dict[str, float]:
    """Get analytic initial guesses for transit parameters.

    Based on Seager & Mallen-Ornelas 2003, ApJ, 585, 1038.
    Provides good starting values for optimization.

    Args:
        depth_ppm: Transit depth in parts per million
        duration_hours: Total transit duration (T14) in hours
        period_days: Orbital period in days
        stellar_density_gcc: Stellar density in g/cm^3 (default: solar)

    Returns:
        Dictionary with initial guesses for rp_rs, a_rs, inc, t0_offset
    """
    # Rp/Rs from depth (assuming no limb darkening - first approximation)
    rp_rs = np.sqrt(depth_ppm / 1e6)

    # a/Rs from stellar density and period (Kepler's 3rd law)
    # a/Rs = (G * rho_star * P^2 / (3*pi))^(1/3)
    grav_const = 6.674e-8  # cgs
    rho_star = stellar_density_gcc  # g/cm^3
    p_seconds = period_days * 86400
    a_rs_from_density = (grav_const * rho_star * p_seconds**2 / (3 * np.pi)) ** (1 / 3)

    # a/Rs from duration (simplified formula for small Rp/Rs and b~0)
    # T14 ~ P/pi * Rs/a * 2
    t14_days = duration_hours / 24.0
    a_rs_from_duration = period_days * 2 / (np.pi * t14_days)

    # Use average of the two estimates
    a_rs_avg = (a_rs_from_density + a_rs_from_duration) / 2

    # Clip to reasonable range
    a_rs_avg = float(np.clip(a_rs_avg, 2, 100))

    # Choose an inclination that corresponds to a transiting geometry.
    # For long-period planets, a/Rs can be large, and a fixed inclination (e.g. 88 deg)
    # implies b >> 1 (no transit), which can trap the optimizer in a flat-model minimum.
    b_init = float(np.clip(0.5, 0.0, max(0.0, 1.0 - rp_rs - 0.05)))
    cosi = float(np.clip(b_init / a_rs_avg, 0.0, 1.0))
    inc = float(np.degrees(np.arccos(cosi)))
    inc = float(np.clip(inc, 70.0, 89.99))

    return {
        "rp_rs": float(rp_rs),
        "a_rs": a_rs_avg,
        "inc": inc,
        "t0_offset": 0.0,
    }


def get_ld_coefficients(
    teff: float,
    logg: float,
    feh: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Get quadratic limb darkening coefficients from stellar parameters.

    Uses ldtk to compute TESS-bandpass limb darkening from PHOENIX models.
    Falls back to empirical values if ldtk is unavailable.

    Args:
        teff: Effective temperature in K
        logg: Surface gravity (log g in cgs)
        feh: Metallicity [Fe/H]

    Returns:
        Tuple of ((u1, u2), (u1_err, u2_err)) - coefficients and uncertainties
    """
    try:
        from ldtk import LDPSetCreator, TESSThroughput

        # Create TESS throughput filter
        tess_filter = TESSThroughput()

        # Set up limb darkening profile creator
        # Uncertainties on stellar params propagate to LD uncertainties
        sc = LDPSetCreator(
            teff=(teff, max(100, teff * 0.02)),  # 2% or 100K uncertainty
            logg=(logg, 0.1),
            z=(feh, 0.1),
            filters=[tess_filter],
        )

        # Create profiles and get quadratic coefficients
        ps = sc.create_profiles()
        coeffs, errs = ps.coeffs_qd(do_mc=True, n_mc_samples=1000)

        u1 = float(coeffs[0, 0])
        u2 = float(coeffs[0, 1])
        u1_err = float(errs[0, 0])
        u2_err = float(errs[0, 1])

        logger.info(f"ldtk LD coefficients: u1={u1:.3f}+-{u1_err:.3f}, u2={u2:.3f}+-{u2_err:.3f}")
        return (u1, u2), (u1_err, u2_err)

    except ImportError:
        logger.warning("ldtk not available, using empirical LD coefficients")
    except Exception as e:
        logger.warning(f"ldtk failed: {e}, using empirical LD coefficients")

    # Fallback: empirical values for solar-like star in TESS band
    # From Claret 2017 for Teff~5800K, logg~4.44, [Fe/H]~0
    u1 = 0.32
    u2 = 0.25
    # Conservative uncertainties for fallback
    u1_err = 0.10
    u2_err = 0.10

    return (u1, u2), (u1_err, u2_err)


def compute_batman_model(
    time: NDArray[np.float64],
    period: float,
    t0: float,
    rp_rs: float,
    a_rs: float,
    inc: float,
    u: tuple[float, float],
    exp_time: float | None = None,
) -> NDArray[np.float64]:
    """Compute batman transit light curve.

    Args:
        time: Time array in days (BTJD)
        period: Orbital period in days
        t0: Mid-transit time in days (BTJD)
        rp_rs: Planet-to-star radius ratio
        a_rs: Scaled semi-major axis (a/Rs)
        inc: Orbital inclination in degrees
        u: Quadratic limb darkening coefficients (u1, u2)
        exp_time: Exposure time in days (auto-detected if None)

    Returns:
        Model flux array (normalized to 1.0 out of transit)
    """
    import batman

    if exp_time is None:
        exp_time = detect_exposure_time(time)

    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = list(u)

    # Use supersampling for long-cadence data (>1 minute)
    cadence_minutes = exp_time * 24 * 60
    supersample = 3 if cadence_minutes > 1.0 else 1

    m = batman.TransitModel(params, time, supersample_factor=supersample, exp_time=exp_time)
    result: NDArray[np.float64] = m.light_curve(params)
    return result


def compute_derived_parameters(
    rp_rs: float,
    a_rs: float,
    inc: float,
    period: float,
) -> dict[str, float]:
    """Compute derived transit parameters.

    Args:
        rp_rs: Planet-to-star radius ratio
        a_rs: Scaled semi-major axis
        inc: Orbital inclination in degrees
        period: Orbital period in days

    Returns:
        Dictionary with derived parameters:
        - transit_depth_ppm: Transit depth in ppm
        - duration_hours: Transit duration in hours
        - impact_parameter: Impact parameter b
        - stellar_density_gcc: Stellar density in g/cm^3
    """
    # Transit depth (ppm)
    transit_depth_ppm = (rp_rs**2) * 1e6

    # Impact parameter
    inc_rad = np.radians(inc)
    impact_parameter = a_rs * np.cos(inc_rad)

    # Transit duration (T14) in hours
    # T14 = P/pi * arcsin(sqrt((1+Rp/Rs)^2 - b^2) / (a/Rs * sin(i)))
    factor = np.sqrt((1 + rp_rs) ** 2 - impact_parameter**2)
    if factor > 0 and a_rs * np.sin(inc_rad) > 0:
        duration_days = (period / np.pi) * np.arcsin(factor / (a_rs * np.sin(inc_rad)))
    else:
        # Fallback for edge cases
        duration_days = 0.1  # ~2.4 hours

    duration_hours = duration_days * 24.0

    # Stellar density from a/Rs and period (Kepler's 3rd law)
    # rho = 3*pi / (G * P^2) * (a/Rs)^3
    grav_const = 6.674e-8  # cgs
    p_seconds = period * 86400
    stellar_density_gcc = (3 * np.pi / (grav_const * p_seconds**2)) * (a_rs**3)

    return {
        "transit_depth_ppm": float(transit_depth_ppm),
        "duration_hours": float(duration_hours),
        "impact_parameter": float(impact_parameter),
        "stellar_density_gcc": float(stellar_density_gcc),
    }


# =============================================================================
# Fitting Functions
# =============================================================================


def fit_optimize(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0_init: float,
    u_prior: tuple[float, float],
    initial_guess: dict[str, float] | None = None,
) -> tuple[NDArray[np.float64], float, bool]:
    """Fast least-squares transit fit using L-BFGS-B.

    Args:
        time: Time array in days
        flux: Normalized flux array
        flux_err: Flux uncertainties
        period: Orbital period in days (fixed)
        t0_init: Initial transit epoch in days
        u_prior: Limb darkening coefficients (fixed in optimize mode)
        initial_guess: Optional initial parameter guesses

    Returns:
        Tuple of (best_fit_params, chi_squared, converged)
        where best_fit_params = [rp_rs, a_rs, inc, t0_offset]
    """
    exp_time = detect_exposure_time(time)

    # Initial parameters
    if initial_guess is not None:
        x0 = np.array(
            [
                initial_guess.get("rp_rs", 0.1),
                initial_guess.get("a_rs", 15.0),
                initial_guess.get("inc", 88.0),
                initial_guess.get("t0_offset", 0.0),
            ],
            dtype=np.float64,
        )
    else:
        x0 = np.array([0.1, 15.0, 88.0, 0.0], dtype=np.float64)

    # Parameter bounds
    # rp_rs: 0.001 to 0.3 (sub-earth to ~3 Jupiter)
    # a_rs: 2 to 100 (close-in to wide orbits)
    # inc: 70 to 90 (bound below 90 to avoid degeneracy)
    # t0_offset: small range around initial t0
    bounds = [
        (0.001, 0.3),  # rp_rs
        (2.0, 100.0),  # a_rs
        (70.0, 90.0),  # inc (bounded to avoid 180-i degeneracy)
        (-0.05, 0.05),  # t0_offset (days)
    ]

    def neg_log_likelihood(theta: NDArray[np.float64]) -> float:
        rp_rs, a_rs, inc, t0_offset = theta

        # Enforce a transiting geometry; otherwise batman can produce a flat light curve
        # and the optimizer may settle on a non-physical "no transit" solution.
        impact_parameter = a_rs * np.cos(np.deg2rad(inc))
        if impact_parameter > 1.0 + rp_rs:
            return 1e9 + 1e8 * float((impact_parameter - (1.0 + rp_rs)) ** 2)

        try:
            model = compute_batman_model(
                time,
                period,
                t0_init + t0_offset,
                rp_rs,
                a_rs,
                inc,
                u_prior,
                exp_time,
            )
            chi2 = float(np.sum(((flux - model) / flux_err) ** 2))
            return chi2
        except Exception:
            return 1e10

    result = minimize(neg_log_likelihood, x0, bounds=bounds, method="L-BFGS-B")

    return result.x, float(result.fun), bool(result.success)


def fit_mcmc(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0_init: float,
    u_prior: tuple[float, float],
    u_err: tuple[float, float],
    fit_limb_darkening: bool = False,
    nwalkers: int = 32,
    nsamples: int = 2000,
    nburn: int = 500,
    initial_guess: dict[str, float] | None = None,
) -> tuple[NDArray[np.float64], dict[str, float], float, list[str]]:
    """MCMC transit fit with full posteriors.

    Uses emcee for sampling and arviz for Gelman-Rubin diagnostics.

    Args:
        time: Time array in days
        flux: Normalized flux array
        flux_err: Flux uncertainties
        period: Orbital period in days (fixed)
        t0_init: Initial transit epoch in days
        u_prior: Limb darkening coefficient means from ldtk
        u_err: Limb darkening coefficient uncertainties from ldtk
        fit_limb_darkening: If True, fit u1/u2; if False, fix to u_prior
        nwalkers: Number of MCMC walkers
        nsamples: Number of samples after burn-in
        nburn: Number of burn-in samples to discard
        initial_guess: Optional initial parameter guesses

    Returns:
        Tuple of (samples, gelman_rubin_dict, acceptance_rate, labels)
    """
    try:
        import arviz as az
        import emcee
    except ImportError as e:
        raise ImportError(
            "MCMC fitting requires the 'fit' extra. "
            "Install with: pip install 'bittr-tess-vetter[fit]'"
        ) from e

    exp_time = detect_exposure_time(time)

    if fit_limb_darkening:
        ndim = 6  # rp_rs, a_rs, inc, t0_offset, u1, u2
        labels = ["rp_rs", "a_rs", "inc", "t0_offset", "u1", "u2"]
    else:
        ndim = 4  # rp_rs, a_rs, inc, t0_offset
        labels = ["rp_rs", "a_rs", "inc", "t0_offset"]

    def log_prior(theta: NDArray[np.float64]) -> float:
        if fit_limb_darkening:
            rp_rs, a_rs, inc, t0_offset, u1, u2 = theta
        else:
            rp_rs, a_rs, inc, t0_offset = theta
            u1, u2 = u_prior

        # Uniform priors with bounds
        if not (0.001 < rp_rs < 0.3):
            return -np.inf
        if not (2.0 < a_rs < 100.0):
            return -np.inf
        if not (70.0 < inc < 90.0):
            return -np.inf
        if not (-0.05 < t0_offset < 0.05):
            return -np.inf

        # Require a transit to be geometrically possible (allow grazing).
        impact_parameter = a_rs * np.cos(np.deg2rad(inc))
        if impact_parameter > 1.0 + rp_rs:
            return -np.inf

        lp = 0.0

        # Gaussian priors on limb darkening from ldtk
        if fit_limb_darkening:
            if not (0.0 < u1 < 1.0):
                return -np.inf
            if not (-0.5 < u2 < 1.0):
                return -np.inf
            # Add Gaussian prior penalty from ldtk
            lp += -0.5 * ((u1 - u_prior[0]) / u_err[0]) ** 2
            lp += -0.5 * ((u2 - u_prior[1]) / u_err[1]) ** 2

        return lp

    def log_likelihood(theta: NDArray[np.float64]) -> float:
        if fit_limb_darkening:
            rp_rs, a_rs, inc, t0_offset, u1, u2 = theta
            u = (u1, u2)
        else:
            rp_rs, a_rs, inc, t0_offset = theta
            u = u_prior

        try:
            model = compute_batman_model(
                time,
                period,
                t0_init + t0_offset,
                rp_rs,
                a_rs,
                inc,
                u,
                exp_time,
            )
            chi2 = np.sum(((flux - model) / flux_err) ** 2)
            return float(-0.5 * chi2)
        except Exception:
            return -np.inf

    def log_probability(theta: NDArray[np.float64]) -> float:
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # Initialize walkers near initial guess
    if initial_guess is not None:
        if fit_limb_darkening:
            p0 = np.array(
                [
                    initial_guess.get("rp_rs", 0.1),
                    initial_guess.get("a_rs", 15.0),
                    initial_guess.get("inc", 88.0),
                    initial_guess.get("t0_offset", 0.0),
                    u_prior[0],
                    u_prior[1],
                ],
                dtype=np.float64,
            )
        else:
            p0 = np.array(
                [
                    initial_guess.get("rp_rs", 0.1),
                    initial_guess.get("a_rs", 15.0),
                    initial_guess.get("inc", 88.0),
                    initial_guess.get("t0_offset", 0.0),
                ],
                dtype=np.float64,
            )
    else:
        if fit_limb_darkening:
            p0 = np.array([0.1, 15.0, 88.0, 0.0, u_prior[0], u_prior[1]], dtype=np.float64)
        else:
            p0 = np.array([0.1, 15.0, 88.0, 0.0], dtype=np.float64)

    # Small perturbations for walker initialization
    pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)

    # Run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(pos, nsamples + nburn, progress=False)

    # Discard burn-in and flatten
    samples = sampler.get_chain(discard=nburn, flat=True)

    # Compute Gelman-Rubin diagnostic with arviz
    # emcee returns chains in shape (draws, chains, ndim); arviz expects (chains, draws, ...).
    chains_raw = sampler.get_chain(discard=nburn)
    assert chains_raw is not None, "MCMC did not produce chains"
    chains: NDArray[np.float64] = chains_raw
    chains = np.swapaxes(chains, 0, 1)  # (chains=nwalkers, draws, ndim)

    # Convert to arviz InferenceData
    var_dict = {labels[i]: chains[:, :, i] for i in range(ndim)}
    idata = az.from_dict(posterior=var_dict)
    rhat = az.rhat(idata)  # type: ignore[no-untyped-call]
    gelman_rubin = {label: float(rhat[label].values) for label in labels}

    acceptance_rate = float(np.mean(sampler.acceptance_fraction))

    # Ensure samples is not None (should always have samples after run_mcmc)
    assert samples is not None, "MCMC did not produce samples"
    return samples, gelman_rubin, acceptance_rate, labels


def compute_uncertainties(
    samples: NDArray[np.float64],
    labels: list[str],
) -> dict[str, ParameterEstimate]:
    """Compute parameter estimates from MCMC samples.

    Computes median and 68% credible intervals from posterior samples.

    Args:
        samples: MCMC samples array (n_samples, n_params)
        labels: Parameter names

    Returns:
        Dictionary mapping parameter names to ParameterEstimate objects
    """
    results = {}
    for i, label in enumerate(labels):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        value = float(mcmc[1])
        uncertainty = float((mcmc[2] - mcmc[0]) / 2)
        credible_interval = (float(mcmc[0]), float(mcmc[2]))
        results[label] = ParameterEstimate(
            value=value,
            uncertainty=uncertainty,
            credible_interval_68=credible_interval,
        )
    return results


# =============================================================================
# Main Fitting Function
# =============================================================================


def fit_transit_model(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period: float,
    t0: float,
    stellar_params: dict[str, float],
    duration: float | None = None,
    fit_limb_darkening: bool = False,
    method: Literal["optimize", "mcmc"] = "optimize",
    mcmc_samples: int = 2000,
    mcmc_burn: int = 500,
) -> TransitFitResult:
    """Fit physical transit model with limb darkening to light curve.

    Uses batman for transit light curve generation and either
    scipy.optimize or emcee MCMC for parameter estimation.

    Args:
        time: Time array in days (BTJD)
        flux: Normalized flux array (median ~1.0)
        flux_err: Flux uncertainties
        period: Orbital period in days (fixed)
        t0: Initial transit epoch in days (BTJD)
        stellar_params: Dictionary with 'teff', 'logg', 'feh' for LD coefficients
        duration: Initial duration guess in hours (auto-estimated if None)
        fit_limb_darkening: If True, fit LD coeffs; if False, use ldtk priors
        method: "optimize" (fast) or "mcmc" (full posteriors)
        mcmc_samples: Number of MCMC samples after burn-in
        mcmc_burn: Number of burn-in samples to discard

    Returns:
        TransitFitResult with fitted parameters and diagnostics
    """
    # Get stellar parameters with defaults
    teff = stellar_params.get("teff", 5800.0)
    logg = stellar_params.get("logg", 4.44)
    feh = stellar_params.get("feh", 0.0)

    # Get limb darkening coefficients and uncertainties
    (u1, u2), (u1_err, u2_err) = get_ld_coefficients(teff, logg, feh)
    u_prior = (u1, u2)
    u_err = (u1_err, u2_err)

    # Get initial guesses from quick estimate
    duration_hours = duration if duration is not None else 3.0

    # First, estimate depth from data (in-transit vs out-of-transit flux).
    # Use the (provided/estimated) duration to avoid selecting most of the orbit as
    # "in transit" for long-period planets, which biases depth toward ~0.
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    half_duration_phase = (duration_hours / 2.0) / 24.0 / period
    half_duration_phase = float(np.clip(half_duration_phase, 5e-4, 0.2))
    in_transit = np.abs(phase) < (1.25 * half_duration_phase)
    out_transit = np.abs(phase) > (3.0 * half_duration_phase)

    if np.sum(in_transit) > 5 and np.sum(out_transit) > 10:
        flux_in = float(np.median(flux[in_transit]))
        flux_out = float(np.median(flux[out_transit]))
        depth_ppm = max(100, (flux_out - flux_in) / flux_out * 1e6)
    else:
        depth_ppm = 10000  # 1% default

    # Get stellar density if available
    stellar_density_gcc = stellar_params.get("stellar_density_gcc", 1.41)

    initial_guess = quick_estimate(depth_ppm, duration_hours, period, stellar_density_gcc)

    # Run fitting
    if method == "optimize":
        best_params, chi2, converged = fit_optimize(
            time, flux, flux_err, period, t0, u_prior, initial_guess
        )

        rp_rs, a_rs, inc, t0_offset = best_params

        # Estimate uncertainties from chi2 curvature (simple approximation)
        # This is a rough estimate; MCMC gives proper uncertainties
        n_points = len(flux)
        rp_rs_err = rp_rs * 0.05  # ~5% uncertainty (rough)
        a_rs_err = a_rs * 0.05
        inc_err = 0.5  # degrees
        t0_err = 0.001  # days

        params = {
            "rp_rs": ParameterEstimate(rp_rs, rp_rs_err),
            "a_rs": ParameterEstimate(a_rs, a_rs_err),
            "inc": ParameterEstimate(inc, inc_err),
            "t0_offset": ParameterEstimate(t0_offset, t0_err),
            "u1": ParameterEstimate(u1, u1_err),
            "u2": ParameterEstimate(u2, u2_err),
        }

        # Compute chi-squared
        model = compute_batman_model(time, period, t0 + t0_offset, rp_rs, a_rs, inc, u_prior)
        residuals = flux - model
        dof = n_points - 4  # 4 free parameters
        reduced_chi2 = chi2 / max(dof, 1)

        mcmc_diagnostics = None

    else:  # method == "mcmc"
        samples, gelman_rubin, acceptance_rate, labels = fit_mcmc(
            time,
            flux,
            flux_err,
            period,
            t0,
            u_prior,
            u_err,
            fit_limb_darkening=fit_limb_darkening,
            nsamples=mcmc_samples,
            nburn=mcmc_burn,
            initial_guess=initial_guess,
        )

        params = compute_uncertainties(samples, labels)

        # Add LD parameters if not fitted
        if not fit_limb_darkening:
            params["u1"] = ParameterEstimate(u1, u1_err)
            params["u2"] = ParameterEstimate(u2, u2_err)

        # Get best-fit values
        rp_rs = params["rp_rs"].value
        a_rs = params["a_rs"].value
        inc = params["inc"].value
        t0_offset = params["t0_offset"].value

        # Compute chi-squared at best fit
        u = (
            params.get("u1", ParameterEstimate(u1, u1_err)).value,
            params.get("u2", ParameterEstimate(u2, u2_err)).value,
        )
        model = compute_batman_model(time, period, t0 + t0_offset, rp_rs, a_rs, inc, u)
        residuals = flux - model
        chi2 = float(np.sum((residuals / flux_err) ** 2))
        n_params = 6 if fit_limb_darkening else 4
        dof = len(flux) - n_params
        reduced_chi2 = chi2 / max(dof, 1)

        mcmc_diagnostics = {
            "n_samples": mcmc_samples,
            "n_burn": mcmc_burn,
            "acceptance_rate": round(acceptance_rate, 3),
            "gelman_rubin": {k: round(v, 4) for k, v in gelman_rubin.items()},
        }

        converged = all(v < 1.1 for v in gelman_rubin.values())

    # Compute derived parameters
    derived = compute_derived_parameters(rp_rs, a_rs, inc, period)

    # Compute BIC
    n_points = len(flux)
    n_params = 6 if (method == "mcmc" and fit_limb_darkening) else 4
    bic = chi2 + n_params * np.log(n_points)

    # Compute RMS
    rms_ppm = float(np.std(residuals) * 1e6)

    # Build model light curve for output (phase-folded)
    phase_output = np.linspace(-0.1, 0.1, 500)
    time_output = t0 + phase_output * period
    u = (
        params.get("u1", ParameterEstimate(u1, u1_err)).value,
        params.get("u2", ParameterEstimate(u2, u2_err)).value,
    )
    flux_model_output = compute_batman_model(
        time_output, period, t0 + t0_offset, rp_rs, a_rs, inc, u
    )

    # Bin observed data to phase
    obs_phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    phase_mask = np.abs(obs_phase) < 0.15
    phase_binned = obs_phase[phase_mask]
    flux_binned = flux[phase_mask]
    flux_err_binned = flux_err[phase_mask]

    # Sort by phase
    sort_idx = np.argsort(phase_binned)
    phase_binned = phase_binned[sort_idx]
    flux_binned = flux_binned[sort_idx]
    flux_err_binned = flux_err_binned[sort_idx]

    return TransitFitResult(
        fit_method=method,
        stellar_params={"teff": teff, "logg": logg, "feh": feh},
        rp_rs=params["rp_rs"],
        a_rs=params["a_rs"],
        inc=params["inc"],
        t0=ParameterEstimate(t0 + t0_offset, params["t0_offset"].uncertainty),
        u1=params.get("u1", ParameterEstimate(u1, u1_err)),
        u2=params.get("u2", ParameterEstimate(u2, u2_err)),
        transit_depth_ppm=derived["transit_depth_ppm"],
        duration_hours=derived["duration_hours"],
        impact_parameter=derived["impact_parameter"],
        stellar_density_gcc=derived["stellar_density_gcc"],
        chi_squared=reduced_chi2,
        bic=float(bic),
        rms_ppm=rms_ppm,
        phase=[float(p) for p in phase_output],
        flux_model=[float(f) for f in flux_model_output],
        flux_data=[float(f) for f in flux_binned[:500]] if len(flux_binned) > 0 else [],
        flux_err=[float(f) for f in flux_err_binned[:500]] if len(flux_err_binned) > 0 else [],
        mcmc_diagnostics=mcmc_diagnostics,
        converged=converged,
    )
