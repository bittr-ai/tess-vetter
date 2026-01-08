"""Model competition for artifact/systematics detection.

This module provides model competition between:
- M1: Transit-only (box transit model)
- M2: Transit + sinusoidal variability (at P and harmonics)
- M3: EB-like (odd/even depth difference + optional secondary eclipse)

Uses information criteria (AIC/BIC) to select the best model and compute
artifact risk scores. Known TESS systematic periods are also checked.

Phase 3.6 deliverable - reduces false positives by detecting non-transit signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "ModelType",
    "ModelFit",
    "ModelCompetitionResult",
    "ArtifactPrior",
    "KNOWN_ARTIFACT_PERIODS",
    "fit_transit_only",
    "fit_transit_sinusoid",
    "fit_eb_like",
    "run_model_competition",
    "compute_artifact_prior",
    "check_period_alias",
]

# =============================================================================
# Type Definitions
# =============================================================================

ModelType = Literal["transit_only", "transit_sinusoid", "eb_like"]


# =============================================================================
# Known TESS Systematic Periods
# =============================================================================

# Known TESS systematic periods (days)
KNOWN_ARTIFACT_PERIODS: list[float] = [
    13.7,  # spacecraft orbital period
    27.4,  # ~2x orbital
    1.0,  # daily systematics
    0.5,  # half-day
    6.85,  # ~0.5x orbital
    41.1,  # ~3x orbital
]


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class ModelFit:
    """Result of fitting a single model.

    Attributes:
        model_type: Type of model fitted
        n_params: Number of free parameters
        log_likelihood: Log-likelihood of the fit
        aic: Akaike Information Criterion (-2*logL + 2*k)
        bic: Bayesian Information Criterion (-2*logL + k*ln(n))
        residual_rms: RMS of residuals
        fitted_params: Dictionary of fitted parameter values
    """

    model_type: ModelType
    n_params: int
    log_likelihood: float
    aic: float
    bic: float
    residual_rms: float
    fitted_params: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_type": self.model_type,
            "n_params": self.n_params,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "residual_rms": self.residual_rms,
            "fitted_params": self.fitted_params,
        }


@dataclass
class ModelCompetitionResult:
    """Result of model competition.

    Attributes:
        fits: Dictionary mapping model type to ModelFit result
        winner: The model type with the best (lowest) BIC
        winner_margin: Delta BIC vs second best model
        model_competition_label: Human-readable label for the result
        artifact_risk: Probability signal is artifact/EB (0-1)
        warnings: List of warning messages
    """

    fits: dict[ModelType, ModelFit]
    winner: ModelType
    winner_margin: float
    model_competition_label: str  # "TRANSIT" | "SINUSOID" | "EB_LIKE" | "AMBIGUOUS"
    artifact_risk: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "fits": {k: v.to_dict() for k, v in self.fits.items()},
            "winner": self.winner,
            "winner_margin": self.winner_margin,
            "model_competition_label": self.model_competition_label,
            "artifact_risk": self.artifact_risk,
            "warnings": self.warnings,
        }


@dataclass
class ArtifactPrior:
    """Prior probability of artifact based on metadata.

    Attributes:
        period_alias_risk: Risk from proximity to known artifact periods (0-1)
        sector_quality_risk: Risk from sector quality flags (0-1)
        scattered_light_risk: Risk from scattered light periods (0-1)
        combined_risk: Weighted combination of all risks (0-1)
    """

    period_alias_risk: float
    sector_quality_risk: float
    scattered_light_risk: float
    combined_risk: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "period_alias_risk": self.period_alias_risk,
            "sector_quality_risk": self.sector_quality_risk,
            "scattered_light_risk": self.scattered_light_risk,
            "combined_risk": self.combined_risk,
        }


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_log_likelihood(
    residuals: np.ndarray,
    flux_err: np.ndarray,
) -> float:
    """Compute Gaussian log-likelihood from residuals and errors.

    Args:
        residuals: Array of residuals (observed - model)
        flux_err: Array of flux uncertainties

    Returns:
        Log-likelihood value
    """
    # Gaussian log-likelihood: -0.5 * sum((r/sigma)^2 + log(2*pi*sigma^2))
    var = flux_err**2
    chi2 = float(np.sum(residuals**2 / var))
    log_det = float(np.sum(np.log(2 * np.pi * var)))
    return -0.5 * (chi2 + log_det)


def _compute_aic_bic(
    log_likelihood: float,
    n_params: int,
    n_points: int,
) -> tuple[float, float]:
    """Compute AIC and BIC from log-likelihood.

    Args:
        log_likelihood: Log-likelihood of the model
        n_params: Number of free parameters
        n_points: Number of data points

    Returns:
        Tuple of (AIC, BIC)
    """
    aic = -2 * log_likelihood + 2 * n_params
    bic = -2 * log_likelihood + n_params * np.log(n_points)
    return float(aic), float(bic)


def _box_transit_template(
    time: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
) -> np.ndarray:
    """Compute box transit template (1 in transit, 0 out of transit).

    Args:
        time: Time array in days (BTJD)
        period: Orbital period in days
        t0: Transit epoch in BTJD
        duration_hours: Transit duration in hours

    Returns:
        Template array (1 during transit, 0 outside)
    """
    duration_days = duration_hours / 24.0
    half_duration = duration_days / 2.0

    # Compute phase relative to t0
    phase = (time - t0) / period
    phase = phase - np.floor(phase + 0.5)  # Center on [-0.5, 0.5]

    # In transit if |phase * period| < half_duration
    in_transit = np.abs(phase * period) < half_duration
    return np.asarray(in_transit, dtype=np.float64)


# =============================================================================
# Model Fitting Functions
# =============================================================================


def fit_transit_only(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
) -> ModelFit:
    """Fit transit-only model (box transit).

    Uses weighted least squares to fit depth:
        model = 1 - depth * template

    Args:
        time: Time array in days (BTJD)
        flux: Normalized flux array
        flux_err: Flux uncertainty array
        period: Orbital period in days
        t0: Transit epoch in BTJD
        duration_hours: Transit duration in hours

    Returns:
        ModelFit with fitted depth and fit statistics
    """
    template = _box_transit_template(time, period, t0, duration_hours)
    n_points = len(time)

    # Weighted least squares for depth: minimize sum(w * (flux - 1 + depth*template)^2)
    # d/d(depth) = 0 gives: depth = sum(w * template * (1-flux)) / sum(w * template^2)
    w = 1.0 / (flux_err**2 + 1e-20)

    numerator = np.sum(w * template * (1.0 - flux))
    denominator = np.sum(w * template**2)

    # Handle case with no in-transit points
    depth = 0.0 if denominator < 1e-20 else float(numerator / denominator)

    # Compute model and residuals
    model = 1.0 - depth * template
    residuals = flux - model

    # Compute fit statistics
    log_lik = _compute_log_likelihood(residuals, flux_err)
    n_params = 1  # depth only (period, t0, duration are fixed)
    aic, bic = _compute_aic_bic(log_lik, n_params, n_points)
    residual_rms = float(np.sqrt(np.mean(residuals**2)))

    return ModelFit(
        model_type="transit_only",
        n_params=n_params,
        log_likelihood=log_lik,
        aic=aic,
        bic=bic,
        residual_rms=residual_rms,
        fitted_params={
            "depth": depth,
            "depth_ppm": depth * 1e6,
        },
    )


def fit_transit_sinusoid(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    *,
    n_harmonics: int = 2,
) -> ModelFit:
    """Fit transit + sinusoidal variability at P and harmonics.

    Model:
        model = 1 - depth * template + sum_k(A_k * sin(2*pi*k*t/P + phi_k))

    Uses linear least squares by expanding sin(omega*t + phi) as:
        sin(omega*t + phi) = a*cos(omega*t) + b*sin(omega*t)
    where A = sqrt(a^2 + b^2), phi = atan2(a, b)

    Args:
        time: Time array in days (BTJD)
        flux: Normalized flux array
        flux_err: Flux uncertainty array
        period: Orbital period in days
        t0: Transit epoch in BTJD
        duration_hours: Transit duration in hours
        n_harmonics: Number of sinusoidal harmonics to include

    Returns:
        ModelFit with fitted depth, sinusoid amplitudes, and fit statistics
    """
    template = _box_transit_template(time, period, t0, duration_hours)
    n_points = len(time)
    w = 1.0 / (flux_err**2 + 1e-20)

    # Build design matrix for weighted least squares
    # Columns: [template, cos(2*pi*t/P), sin(2*pi*t/P), cos(4*pi*t/P), sin(4*pi*t/P), ...]
    n_sinusoid_params = 2 * n_harmonics
    n_cols = 1 + n_sinusoid_params  # template + sinusoids

    design = np.zeros((n_points, n_cols), dtype=np.float64)
    design[:, 0] = template  # Transit template

    for k in range(1, n_harmonics + 1):
        omega = 2 * np.pi * k / period
        design[:, 2 * k - 1] = np.cos(omega * time)
        design[:, 2 * k] = np.sin(omega * time)

    # Weight the design matrix and target
    sqrt_w = np.sqrt(w)
    design_weighted = design * sqrt_w[:, np.newaxis]
    target_weighted = (1.0 - flux) * sqrt_w

    # Solve weighted least squares: X.T @ W @ X @ params = X.T @ W @ target
    # For (1 - flux) as target, transit depth is the first coefficient
    try:
        params, residuals_ls, rank, s = np.linalg.lstsq(
            design_weighted, target_weighted, rcond=None
        )
    except np.linalg.LinAlgError:
        # Fallback to transit-only if lstsq fails
        return fit_transit_only(time, flux, flux_err, period, t0, duration_hours)

    # Extract fitted parameters
    depth = float(params[0])

    # Compute sinusoid amplitudes and phases
    sinusoid_params: dict[str, float] = {}
    for k in range(1, n_harmonics + 1):
        a_cos = float(params[2 * k - 1])
        a_sin = float(params[2 * k])
        amplitude = np.sqrt(a_cos**2 + a_sin**2)
        phase = np.arctan2(a_cos, a_sin)  # radians
        sinusoid_params[f"amplitude_k{k}"] = amplitude
        sinusoid_params[f"phase_k{k}_rad"] = phase

    # Compute model and residuals
    model_offset = design @ params
    model = 1.0 - model_offset
    residuals_fit = flux - model

    # Compute fit statistics
    log_lik = _compute_log_likelihood(residuals_fit, flux_err)
    n_params = 1 + n_sinusoid_params  # depth + sinusoid coefficients
    aic, bic = _compute_aic_bic(log_lik, n_params, n_points)
    residual_rms = float(np.sqrt(np.mean(residuals_fit**2)))

    fitted_params = {
        "depth": depth,
        "depth_ppm": depth * 1e6,
        "n_harmonics": n_harmonics,
        **sinusoid_params,
    }

    return ModelFit(
        model_type="transit_sinusoid",
        n_params=n_params,
        log_likelihood=log_lik,
        aic=aic,
        bic=bic,
        residual_rms=residual_rms,
        fitted_params=fitted_params,
    )


def fit_eb_like(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
) -> ModelFit:
    """Fit EB-like model with odd/even depth difference and secondary eclipse.

    Model allows:
    - Different depths for odd and even transits
    - Optional secondary eclipse at phase 0.5

    Args:
        time: Time array in days (BTJD)
        flux: Normalized flux array
        flux_err: Flux uncertainty array
        period: Orbital period in days
        t0: Transit epoch in BTJD
        duration_hours: Transit duration in hours

    Returns:
        ModelFit with fitted odd/even depths, secondary depth, and fit statistics
    """
    n_points = len(time)
    w = 1.0 / (flux_err**2 + 1e-20)

    duration_days = duration_hours / 24.0
    half_duration = duration_days / 2.0

    # Compute phase relative to t0
    phase = (time - t0) / period
    phase_centered = phase - np.floor(phase + 0.5)  # Center on [-0.5, 0.5]

    # Determine which transit each point belongs to
    orbit_number = np.floor(phase + 0.5)
    is_odd = (orbit_number.astype(int) % 2) == 1

    # Primary transit (odd epochs)
    in_primary_odd = (np.abs(phase_centered * period) < half_duration) & is_odd
    # Primary transit (even epochs)
    in_primary_even = (np.abs(phase_centered * period) < half_duration) & ~is_odd

    # Secondary eclipse at phase 0.5 (assuming circular orbit)
    phase_secondary = phase_centered - 0.5
    phase_secondary = phase_secondary - np.round(phase_secondary)
    in_secondary = np.abs(phase_secondary * period) < half_duration

    # Build design matrix
    # Columns: [odd_transit, even_transit, secondary]
    design = np.zeros((n_points, 3), dtype=np.float64)
    design[:, 0] = in_primary_odd.astype(np.float64)
    design[:, 1] = in_primary_even.astype(np.float64)
    design[:, 2] = in_secondary.astype(np.float64)

    # Weighted least squares
    sqrt_w = np.sqrt(w)
    design_weighted = design * sqrt_w[:, np.newaxis]
    target_weighted = (1.0 - flux) * sqrt_w

    try:
        params, _, _, _ = np.linalg.lstsq(design_weighted, target_weighted, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to transit-only if lstsq fails
        return fit_transit_only(time, flux, flux_err, period, t0, duration_hours)

    # Extract parameters
    depth_odd = float(params[0])
    depth_even = float(params[1])
    depth_secondary = float(params[2])

    # Compute model and residuals
    model_offset = design @ params
    model = 1.0 - model_offset
    residuals = flux - model

    # Compute fit statistics
    log_lik = _compute_log_likelihood(residuals, flux_err)
    n_params = 3  # depth_odd, depth_even, depth_secondary
    aic, bic = _compute_aic_bic(log_lik, n_params, n_points)
    residual_rms = float(np.sqrt(np.mean(residuals**2)))

    # Compute derived quantities
    depth_avg = (depth_odd + depth_even) / 2.0
    depth_diff = abs(depth_odd - depth_even)
    depth_diff_frac = depth_diff / (depth_avg + 1e-20)

    fitted_params = {
        "depth_odd": depth_odd,
        "depth_even": depth_even,
        "depth_odd_ppm": depth_odd * 1e6,
        "depth_even_ppm": depth_even * 1e6,
        "depth_secondary": depth_secondary,
        "depth_secondary_ppm": depth_secondary * 1e6,
        "depth_avg": depth_avg,
        "depth_avg_ppm": depth_avg * 1e6,
        "odd_even_diff_frac": depth_diff_frac,
    }

    return ModelFit(
        model_type="eb_like",
        n_params=n_params,
        log_likelihood=log_lik,
        aic=aic,
        bic=bic,
        residual_rms=residual_rms,
        fitted_params=fitted_params,
    )


# =============================================================================
# Model Competition
# =============================================================================


def run_model_competition(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    *,
    bic_threshold: float = 10.0,
    n_harmonics: int = 2,
) -> ModelCompetitionResult:
    """Run all models and select winner based on BIC.

    Args:
        time: Time array in days (BTJD)
        flux: Normalized flux array
        flux_err: Flux uncertainty array
        period: Orbital period in days
        t0: Transit epoch in BTJD
        duration_hours: Transit duration in hours
        bic_threshold: BIC difference threshold for "strong evidence" (default 10.0)
        n_harmonics: Number of sinusoidal harmonics for transit_sinusoid model

    Returns:
        ModelCompetitionResult with all fits and winning model
    """
    warnings: list[str] = []

    # Fit all models
    fit_transit = fit_transit_only(time, flux, flux_err, period, t0, duration_hours)
    fit_sinusoid = fit_transit_sinusoid(
        time, flux, flux_err, period, t0, duration_hours, n_harmonics=n_harmonics
    )
    fit_eb = fit_eb_like(time, flux, flux_err, period, t0, duration_hours)

    fits: dict[ModelType, ModelFit] = {
        "transit_only": fit_transit,
        "transit_sinusoid": fit_sinusoid,
        "eb_like": fit_eb,
    }

    # Select winner by BIC (lower is better)
    bic_values: dict[ModelType, float] = {k: v.bic for k, v in fits.items()}
    sorted_models: list[ModelType] = sorted(
        bic_values.keys(), key=lambda k: bic_values[k]
    )
    winner: ModelType = sorted_models[0]
    second_best: ModelType = sorted_models[1]

    winner_margin = bic_values[second_best] - bic_values[winner]

    # Determine label and artifact risk
    if winner_margin < bic_threshold:
        # No clear winner
        label = "AMBIGUOUS"
        # Artifact risk is moderate if non-transit models are competitive
        artifact_risk = 0.5
        warnings.append(
            f"Model competition inconclusive (delta_BIC={winner_margin:.1f} < {bic_threshold})"
        )
    elif winner == "transit_only":
        label = "TRANSIT"
        artifact_risk = 0.0
    elif winner == "transit_sinusoid":
        label = "SINUSOID"
        artifact_risk = 0.8
        warnings.append(
            "Sinusoidal variability model preferred - signal may be stellar rotation or pulsation"
        )
    else:  # eb_like
        label = "EB_LIKE"
        artifact_risk = 0.9
        # Check for significant odd/even difference
        depth_diff_frac = fit_eb.fitted_params.get("odd_even_diff_frac", 0.0)
        if depth_diff_frac > 0.1:
            warnings.append(
                f"Significant odd/even depth difference ({depth_diff_frac:.1%}) - likely eclipsing binary"
            )
        if fit_eb.fitted_params.get("depth_secondary", 0.0) > 0.0001:
            warnings.append("Secondary eclipse detected - likely eclipsing binary")

    return ModelCompetitionResult(
        fits=fits,
        winner=winner,
        winner_margin=winner_margin,
        model_competition_label=label,
        artifact_risk=artifact_risk,
        warnings=warnings,
    )


# =============================================================================
# Artifact Prior Functions
# =============================================================================


def check_period_alias(
    period: float,
    known_periods: list[float] | None = None,
    tolerance: float = 0.01,
) -> tuple[bool, float | None, float]:
    """Check if period is near a known artifact period.

    Args:
        period: Period to check in days
        known_periods: List of known artifact periods (default: KNOWN_ARTIFACT_PERIODS)
        tolerance: Fractional tolerance for period matching (default 0.01 = 1%)

    Returns:
        Tuple of (is_alias, closest_known_period, fractional_difference)
        - is_alias: True if period matches a known artifact period
        - closest_known_period: The matched period (None if no match)
        - fractional_difference: Fractional difference to closest period
    """
    if known_periods is None:
        known_periods = KNOWN_ARTIFACT_PERIODS

    if not known_periods or period <= 0:
        return False, None, float("inf")

    # Find closest known period
    closest_period = None
    min_diff = float("inf")

    for known_p in known_periods:
        if known_p <= 0:
            continue
        frac_diff = abs(period - known_p) / known_p
        if frac_diff < min_diff:
            min_diff = frac_diff
            closest_period = known_p

    is_alias = min_diff <= tolerance

    return is_alias, closest_period, min_diff


def compute_artifact_prior(
    period: float,
    sector: int | None = None,
    quality_flags: dict[str, Any] | None = None,
    *,
    alias_tolerance: float = 0.01,
) -> ArtifactPrior:
    """Compute artifact prior from metadata.

    Args:
        period: Orbital period in days
        sector: TESS sector number (optional, for sector-specific quality)
        quality_flags: Dictionary of quality flags (optional)
        alias_tolerance: Fractional tolerance for period alias matching

    Returns:
        ArtifactPrior with risk scores
    """
    # Period alias risk
    is_alias, closest, frac_diff = check_period_alias(period, tolerance=alias_tolerance)
    if is_alias:
        # Higher risk the closer to known period
        period_alias_risk = 1.0 - frac_diff / alias_tolerance
    else:
        # Risk decreases with distance from known periods
        period_alias_risk = max(0.0, 1.0 - frac_diff) * 0.5

    # Sector quality risk (placeholder - would be populated from sector metadata)
    sector_quality_risk = 0.0
    if quality_flags is not None:
        # Check for known sector quality issues
        if quality_flags.get("scattered_light", False):
            sector_quality_risk = 0.5
        if quality_flags.get("high_background", False):
            sector_quality_risk = max(sector_quality_risk, 0.3)
        if quality_flags.get("momentum_dump", False):
            sector_quality_risk = max(sector_quality_risk, 0.2)

    # Scattered light risk (placeholder)
    scattered_light_risk = 0.0
    if sector is not None:
        # Some sectors have known scattered light issues
        # This would be populated from a sector quality database
        pass

    # Combined risk (weighted average)
    weights = [0.5, 0.3, 0.2]  # period alias, sector quality, scattered light
    risks = [period_alias_risk, sector_quality_risk, scattered_light_risk]
    combined_risk = sum(w * r for w, r in zip(weights, risks, strict=True))

    return ArtifactPrior(
        period_alias_risk=period_alias_risk,
        sector_quality_risk=sector_quality_risk,
        scattered_light_risk=scattered_light_risk,
        combined_risk=combined_risk,
    )
