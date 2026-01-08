"""PRF-based aperture depth prediction and conflict detection.

This module provides PRF-aware prediction of transit depth vs aperture curves
for host hypothesis testing. It computes expected depth-vs-aperture curves
based on PRF models and detects conflicts between localization and aperture
evidence.

Key types:
- AperturePrediction: Predicted depth-vs-aperture curve for a hypothesis
- ApertureConflict: Detected conflict between evidence sources

Key functions:
- predict_depth_vs_aperture: Predict observed depth curve for a hypothesis
- predict_all_hypotheses: Predict curves for all hypotheses
- propagate_aperture_uncertainty: Add uncertainty estimates to predictions
- detect_aperture_conflict: Detect conflicts between localization and aperture
- compute_aperture_chi2: Compute chi-squared goodness of fit

CRITICAL: This module must remain pure compute:
- NO file I/O (open, Path, etc.)
- NO network access
- NO imports of astropy, lightkurve at module level
- ONLY numpy and scipy dependencies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .prf_psf import PRFModel

__all__ = [
    "AperturePrediction",
    "ApertureConflict",
    "predict_depth_vs_aperture",
    "predict_all_hypotheses",
    "propagate_aperture_uncertainty",
    "detect_aperture_conflict",
    "compute_aperture_chi2",
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True)
class AperturePrediction:
    """Predicted depth-vs-aperture curve for a hypothesis.

    Attributes:
        source_id: Unique identifier for the source hypothesis.
        radii_px: List of aperture radii in pixels.
        predicted_depths: Predicted observed depths (depth_obs = depth_true * f_host/f_total).
        host_fractions: Host fraction at each aperture (f_host / f_total).
        uncertainties: Propagated uncertainties for each depth (None if not computed).
    """

    source_id: str
    radii_px: list[float]
    predicted_depths: list[float]
    host_fractions: list[float]
    uncertainties: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_id": self.source_id,
            "radii_px": self.radii_px,
            "predicted_depths": self.predicted_depths,
            "host_fractions": self.host_fractions,
            "uncertainties": self.uncertainties,
        }


@dataclass(frozen=True)
class ApertureConflict:
    """Detected conflict between localization and aperture evidence.

    Attributes:
        localization_best: Source ID preferred by localization.
        aperture_best: Source ID preferred by aperture analysis.
        conflict_type: Type of conflict (always "CONFLICT_APERTURE_LOCALIZATION").
        margin: How strong the disagreement is (chi2 difference or similar).
        significance: Statistical significance (p-value or similar).
        explanation: Human-readable explanation of the conflict.
        recommended_verdict: Recommended verdict given the conflict.
    """

    localization_best: str
    aperture_best: str
    conflict_type: str = field(default="CONFLICT_APERTURE_LOCALIZATION")
    margin: float = 0.0
    significance: float = 0.0
    explanation: str = ""
    recommended_verdict: str = "AMBIGUOUS"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "localization_best": self.localization_best,
            "aperture_best": self.aperture_best,
            "conflict_type": self.conflict_type,
            "margin": self.margin,
            "significance": self.significance,
            "explanation": self.explanation,
            "recommended_verdict": self.recommended_verdict,
        }


# =============================================================================
# Aperture Mask Helpers
# =============================================================================


def _create_circular_aperture_mask(
    shape: tuple[int, int],
    center_row: float,
    center_col: float,
    radius_px: float,
) -> NDArray[np.bool_]:
    """Create a circular aperture mask.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the mask as (n_rows, n_cols).
    center_row : float
        Row coordinate of aperture center.
    center_col : float
        Column coordinate of aperture center.
    radius_px : float
        Radius of the aperture in pixels.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask where True indicates pixels inside the aperture.
    """
    n_rows, n_cols = shape
    row_indices, col_indices = np.ogrid[:n_rows, :n_cols]
    distance = np.sqrt((row_indices - center_row) ** 2 + (col_indices - center_col) ** 2)
    mask: NDArray[np.bool_] = distance <= radius_px
    return mask


# =============================================================================
# PRF-based Depth Prediction
# =============================================================================


def predict_depth_vs_aperture(
    hypothesis_row: float,
    hypothesis_col: float,
    depth_true: float,
    aperture_radii: list[float],
    prf_model: PRFModel,
    stamp_shape: tuple[int, int],
    *,
    aperture_center: tuple[float, float] | None = None,
    other_sources: list[tuple[float, float, float]] | None = None,
    background_params: tuple[float, float, float] | None = None,
) -> AperturePrediction:
    """Predict observed depth vs aperture for a hypothesis.

    Uses PRF model to compute host fraction at each aperture radius.
    The observed depth is modulated by the fraction of the host's flux
    captured by the aperture relative to total flux.

    Parameters
    ----------
    hypothesis_row : float
        Row coordinate of the hypothesis source.
    hypothesis_col : float
        Column coordinate of the hypothesis source.
    depth_true : float
        True transit depth (ppm) assuming 100% of signal is from this source.
    aperture_radii : list[float]
        List of aperture radii to test (in pixels).
    prf_model : PRFModel
        PRF model to use for flux computation.
    stamp_shape : tuple[int, int]
        Shape of the TPF stamp as (n_rows, n_cols).
    aperture_center : tuple[float, float] | None, optional
        Center of the apertures as (row, col). If None, uses stamp center.
    other_sources : list[tuple[float, float, float]] | None, optional
        List of other sources as (row, col, flux_ratio). flux_ratio is
        the ratio of the other source's flux to the hypothesis source.
        Used for dilution calculation.
    background_params : tuple[float, float, float] | None, optional
        Background gradient as (b0, bx, by). Currently unused but reserved
        for future improvements.

    Returns
    -------
    AperturePrediction
        Predicted depth-vs-aperture curve for this hypothesis.

    Notes
    -----
    The model is:
        depth_obs(ap) = depth_true * f_host(ap) / f_total(ap)

    where f_host(ap) is the PRF flux of the hypothesis source within the
    aperture, and f_total(ap) is the total flux from all sources within
    the aperture.
    """
    if aperture_center is None:
        aperture_center = ((stamp_shape[0] - 1) / 2.0, (stamp_shape[1] - 1) / 2.0)

    center_row, center_col = aperture_center

    # Evaluate PRF for hypothesis source (normalized)
    host_prf = prf_model.evaluate(
        hypothesis_row,
        hypothesis_col,
        stamp_shape,
        normalize=True,
    )

    # Evaluate PRF for other sources if provided
    other_prfs: list[tuple[NDArray[np.float64], float]] = []
    if other_sources:
        for src_row, src_col, flux_ratio in other_sources:
            src_prf = prf_model.evaluate(
                src_row,
                src_col,
                stamp_shape,
                normalize=True,
            )
            other_prfs.append((src_prf, flux_ratio))

    predicted_depths: list[float] = []
    host_fractions: list[float] = []

    for radius in aperture_radii:
        # Create circular aperture mask
        mask = _create_circular_aperture_mask(stamp_shape, center_row, center_col, radius)

        # Host flux within aperture (PRF is normalized, so this is fractional)
        f_host = float(np.sum(host_prf[mask]))

        # Total flux within aperture (host + other sources)
        f_total = f_host
        for src_prf, flux_ratio in other_prfs:
            f_total += float(np.sum(src_prf[mask])) * flux_ratio

        # Compute host fraction and observed depth
        host_fraction = f_host / f_total if f_total > 1e-10 else 1.0

        depth_obs = depth_true * host_fraction

        predicted_depths.append(depth_obs)
        host_fractions.append(host_fraction)

    return AperturePrediction(
        source_id="hypothesis",  # Will be set by caller
        radii_px=list(aperture_radii),
        predicted_depths=predicted_depths,
        host_fractions=host_fractions,
        uncertainties=None,
    )


def predict_all_hypotheses(
    hypotheses: list[dict[str, Any]],
    depth_estimate: float,
    aperture_radii: list[float],
    prf_model: PRFModel,
    stamp_shape: tuple[int, int],
    *,
    aperture_center: tuple[float, float] | None = None,
) -> dict[str, AperturePrediction]:
    """Predict depth-vs-aperture for all hypotheses.

    For each hypothesis, computes the expected observed depth at each
    aperture radius, accounting for dilution from other sources.

    Parameters
    ----------
    hypotheses : list[dict]
        List of hypotheses. Each dict must contain:
        - 'source_id': str, unique identifier
        - 'row': float, row pixel coordinate
        - 'col': float, column pixel coordinate
        - 'flux_ratio': float, optional flux ratio relative to brightest source
    depth_estimate : float
        Estimated true transit depth (ppm).
    aperture_radii : list[float]
        List of aperture radii to test (in pixels).
    prf_model : PRFModel
        PRF model to use for flux computation.
    stamp_shape : tuple[int, int]
        Shape of the TPF stamp as (n_rows, n_cols).
    aperture_center : tuple[float, float] | None, optional
        Center of the apertures as (row, col). If None, uses stamp center.

    Returns
    -------
    dict[str, AperturePrediction]
        Dictionary mapping source_id to AperturePrediction.
    """
    if not hypotheses:
        return {}

    if aperture_center is None:
        aperture_center = ((stamp_shape[0] - 1) / 2.0, (stamp_shape[1] - 1) / 2.0)

    predictions: dict[str, AperturePrediction] = {}

    for hyp in hypotheses:
        source_id = str(hyp.get("source_id", "unknown"))
        row = float(hyp.get("row", 0.0))
        col = float(hyp.get("col", 0.0))
        flux_ratio = float(hyp.get("flux_ratio", 1.0))

        # Build list of other sources (all except current hypothesis)
        other_sources: list[tuple[float, float, float]] = []
        for other in hypotheses:
            other_id = str(other.get("source_id", "unknown"))
            if other_id == source_id:
                continue
            other_row = float(other.get("row", 0.0))
            other_col = float(other.get("col", 0.0))
            other_flux = float(other.get("flux_ratio", 1.0))
            # Flux ratio relative to hypothesis source
            relative_flux = other_flux / flux_ratio if flux_ratio > 0 else 1.0
            other_sources.append((other_row, other_col, relative_flux))

        prediction = predict_depth_vs_aperture(
            hypothesis_row=row,
            hypothesis_col=col,
            depth_true=depth_estimate,
            aperture_radii=aperture_radii,
            prf_model=prf_model,
            stamp_shape=stamp_shape,
            aperture_center=aperture_center,
            other_sources=other_sources if other_sources else None,
        )

        # Update source_id in prediction
        predictions[source_id] = AperturePrediction(
            source_id=source_id,
            radii_px=prediction.radii_px,
            predicted_depths=prediction.predicted_depths,
            host_fractions=prediction.host_fractions,
            uncertainties=prediction.uncertainties,
        )

    return predictions


# =============================================================================
# Uncertainty Propagation
# =============================================================================


def propagate_aperture_uncertainty(
    prediction: AperturePrediction,
    depth_uncertainty: float,
    prf_position_uncertainty: float = 0.1,
    photometric_noise: list[float] | None = None,
) -> AperturePrediction:
    """Add uncertainty estimates to prediction.

    MVP implementation: quadrature sum of depth uncertainty and
    PRF position uncertainty contribution.

    Parameters
    ----------
    prediction : AperturePrediction
        The prediction to add uncertainties to.
    depth_uncertainty : float
        Uncertainty in the true depth estimate (ppm).
    prf_position_uncertainty : float, optional
        Position uncertainty in pixels. Default is 0.1 pixels.
        Contributes to depth uncertainty via PRF gradient.
    photometric_noise : list[float] | None, optional
        Per-aperture photometric noise in ppm. If provided, must have
        same length as radii_px.

    Returns
    -------
    AperturePrediction
        New prediction with uncertainties populated.

    Notes
    -----
    The uncertainty model is:
        sigma_depth^2 = (host_fraction * sigma_true)^2
                      + (depth_obs * sigma_position_effect)^2
                      + sigma_photometric^2

    The position effect is approximated as:
        sigma_position_effect ~ prf_position_uncertainty / sqrt(n_pix_in_aperture)

    This is a simplification that works reasonably well for typical
    TESS aperture sizes and PSF widths.
    """
    n_apertures = len(prediction.radii_px)
    uncertainties: list[float] = []

    for i in range(n_apertures):
        host_fraction = prediction.host_fractions[i]
        depth_obs = prediction.predicted_depths[i]
        radius = prediction.radii_px[i]

        # Depth uncertainty contribution (scaled by host fraction)
        sigma_depth = host_fraction * depth_uncertainty

        # Position uncertainty contribution
        # Approximate number of pixels in aperture
        n_pix = np.pi * radius**2
        sigma_position_effect = prf_position_uncertainty / np.sqrt(max(1.0, n_pix))
        sigma_position = abs(depth_obs) * sigma_position_effect

        # Photometric noise contribution
        sigma_photometric = 0.0
        if photometric_noise is not None and i < len(photometric_noise):
            sigma_photometric = photometric_noise[i]

        # Quadrature sum
        sigma_total = float(np.sqrt(sigma_depth**2 + sigma_position**2 + sigma_photometric**2))
        uncertainties.append(sigma_total)

    return AperturePrediction(
        source_id=prediction.source_id,
        radii_px=prediction.radii_px,
        predicted_depths=prediction.predicted_depths,
        host_fractions=prediction.host_fractions,
        uncertainties=uncertainties,
    )


# =============================================================================
# Conflict Detection
# =============================================================================


def compute_aperture_chi2(
    observed: list[float],
    predicted: list[float],
    uncertainties: list[float],
) -> tuple[float, float]:
    """Compute chi-squared and p-value for aperture fit.

    Parameters
    ----------
    observed : list[float]
        Observed depths at each aperture (ppm).
    predicted : list[float]
        Predicted depths at each aperture (ppm).
    uncertainties : list[float]
        Uncertainties for each aperture (ppm).

    Returns
    -------
    tuple[float, float]
        (chi_squared, p_value). The chi-squared statistic and its p-value.
        p_value is computed using the chi-squared distribution with
        n_points - 1 degrees of freedom.
    """
    n = len(observed)
    if n == 0 or len(predicted) != n or len(uncertainties) != n:
        return (float("nan"), float("nan"))

    # Filter out invalid values
    valid_mask = []
    for i in range(n):
        obs = observed[i]
        pred = predicted[i]
        unc = uncertainties[i]
        if np.isfinite(obs) and np.isfinite(pred) and np.isfinite(unc) and unc > 0:
            valid_mask.append(i)

    n_valid = len(valid_mask)
    if n_valid == 0:
        return (float("nan"), float("nan"))

    chi2 = 0.0
    for i in valid_mask:
        residual = (observed[i] - predicted[i]) / uncertainties[i]
        chi2 += residual**2

    # Degrees of freedom: n_points - 1 (fitting one parameter: depth scaling)
    dof = max(1, n_valid - 1)

    # P-value from chi-squared distribution
    p_value = float(1.0 - stats.chi2.cdf(chi2, dof))

    return (float(chi2), p_value)


def detect_aperture_conflict(
    localization_result: dict[str, Any],
    observed_depths: list[float],
    predictions: dict[str, AperturePrediction],
    *,
    observed_uncertainties: list[float] | None = None,
    margin_threshold: float = 2.0,
    chi2_threshold: float = 0.05,
) -> ApertureConflict | None:
    """Detect conflict between localization and aperture evidence.

    Compares the best hypothesis from localization with the best
    hypothesis from aperture curve fitting. If they disagree
    significantly, returns an ApertureConflict.

    Parameters
    ----------
    localization_result : dict
        Result from mlx_localize or similar. Expected keys:
        - 'consensus_best_source_id' or 'best_source_id': str
        - 'margin': float (optional)
    observed_depths : list[float]
        Observed depths at each aperture (ppm).
    predictions : dict[str, AperturePrediction]
        Dictionary mapping source_id to predicted depth curves.
    observed_uncertainties : list[float] | None, optional
        Uncertainties for observed depths. If None, uses 10% of
        observed values as default uncertainty.
    margin_threshold : float, optional
        Minimum chi2 difference to consider sources distinguishable.
        Default is 2.0.
    chi2_threshold : float, optional
        P-value threshold for good fit. Default is 0.05.

    Returns
    -------
    ApertureConflict | None
        Conflict object if significant conflict detected, None otherwise.

    Notes
    -----
    Conflict is detected when:
    1. Localization prefers source A
    2. Aperture analysis prefers source B (lowest chi2)
    3. The chi2 difference exceeds margin_threshold
    """
    if not predictions or not observed_depths:
        return None

    # Get localization best source
    localization_best = localization_result.get("consensus_best_source_id")
    if localization_best is None:
        localization_best = localization_result.get("best_source_id")
    if localization_best is None:
        return None
    localization_best = str(localization_best)

    # Default uncertainties if not provided
    if observed_uncertainties is None:
        observed_uncertainties = [
            max(abs(d) * 0.1, 10.0)
            for d in observed_depths  # 10% or 10 ppm minimum
        ]

    # Compute chi2 for each hypothesis
    chi2_by_source: dict[str, float] = {}
    pvalue_by_source: dict[str, float] = {}

    for source_id, pred in predictions.items():
        # Use prediction uncertainties if available, otherwise observed
        if pred.uncertainties is not None:
            uncertainties = pred.uncertainties
        else:
            uncertainties = observed_uncertainties

        chi2, pvalue = compute_aperture_chi2(
            observed_depths,
            pred.predicted_depths,
            uncertainties,
        )

        if np.isfinite(chi2):
            chi2_by_source[source_id] = chi2
            pvalue_by_source[source_id] = pvalue

    if not chi2_by_source:
        return None

    # Find best aperture fit (lowest chi2)
    aperture_best = min(chi2_by_source.keys(), key=lambda k: chi2_by_source[k])
    best_chi2 = chi2_by_source[aperture_best]

    # Check if localization_best is different from aperture_best
    if localization_best == aperture_best:
        return None  # No conflict

    # Compute margin (chi2 difference)
    localization_chi2 = chi2_by_source.get(localization_best)
    if localization_chi2 is None:
        # Localization best not in predictions - conflict
        margin = float("inf")
        significance = 0.0
    else:
        margin = localization_chi2 - best_chi2
        # Significance: p-value difference
        significance = pvalue_by_source.get(localization_best, 1.0)

    # Check if margin is significant
    if margin < margin_threshold:
        return None  # Not significant enough

    # Build explanation
    explanation = (
        f"Localization prefers '{localization_best}' but aperture analysis prefers "
        f"'{aperture_best}' (chi2 difference: {margin:.2f}). "
        f"Aperture fit p-value for localization best: {significance:.4f}."
    )

    # Recommended verdict: always AMBIGUOUS for Phase 3.4
    # Future: could differentiate based on calibration (significance < chi2_threshold)
    recommended_verdict = "AMBIGUOUS"

    return ApertureConflict(
        localization_best=localization_best,
        aperture_best=aperture_best,
        conflict_type="CONFLICT_APERTURE_LOCALIZATION",
        margin=margin,
        significance=significance,
        explanation=explanation,
        recommended_verdict=recommended_verdict,
    )
