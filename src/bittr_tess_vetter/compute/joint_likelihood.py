"""Joint multi-sector likelihood computation for localization.

This module provides the core algorithms for computing joint log-likelihood
across multiple TESS sectors for host hypothesis testing. It replaces
Phase 2's simple voting with proper likelihood-based inference.

Phase 3.2 Joint Likelihood Model:
- Per-hypothesis joint log-likelihood: sum(sector_weight * sector_loglike)
- Automatic sector quality weighting based on residual diagnostics
- Best hypothesis selection with ambiguity detection

CRITICAL: This module must remain pure compute:
- NO file I/O (open, Path, etc.)
- NO network access
- NO imports of astropy, lightkurve
- ONLY numpy and scipy dependencies (minimal)

These functions are designed to work with pre-validated SectorEvidence objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .joint_inference_schemas import SectorEvidence

__all__ = [
    "assess_sector_quality",
    "compute_sector_weights",
    "compute_joint_log_likelihood",
    "compute_all_hypotheses_joint",
    "select_best_hypothesis_joint",
]


# =============================================================================
# Default Thresholds
# =============================================================================

DEFAULT_RESIDUAL_THRESHOLD = 2.0
"""Residual RMS threshold in MAD units for downweighting.

Sectors with normalized residual RMS above this threshold receive reduced weight.
"""

DEFAULT_MIN_SNR = 3.0
"""Minimum signal-to-noise ratio for full quality weight.

Sectors with low SNR (from shallow transits or high noise) are downweighted.
"""

DEFAULT_MARGIN_THRESHOLD = 2.0
"""Log-likelihood margin threshold for resolved vs ambiguous verdict.

If delta_log_likelihood < margin_threshold, the result is AMBIGUOUS.
"""


# =============================================================================
# Sector Quality Assessment
# =============================================================================


def assess_sector_quality(
    evidence: SectorEvidence,
    *,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    min_snr: float = DEFAULT_MIN_SNR,
) -> tuple[float, str | None]:
    """Assess sector quality for weighting in joint inference.

    Evaluates a sector's quality based on residual RMS and signal characteristics
    to determine its weight contribution to joint likelihood.

    Parameters
    ----------
    evidence : SectorEvidence
        Per-sector evidence containing residual_rms and hypothesis scores.
    residual_threshold : float, optional
        Residual RMS threshold (in MAD units) above which sector is downweighted.
        Default is DEFAULT_RESIDUAL_THRESHOLD (2.0).
    min_snr : float, optional
        Minimum SNR for full quality weight.
        Default is DEFAULT_MIN_SNR (3.0).

    Returns
    -------
    tuple[float, str | None]
        (quality_weight, downweight_reason)
        - quality_weight: Weight in [0, 1] for this sector
        - downweight_reason: Explanation for reduced weight, or None if full weight

    Notes
    -----
    Quality assessment considers:
    1. High residual RMS: indicates poor model fit or systematics
    2. Low amplitude (proxy for SNR): indicates weak transit signal
    3. Few hypotheses: may indicate data issues

    The weight is computed as a product of penalty factors:
        weight = penalty_residual * penalty_snr

    Each penalty is in [0, 1], so the overall weight is also in [0, 1].
    """
    weight = 1.0
    reason: str | None = None

    # Check if sector already has quality_weight < 1 (pre-set by upstream)
    if evidence.quality_weight < 1.0:
        return evidence.quality_weight, evidence.downweight_reason

    # Check residual RMS
    # We use a soft penalty: weight = 1 / (1 + (residual_rms / threshold - 1))
    # This gives full weight at threshold, and decreasing weight above
    if evidence.residual_rms > 0:
        # Normalize by a baseline (use threshold as reference)
        # Higher residuals get lower weights via soft penalty
        normalized_residual = evidence.residual_rms / (residual_threshold * 1e-4)
        if normalized_residual > 1.0:
            residual_penalty = 1.0 / (1.0 + (normalized_residual - 1.0) ** 2)
            if residual_penalty < weight:
                weight = residual_penalty
                reason = "high_residual"

    # Check transit amplitude (proxy for SNR)
    if evidence.hypotheses:
        best_hyp = evidence.hypotheses[0]
        amplitude = best_hyp.get("fit_amplitude")
        if amplitude is not None and amplitude != 0:
            # Transit amplitude should be negative (flux decrease)
            # Estimate SNR as |amplitude| / residual_rms
            abs_amp = abs(float(amplitude))
            if evidence.residual_rms > 0 and abs_amp > 0:
                snr_estimate = abs_amp / evidence.residual_rms
                if snr_estimate < min_snr:
                    snr_penalty = snr_estimate / min_snr
                    if snr_penalty < weight:
                        weight = snr_penalty
                        reason = "low_snr"

    # Ensure weight is in valid range
    weight = float(np.clip(weight, 0.0, 1.0))

    return weight, reason


def compute_sector_weights(
    sector_evidence: list[SectorEvidence],
    *,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
    min_snr: float = DEFAULT_MIN_SNR,
) -> dict[int, float]:
    """Compute quality-based weights for all sectors.

    Assesses each sector's quality and returns a mapping of sector numbers
    to their weights for use in joint likelihood computation.

    Parameters
    ----------
    sector_evidence : list[SectorEvidence]
        List of per-sector evidence from localization analysis.
    residual_threshold : float, optional
        Residual RMS threshold for downweighting.
        Default is DEFAULT_RESIDUAL_THRESHOLD.
    min_snr : float, optional
        Minimum SNR for full quality weight.
        Default is DEFAULT_MIN_SNR.

    Returns
    -------
    dict[int, float]
        Mapping of sector number to quality weight.
        All weights are in [0, 1].

    Examples
    --------
    >>> weights = compute_sector_weights(sector_evidence_list)
    >>> weights
    {15: 1.0, 42: 0.7, 67: 1.0}
    """
    weights: dict[int, float] = {}

    for evidence in sector_evidence:
        quality_weight, _ = assess_sector_quality(
            evidence,
            residual_threshold=residual_threshold,
            min_snr=min_snr,
        )
        weights[evidence.sector] = quality_weight

    return weights


# =============================================================================
# Joint Likelihood Computation
# =============================================================================


def _get_hypothesis_log_likelihood(
    evidence: SectorEvidence,
    hypothesis_id: str,
) -> float | None:
    """Extract log-likelihood for a hypothesis from sector evidence.

    Parameters
    ----------
    evidence : SectorEvidence
        Sector evidence containing hypothesis scores.
    hypothesis_id : str
        Source ID of the hypothesis.

    Returns
    -------
    float | None
        Log-likelihood (negative fit_loss) or None if not found.
    """
    for hyp in evidence.hypotheses:
        if hyp.get("source_id") == hypothesis_id:
            fit_loss = hyp.get("fit_loss")
            if fit_loss is not None and np.isfinite(fit_loss):
                # Convert fit_loss (SSE) to log-likelihood
                # Assuming Gaussian errors: log L = -0.5 * SSE / sigma^2
                # We use -fit_loss as a proxy (up to constant factor)
                return -float(fit_loss)
            return None
    return None


def compute_joint_log_likelihood(
    sector_evidence: list[SectorEvidence],
    hypothesis_id: str,
    *,
    sector_weights: dict[int, float] | None = None,
    downweight_high_residual: bool = True,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
) -> tuple[float, dict[int, float]]:
    """Compute joint log-likelihood for a hypothesis across sectors.

    Combines per-sector log-likelihoods using quality-based weighting to
    produce a single joint log-likelihood for the hypothesis.

    Parameters
    ----------
    sector_evidence : list[SectorEvidence]
        Per-sector evidence from localization analysis.
    hypothesis_id : str
        Source ID of the hypothesis to evaluate.
    sector_weights : dict[int, float] | None, optional
        Pre-computed sector weights. If None, weights are computed
        from sector quality assessment.
    downweight_high_residual : bool, optional
        Whether to downweight sectors with high residual RMS.
        Default is True.
    residual_threshold : float, optional
        Residual threshold for downweighting (MAD units).
        Default is DEFAULT_RESIDUAL_THRESHOLD.

    Returns
    -------
    tuple[float, dict[int, float]]
        (joint_loglike, per_sector_contributions)
        - joint_loglike: Sum of weighted per-sector log-likelihoods
        - per_sector_contributions: Sector -> weighted contribution mapping

    Notes
    -----
    Joint log-likelihood is computed as:

        joint_loglike = sum(w[s] * loglike[s]) for s in sectors

    where w[s] is the quality weight for sector s, and loglike[s] is
    the per-sector log-likelihood for the hypothesis.

    The per_sector_contributions dict shows each sector's weighted
    contribution, useful for debugging and diagnostics.

    Examples
    --------
    >>> joint_ll, contributions = compute_joint_log_likelihood(
    ...     sector_evidence,
    ...     "tic:123456789",
    ... )
    >>> joint_ll
    -245.67
    >>> contributions
    {15: -100.0, 42: -145.67}
    """
    # Compute sector weights if not provided
    if sector_weights is None:
        if downweight_high_residual:
            sector_weights = compute_sector_weights(
                sector_evidence,
                residual_threshold=residual_threshold,
            )
        else:
            sector_weights = {ev.sector: 1.0 for ev in sector_evidence}

    joint_loglike = 0.0
    per_sector_contributions: dict[int, float] = {}

    for evidence in sector_evidence:
        sector = evidence.sector
        weight = sector_weights.get(sector, 1.0)

        # Skip zero-weight sectors
        if weight <= 0:
            per_sector_contributions[sector] = 0.0
            continue

        # Get per-sector log-likelihood
        sector_ll = _get_hypothesis_log_likelihood(evidence, hypothesis_id)

        if sector_ll is None:
            # Hypothesis not found in this sector
            per_sector_contributions[sector] = 0.0
            continue

        # Weighted contribution
        contribution = weight * sector_ll
        per_sector_contributions[sector] = contribution
        joint_loglike += contribution

    return joint_loglike, per_sector_contributions


def compute_all_hypotheses_joint(
    sector_evidence: list[SectorEvidence],
    hypotheses: list[str],
    *,
    sector_weights: dict[int, float] | None = None,
    downweight_high_residual: bool = True,
    residual_threshold: float = DEFAULT_RESIDUAL_THRESHOLD,
) -> dict[str, float]:
    """Compute joint log-likelihood for all hypotheses.

    Evaluates each hypothesis across all sectors and returns their
    joint log-likelihoods for comparison.

    Parameters
    ----------
    sector_evidence : list[SectorEvidence]
        Per-sector evidence from localization analysis.
    hypotheses : list[str]
        List of source IDs to evaluate.
    sector_weights : dict[int, float] | None, optional
        Pre-computed sector weights. If None, weights are computed.
    downweight_high_residual : bool, optional
        Whether to downweight high-residual sectors.
        Default is True.
    residual_threshold : float, optional
        Residual threshold for downweighting.
        Default is DEFAULT_RESIDUAL_THRESHOLD.

    Returns
    -------
    dict[str, float]
        Mapping of hypothesis source_id to joint log-likelihood.
        Higher values indicate better fit.

    Examples
    --------
    >>> joint_loglikes = compute_all_hypotheses_joint(
    ...     sector_evidence,
    ...     ["tic:123456789", "gaia_dr3:987654321"],
    ... )
    >>> joint_loglikes
    {'tic:123456789': -200.0, 'gaia_dr3:987654321': -250.0}
    """
    # Pre-compute weights once for all hypotheses
    if sector_weights is None:
        if downweight_high_residual:
            sector_weights = compute_sector_weights(
                sector_evidence,
                residual_threshold=residual_threshold,
            )
        else:
            sector_weights = {ev.sector: 1.0 for ev in sector_evidence}

    joint_loglikes: dict[str, float] = {}

    for hypothesis_id in hypotheses:
        joint_ll, _ = compute_joint_log_likelihood(
            sector_evidence,
            hypothesis_id,
            sector_weights=sector_weights,
            downweight_high_residual=False,  # Already have weights
        )
        joint_loglikes[hypothesis_id] = joint_ll

    return joint_loglikes


def select_best_hypothesis_joint(
    joint_loglikes: dict[str, float],
    *,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> tuple[str, str, float]:
    """Select best hypothesis and determine verdict from joint log-likelihoods.

    Identifies the hypothesis with highest joint log-likelihood and
    determines whether the result is resolved or ambiguous based on
    the margin to the runner-up.

    Parameters
    ----------
    joint_loglikes : dict[str, float]
        Mapping of hypothesis source_id to joint log-likelihood.
    margin_threshold : float, optional
        Minimum log-likelihood margin to consider result "resolved".
        Default is DEFAULT_MARGIN_THRESHOLD (2.0).

    Returns
    -------
    tuple[str, str, float]
        (best_source_id, verdict, delta_loglike)
        - best_source_id: Source ID with highest joint log-likelihood
        - verdict: "ON_TARGET" | "OFF_TARGET" | "AMBIGUOUS"
        - delta_loglike: Log-likelihood difference (best - runner_up)

    Notes
    -----
    Verdict determination:
    - "ON_TARGET": best hypothesis contains "target" and margin >= threshold
    - "OFF_TARGET": best hypothesis is not target and margin >= threshold
    - "AMBIGUOUS": margin < threshold (cannot distinguish)

    If there is only one hypothesis, delta_loglike is set to inf and
    verdict is determined by whether it's the target.

    Examples
    --------
    >>> best_id, verdict, delta = select_best_hypothesis_joint(
    ...     {'tic:123': -100.0, 'gaia:456': -150.0},
    ... )
    >>> best_id
    'tic:123'
    >>> verdict
    'ON_TARGET'  # Assuming TIC ID contains 'target' in name
    >>> delta
    50.0
    """
    if not joint_loglikes:
        return "", "AMBIGUOUS", 0.0

    # Sort hypotheses by joint log-likelihood (descending)
    sorted_hyps = sorted(
        joint_loglikes.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    best_source_id = sorted_hyps[0][0]
    best_ll = sorted_hyps[0][1]

    # Compute delta to runner-up
    if len(sorted_hyps) > 1:
        runner_up_ll = sorted_hyps[1][1]
        delta_loglike = best_ll - runner_up_ll
    else:
        delta_loglike = float("inf")

    # Determine verdict
    # Check if best is "target" (case-insensitive check for common patterns)
    is_target = "target" in best_source_id.lower()

    if not np.isfinite(delta_loglike) or delta_loglike >= margin_threshold:
        verdict = "ON_TARGET" if is_target else "OFF_TARGET"
    else:
        verdict = "AMBIGUOUS"

    # Ensure delta_loglike is finite for output
    if not np.isfinite(delta_loglike):
        delta_loglike = 0.0 if len(sorted_hyps) <= 1 else float(best_ll - sorted_hyps[1][1])

    return best_source_id, verdict, float(delta_loglike)
