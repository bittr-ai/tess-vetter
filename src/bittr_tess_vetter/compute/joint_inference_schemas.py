"""Joint multi-sector inference output schemas for localization.

This module defines the output contract for joint multi-sector localization,
which combines evidence from multiple TESS sectors into a single inference
about the transit host.

Phase 3.2 introduces joint inference (vs Phase 2's sector voting):
- Single joint likelihood over all sectors
- Per-sector evidence with quality weights
- Calibrated resolved_probability (added by Phase 3.5)

CRITICAL: This module must remain pure data structures:
- NO file I/O (open, Path, etc.)
- NO network access
- NO imports of astropy, lightkurve
- ONLY standard library and numpy dependencies

These schemas are designed to be JSON-serializable for downstream applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .pixel_host_hypotheses import (
    FLIP_RATE_MIXED_THRESHOLD,
    FLIP_RATE_UNSTABLE_THRESHOLD,
    MARGIN_RESOLVE_THRESHOLD,
    HypothesisScore,
)

__all__ = [
    "SectorEvidence",
    "JointInferenceResult",
    "sector_evidence_to_dict",
    "sector_evidence_from_dict",
    "joint_result_to_dict",
    "joint_result_from_dict",
    "to_evidence_block",
    "create_joint_result_from_sectors",
    # Inference modes
    "InferenceMode",
]

# Type alias for inference mode
InferenceMode = Literal["vote", "joint"]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SectorEvidence:
    """Per-sector evidence for joint inference.

    Captures the localization evidence from a single TESS sector,
    including hypothesis rankings, quality metrics, and nuisance parameters.

    Attributes
    ----------
    sector : int
        TESS sector number (1-99+).
    tpf_fits_ref : str
        Reference to the cached TPF FITS data (e.g., 'tpf_fits:123456789:15:spoc').
    hypotheses : list[HypothesisScore]
        Ranked list of source hypotheses for this sector.
        Best hypothesis is first (lowest fit_loss).
    residual_rms : float
        RMS of residuals after fitting the best hypothesis model.
        Higher values indicate poorer fit or systematics.
    quality_weight : float
        Weight assigned to this sector in joint inference (0-1).
        Lower weights reduce this sector's contribution to the joint likelihood.
    downweight_reason : str | None
        Explanation for reduced quality_weight, e.g.:
        - "high_residual": residual_rms exceeds threshold
        - "systematic_pattern": correlated residuals detected
        - "few_transits": insufficient transit events
        - None if quality_weight is 1.0 (no downweighting)
    nuisance_params : dict[str, float]
        Per-sector fitted nuisance parameters:
        - "background": fitted background level
        - "jitter": per-sector blur/jitter term
        - "amplitude_scale": flux amplitude scaling factor
    """

    sector: int
    tpf_fits_ref: str
    hypotheses: list[HypothesisScore]
    residual_rms: float
    quality_weight: float
    downweight_reason: str | None
    nuisance_params: dict[str, float] = field(default_factory=dict)


@dataclass
class JointInferenceResult:
    """Result of joint multi-sector localization.

    Combines evidence from multiple sectors into a single inference about
    which source is the transit host. Designed for Phase 3 joint likelihood
    inference (vs Phase 2's sector voting).

    Decision Fields
    ---------------
    joint_best_source_id : str
        Source ID of the preferred host hypothesis based on joint evidence.
        This is the source with highest posterior probability.
    verdict : str
        Localization verdict:
        - "ON_TARGET": transit is on the primary target (TIC ID)
        - "OFF_TARGET": transit is on a nearby source (blend/EB)
        - "AMBIGUOUS": cannot distinguish between hypotheses
        - "INVALID": insufficient data for localization

    Calibrated Probability Fields
    -----------------------------
    resolved_probability : float | None
        Calibrated probability that the verdict is correct.
        This is populated by Phase 3.5 calibration; None until then.
        Range: 0.0-1.0, where 1.0 = certain.
    calibration_version : str | None
        Version identifier for the calibration model used, e.g.:
        - "v3.5.0-discovery": discovery-grade calibration
        - "v3.5.0-strict": strict-grade calibration
        - None if not yet calibrated

    Evidence Fields
    ---------------
    joint_log_likelihood : float
        Log-likelihood of the best hypothesis under joint model.
        Higher values indicate better fit.
    delta_log_likelihood : float
        Log-likelihood difference between best and second-best hypothesis.
        Higher values indicate more confident discrimination.
    posterior_odds : float | None
        Posterior odds ratio of best vs second-best hypothesis.
        Computed as exp(delta_log_likelihood) when available.

    Per-Sector Breakdown
    --------------------
    sector_evidence : list[SectorEvidence]
        Per-sector evidence used in joint inference.
        Each entry contains hypothesis rankings, quality weights, etc.
    sector_weights : dict[int, float]
        Mapping of sector number to its weight in joint inference.
        Convenience accessor for sector_evidence[i].quality_weight.

    Consistency Fields
    ------------------
    flip_rate : float
        Fraction of sectors where the best hypothesis differs from joint_best_source_id.
        Range: 0.0-1.0, where 0.0 = all sectors agree.
    consistency_verdict : str
        Summary of multi-sector consistency:
        - "stable": flip_rate < 0.3 (sectors largely agree)
        - "mixed": 0.3 <= flip_rate < 0.5 (some disagreement)
        - "flipping": flip_rate >= 0.5 (sectors contradict each other)

    Diagnostic Fields
    -----------------
    hypotheses_considered : list[str]
        Source IDs of all hypotheses evaluated in joint inference.
    computation_time_seconds : float
        Wall-clock time for joint inference computation.
    warnings : list[str]
        Non-fatal warnings encountered during inference.
    blobs : dict[str, str]
        References to large artifacts stored in blob cache:
        - "residual_images": per-sector residual images
        - "per_sector_table": detailed per-sector results
    """

    # Decision
    joint_best_source_id: str
    verdict: str  # ON_TARGET | OFF_TARGET | AMBIGUOUS | INVALID

    # Calibrated probabilities (populated by Phase 3.5)
    resolved_probability: float | None
    calibration_version: str | None

    # Evidence
    joint_log_likelihood: float
    delta_log_likelihood: float
    posterior_odds: float | None

    # Per-sector breakdown
    sector_evidence: list[SectorEvidence]
    sector_weights: dict[int, float]

    # Consistency
    flip_rate: float
    consistency_verdict: str  # stable | mixed | flipping

    # Diagnostics
    hypotheses_considered: list[str]
    computation_time_seconds: float
    warnings: list[str] = field(default_factory=list)
    blobs: dict[str, str] = field(default_factory=dict)


# =============================================================================
# Serialization Helpers
# =============================================================================


def sector_evidence_to_dict(evidence: SectorEvidence) -> dict[str, Any]:
    """Convert SectorEvidence to JSON-serializable dict.

    Parameters
    ----------
    evidence : SectorEvidence
        The sector evidence to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary.
    """
    return {
        "sector": evidence.sector,
        "tpf_fits_ref": evidence.tpf_fits_ref,
        "hypotheses": [dict(h) for h in evidence.hypotheses],
        "residual_rms": float(evidence.residual_rms),
        "quality_weight": float(evidence.quality_weight),
        "downweight_reason": evidence.downweight_reason,
        "nuisance_params": {k: float(v) for k, v in evidence.nuisance_params.items()},
    }


def sector_evidence_from_dict(d: dict[str, Any]) -> SectorEvidence:
    """Create SectorEvidence from JSON dict.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary from sector_evidence_to_dict.

    Returns
    -------
    SectorEvidence
        Reconstructed dataclass instance.
    """
    hypotheses: list[HypothesisScore] = []
    for h in d.get("hypotheses", []):
        hypotheses.append(
            HypothesisScore(
                source_id=str(h.get("source_id", "")),
                source_name=str(h.get("source_name", "")),
                fit_loss=float(h.get("fit_loss", float("inf"))),
                delta_loss=float(h.get("delta_loss", 0.0)),
                rank=int(h.get("rank", 0)),
                fit_amplitude=h.get("fit_amplitude"),
                fit_background=h.get("fit_background"),
            )
        )

    return SectorEvidence(
        sector=int(d["sector"]),
        tpf_fits_ref=str(d["tpf_fits_ref"]),
        hypotheses=hypotheses,
        residual_rms=float(d.get("residual_rms", 0.0)),
        quality_weight=float(d.get("quality_weight", 1.0)),
        downweight_reason=d.get("downweight_reason"),
        nuisance_params={k: float(v) for k, v in d.get("nuisance_params", {}).items()},
    )


def joint_result_to_dict(result: JointInferenceResult) -> dict[str, Any]:
    """Convert JointInferenceResult to JSON-serializable dict.

    Parameters
    ----------
    result : JointInferenceResult
        The joint inference result to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary suitable for downstream applications.
    """
    return {
        # Decision
        "joint_best_source_id": result.joint_best_source_id,
        "verdict": result.verdict,
        # Calibration (may be None)
        "resolved_probability": result.resolved_probability,
        "calibration_version": result.calibration_version,
        # Evidence
        "joint_log_likelihood": float(result.joint_log_likelihood),
        "delta_log_likelihood": float(result.delta_log_likelihood),
        "posterior_odds": (
            float(result.posterior_odds) if result.posterior_odds is not None else None
        ),
        # Per-sector breakdown
        "sector_evidence": [sector_evidence_to_dict(se) for se in result.sector_evidence],
        "sector_weights": {int(k): float(v) for k, v in result.sector_weights.items()},
        # Consistency
        "flip_rate": float(result.flip_rate),
        "consistency_verdict": result.consistency_verdict,
        # Diagnostics
        "hypotheses_considered": list(result.hypotheses_considered),
        "computation_time_seconds": float(result.computation_time_seconds),
        "warnings": list(result.warnings),
        "blobs": dict(result.blobs),
    }


def joint_result_from_dict(d: dict[str, Any]) -> JointInferenceResult:
    """Create JointInferenceResult from JSON dict.

    Parameters
    ----------
    d : dict[str, Any]
        Dictionary from joint_result_to_dict.

    Returns
    -------
    JointInferenceResult
        Reconstructed dataclass instance.
    """
    sector_evidence = [sector_evidence_from_dict(se) for se in d.get("sector_evidence", [])]

    return JointInferenceResult(
        # Decision
        joint_best_source_id=str(d["joint_best_source_id"]),
        verdict=str(d["verdict"]),
        # Calibration
        resolved_probability=d.get("resolved_probability"),
        calibration_version=d.get("calibration_version"),
        # Evidence
        joint_log_likelihood=float(d.get("joint_log_likelihood", 0.0)),
        delta_log_likelihood=float(d.get("delta_log_likelihood", 0.0)),
        posterior_odds=d.get("posterior_odds"),
        # Per-sector breakdown
        sector_evidence=sector_evidence,
        sector_weights={int(k): float(v) for k, v in d.get("sector_weights", {}).items()},
        # Consistency
        flip_rate=float(d.get("flip_rate", 0.0)),
        consistency_verdict=str(d.get("consistency_verdict", "stable")),
        # Diagnostics
        hypotheses_considered=list(d.get("hypotheses_considered", [])),
        computation_time_seconds=float(d.get("computation_time_seconds", 0.0)),
        warnings=list(d.get("warnings", [])),
        blobs=dict(d.get("blobs", {})),
    )


def to_evidence_block(result: JointInferenceResult) -> dict[str, Any]:
    """Convert JointInferenceResult to evidence block for run_vetting_pipeline.

    Creates a summary evidence block suitable for inclusion in the vetting
    pipeline's evidence packet. This format is designed for the evidence-first
    vetting approach where multiple evidence sources are aggregated.

    Parameters
    ----------
    result : JointInferenceResult
        The joint inference result to summarize.

    Returns
    -------
    dict[str, Any]
        Evidence block with standardized keys:
        - "source": evidence source identifier
        - "verdict": localization verdict (categorical output of the inference)
        - "key_metrics": numeric metrics from the inference
        - "warnings": raw warnings emitted by the inference
        - "details_ref": blob reference for full details
    """
    return {
        "source": "joint_multi_sector_localization",
        "version": "3.2.0",
        "verdict": result.verdict,
        "key_metrics": {
            "joint_best_source_id": result.joint_best_source_id,
            "delta_log_likelihood": result.delta_log_likelihood,
            "flip_rate": result.flip_rate,
            "n_sectors": len(result.sector_evidence),
            "resolved_probability": result.resolved_probability,
            "calibration_version": result.calibration_version,
        },
        "warnings": list(result.warnings or []),
        "details_ref": result.blobs.get("per_sector_table"),
    }


# =============================================================================
# Factory Functions
# =============================================================================


def _create_joint_result_vote_mode(
    sector_results: list[SectorEvidence],
    hypotheses: list[str],
    *,
    margin_threshold: float,
    flip_rate_threshold: float,
    start_time: float,
) -> JointInferenceResult:
    """Create JointInferenceResult using Phase 2 weighted vote mode.

    This is the original implementation for backward compatibility.
    Uses weighted voting by quality_weight for each source's best-in-sector wins.
    """
    # Multi-sector case: weighted voting
    # Count weighted votes for each source_id
    vote_weights: dict[str, float] = {}
    total_log_likelihood: dict[str, float] = {}
    sector_best: list[str] = []

    for sector_ev in sector_results:
        if not sector_ev.hypotheses:
            continue

        best = sector_ev.hypotheses[0]
        source_id = best["source_id"]
        weight = sector_ev.quality_weight

        vote_weights[source_id] = vote_weights.get(source_id, 0.0) + weight

        # Sum log-likelihoods (negative fit_loss as proxy)
        ll = -best["fit_loss"] if np.isfinite(best["fit_loss"]) else 0.0
        total_log_likelihood[source_id] = total_log_likelihood.get(source_id, 0.0) + ll * weight

        sector_best.append(source_id)

    import time as time_module

    if not vote_weights:
        return JointInferenceResult(
            joint_best_source_id="",
            verdict="INVALID",
            resolved_probability=None,
            calibration_version=None,
            joint_log_likelihood=0.0,
            delta_log_likelihood=0.0,
            posterior_odds=None,
            sector_evidence=sector_results,
            sector_weights={se.sector: se.quality_weight for se in sector_results},
            flip_rate=0.0,
            consistency_verdict="stable",
            hypotheses_considered=hypotheses,
            computation_time_seconds=time_module.perf_counter() - start_time,
            warnings=["No valid hypotheses in any sector"],
            blobs={},
        )

    # Find best by weighted vote, tie-break by total log-likelihood
    sorted_sources = sorted(
        vote_weights.keys(),
        key=lambda s: (vote_weights[s], total_log_likelihood.get(s, 0.0)),
        reverse=True,
    )

    joint_best = sorted_sources[0]
    joint_ll = total_log_likelihood.get(joint_best, 0.0)

    # Compute delta log-likelihood to runner-up
    if len(sorted_sources) > 1:
        runner_up = sorted_sources[1]
        delta_ll = joint_ll - total_log_likelihood.get(runner_up, 0.0)
    else:
        delta_ll = float("inf") if np.isfinite(joint_ll) else 0.0

    # Compute flip rate
    n_confident = len([s for s in sector_best if s])
    if n_confident > 0:
        n_flips = sum(1 for s in sector_best if s and s != joint_best)
        flip_rate = n_flips / n_confident
    else:
        flip_rate = 0.0

    # Determine consistency verdict
    if flip_rate >= flip_rate_threshold:
        consistency_verdict = "flipping"
    elif flip_rate >= FLIP_RATE_MIXED_THRESHOLD:
        consistency_verdict = "mixed"
    else:
        consistency_verdict = "stable"

    # Determine final verdict
    is_target = "target" in joint_best.lower()
    if consistency_verdict == "flipping":
        verdict = "AMBIGUOUS"
    elif delta_ll >= margin_threshold:
        verdict = "ON_TARGET" if is_target else "OFF_TARGET"
    else:
        verdict = "AMBIGUOUS"

    # Compute posterior odds
    posterior_odds: float | None = None
    if np.isfinite(delta_ll) and 0 < delta_ll < 100:
        posterior_odds = float(np.exp(delta_ll))

    # Build sector_weights dict
    sector_weights = {se.sector: se.quality_weight for se in sector_results}

    # Collect warnings
    warnings: list[str] = []
    low_weight_sectors = [se.sector for se in sector_results if se.quality_weight < 0.5]
    if low_weight_sectors:
        warnings.append(f"Low quality weight sectors: {low_weight_sectors}")

    return JointInferenceResult(
        joint_best_source_id=joint_best,
        verdict=verdict,
        resolved_probability=None,  # Calibrated by Phase 3.5
        calibration_version=None,
        joint_log_likelihood=joint_ll,
        delta_log_likelihood=delta_ll if np.isfinite(delta_ll) else 0.0,
        posterior_odds=posterior_odds,
        sector_evidence=sector_results,
        sector_weights=sector_weights,
        flip_rate=flip_rate,
        consistency_verdict=consistency_verdict,
        hypotheses_considered=hypotheses,
        computation_time_seconds=time_module.perf_counter() - start_time,
        warnings=warnings,
        blobs={},
    )


def _create_joint_result_joint_mode(
    sector_results: list[SectorEvidence],
    hypotheses: list[str],
    *,
    margin_threshold: float,
    flip_rate_threshold: float,
    downweight_high_residual: bool,
    start_time: float,
) -> JointInferenceResult:
    """Create JointInferenceResult using Phase 3.2 joint likelihood mode.

    Uses true joint log-likelihood computation across all sectors,
    with quality-based sector weighting.
    """
    import time as time_module

    from .joint_likelihood import (
        compute_all_hypotheses_joint,
        compute_sector_weights,
        select_best_hypothesis_joint,
    )

    # Collect all hypothesis IDs from sector evidence
    all_hyp_ids: set[str] = set(hypotheses)
    for sector_ev in sector_results:
        for hyp in sector_ev.hypotheses:
            source_id = hyp.get("source_id")
            if source_id:
                all_hyp_ids.add(str(source_id))

    hyp_list = list(all_hyp_ids)

    # Compute sector weights based on quality
    if downweight_high_residual:
        sector_weights = compute_sector_weights(sector_results)
    else:
        sector_weights = {se.sector: se.quality_weight for se in sector_results}

    # Update SectorEvidence with computed weights (for consistency in output)
    for sector_ev in sector_results:
        sector_ev.quality_weight = sector_weights.get(sector_ev.sector, 1.0)

    # Compute joint log-likelihood for all hypotheses
    joint_loglikes = compute_all_hypotheses_joint(
        sector_results,
        hyp_list,
        sector_weights=sector_weights,
        downweight_high_residual=False,  # Already computed weights
    )

    # Select best hypothesis
    joint_best, verdict_from_ll, delta_ll = select_best_hypothesis_joint(
        joint_loglikes,
        margin_threshold=margin_threshold,
    )

    # Get joint log-likelihood for best hypothesis
    joint_ll = joint_loglikes.get(joint_best, 0.0)

    # Compute flip rate (sectors where best != joint_best)
    sector_best: list[str] = []
    for sector_ev in sector_results:
        if sector_ev.hypotheses:
            sector_best.append(sector_ev.hypotheses[0].get("source_id", ""))

    n_confident = len([s for s in sector_best if s])
    if n_confident > 0:
        n_flips = sum(1 for s in sector_best if s and s != joint_best)
        flip_rate = n_flips / n_confident
    else:
        flip_rate = 0.0

    # Determine consistency verdict
    if flip_rate >= flip_rate_threshold:
        consistency_verdict = "flipping"
    elif flip_rate >= FLIP_RATE_MIXED_THRESHOLD:
        consistency_verdict = "mixed"
    else:
        consistency_verdict = "stable"

    # Override verdict if consistency is flipping
    verdict = "AMBIGUOUS" if consistency_verdict == "flipping" else verdict_from_ll

    # Compute posterior odds
    posterior_odds: float | None = None
    if np.isfinite(delta_ll) and 0 < delta_ll < 100:
        posterior_odds = float(np.exp(delta_ll))

    # Collect warnings
    warnings: list[str] = []
    low_weight_sectors = [se.sector for se in sector_results if se.quality_weight < 0.5]
    if low_weight_sectors:
        warnings.append(f"Low quality weight sectors: {low_weight_sectors}")

    return JointInferenceResult(
        joint_best_source_id=joint_best,
        verdict=verdict,
        resolved_probability=None,  # Calibrated by Phase 3.5
        calibration_version=None,
        joint_log_likelihood=joint_ll,
        delta_log_likelihood=delta_ll if np.isfinite(delta_ll) else 0.0,
        posterior_odds=posterior_odds,
        sector_evidence=sector_results,
        sector_weights=sector_weights,
        flip_rate=flip_rate,
        consistency_verdict=consistency_verdict,
        hypotheses_considered=hypotheses,
        computation_time_seconds=time_module.perf_counter() - start_time,
        warnings=warnings,
        blobs={},
    )


def create_joint_result_from_sectors(
    sector_results: list[SectorEvidence],
    hypotheses: list[str],
    *,
    margin_threshold: float = MARGIN_RESOLVE_THRESHOLD,
    flip_rate_threshold: float = FLIP_RATE_UNSTABLE_THRESHOLD,
    inference_mode: InferenceMode = "vote",
    downweight_high_residual: bool = True,
) -> JointInferenceResult:
    """Create JointInferenceResult from per-sector evidence.

    This factory function aggregates per-sector hypothesis rankings into
    a joint decision. Supports two inference modes:

    - "vote" (default): Phase 2 weighted voting for backward compatibility
    - "joint": Phase 3.2 true joint log-likelihood optimization

    Parameters
    ----------
    sector_results : list[SectorEvidence]
        Per-sector evidence from localization analysis.
        Each entry contains hypothesis rankings and quality weights.
    hypotheses : list[str]
        List of source IDs being considered as host hypotheses.
    margin_threshold : float, optional
        Minimum margin to consider a result "resolved" vs "ambiguous".
        Default is MARGIN_RESOLVE_THRESHOLD (2.0).
    flip_rate_threshold : float, optional
        Flip rate above which consistency is "flipping".
        Default is FLIP_RATE_UNSTABLE_THRESHOLD (0.5).
    inference_mode : InferenceMode, optional
        Inference mode to use:
        - "vote": Phase 2 weighted voting (default, backward compatible)
        - "joint": Phase 3.2 joint log-likelihood
    downweight_high_residual : bool, optional
        Whether to downweight sectors with high residual RMS.
        Only used in "joint" mode. Default is True.

    Returns
    -------
    JointInferenceResult
        Joint inference result with decision based on selected inference mode.

    Notes
    -----
    Vote mode (Phase 2):
    - Weighted vote: sum quality_weight for each source's best-in-sector wins
    - Joint log-likelihood: sum of sector log-likelihoods (approximation)
    - Delta log-likelihood: margin between best and second-best

    Joint mode (Phase 3.2):
    - True joint log-likelihood optimization over all sectors
    - Automatic sector quality weighting based on residual diagnostics
    - Proper hypothesis comparison via log-likelihood difference

    Examples
    --------
    >>> # Use Phase 2 voting (default, backward compatible)
    >>> result = create_joint_result_from_sectors(sectors, hyps)

    >>> # Use Phase 3.2 joint likelihood
    >>> result = create_joint_result_from_sectors(
    ...     sectors, hyps, inference_mode="joint"
    ... )
    """
    import time

    start_time = time.perf_counter()

    # Edge case: no sectors
    if not sector_results:
        return JointInferenceResult(
            joint_best_source_id="",
            verdict="INVALID",
            resolved_probability=None,
            calibration_version=None,
            joint_log_likelihood=0.0,
            delta_log_likelihood=0.0,
            posterior_odds=None,
            sector_evidence=[],
            sector_weights={},
            flip_rate=0.0,
            consistency_verdict="stable",
            hypotheses_considered=hypotheses,
            computation_time_seconds=0.0,
            warnings=["No sector evidence provided"],
            blobs={},
        )

    # Edge case: single sector (same handling for both modes)
    if len(sector_results) == 1:
        single = sector_results[0]
        if not single.hypotheses:
            return JointInferenceResult(
                joint_best_source_id="",
                verdict="INVALID",
                resolved_probability=None,
                calibration_version=None,
                joint_log_likelihood=0.0,
                delta_log_likelihood=0.0,
                posterior_odds=None,
                sector_evidence=sector_results,
                sector_weights={single.sector: single.quality_weight},
                flip_rate=0.0,
                consistency_verdict="stable",
                hypotheses_considered=hypotheses,
                computation_time_seconds=time.perf_counter() - start_time,
                warnings=["Single sector with no hypotheses"],
                blobs={},
            )

        best_hyp = single.hypotheses[0]
        delta = single.hypotheses[1]["delta_loss"] if len(single.hypotheses) > 1 else float("inf")

        # Determine verdict
        is_target = "target" in best_hyp["source_id"].lower()
        if delta >= margin_threshold:
            verdict = "ON_TARGET" if is_target else "OFF_TARGET"
        else:
            verdict = "AMBIGUOUS"

        # Approximate joint log-likelihood from fit_loss (negative SSE -> log-likelihood)
        joint_ll = -best_hyp["fit_loss"] if np.isfinite(best_hyp["fit_loss"]) else 0.0

        return JointInferenceResult(
            joint_best_source_id=best_hyp["source_id"],
            verdict=verdict,
            resolved_probability=None,  # Calibrated by Phase 3.5
            calibration_version=None,
            joint_log_likelihood=joint_ll,
            delta_log_likelihood=delta if np.isfinite(delta) else 0.0,
            posterior_odds=float(np.exp(delta)) if np.isfinite(delta) and delta < 100 else None,
            sector_evidence=sector_results,
            sector_weights={single.sector: single.quality_weight},
            flip_rate=0.0,
            consistency_verdict="stable",
            hypotheses_considered=hypotheses,
            computation_time_seconds=time.perf_counter() - start_time,
            warnings=[],
            blobs={},
        )

    # Multi-sector case: dispatch based on inference mode
    if inference_mode == "joint":
        return _create_joint_result_joint_mode(
            sector_results,
            hypotheses,
            margin_threshold=margin_threshold,
            flip_rate_threshold=flip_rate_threshold,
            downweight_high_residual=downweight_high_residual,
            start_time=start_time,
        )
    else:
        # Default: vote mode (Phase 2 compatible)
        return _create_joint_result_vote_mode(
            sector_results,
            hypotheses,
            margin_threshold=margin_threshold,
            flip_rate_threshold=flip_rate_threshold,
            start_time=start_time,
        )
