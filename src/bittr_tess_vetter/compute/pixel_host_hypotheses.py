"""Core algorithms for pixel-level host hypothesis testing.

This module provides quantitative methods to identify which source is
the host of a transit signal in TESS data. It implements:

1. PRF-lite hypothesis scoring: Fit a Gaussian PRF model to difference images
   and rank source hypotheses by fit quality.

2. Multi-sector aggregation: Combine per-sector results to compute consensus
   and detect inconsistencies that may indicate blends or systematics.

3. Aperture hypothesis fitting: Use depth-vs-aperture curves to identify
   the true host by predicting observed depths from PRF weights.

CRITICAL: This module must remain pure compute:
- NO file I/O (open, Path, etc.)
- NO network access
- NO imports of astropy, lightkurve
- ONLY numpy and scipy dependencies

These functions are designed to work with pre-validated numpy arrays.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from .pixel_prf_lite import build_prf_model

__all__ = [
    "score_hypotheses_prf_lite",
    "aggregate_multi_sector",
    "fit_aperture_hypothesis",
    "HypothesisScore",
    "MultiSectorConsensus",
    "ApertureHypothesisFit",
    # Thresholds
    "MARGIN_RESOLVE_THRESHOLD",
    "FLIP_RATE_MIXED_THRESHOLD",
    "FLIP_RATE_UNSTABLE_THRESHOLD",
]

# ==============================================================================
# Default Thresholds
# ==============================================================================

MARGIN_RESOLVE_THRESHOLD = 2.0
"""Minimum delta_loss margin to prefer one host over another.

If the margin (best vs runner-up) is below this threshold, the result
is considered 'ambiguous' and no host is preferred.
"""

FLIP_RATE_MIXED_THRESHOLD = 0.3
"""Flip rate above which multi-sector consistency is 'mixed'.

Flip rate = fraction of sectors where best_source_id != consensus_best_source_id.
"""

FLIP_RATE_UNSTABLE_THRESHOLD = 0.5
"""Flip rate above which multi-sector consistency is 'flipping'.

This indicates the host preference is unstable across sectors.
"""


# ==============================================================================
# Type Definitions
# ==============================================================================


class HypothesisScore(TypedDict, total=False):
    """Per-hypothesis scoring result from PRF-based fitting.

    Core fields (always present in all backends):
    - source_id, source_name: Identifiers for the source hypothesis
    - fit_loss, delta_loss, rank: Fit quality metrics
    - fit_amplitude, fit_background: Fitted parameters

    Extended fields (present when using parametric PRF backend):
    - log_likelihood: Log-likelihood of the fit
    - fit_residual_rms: RMS of fit residuals
    - fitted_background: (b0, bx, by) gradient tuple
    - prf_backend: Backend used for scoring

    Position fields (present when position is tracked):
    - row, col: Pixel coordinates of the source
    - source_row, source_col: Alias for row/col (for backward compat)
    - source_ra_deg, source_dec_deg: Sky coordinates
    - distance_from_centroid: Distance from difference image centroid
    - prf_weight: PRF weight at this position
    """

    # Core fields (always present)
    source_id: str
    source_name: str
    fit_loss: float  # SSE-like, lower is better
    delta_loss: float  # fit_loss - best_fit_loss (0 for best)
    rank: int  # 1 for best
    fit_amplitude: float | None
    fit_background: float | None

    # Extended fields for PRF-based scoring
    log_likelihood: float | None
    fit_residual_rms: float | None
    fitted_background: tuple[float, float, float] | None
    prf_backend: str | None

    # Position fields
    row: float
    col: float
    source_row: float
    source_col: float
    source_ra_deg: float | None
    source_dec_deg: float | None
    distance_from_centroid: float | None
    prf_weight: float | None


class MultiSectorConsensus(TypedDict):
    """Result of aggregating per-sector hypothesis rankings."""

    consensus_best_source_id: str | None
    consensus_margin: float | None
    disagreement_flag: str  # "stable" | "mixed" | "flipping"
    flip_rate: float
    n_sectors_total: int
    n_sectors_supporting_best: int


class ApertureHypothesisFit(TypedDict):
    """Result of aperture-based host hypothesis testing."""

    best_source_id: str | None
    margin: float | None
    host_ambiguity: str  # "resolved" | "ambiguous" | "conflicted"
    hypothesis_fits: list[dict[str, Any]]  # Per-hypothesis fit details


# ==============================================================================
# Core Algorithms
# ==============================================================================


def score_hypotheses_prf_lite(
    diff_image: NDArray[np.float64],
    hypotheses: list[dict[str, Any]],
    *,
    sigma: float = 1.5,
    seed: int | None = None,
) -> list[HypothesisScore]:
    """Score source hypotheses by fitting PRF-lite models to a difference image.

    For each hypothesis, fits a scaled Gaussian PRF model plus background offset
    to the difference image using least squares. Lower fit_loss indicates a
    better match (the source is more likely the transit host).

    Parameters
    ----------
    diff_image : NDArray[np.float64]
        2D difference image (out-of-transit minus in-transit). Shape (n_rows, n_cols).
        Positive values indicate flux decrease (transit signal).
    hypotheses : list[dict]
        List of source hypotheses to test. Each dict must contain:
        - 'source_id': str, unique identifier (e.g., "gaia_dr3:12345" or "target")
        - 'source_name': str, human-readable name
        - 'row': float, row pixel coordinate of the source
        - 'col': float, column pixel coordinate of the source
    sigma : float, optional
        Gaussian sigma for PRF model in pixels. Default is 1.5.
    seed : int | None, optional
        Random seed for reproducibility. Currently unused but reserved for
        future bootstrap uncertainty estimation.

    Returns
    -------
    list[HypothesisScore]
        Ranked list of hypothesis scores (best first). Each entry contains:
        - source_id, source_name: identifiers from input
        - fit_loss: sum of squared residuals (lower is better)
        - delta_loss: difference from best (0 for best hypothesis)
        - rank: 1 for best, 2 for second best, etc.
        - fit_amplitude: fitted PSF amplitude
        - fit_background: fitted background offset

    Raises
    ------
    ValueError
        If diff_image is not 2D or hypotheses list is empty.

    Notes
    -----
    The fitting model is:

        model(x) = amplitude * PRF(x | center=hypothesis_location) + background

    where PRF is a normalized 2D Gaussian. We solve for amplitude and background
    via linear least squares:

        minimize sum((diff_image - amplitude * PRF - background)^2)

    This has a closed-form solution using the normal equations.

    Examples
    --------
    >>> diff_img = np.zeros((11, 11))
    >>> diff_img[5, 5] = 100  # Strong signal at center
    >>> hypotheses = [
    ...     {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
    ...     {"source_id": "neighbor", "source_name": "Neighbor", "row": 8.0, "col": 8.0},
    ... ]
    >>> scores = score_hypotheses_prf_lite(diff_img, hypotheses)
    >>> scores[0]["source_id"]  # Best hypothesis
    'target'
    """
    # Set seed for reproducibility (for future bootstrap)
    if seed is not None:
        np.random.seed(seed)

    if diff_image.ndim != 2:
        raise ValueError(f"diff_image must be 2D, got shape {diff_image.shape}")
    if len(hypotheses) == 0:
        raise ValueError("hypotheses list cannot be empty")

    shape = diff_image.shape

    # Flatten difference image for linear algebra
    y = diff_image.ravel()

    # Handle NaN/inf values by masking
    valid_mask = np.isfinite(y)
    n_valid = np.sum(valid_mask)

    if n_valid < 3:
        # Not enough valid pixels for fitting
        # Return all hypotheses with infinite loss
        results: list[HypothesisScore] = []
        for i, hyp in enumerate(hypotheses):
            results.append(
                HypothesisScore(
                    source_id=str(hyp.get("source_id", f"hyp_{i}")),
                    source_name=str(hyp.get("source_name", f"Hypothesis {i}")),
                    fit_loss=float("inf"),
                    delta_loss=0.0,
                    rank=1,
                    fit_amplitude=None,
                    fit_background=None,
                )
            )
        return results

    # Score each hypothesis
    scores_raw: list[tuple[float, float, float, dict[str, Any]]] = []  # (loss, amp, bg, hyp)

    for hyp in hypotheses:
        row = float(hyp["row"])
        col = float(hyp["col"])

        # Build PRF model for this hypothesis
        prf = build_prf_model(row, col, shape, sigma=sigma)
        prf_flat = prf.ravel()

        # Design matrix: [prf, ones] for [amplitude, background]
        # Only use valid pixels
        design = np.column_stack([prf_flat[valid_mask], np.ones(n_valid)])
        y_valid = y[valid_mask]

        # Solve least squares: minimize ||y - design @ [amp, bg]||^2
        # Normal equations: (design^T design) @ [amp, bg] = design^T @ y
        try:
            gram = design.T @ design
            rhs = design.T @ y_valid
            params = np.linalg.solve(gram, rhs)
            amplitude, background = params[0], params[1]

            # Compute residuals and loss (SSE)
            predicted = amplitude * prf_flat[valid_mask] + background
            residuals = y_valid - predicted
            loss = float(np.sum(residuals**2))
        except np.linalg.LinAlgError:
            # Singular matrix - set high loss
            amplitude, background = 0.0, 0.0
            loss = float("inf")

        scores_raw.append((loss, amplitude, background, hyp))

    # Sort by loss (ascending)
    scores_raw.sort(key=lambda x: x[0])
    best_loss = scores_raw[0][0]

    # Build output with ranks and delta_loss
    results = []
    for rank, (loss, amp, bg, hyp) in enumerate(scores_raw, start=1):
        delta = loss - best_loss if np.isfinite(loss) and np.isfinite(best_loss) else 0.0
        results.append(
            HypothesisScore(
                source_id=str(hyp.get("source_id", f"hyp_{rank}")),
                source_name=str(hyp.get("source_name", f"Hypothesis {rank}")),
                fit_loss=loss,
                delta_loss=delta,
                rank=rank,
                fit_amplitude=float(amp) if np.isfinite(amp) else None,
                fit_background=float(bg) if np.isfinite(bg) else None,
            )
        )

    return results


def aggregate_multi_sector(
    per_sector_results: list[dict[str, Any]],
    *,
    margin_threshold: float = MARGIN_RESOLVE_THRESHOLD,
    flip_rate_mixed: float = FLIP_RATE_MIXED_THRESHOLD,
    flip_rate_unstable: float = FLIP_RATE_UNSTABLE_THRESHOLD,
) -> MultiSectorConsensus:
    """Aggregate per-sector hypothesis rankings into a multi-sector consensus.

    Computes which source is most consistently identified as the host across
    sectors and detects inconsistencies that may indicate blends or systematics.

    Parameters
    ----------
    per_sector_results : list[dict]
        List of per-sector results. Each dict should contain:
        - 'sector': int, sector number
        - 'best_source_id': str | None, best hypothesis for this sector
        - 'margin': float | None, delta_loss between best and runner-up
        - 'hypotheses_ranked': list[HypothesisScore], full ranking (optional)
        - 'status': str, "ok" | "invalid" | "skipped" (optional, default "ok")
    margin_threshold : float, optional
        Minimum margin to consider a sector's preference "confident".
        Default is MARGIN_RESOLVE_THRESHOLD (2.0).
    flip_rate_mixed : float, optional
        Flip rate above which disagreement_flag becomes "mixed".
        Default is FLIP_RATE_MIXED_THRESHOLD (0.3).
    flip_rate_unstable : float, optional
        Flip rate above which disagreement_flag becomes "flipping".
        Default is FLIP_RATE_UNSTABLE_THRESHOLD (0.5).

    Returns
    -------
    MultiSectorConsensus
        Aggregated consensus containing:
        - consensus_best_source_id: most consistently preferred source
        - consensus_margin: combined evidence margin
        - disagreement_flag: "stable" | "mixed" | "flipping"
        - flip_rate: fraction of sectors disagreeing with consensus
        - n_sectors_total: total number of sectors provided
        - n_sectors_supporting_best: sectors agreeing with consensus

    Notes
    -----
    The consensus is computed as a simple vote over per-sector `best_source_id`,
    with ties broken by the total `margin` summed over sectors where that source
    won. This avoids deep dependencies on per-sector full rankings while still
    rewarding sectors with higher evidence margins.

    A sector "flips" if its best_source_id differs from the consensus.
    The flip_rate is computed only over sectors with confident preferences
    (margin >= margin_threshold).

    Examples
    --------
    >>> results = [
    ...     {"sector": 1, "best_source_id": "target", "margin": 5.0},
    ...     {"sector": 2, "best_source_id": "target", "margin": 3.0},
    ...     {"sector": 3, "best_source_id": "target", "margin": 4.0},
    ... ]
    >>> consensus = aggregate_multi_sector(results)
    >>> consensus["disagreement_flag"]
    'stable'
    >>> consensus["flip_rate"]
    0.0
    """
    n_sectors_total = len(per_sector_results)

    if n_sectors_total == 0:
        return MultiSectorConsensus(
            consensus_best_source_id=None,
            consensus_margin=None,
            disagreement_flag="stable",
            flip_rate=0.0,
            n_sectors_total=0,
            n_sectors_supporting_best=0,
        )

    # Filter to valid sectors only
    valid_sectors = [
        r for r in per_sector_results if r.get("status", "ok") == "ok" and r.get("best_source_id")
    ]

    if len(valid_sectors) == 0:
        return MultiSectorConsensus(
            consensus_best_source_id=None,
            consensus_margin=None,
            disagreement_flag="stable",
            flip_rate=0.0,
            n_sectors_total=n_sectors_total,
            n_sectors_supporting_best=0,
        )

    # Collect all unique source_ids and sum their delta_loss across sectors
    # For the best source in each sector, delta_loss = 0
    # For others, delta_loss > 0
    # Summing gives total "penalty" - lowest total is the consensus best

    # Method 1: Simple vote by best_source_id
    vote_counts: dict[str, int] = {}
    margin_sums: dict[str, float] = {}

    for r in valid_sectors:
        source_id = r["best_source_id"]
        margin = r.get("margin", 0.0) or 0.0

        vote_counts[source_id] = vote_counts.get(source_id, 0) + 1
        margin_sums[source_id] = margin_sums.get(source_id, 0.0) + margin

    # Consensus is the source with most votes, tie-break by total margin
    sorted_sources = sorted(
        vote_counts.keys(),
        key=lambda s: (vote_counts[s], margin_sums.get(s, 0)),
        reverse=True,
    )

    consensus_best = sorted_sources[0]
    n_supporting = vote_counts[consensus_best]

    # Compute consensus margin as sum of margins when this source won
    consensus_margin_val = 0.0
    for r in valid_sectors:
        if r["best_source_id"] == consensus_best:
            consensus_margin_val += r.get("margin", 0.0) or 0.0

    # Count "flips" - sectors where best != consensus AND margin is confident
    confident_sectors = [r for r in valid_sectors if (r.get("margin") or 0.0) >= margin_threshold]

    if len(confident_sectors) == 0:
        # No confident sectors - cannot compute flip rate meaningfully
        flip_rate = 0.0
    else:
        flips = sum(1 for r in confident_sectors if r["best_source_id"] != consensus_best)
        flip_rate = flips / len(confident_sectors)

    # Determine disagreement flag
    if flip_rate >= flip_rate_unstable:
        disagreement_flag = "flipping"
    elif flip_rate >= flip_rate_mixed:
        disagreement_flag = "mixed"
    else:
        disagreement_flag = "stable"

    return MultiSectorConsensus(
        consensus_best_source_id=consensus_best,
        consensus_margin=consensus_margin_val if consensus_margin_val > 0 else None,
        disagreement_flag=disagreement_flag,
        flip_rate=flip_rate,
        n_sectors_total=n_sectors_total,
        n_sectors_supporting_best=n_supporting,
    )


def fit_aperture_hypothesis(
    depths_by_aperture: list[dict[str, Any]],
    prf_weights_by_hypothesis: dict[str, list[float]],
    *,
    margin_threshold: float = MARGIN_RESOLVE_THRESHOLD,
) -> ApertureHypothesisFit:
    """Fit aperture depth curves to identify the transit host.

    For each hypothesis, predicts the observed depth at each aperture size
    based on PRF weights and fits a "true depth" parameter. The hypothesis
    whose predicted curve best matches the observed depths is preferred.

    Parameters
    ----------
    depths_by_aperture : list[dict]
        Observed depths at different apertures. Each dict should contain:
        - 'aperture_id': str, aperture identifier (e.g., "spoc", "r+1")
        - 'depth_ppm': float, observed depth in ppm
        - 'depth_ppm_err': float | None, uncertainty (optional)
    prf_weights_by_hypothesis : dict[str, list[float]]
        For each hypothesis source_id, a list of PRF weights corresponding
        to each aperture in depths_by_aperture (same order). Each weight
        is the fraction of the source's flux captured by that aperture.
    margin_threshold : float, optional
        Minimum RMSE margin to consider host "resolved".
        Default is MARGIN_RESOLVE_THRESHOLD (2.0).

    Returns
    -------
    ApertureHypothesisFit
        Result containing:
        - best_source_id: hypothesis with best fit, or None if ambiguous
        - margin: RMSE difference between best and runner-up
        - host_ambiguity: "resolved" | "ambiguous" | "conflicted"
        - hypothesis_fits: per-hypothesis fit details

    Notes
    -----
    For each hypothesis h, we model:

        depth_observed[a] = depth_true[h] * prf_weight[h][a]

    where a indexes apertures. We fit depth_true[h] via weighted least squares:

        depth_true[h] = sum(w * d_obs) / sum(w * prf_weight)

    where w incorporates both PRF weight and measurement uncertainty.

    The fit quality is measured by RMSE of (observed - predicted) depths.

    Examples
    --------
    >>> depths = [
    ...     {"aperture_id": "spoc", "depth_ppm": 1000.0},
    ...     {"aperture_id": "r+1", "depth_ppm": 800.0},
    ...     {"aperture_id": "r+2", "depth_ppm": 600.0},
    ... ]
    >>> weights = {
    ...     "target": [1.0, 0.8, 0.6],  # PRF centered on target
    ...     "neighbor": [0.2, 0.3, 0.4],  # PRF off-center
    ... }
    >>> result = fit_aperture_hypothesis(depths, weights)
    >>> result["best_source_id"]
    'target'
    """
    if len(depths_by_aperture) == 0:
        return ApertureHypothesisFit(
            best_source_id=None,
            margin=None,
            host_ambiguity="ambiguous",
            hypothesis_fits=[],
        )

    if len(prf_weights_by_hypothesis) == 0:
        return ApertureHypothesisFit(
            best_source_id=None,
            margin=None,
            host_ambiguity="ambiguous",
            hypothesis_fits=[],
        )

    # Extract observed depths and uncertainties (use 1.0 if not provided)
    observed_depths_ppm = np.array([d["depth_ppm"] for d in depths_by_aperture], dtype=np.float64)
    depth_uncertainties_ppm = np.array(
        [d.get("depth_ppm_err") or 1.0 for d in depths_by_aperture], dtype=np.float64
    )

    # Optional: if baseline flux is provided, fit in flux-drop space to account for dilution:
    #
    #   depth_ppm[a] = 1e6 * (depth_true_frac * host_flux_total * w[a]) / baseline_flux[a]
    #
    # Let amp = depth_true_frac * host_flux_total (units: flux). Then:
    #   delta_flux[a] = baseline_flux[a] * depth_ppm[a] / 1e6 = amp * w[a]
    #
    # This makes the hypothesis fit robust to aperture-dependent dilution.
    baseline_flux = np.array(
        [d.get("baseline_flux") for d in depths_by_aperture],
        dtype=np.float64,
    )
    use_baseline = (
        np.all(np.isfinite(baseline_flux))
        and np.all(baseline_flux > 0)
        and baseline_flux.shape == observed_depths_ppm.shape
    )

    if use_baseline:
        observed_delta = baseline_flux * observed_depths_ppm / 1_000_000.0
        uncertainties_delta = baseline_flux * depth_uncertainties_ppm / 1_000_000.0
        inv_var = 1.0 / (uncertainties_delta**2 + 1e-10)
        baseline_median = float(np.nanmedian(baseline_flux))
    else:
        inv_var = 1.0 / (depth_uncertainties_ppm**2 + 1e-10)

    # Fit each hypothesis
    fit_results: list[
        tuple[float, float, str, str | None]
    ] = []  # (rmse, depth_hat, source_id, name)

    for source_id, prf_weights in prf_weights_by_hypothesis.items():
        if len(prf_weights) != len(depths_by_aperture):
            # Mismatched lengths - skip this hypothesis
            fit_results.append((float("inf"), 0.0, source_id, None))
            continue

        weights_arr = np.array(prf_weights, dtype=np.float64)

        # Avoid division by zero
        denom = np.sum(inv_var * weights_arr**2)
        if denom < 1e-10:
            fit_results.append((float("inf"), 0.0, source_id, None))
            continue

        if use_baseline:
            # Fit in flux-drop space: delta_flux ≈ amp * w
            amp_hat = np.sum(inv_var * observed_delta * weights_arr) / denom
            predicted_delta = amp_hat * weights_arr
            # Convert residuals back to ppm for interpretability
            residuals_ppm = (observed_delta - predicted_delta) / baseline_flux * 1_000_000.0
            rmse = float(np.sqrt(np.mean(residuals_ppm**2)))
            depth_true_hat = (
                (amp_hat / baseline_median) * 1_000_000.0 if baseline_median > 0 else float("nan")
            )
        else:
            # Legacy fit: depth_ppm ≈ depth_true_ppm * w
            depth_true_hat = np.sum(inv_var * observed_depths_ppm * weights_arr) / denom
            predicted = depth_true_hat * weights_arr
            residuals = observed_depths_ppm - predicted
            rmse = float(np.sqrt(np.mean(residuals**2)))

        fit_results.append((rmse, depth_true_hat, source_id, None))

    # Sort by RMSE (ascending)
    fit_results.sort(key=lambda x: x[0])

    if len(fit_results) == 0:
        return ApertureHypothesisFit(
            best_source_id=None,
            margin=None,
            host_ambiguity="ambiguous",
            hypothesis_fits=[],
        )

    best_rmse = fit_results[0][0]
    runner_up_rmse = fit_results[1][0] if len(fit_results) > 1 else float("inf")
    margin_val = runner_up_rmse - best_rmse

    # Build hypothesis_fits list
    hypothesis_fits = []
    for rank, (rmse, depth_hat, source_id, _) in enumerate(fit_results, start=1):
        delta = rmse - best_rmse if np.isfinite(rmse) and np.isfinite(best_rmse) else 0.0
        hypothesis_fits.append(
            {
                "source_id": source_id,
                "source_name": source_id,  # Can be enriched by caller
                "depth_true_ppm_hat": float(depth_hat) if np.isfinite(depth_hat) else None,
                "fit_rmse_ppm": float(rmse) if np.isfinite(rmse) else None,
                "delta_fit_rmse_ppm": float(delta),
                "rank": rank,
            }
        )

    # Determine ambiguity
    if not np.isfinite(best_rmse):
        host_ambiguity = "conflicted"
        best_source = None
    elif margin_val < margin_threshold:
        host_ambiguity = "ambiguous"
        best_source = None  # Don't claim a host if margin is too small
    else:
        host_ambiguity = "resolved"
        best_source = fit_results[0][2]

    return ApertureHypothesisFit(
        best_source_id=best_source,
        margin=margin_val if np.isfinite(margin_val) else None,
        host_ambiguity=host_ambiguity,
        hypothesis_fits=hypothesis_fits,
    )
