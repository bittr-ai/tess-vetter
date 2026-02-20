"""PRF-based hypothesis scoring for pixel-level host identification.

This module provides hypothesis scoring using configurable PRF backends,
extending the PRF-lite approach with more accurate parametric models.

The key function is `score_hypotheses_with_prf`, which:
1. Evaluates PRF models at hypothesis positions using specified backend
2. Fits amplitude and optional background gradient to the difference image
3. Computes log-likelihood and fit quality metrics
4. Returns ranked hypotheses with extended diagnostics

CRITICAL: This module must remain pure compute:
- NO file I/O (open, Path, etc.)
- NO network access
- NO imports of astropy, lightkurve
- ONLY numpy and scipy dependencies
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .pixel_host_hypotheses import HypothesisScore
from .pixel_prf_lite import build_prf_model
from .prf_psf import get_prf_model
from .prf_schemas import PRFParams

__all__ = [
    "score_hypotheses_with_prf",
    "PRFBackend",
]

# Type alias for PRF backend selection
PRFBackend = Literal["prf_lite", "parametric", "instrument"]


def _compute_log_likelihood(
    residuals: NDArray[np.float64],
    variance: float | NDArray[np.float64],
) -> float:
    """Compute log-likelihood assuming Gaussian noise.

    Log-likelihood = -0.5 * sum((data - model)^2 / variance)

    Parameters
    ----------
    residuals : NDArray[np.float64]
        Residuals (data - model), 1D or 2D array.
    variance : float | NDArray[np.float64]
        Variance (can be scalar or per-pixel array).

    Returns
    -------
    float
        Log-likelihood value.
    """
    residuals_flat = residuals.ravel()
    if isinstance(variance, np.ndarray):
        variance_flat = variance.ravel()
        # Avoid division by zero
        variance_flat = np.where(variance_flat > 0, variance_flat, 1e-10)
        chi2 = np.sum(residuals_flat**2 / variance_flat)
    else:
        chi2 = np.sum(residuals_flat**2) / max(variance, 1e-10)

    return float(-0.5 * chi2)


def _fit_prf_with_background(
    diff_image: NDArray[np.float64],
    prf_model: NDArray[np.float64],
    *,
    fit_background: bool = True,
    valid_mask: NDArray[np.bool_] | None = None,
) -> tuple[float, tuple[float, float, float] | None, NDArray[np.float64]]:
    """Fit PRF amplitude and optional background gradient to difference image.

    Model: diff_image = amplitude * prf_model + b0 + bx*(col - c0) + by*(row - r0)

    Parameters
    ----------
    diff_image : NDArray[np.float64]
        2D difference image (out-of-transit - in-transit).
    prf_model : NDArray[np.float64]
        2D normalized PRF model (same shape as diff_image).
    fit_background : bool
        If True, fit a background gradient (b0, bx, by). Otherwise, only fit amplitude.
    valid_mask : NDArray[np.bool_] | None
        Optional mask indicating valid pixels (True = include).

    Returns
    -------
    tuple[float, tuple[float, float, float] | None, NDArray[np.float64]]
        - amplitude: fitted PRF amplitude
        - background: (b0, bx, by) tuple if fit_background=True, else None
        - residuals: residual array (diff_image - model)
    """
    shape = diff_image.shape
    n_rows, n_cols = shape

    # Flatten arrays
    y = diff_image.ravel()
    prf_flat = prf_model.ravel()

    # Create valid mask if not provided
    if valid_mask is None:  # noqa: SIM108
        valid_mask = np.isfinite(y)
    else:
        valid_mask = valid_mask.ravel() & np.isfinite(y)

    n_valid = int(np.sum(valid_mask))
    if n_valid < 3:
        # Not enough valid pixels
        return 0.0, None, diff_image.copy()

    # Build coordinate grids for background model
    row_grid, col_grid = np.mgrid[0:n_rows, 0:n_cols]
    center_row = (n_rows - 1) / 2.0
    center_col = (n_cols - 1) / 2.0
    row_rel = (row_grid - center_row).ravel()
    col_rel = (col_grid - center_col).ravel()

    if fit_background:
        # Design matrix: [prf, 1, col_rel, row_rel] for [amplitude, b0, bx, by]
        design = np.column_stack(
            [
                prf_flat[valid_mask],
                np.ones(n_valid),
                col_rel[valid_mask],
                row_rel[valid_mask],
            ]
        )
        y_valid = y[valid_mask]

        try:
            gram = design.T @ design
            rhs = design.T @ y_valid
            params = np.linalg.solve(gram, rhs)
            amplitude, b0, bx, by = params[0], params[1], params[2], params[3]
            background = (float(b0), float(bx), float(by))

            # Compute model and residuals
            model_flat = amplitude * prf_flat + b0 + bx * col_rel + by * row_rel
            residuals_flat = y - model_flat
            residuals = residuals_flat.reshape(shape)

        except np.linalg.LinAlgError:
            amplitude = 0.0
            background = (0.0, 0.0, 0.0)
            residuals = diff_image.copy()
    else:
        # Simpler fit: just amplitude and constant background
        design = np.column_stack([prf_flat[valid_mask], np.ones(n_valid)])
        y_valid = y[valid_mask]

        try:
            gram = design.T @ design
            rhs = design.T @ y_valid
            params = np.linalg.solve(gram, rhs)
            amplitude, b0 = params[0], params[1]
            background = None

            # Compute model and residuals
            model_flat = amplitude * prf_flat + b0
            residuals_flat = y - model_flat
            residuals = residuals_flat.reshape(shape)

        except np.linalg.LinAlgError:
            amplitude = 0.0
            background = None
            residuals = diff_image.copy()

    return float(amplitude), background, residuals


def score_hypotheses_with_prf(
    diff_image: NDArray[np.float64],
    hypotheses: list[dict[str, Any]],
    *,
    prf_backend: PRFBackend = "prf_lite",
    prf_params: PRFParams | None = None,
    fit_background: bool = True,
    fit_position: bool = False,  # Reserved for future position refinement
    seed: int | None = None,
    variance: float | NDArray[np.float64] | None = None,
) -> list[HypothesisScore]:
    """Score source hypotheses using specified PRF backend.

    For each hypothesis, evaluates a PRF model at the source position,
    fits amplitude (and optionally background gradient) to the difference
    image, and computes fit quality metrics including log-likelihood.

    Parameters
    ----------
    diff_image : NDArray[np.float64]
        2D difference image (out-of-transit minus in-transit). Shape (n_rows, n_cols).
        Positive values indicate flux decrease (transit signal).
    hypotheses : list[dict]
        List of source hypotheses to test. Each dict must contain:
        - 'source_id': str, unique identifier
        - 'source_name': str, human-readable name
        - 'row': float, row pixel coordinate of the source
        - 'col': float, column pixel coordinate of the source
    prf_backend : PRFBackend
        PRF backend to use:
        - 'prf_lite': Simple Gaussian PRF (fast, backward compatible)
        - 'parametric': Elliptical Gaussian with background gradient
        - 'instrument': Instrument-aware PRF (future, raises if not available)
    prf_params : PRFParams | None
        Optional PRF parameters for 'parametric' backend. If None, uses defaults.
    fit_background : bool
        If True, fit a background gradient (b0 + bx*x + by*y). Default True.
    fit_position : bool
        Reserved for future position refinement. Currently unused.
    seed : int | None
        Random seed for reproducibility. Currently used for future bootstrap.
    variance : float | NDArray[np.float64] | None
        Variance for log-likelihood calculation. If None, uses empirical estimate.

    Returns
    -------
    list[HypothesisScore]
        Ranked list of hypothesis scores (best first). Each entry contains:
        - source_id, source_name: identifiers from input
        - fit_loss: sum of squared residuals (lower is better)
        - delta_loss: difference from best (0 for best hypothesis)
        - rank: 1 for best, 2 for second best, etc.
        - fit_amplitude: fitted PSF amplitude
        - fit_background: fitted constant background (legacy field)
        - log_likelihood: log-likelihood of the fit (new)
        - fit_residual_rms: RMS of fit residuals (new)
        - fitted_background: (b0, bx, by) tuple if fit_background=True (new)
        - prf_backend: backend used for this scoring (new)

    Raises
    ------
    ValueError
        If diff_image is not 2D, hypotheses list is empty, or backend unavailable.

    Notes
    -----
    The parametric backend uses an elliptical Gaussian PSF model with optional
    rotation and background gradient, providing more accurate fits than prf_lite
    at the cost of additional computation.

    For prf_lite backend, this function is nearly equivalent to
    score_hypotheses_prf_lite but adds the extended output fields.

    Examples
    --------
    >>> diff_img = np.zeros((11, 11))
    >>> diff_img[5, 5] = 100  # Signal at center
    >>> hypotheses = [
    ...     {"source_id": "target", "source_name": "Target", "row": 5.0, "col": 5.0},
    ...     {"source_id": "neighbor", "source_name": "Neighbor", "row": 8.0, "col": 8.0},
    ... ]
    >>> scores = score_hypotheses_with_prf(diff_img, hypotheses, prf_backend="parametric")
    >>> scores[0]["source_id"]
    'target'
    >>> scores[0]["log_likelihood"] is not None
    True
    """
    # Set seed for reproducibility (for future bootstrap)
    if seed is not None:
        np.random.seed(seed)

    if diff_image.ndim != 2:
        raise ValueError(f"diff_image must be 2D, got shape {diff_image.shape}")
    if len(hypotheses) == 0:
        raise ValueError("hypotheses list cannot be empty")

    shape = diff_image.shape

    # Validate backend
    if prf_backend not in ("prf_lite", "parametric", "instrument"):
        raise ValueError(f"Unknown prf_backend: {prf_backend}")

    if prf_backend == "instrument":
        raise ValueError(
            "Instrument PRF backend not yet available. Use 'prf_lite' or 'parametric'."
        )

    # Estimate variance if not provided (using median absolute deviation)
    variance_val: float | NDArray[np.float64]
    if variance is None:
        valid_pixels = diff_image[np.isfinite(diff_image)]
        if len(valid_pixels) > 0:
            mad = float(np.median(np.abs(valid_pixels - np.median(valid_pixels))))
            estimated_var = (1.4826 * mad) ** 2  # Robust variance estimate
            variance_val = estimated_var if estimated_var >= 1e-10 else 1.0
        else:
            variance_val = 1.0
    else:
        variance_val = variance

    # Create valid mask
    valid_mask = np.isfinite(diff_image)
    n_valid = int(np.sum(valid_mask))

    if n_valid < 3:
        # Not enough valid pixels for fitting
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
                    log_likelihood=None,
                    fit_residual_rms=None,
                    fitted_background=None,
                    prf_backend=prf_backend,
                )
            )
        return results

    # Get PRF model based on backend
    if prf_backend == "parametric":
        if prf_params is None:
            prf_params = PRFParams()  # Default isotropic Gaussian
        prf_model_obj = get_prf_model("parametric", params=prf_params)
    else:
        # prf_lite uses simple Gaussian
        prf_model_obj = None

    # Score each hypothesis
    scores_raw: list[
        tuple[
            float,  # loss (SSE)
            float,  # amplitude
            tuple[float, float, float] | None,  # background gradient
            float,  # log_likelihood
            float,  # residual_rms
            dict[str, Any],  # hypothesis
        ]
    ] = []

    for hyp in hypotheses:
        row = float(hyp["row"])
        col = float(hyp["col"])

        # Build PRF model for this hypothesis
        if prf_backend == "parametric" and prf_model_obj is not None:
            # Use parametric PSF model (normalized, no background in model itself)
            prf = prf_model_obj.evaluate(row, col, shape, normalize=True)
        else:
            # Use prf_lite (simple Gaussian)
            sigma = prf_params.effective_sigma if prf_params else 1.5
            prf = build_prf_model(row, col, shape, sigma=sigma)

        # Fit amplitude and background
        amplitude, bg_gradient, residuals = _fit_prf_with_background(
            diff_image,
            prf,
            fit_background=fit_background,
            valid_mask=valid_mask,
        )

        # Compute loss (SSE) and log-likelihood
        residuals_valid = residuals[valid_mask]
        loss = float(np.sum(residuals_valid**2))
        log_ll = _compute_log_likelihood(residuals_valid, variance_val)
        residual_rms = float(np.sqrt(np.mean(residuals_valid**2)))

        scores_raw.append((loss, amplitude, bg_gradient, log_ll, residual_rms, hyp))

    # Sort by loss (ascending) - lower loss is better
    scores_raw.sort(key=lambda x: x[0])
    best_loss = scores_raw[0][0]

    # Build output with ranks and delta_loss
    output_results: list[HypothesisScore] = []
    for rank, (loss, amp, bg_grad, log_ll, rms, hyp) in enumerate(scores_raw, start=1):
        delta = loss - best_loss if np.isfinite(loss) and np.isfinite(best_loss) else 0.0

        # Legacy fit_background field (constant only)
        legacy_bg = bg_grad[0] if bg_grad is not None else None

        output_results.append(
            HypothesisScore(
                source_id=str(hyp.get("source_id", f"hyp_{rank}")),
                source_name=str(hyp.get("source_name", f"Hypothesis {rank}")),
                fit_loss=loss,
                delta_loss=delta,
                rank=rank,
                fit_amplitude=float(amp) if np.isfinite(amp) else None,
                fit_background=float(legacy_bg)
                if legacy_bg is not None and np.isfinite(legacy_bg)
                else None,
                log_likelihood=float(log_ll) if np.isfinite(log_ll) else None,
                fit_residual_rms=float(rms) if np.isfinite(rms) else None,
                fitted_background=bg_grad if bg_grad is not None else None,
                prf_backend=prf_backend,
            )
        )

    return output_results
