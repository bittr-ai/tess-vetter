"""Differentiable transit scoring primitives using MLX (Apple Silicon).

This module is intentionally *analysis-first*: it does not attempt to replace TLS.
Instead it provides smooth, differentiable scoring functions for:
- detectability thresholding (depth vs score)
- attribution (gradients w.r.t. flux, integrated gradients)

MLX is an optional dependency. Import errors are raised with an actionable message.

Scientific lineage:
    - The box-like template + least-squares depth fit is in the BLS / matched-filter family
      (Kovács, Zucker & Mazeh 2002).
    - Integrated Gradients attribution is from Sundararajan, Taly & Yan 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "MLX is not installed. Install it (Apple Silicon only), e.g.: `pip install mlx`."
        ) from e
    return mx


@dataclass(frozen=True)
class MlxTopKScoreResult:
    """Result of scoring a small set of candidate periods."""

    top_k_periods: Any  # mx.array
    scores: Any  # mx.array shape (k,)
    weights: Any  # mx.array shape (k,)


@dataclass(frozen=True)
class MlxT0RefinementResult:
    """Result of local t0 refinement for a fixed-period MLX score."""

    t0_best_btjd: float
    score_best: float
    score_at_input: float
    delta_score: float
    t0_grid_btjd: np.ndarray
    scores: np.ndarray


def _template_batched(
    *,
    mx: Any,
    time: Any,  # (N,)
    period_days: float,
    t0s_btjd: Any,  # (K,)
    duration_hours: float,
    ingress_egress_fraction: float,
    sharpness: float,
) -> Any:  # (K, N)
    period = mx.array(float(period_days))
    duration_days = mx.array(float(duration_hours) / 24.0)

    half_duration = duration_days / 2.0
    ingress = mx.maximum(duration_days * mx.array(float(ingress_egress_fraction)), mx.array(1e-6))
    k = mx.array(float(sharpness)) / ingress

    phase = (time[None, :] - t0s_btjd[:, None]) / period
    phase = phase - mx.floor(phase + mx.array(0.5))
    dt = mx.abs(phase * period)
    tmpl = mx.sigmoid(k * (half_duration - dt))
    return mx.clip(tmpl, 0.0, 1.0)


def _batched_scores(
    *,
    mx: Any,
    time: Any,
    flux: Any,
    flux_err: Any | None,
    period_days: float,
    t0s_btjd: Any,
    duration_hours: float,
    ingress_egress_fraction: float,
    sharpness: float,
    eps: float,
) -> Any:
    y = mx.array(1.0) - flux

    if flux_err is None:
        w = mx.ones_like(y)
    else:
        w = mx.array(1.0) / mx.maximum(flux_err * flux_err, mx.array(float(eps)))

    tmpl = _template_batched(
        mx=mx,
        time=time,
        period_days=period_days,
        t0s_btjd=t0s_btjd,
        duration_hours=duration_hours,
        ingress_egress_fraction=ingress_egress_fraction,
        sharpness=sharpness,
    )

    denom = mx.sum(w[None, :] * tmpl * tmpl, axis=1) + mx.array(float(eps))
    depth_hat = mx.sum(w[None, :] * tmpl * y[None, :], axis=1) / denom
    depth_sigma = mx.sqrt(mx.array(1.0) / denom)
    score = depth_hat / mx.maximum(depth_sigma, mx.array(float(eps)))
    return score, depth_hat, depth_sigma


def smooth_box_template(
    *,
    time: Any,  # mx.array
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    ingress_egress_fraction: float = 0.2,
    sharpness: float = 30.0,
) -> Any:
    """Create a smooth (differentiable) transit template in [0, 1].

    The template is a smoothed box (sigmoid edges), centered on the transit.
    It avoids hard discontinuities at ingress/egress.
    """
    mx = _require_mlx()

    period = mx.array(period_days)
    t0 = mx.array(t0_btjd)
    duration_days = mx.array(duration_hours / 24.0)

    # Phase in [-0.5, 0.5)
    phase = (time - t0) / period
    phase = phase - mx.floor(phase + 0.5)
    dt = mx.abs(phase * period)  # time offset from transit center, in days

    # Smoothed "inside transit" indicator.
    #
    # We treat `duration_hours` as the full (first-to-fourth contact) duration.
    # A sharp box would be: inside = 1 if dt <= duration/2 else 0.
    # We smooth that edge over a characteristic timescale set by
    # `ingress_egress_fraction * duration`, controlled by `sharpness`.
    half_duration = duration_days / 2.0
    ingress = mx.maximum(duration_days * mx.array(ingress_egress_fraction), mx.array(1e-6))
    k = mx.array(sharpness) / ingress
    template = mx.sigmoid(k * (half_duration - dt))
    return mx.clip(template, 0.0, 1.0)


def score_fixed_period(
    *,
    time: Any,  # mx.array
    flux: Any,  # mx.array (normalized, ~1)
    flux_err: Any | None,  # mx.array or None
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    ingress_egress_fraction: float = 0.2,
    sharpness: float = 30.0,
    eps: float = 1e-12,
) -> Any:
    """Differentiable matched-filter style detection score at a fixed period.

    Model:
        flux ~ 1 - depth * template

    We estimate depth by weighted least squares against the template, then return
    a z-score-like statistic depth_hat / depth_sigma.
    """
    mx = _require_mlx()

    template = smooth_box_template(
        time=time,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        ingress_egress_fraction=ingress_egress_fraction,
        sharpness=sharpness,
    )

    y = mx.array(1.0) - flux

    if flux_err is None:
        w = mx.ones_like(y)
    else:
        sigma2 = mx.maximum(flux_err * flux_err, mx.array(eps))
        w = mx.array(1.0) / sigma2

    denom = mx.sum(w * template * template) + mx.array(eps)
    depth_hat = mx.sum(w * template * y) / denom
    depth_sigma = mx.sqrt(mx.array(1.0) / denom)
    score = depth_hat / mx.maximum(depth_sigma, mx.array(eps))
    return score


def score_top_k_periods(
    *,
    time: Any,  # mx.array
    flux: Any,  # mx.array
    flux_err: Any | None,
    periods_days_top_k: Any,  # mx.array (k,)
    t0_btjd: float,
    duration_hours: float,
    temperature: float = 0.5,
    ingress_egress_fraction: float = 0.2,
    sharpness: float = 30.0,
) -> MlxTopKScoreResult:
    """Score a small candidate period set and compute deterministic soft weights."""
    mx = _require_mlx()

    # Convert scalar parameters to MLX arrays once, outside vmap
    t0_mx = mx.array(t0_btjd)
    duration_days_mx = mx.array(duration_hours / 24.0)
    half_duration = duration_days_mx / 2.0
    ingress = mx.maximum(duration_days_mx * mx.array(ingress_egress_fraction), mx.array(1e-6))
    k_sharpness = mx.array(sharpness) / ingress

    # Precompute weights for flux_err
    y = mx.array(1.0) - flux
    if flux_err is None:
        w = mx.ones_like(y)
    else:
        sigma2 = mx.maximum(flux_err * flux_err, mx.array(1e-12))
        w = mx.array(1.0) / sigma2

    def _score_one_pure_mlx(period_days_mx: Any) -> Any:
        """Score at a single period using pure MLX operations (vmap-safe)."""
        # Phase in [-0.5, 0.5)
        phase = (time - t0_mx) / period_days_mx
        phase = phase - mx.floor(phase + mx.array(0.5))
        dt = mx.abs(phase * period_days_mx)

        # Smoothed template
        template = mx.sigmoid(k_sharpness * (half_duration - dt))
        template = mx.clip(template, 0.0, 1.0)

        # Matched-filter score
        eps = mx.array(1e-12)
        denom = mx.sum(w * template * template) + eps
        depth_hat = mx.sum(w * template * y) / denom
        depth_sigma = mx.sqrt(mx.array(1.0) / denom)
        score = depth_hat / mx.maximum(depth_sigma, eps)
        return score

    scores = mx.vmap(_score_one_pure_mlx)(periods_days_top_k)
    weights = mx.softmax(scores / mx.array(max(temperature, 1e-6)))
    return MlxTopKScoreResult(
        top_k_periods=periods_days_top_k,
        scores=scores,
        weights=weights,
    )


def integrated_gradients(
    *,
    score_fn: Any,  # Callable[[mx.array], mx.array]
    flux: Any,  # mx.array
    baseline: Any,  # mx.array
    steps: int = 50,
) -> Any:
    """Compute integrated gradients attribution for score_fn w.r.t. flux."""
    mx = _require_mlx()
    alphas = mx.linspace(0.0, 1.0, steps)
    grads = []
    grad_fn = mx.grad(score_fn)
    for a in alphas.tolist():  # small loop, stable and explicit
        a_mx = mx.array(float(a))
        interp = baseline + a_mx * (flux - baseline)
        grads.append(grad_fn(interp))
    avg_grad = mx.mean(mx.stack(grads), axis=0)
    return (flux - baseline) * avg_grad


def score_fixed_period_refine_t0(
    *,
    time: Any,  # mx.array
    flux: Any,  # mx.array (normalized, ~1)
    flux_err: Any | None,  # mx.array or None
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    t0_scan_n: int = 81,
    t0_scan_half_span_minutes: float | None = None,
    ingress_egress_fraction: float = 0.2,
    sharpness: float = 30.0,
    eps: float = 1e-12,
) -> MlxT0RefinementResult:
    """Refine `t0` locally by scanning a small window and taking the best score."""
    mx = _require_mlx()

    n = int(t0_scan_n)
    if n < 21:
        raise ValueError(f"t0_scan_n must be >= 21, got {t0_scan_n}")
    if n % 2 == 0:
        n += 1

    if t0_scan_half_span_minutes is None:
        half_span_minutes = float(min(120.0, max(10.0, 0.5 * float(duration_hours) * 60.0)))
    else:
        half_span_minutes = float(t0_scan_half_span_minutes)

    half_span_days = half_span_minutes / (24.0 * 60.0)
    t0_grid = (float(t0_btjd) + np.linspace(-half_span_days, half_span_days, n)).astype(np.float64)

    scores_mx, _depth_hat_mx, _depth_sigma_mx = _batched_scores(
        mx=mx,
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=float(period_days),
        t0s_btjd=mx.array(t0_grid),
        duration_hours=float(duration_hours),
        ingress_egress_fraction=float(ingress_egress_fraction),
        sharpness=float(sharpness),
        eps=float(eps),
    )
    mx.eval(scores_mx)

    scores = np.asarray(scores_mx).astype(np.float64)
    best_index = int(np.argmax(scores))
    t0_best = float(t0_grid[best_index])
    score_best = float(scores[best_index])

    input_index = int(np.argmin(np.abs(t0_grid - float(t0_btjd))))
    score_at_input = float(scores[input_index])

    return MlxT0RefinementResult(
        t0_best_btjd=t0_best,
        score_best=score_best,
        score_at_input=score_at_input,
        delta_score=float(score_best - score_at_input),
        t0_grid_btjd=t0_grid,
        scores=scores,
    )
