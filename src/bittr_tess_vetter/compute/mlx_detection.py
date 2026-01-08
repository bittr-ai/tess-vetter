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

    def _score_one(p: Any) -> Any:
        return score_fixed_period(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period_days=float(p),
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
            ingress_egress_fraction=ingress_egress_fraction,
            sharpness=sharpness,
        )

    scores = mx.vmap(_score_one)(periods_days_top_k)
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
