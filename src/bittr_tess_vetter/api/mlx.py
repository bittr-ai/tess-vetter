"""MLX detection primitives for the public API (optional dependency).

This module is a thin facade over `bittr_tess_vetter.compute.mlx_detection` to provide
stable import paths for host applications (e.g., MCP tools).

Importing this module does not require MLX to be installed. Calling any of the
functions will raise an actionable ImportError if MLX is missing.

Novelty: new (API facade over standard methods)

References:
    [1] Kovács et al. 2002 (2002A&A...391..369K): Box-fitting Least Squares (BLS) methodology
    [2] Sundararajan et al. 2017 (arXiv:1703.01365): Integrated Gradients attribution
"""

from __future__ import annotations

import importlib.util

from bittr_tess_vetter.api.references import KOVACS_2002, SUNDARARAJAN_2017, cite, cites
from bittr_tess_vetter.compute import mlx_detection as _mlx_detection
from bittr_tess_vetter.compute.mlx_detection import (
    MlxT0RefinementResult,
    MlxTopKScoreResult,
)


def _mlx_is_available() -> bool:
    # MLX is an optional dependency; check availability without importing.
    return importlib.util.find_spec("mlx") is not None


MLX_AVAILABLE = _mlx_is_available()

REFERENCES = [ref.to_dict() for ref in [KOVACS_2002, SUNDARARAJAN_2017]]


@cites(
    cite(KOVACS_2002, "box-fitting / matched-filter scoring lineage"),
)
def smooth_box_template(
    *,
    time: object,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    ingress_egress_fraction: float = 0.2,
    sharpness: float = 30.0,
) -> object:
    return _mlx_detection.smooth_box_template(
        time=time,  # type: ignore[arg-type]
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        ingress_egress_fraction=ingress_egress_fraction,
        sharpness=sharpness,
    )


@cites(
    cite(KOVACS_2002, "least-squares depth estimate against box template"),
)
def score_fixed_period(
    *,
    time: object,
    flux: object,
    flux_err: object | None,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    ingress_egress_fraction: float = 0.2,
    sharpness: float = 30.0,
    eps: float = 1e-12,
) -> object:
    return _mlx_detection.score_fixed_period(
        time=time,  # type: ignore[arg-type]
        flux=flux,  # type: ignore[arg-type]
        flux_err=flux_err,  # type: ignore[arg-type]
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        ingress_egress_fraction=ingress_egress_fraction,
        sharpness=sharpness,
        eps=eps,
    )


@cites(
    cite(KOVACS_2002, "epoch/phase search and local refinement within a template-based detector"),
)
def score_fixed_period_refine_t0(
    *,
    time: object,
    flux: object,
    flux_err: object | None,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    t0_scan_n: int = 81,
    t0_scan_half_span_minutes: float | None = None,
    ingress_egress_fraction: float = 0.2,
    sharpness: float = 30.0,
    eps: float = 1e-12,
) -> MlxT0RefinementResult:
    return _mlx_detection.score_fixed_period_refine_t0(
        time=time,  # type: ignore[arg-type]
        flux=flux,  # type: ignore[arg-type]
        flux_err=flux_err,  # type: ignore[arg-type]
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        t0_scan_n=t0_scan_n,
        t0_scan_half_span_minutes=t0_scan_half_span_minutes,
        ingress_egress_fraction=ingress_egress_fraction,
        sharpness=sharpness,
        eps=eps,
    )


@cites(
    cite(KOVACS_2002, "scoring candidate periods with box-like templates"),
)
def score_top_k_periods(
    *,
    time: object,
    flux: object,
    flux_err: object | None,
    periods_days_top_k: object,
    t0_btjd: float,
    duration_hours: float,
    temperature: float = 0.5,
    ingress_egress_fraction: float = 0.2,
    sharpness: float = 30.0,
) -> MlxTopKScoreResult:
    return _mlx_detection.score_top_k_periods(
        time=time,  # type: ignore[arg-type]
        flux=flux,  # type: ignore[arg-type]
        flux_err=flux_err,  # type: ignore[arg-type]
        periods_days_top_k=periods_days_top_k,  # type: ignore[arg-type]
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        temperature=temperature,
        ingress_egress_fraction=ingress_egress_fraction,
        sharpness=sharpness,
    )


@cites(
    cite(SUNDARARAJAN_2017, "Integrated Gradients attribution method"),
)
def integrated_gradients(
    *,
    score_fn: object,
    flux: object,
    baseline: object,
    steps: int = 50,
) -> object:
    return _mlx_detection.integrated_gradients(
        score_fn=score_fn,
        flux=flux,  # type: ignore[arg-type]
        baseline=baseline,  # type: ignore[arg-type]
        steps=steps,
    )

__all__ = [
    "MLX_AVAILABLE",
    "REFERENCES",
    "MlxTopKScoreResult",
    "MlxT0RefinementResult",
    "smooth_box_template",
    "score_fixed_period",
    "score_fixed_period_refine_t0",
    "score_top_k_periods",
    "integrated_gradients",
]
