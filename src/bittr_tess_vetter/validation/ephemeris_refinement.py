from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.validation.ephemeris_specificity import (
    SmoothTemplateConfig,
    score_fixed_period_numpy,
)


@dataclass(frozen=True)
class EphemerisRefinementCandidate:
    period_days: float
    t0_btjd: float
    duration_hours: float


@dataclass(frozen=True)
class EphemerisRefinementConfig:
    steps: int = 25
    lr: float = 0.05
    duration_bounds_mode: Literal["absolute", "relative"] = "absolute"
    duration_min_hours: float = 0.5
    duration_max_hours: float = 12.0
    duration_min_factor: float = 0.5
    duration_max_factor: float = 2.0
    t0_window_phase: float = 0.02
    smooth_template: SmoothTemplateConfig = SmoothTemplateConfig()


@dataclass(frozen=True)
class EphemerisRefinementCandidateResult:
    period_days: float
    t0_init_btjd: float
    duration_init_hours: float
    t0_refined_btjd: float
    duration_refined_hours: float
    t0_shift_cycles: float
    t0_shift_cycles_wrapped: float
    duration_ratio_refined_over_init: float
    depth_hat_ppm: float
    depth_sigma_ppm: float
    score_z: float
    steps: int
    lr: float
    t0_window_phase: float
    duration_bounds_mode: str
    duration_bounds_hours: list[float]
    ingress_egress_fraction: float
    sharpness: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_days": float(self.period_days),
            "t0_init_btjd": float(self.t0_init_btjd),
            "duration_init_hours": float(self.duration_init_hours),
            "t0_refined_btjd": float(self.t0_refined_btjd),
            "duration_refined_hours": float(self.duration_refined_hours),
            "t0_shift_cycles": float(self.t0_shift_cycles),
            "t0_shift_cycles_wrapped": float(self.t0_shift_cycles_wrapped),
            "duration_ratio_refined_over_init": float(self.duration_ratio_refined_over_init),
            "depth_hat_ppm": float(self.depth_hat_ppm),
            "depth_sigma_ppm": float(self.depth_sigma_ppm),
            "score_z": float(self.score_z),
            "steps": int(self.steps),
            "lr": float(self.lr),
            "t0_window_phase": float(self.t0_window_phase),
            "duration_bounds_mode": str(self.duration_bounds_mode),
            "duration_bounds_hours": [float(x) for x in self.duration_bounds_hours],
            "ingress_egress_fraction": float(self.ingress_egress_fraction),
            "sharpness": float(self.sharpness),
        }


@dataclass(frozen=True)
class EphemerisRefinementRunResult:
    n_points_used: int
    refined: list[EphemerisRefinementCandidateResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_points_used": int(self.n_points_used),
            "refined": [row.to_dict() for row in self.refined],
        }


def _duration_bounds_hours(
    *,
    duration_init_hours: float,
    config: EphemerisRefinementConfig,
) -> tuple[float, float]:
    if config.duration_bounds_mode not in {"absolute", "relative"}:
        raise ValueError("duration_bounds_mode must be 'absolute' or 'relative'")
    if config.duration_bounds_mode == "relative":
        dur_min_h = float(duration_init_hours) * float(config.duration_min_factor)
        dur_max_h = float(duration_init_hours) * float(config.duration_max_factor)
    else:
        dur_min_h = float(config.duration_min_hours)
        dur_max_h = float(config.duration_max_hours)

    # Clamp and validate (consistent with legacy astro implementation).
    dur_min_h = max(1e-3, min(dur_min_h, 48.0))
    dur_max_h = max(1e-3, min(dur_max_h, 48.0))
    if dur_max_h <= dur_min_h:
        dur_max_h = dur_min_h + 0.5
    return float(dur_min_h), float(dur_max_h)


def refine_one_candidate_numpy(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None,
    candidate: EphemerisRefinementCandidate,
    config: EphemerisRefinementConfig,
) -> EphemerisRefinementCandidateResult:
    """Refine one candidate's (t0, duration) by maximizing smooth-template score.

    Notes:
    - This is a deterministic, CPU-only refinement utility intended as a fallback or
      a lightweight local "polish" step, not a global search.
    - The objective is the smooth-template z-score (depth_hat / depth_sigma).
    - Bounds are enforced via a stable parameterization to prevent drifting to
      unrelated epochs/durations.
    """
    period_days = float(candidate.period_days)
    t0_btjd = float(candidate.t0_btjd)
    duration_init_hours = float(candidate.duration_hours)

    dur_min_h, dur_max_h = _duration_bounds_hours(
        duration_init_hours=duration_init_hours,
        config=config,
    )
    delta_max_days = float(config.t0_window_phase) * period_days

    ferr = (
        np.asarray(flux_err, dtype=np.float64)
        if flux_err is not None
        else np.ones_like(flux, dtype=np.float64)
    )

    def _unpack(params: NDArray[np.float64]) -> tuple[float, float]:
        raw_dt0 = float(params[0])
        raw_dur = float(params[1])
        t0 = t0_btjd + delta_max_days * math.tanh(raw_dt0)
        dur_h = dur_min_h + (dur_max_h - dur_min_h) / (1.0 + math.exp(-raw_dur))
        return float(t0), float(dur_h)

    def _score_depth_sigma(t0: float, dur_h: float) -> tuple[float, float, float]:
        res = score_fixed_period_numpy(
            time=np.asarray(time, dtype=np.float64),
            flux=np.asarray(flux, dtype=np.float64),
            flux_err=ferr,
            period_days=period_days,
            t0_btjd=float(t0),
            duration_hours=float(dur_h),
            config=config.smooth_template,
        )
        return float(res.score), float(res.depth_hat), float(res.depth_sigma)

    def _loss(params: NDArray[np.float64]) -> float:
        t0, dur_h = _unpack(params)
        score, _, _ = _score_depth_sigma(t0, dur_h)
        return -float(score)

    def _numerical_grad(params: NDArray[np.float64], eps: float = 1e-5) -> NDArray[np.float64]:
        grad = np.zeros_like(params)
        for i in range(int(len(params))):
            params_p = params.copy()
            params_m = params.copy()
            params_p[i] += eps
            params_m[i] -= eps
            grad[i] = (_loss(params_p) - _loss(params_m)) / (2.0 * eps)
        return grad

    steps = int(config.steps)
    lr = float(config.lr)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if lr <= 0.0:
        raise ValueError("lr must be > 0")

    # Adam optimizer on bounded parameterization.
    params = np.array([0.0, 0.0], dtype=np.float64)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step_i in range(steps):
        g = _numerical_grad(params)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        t_step = float(step_i + 1)
        m_hat = m / (1.0 - beta1**t_step)
        v_hat = v / (1.0 - beta2**t_step)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

    t0_final, dur_final_h = _unpack(params)
    score, depth_hat, depth_sigma = _score_depth_sigma(t0_final, dur_final_h)

    depth_ppm = depth_hat * 1_000_000.0
    depth_sigma_ppm = depth_sigma * 1_000_000.0
    t0_shift_cycles = (t0_final - t0_btjd) / period_days
    t0_shift_cycles_wrapped = ((t0_shift_cycles + 0.5) % 1.0) - 0.5
    dur_ratio = dur_final_h / max(duration_init_hours, 1e-9)

    return EphemerisRefinementCandidateResult(
        period_days=period_days,
        t0_init_btjd=t0_btjd,
        duration_init_hours=duration_init_hours,
        t0_refined_btjd=float(t0_final),
        duration_refined_hours=float(dur_final_h),
        t0_shift_cycles=float(t0_shift_cycles),
        t0_shift_cycles_wrapped=float(t0_shift_cycles_wrapped),
        duration_ratio_refined_over_init=float(dur_ratio),
        depth_hat_ppm=float(depth_ppm),
        depth_sigma_ppm=float(depth_sigma_ppm),
        score_z=float(score),
        steps=int(steps),
        lr=float(lr),
        t0_window_phase=float(config.t0_window_phase),
        duration_bounds_mode=str(config.duration_bounds_mode),
        duration_bounds_hours=[float(dur_min_h), float(dur_max_h)],
        ingress_egress_fraction=float(config.smooth_template.ingress_egress_fraction),
        sharpness=float(config.smooth_template.sharpness),
    )


def refine_candidates_numpy(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64] | None,
    candidates: list[EphemerisRefinementCandidate],
    config: EphemerisRefinementConfig,
) -> EphemerisRefinementRunResult:
    refined = [
        refine_one_candidate_numpy(
            time=time,
            flux=flux,
            flux_err=flux_err,
            candidate=c,
            config=config,
        )
        for c in candidates
    ]
    return EphemerisRefinementRunResult(n_points_used=int(time.size), refined=refined)
