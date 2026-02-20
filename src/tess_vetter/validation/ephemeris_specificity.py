from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tess_vetter.validation.base import get_in_transit_mask


@dataclass(frozen=True)
class SmoothTemplateConfig:
    ingress_egress_fraction: float = 0.2
    sharpness: float = 30.0


@dataclass(frozen=True)
class SmoothTemplateScoreResult:
    score: float
    depth_hat: float
    depth_sigma: float
    template: NDArray[np.float64]


@dataclass(frozen=True)
class PhaseShiftNullResult:
    n_trials: int
    strategy: Literal["grid", "random"]
    null_mean: float
    null_std: float
    z_score: float
    p_value_one_sided: float


@dataclass(frozen=True)
class ConcentrationMetrics:
    in_transit_contribution_abs: float
    max_point_fraction_abs: float
    top_5_fraction_abs: float
    effective_n_points: float
    n_in_transit: int


@dataclass(frozen=True)
class DepthThresholdResult:
    backend: Literal["numpy"]
    score_current: float
    target_score: float
    depth_hat_ppm: float
    depth_sigma_ppm: float
    depth_needed_ppm: float
    dscore_ddepth_per_fraction: float


def downsample_evenly(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    max_points: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    n = int(len(time))
    if n <= max_points:
        return time, flux, flux_err, n
    idx = np.linspace(0, n - 1, int(max_points), dtype=int)
    return time[idx], flux[idx], flux_err[idx], int(len(idx))


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def smooth_box_template_numpy(
    *,
    time: NDArray[np.float64],
    period_days: float,
    t0_btjd: NDArray[np.float64] | float,
    duration_hours: float,
    config: SmoothTemplateConfig,
) -> NDArray[np.float64]:
    """NumPy implementation mirroring `tess_vetter.compute.mlx_detection.smooth_box_template`."""
    period = float(period_days)
    t0 = np.asarray(t0_btjd, dtype=np.float64)
    duration_days = float(duration_hours) / 24.0
    half_duration = duration_days / 2.0
    ingress = max(duration_days * float(config.ingress_egress_fraction), 1e-6)
    k = float(config.sharpness) / ingress

    # Phase in [-0.5, 0.5)
    phase = (np.asarray(time, dtype=np.float64) - t0) / period
    phase = phase - np.floor(phase + 0.5)
    dt = np.abs(phase * period)  # days from transit center
    template = _sigmoid(k * (half_duration - dt))
    return np.clip(template, 0.0, 1.0)


def score_fixed_period_numpy(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    config: SmoothTemplateConfig,
    eps: float = 1e-12,
) -> SmoothTemplateScoreResult:
    """Return the MLX-like smooth-template score and fitted depth."""
    template = smooth_box_template_numpy(
        time=time,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        config=config,
    )

    y = 1.0 - flux
    sigma2 = np.maximum(flux_err * flux_err, eps)
    w = 1.0 / sigma2

    denom = float(np.sum(w * template * template) + eps)
    depth_hat = float(np.sum(w * template * y) / denom)
    depth_sigma = float(math.sqrt(1.0 / denom))
    score = float(depth_hat / max(depth_sigma, eps))
    return SmoothTemplateScoreResult(
        score=score, depth_hat=depth_hat, depth_sigma=depth_sigma, template=template
    )


def compute_depth_threshold_numpy(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    target_score: float,
    config: SmoothTemplateConfig,
) -> DepthThresholdResult:
    """Compute the additional depth needed (in ppm) to reach a target smooth-template score."""
    res = score_fixed_period_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
        config=config,
    )
    depth_hat = float(res.depth_hat)
    depth_sigma = float(res.depth_sigma)
    score_current = float(res.score)

    # score = depth_hat / depth_sigma, so d(score)/d(depth_hat) = 1/depth_sigma.
    dscore_ddepth = 1.0 / max(depth_sigma, 1e-12)
    depth_needed = max(float(target_score) * depth_sigma - depth_hat, 0.0)
    return DepthThresholdResult(
        backend="numpy",
        score_current=float(score_current),
        target_score=float(target_score),
        depth_hat_ppm=float(depth_hat * 1e6),
        depth_sigma_ppm=float(depth_sigma * 1e6),
        depth_needed_ppm=float(depth_needed * 1e6),
        dscore_ddepth_per_fraction=float(dscore_ddepth),
    )


def phase_shift_t0s(
    *,
    t0: float,
    period: float,
    n: int,
    strategy: Literal["grid", "random"],
    random_seed: int,
) -> NDArray[np.float64]:
    if strategy == "random":
        rng = np.random.default_rng(int(random_seed))
        phases = rng.uniform(0.0, 1.0, size=int(n))
        phases = np.where(phases < 1e-3, phases + 1e-3, phases)
        return (t0 + phases * period).astype(np.float64)

    phases = (np.arange(1, int(n) + 1, dtype=np.float64) / (float(n) + 1.0)).astype(np.float64)
    return (t0 + phases * period).astype(np.float64)


def scores_for_t0s_numpy(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period_days: float,
    t0s: NDArray[np.float64],
    duration_hours: float,
    config: SmoothTemplateConfig,
    chunk_size: int = 25,
) -> NDArray[np.float64]:
    y = 1.0 - flux
    w = 1.0 / np.maximum(flux_err * flux_err, 1e-12)

    out = np.empty(len(t0s), dtype=np.float64)
    for i in range(0, len(t0s), int(chunk_size)):
        t0_chunk = t0s[i : i + int(chunk_size)]
        template = smooth_box_template_numpy(
            time=time[None, :],
            period_days=period_days,
            t0_btjd=t0_chunk[:, None],
            duration_hours=duration_hours,
            config=config,
        )
        denom = np.sum(w[None, :] * template * template, axis=1) + 1e-12
        depth_hat = np.sum(w[None, :] * template * y[None, :], axis=1) / denom
        depth_sigma = np.sqrt(1.0 / denom)
        out[i : i + int(chunk_size)] = depth_hat / np.maximum(depth_sigma, 1e-12)
    return out


def compute_phase_shift_null(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    observed_score: float,
    n_trials: int,
    strategy: Literal["grid", "random"],
    random_seed: int,
    config: SmoothTemplateConfig,
) -> PhaseShiftNullResult:
    t0s = phase_shift_t0s(
        t0=float(t0_btjd),
        period=float(period_days),
        n=int(n_trials),
        strategy=strategy,
        random_seed=int(random_seed),
    )
    null_scores = scores_for_t0s_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=float(period_days),
        t0s=t0s,
        duration_hours=float(duration_hours),
        config=config,
    )
    null_mean = float(np.mean(null_scores))
    null_std = float(np.std(null_scores))
    z_score = float((observed_score - null_mean) / null_std) if null_std > 0 else float("nan")
    p_value = float((np.sum(null_scores >= observed_score) + 1.0) / (len(null_scores) + 1.0))
    return PhaseShiftNullResult(
        n_trials=int(len(null_scores)),
        strategy=strategy,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z_score,
        p_value_one_sided=p_value,
    )


def compute_concentration_metrics(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    template: NDArray[np.float64],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
) -> ConcentrationMetrics:
    in_mask = get_in_transit_mask(time, float(period_days), float(t0_btjd), float(duration_hours))
    n_in_transit = int(np.sum(in_mask))

    y = 1.0 - flux
    w = 1.0 / np.maximum(flux_err * flux_err, 1e-12)
    contrib = w * template * y

    abs_contrib = np.abs(contrib)
    abs_total = float(np.sum(abs_contrib))
    if abs_total > 0:
        in_transit_contribution = float(np.sum(abs_contrib[in_mask]) / abs_total)
        max_point_fraction = float(np.max(abs_contrib) / abs_total)
        sorted_abs = np.sort(abs_contrib)[::-1]
        top_5_fraction = float(np.sum(sorted_abs[:5]) / abs_total)
        p = abs_contrib / abs_total
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        effective_n = float(np.exp(entropy))
    else:
        in_transit_contribution = float("nan")
        max_point_fraction = float("nan")
        top_5_fraction = float("nan")
        effective_n = float("nan")

    return ConcentrationMetrics(
        in_transit_contribution_abs=in_transit_contribution,
        max_point_fraction_abs=max_point_fraction,
        top_5_fraction_abs=top_5_fraction,
        effective_n_points=effective_n,
        n_in_transit=n_in_transit,
    )


@dataclass(frozen=True)
class LocalT0SensitivityResult:
    backend: Literal["numpy"]
    t0_best_btjd: float
    score_at_input: float
    score_best: float
    delta_score: float
    curvature: float
    fwhm_minutes: float
    n_grid: int
    half_span_minutes: float


def _estimate_curvature_from_grid(
    *,
    t0s: NDArray[np.float64],
    scores: NDArray[np.float64],
    best_index: int,
) -> float:
    if best_index <= 0 or best_index >= len(t0s) - 1:
        return float("nan")
    x = t0s[best_index - 1 : best_index + 2]
    y = scores[best_index - 1 : best_index + 2]
    try:
        a, _b, _c = np.polyfit(x, y, 2)
    except Exception:
        return float("nan")
    return float(2.0 * a)


def _estimate_fwhm_minutes(
    *,
    t0s: NDArray[np.float64],
    scores: NDArray[np.float64],
    best_index: int,
) -> float:
    if len(t0s) < 5:
        return float("nan")
    peak = float(scores[best_index])
    baseline = float(np.median(scores))
    thresh = (peak + baseline) / 2.0

    left = best_index
    while left > 0 and float(scores[left]) >= thresh:
        left -= 1
    right = best_index
    while right < len(scores) - 1 and float(scores[right]) >= thresh:
        right += 1
    if left == best_index or right == best_index:
        return float("nan")
    width_days = float(t0s[right] - t0s[left])
    return float(width_days * 24.0 * 60.0)


def compute_local_t0_sensitivity_numpy(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    config: SmoothTemplateConfig,
    n_grid: int = 81,
    half_span_minutes: float | None = None,
) -> LocalT0SensitivityResult:
    if half_span_minutes is None:
        half_span_minutes = float(duration_hours) * 60.0
    half_span_days = float(half_span_minutes) / (24.0 * 60.0)

    t0s = (float(t0_btjd) + np.linspace(-half_span_days, half_span_days, int(n_grid))).astype(
        np.float64
    )
    scores = scores_for_t0s_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=float(period_days),
        t0s=t0s,
        duration_hours=float(duration_hours),
        config=config,
    )

    best_index = int(np.argmax(scores))
    t0_best = float(t0s[best_index])
    score_best = float(scores[best_index])
    score_at_input = float(scores[int(np.argmin(np.abs(t0s - float(t0_btjd))))])

    curvature = _estimate_curvature_from_grid(t0s=t0s, scores=scores, best_index=best_index)
    fwhm_minutes = _estimate_fwhm_minutes(t0s=t0s, scores=scores, best_index=best_index)
    return LocalT0SensitivityResult(
        backend="numpy",
        t0_best_btjd=t0_best,
        score_at_input=score_at_input,
        score_best=score_best,
        delta_score=float(score_best - score_at_input),
        curvature=float(curvature),
        fwhm_minutes=float(fwhm_minutes),
        n_grid=int(n_grid),
        half_span_minutes=float(half_span_minutes),
    )
