from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.validation.ephemeris_specificity import (
    ConcentrationMetrics,
    LocalT0SensitivityResult,
    PhaseShiftNullResult,
    SmoothTemplateConfig,
    SmoothTemplateScoreResult,
    compute_concentration_metrics,
    compute_local_t0_sensitivity_numpy,
    compute_phase_shift_null,
    score_fixed_period_numpy,
)


@dataclass(frozen=True)
class PeriodNeighborhoodResult:
    period_grid_days: NDArray[np.float64]
    scores: NDArray[np.float64]
    best_period_days: float
    best_score: float
    score_at_input: float
    second_best_score: float
    peak_to_next: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_grid_days": [float(x) for x in self.period_grid_days],
            "scores": [float(x) for x in self.scores],
            "best_period_days": float(self.best_period_days),
            "best_score": float(self.best_score),
            "score_at_input": float(self.score_at_input),
            "second_best_score": float(self.second_best_score),
            "peak_to_next": float(self.peak_to_next),
        }


@dataclass(frozen=True)
class AblationResult:
    n_removed: int
    score: float
    score_drop_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_removed": int(self.n_removed),
            "score": float(self.score),
            "score_drop_fraction": float(self.score_drop_fraction),
        }


@dataclass(frozen=True)
class EphemerisReliabilityRegimeResult:
    base: SmoothTemplateScoreResult
    phase_shift_null: PhaseShiftNullResult
    null_percentile: float
    period_neighborhood: PeriodNeighborhoodResult
    harmonics: dict[str, Any]
    concentration: ConcentrationMetrics
    top_contribution_fractions: dict[str, float]
    ablation: list[AblationResult]
    max_ablation_score_drop_fraction: float
    t0_sensitivity: LocalT0SensitivityResult
    label: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": {
                "score": float(self.base.score),
                "depth_hat": float(self.base.depth_hat),
                "depth_sigma": float(self.base.depth_sigma),
                "depth_hat_ppm": float(self.base.depth_hat * 1e6),
                "depth_sigma_ppm": float(self.base.depth_sigma * 1e6),
            },
            "phase_shift_null": self.phase_shift_null.__dict__,
            "null_percentile": float(self.null_percentile),
            "period_neighborhood": self.period_neighborhood.to_dict(),
            "harmonics": dict(self.harmonics),
            "concentration": self.concentration.__dict__,
            "top_contribution_fractions": {
                str(k): float(v) for k, v in self.top_contribution_fractions.items()
            },
            "ablation": [row.to_dict() for row in self.ablation],
            "max_ablation_score_drop_fraction": float(self.max_ablation_score_drop_fraction),
            "t0_sensitivity": self.t0_sensitivity.__dict__,
            "label": str(self.label),
            "warnings": list(self.warnings),
        }


def _period_grid_around(
    *,
    period_days: float,
    period_jitter_frac: float,
    n: int,
) -> NDArray[np.float64]:
    p0 = float(period_days)
    frac = float(period_jitter_frac)
    n = int(n)
    if n < 3:
        raise ValueError("period_jitter_n must be >= 3")
    if n % 2 == 0:
        n += 1
    grid = p0 * (1.0 + np.linspace(-frac, frac, n, dtype=np.float64))
    return np.clip(grid, 0.05, 1000.0).astype(np.float64)


def compute_reliability_regime_numpy(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    config: SmoothTemplateConfig,
    n_phase_shifts: int = 200,
    phase_shift_strategy: Literal["grid", "random"] = "grid",
    random_seed: int = 0,
    period_jitter_frac: float = 0.002,
    period_jitter_n: int = 21,
    include_harmonics: bool = True,
    ablation_top_ns: tuple[int, ...] = (1, 3, 5),
    contribution_top_n: int = 10,
    t0_scan_n: int = 81,
    t0_scan_half_span_minutes: float | None = None,
    # Warning thresholds (policy-ish but still lightweight)
    p_value_warn_threshold: float = 0.01,
    peak_ratio_warn_threshold: float = 1.5,
    ablation_score_drop_warn_threshold: float = 0.5,
    top_contribution_warn_fraction: float = 0.35,
) -> EphemerisReliabilityRegimeResult:
    base = score_fixed_period_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
        config=config,
    )

    null = compute_phase_shift_null(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
        observed_score=float(base.score),
        n_trials=int(n_phase_shifts),
        strategy=phase_shift_strategy,
        random_seed=int(random_seed),
        config=config,
    )
    p_value = float(null.p_value_one_sided)
    null_percentile = float(1.0 - p_value) if np.isfinite(p_value) else float("nan")

    period_grid = _period_grid_around(
        period_days=float(period_days),
        period_jitter_frac=float(period_jitter_frac),
        n=int(period_jitter_n),
    )
    scores = np.empty_like(period_grid)
    for i, p in enumerate(period_grid):
        scores[i] = float(
            score_fixed_period_numpy(
                time=time,
                flux=flux,
                flux_err=flux_err,
                period_days=float(p),
                t0_btjd=float(t0_btjd),
                duration_hours=float(duration_hours),
                config=config,
            ).score
        )

    best_idx = int(np.argmax(scores))
    best_period = float(period_grid[best_idx])
    best_score = float(scores[best_idx])
    score_at_input = float(scores[int(np.argmin(np.abs(period_grid - float(period_days))))])
    scores_wo_best = np.delete(scores, best_idx)
    second_best = float(np.max(scores_wo_best)) if scores_wo_best.size else float("nan")
    peak_to_next = (
        float(abs(best_score) / abs(second_best))
        if np.isfinite(second_best) and abs(second_best) > 0
        else float("nan")
    )
    neighborhood = PeriodNeighborhoodResult(
        period_grid_days=period_grid,
        scores=scores,
        best_period_days=best_period,
        best_score=best_score,
        score_at_input=score_at_input,
        second_best_score=second_best,
        peak_to_next=peak_to_next,
    )

    harmonics: dict[str, Any] = {}
    if include_harmonics:
        for mult in (0.5, 2.0):
            ph = float(period_days) * float(mult)
            if ph < 0.05 or ph > 1000.0:
                continue
            h_score = score_fixed_period_numpy(
                time=time,
                flux=flux,
                flux_err=flux_err,
                period_days=float(ph),
                t0_btjd=float(t0_btjd),
                duration_hours=float(duration_hours),
                config=config,
            ).score
            harmonics[f"period_x{mult:g}"] = {"period_days": float(ph), "score": float(h_score)}

    y = 1.0 - flux
    w = 1.0 / np.maximum(flux_err * flux_err, 1e-12)
    contrib = (w * base.template * y).astype(np.float64)
    abs_contrib = np.abs(contrib)
    order = np.argsort(-abs_contrib)
    total = float(np.sum(abs_contrib)) + 1e-18

    contribution_top_n = int(max(5, min(int(contribution_top_n), int(order.size))))
    top_fracs: dict[str, float] = {}
    for k in (1, 3, 5, min(10, contribution_top_n), contribution_top_n):
        kk = int(max(1, min(int(k), int(order.size))))
        top_fracs[f"top_{kk}_fraction"] = float(np.sum(abs_contrib[order[:kk]]) / total)

    concentration = compute_concentration_metrics(
        time=time,
        flux=flux,
        flux_err=flux_err,
        template=base.template,
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
    )

    ablation: list[AblationResult] = []
    ablation_drops: list[float] = []
    base_abs = float(abs(base.score))
    for n_remove in ablation_top_ns:
        n_remove = int(max(1, min(int(n_remove), int(order.size))))
        mask = np.ones(order.size, dtype=bool)
        mask[order[:n_remove]] = False
        s_masked = float(
            score_fixed_period_numpy(
                time=time[mask],
                flux=flux[mask],
                flux_err=flux_err[mask],
                period_days=float(period_days),
                t0_btjd=float(t0_btjd),
                duration_hours=float(duration_hours),
                config=config,
            ).score
        )
        drop = float((base_abs - abs(s_masked)) / max(base_abs, 1e-12))
        drop = float(max(drop, 0.0))
        ablation_drops.append(drop)
        ablation.append(AblationResult(n_removed=n_remove, score=s_masked, score_drop_fraction=drop))

    max_ablation_drop = float(np.max(np.asarray(ablation_drops))) if ablation_drops else 0.0

    if t0_scan_half_span_minutes is None:
        t0_scan_half_span_minutes = float(duration_hours) * 60.0 * 1.5
    t0_sens = compute_local_t0_sensitivity_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
        config=config,
        n_grid=int(t0_scan_n),
        half_span_minutes=float(t0_scan_half_span_minutes),
    )

    warnings: list[str] = []
    label = "ok"

    if np.isfinite(p_value) and p_value > float(p_value_warn_threshold):
        warnings.append(
            f"Phase-shift null p={p_value:.3g} > {float(p_value_warn_threshold):.3g} (false-alarm dominated)"
        )
        label = "low_reliability"

    if np.isfinite(peak_to_next) and peak_to_next < float(peak_ratio_warn_threshold):
        warnings.append(
            f"Period neighborhood confusable (peak/next={peak_to_next:.2f} < {float(peak_ratio_warn_threshold):.2f})"
        )
        if label == "ok":
            label = "pipeline_sensitive"

    if max_ablation_drop > float(ablation_score_drop_warn_threshold):
        warnings.append(
            f"Few-point dominated (max score drop {max_ablation_drop:.2f} > {float(ablation_score_drop_warn_threshold):.2f})"
        )
        if label == "ok":
            label = "pipeline_sensitive"

    top5 = float(top_fracs.get("top_5_fraction", 0.0))
    if top5 > float(top_contribution_warn_fraction):
        warnings.append(
            f"Few-point dominated (top_5_fraction {top5:.2f} > {float(top_contribution_warn_fraction):.2f})"
        )
        if label == "ok":
            label = "pipeline_sensitive"

    return EphemerisReliabilityRegimeResult(
        base=base,
        phase_shift_null=null,
        null_percentile=null_percentile,
        period_neighborhood=neighborhood,
        harmonics=harmonics,
        concentration=concentration,
        top_contribution_fractions=top_fracs,
        ablation=ablation,
        max_ablation_score_drop_fraction=max_ablation_drop,
        t0_sensitivity=t0_sens,
        label=label,
        warnings=warnings,
    )

