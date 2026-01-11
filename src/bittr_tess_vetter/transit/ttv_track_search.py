"""TTV-aware transit detection via per-window timing-offset track search.

This module implements a RIVERS-like "track search" that can recover signals
with large transit timing variations (TTVs) that periodic-only BLS/TLS can miss.

Core idea:
  - Assume a base ephemeris (period, t0) and a set of observing windows (e.g.,
    sectors / continuous stretches separated by data gaps).
  - Search over bounded timing offsets per window and score each offset "track"
    by the improvement of a simple box transit model vs. a periodic-only model.

This is intentionally conservative and "physics-light": it is a detection aid,
not a full dynamical TTV model.
"""

from __future__ import annotations

import itertools
import time as time_module
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TTVSearchBudget:
    """Budget controls for TTV track search."""

    max_runtime_seconds: float = 60.0
    max_period_evaluations: int = 200
    max_track_hypotheses: int = 20_000

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_period_evaluations": self.max_period_evaluations,
            "max_track_hypotheses": self.max_track_hypotheses,
        }


@dataclass(frozen=True)
class TTVTrackHypothesis:
    """A TTV timing-offset track hypothesis."""

    track_id: str
    base_period_days: float
    base_t0_btjd: float
    window_offsets_days: list[float]
    score: float
    score_improvement: float
    n_transits_matched: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "base_period_days": round(self.base_period_days, 10),
            "base_t0_btjd": round(self.base_t0_btjd, 8),
            "window_offsets_days": [round(x, 8) for x in self.window_offsets_days],
            "score": round(self.score, 6),
            "score_improvement": round(self.score_improvement, 6),
            "n_transits_matched": self.n_transits_matched,
        }


@dataclass(frozen=True)
class TTVTrackCandidate:
    """A candidate recovered by a TTV track search."""

    best_track: TTVTrackHypothesis
    alternative_tracks: list[TTVTrackHypothesis]
    periodic_score: float
    per_transit_residuals: list[float]
    runtime_seconds: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_track": self.best_track.to_dict(),
            "alternative_tracks": [t.to_dict() for t in self.alternative_tracks],
            "periodic_score": round(self.periodic_score, 6),
            "per_transit_residuals": [round(x, 10) for x in self.per_transit_residuals],
            "runtime_seconds": round(self.runtime_seconds, 3),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class TTVTrackSearchResult:
    """Full result from a TTV track search."""

    candidates: list[TTVTrackCandidate]
    n_periods_searched: int
    n_tracks_evaluated: int
    runtime_seconds: float
    budget_exhausted: bool
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "n_periods_searched": self.n_periods_searched,
            "n_tracks_evaluated": self.n_tracks_evaluated,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "budget_exhausted": self.budget_exhausted,
            "provenance": self.provenance,
        }


def identify_observing_windows(
    time_btjd: np.ndarray,
    *,
    gap_threshold_days: float = 5.0,
) -> list[tuple[float, float]]:
    """Identify observing windows separated by large time gaps."""
    if time_btjd.size == 0:
        return []
    t = np.asarray(time_btjd, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return []
    t = np.sort(t)
    gaps = np.diff(t)
    gap_indices = np.where(gaps > gap_threshold_days)[0]

    if gap_indices.size == 0:
        return [(float(t[0]), float(t[-1]))]

    windows: list[tuple[float, float]] = []
    windows.append((float(t[0]), float(t[gap_indices[0]])))
    for i in range(len(gap_indices) - 1):
        start_idx = int(gap_indices[i] + 1)
        end_idx = int(gap_indices[i + 1])
        windows.append((float(t[start_idx]), float(t[end_idx])))
    windows.append((float(t[int(gap_indices[-1] + 1)]), float(t[-1])))
    return windows


def should_run_ttv_search(
    time_btjd: np.ndarray,
    *,
    min_baseline_days: float = 100.0,
    min_windows: int = 3,
    gap_threshold_days: float = 5.0,
) -> bool:
    """Heuristic guardrail: TTV track search is only meaningful on long baselines."""
    t = np.asarray(time_btjd, dtype=float)
    t = t[np.isfinite(t)]
    if t.size == 0:
        return False
    baseline = float(np.nanmax(t) - np.nanmin(t))
    if baseline < min_baseline_days:
        return False
    windows = identify_observing_windows(t, gap_threshold_days=gap_threshold_days)
    return len(windows) >= min_windows


def _box_model(
    time_btjd: np.ndarray,
    t_center_btjd: float,
    *,
    depth_frac: float,
    duration_days: float,
) -> np.ndarray:
    model = np.ones_like(time_btjd, dtype=float)
    half_dur = float(duration_days) / 2.0
    in_transit = np.abs(time_btjd - float(t_center_btjd)) < half_dur
    model[in_transit] = 1.0 - float(depth_frac)
    return model


def _expected_transit_times(
    time_btjd: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
) -> list[float]:
    time_min = float(np.nanmin(time_btjd))
    time_max = float(np.nanmax(time_btjd))
    n_min = int(np.floor((time_min - t0_btjd) / period_days)) - 1
    n_max = int(np.ceil((time_max - t0_btjd) / period_days)) + 1
    transit_times = [t0_btjd + n * period_days for n in range(n_min, n_max + 1)]
    return [t for t in transit_times if (time_min - period_days) <= t <= (time_max + period_days)]


def score_periodic_model(
    time_btjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_frac: float | None = None,
) -> tuple[float, float, int]:
    """Score a periodic box model using chi^2 improvement vs. flat baseline."""
    duration_days = float(duration_hours) / 24.0
    transit_times = _expected_transit_times(time_btjd, period_days=period_days, t0_btjd=t0_btjd)
    if not transit_times:
        return 0.0, 0.0, 0

    if depth_frac is None:
        in_transit_mask = np.zeros(time_btjd.size, dtype=bool)
        for t_transit in transit_times:
            in_transit_mask |= np.abs(time_btjd - t_transit) < duration_days / 2.0
        out_transit_mask = ~in_transit_mask
        if np.any(in_transit_mask) and np.any(out_transit_mask):
            baseline = float(np.nanmedian(flux[out_transit_mask]))
            transit_flux = float(np.nanmedian(flux[in_transit_mask]))
            depth_frac = max(1e-8, baseline - transit_flux)
        else:
            depth_frac = 1e-3

    model = np.ones_like(flux, dtype=float)
    for t_transit in transit_times:
        in_transit = np.abs(time_btjd - t_transit) < duration_days / 2.0
        model[in_transit] = 1.0 - float(depth_frac)

    valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0) & np.isfinite(time_btjd)
    if not np.any(valid):
        return 0.0, float(depth_frac), 0

    residuals_null = (flux[valid] - 1.0) / flux_err[valid]
    residuals_model = (flux[valid] - model[valid]) / flux_err[valid]

    chi2_null = float(np.sum(residuals_null**2))
    chi2_model = float(np.sum(residuals_model**2))
    score = max(0.0, chi2_null - chi2_model)

    n_transits = len(transit_times)
    if n_transits > 0:
        score = score / np.sqrt(n_transits)
    return score, float(depth_frac), n_transits


def score_track_hypothesis(
    time_btjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    window_boundaries: list[tuple[float, float]],
    window_offsets_days: list[float],
    depth_frac: float | None = None,
) -> tuple[float, int, list[float]]:
    """Score a per-window-offset track hypothesis."""
    if len(window_offsets_days) != len(window_boundaries):
        raise ValueError(
            f"window_offsets_days length ({len(window_offsets_days)}) must match "
            f"window_boundaries length ({len(window_boundaries)})"
        )
    duration_days = float(duration_hours) / 24.0
    linear_transits = _expected_transit_times(time_btjd, period_days=period_days, t0_btjd=t0_btjd)
    if not linear_transits:
        return 0.0, 0, []

    adjusted_transits: list[float] = []
    for t_linear in linear_transits:
        for w_idx, (w_start, w_end) in enumerate(window_boundaries):
            if (w_start - period_days) <= t_linear <= (w_end + period_days):
                adjusted_transits.append(t_linear + float(window_offsets_days[w_idx]))
                break

    if not adjusted_transits:
        return 0.0, 0, []

    if depth_frac is None:
        in_transit_mask = np.zeros(time_btjd.size, dtype=bool)
        for t_transit in adjusted_transits:
            in_transit_mask |= np.abs(time_btjd - t_transit) < duration_days / 2.0
        out_transit_mask = ~in_transit_mask
        if np.any(in_transit_mask) and np.any(out_transit_mask):
            baseline = float(np.nanmedian(flux[out_transit_mask]))
            transit_flux = float(np.nanmedian(flux[in_transit_mask]))
            depth_frac = max(1e-8, baseline - transit_flux)
        else:
            depth_frac = 1e-3

    model = np.ones_like(flux, dtype=float)
    for t_transit in adjusted_transits:
        in_transit = np.abs(time_btjd - t_transit) < duration_days / 2.0
        model[in_transit] = 1.0 - float(depth_frac)

    valid = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0) & np.isfinite(time_btjd)
    if not np.any(valid):
        return 0.0, 0, []

    residuals_null = (flux[valid] - 1.0) / flux_err[valid]
    residuals_model = (flux[valid] - model[valid]) / flux_err[valid]
    chi2_null = float(np.sum(residuals_null**2))
    chi2_model = float(np.sum(residuals_model**2))
    score = max(0.0, chi2_null - chi2_model)

    n_transits = len(adjusted_transits)
    if n_transits > 0:
        score = score / np.sqrt(n_transits)

    per_transit_residuals: list[float] = []
    for t_transit in adjusted_transits:
        window_mask = np.abs(time_btjd - t_transit) < duration_days
        if np.any(window_mask & valid):
            per_transit_residuals.append(
                float(
                    np.sqrt(
                        np.mean((flux[window_mask & valid] - model[window_mask & valid]) ** 2)
                    )
                )
            )
    return score, n_transits, per_transit_residuals


def generate_track_grid(
    n_windows: int,
    *,
    max_offset_days: float,
    n_offset_steps: int,
    max_tracks: int,
    random_seed: int,
) -> list[list[float]]:
    """Generate a grid of per-window offsets (deterministic)."""
    if n_windows <= 0:
        return []
    if n_offset_steps <= 0:
        return [[0.0] * n_windows]

    if n_offset_steps == 1:
        single_window_offsets = [0.0]
    else:
        single_window_offsets = np.linspace(-max_offset_days, max_offset_days, n_offset_steps).tolist()

    total = n_offset_steps**n_windows
    if total <= max_tracks:
        return [list(t) for t in itertools.product(single_window_offsets, repeat=n_windows)]

    rng = np.random.default_rng(int(random_seed))
    tracks: list[list[float]] = [[0.0] * n_windows]
    for _ in range(max_tracks - 1):
        tracks.append([float(rng.choice(single_window_offsets)) for _ in range(n_windows)])
    return tracks


def generate_adaptive_track_grid(
    n_windows: int,
    *,
    max_offset_days: float,
    n_offset_steps: int,
    max_tracks: int,
    random_seed: int,
    prioritize_smooth: bool = True,
) -> list[list[float]]:
    """Generate an offset grid with physically-plausible patterns first."""
    if n_windows <= 0:
        return []

    tracks: list[list[float]] = [[0.0] * n_windows]
    if max_tracks <= 1:
        return tracks

    # Sinusoidal patterns (common TTV phenomenology)
    n_sinusoids = min(max_tracks // 4, 50)
    for i in range(n_sinusoids):
        ttv_period = max(2.0, n_windows / (1.0 + i * 0.5))
        phase = i * np.pi / 10.0
        tracks.append(
            [float(max_offset_days * np.sin(2 * np.pi * w / ttv_period + phase)) for w in range(n_windows)]
        )

    # Linear drift patterns
    n_linear = min(max_tracks // 8, 20)
    for i in range(n_linear):
        slope = float(max_offset_days) * (2 * i / max(n_linear - 1, 1) - 1)
        tracks.append(
            [float(slope * (w - n_windows / 2) / max(n_windows / 2, 1)) for w in range(n_windows)]
        )

    rng = np.random.default_rng(int(random_seed))
    single_offsets = np.linspace(-max_offset_days, max_offset_days, max(n_offset_steps, 2))
    n_remaining = max_tracks - len(tracks)
    for _ in range(max(n_remaining, 0)):
        if prioritize_smooth:
            offsets = [0.0]
            for _ in range(n_windows - 1):
                step = float(rng.uniform(-max_offset_days / 3, max_offset_days / 3))
                offsets.append(float(np.clip(offsets[-1] + step, -max_offset_days, max_offset_days)))
            tracks.append(offsets)
        else:
            tracks.append([float(rng.choice(single_offsets)) for _ in range(n_windows)])
    return tracks[:max_tracks]


def estimate_search_cost(
    time_btjd: np.ndarray,
    *,
    period_steps: int,
    n_offset_steps: int,
    max_tracks_per_period: int,
    budget: TTVSearchBudget,
    gap_threshold_days: float = 5.0,
) -> dict[str, int | float]:
    """Estimate compute cost for a given configuration."""
    windows = identify_observing_windows(time_btjd, gap_threshold_days=gap_threshold_days)
    n_windows = len(windows)
    period_steps_actual = min(int(period_steps), int(budget.max_period_evaluations))
    tracks_per_period = min(int(max_tracks_per_period), int(n_offset_steps**max(n_windows, 1)))
    theoretical_total = period_steps_actual * tracks_per_period
    budget_limited = min(theoretical_total, int(budget.max_track_hypotheses))
    estimated_seconds = budget_limited / 1000.0  # rough heuristic
    return {
        "n_windows": n_windows,
        "period_steps": period_steps_actual,
        "tracks_per_period": tracks_per_period,
        "theoretical_total_tracks": theoretical_total,
        "budget_limited_tracks": budget_limited,
        "estimated_seconds": min(float(estimated_seconds), float(budget.max_runtime_seconds)),
        "will_hit_budget": theoretical_total > budget.max_track_hypotheses,
    }


def run_ttv_track_search(
    time_btjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    period_span_fraction: float = 0.005,
    period_steps: int = 50,
    max_offset_days: float = 0.1,
    n_offset_steps: int = 5,
    max_tracks_per_period: int = 200,
    min_score_improvement: float = 2.0,
    gap_threshold_days: float = 5.0,
    budget: TTVSearchBudget | None = None,
    random_seed: int = 42,
) -> TTVTrackSearchResult:
    """Run a TTV track search around a known ephemeris (production entrypoint).

    Args:
        time_btjd: Time array (BTJD)
        flux: Normalized flux array (baseline ~1.0)
        flux_err: Flux uncertainties
        period_days: Base period in days
        t0_btjd: Base epoch in BTJD
        duration_hours: Transit duration in hours
        period_span_fraction: Search +/- this fraction around period_days
        period_steps: Number of period grid points
        max_offset_days: Max per-window timing offset (days)
        n_offset_steps: Grid resolution for per-window offsets
        max_tracks_per_period: Max offset tracks evaluated per period
        min_score_improvement: Minimum improvement over periodic to report
        gap_threshold_days: Gap size threshold for window splitting
        budget: Optional compute budget
        random_seed: Seed for deterministic grid sampling
    """
    if budget is None:
        budget = TTVSearchBudget()

    time_btjd = np.asarray(time_btjd, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    if not (time_btjd.shape == flux.shape == flux_err.shape):
        raise ValueError("time_btjd, flux, and flux_err must have the same shape")
    if period_days <= 0 or duration_hours <= 0:
        raise ValueError("period_days and duration_hours must be positive")

    start_time = time_module.time()
    windows = identify_observing_windows(time_btjd, gap_threshold_days=gap_threshold_days)
    n_windows = len(windows)
    if n_windows == 0:
        return TTVTrackSearchResult(
            candidates=[],
            n_periods_searched=0,
            n_tracks_evaluated=0,
            runtime_seconds=0.0,
            budget_exhausted=False,
            provenance={"error": "no_observing_windows"},
        )

    period_steps = min(int(period_steps), int(budget.max_period_evaluations))
    span = float(period_span_fraction)
    if span < 0:
        raise ValueError("period_span_fraction must be >= 0")
    p_min = float(period_days) * (1.0 - span)
    p_max = float(period_days) * (1.0 + span)
    periods = np.linspace(p_min, p_max, max(period_steps, 1))

    tracks_per_period = min(int(max_tracks_per_period), int(budget.max_track_hypotheses // max(period_steps, 1)))
    candidates: list[TTVTrackCandidate] = []
    n_periods_searched = 0
    n_tracks_evaluated = 0
    budget_exhausted = False

    for p in periods:
        elapsed = time_module.time() - start_time
        if elapsed > budget.max_runtime_seconds or n_tracks_evaluated >= budget.max_track_hypotheses:
            budget_exhausted = True
            break

        n_periods_searched += 1
        periodic_score, depth_frac, n_transits = score_periodic_model(
            time_btjd,
            flux,
            flux_err,
            period_days=float(p),
            t0_btjd=float(t0_btjd),
            duration_hours=float(duration_hours),
        )
        if n_transits < 2:
            continue

        tracks = generate_adaptive_track_grid(
            n_windows,
            max_offset_days=float(max_offset_days),
            n_offset_steps=int(n_offset_steps),
            max_tracks=int(tracks_per_period),
            random_seed=int(random_seed),
        )

        best_track: TTVTrackHypothesis | None = None
        best_residuals: list[float] = []
        alternatives: list[TTVTrackHypothesis] = []

        for track_idx, offsets in enumerate(tracks):
            n_tracks_evaluated += 1
            if n_tracks_evaluated >= budget.max_track_hypotheses:
                budget_exhausted = True
                break

            track_score, n_matched, residuals = score_track_hypothesis(
                time_btjd,
                flux,
                flux_err,
                period_days=float(p),
                t0_btjd=float(t0_btjd),
                duration_hours=float(duration_hours),
                window_boundaries=windows,
                window_offsets_days=offsets,
                depth_frac=depth_frac,
            )
            improvement = float(track_score - periodic_score)
            if improvement < min_score_improvement:
                continue

            hypothesis = TTVTrackHypothesis(
                track_id=f"ttv_track:p{p:.6f}:k{track_idx}",
                base_period_days=float(p),
                base_t0_btjd=float(t0_btjd),
                window_offsets_days=[float(x) for x in offsets],
                score=float(track_score),
                score_improvement=float(improvement),
                n_transits_matched=int(n_matched),
            )
            if best_track is None or hypothesis.score_improvement > best_track.score_improvement:
                if best_track is not None:
                    alternatives.append(best_track)
                best_track = hypothesis
                best_residuals = residuals
            else:
                alternatives.append(hypothesis)

        if best_track is not None:
            candidates.append(
                TTVTrackCandidate(
                    best_track=best_track,
                    alternative_tracks=sorted(alternatives, key=lambda t: -t.score_improvement)[:5],
                    periodic_score=float(periodic_score),
                    per_transit_residuals=[float(x) for x in best_residuals],
                    runtime_seconds=float(time_module.time() - start_time),
                    provenance={
                        "n_windows": n_windows,
                        "gap_threshold_days": gap_threshold_days,
                        "max_offset_days": max_offset_days,
                        "n_offset_steps": n_offset_steps,
                        "tracks_per_period": tracks_per_period,
                        "random_seed": random_seed,
                    },
                )
            )

    candidates.sort(key=lambda c: -c.best_track.score_improvement)
    runtime = float(time_module.time() - start_time)
    return TTVTrackSearchResult(
        candidates=candidates,
        n_periods_searched=int(n_periods_searched),
        n_tracks_evaluated=int(n_tracks_evaluated),
        runtime_seconds=runtime,
        budget_exhausted=bool(budget_exhausted),
        provenance={
            "period_days": float(period_days),
            "t0_btjd": float(t0_btjd),
            "duration_hours": float(duration_hours),
            "period_span_fraction": float(period_span_fraction),
            "period_steps": int(period_steps),
            "min_score_improvement": float(min_score_improvement),
            "budget": budget.to_dict(),
        },
    )

