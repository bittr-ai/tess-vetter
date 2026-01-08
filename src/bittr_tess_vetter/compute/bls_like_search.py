from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BlsLikeSearchResult:
    method: str
    best_period_days: float
    best_t0_btjd: float
    best_duration_hours: float
    score: float
    runtime_seconds: float
    notes: dict[str, Any]


@dataclass(frozen=True)
class BlsLikeCandidate:
    period_days: float
    t0_btjd: float
    duration_hours: float
    score: float


def _rolling_mean_circular(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be >= 1")
    if window == 1:
        return x.copy()
    pad = window - 1
    x2 = np.concatenate([x, x[:pad]])
    c = np.cumsum(np.insert(x2, 0, 0.0))
    out = (c[window:] - c[:-window]) / window
    return out[: x.size]


def _phase_bin_means(time_btjd: np.ndarray, flux: np.ndarray, period: float, nbins: int) -> tuple[np.ndarray, np.ndarray]:
    phase = ((time_btjd % period) / period) * nbins
    bins = np.floor(phase).astype(np.int64)
    bins = np.clip(bins, 0, nbins - 1)
    counts = np.bincount(bins, minlength=nbins).astype(np.int64)
    sums = np.bincount(bins, weights=flux, minlength=nbins).astype(np.float64)
    means = np.zeros(nbins, dtype=np.float64)
    m = counts > 0
    means[m] = sums[m] / counts[m]
    means[~m] = np.nan
    return means, counts


def _bls_score_from_binned_flux(
    binned_flux: np.ndarray,
    binned_counts: np.ndarray,
    *,
    duration_bins: int,
) -> tuple[float, int]:
    valid = binned_counts > 0
    if int(valid.sum()) < max(5, duration_bins + 2):
        return float("-inf"), 0

    y = binned_flux.copy()
    y[~valid] = np.nan
    overall = float(np.nanmedian(y))

    rm = _rolling_mean_circular(np.nan_to_num(y, nan=overall), duration_bins)
    min_idx = int(np.argmin(rm))
    min_val = float(rm[min_idx])

    depth = overall - min_val
    scatter = float(np.nanstd(y[valid])) + 1e-12
    score = depth / scatter
    return float(score), min_idx


def bls_like_search_numpy(
    *,
    time_btjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None,
    period_grid: np.ndarray,
    duration_hours_grid: list[float],
    nbins: int = 200,
    local_refine_steps: int = 11,
    local_refine_width_phase: float = 0.02,
) -> BlsLikeSearchResult:
    start = time.perf_counter()

    best = {
        "score": float("-inf"),
        "period": float("nan"),
        "t0": float("nan"),
        "dur_h": float("nan"),
    }

    if flux_err is not None:
        w = 1.0 / np.maximum(flux_err * flux_err, 1e-12)
    else:
        w = None

    for period in period_grid:
        bmeans, bcounts = _phase_bin_means(time_btjd, flux, float(period), nbins)
        for dur_h in duration_hours_grid:
            dur_days = float(dur_h) / 24.0
            dur_bins = max(1, int(round((dur_days / float(period)) * nbins)))
            score, min_bin = _bls_score_from_binned_flux(bmeans, bcounts, duration_bins=dur_bins)
            if not np.isfinite(score):
                continue

            t0_phase = ((min_bin + 0.5 * dur_bins) / nbins) % 1.0
            t0_mod = t0_phase * float(period)
            t_mid = float(np.nanmedian(time_btjd))
            k = float(np.round((t_mid - t0_mod) / float(period)))
            t0_guess = float(t0_mod + k * float(period))

            scan_phases = np.linspace(-local_refine_width_phase, local_refine_width_phase, local_refine_steps)
            best_local = float("-inf")
            best_t0 = t0_guess

            duration_days = dur_days
            for dphi in scan_phases:
                t0_try = float(t0_guess + dphi * float(period))
                phase = ((time_btjd - t0_try) / float(period)) % 1.0
                phase = np.minimum(phase, 1.0 - phase)
                in_tr = phase * float(period) <= (duration_days / 2.0)
                template = in_tr.astype(np.float64)

                y = 1.0 - flux
                if w is None:
                    denom = float(np.sum(template * template) + 1e-12)
                    depth_hat = float(np.sum(template * y) / denom)
                    depth_sigma = float(math.sqrt(1.0 / denom))
                else:
                    denom = float(np.sum(w * template * template) + 1e-12)
                    depth_hat = float(np.sum(w * template * y) / denom)
                    depth_sigma = float(math.sqrt(1.0 / denom))

                z = depth_hat / max(depth_sigma, 1e-12)
                if z > best_local:
                    best_local = float(z)
                    best_t0 = t0_try

            if best_local > best["score"]:
                best["score"] = float(best_local)
                best["period"] = float(period)
                best["t0"] = float(best_t0)
                best["dur_h"] = float(dur_h)

    end = time.perf_counter()
    return BlsLikeSearchResult(
        method="numpy_bls_like",
        best_period_days=float(best["period"]),
        best_t0_btjd=float(best["t0"]),
        best_duration_hours=float(best["dur_h"]),
        score=float(best["score"]),
        runtime_seconds=float(end - start),
        notes={
            "nbins": int(nbins),
            "local_refine_steps": int(local_refine_steps),
            "local_refine_width_phase": float(local_refine_width_phase),
            "n_periods": int(len(period_grid)),
            "n_durations": int(len(duration_hours_grid)),
        },
    )


def bls_like_search_numpy_top_k(
    *,
    time_btjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None,
    period_grid: np.ndarray,
    duration_hours_grid: list[float],
    top_k: int = 10,
    nbins: int = 200,
    local_refine_steps: int = 11,
    local_refine_width_phase: float = 0.02,
) -> tuple[BlsLikeSearchResult, list[BlsLikeCandidate]]:
    """Return best result and the top-k period candidates (descending by score)."""
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    start = time.perf_counter()

    if flux_err is not None:
        w = 1.0 / np.maximum(flux_err * flux_err, 1e-12)
    else:
        w = None

    candidates: list[BlsLikeCandidate] = []

    for period in period_grid:
        best_local = float("-inf")
        best_t0 = float("nan")
        best_dur_h = float("nan")

        bmeans, bcounts = _phase_bin_means(time_btjd, flux, float(period), nbins)
        for dur_h in duration_hours_grid:
            dur_days = float(dur_h) / 24.0
            dur_bins = max(1, int(round((dur_days / float(period)) * nbins)))
            score, min_bin = _bls_score_from_binned_flux(bmeans, bcounts, duration_bins=dur_bins)
            if not np.isfinite(score):
                continue

            t0_phase = ((min_bin + 0.5 * dur_bins) / nbins) % 1.0
            t0_mod = t0_phase * float(period)
            t_mid = float(np.nanmedian(time_btjd))
            k = float(np.round((t_mid - t0_mod) / float(period)))
            t0_guess = float(t0_mod + k * float(period))

            scan_phases = np.linspace(-local_refine_width_phase, local_refine_width_phase, local_refine_steps)
            duration_days = dur_days
            for dphi in scan_phases:
                t0_try = float(t0_guess + dphi * float(period))
                phase = ((time_btjd - t0_try) / float(period)) % 1.0
                phase = np.minimum(phase, 1.0 - phase)
                in_tr = phase * float(period) <= (duration_days / 2.0)
                template = in_tr.astype(np.float64)

                y = 1.0 - flux
                if w is None:
                    denom = float(np.sum(template * template) + 1e-12)
                    depth_hat = float(np.sum(template * y) / denom)
                    depth_sigma = float(math.sqrt(1.0 / denom))
                else:
                    denom = float(np.sum(w * template * template) + 1e-12)
                    depth_hat = float(np.sum(w * template * y) / denom)
                    depth_sigma = float(math.sqrt(1.0 / denom))

                z = depth_hat / max(depth_sigma, 1e-12)
                if z > best_local:
                    best_local = float(z)
                    best_t0 = t0_try
                    best_dur_h = float(dur_h)

        if np.isfinite(best_local):
            candidates.append(
                BlsLikeCandidate(
                    period_days=float(period),
                    t0_btjd=float(best_t0),
                    duration_hours=float(best_dur_h),
                    score=float(best_local),
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[: min(int(top_k), len(candidates))]

    end = time.perf_counter()

    best = candidates[0] if candidates else BlsLikeCandidate(float("nan"), float("nan"), float("nan"), float("-inf"))
    best_res = BlsLikeSearchResult(
        method="numpy_bls_like",
        best_period_days=float(best.period_days),
        best_t0_btjd=float(best.t0_btjd),
        best_duration_hours=float(best.duration_hours),
        score=float(best.score),
        runtime_seconds=float(end - start),
        notes={
            "nbins": int(nbins),
            "local_refine_steps": int(local_refine_steps),
            "local_refine_width_phase": float(local_refine_width_phase),
            "n_periods": int(len(period_grid)),
            "n_durations": int(len(duration_hours_grid)),
            "top_k": int(top_k),
        },
    )
    return best_res, candidates


__all__ = [
    "BlsLikeCandidate",
    "BlsLikeSearchResult",
    "bls_like_search_numpy",
    "bls_like_search_numpy_top_k",
]
