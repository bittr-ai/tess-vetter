"""Alias and harmonic diagnostics (metrics-only).

This module provides lightweight, deterministic computations to help detect
period aliases/harmonics (e.g., EB P/2 vs P) using only a light curve.

Design notes:
- Implementation is open-safe: no curated alias family lists are included.
- Host apps can choose thresholds/policy; this module exposes both raw scores
  and a simple ratio-based classification helper.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

AliasClass = Literal["ALIAS_STRONG", "ALIAS_WEAK", "NONE"]

HARMONIC_LABELS = ("P", "P/2", "2P", "P/3", "3P")


@dataclass
class HarmonicScore:
    """Score at a specific harmonic period."""

    harmonic: str
    period: float
    score: float
    depth_ppm: float
    duration_hours: float | None


@dataclass
class PhaseShiftEvent:
    """A significant event at a phase other than the primary transit."""

    phase: float
    significance: float
    depth_ppm: float
    n_points: int


def _compute_box_depth(
    time: NDArray[np.floating[Any]],
    flux: NDArray[np.floating[Any]],
    flux_err: NDArray[np.floating[Any]],
    period: float,
    t0: float,
    duration_hours: float,
) -> tuple[float, float, float]:
    """Compute a simple box depth + significance at a given period."""
    duration_days = duration_hours / 24.0

    phase = ((time - t0) % period) / period
    half_dur_phase = (duration_days / period) / 2.0
    in_transit = (phase < half_dur_phase) | (phase > 1.0 - half_dur_phase)

    # Avoid secondary at phase ~0.5 for the out-of-transit baseline.
    out_transit = ((phase > 0.15) & (phase < 0.35)) | ((phase > 0.65) & (phase < 0.85))

    n_in = int(np.sum(in_transit))
    n_out = int(np.sum(out_transit))
    if n_in < 3 or n_out < 10:
        return 0.0, 0.0, 0.0

    flux_in = flux[in_transit]
    flux_out = flux[out_transit]
    err_in = flux_err[in_transit]
    err_out = flux_err[out_transit]

    w_in = 1.0 / (err_in**2 + 1e-10)
    w_out = 1.0 / (err_out**2 + 1e-10)

    mean_in = float(np.sum(flux_in * w_in) / np.sum(w_in))
    mean_out = float(np.sum(flux_out * w_out) / np.sum(w_out))

    depth = mean_out - mean_in
    depth_ppm = depth * 1e6

    var_in = 1.0 / float(np.sum(w_in))
    var_out = 1.0 / float(np.sum(w_out))
    depth_err = float(np.sqrt(var_in + var_out))
    depth_err_ppm = depth_err * 1e6

    significance = depth / depth_err if depth_err > 0 else 0.0
    return float(depth_ppm), float(depth_err_ppm), float(significance)


def _harmonic_period(base_period: float, harmonic: str) -> float:
    if harmonic == "P":
        return base_period
    if harmonic == "P/2":
        return base_period / 2.0
    if harmonic == "2P":
        return base_period * 2.0
    if harmonic == "P/3":
        return base_period / 3.0
    if harmonic == "3P":
        return base_period * 3.0
    raise ValueError(f"Unknown harmonic: {harmonic}")


def compute_harmonic_scores(
    time: NDArray[np.floating[Any]],
    flux: NDArray[np.floating[Any]],
    flux_err: NDArray[np.floating[Any]],
    base_period: float,
    base_t0: float,
    *,
    harmonics: list[str] | None = None,
    duration_hours: float = 2.0,
    scorer: Callable[..., tuple[float, float, float]] | None = None,
) -> list[HarmonicScore]:
    """Compute detection scores at harmonic periods."""
    if harmonics is None:
        harmonics = list(HARMONIC_LABELS)

    if scorer is None:
        scorer = _compute_box_depth

    scores: list[HarmonicScore] = []
    for harmonic in harmonics:
        period = _harmonic_period(base_period, harmonic)

        # Scale duration modestly with period to keep masks comparable.
        scaled_duration = duration_hours
        if harmonic in ("P/2", "P/3", "2P", "3P"):
            scaled_duration = duration_hours * (period / base_period) ** 0.5
        scaled_duration = float(max(0.5, min(scaled_duration, 12.0)))

        depth_ppm, _depth_err_ppm, significance = scorer(
            time, flux, flux_err, period, base_t0, scaled_duration
        )

        scores.append(
            HarmonicScore(
                harmonic=harmonic,
                period=float(period),
                score=float(significance),
                depth_ppm=float(depth_ppm),
                duration_hours=float(scaled_duration),
            )
        )

    return scores


def detect_phase_shift_events(
    time: NDArray[np.floating[Any]],
    flux: NDArray[np.floating[Any]],
    flux_err: NDArray[np.floating[Any]],
    period: float,
    t0: float,
    *,
    n_phase_bins: int = 10,
    significance_threshold: float = 3.0,
) -> list[PhaseShiftEvent]:
    """Find significant events at phases other than the primary transit."""
    _ = flux_err  # noise is estimated from a global baseline region for simplicity

    phase = ((time - t0) % period) / period

    events: list[PhaseShiftEvent] = []

    baseline_mask = ((phase > 0.1) & (phase < 0.4)) | ((phase > 0.6) & (phase < 0.9))
    if int(np.sum(baseline_mask)) < 20:
        return events

    baseline_flux = float(np.median(flux[baseline_mask]))
    baseline_std = float(np.std(flux[baseline_mask]))
    if baseline_std <= 0:
        return events

    bin_edges = np.linspace(0, 1, n_phase_bins + 1)
    for i in range(n_phase_bins):
        bin_center = float((bin_edges[i] + bin_edges[i + 1]) / 2)

        # Skip the primary transit bin (phase ~ 0)
        if bin_center < 0.05 or bin_center > 0.95:
            continue

        in_bin = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        n_points = int(np.sum(in_bin))
        if n_points < 3:
            continue

        bin_flux = float(np.mean(flux[in_bin]))
        bin_err = baseline_std / float(np.sqrt(n_points))

        depth = baseline_flux - bin_flux
        significance = depth / bin_err if bin_err > 0 else 0.0

        if significance >= significance_threshold:
            events.append(
                PhaseShiftEvent(
                    phase=bin_center,
                    significance=float(significance),
                    depth_ppm=float(depth * 1e6),
                    n_points=n_points,
                )
            )

    return events


def compute_secondary_significance(
    time: NDArray[np.floating[Any]],
    flux: NDArray[np.floating[Any]],
    flux_err: NDArray[np.floating[Any]],
    period: float,
    t0: float,
    duration_hours: float,
) -> float:
    """Compute significance of secondary eclipse at phase 0.5 (sigma)."""
    _ = flux_err  # use OOT scatter for a simple, robust noise estimate

    duration_days = duration_hours / 24.0
    phase = ((time - t0) % period) / period

    half_dur_phase = (duration_days / period) / 2.0
    in_secondary = (phase > 0.5 - half_dur_phase) & (phase < 0.5 + half_dur_phase)
    out_transit = ((phase > 0.15) & (phase < 0.35)) | ((phase > 0.65) & (phase < 0.85))

    n_sec = int(np.sum(in_secondary))
    n_out = int(np.sum(out_transit))
    if n_sec < 3 or n_out < 10:
        return 0.0

    mean_secondary = float(np.mean(flux[in_secondary]))
    mean_out = float(np.mean(flux[out_transit]))
    std_out = float(np.std(flux[out_transit]))
    if std_out <= 0:
        return 0.0

    depth = mean_out - mean_secondary
    err = std_out / float(np.sqrt(n_sec))
    significance = depth / err if err > 0 else 0.0
    return float(max(0.0, significance))


def classify_alias(
    harmonic_scores: list[HarmonicScore],
    base_score: float,
    *,
    strong_ratio_threshold: float = 1.5,
    weak_ratio_threshold: float = 1.1,
) -> tuple[AliasClass, str, float]:
    """Classify alias status based on harmonic scores."""
    if not harmonic_scores:
        return "NONE", "P", 1.0

    best_other: HarmonicScore | None = None
    for hs in harmonic_scores:
        if hs.harmonic == "P":
            continue
        if best_other is None or hs.score > best_other.score:
            best_other = hs

    if best_other is None:
        return "NONE", "P", 1.0

    if base_score <= 0:
        if best_other.score > 0:
            return "ALIAS_STRONG", best_other.harmonic, float("inf")
        return "NONE", "P", 1.0

    ratio = float(best_other.score / base_score)
    if ratio >= strong_ratio_threshold:
        return "ALIAS_STRONG", best_other.harmonic, ratio
    if ratio >= weak_ratio_threshold:
        return "ALIAS_WEAK", best_other.harmonic, ratio
    return "NONE", "P", ratio


__all__ = [
    "AliasClass",
    "HarmonicScore",
    "PhaseShiftEvent",
    "HARMONIC_LABELS",
    "classify_alias",
    "compute_harmonic_scores",
    "compute_secondary_significance",
    "detect_phase_shift_events",
]
