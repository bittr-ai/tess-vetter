from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bittr_tess_vetter.validation.base import (
    get_in_transit_mask,
    get_out_of_transit_mask,
    measure_transit_depth,
)


@dataclass(frozen=True)
class PhaseCoverageResult:
    coverage_fraction: float
    transit_phase_coverage: float
    bins_with_data: int
    total_bins: int


def compute_depth_over_depth_err_snr(
    *,
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_fractional: float,
    buffer_factor: float = 3.0,
) -> tuple[float, float]:
    in_mask = get_in_transit_mask(time, period_days, t0_btjd, duration_hours)
    out_mask = get_out_of_transit_mask(
        time, period_days, t0_btjd, duration_hours, buffer_factor=buffer_factor
    )
    _, depth_err = measure_transit_depth(flux, in_mask, out_mask)
    depth_err_used = float(depth_err)
    snr = abs(float(depth_fractional)) / depth_err_used if depth_err_used > 0 else 0.0
    return snr, depth_err_used


def compute_phase_coverage(
    *,
    time: np.ndarray,
    period_days: float,
    t0_btjd: float,
    n_bins: int = 20,
    transit_bins: tuple[int, ...] = (0, 1, -1),
) -> PhaseCoverageResult:
    if not np.isfinite(period_days) or float(period_days) <= 0:
        raise ValueError(f"period_days must be positive and finite, got {period_days}")
    if not np.isfinite(t0_btjd):
        raise ValueError(f"t0_btjd must be finite, got {t0_btjd}")
    if int(n_bins) <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    phase = ((time - t0_btjd) / period_days) % 1.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts, _ = np.histogram(phase, bins=bin_edges)
    bins_with_data = int(np.sum(bin_counts > 0))
    coverage_fraction = float(bins_with_data / n_bins) if n_bins > 0 else 0.0

    if n_bins <= 0 or len(transit_bins) == 0:
        transit_phase_coverage = 0.0
    else:
        n_hit = 0
        for b in transit_bins:
            n_hit += 1 if bin_counts[b % n_bins] > 0 else 0
        transit_phase_coverage = float(n_hit / len(transit_bins))

    return PhaseCoverageResult(
        coverage_fraction=coverage_fraction,
        transit_phase_coverage=transit_phase_coverage,
        bins_with_data=bins_with_data,
        total_bins=int(n_bins),
    )
