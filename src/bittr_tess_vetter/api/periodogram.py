"""Periodogram API surface (host-facing).

This module provides a stable facade for periodogram operations so host
applications don't need to import from internal `compute.*` modules.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from bittr_tess_vetter.compute.periodogram import (  # noqa: F401
    PerformancePreset,
    auto_periodogram,
    compute_bls_model,
    detect_sector_gaps,
    ls_periodogram,
    merge_candidates,
    refine_period,
    search_planets,
    split_by_sectors,
    tls_search,
    tls_search_per_sector,
)
from bittr_tess_vetter.domain.detection import PeriodogramPeak, PeriodogramResult  # noqa: F401


def run_periodogram(
    *,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    min_period: float = 0.5,
    max_period: float | None = None,
    preset: Literal["fast", "thorough", "deep"] | str = "fast",
    method: Literal["tls", "ls", "auto"] = "auto",
    max_planets: int = 1,
    data_ref: str = "",
    tic_id: int | None = None,
    stellar_radius_rsun: float | None = None,
    stellar_mass_msun: float | None = None,
    use_threads: int | None = None,
    per_sector: bool = True,
    downsample_factor: int = 1,
) -> PeriodogramResult:
    """Host-facing wrapper for periodogram analysis.

    This is a convenience wrapper around `auto_periodogram` matching the
    host/MCP style naming.
    """
    return auto_periodogram(
        time=np.asarray(time, dtype=np.float64),
        flux=np.asarray(flux, dtype=np.float64),
        flux_err=np.asarray(flux_err, dtype=np.float64) if flux_err is not None else None,
        min_period=float(min_period),
        max_period=float(max_period) if max_period is not None else None,
        preset=str(preset),
        method=method,  # type: ignore[arg-type]
        n_peaks=5,
        data_ref=str(data_ref),
        tic_id=tic_id,
        stellar_radius_rsun=stellar_radius_rsun,
        stellar_mass_msun=stellar_mass_msun,
        max_planets=int(max_planets),
        use_threads=use_threads,
        per_sector=per_sector,
        downsample_factor=int(downsample_factor),
    )


def compute_transit_model(
    *,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    depth_ppm: float,
) -> dict[str, float | int]:
    """Compute a simple box transit model and fit diagnostics.

    Returns metrics only (no model array) for host/MCP use.
    """
    time_arr = np.asarray(time, dtype=np.float64)
    flux_arr = np.asarray(flux, dtype=np.float64)
    flux_err_arr = np.asarray(flux_err, dtype=np.float64)

    depth_fractional = float(depth_ppm) / 1_000_000.0
    model = compute_bls_model(
        time=time_arr,
        period=float(period),
        t0=float(t0),
        duration_hours=float(duration_hours),
        depth=float(depth_fractional),
    )

    finite_mask = np.isfinite(time_arr) & np.isfinite(flux_arr) & np.isfinite(flux_err_arr)
    n_finite = int(np.sum(finite_mask))
    if n_finite <= 4:
        raise ValueError("Insufficient finite points to compute diagnostics (need >4).")

    residuals = flux_arr[finite_mask] - model[finite_mask]
    rms = float(np.sqrt(np.mean(residuals**2)))
    chi2 = float(np.sum((residuals / flux_err_arr[finite_mask]) ** 2))
    reduced_chi2 = float(chi2 / (n_finite - 4))

    in_transit = model < (1.0 - depth_fractional / 2)
    n_in_transit = int(np.sum(in_transit & finite_mask))

    return {
        "period": float(period),
        "t0": float(t0),
        "duration_hours": float(duration_hours),
        "depth_ppm": float(depth_ppm),
        "rms_residual": float(rms),
        "chi2": float(chi2),
        "reduced_chi2": float(reduced_chi2),
        "n_in_transit": int(n_in_transit),
    }


__all__ = [
    "PerformancePreset",
    "PeriodogramPeak",
    "PeriodogramResult",
    "run_periodogram",
    "compute_transit_model",
    "auto_periodogram",
    "compute_bls_model",
    "detect_sector_gaps",
    "ls_periodogram",
    "merge_candidates",
    "refine_period",
    "search_planets",
    "split_by_sectors",
    "tls_search",
    "tls_search_per_sector",
]
