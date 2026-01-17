"""Sector-level ephemeris metrics for stitched or labeled time series.

This module provides **metrics-only** helpers for quantifying whether a transit
signal (fixed ephemeris) is consistently measurable across sectors/chunks.

These helpers are intentionally policy-free: they report scores/uncertainties
but do not apply pass/fail thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.api.ephemeris_specificity import (
    SmoothTemplateConfig,
    score_fixed_period_numpy,
)
from bittr_tess_vetter.api.stitch import StitchedLC, _infer_cadence_seconds
from bittr_tess_vetter.validation.base import get_in_transit_mask, get_out_of_transit_mask


@dataclass(frozen=True)
class SectorEphemerisMetrics:
    """Per-sector fixed-ephemeris diagnostic metrics."""

    sector: int
    n_total: int
    n_valid: int
    time_start_btjd: float
    time_end_btjd: float
    duration_days: float
    cadence_seconds: float
    n_in_transit: int
    n_out_of_transit: int
    depth_hat_ppm: float
    depth_sigma_ppm: float
    score: float
    flux_mad_ppm: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sector": int(self.sector),
            "n_total": int(self.n_total),
            "n_valid": int(self.n_valid),
            "time_start_btjd": float(self.time_start_btjd),
            "time_end_btjd": float(self.time_end_btjd),
            "duration_days": float(self.duration_days),
            "cadence_seconds": float(self.cadence_seconds),
            "n_in_transit": int(self.n_in_transit),
            "n_out_of_transit": int(self.n_out_of_transit),
            "depth_hat_ppm": float(self.depth_hat_ppm),
            "depth_sigma_ppm": float(self.depth_sigma_ppm),
            "score": float(self.score),
            "flux_mad_ppm": float(self.flux_mad_ppm),
        }


def _mad_ppm(flux: NDArray[np.float64]) -> float:
    flux = np.asarray(flux, dtype=np.float64)
    finite = np.isfinite(flux)
    if not np.any(finite):
        return float("nan")
    med = float(np.nanmedian(flux[finite]))
    mad = float(np.nanmedian(np.abs(flux[finite] - med)))
    return float(mad * 1e6)


def compute_sector_ephemeris_metrics(
    *,
    time: NDArray[np.floating[Any]],
    flux: NDArray[np.floating[Any]],
    flux_err: NDArray[np.floating[Any]],
    sector: NDArray[np.integer[Any]],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    template_config: SmoothTemplateConfig | None = None,
    oot_buffer_factor: float = 3.0,
) -> list[SectorEphemerisMetrics]:
    """Compute per-sector fixed-ephemeris metrics from labeled arrays.

    Args:
        time: BTJD time array.
        flux: normalized flux array (median ~1.0).
        flux_err: flux uncertainties.
        sector: integer sector label per cadence (same length as time).
        period_days: candidate period.
        t0_btjd: mid-transit reference epoch (BTJD).
        duration_hours: transit duration.
        template_config: Smooth-template config used for score/depth estimation.
        oot_buffer_factor: out-of-transit mask buffer multiplier (technical parameter).

    Returns:
        List of metrics objects, sorted by sector id.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)
    sector = np.asarray(sector, dtype=np.int32)

    if time.shape != flux.shape or time.shape != flux_err.shape or time.shape != sector.shape:
        raise ValueError("time/flux/flux_err/sector must have the same shape")

    cfg = template_config or SmoothTemplateConfig()

    out: list[SectorEphemerisMetrics] = []
    for sec in sorted({int(s) for s in sector}):
        m = sector == int(sec)
        t = time[m]
        f = flux[m]
        e = flux_err[m]

        n_total = int(len(t))
        finite = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
        t = t[finite]
        f = f[finite]
        e = e[finite]
        n_valid = int(len(t))

        if n_valid == 0:
            out.append(
                SectorEphemerisMetrics(
                    sector=int(sec),
                    n_total=n_total,
                    n_valid=0,
                    time_start_btjd=float("nan"),
                    time_end_btjd=float("nan"),
                    duration_days=float("nan"),
                    cadence_seconds=float("nan"),
                    n_in_transit=0,
                    n_out_of_transit=0,
                    depth_hat_ppm=float("nan"),
                    depth_sigma_ppm=float("nan"),
                    score=float("nan"),
                    flux_mad_ppm=float("nan"),
                )
            )
            continue

        sort_idx = np.argsort(t)
        t = t[sort_idx]
        f = f[sort_idx]
        e = e[sort_idx]

        in_mask = get_in_transit_mask(t, float(period_days), float(t0_btjd), float(duration_hours))
        out_mask = get_out_of_transit_mask(
            t,
            float(period_days),
            float(t0_btjd),
            float(duration_hours),
            buffer_factor=float(oot_buffer_factor),
        )

        res = score_fixed_period_numpy(
            time=t,
            flux=f,
            flux_err=e,
            period_days=float(period_days),
            t0_btjd=float(t0_btjd),
            duration_hours=float(duration_hours),
            config=cfg,
        )

        cadence_seconds = _infer_cadence_seconds(
            t, np.full(len(t), int(sec), dtype=np.int32), default_seconds=120.0
        )

        out.append(
            SectorEphemerisMetrics(
                sector=int(sec),
                n_total=n_total,
                n_valid=n_valid,
                time_start_btjd=float(np.min(t)),
                time_end_btjd=float(np.max(t)),
                duration_days=float(np.max(t) - np.min(t)),
                cadence_seconds=float(cadence_seconds),
                n_in_transit=int(np.sum(in_mask)),
                n_out_of_transit=int(np.sum(out_mask)),
                depth_hat_ppm=float(res.depth_hat * 1e6),
                depth_sigma_ppm=float(res.depth_sigma * 1e6),
                score=float(res.score),
                flux_mad_ppm=float(_mad_ppm(f)),
            )
        )

    return out


def compute_sector_ephemeris_metrics_from_stitched(
    *,
    stitched: StitchedLC,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    template_config: SmoothTemplateConfig | None = None,
    oot_buffer_factor: float = 3.0,
) -> list[SectorEphemerisMetrics]:
    """Convenience wrapper for :class:`~bittr_tess_vetter.api.stitch.StitchedLC`."""
    return compute_sector_ephemeris_metrics(
        time=stitched.time,
        flux=stitched.flux,
        flux_err=stitched.flux_err,
        sector=stitched.sector,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        template_config=template_config,
        oot_buffer_factor=oot_buffer_factor,
    )


__all__ = [
    "SectorEphemerisMetrics",
    "compute_sector_ephemeris_metrics",
    "compute_sector_ephemeris_metrics_from_stitched",
]
