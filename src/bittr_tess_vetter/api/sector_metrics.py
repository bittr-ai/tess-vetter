"""Sector-level ephemeris metrics for stitched or labeled time series.

This module provides **metrics-only** helpers for quantifying whether a transit
signal (fixed ephemeris) is consistently measurable across sectors/chunks.

These helpers are intentionally policy-free: they report scores/uncertainties
but do not apply pass/fail thresholds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.api.ephemeris_specificity import (
    SmoothTemplateConfig,
    score_fixed_period_numpy,
)
from bittr_tess_vetter.api.stitch import StitchedLC, _infer_cadence_seconds
from bittr_tess_vetter.validation.base import get_in_transit_mask, get_out_of_transit_mask
from bittr_tess_vetter.validation.sector_consistency import SectorMeasurement

SECTOR_MEASUREMENTS_SCHEMA_VERSION = 1


class V21SectorMeasurementRow(TypedDict):
    """Stable JSON row schema for V21 sector measurements."""

    sector: int
    depth_ppm: float
    depth_err_ppm: float
    duration_hours: float
    duration_err_hours: float
    n_transits: int
    shape_metric: float
    quality_weight: float


class V21SectorMeasurementsPayload(TypedDict):
    """Stable JSON payload schema for V21 sector measurements."""

    schema_version: int
    measurements: list[V21SectorMeasurementRow]


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


def _coerce_sector_measurement_row(row: Mapping[str, Any]) -> SectorMeasurement:
    if "sector" not in row or "depth_ppm" not in row or "depth_err_ppm" not in row:
        raise ValueError("each sector measurement row must include sector, depth_ppm, and depth_err_ppm")
    return SectorMeasurement(
        sector=int(row["sector"]),
        depth_ppm=float(row["depth_ppm"]),
        depth_err_ppm=float(row["depth_err_ppm"]),
        duration_hours=float(row.get("duration_hours", 0.0)),
        duration_err_hours=float(row.get("duration_err_hours", 0.0)),
        n_transits=int(row.get("n_transits", 0)),
        shape_metric=float(row.get("shape_metric", 0.0)),
        quality_weight=float(row.get("quality_weight", 1.0)),
    )


def serialize_v21_sector_measurements(
    measurements: list[SectorMeasurement] | list[SectorEphemerisMetrics],
) -> V21SectorMeasurementsPayload:
    """Serialize sector measurements to a stable, CLI-friendly V21 payload."""
    if not measurements:
        return {
            "schema_version": SECTOR_MEASUREMENTS_SCHEMA_VERSION,
            "measurements": [],
        }

    rows: list[V21SectorMeasurementRow] = []
    first = measurements[0]
    if isinstance(first, SectorEphemerisMetrics):
        for m in cast(list[SectorEphemerisMetrics], measurements):
            rows.append(
                {
                    "sector": int(m.sector),
                    "depth_ppm": float(m.depth_hat_ppm),
                    "depth_err_ppm": float(m.depth_sigma_ppm),
                    "duration_hours": 0.0,
                    "duration_err_hours": 0.0,
                    "n_transits": 0,
                    "shape_metric": 0.0,
                    "quality_weight": 1.0,
                }
            )
    else:
        for m in cast(list[SectorMeasurement], measurements):
            rows.append(
                {
                    "sector": int(m.sector),
                    "depth_ppm": float(m.depth_ppm),
                    "depth_err_ppm": float(m.depth_err_ppm),
                    "duration_hours": float(m.duration_hours),
                    "duration_err_hours": float(m.duration_err_hours),
                    "n_transits": int(m.n_transits),
                    "shape_metric": float(m.shape_metric),
                    "quality_weight": float(m.quality_weight),
                }
            )

    return {
        "schema_version": SECTOR_MEASUREMENTS_SCHEMA_VERSION,
        "measurements": rows,
    }


def deserialize_v21_sector_measurements(
    payload: str | Mapping[str, Any],
) -> list[SectorMeasurement]:
    """Deserialize V21 sector-measurement payload from JSON text or mapping."""
    data: Mapping[str, Any]
    if isinstance(payload, str):
        loaded = json.loads(payload)
        if not isinstance(loaded, dict):
            raise ValueError("V21 sector measurements payload must be a JSON object")
        data = cast(Mapping[str, Any], loaded)
    else:
        data = payload

    version = int(data.get("schema_version", -1))
    if version != SECTOR_MEASUREMENTS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported V21 sector measurements schema_version: {version} "
            f"(expected {SECTOR_MEASUREMENTS_SCHEMA_VERSION})"
        )

    raw_rows = data.get("measurements")
    if not isinstance(raw_rows, list):
        raise ValueError("V21 sector measurements payload must include a list 'measurements'")

    rows: list[SectorMeasurement] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("each sector measurement must be an object")
        rows.append(_coerce_sector_measurement_row(raw))
    return rows


__all__ = [
    "SECTOR_MEASUREMENTS_SCHEMA_VERSION",
    "SectorEphemerisMetrics",
    "V21SectorMeasurementRow",
    "V21SectorMeasurementsPayload",
    "compute_sector_ephemeris_metrics",
    "compute_sector_ephemeris_metrics_from_stitched",
    "serialize_v21_sector_measurements",
    "deserialize_v21_sector_measurements",
]
