"""Light curve cache-oriented types (host-facing).

This module provides a small contract for representing cached light curve data
and referencing it from higher-level tool surfaces.

It intentionally includes:
- `LightCurveData`: the internal ndarray container (immutable arrays)
- `make_data_ref`: deterministic cache key helper
- `LightCurveRef`: metadata-only reference (no raw arrays)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from bittr_tess_vetter.domain.lightcurve import LightCurveData, make_data_ref


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LightCurveRef(FrozenModel):
    """Metadata-only light curve reference suitable for API responses."""

    data_ref: str
    tic_id: int
    sector: int
    n_points: int
    n_valid: int
    duration_days: float
    cadence_seconds: float
    median_flux: float
    flux_std: float
    gap_fraction: float
    quality_flags_present: list[int]

    @classmethod
    def from_data(cls, data: LightCurveData, flux_type: str = "pdcsap") -> "LightCurveRef":
        return cls(
            data_ref=make_data_ref(data.tic_id, data.sector, flux_type),
            tic_id=data.tic_id,
            sector=data.sector,
            n_points=data.n_points,
            n_valid=data.n_valid,
            duration_days=data.duration_days,
            cadence_seconds=data.cadence_seconds,
            median_flux=data.median_flux,
            flux_std=data.flux_std,
            gap_fraction=data.gap_fraction,
            quality_flags_present=data.quality_flags_present,
        )


__all__ = ["LightCurveData", "LightCurveRef", "make_data_ref"]

