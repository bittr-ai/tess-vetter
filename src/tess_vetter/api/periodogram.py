"""Periodogram API surface (host-facing).

This module provides a stable facade for periodogram operations so host
applications don't need to import from internal `compute.*` modules.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from tess_vetter.api.contracts import (
    callable_input_schema_from_signature,
    model_input_schema,
    model_output_schema,
)
from tess_vetter.api.references import (
    HIPPKE_HELLER_2019_TLS,
    LOMB_1976,
    SCARGLE_1982,
    cite,
    cites,
)
from tess_vetter.api.transit_model import compute_transit_model
from tess_vetter.compute.periodogram import (  # noqa: F401
    PerformancePreset,
    auto_periodogram,
    cluster_cross_sector_candidates,
    compute_bls_model,
    detect_sector_gaps,
    ls_periodogram,
    merge_candidates,
    search_planets,
    split_by_sectors,
    tls_search,
    tls_search_per_sector,
)
from tess_vetter.compute.periodogram import refine_period as _refine_period_compute
from tess_vetter.domain.detection import PeriodogramPeak, PeriodogramResult  # noqa: F401


class RunPeriodogramRequest(BaseModel):
    """Typed request payload for periodogram API boundary contracts."""

    model_config = ConfigDict(extra="forbid")

    time: list[float]
    flux: list[float]
    flux_err: list[float] | None = None
    min_period: float = 0.5
    max_period: float | None = None
    preset: Literal["fast", "thorough", "deep"] | str = "fast"
    method: Literal["tls", "ls", "auto"] = "auto"
    max_planets: int = 1
    data_ref: str = ""
    tic_id: int | None = None
    stellar_radius_rsun: float | None = None
    stellar_mass_msun: float | None = None
    use_threads: int | None = None
    per_sector: bool = True
    downsample_factor: int = 1


class RunPeriodogramResponse(BaseModel):
    """Typed response payload for periodogram API boundary contracts."""

    model_config = ConfigDict(extra="forbid")

    result: PeriodogramResult


class RefinePeriodRequest(BaseModel):
    """Typed request payload for period-refinement boundary contracts."""

    model_config = ConfigDict(extra="forbid")

    time: list[float]
    flux: list[float]
    flux_err: list[float] | None
    initial_period: float
    initial_duration: float
    refine_factor: float = 0.1
    n_refine: int = 100
    tic_id: int | None = None
    stellar_radius_rsun: float | None = None
    stellar_mass_msun: float | None = None


class RefinePeriodResponse(BaseModel):
    """Typed response payload for period-refinement boundary contracts."""

    model_config = ConfigDict(extra="forbid")

    period: float
    t0: float
    power: float


@cites(
    cite(HIPPKE_HELLER_2019_TLS, "Transit Least Squares (TLS) periodogram for transit detection"),
    cite(LOMB_1976, "Lomb periodogram for unevenly spaced time series"),
    cite(SCARGLE_1982, "Lomb-Scargle normalization/statistics for uneven sampling"),
)
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

    Notes:
        `method="tls"` is for transit detection.

        `method="ls"` is for rotation/variability (sinusoidal) detection and is
        not a transit-search algorithm. Its result will have
        ``signal_type="sinusoidal"`` and no meaningful transit duration/depth.
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


@cites(cite(HIPPKE_HELLER_2019_TLS, "TLS refinement around an initial period estimate"))
def refine_period(
    *,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None,
    initial_period: float,
    initial_duration: float,
    refine_factor: float = 0.1,
    n_refine: int = 100,
    tic_id: int | None = None,
    stellar_radius_rsun: float | None = None,
    stellar_mass_msun: float | None = None,
) -> tuple[float, float, float]:
    """Refine a period estimate with a constrained TLS search.

    This wrapper exists so host applications can call refinement without
    importing internal `compute.*` modules.
    """
    return _refine_period_compute(
        np.asarray(time, dtype=np.float64),
        np.asarray(flux, dtype=np.float64),
        np.asarray(flux_err, dtype=np.float64) if flux_err is not None else None,
        initial_period=float(initial_period),
        initial_duration=float(initial_duration),
        refine_factor=float(refine_factor),
        n_refine=int(n_refine),
        tic_id=tic_id,
        stellar_radius_rsun=stellar_radius_rsun,
        stellar_mass_msun=stellar_mass_msun,
    )


RUN_PERIODOGRAM_INPUT_SCHEMA = model_input_schema(RunPeriodogramRequest)
RUN_PERIODOGRAM_OUTPUT_SCHEMA = model_output_schema(RunPeriodogramResponse)
RUN_PERIODOGRAM_CALL_SCHEMA = callable_input_schema_from_signature(run_periodogram)
REFINE_PERIOD_INPUT_SCHEMA = model_input_schema(RefinePeriodRequest)
REFINE_PERIOD_OUTPUT_SCHEMA = model_output_schema(RefinePeriodResponse)
REFINE_PERIOD_CALL_SCHEMA = callable_input_schema_from_signature(refine_period)


__all__ = [
    "PerformancePreset",
    "PeriodogramPeak",
    "PeriodogramResult",
    "REFINE_PERIOD_CALL_SCHEMA",
    "REFINE_PERIOD_INPUT_SCHEMA",
    "REFINE_PERIOD_OUTPUT_SCHEMA",
    "RefinePeriodRequest",
    "RefinePeriodResponse",
    "RUN_PERIODOGRAM_CALL_SCHEMA",
    "RUN_PERIODOGRAM_INPUT_SCHEMA",
    "RUN_PERIODOGRAM_OUTPUT_SCHEMA",
    "RunPeriodogramRequest",
    "RunPeriodogramResponse",
    "run_periodogram",
    "compute_transit_model",
    "auto_periodogram",
    "cluster_cross_sector_candidates",
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
