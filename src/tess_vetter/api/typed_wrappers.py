"""Additive typed wrapper entrypoints for selected public API operations.

These wrappers provide explicit Pydantic input/output contracts around existing
API functions without changing underlying behavior.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tess_vetter.api.periodogram import run_periodogram as _run_periodogram
from tess_vetter.api.pipeline import describe_checks as _describe_checks
from tess_vetter.api.pipeline import list_checks as _list_checks
from tess_vetter.domain.detection import PeriodogramPeak, PeriodogramResult
from tess_vetter.validation.registry import CheckRegistry


class CheckRequirementsModel(BaseModel):
    """Structured requirements for a vetting check entry."""

    model_config = ConfigDict(extra="forbid")

    needs_tpf: bool
    needs_network: bool
    needs_ra_dec: bool
    needs_tic_id: bool
    needs_stellar: bool
    optional_deps: list[str]


class CheckInfoModel(BaseModel):
    """Structured check metadata returned by `list_checks_typed`."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    tier: str
    requirements: CheckRequirementsModel
    citations: list[str]


class DescribeChecksModel(BaseModel):
    """Typed description payload returned by `describe_checks_typed`."""

    model_config = ConfigDict(extra="forbid")

    description: str


class RunPeriodogramInputModel(BaseModel):
    """Typed input contract for `run_periodogram_typed`."""

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


class RunPeriodogramOutputModel(BaseModel):
    """Typed output contract for `run_periodogram_typed`."""

    model_config = ConfigDict(extra="forbid")

    result: PeriodogramResult


class ListChecksOutputModel(BaseModel):
    """Typed output contract for `list_checks_typed`."""

    model_config = ConfigDict(extra="forbid")

    checks: list[CheckInfoModel] = Field(default_factory=list)


def list_checks_typed(registry: CheckRegistry | None = None) -> ListChecksOutputModel:
    """Typed wrapper around `tess_vetter.api.pipeline.list_checks`."""
    raw = _list_checks(registry)
    return ListChecksOutputModel(checks=[CheckInfoModel.model_validate(item) for item in raw])


def describe_checks_typed(registry: CheckRegistry | None = None) -> DescribeChecksModel:
    """Typed wrapper around `tess_vetter.api.pipeline.describe_checks`."""
    return DescribeChecksModel(description=_describe_checks(registry))


def run_periodogram_typed(**kwargs: Any) -> RunPeriodogramOutputModel:
    """Typed wrapper around `tess_vetter.api.periodogram.run_periodogram`."""
    payload = RunPeriodogramInputModel.model_validate(kwargs)
    result = _run_periodogram(
        time=np.asarray(payload.time, dtype=np.float64),
        flux=np.asarray(payload.flux, dtype=np.float64),
        flux_err=(
            np.asarray(payload.flux_err, dtype=np.float64) if payload.flux_err is not None else None
        ),
        min_period=payload.min_period,
        max_period=payload.max_period,
        preset=payload.preset,
        method=payload.method,
        max_planets=payload.max_planets,
        data_ref=payload.data_ref,
        tic_id=payload.tic_id,
        stellar_radius_rsun=payload.stellar_radius_rsun,
        stellar_mass_msun=payload.stellar_mass_msun,
        use_threads=payload.use_threads,
        per_sector=payload.per_sector,
        downsample_factor=payload.downsample_factor,
    )
    return RunPeriodogramOutputModel(result=result)


__all__ = [
    "CheckInfoModel",
    "CheckRequirementsModel",
    "DescribeChecksModel",
    "ListChecksOutputModel",
    "PeriodogramPeak",
    "PeriodogramResult",
    "RunPeriodogramInputModel",
    "RunPeriodogramOutputModel",
    "describe_checks_typed",
    "list_checks_typed",
    "run_periodogram_typed",
]
