"""Thin workflow orchestration helpers (researcher-facing, policy-free).

This module composes existing public APIs to reduce notebook glue while keeping
outputs transparent and machine-friendly.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

from tess_vetter.api.datasets import LocalDataset
from tess_vetter.api.per_sector import PerSectorVettingResult, per_sector_vet
from tess_vetter.api.stitch import StitchedLC, stitch_lightcurves
from tess_vetter.api.types import (
    Candidate,
    LightCurve,
    StellarParams,
    TPFStamp,
    VettingBundleResult,
)
from tess_vetter.api.vet import vet_candidate

WORKFLOW_SCHEMA_VERSION = 1
WORKFLOW_PROVENANCE_SCHEMA_VERSION = 1


class WorkflowStitchedDiagnostics(TypedDict):
    """Stable row contract for stitched per-sector diagnostics."""

    sector: int
    n_cadences: int
    median_flux: float
    mad_flux: float
    normalization_factor: float
    normalization_warning: str | None
    quality_flags_summary: dict[str, int]


class WorkflowStitchedSummary(TypedDict):
    """Stable stitched summary contract emitted by WorkflowResult."""

    normalization_policy_version: str
    sectors: list[int]
    per_sector_diagnostics: list[WorkflowStitchedDiagnostics]


class WorkflowProvenance(TypedDict):
    """Stable provenance contract emitted by WorkflowResult."""

    schema_version: int
    package_version: str
    preset: str
    checks: list[str] | None
    network: bool
    run_per_sector: bool
    tic_id: int | None
    has_dataset: bool
    has_lc_by_sector: bool
    has_tpf_by_sector: bool


class WorkflowResultPayload(TypedDict):
    """Stable JSON payload contract returned by WorkflowResult.to_dict()."""

    schema_version: int
    bundle: dict[str, Any]
    per_sector: dict[str, Any] | None
    stitched: WorkflowStitchedSummary | None
    provenance: WorkflowProvenance


def _package_version() -> str:
    try:
        return str(importlib.metadata.version("tess-vetter"))
    except Exception:
        return "unknown"


def _stitch_from_mapping(lc_by_sector: dict[int, LightCurve]) -> StitchedLC:
    lc_list: list[dict[str, Any]] = []
    for sec in sorted(lc_by_sector.keys()):
        lc = lc_by_sector[int(sec)]
        t = np.asarray(lc.time, dtype=np.float64)
        f = np.asarray(lc.flux, dtype=np.float64)
        e = (
            np.asarray(lc.flux_err, dtype=np.float64)
            if lc.flux_err is not None
            else np.zeros_like(f, dtype=np.float64)
        )
        q = (
            np.asarray(lc.quality, dtype=np.int32)
            if lc.quality is not None
            else np.zeros_like(t, dtype=np.int32)
        )
        lc_list.append({"time": t, "flux": f, "flux_err": e, "sector": int(sec), "quality": q})
    return stitch_lightcurves(lc_list)


@dataclass(frozen=True)
class WorkflowResult:
    """Structured output from :func:`run_candidate_workflow`."""

    schema_version: int
    bundle: VettingBundleResult
    per_sector: PerSectorVettingResult | None
    stitched: WorkflowStitchedSummary | None
    provenance: WorkflowProvenance

    def to_dict(self) -> WorkflowResultPayload:
        return {
            "schema_version": int(self.schema_version),
            "bundle": self.bundle.model_dump(),
            "per_sector": self.per_sector.to_dict() if self.per_sector is not None else None,
            "stitched": dict(self.stitched) if self.stitched is not None else None,
            "provenance": dict(self.provenance),
        }


def run_candidate_workflow(
    *,
    lc: LightCurve | None = None,
    lc_by_sector: dict[int, LightCurve] | None = None,
    dataset: LocalDataset | None = None,
    candidate: Candidate,
    stellar: StellarParams | None = None,
    tpf_by_sector: dict[int, TPFStamp] | None = None,
    preset: str = "default",
    checks: list[str] | None = None,
    network: bool = False,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    tic_id: int | None = None,
    run_per_sector: bool = True,
    extra_context: dict[str, Any] | None = None,
) -> WorkflowResult:
    """Run a common researcher workflow with transparent, structured outputs.

    This helper does not download data. Provide `lc`, `lc_by_sector`, or `dataset`.
    """
    if dataset is not None:
        if lc is not None or lc_by_sector is not None or tpf_by_sector is not None:
            raise ValueError("Provide either dataset=... OR (lc/lc_by_sector/tpf_by_sector), not both")
        lc_by_sector = dict(dataset.lc_by_sector)
        tpf_by_sector = dict(dataset.tpf_by_sector)

    if lc is None and not lc_by_sector:
        raise ValueError("Provide lc=..., lc_by_sector=..., or dataset=...")

    stitched_summary: WorkflowStitchedSummary | None = None
    lc_for_bundle: LightCurve
    tpf_for_bundle: TPFStamp | None = None

    if lc is not None:
        lc_for_bundle = lc
        if tpf_by_sector and len(tpf_by_sector) == 1:
            tpf_for_bundle = next(iter(tpf_by_sector.values()))
    else:
        assert lc_by_sector is not None
        stitched = _stitch_from_mapping(lc_by_sector)
        lc_for_bundle = LightCurve(
            time=stitched.time,
            flux=stitched.flux,
            flux_err=stitched.flux_err,
            quality=stitched.quality,
        )
        stitched_summary = {
            "normalization_policy_version": str(stitched.normalization_policy_version),
            "sectors": sorted({int(s.sector) for s in stitched.per_sector_diagnostics}),
            "per_sector_diagnostics": [
                {
                    "sector": int(d.sector),
                    "n_cadences": int(d.n_cadences),
                    "median_flux": float(d.median_flux),
                    "mad_flux": float(d.mad_flux),
                    "normalization_factor": float(d.normalization_factor),
                    "normalization_warning": d.normalization_warning,
                    "quality_flags_summary": dict(d.quality_flags_summary),
                }
                for d in stitched.per_sector_diagnostics
            ],
        }
        # Do not pass a single TPF when we have multiple sectors; use per-sector vetting instead.
        if tpf_by_sector and len(tpf_by_sector) == 1:
            tpf_for_bundle = next(iter(tpf_by_sector.values()))

    bundle = vet_candidate(
        lc_for_bundle,
        candidate,
        stellar=stellar,
        tpf=tpf_for_bundle,
        network=network,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        tic_id=tic_id,
        preset=preset,
        checks=checks,
        context=extra_context,
    )

    per_sector: PerSectorVettingResult | None = None
    if run_per_sector and lc_by_sector:
        per_sector = per_sector_vet(
            lc_by_sector,
            candidate,
            stellar=stellar,
            tpf_by_sector=tpf_by_sector,
            preset=preset,
            checks=checks,
            network=network,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            tic_id=tic_id,
            extra_context=extra_context,
        )

    provenance: WorkflowProvenance = {
        "schema_version": WORKFLOW_PROVENANCE_SCHEMA_VERSION,
        "package_version": _package_version(),
        "preset": str(preset),
        "checks": list(checks) if checks is not None else None,
        "network": bool(network),
        "run_per_sector": bool(run_per_sector),
        "tic_id": int(tic_id) if tic_id is not None else None,
        "has_dataset": dataset is not None,
        "has_lc_by_sector": bool(lc_by_sector),
        "has_tpf_by_sector": bool(tpf_by_sector),
    }

    return WorkflowResult(
        schema_version=WORKFLOW_SCHEMA_VERSION,
        bundle=bundle,
        per_sector=per_sector,
        stitched=stitched_summary,
        provenance=provenance,
    )


__all__ = [
    "WORKFLOW_PROVENANCE_SCHEMA_VERSION",
    "WORKFLOW_SCHEMA_VERSION",
    "WorkflowProvenance",
    "WorkflowResult",
    "WorkflowResultPayload",
    "WorkflowStitchedDiagnostics",
    "WorkflowStitchedSummary",
    "run_candidate_workflow",
]
