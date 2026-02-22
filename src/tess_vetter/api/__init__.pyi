from __future__ import annotations

from pathlib import Path
from typing import Any

from tess_vetter.api.datasets import LocalDataset
from tess_vetter.api.export import ExportFormat
from tess_vetter.api.generate_report import GenerateReportResult
from tess_vetter.api.per_sector import PerSectorVettingResult
from tess_vetter.api.pipeline import PipelineConfig
from tess_vetter.api.types import (
    Candidate,
    CheckResult,
    LightCurve,
    StellarParams,
    TPFStamp,
    VettingBundleResult,
)
from tess_vetter.api.workflow import WorkflowResult
from tess_vetter.validation.registry import CheckRegistry

MLX_AVAILABLE: bool
MATPLOTLIB_AVAILABLE: bool
VettingPipeline: Any
VettingSession: Any
CheckTier: Any
CheckRequirements: Any
VettingCheck: Any

# Core golden-path entry points

def vet_candidate(
    lc: LightCurve,
    candidate: Candidate,
    *,
    stellar: StellarParams | None = ...,
    tpf: TPFStamp | None = ...,
    network: bool = ...,
    preset: str = ...,
    checks: list[str] | None = ...,
    pipeline_config: PipelineConfig | None = ...,
    **kwargs: Any,
) -> VettingBundleResult: ...

def vet_many(
    lc: LightCurve,
    candidates: list[Candidate],
    *,
    stellar: StellarParams | None = ...,
    tpf: TPFStamp | None = ...,
    network: bool = ...,
    preset: str = ...,
    checks: list[str] | None = ...,
    pipeline_config: PipelineConfig | None = ...,
    **kwargs: Any,
) -> tuple[list[VettingBundleResult], list[dict[str, Any]]]: ...

def run_periodogram(
    *,
    time: Any,
    flux: Any,
    flux_err: Any | None = ...,
    min_period: float = ...,
    max_period: float | None = ...,
    preset: str = ...,
    method: str = ...,
    max_planets: int = ...,
    **kwargs: Any,
) -> Any: ...

def list_checks(registry: CheckRegistry | None = ...) -> list[dict[str, Any]]: ...
def describe_checks(registry: CheckRegistry | None = ...) -> str: ...

def generate_report(
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    *,
    depth_ppm: float | None = ...,
    stellar: StellarParams | None = ...,
    include_html: bool = ...,
    pipeline_config: PipelineConfig | None = ...,
    **kwargs: Any,
) -> GenerateReportResult: ...

# Workflow/per-sector helpers commonly used by agents

def per_sector_vet(
    lc_by_sector: dict[int, LightCurve],
    candidate: Candidate,
    *,
    stellar: StellarParams | None = ...,
    tpf_by_sector: dict[int, TPFStamp] | None = ...,
    preset: str = ...,
    checks: list[str] | None = ...,
    network: bool = ...,
    **kwargs: Any,
) -> PerSectorVettingResult: ...

def run_candidate_workflow(
    *,
    lc: LightCurve | None = ...,
    lc_by_sector: dict[int, LightCurve] | None = ...,
    dataset: LocalDataset | None = ...,
    candidate: Candidate,
    stellar: StellarParams | None = ...,
    tpf_by_sector: dict[int, TPFStamp] | None = ...,
    preset: str = ...,
    checks: list[str] | None = ...,
    network: bool = ...,
    run_per_sector: bool = ...,
    **kwargs: Any,
) -> WorkflowResult: ...

def run_check(
    *,
    lc: LightCurve,
    candidate: Candidate,
    check_id: str,
    stellar: StellarParams | None = ...,
    tpf: TPFStamp | None = ...,
    network: bool = ...,
    preset: str = ...,
    registry: CheckRegistry | None = ...,
    pipeline_config: PipelineConfig | None = ...,
    **kwargs: Any,
) -> CheckResult: ...

def run_checks(
    *,
    lc: LightCurve,
    candidate: Candidate,
    check_ids: list[str],
    stellar: StellarParams | None = ...,
    tpf: TPFStamp | None = ...,
    network: bool = ...,
    preset: str = ...,
    registry: CheckRegistry | None = ...,
    pipeline_config: PipelineConfig | None = ...,
    **kwargs: Any,
) -> list[CheckResult]: ...

def export_bundle(
    bundle: VettingBundleResult,
    *,
    format: ExportFormat,
    path: str | Path | None = ...,
    include_raw: bool = ...,
    title: str = ...,
) -> str | None: ...

def hydrate_cache_from_dataset(
    *,
    dataset: LocalDataset,
    tic_id: int,
    flux_type: str = ...,
    cache_dir: str | Path | None = ...,
    cadence_seconds: float = ...,
    sectors: list[int] | None = ...,
) -> Any: ...

def load_contrast_curve_exofop_tbl(path: str | Path, *, filter: str | None = ...) -> Any: ...

# Reporting helpers

def format_check_result(result: Any, **kwargs: Any) -> str: ...
def format_vetting_table(bundle: Any, **kwargs: Any) -> str: ...
def summarize_bundle(bundle: Any, **kwargs: Any) -> Any: ...
def render_validation_report_markdown(*, title: str, bundle: Any, **kwargs: Any) -> str: ...

# Convenience aliases
vet: Any
periodogram: Any
localize: Any

# Additional public names exported at runtime; kept permissive here.
localize_transit_source: Any
fit_transit: Any
calculate_fpp: Any
