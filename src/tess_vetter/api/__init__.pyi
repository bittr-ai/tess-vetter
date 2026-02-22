from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
    ra_deg: float | None = ...,
    dec_deg: float | None = ...,
    tic_id: int | None = ...,
    preset: str = ...,
    checks: list[str] | None = ...,
    context: dict[str, Any] | None = ...,
    pipeline_config: PipelineConfig | None = ...,
) -> VettingBundleResult: ...

def vet_many(
    lc: LightCurve,
    candidates: list[Candidate],
    *,
    stellar: StellarParams | None = ...,
    tpf: TPFStamp | None = ...,
    network: bool = ...,
    ra_deg: float | None = ...,
    dec_deg: float | None = ...,
    tic_id: int | None = ...,
    preset: str = ...,
    checks: list[str] | None = ...,
    context: dict[str, Any] | None = ...,
    pipeline_config: PipelineConfig | None = ...,
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
    data_ref: str = ...,
    tic_id: int | None = ...,
    stellar_radius_rsun: float | None = ...,
    stellar_mass_msun: float | None = ...,
    use_threads: int | None = ...,
    per_sector: bool = ...,
    downsample_factor: int = ...,
) -> Any: ...

def list_checks(registry: CheckRegistry | None = ...) -> list[dict[str, Any]]: ...
def describe_checks(registry: CheckRegistry | None = ...) -> str: ...
def get_legacy_dynamic_export_policy_registry() -> dict[str, str]: ...
def list_legacy_dynamic_exports() -> list[str]: ...
def is_legacy_dynamic_export(name: str) -> bool: ...
def is_unloadable_export(name: str) -> bool: ...
def is_agent_actionable_export(name: str) -> bool: ...

def generate_report(
    tic_id: int,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    *,
    depth_ppm: float | None = ...,
    stellar: StellarParams | None = ...,
    toi: str | None = ...,
    sectors: list[int] | None = ...,
    flux_type: str = ...,
    mast_client: Any | None = ...,
    include_html: bool = ...,
    include_v03: bool = ...,
    bin_minutes: float = ...,
    max_lc_points: int = ...,
    max_phase_points: int = ...,
    include_additional_plots: bool = ...,
    max_transit_windows: int = ...,
    max_points_per_window: int = ...,
    max_timing_points: int = ...,
    include_lc_robustness: bool = ...,
    max_lc_robustness_epochs: int = ...,
    check_config: dict[str, dict[str, Any]] | None = ...,
    pipeline_config: PipelineConfig | None = ...,
    include_enrichment: bool = ...,
    enrichment_config: Any | None = ...,
    custom_views: dict[str, Any] | None = ...,
    progress_callback: Any | None = ...,
    vet_result: VettingBundleResult | dict[str, Any] | None = ...,
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
    ra_deg: float | None = ...,
    dec_deg: float | None = ...,
    tic_id: int | None = ...,
    extra_context: dict[str, Any] | None = ...,
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
    ra_deg: float | None = ...,
    dec_deg: float | None = ...,
    tic_id: int | None = ...,
    run_per_sector: bool = ...,
    extra_context: dict[str, Any] | None = ...,
) -> WorkflowResult: ...

def run_check(
    *,
    lc: LightCurve,
    candidate: Candidate,
    check_id: str,
    stellar: StellarParams | None = ...,
    tpf: TPFStamp | None = ...,
    network: bool = ...,
    ra_deg: float | None = ...,
    dec_deg: float | None = ...,
    tic_id: int | None = ...,
    context: Mapping[str, object] | None = ...,
    preset: str = ...,
    registry: CheckRegistry | None = ...,
    pipeline_config: PipelineConfig | None = ...,
) -> CheckResult: ...

def run_checks(
    *,
    lc: LightCurve,
    candidate: Candidate,
    check_ids: list[str],
    stellar: StellarParams | None = ...,
    tpf: TPFStamp | None = ...,
    network: bool = ...,
    ra_deg: float | None = ...,
    dec_deg: float | None = ...,
    tic_id: int | None = ...,
    context: Mapping[str, object] | None = ...,
    preset: str = ...,
    registry: CheckRegistry | None = ...,
    pipeline_config: PipelineConfig | None = ...,
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

def format_check_result(
    result: Any,
    *,
    include_header: bool = ...,
    include_metrics: bool = ...,
    metric_keys: Sequence[str] = ...,
    max_metrics: int = ...,
    include_flags: bool = ...,
    include_notes: bool = ...,
    include_provenance: bool = ...,
) -> str: ...
def format_vetting_table(bundle: Any, *, options: Any | None = ...) -> str: ...
def summarize_bundle(
    bundle: Any,
    *,
    check_ids: Sequence[str] | None = ...,
    include_metrics: bool = ...,
    metric_keys: Sequence[str] | None = ...,
    include_flags: bool = ...,
    include_notes: bool = ...,
    include_provenance: bool = ...,
    include_inputs_summary: bool = ...,
) -> Any: ...
def render_validation_report_markdown(
    *,
    title: str,
    bundle: Any,
    include_table: bool = ...,
    table_options: Any | None = ...,
    extra_sections: Iterable[tuple[str, str]] | None = ...,
) -> str: ...

# Convenience aliases
vet: Any
periodogram: Any
localize: Any

# Additional public names exported at runtime; kept permissive here.
localize_transit_source: Any
fit_transit: Any
calculate_fpp: Any
