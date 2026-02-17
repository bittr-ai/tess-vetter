"""Typed report assembly pipeline: context -> summary blocks -> payload parts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from bittr_tess_vetter.report._references import (
    reference_entries,
    refs_for_check,
    refs_for_summary_block,
)
from bittr_tess_vetter.report._summary_builders import (
    _build_alias_scalar_summary,
    _build_data_gap_summary,
    _build_lc_robustness_summary,
    _build_noise_summary,
    _build_odd_even_summary,
    _build_secondary_scan_summary,
    _build_timing_summary,
    _build_variability_summary,
    _model_dump_like,
)
from bittr_tess_vetter.report._summary_verdict import build_summary_verdict
from bittr_tess_vetter.validation.result_schema import CheckResult, VettingBundleResult


@dataclass(frozen=True)
class ReportAssemblyContext:
    """Typed input contract for report assembly."""

    tic_id: int | None
    toi: str | None
    candidate: Any | None
    stellar: Any | None
    lc_summary: Any | None
    check_execution: Any | None
    checks: dict[str, CheckResult]
    bundle: VettingBundleResult | None
    enrichment: Any | None
    lc_robustness: Any | None
    full_lc: Any | None
    phase_folded: Any | None
    per_transit_stack: Any | None
    local_detrend: Any | None
    oot_context: Any | None
    timing_series: Any | None
    timing_summary_series: Any | None
    alias_summary: Any | None
    odd_even_phase: Any | None
    secondary_scan: Any | None
    checks_run: list[str]


@dataclass(frozen=True)
class SummaryBlockSpec:
    """A single summary block builder entry in the registry."""

    key: str
    builder: Callable[[ReportAssemblyContext], Any]
    reference_block: str | None = None


def _block_odd_even(ctx: ReportAssemblyContext) -> dict[str, Any]:
    return _build_odd_even_summary(ctx.checks)


def _block_noise(ctx: ReportAssemblyContext) -> dict[str, Any]:
    return _build_noise_summary(ctx.lc_summary, ctx.lc_robustness)


def _block_variability(ctx: ReportAssemblyContext) -> dict[str, Any]:
    return _build_variability_summary(
        ctx.lc_summary,
        ctx.timing_series,
        ctx.alias_summary,
        ctx.stellar,
    )


def _block_alias(ctx: ReportAssemblyContext) -> dict[str, Any]:
    return _build_alias_scalar_summary(ctx.alias_summary)


def _block_timing(ctx: ReportAssemblyContext) -> dict[str, Any]:
    timing_source = ctx.timing_summary_series or ctx.timing_series
    return _build_timing_summary(timing_source, ctx.checks)


def _block_secondary_scan(ctx: ReportAssemblyContext) -> dict[str, Any]:
    return _build_secondary_scan_summary(ctx.secondary_scan)


def _block_data_gap(ctx: ReportAssemblyContext) -> dict[str, Any]:
    return _build_data_gap_summary(ctx.checks)


def _block_lc_robustness(ctx: ReportAssemblyContext) -> dict[str, Any] | None:
    return _build_lc_robustness_summary(ctx.lc_robustness)


SUMMARY_BLOCK_REGISTRY: tuple[SummaryBlockSpec, ...] = (
    SummaryBlockSpec("odd_even_summary", _block_odd_even, "odd_even_summary"),
    SummaryBlockSpec("noise_summary", _block_noise, "noise_summary"),
    SummaryBlockSpec("variability_summary", _block_variability, "variability_summary"),
    SummaryBlockSpec("alias_scalar_summary", _block_alias, "alias_scalar_summary"),
    SummaryBlockSpec("timing_summary", _block_timing, "timing_summary"),
    SummaryBlockSpec("secondary_scan_summary", _block_secondary_scan, "secondary_scan_summary"),
    SummaryBlockSpec("data_gap_summary", _block_data_gap, "data_gap_summary"),
    SummaryBlockSpec("lc_robustness_summary", _block_lc_robustness),
)


def assemble_summary(ctx: ReportAssemblyContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble base summary and check overlay payload from typed context."""
    summary: dict[str, Any] = {
        "tic_id": ctx.tic_id,
        "toi": ctx.toi,
        "checks_run": list(ctx.checks_run),
    }
    check_overlays: dict[str, Any] = {}

    if ctx.candidate is not None:
        eph = ctx.candidate.ephemeris
        summary["ephemeris"] = {
            "period_days": eph.period_days,
            "t0_btjd": eph.t0_btjd,
            "duration_hours": eph.duration_hours,
        }
        summary["input_depth_ppm"] = getattr(ctx.candidate, "depth_ppm", None)

    if ctx.stellar is not None:
        summary["stellar"] = _model_dump_like(ctx.stellar)

    if ctx.lc_summary is not None:
        summary["lc_summary"] = asdict(ctx.lc_summary)

    if ctx.check_execution is not None:
        summary["check_execution"] = asdict(ctx.check_execution)

    checks_summary: dict[str, Any] = {}
    reference_ids: set[str] = set()
    for check_id, check in ctx.checks.items():
        method_refs = refs_for_check(check_id)
        reference_ids.update(method_refs)
        checks_summary[check_id] = {
            "id": check.id,
            "name": check.name,
            "status": check.status,
            "confidence": check.confidence,
            "metrics": check.metrics,
            "flags": check.flags,
            "notes": check.notes,
            "provenance": check.provenance,
            "method_refs": method_refs,
        }
        if isinstance(check.raw, dict) and "plot_data" in check.raw:
            check_overlays[check_id] = check.raw["plot_data"]
    summary["checks"] = checks_summary

    for block in SUMMARY_BLOCK_REGISTRY:
        summary[block.key] = block.builder(ctx)
        if block.reference_block is not None:
            reference_ids.update(refs_for_summary_block(block.reference_block))
    summary["references"] = reference_entries(reference_ids)

    if ctx.bundle is not None:
        summary["bundle_summary"] = {
            "n_checks": len(ctx.bundle.results),
            "n_ok": ctx.bundle.n_passed,
            "n_failed": ctx.bundle.n_failed,
            "n_skipped": ctx.bundle.n_unknown,
            "failed_ids": ctx.bundle.failed_check_ids,
        }

    summary.update(
        build_summary_verdict(
            bundle=ctx.bundle,
            checks=ctx.checks,
            noise_summary=summary.get("noise_summary"),
        )
    )

    if ctx.enrichment is not None:
        summary["enrichment"] = asdict(ctx.enrichment)

    return summary, check_overlays


def assemble_plot_data(
    ctx: ReportAssemblyContext,
    *,
    check_overlays: dict[str, Any],
) -> dict[str, Any]:
    """Assemble plot_data payload from typed context."""
    plot_data: dict[str, Any] = {}
    if check_overlays:
        plot_data["check_overlays"] = check_overlays
    if ctx.lc_robustness is not None:
        plot_data["lc_robustness"] = asdict(ctx.lc_robustness)
    if ctx.full_lc is not None:
        plot_data["full_lc"] = asdict(ctx.full_lc)
    if ctx.phase_folded is not None:
        plot_data["phase_folded"] = asdict(ctx.phase_folded)
    if ctx.per_transit_stack is not None:
        plot_data["per_transit_stack"] = asdict(ctx.per_transit_stack)
    if ctx.local_detrend is not None:
        plot_data["local_detrend"] = asdict(ctx.local_detrend)
    if ctx.oot_context is not None:
        plot_data["oot_context"] = asdict(ctx.oot_context)
    if ctx.timing_series is not None:
        plot_data["timing_series"] = asdict(ctx.timing_series)
    if ctx.alias_summary is not None:
        plot_data["alias_summary"] = asdict(ctx.alias_summary)
    if ctx.odd_even_phase is not None:
        plot_data["odd_even_phase"] = asdict(ctx.odd_even_phase)
    if ctx.secondary_scan is not None:
        plot_data["secondary_scan"] = asdict(ctx.secondary_scan)
    return plot_data
