"""Public helper for vet-usable LC summary blocks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from bittr_tess_vetter.report._data import ReportData
from bittr_tess_vetter.report._summary_builders import (
    _build_alias_scalar_summary,
    _build_lc_robustness_summary,
    _build_noise_summary,
    _build_odd_even_summary,
    _build_variability_summary,
)


def build_vet_lc_summary_blocks(report: ReportData) -> dict[str, Any]:
    """Build vet-facing LC summary blocks from ``ReportData``.

    This is a public seam for callers that need report-derived scalar
    summaries without importing private report internals.
    """
    return {
        "lc_summary": asdict(report.lc_summary) if report.lc_summary is not None else None,
        "noise_summary": _build_noise_summary(report.lc_summary, report.lc_robustness),
        "variability_summary": _build_variability_summary(report.lc_summary, report.timing_series),
        "lc_robustness_summary": _build_lc_robustness_summary(report.lc_robustness),
        "odd_even_summary": _build_odd_even_summary(report.checks),
        "alias_scalar_summary": _build_alias_scalar_summary(report.alias_summary),
    }


__all__ = ["build_vet_lc_summary_blocks"]
