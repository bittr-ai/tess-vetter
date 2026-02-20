"""LC-only report data assembly and rendering module.

Public API:
    build_report   -- Assemble a ReportData from light curve + candidate
    render_html    -- Render a ReportData to self-contained HTML
    ReportData     -- Structured report data packet
    LCSummary      -- Light curve vital signs
    FullLCPlotData -- Plot-ready full LC arrays
    PhaseFoldedPlotData -- Plot-ready phase-folded arrays
"""

from tess_vetter.report._build_core import build_report
from tess_vetter.report._data import (
    AliasHarmonicSummaryData,
    CheckExecutionState,
    EnrichmentBlockData,
    FullLCPlotData,
    LCFPSignals,
    LCRobustnessData,
    LCRobustnessEpochMetrics,
    LCRobustnessMetrics,
    LCRobustnessRedNoiseMetrics,
    LCSummary,
    LocalDetrendDiagnosticPlotData,
    LocalDetrendWindowData,
    OddEvenPhasePlotData,
    OOTContextPlotData,
    PerTransitStackPlotData,
    PhaseFoldedPlotData,
    ReportData,
    ReportEnrichmentData,
    SecondaryScanPlotData,
    SecondaryScanQuality,
    SecondaryScanRenderHints,
    TransitTimingPlotData,
    TransitWindowData,
)
from tess_vetter.report._render_html import render_html, render_html_from_payload
from tess_vetter.report.field_catalog import FIELD_CATALOG, FieldKey, FieldSpec
from tess_vetter.report.schema import (
    BundleSummaryModel,
    CheckExecutionSummaryModel,
    CheckSummaryModel,
    ReportPayloadMetaModel,
    ReportPayloadModel,
    ReportPlotDataModel,
    ReportSummaryModel,
    report_payload_json_schema,
)
from tess_vetter.report.vet_lc_summary import build_vet_lc_summary_blocks

__all__ = [
    "build_report",
    "render_html",
    "render_html_from_payload",
    "report_payload_json_schema",
    "ReportPayloadModel",
    "ReportSummaryModel",
    "ReportPlotDataModel",
    "ReportPayloadMetaModel",
    "CheckSummaryModel",
    "CheckExecutionSummaryModel",
    "BundleSummaryModel",
    "build_vet_lc_summary_blocks",
    "FieldKey",
    "FieldSpec",
    "FIELD_CATALOG",
    "ReportData",
    "CheckExecutionState",
    "LCSummary",
    "AliasHarmonicSummaryData",
    "EnrichmentBlockData",
    "FullLCPlotData",
    "PhaseFoldedPlotData",
    "ReportEnrichmentData",
    "TransitWindowData",
    "PerTransitStackPlotData",
    "LocalDetrendWindowData",
    "LocalDetrendDiagnosticPlotData",
    "OOTContextPlotData",
    "LCRobustnessData",
    "LCRobustnessEpochMetrics",
    "LCRobustnessMetrics",
    "LCRobustnessRedNoiseMetrics",
    "LCFPSignals",
    "TransitTimingPlotData",
    "OddEvenPhasePlotData",
    "SecondaryScanPlotData",
    "SecondaryScanQuality",
    "SecondaryScanRenderHints",
]
