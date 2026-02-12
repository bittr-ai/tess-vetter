"""LC-only report data assembly and rendering module.

Public API:
    build_report   -- Assemble a ReportData from light curve + candidate
    render_html    -- Render a ReportData to self-contained HTML
    ReportData     -- Structured report data packet
    LCSummary      -- Light curve vital signs
    FullLCPlotData -- Plot-ready full LC arrays
    PhaseFoldedPlotData -- Plot-ready phase-folded arrays
"""

from bittr_tess_vetter.report._build import build_report
from bittr_tess_vetter.report._data import (
    AliasHarmonicSummaryData,
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
from bittr_tess_vetter.report._render_html import render_html, render_html_from_payload

__all__ = [
    "build_report",
    "render_html",
    "render_html_from_payload",
    "ReportData",
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
