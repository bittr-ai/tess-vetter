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
    FullLCPlotData,
    LCSummary,
    LocalDetrendDiagnosticPlotData,
    LocalDetrendWindowData,
    OddEvenPhasePlotData,
    OOTContextPlotData,
    PerTransitStackPlotData,
    Phase15Data,
    Phase15EpochMetrics,
    Phase15FPSignals,
    Phase15RedNoiseMetrics,
    Phase15RobustnessMetrics,
    PhaseFoldedPlotData,
    ReportData,
    SecondaryScanPlotData,
    SecondaryScanQuality,
    SecondaryScanRenderHints,
    TransitTimingPlotData,
    TransitWindowData,
)
from bittr_tess_vetter.report._render_html import render_html

__all__ = [
    "build_report",
    "render_html",
    "ReportData",
    "LCSummary",
    "AliasHarmonicSummaryData",
    "FullLCPlotData",
    "PhaseFoldedPlotData",
    "TransitWindowData",
    "PerTransitStackPlotData",
    "LocalDetrendWindowData",
    "LocalDetrendDiagnosticPlotData",
    "OOTContextPlotData",
    "Phase15Data",
    "Phase15EpochMetrics",
    "Phase15RobustnessMetrics",
    "Phase15RedNoiseMetrics",
    "Phase15FPSignals",
    "TransitTimingPlotData",
    "OddEvenPhasePlotData",
    "SecondaryScanPlotData",
    "SecondaryScanQuality",
    "SecondaryScanRenderHints",
]
