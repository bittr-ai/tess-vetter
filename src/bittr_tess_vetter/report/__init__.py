"""LC-only report data assembly module.

Public API:
    build_report   -- Assemble a ReportData from light curve + candidate
    ReportData     -- Structured report data packet
    LCSummary      -- Light curve vital signs
    FullLCPlotData -- Plot-ready full LC arrays
    PhaseFoldedPlotData -- Plot-ready phase-folded arrays
"""

from bittr_tess_vetter.report._build import build_report
from bittr_tess_vetter.report._data import (
    FullLCPlotData,
    LCSummary,
    PhaseFoldedPlotData,
    ReportData,
)

__all__ = [
    "build_report",
    "ReportData",
    "LCSummary",
    "FullLCPlotData",
    "PhaseFoldedPlotData",
]
