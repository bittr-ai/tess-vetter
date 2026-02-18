"""Summary and verdict helpers for normalized contrast curves."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from bittr_tess_vetter.contrast_curves.models import RulingSummary
from bittr_tess_vetter.contrast_curves.normalize import MAX_RELEVANT_SEPARATION_ARCSEC


def build_ruling_summary(normalized: Mapping[str, Any]) -> RulingSummary:
    """Build compact rule-based coverage summary from normalized arrays."""
    sep = np.asarray(normalized.get("separation_arcsec", []), dtype=np.float64)
    dmag = np.asarray(normalized.get("delta_mag", []), dtype=np.float64)
    if sep.size < 2 or dmag.size < 2:
        return {
            "status": "no_data",
            "quality_assessment": "none",
            "notes": ["No usable contrast-curve points available."],
        }

    dmag_at_05 = float(np.interp(0.5, sep, dmag))
    dmag_at_10 = float(np.interp(1.0, sep, dmag))
    dmag_at_20 = float(np.interp(2.0, sep, dmag))

    quality_assessment = "shallow"
    if dmag_at_05 >= 6.5:
        quality_assessment = "deep"
    elif dmag_at_05 >= 4.0:
        quality_assessment = "moderate"

    notes = [
        "Higher delta_mag means stronger exclusion of faint companions.",
        "Use strongest available curve when running FPP with contrast constraints.",
    ]
    if bool(normalized.get("truncation_applied")):
        notes.append(
            f"Curve was truncated to <= {float(normalized.get('truncation_max_separation_arcsec', MAX_RELEVANT_SEPARATION_ARCSEC)):.1f} arcsec for plausibility."
        )
    extrapolated = []
    sep_min = float(np.min(sep))
    sep_max = float(np.max(sep))
    for probe in (0.5, 1.0, 2.0):
        if probe < sep_min or probe > sep_max:
            extrapolated.append(probe)
    if extrapolated:
        notes.append(
            "One or more summary depths are extrapolated outside sampled separations: "
            + ", ".join(f"{x:.1f}\"" for x in extrapolated)
            + "."
        )

    return {
        "status": "ok",
        "quality_assessment": quality_assessment,
        "max_delta_mag_at_0p5_arcsec": dmag_at_05,
        "max_delta_mag_at_1p0_arcsec": dmag_at_10,
        "max_delta_mag_at_2p0_arcsec": dmag_at_20,
        "innermost_separation_arcsec": sep_min,
        "outermost_separation_arcsec": sep_max,
        "dmag_at_0p5_arcsec": dmag_at_05,
        "dmag_at_1p0_arcsec": dmag_at_10,
        "dmag_at_2p0_arcsec": dmag_at_20,
        "inner_working_angle_arcsec": sep_min,
        "outer_working_angle_arcsec": sep_max,
        "notes": notes,
    }


def derive_contrast_verdict(ruling_summary: RulingSummary) -> tuple[str, str]:
    """Return canonical verdict token and JSON path source for contrast outputs."""
    status = str(ruling_summary.get("status") or "").lower()
    if status != "ok":
        return "NO_CONTRAST_CURVES", "$.ruling_summary.status"

    quality = str(ruling_summary.get("quality_assessment") or "").lower()
    if quality == "deep":
        return "CONTRAST_CURVES_STRONG", "$.ruling_summary.quality_assessment"
    if quality == "moderate":
        return "CONTRAST_CURVES_MODERATE", "$.ruling_summary.quality_assessment"
    return "CONTRAST_CURVES_LIMITED", "$.ruling_summary.quality_assessment"
