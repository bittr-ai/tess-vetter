"""Typed payloads returned by contrast-curve utilities."""

from __future__ import annotations

from typing import TypedDict


class NormalizedContrastCurve(TypedDict, total=False):
    separation_arcsec: list[float]
    delta_mag: list[float]
    delta_flux_ratio: list[float]
    n_points: int
    separation_arcsec_min: float
    separation_arcsec_max: float
    delta_mag_min: float
    delta_mag_max: float
    truncation_applied: bool
    truncation_max_separation_arcsec: float
    n_points_before_truncation: int
    n_points_dropped_by_truncation: int
    original_separation_arcsec_max: float


class RulingSummary(TypedDict, total=False):
    status: str
    quality_assessment: str
    notes: list[str]
    max_delta_mag_at_0p5_arcsec: float
    max_delta_mag_at_1p0_arcsec: float
    max_delta_mag_at_2p0_arcsec: float
    innermost_separation_arcsec: float
    outermost_separation_arcsec: float
    dmag_at_0p5_arcsec: float
    dmag_at_1p0_arcsec: float
    dmag_at_2p0_arcsec: float
    inner_working_angle_arcsec: float
    outer_working_angle_arcsec: float
