"""Pixel vetting report generator (metrics-only).

This module provides a small report model and helper that bundles pixel-level
analysis outputs into one object for host-side consumption. It does not apply
any pass/warn/reject policy.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bittr_tess_vetter.pixel.aperture import ApertureDependenceResult
from bittr_tess_vetter.pixel.centroid import CentroidResult
from bittr_tess_vetter.pixel.difference import DifferenceImageResult


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "__dict__"):
        d = dict(obj.__dict__)
        # Normalize nested immutable containers to JSON-friendly types.
        if "warnings" in d and isinstance(d["warnings"], tuple):
            d["warnings"] = list(d["warnings"])
        if "flags" in d and isinstance(d["flags"], tuple):
            d["flags"] = list(d["flags"])
        return d
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError(f"Unsupported report payload type: {type(obj)}")


class PixelVetReport(BaseModel):
    """Bundle of pixel-level diagnostics (no policy)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    centroid: dict[str, Any] | None = Field(default=None)
    difference_image: dict[str, Any] | None = Field(default=None)
    aperture_dependence: dict[str, Any] | None = Field(default=None)

    quality_flags: list[str] = Field(default_factory=list)
    manifest_ref: str | None = Field(default=None)
    plot_refs: list[str] = Field(default_factory=list)
    plots: list[dict[str, Any]] = Field(default_factory=list)


def generate_pixel_vet_report(
    centroid_result: CentroidResult | None = None,
    diff_image_result: DifferenceImageResult | None = None,
    aperture_result: ApertureDependenceResult | None = None,
    *,
    manifest_ref: str | None = None,
    plot_refs: list[str] | None = None,
    plots: list[dict[str, Any]] | None = None,
) -> PixelVetReport:
    """Generate a `PixelVetReport` bundling available pixel diagnostics."""
    centroid = _as_dict(centroid_result) if centroid_result is not None else None
    diff = _as_dict(diff_image_result) if diff_image_result is not None else None
    ap = _as_dict(aperture_result) if aperture_result is not None else None

    flags: list[str] = []
    if centroid is None:
        flags.append("MISSING_CENTROID_RESULT")
    else:
        flags.extend([str(w) for w in centroid.get("warnings", [])])

    if diff is None:
        flags.append("MISSING_DIFFERENCE_IMAGE_RESULT")

    if ap is None:
        flags.append("MISSING_APERTURE_DEPENDENCE_RESULT")
    else:
        flags.extend([str(f) for f in ap.get("flags", [])])

    return PixelVetReport(
        centroid=centroid,
        difference_image=diff,
        aperture_dependence=ap,
        quality_flags=flags,
        manifest_ref=manifest_ref,
        plot_refs=list(plot_refs or []),
        plots=list(plots or []),
    )


__all__ = ["PixelVetReport", "generate_pixel_vet_report"]

