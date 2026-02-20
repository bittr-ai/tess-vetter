from __future__ import annotations

import pytest

from tess_vetter.api.fpp import ContrastCurve
from tess_vetter.contrast_curves import (
    ContrastCurveParseError,
    build_ruling_summary,
    normalize_contrast_curve,
)


def test_normalize_contrast_curve_truncates_outer_artifact_values() -> None:
    curve = ContrastCurve(
        separation_arcsec=[0.1, 0.5, 4.5],
        delta_mag=[5.1, 7.2, 60.0],
        filter="Ks",
    )

    normalized = normalize_contrast_curve(curve)
    ruling = build_ruling_summary(normalized)

    assert normalized["n_points"] == 2
    assert normalized["delta_mag_max"] == pytest.approx(7.2)
    assert normalized["separation_arcsec_max"] <= 0.5
    assert normalized["truncation_applied"] is True
    assert normalized["n_points_before_truncation"] == 3
    assert normalized["n_points_dropped_by_truncation"] == 1
    assert normalized["original_separation_arcsec_max"] == pytest.approx(4.5)
    assert ruling["status"] == "ok"
    assert ruling["max_delta_mag_at_2p0_arcsec"] == pytest.approx(7.2)


def test_normalize_contrast_curve_requires_inner_coverage_after_truncation() -> None:
    curve = ContrastCurve(
        separation_arcsec=[4.2, 5.1],
        delta_mag=[4.0, 4.5],
        filter="Ks",
    )

    with pytest.raises(ContrastCurveParseError, match="Fewer than 2 data points within 4.0 arcsec"):
        normalize_contrast_curve(curve)


def test_normalize_contrast_curve_still_rejects_implausible_inner_values() -> None:
    curve = ContrastCurve(
        separation_arcsec=[0.1, 0.3, 0.5],
        delta_mag=[-3.5, 1.0, 2.0],
        filter="Ks",
    )

    with pytest.raises(ContrastCurveParseError, match="Implausible delta_mag range"):
        normalize_contrast_curve(curve)


def test_normalize_contrast_curve_no_truncation_metadata() -> None:
    curve = ContrastCurve(
        separation_arcsec=[0.1, 0.5, 1.0, 3.9],
        delta_mag=[3.0, 4.5, 5.4, 6.2],
        filter="Ks",
    )

    normalized = normalize_contrast_curve(curve)
    assert normalized["truncation_applied"] is False
    assert normalized["n_points_before_truncation"] == normalized["n_points"]
    assert normalized["n_points_dropped_by_truncation"] == 0
    assert normalized["original_separation_arcsec_max"] == pytest.approx(normalized["separation_arcsec_max"])
    assert normalized["truncation_max_separation_arcsec"] == pytest.approx(4.0)


def test_normalize_contrast_curve_keeps_4_arcsec_boundary_point() -> None:
    curve = ContrastCurve(
        separation_arcsec=[0.5, 4.0, 5.5],
        delta_mag=[5.0, 7.0, 12.0],
        filter="Ks",
    )

    normalized = normalize_contrast_curve(curve)
    assert normalized["n_points"] == 2
    assert normalized["separation_arcsec_max"] == pytest.approx(4.0)


def test_normalize_contrast_curve_counts_dropped_points_after_dedup() -> None:
    curve = ContrastCurve(
        separation_arcsec=[0.5, 1.0, 5.0, 5.0],
        delta_mag=[5.0, 6.0, 8.0, 9.0],
        filter="Ks",
    )

    normalized = normalize_contrast_curve(curve)
    assert normalized["n_points_before_truncation"] == 3
    assert normalized["n_points"] == 2
    assert normalized["n_points_dropped_by_truncation"] == 1
    assert normalized["original_separation_arcsec_max"] == pytest.approx(5.0)


def test_build_ruling_summary_includes_extrapolation_note() -> None:
    curve = ContrastCurve(
        separation_arcsec=[3.2, 3.8],
        delta_mag=[8.0, 8.5],
        filter="Ks",
    )
    normalized = normalize_contrast_curve(curve)
    ruling = build_ruling_summary(normalized)
    notes = " ".join(ruling.get("notes", []))
    assert "extrapolated" in notes.lower()
