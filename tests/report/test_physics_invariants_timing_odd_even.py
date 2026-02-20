from __future__ import annotations

import pytest

from tess_vetter.api.types import ok_result
from tess_vetter.report import ReportData, TransitTimingPlotData


def test_timing_summary_deterministic_physics_invariants() -> None:
    epochs = list(range(1, 9))
    oc_seconds = [float(((-1) ** epoch) * (5 * epoch)) for epoch in epochs]
    snr = [float(8 + (epoch % 4)) for epoch in epochs]

    report = ReportData(
        timing_series=TransitTimingPlotData(
            epochs=epochs,
            oc_seconds=oc_seconds,
            snr=snr,
            rms_seconds=10.0,
            periodicity_score=0.5,
            linear_trend_sec_per_epoch=0.25,
        ),
        checks_run=[],
    )

    payload_a = report.to_json()
    payload_b = report.to_json()
    summary_a = payload_a["summary"]["timing_summary"]
    summary_b = payload_b["summary"]["timing_summary"]

    assert summary_a == summary_b, "timing_summary must be deterministic"

    assert summary_a["n_epochs_measured"] == 8
    assert summary_a["outlier_count"] == 2
    assert summary_a["deepest_epoch"] == 3

    assert summary_a["rms_seconds"] == pytest.approx(10.0, abs=1e-3)
    assert summary_a["periodicity_score"] == pytest.approx(0.5, abs=1e-3)
    assert summary_a["linear_trend_sec_per_epoch"] == pytest.approx(0.25, abs=1e-3)
    assert summary_a["max_abs_oc_seconds"] == pytest.approx(40.0, abs=1e-3)
    assert summary_a["max_snr"] == pytest.approx(11.0, abs=1e-3)
    assert summary_a["snr_median"] == pytest.approx(9.5, abs=1e-3)
    assert summary_a["oc_median"] == pytest.approx(2.5, abs=1e-3)
    assert summary_a["outlier_fraction"] == pytest.approx(0.25, abs=1e-3)


@pytest.mark.parametrize(
    ("depth_diff_sigma", "expected_significant", "expected_flags"),
    [
        (3.0, True, ["ODD_EVEN_MISMATCH"]),
        (3.05, True, ["ODD_EVEN_MISMATCH"]),
        (2.95, False, []),
    ],
)
def test_odd_even_summary_deterministic_threshold_flags(
    depth_diff_sigma: float,
    expected_significant: bool,
    expected_flags: list[str],
) -> None:
    odd_depth_ppm = 1520.0
    even_depth_ppm = 1010.0
    depth_diff_ppm = odd_depth_ppm - even_depth_ppm

    odd_even_check = ok_result(
        id="V01",
        name="odd_even_depth",
        confidence=0.9,
        metrics={
            "depth_odd_ppm": odd_depth_ppm,
            "depth_even_ppm": even_depth_ppm,
            "delta_ppm": depth_diff_ppm,
            "delta_sigma": depth_diff_sigma,
        },
    )

    report = ReportData(checks={"V01": odd_even_check}, checks_run=["V01"])
    payload_a = report.to_json()
    payload_b = report.to_json()
    summary_a = payload_a["summary"]["odd_even_summary"]
    summary_b = payload_b["summary"]["odd_even_summary"]

    assert summary_a == summary_b, "odd_even_summary must be deterministic"

    assert summary_a["is_significant"] is expected_significant
    assert summary_a["flags"] == expected_flags

    assert summary_a["odd_depth_ppm"] == pytest.approx(1520.0, abs=0.1)
    assert summary_a["even_depth_ppm"] == pytest.approx(1010.0, abs=0.1)
    assert summary_a["depth_diff_ppm"] == pytest.approx(510.0, abs=0.1)
    assert summary_a["depth_diff_sigma"] == pytest.approx(depth_diff_sigma, abs=0.1)
