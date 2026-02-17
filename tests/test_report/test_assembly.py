from __future__ import annotations

from bittr_tess_vetter.api.types import ok_result
from bittr_tess_vetter.report._assembly import (
    SUMMARY_BLOCK_REGISTRY,
    ReportAssemblyContext,
    assemble_plot_data,
    assemble_summary,
)
from bittr_tess_vetter.report._data import (
    LCFPSignals,
    LCRobustnessData,
    LCRobustnessMetrics,
    LCRobustnessRedNoiseMetrics,
)
from bittr_tess_vetter.report._references import refs_for_check


def _base_context(*, lc_robustness: LCRobustnessData | None = None) -> ReportAssemblyContext:
    v01 = ok_result(
        id="V01",
        name="odd_even_depth",
        confidence=0.9,
        metrics={
            "depth_odd_ppm": 1100.0,
            "depth_even_ppm": 1000.0,
            "delta_sigma": 2.0,
        },
        raw={"plot_data": {"example": [1, 2, 3]}},
    )
    return ReportAssemblyContext(
        tic_id=123,
        toi="TOI-123.01",
        candidate=None,
        stellar=None,
        lc_summary=None,
        check_execution=None,
        checks={"V01": v01},
        bundle=None,
        enrichment=None,
        lc_robustness=lc_robustness,
        full_lc=None,
        phase_folded=None,
        per_transit_stack=None,
        local_detrend=None,
        oot_context=None,
        timing_series=None,
        timing_summary_series=None,
        alias_summary=None,
        odd_even_phase=None,
        secondary_scan=None,
        checks_run=["V01"],
    )


def test_summary_registry_contains_expected_blocks() -> None:
    keys = [spec.key for spec in SUMMARY_BLOCK_REGISTRY]
    assert keys == [
        "odd_even_summary",
        "noise_summary",
        "variability_summary",
        "stellar_contamination_summary",
        "alias_scalar_summary",
        "ephemeris_schedulability_summary",
        "timing_summary",
        "secondary_scan_summary",
        "data_gap_summary",
        "lc_robustness_summary",
    ]


def test_assemble_summary_adds_check_method_refs_and_references() -> None:
    summary, check_overlays = assemble_summary(_base_context())

    assert summary["checks_run"] == ["V01"]
    assert "V01" in summary["checks"]
    assert summary["checks"]["V01"]["method_refs"] == refs_for_check("V01")
    assert isinstance(summary["references"], list)
    assert summary["odd_even_summary"]["depth_diff_ppm"] == 100.0
    assert "stellar_contamination_summary" in summary
    assert "ephemeris_schedulability_summary" in summary
    assert summary["verdict"] == "ALL_CHECKS_PASSED"
    assert summary["verdict_source"] == "$.summary.checks"
    assert summary["caveats"] == []
    assert check_overlays == {"V01": {"example": [1, 2, 3]}}


def test_assemble_plot_data_includes_robustness_and_overlays() -> None:
    lc_robustness = LCRobustnessData(
        version="1",
        baseline_window_mult=2.0,
        per_epoch=[],
        robustness=LCRobustnessMetrics(
            n_epochs_measured=0,
            loto_snr_min=None,
            loto_snr_max=None,
            loto_snr_mean=None,
            loto_depth_ppm_min=None,
            loto_depth_ppm_max=None,
            loto_depth_shift_ppm_max=None,
            dominance_index=None,
        ),
        red_noise=LCRobustnessRedNoiseMetrics(beta_30m=1.1, beta_60m=1.2, beta_duration=1.3),
        fp_signals=LCFPSignals(
            odd_even_depth_diff_sigma=2.0,
            secondary_depth_sigma=1.0,
            phase_0p5_bin_depth_ppm=5.0,
            v_shape_metric=0.2,
            asymmetry_sigma=0.1,
        ),
    )
    ctx = _base_context(lc_robustness=lc_robustness)
    summary, check_overlays = assemble_summary(ctx)
    plot_data = assemble_plot_data(ctx, check_overlays=check_overlays)

    assert summary["lc_robustness_summary"] is not None
    assert summary["lc_robustness_summary"]["beta_duration"] == 1.3
    assert "lc_robustness" in plot_data
    assert plot_data["check_overlays"] == {"V01": {"example": [1, 2, 3]}}


def test_stellar_contamination_summary_is_present_and_null_safe_when_inputs_missing() -> None:
    summary, _ = assemble_summary(_base_context())
    contamination = summary["stellar_contamination_summary"]

    assert contamination["risk_scalar"] is None
    assert contamination["n_components_available"] == 0
    assert contamination["n_components_total"] == 4
    assert contamination["components"]["variability_index"]["raw_value"] is None
    assert contamination["components"]["variability_index"]["transformed_value"] is None


def test_ephemeris_schedulability_summary_is_null_safe_when_v17_missing() -> None:
    summary, _ = assemble_summary(_base_context())
    sched = summary["ephemeris_schedulability_summary"]
    assert sched["scalar"] is None
    assert sched["components"] == {}
    assert sched["provenance"]["source_check"] == "V17"
