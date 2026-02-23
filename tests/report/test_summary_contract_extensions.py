from __future__ import annotations

from tess_vetter.api.types import StellarParams
from tess_vetter.report import (
    AliasHarmonicSummaryData,
    LCSummary,
    ReportData,
    TransitTimingPlotData,
)
from tess_vetter.validation.result_schema import (
    VettingBundleResult,
    error_result,
    ok_result,
    skipped_result,
)


def _minimal_lc_summary() -> LCSummary:
    return LCSummary(
        n_points=10,
        n_valid=10,
        n_transits=2,
        n_in_transit_total=4,
        duration_days=5.0,
        cadence_seconds=120.0,
        flux_std_ppm=100.0,
        flux_mad_ppm=100.0,
        gap_fraction=0.0,
        snr=5.0,
        depth_ppm=500.0,
        depth_err_ppm=50.0,
    )


def test_summary_verdict_all_ok_and_source() -> None:
    v01 = ok_result(id="V01", name="odd_even_depth", metrics={"delta_sigma": 0.2})
    bundle = VettingBundleResult.from_checks([v01])
    payload = ReportData(checks={"V01": v01}, bundle=bundle, checks_run=["V01"]).to_json()
    summary = payload["summary"]

    assert summary["verdict"] == "ALL_CHECKS_PASSED"
    assert summary["verdict_source"] == "$.summary.bundle_summary"
    assert summary["caveats"] == []


def test_summary_verdict_without_bundle_uses_checks_source_path() -> None:
    v01 = ok_result(id="V01", name="odd_even_depth", metrics={"delta_sigma": 0.2})
    payload = ReportData(checks={"V01": v01}, checks_run=["V01"]).to_json()
    summary = payload["summary"]

    assert summary["verdict"] == "ALL_CHECKS_PASSED"
    assert summary["verdict_source"] == "$.summary.checks"


def test_summary_verdict_failed_and_red_noise_caveat_qualified() -> None:
    failed = error_result(
        id="V02",
        name="secondary_eclipse",
        error="FAILED",
        notes=["Insufficient baseline for red noise, using default inflation"],
    )
    bundle = VettingBundleResult.from_checks([failed])
    payload = ReportData(checks={"V02": failed}, bundle=bundle, checks_run=["V02"]).to_json()
    summary = payload["summary"]

    assert summary["verdict"] == "CHECK_FAILED:V02_WITH_CAVEATS"
    assert summary["caveats"] == ["RED_NOISE_CAVEAT"]


def test_summary_verdict_flagged_for_skipped_checks() -> None:
    skipped = skipped_result(id="V13", name="data_gaps", reason_flag="INSUFFICIENT_DATA")
    bundle = VettingBundleResult.from_checks([skipped])
    payload = ReportData(checks={"V13": skipped}, bundle=bundle, checks_run=["V13"]).to_json()
    summary = payload["summary"]

    assert summary["verdict"] == "CHECKS_FLAGGED:V13"


def test_variability_summary_uses_alias_signal_to_avoid_low_label() -> None:
    alias = AliasHarmonicSummaryData(
        harmonic_labels=["P", "P/2", "2P"],
        periods=[4.0, 2.0, 8.0],
        scores=[1.0, 1.2, 0.1],
        harmonic_depth_ppm=[600.0, 650.0, 100.0],
        best_harmonic="P/2",
        best_ratio_over_p=1.2,
        classification="ALIAS_WEAK",
        phase_shift_event_count=0,
        phase_shift_peak_sigma=0.0,
        secondary_significance=0.0,
    )
    payload = ReportData(
        lc_summary=_minimal_lc_summary(),
        timing_series=TransitTimingPlotData(
            epochs=[1, 2],
            oc_seconds=[2.0, -2.0],
            snr=[8.0, 8.5],
            rms_seconds=3.0,
            periodicity_score=0.1,
            linear_trend_sec_per_epoch=0.0,
        ),
        alias_summary=alias,
        checks_run=[],
    ).to_json()
    variability = payload["summary"]["variability_summary"]
    alias_scalar = payload["summary"]["alias_scalar_summary"]

    assert variability["classification"] == "moderate_variability"
    assert "PERIODIC_SIGNAL_PRESENT" in variability["flags"]
    assert variability["rotation_context"]["status"] == "INCOMPLETE_INPUTS"
    assert "MISSING_ROTATION_PERIOD" in variability["rotation_context"]["quality_flags"]
    assert alias_scalar["alias_interpretation"] == "weak_alias_candidate"


def test_timing_summary_surfaces_v04_depth_stability_scalars() -> None:
    v04 = ok_result(
        id="V04",
        name="depth_stability",
        metrics={
            "n_transits_measured": 7,
            "depth_scatter_ppm": 145.2,
            "chi2_reduced": 1.8,
        },
    )
    payload = ReportData(
        checks={"V04": v04},
        timing_series=TransitTimingPlotData(
            epochs=[1, 2, 3],
            oc_seconds=[5.0, -4.0, 2.0],
            snr=[10.0, 9.0, 11.0],
            rms_seconds=4.0,
            periodicity_score=0.5,
            linear_trend_sec_per_epoch=0.1,
        ),
        checks_run=["V04"],
    ).to_json()
    timing = payload["summary"]["timing_summary"]

    assert timing["n_transits_measured"] == 7
    assert timing["depth_scatter_ppm"] == 145.2
    assert timing["chi2_reduced"] == 1.8


def test_variability_summary_rotation_context_uses_stellar_radius_when_available() -> None:
    payload = ReportData(
        lc_summary=_minimal_lc_summary(),
        stellar=StellarParams(radius=1.4, mass=1.2, tmag=9.1, teff=6400.0),
        checks_run=[],
    ).to_json()
    rotation_context = payload["summary"]["variability_summary"]["rotation_context"]
    assert rotation_context["stellar_radius_rsun"] == 1.4
    assert rotation_context["v_eq_est_kms"] is None
    assert "MISSING_ROTATION_PERIOD" in rotation_context["quality_flags"]


def test_variability_periodicity_falls_back_to_alias_when_timing_missing() -> None:
    alias = AliasHarmonicSummaryData(
        harmonic_labels=["P", "P/2", "2P"],
        periods=[4.0, 2.0, 8.0],
        scores=[0.9, 1.1, 0.2],
        harmonic_depth_ppm=[500.0, 620.0, 120.0],
        best_harmonic="P/2",
        best_ratio_over_p=0.5,
        classification="ALIAS_WEAK",
        phase_shift_event_count=0,
        phase_shift_peak_sigma=3.4,
        secondary_significance=0.2,
    )
    payload = ReportData(
        lc_summary=_minimal_lc_summary(),
        alias_summary=alias,
        checks_run=[],
    ).to_json()

    variability = payload["summary"]["variability_summary"]
    contamination = payload["summary"]["stellar_contamination_summary"]

    assert variability["periodicity_score"] == 3.4
    assert (
        variability["semantics"]["periodicity_source"]
        == "alias_summary.{classification,phase_shift_peak_sigma,secondary_significance,phase_shift_event_count}"
    )
    assert contamination["components"]["periodicity_score"]["raw_value"] == 3.4
    assert contamination["components"]["periodicity_score"]["transformed_value"] is not None
