"""Tests for the report module (Phase 1).

Covers:
- Unit test: ReportData construction + to_json() round-trip
- Integration test: synthetic LC with injected transit -> build_report()
- Schema test: JSON serialization key/type verification
- Plot data passthrough test: checks["V01"].raw["plot_data"] intact
- Enabled filter test: include_v03=True adds V03
- Downsampling test: max_lc_points respected, in-transit preserved
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bittr_tess_vetter.api.types import (
    Candidate,
    CheckResult,
    Ephemeris,
    LightCurve,
    StellarParams,
    VettingBundleResult,
    ok_result,
)
from bittr_tess_vetter.report import (
    AliasHarmonicSummaryData,
    EnrichmentBlockData,
    FullLCPlotData,
    LCSummary,
    PhaseFoldedPlotData,
    ReportData,
    ReportEnrichmentData,
    SecondaryScanPlotData,
    SecondaryScanQuality,
    TransitTimingPlotData,
    build_report,
)
from bittr_tess_vetter.report._build import (
    _bin_phase_data,
    _downsample_phase_preserving_transit,
    _downsample_preserving_transits,
    _validate_build_inputs,
)
from bittr_tess_vetter.report._data import _scrub_non_finite

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_box_transit_lc(
    *,
    period_days: float = 3.5,
    t0_btjd: float = 0.5,
    duration_hours: float = 2.5,
    baseline_days: float = 27.0,
    cadence_minutes: float = 10.0,
    depth_frac: float = 0.01,
    noise_ppm: float = 50.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic light curve with box-shaped transits."""
    rng = np.random.default_rng(seed)
    dt_days = cadence_minutes / (24.0 * 60.0)
    time = np.arange(0.0, baseline_days, dt_days, dtype=np.float64)
    flux = np.ones_like(time)
    flux += rng.normal(0.0, noise_ppm * 1e-6, size=time.size)
    flux_err = np.full_like(time, noise_ppm * 1e-6)

    duration_days = duration_hours / 24.0
    half_phase = (duration_days / period_days) / 2.0
    phase = ((time - t0_btjd) / period_days) % 1.0
    phase_dist = np.minimum(phase, 1.0 - phase)
    in_transit = phase_dist < half_phase

    flux[in_transit] *= 1.0 - depth_frac

    return time, flux, flux_err


def _assert_scalar_only_summary_block(block: dict[str, object], *, block_name: str) -> None:
    """Assert summary block has only scalar leaf values."""
    assert isinstance(block, dict), f"{block_name} must be a dict"
    scalar_types = (bool, int, float, str)
    for key, value in block.items():
        assert value is None or isinstance(value, scalar_types), (
            f"{block_name}.{key} must be scalar, got {type(value).__name__}"
        )


# ---------------------------------------------------------------------------
# Unit test: ReportData construction + to_json() round-trip
# ---------------------------------------------------------------------------


def test_report_data_to_json_round_trip() -> None:
    """Construct ReportData with mock data and round-trip through to_json()."""
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    lc_summary = LCSummary(
        n_points=1000,
        n_valid=990,
        n_transits=7,
        n_in_transit_total=70,
        duration_days=27.0,
        cadence_seconds=120.0,
        flux_std_ppm=100.0,
        flux_mad_ppm=90.0,
        gap_fraction=0.01,
        snr=25.0,
        depth_ppm=10000.0,
        depth_err_ppm=400.0,
    )

    full_lc = FullLCPlotData(
        time=[1.0, 2.0, 3.0],
        flux=[1.0, 0.99, 1.0],
        transit_mask=[False, True, False],
    )

    phase_folded = PhaseFoldedPlotData(
        phase=[-0.1, 0.0, 0.1],
        flux=[1.0, 0.99, 1.0],
        bin_centers=[-0.05, 0.05],
        bin_flux=[0.995, 0.995],
        bin_err=[0.001, 0.001],
        bin_minutes=30.0,
        transit_duration_phase=0.0298,  # 2.5h / (3.5d * 24)
        phase_range=(-0.0893, 0.0893),  # ±3 * transit_duration_phase
    )

    mock_check = ok_result(
        id="V01",
        name="odd_even_depth",
        metrics={"rel_diff": 0.05},
        confidence=0.9,
    )

    bundle = VettingBundleResult.from_checks([mock_check])

    report = ReportData(
        tic_id=12345678,
        toi="TOI-1234.01",
        candidate=candidate,
        lc_summary=lc_summary,
        checks={"V01": mock_check},
        bundle=bundle,
        full_lc=full_lc,
        phase_folded=phase_folded,
        checks_run=["V01"],
    )

    j = report.to_json()

    # Round-trip through JSON serialization
    serialized = json.dumps(j)
    deserialized = json.loads(serialized)

    assert deserialized["schema_version"] == "1.0.0"
    assert deserialized["summary"]["tic_id"] == 12345678
    assert deserialized["summary"]["toi"] == "TOI-1234.01"
    assert deserialized["summary"]["checks_run"] == ["V01"]
    assert deserialized["summary"]["ephemeris"]["period_days"] == 3.5
    assert deserialized["summary"]["input_depth_ppm"] == 10000.0
    assert deserialized["summary"]["lc_summary"]["snr"] == 25.0
    assert "V01" in deserialized["summary"]["checks"]
    assert deserialized["summary"]["bundle_summary"]["n_ok"] == 1
    assert deserialized["summary"]["bundle_summary"]["n_failed"] == 0
    assert deserialized["plot_data"]["full_lc"]["time"] == [1.0, 2.0, 3.0]
    assert deserialized["plot_data"]["phase_folded"]["bin_minutes"] == 30.0


# ---------------------------------------------------------------------------
# Integration test: synthetic LC -> build_report() -> verify fields
# ---------------------------------------------------------------------------


def test_build_report_integration() -> None:
    """Build report from synthetic LC with injected transit."""
    time, flux, flux_err = _make_box_transit_lc(
        depth_frac=0.01, noise_ppm=50.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate, tic_id=99999, toi="TOI-9999.01")

    # Identity
    assert report.tic_id == 99999
    assert report.toi == "TOI-9999.01"
    assert report.version == "1.0.0"

    # Candidate stored for provenance
    assert report.candidate is candidate

    # LC Summary populated
    assert report.lc_summary is not None
    assert report.lc_summary.n_points > 0
    assert report.lc_summary.n_valid > 0
    assert report.lc_summary.n_transits > 0
    assert report.lc_summary.snr > 0.0
    assert report.lc_summary.depth_ppm > 0.0
    assert report.lc_summary.flux_std_ppm > 0.0
    assert report.lc_summary.cadence_seconds > 0.0

    # Checks populated (default: V01, V02, V04, V05, V13, V15)
    assert set(report.checks.keys()) == {"V01", "V02", "V04", "V05", "V13", "V15"}
    assert report.checks_run == ["V01", "V02", "V04", "V05", "V13", "V15"]

    # Bundle populated
    assert report.bundle is not None
    assert len(report.bundle.results) == 6

    # Plot data populated
    assert report.full_lc is not None
    assert len(report.full_lc.time) > 0
    assert len(report.full_lc.flux) == len(report.full_lc.time)
    assert len(report.full_lc.transit_mask) == len(report.full_lc.time)
    assert any(report.full_lc.transit_mask)  # some in-transit points

    assert report.phase_folded is not None
    assert len(report.phase_folded.phase) > 0
    assert len(report.phase_folded.bin_centers) > 0
    assert report.phase_folded.bin_minutes == 30.0
    assert report.phase_folded.y_range_suggested is not None
    assert report.phase_folded.depth_reference_flux is not None
    assert report.per_transit_stack is not None
    assert report.local_detrend is not None
    assert report.oot_context is not None
    assert report.timing_series is not None
    assert report.alias_summary is not None
    assert report.lc_robustness is not None
    assert report.odd_even_phase is not None
    assert report.secondary_scan is not None
    assert report.secondary_scan.quality is not None
    assert report.secondary_scan.render_hints is not None


# ---------------------------------------------------------------------------
# Schema test: JSON serialization key/type verification
# ---------------------------------------------------------------------------


def test_json_schema_keys_and_types() -> None:
    """Verify all expected keys present and types correct in JSON output."""
    time, flux, flux_err = _make_box_transit_lc()
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate)
    j = report.to_json()

    assert isinstance(j["schema_version"], str)
    assert isinstance(j["summary"], dict)
    assert isinstance(j["plot_data"], dict)
    assert isinstance(j["payload_meta"], dict)

    s = j["summary"]
    p = j["plot_data"]
    assert s.get("tic_id") is None  # not provided
    assert s.get("toi") is None
    assert isinstance(s["checks_run"], list)
    assert isinstance(s["checks"], dict)
    assert isinstance(s["references"], list)
    assert isinstance(s["odd_even_summary"], dict)
    assert isinstance(s["noise_summary"], dict)
    assert isinstance(s["variability_summary"], dict)

    # Ephemeris
    assert isinstance(s["ephemeris"], dict)
    assert isinstance(s["ephemeris"]["period_days"], float)
    assert isinstance(s["ephemeris"]["t0_btjd"], float)
    assert isinstance(s["ephemeris"]["duration_hours"], float)

    # LC Summary
    assert isinstance(s["lc_summary"], dict)
    lc_sum = s["lc_summary"]
    assert isinstance(lc_sum["n_points"], int)
    assert isinstance(lc_sum["n_valid"], int)
    assert isinstance(lc_sum["n_transits"], int)
    assert isinstance(lc_sum["n_in_transit_total"], int)
    assert isinstance(lc_sum["duration_days"], float)
    assert isinstance(lc_sum["cadence_seconds"], float)
    assert isinstance(lc_sum["flux_std_ppm"], float)
    assert isinstance(lc_sum["flux_mad_ppm"], float)
    assert isinstance(lc_sum["gap_fraction"], float)
    assert isinstance(lc_sum["snr"], float)
    assert isinstance(lc_sum["depth_ppm"], float)
    # depth_err_ppm can be float or None (None if unmeasurable)
    assert lc_sum["depth_err_ppm"] is None or isinstance(lc_sum["depth_err_ppm"], float)

    # Bundle summary
    assert isinstance(s["bundle_summary"], dict)
    assert isinstance(s["bundle_summary"]["n_checks"], int)
    assert isinstance(s["bundle_summary"]["n_ok"], int)
    assert isinstance(s["bundle_summary"]["n_failed"], int)
    assert isinstance(s["bundle_summary"]["n_skipped"], int)
    assert isinstance(s["bundle_summary"]["failed_ids"], list)
    for check_summary in s["checks"].values():
        assert isinstance(check_summary["method_refs"], list)

    # References should be deduped and sorted by key.
    ref_keys = [ref["key"] for ref in s["references"]]
    assert ref_keys == sorted(set(ref_keys))

    # Deterministic summary blocks.
    assert isinstance(s["odd_even_summary"]["flags"], list)
    assert isinstance(s["noise_summary"]["flags"], list)
    assert isinstance(s["noise_summary"]["semantics"], dict)
    assert s["noise_summary"]["trend_stat"] is None or isinstance(
        s["noise_summary"]["trend_stat"], float
    )
    assert isinstance(s["noise_summary"]["trend_stat_unit"], str)
    assert s["noise_summary"]["trend_stat_unit"] == "relative_flux_per_day"
    assert isinstance(s["variability_summary"]["flags"], list)
    assert isinstance(s["variability_summary"]["semantics"], dict)
    assert isinstance(s["variability_summary"]["classification"], str)

    # Full LC plot data
    assert isinstance(p["full_lc"], dict)
    assert isinstance(p["full_lc"]["time"], list)
    assert isinstance(p["full_lc"]["flux"], list)
    assert isinstance(p["full_lc"]["transit_mask"], list)

    # Phase-folded plot data
    assert isinstance(p["phase_folded"], dict)
    assert isinstance(p["phase_folded"]["phase"], list)
    assert isinstance(p["phase_folded"]["flux"], list)
    assert isinstance(p["phase_folded"]["bin_centers"], list)
    assert isinstance(p["phase_folded"]["bin_flux"], list)
    assert isinstance(p["phase_folded"]["bin_err"], list)
    assert isinstance(p["phase_folded"]["bin_minutes"], float)
    assert isinstance(p["phase_folded"]["transit_duration_phase"], float)
    assert isinstance(p["phase_folded"]["phase_range"], (list, tuple))
    assert len(p["phase_folded"]["phase_range"]) == 2
    assert (
        p["phase_folded"]["y_range_suggested"] is None
        or isinstance(p["phase_folded"]["y_range_suggested"], (list, tuple))
    )
    if p["phase_folded"]["y_range_suggested"] is not None:
        assert len(p["phase_folded"]["y_range_suggested"]) == 2
    assert (
        p["phase_folded"]["depth_reference_flux"] is None
        or isinstance(p["phase_folded"]["depth_reference_flux"], float)
    )

    # Additional LC-only panels
    assert isinstance(p["per_transit_stack"], dict)
    assert isinstance(p["per_transit_stack"]["windows"], list)
    assert isinstance(p["per_transit_stack"]["window_half_hours"], float)
    assert isinstance(p["local_detrend"], dict)
    assert isinstance(p["local_detrend"]["windows"], list)
    assert isinstance(p["local_detrend"]["window_half_hours"], float)
    assert isinstance(p["local_detrend"]["baseline_method"], str)
    assert isinstance(p["oot_context"], dict)
    assert isinstance(p["oot_context"]["flux_sample"], list)
    assert isinstance(p["oot_context"]["flux_residual_ppm_sample"], list)
    assert isinstance(p["oot_context"]["hist_centers"], list)
    assert isinstance(p["oot_context"]["hist_counts"], list)
    assert isinstance(p["oot_context"]["n_oot_points"], int)
    assert isinstance(p["timing_series"], dict)
    assert isinstance(p["timing_series"]["epochs"], list)
    assert isinstance(p["timing_series"]["oc_seconds"], list)
    assert isinstance(p["timing_series"]["snr"], list)
    assert (
        p["timing_series"]["periodicity_score"] is None
        or isinstance(p["timing_series"]["periodicity_score"], float)
    )
    assert isinstance(p["alias_summary"], dict)
    assert isinstance(p["alias_summary"]["harmonic_labels"], list)
    assert isinstance(p["alias_summary"]["periods"], list)
    assert isinstance(p["alias_summary"]["scores"], list)
    assert isinstance(p["alias_summary"]["best_harmonic"], str)
    assert isinstance(p["alias_summary"]["best_ratio_over_p"], float)
    assert isinstance(p["lc_robustness"], dict)
    assert isinstance(p["lc_robustness"]["version"], str)
    assert isinstance(p["lc_robustness"]["per_epoch"], list)
    assert isinstance(p["lc_robustness"]["robustness"], dict)
    assert isinstance(p["lc_robustness"]["red_noise"], dict)
    assert isinstance(p["lc_robustness"]["fp_signals"], dict)
    assert isinstance(p["odd_even_phase"], dict)
    assert isinstance(p["odd_even_phase"]["odd_phase"], list)
    assert isinstance(p["odd_even_phase"]["even_phase"], list)
    assert isinstance(p["secondary_scan"], dict)
    assert isinstance(p["secondary_scan"]["phase"], list)
    assert isinstance(p["secondary_scan"]["bin_centers"], list)
    assert isinstance(p["secondary_scan"]["quality"], dict)
    assert isinstance(p["secondary_scan"]["render_hints"], dict)
    assert isinstance(p["secondary_scan"]["quality"]["phase_coverage_fraction"], float)
    assert isinstance(p["secondary_scan"]["quality"]["largest_phase_gap"], float)
    assert isinstance(p["secondary_scan"]["quality"]["flags"], list)
    assert isinstance(p["secondary_scan"]["render_hints"]["style_mode"], str)
    assert isinstance(p["secondary_scan"]["render_hints"]["connect_bins"], bool)
    assert isinstance(p["secondary_scan"]["render_hints"]["error_bar_stride"], int)

    # Round-trip through json.dumps should work without custom encoders
    serialized = json.dumps(j)
    assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Plot data passthrough test
# ---------------------------------------------------------------------------


def test_check_plot_data_passthrough() -> None:
    """Verify checks["V01"].raw["plot_data"] is intact when present."""
    time, flux, flux_err = _make_box_transit_lc()
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate)

    # V01 should be present
    assert "V01" in report.checks
    v01 = report.checks["V01"]
    assert isinstance(v01, CheckResult)

    # The raw field should contain plot_data if the check produces it
    if v01.raw is not None and "plot_data" in v01.raw:
        plot_data = v01.raw["plot_data"]
        assert isinstance(plot_data, dict)

        # Serialize the report and verify plot_data survives
        j = report.to_json()
        assert "check_overlays" in j["plot_data"]
        assert "V01" in j["plot_data"]["check_overlays"]


def test_summary_references_cover_method_refs() -> None:
    """All method refs attached to checks should resolve in summary.references."""
    time, flux, flux_err = _make_box_transit_lc()
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    j = build_report(lc, candidate).to_json()
    ref_keys = {ref["key"] for ref in j["summary"]["references"]}
    method_refs = {
        ref
        for check in j["summary"]["checks"].values()
        for ref in check.get("method_refs", [])
    }
    assert method_refs.issubset(ref_keys)


def test_alias_scalar_summary_regression_deterministic_and_scalar_only() -> None:
    """Alias scalar summary should be deterministic, scalar-only, and compact."""
    report = ReportData(
        alias_summary=AliasHarmonicSummaryData(
            harmonic_labels=["P", "P/2", "2P"],
            periods=[10.0, 5.0, 20.0],
            scores=[0.6, 0.2, 0.1],
            best_harmonic="P",
            best_ratio_over_p=1.0,
        ),
        checks_run=[],
    )

    payload_a = report.to_json()
    payload_b = report.to_json()

    block_a = payload_a["summary"]["alias_scalar_summary"]
    block_b = payload_b["summary"]["alias_scalar_summary"]
    assert block_a == block_b, "alias_scalar_summary must be deterministic"

    assert set(block_a.keys()) == {
        "best_harmonic",
        "best_ratio_over_p",
        "score_p",
        "score_p_over_2",
        "score_2p",
        "depth_ppm_peak",
    }
    assert block_a["best_harmonic"] == "P"
    assert block_a["best_ratio_over_p"] == pytest.approx(1.0)
    assert block_a["score_p"] == pytest.approx(0.6)
    assert block_a["score_p_over_2"] == pytest.approx(0.2)
    assert block_a["score_2p"] == pytest.approx(0.1)
    assert (
        block_a["depth_ppm_peak"] is None
        or isinstance(block_a["depth_ppm_peak"], float)
    )

    _assert_scalar_only_summary_block(block_a, block_name="alias_scalar_summary")


def test_timing_summary_regression_rules_and_scalar_only() -> None:
    """Timing summary should enforce deterministic denominator and tie-break rules."""
    report = ReportData(
        timing_series=TransitTimingPlotData(
            epochs=[4, 1, 3],
            oc_seconds=[15.0, -30.0, 30.0],
            snr=[8.0, 12.0, 10.0],
            rms_seconds=25.0,
            periodicity_score=0.5,
            linear_trend_sec_per_epoch=0.2,
        ),
        checks_run=[],
    )

    payload_a = report.to_json()
    payload_b = report.to_json()

    block_a = payload_a["summary"]["timing_summary"]
    block_b = payload_b["summary"]["timing_summary"]
    assert block_a == block_b, "timing_summary must be deterministic"

    assert set(block_a.keys()) == {
        "n_epochs_measured",
        "rms_seconds",
        "periodicity_score",
        "linear_trend_sec_per_epoch",
        "max_abs_oc_seconds",
        "max_snr",
        "outlier_count",
        "outlier_fraction",
        "deepest_epoch",
    }
    if block_a["n_epochs_measured"] > 0 and block_a["outlier_fraction"] is not None:
        expected_fraction = block_a["outlier_count"] / block_a["n_epochs_measured"]
        assert block_a["outlier_fraction"] == pytest.approx(expected_fraction)
    assert block_a["deepest_epoch"] == 1
    assert block_a["rms_seconds"] == pytest.approx(25.0)
    assert block_a["linear_trend_sec_per_epoch"] == pytest.approx(0.2)
    assert block_a["max_abs_oc_seconds"] == pytest.approx(30.0)

    _assert_scalar_only_summary_block(block_a, block_name="timing_summary")


def test_secondary_scan_summary_regression_naming_unit_and_scalar_only() -> None:
    """Secondary scan summary should expose ppm depth naming and scalar-only fields."""
    report = ReportData(
        secondary_scan=SecondaryScanPlotData(
            phase=[-0.2, 0.0, 0.2],
            flux=[1.0, 0.9985, 1.0],
            bin_centers=[-0.1, 0.1],
            bin_flux=[0.999, 1.0],
            bin_err=[0.0002, 0.0002],
            bin_minutes=30.0,
            primary_phase=0.0,
            secondary_phase=0.5,
            strongest_dip_phase=0.0,
            strongest_dip_flux=0.9985,
            quality=SecondaryScanQuality(
                n_raw_points=3,
                n_bins=2,
                n_bins_with_error=2,
                phase_coverage_fraction=0.4,
                largest_phase_gap=0.6,
                is_degraded=True,
                flags=["LOW_BIN_COUNT"],
            ),
        ),
        checks_run=[],
    )

    payload_a = report.to_json()
    payload_b = report.to_json()

    block_a = payload_a["summary"]["secondary_scan_summary"]
    block_b = payload_b["summary"]["secondary_scan_summary"]
    assert block_a == block_b, "secondary_scan_summary must be deterministic"

    assert set(block_a.keys()) == {
        "phase_coverage_fraction",
        "largest_phase_gap",
        "n_bins_with_error",
        "strongest_dip_phase",
        "strongest_dip_depth_ppm",
        "is_degraded",
        "quality_flag_count",
    }
    assert "strongest_dip_flux" not in block_a
    assert block_a["strongest_dip_depth_ppm"] == pytest.approx(1500.0)
    assert block_a["quality_flag_count"] == 1

    _assert_scalar_only_summary_block(block_a, block_name="secondary_scan_summary")


# ---------------------------------------------------------------------------
# Enabled filter test: include_v03=True adds V03
# ---------------------------------------------------------------------------


def test_include_v03_adds_v03() -> None:
    """Verify include_v03=True with stellar params includes V03 in results."""
    time, flux, flux_err = _make_box_transit_lc()
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)
    stellar = StellarParams(radius=1.0, mass=1.0)

    report = build_report(lc, candidate, include_v03=True, stellar=stellar)

    assert "V03" in report.checks
    assert "V03" in report.checks_run
    assert set(report.checks.keys()) == {"V01", "V02", "V03", "V04", "V05", "V13", "V15"}


def test_include_v03_without_stellar_disables() -> None:
    """Verify include_v03=True without stellar auto-disables V03 with warning."""
    time, flux, flux_err = _make_box_transit_lc()
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate, include_v03=True)

    assert "V03" not in report.checks
    assert "V03" not in report.checks_run


def test_default_excludes_v03() -> None:
    """Verify default behavior excludes V03."""
    time, flux, flux_err = _make_box_transit_lc()
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate)

    assert "V03" not in report.checks
    assert "V03" not in report.checks_run


# ---------------------------------------------------------------------------
# Downsampling test
# ---------------------------------------------------------------------------


def test_downsampling_respects_max_points_and_preserves_transits() -> None:
    """Verify max_lc_points is respected and in-transit points preserved."""
    time, flux, flux_err = _make_box_transit_lc(
        baseline_days=90.0,
        cadence_minutes=2.0,
        depth_frac=0.01,
        noise_ppm=50.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    # Use a small max to force downsampling
    max_pts = 5000
    report = build_report(lc, candidate, max_lc_points=max_pts)

    assert report.full_lc is not None
    assert len(report.full_lc.time) <= max_pts

    # All in-transit points should be preserved
    n_in_transit = sum(report.full_lc.transit_mask)
    assert n_in_transit > 0

    # Verify the transit mask marks actual transits (flux dips)
    transit_flux = [
        f for f, m in zip(report.full_lc.flux, report.full_lc.transit_mask, strict=True) if m
    ]
    oot_flux = [
        f for f, m in zip(report.full_lc.flux, report.full_lc.transit_mask, strict=True) if not m
    ]
    assert np.mean(transit_flux) < np.mean(oot_flux)


def test_no_downsampling_when_under_limit() -> None:
    """Verify no downsampling when data is under max_lc_points."""
    time, flux, flux_err = _make_box_transit_lc(
        baseline_days=27.0,
        cadence_minutes=10.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate, max_lc_points=50_000)

    assert report.full_lc is not None
    # All valid points should be present (no downsampling)
    valid = np.isfinite(time) & np.isfinite(flux)
    assert len(report.full_lc.time) == int(np.sum(valid))


# ---------------------------------------------------------------------------
# SNR sanity check
# ---------------------------------------------------------------------------


def test_snr_reasonable_for_deep_transit() -> None:
    """Verify SNR is significant for a deep, low-noise transit."""
    time, flux, flux_err = _make_box_transit_lc(
        depth_frac=0.01,
        noise_ppm=50.0,
        baseline_days=27.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate)

    assert report.lc_summary is not None
    # 10000 ppm depth with 50 ppm noise over 27 days -> SNR should be high
    assert report.lc_summary.snr > 10.0
    # Depth should be close to 10000 ppm
    assert abs(report.lc_summary.depth_ppm - 10000.0) < 2000.0


# ---------------------------------------------------------------------------
# Edge case: NaN safety in JSON output
# ---------------------------------------------------------------------------


def test_nan_safety_in_json_output() -> None:
    """Verify to_json() produces valid JSON with no NaN/Inf values."""
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    # Manually construct LCSummary with depth_err_ppm=None (as if detect_transit failed)
    lc_summary = LCSummary(
        n_points=1000,
        n_valid=990,
        n_transits=7,
        n_in_transit_total=70,
        duration_days=27.0,
        cadence_seconds=120.0,
        flux_std_ppm=100.0,
        flux_mad_ppm=90.0,
        gap_fraction=0.01,
        snr=0.0,
        depth_ppm=0.0,
        depth_err_ppm=None,
    )

    report = ReportData(
        candidate=candidate,
        lc_summary=lc_summary,
        checks_run=[],
    )

    j = report.to_json()

    # Must round-trip through json.dumps with allow_nan=False (strict RFC 8259)
    serialized = json.dumps(j, allow_nan=False)
    assert isinstance(serialized, str)

    # depth_err_ppm should be None, not NaN
    assert j["summary"]["lc_summary"]["depth_err_ppm"] is None


# ---------------------------------------------------------------------------
# Edge case: flux_err=None
# ---------------------------------------------------------------------------


def test_build_report_with_flux_err_none() -> None:
    """Verify build_report works when flux_err is None."""
    time, flux, _ = _make_box_transit_lc(
        depth_frac=0.01, noise_ppm=50.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=None)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate)

    assert report.lc_summary is not None
    assert report.lc_summary.n_points > 0
    assert report.lc_summary.n_valid > 0

    # JSON round-trip should work
    j = report.to_json()
    serialized = json.dumps(j, allow_nan=False)
    assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Edge case: single transit visible
# ---------------------------------------------------------------------------


def test_build_report_single_transit() -> None:
    """Verify build_report succeeds with only ~1 transit in the data."""
    time, flux, flux_err = _make_box_transit_lc(
        period_days=25.0,  # long period -> only ~1 transit in 27-day baseline
        t0_btjd=13.0,
        duration_hours=3.0,
        baseline_days=27.0,
        cadence_minutes=10.0,
        depth_frac=0.01,
        noise_ppm=50.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=25.0, t0_btjd=13.0, duration_hours=3.0)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate)

    assert report.lc_summary is not None
    assert report.lc_summary.n_points > 0
    assert report.full_lc is not None
    assert report.phase_folded is not None

    # JSON round-trip should work
    j = report.to_json()
    serialized = json.dumps(j, allow_nan=False)
    assert isinstance(serialized, str)


def test_phase_depth_reference_absent_without_depth_ppm() -> None:
    """Depth reference should be omitted when candidate depth_ppm is unavailable."""
    time, flux, flux_err = _make_box_transit_lc()
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph)  # no depth_ppm

    report = build_report(lc, candidate)

    assert report.phase_folded is not None
    assert report.phase_folded.depth_reference_flux is None


def test_build_report_without_additional_plots() -> None:
    """Payload control: additional panels can be disabled."""
    time, flux, flux_err = _make_box_transit_lc()
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate, include_additional_plots=False)
    assert report.per_transit_stack is None
    assert report.odd_even_phase is None
    assert report.secondary_scan is None
    assert report.lc_robustness is not None
    j = report.to_json()
    assert "per_transit_stack" not in j["plot_data"]
    assert "odd_even_phase" not in j["plot_data"]
    assert "secondary_scan" not in j["plot_data"]
    assert "lc_robustness" in j["plot_data"]


def test_build_report_without_lc_robustness() -> None:
    """Payload control: lc_robustness block can be disabled."""
    time, flux, flux_err = _make_box_transit_lc()
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report = build_report(lc, candidate, include_lc_robustness=False)
    assert report.lc_robustness is None
    j = report.to_json()
    assert "lc_robustness" not in j["plot_data"]


def test_secondary_scan_preserves_full_orbit_phase_coverage() -> None:
    """Secondary scan should remain full-orbit, not transit-window focused."""
    time, flux, flux_err = _make_box_transit_lc(
        baseline_days=90.0,
        cadence_minutes=2.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    # Force downsampling so we exercise the secondary-scan decimator.
    report = build_report(lc, candidate, max_phase_points=2_000)
    assert report.secondary_scan is not None
    phase = np.asarray(report.secondary_scan.phase, dtype=np.float64)
    assert len(phase) > 0

    # Full-orbit scan should retain broad phase coverage.
    assert float(np.min(phase)) < -0.4
    assert float(np.max(phase)) > 0.4
    q = report.secondary_scan.quality
    assert q is not None
    assert 0.0 <= q.phase_coverage_fraction <= 1.0
    assert q.largest_phase_gap >= 0.0
    hints = report.secondary_scan.render_hints
    assert hints is not None
    assert hints.style_mode in {"normal", "degraded"}


def test_secondary_scan_bins_independent_of_display_downsampling() -> None:
    """Binned secondary-scan diagnostics should be stable vs raw-point caps."""
    time, flux, flux_err = _make_box_transit_lc(
        baseline_days=90.0,
        cadence_minutes=2.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    report_low = build_report(lc, candidate, max_phase_points=800)
    report_high = build_report(lc, candidate, max_phase_points=8_000)
    assert report_low.secondary_scan is not None
    assert report_high.secondary_scan is not None

    low = report_low.secondary_scan
    high = report_high.secondary_scan
    assert low.bin_centers == high.bin_centers
    assert low.bin_flux == high.bin_flux
    assert low.strongest_dip_phase == high.strongest_dip_phase


# ---------------------------------------------------------------------------
# Edge case: all points in transit for downsampling
# ---------------------------------------------------------------------------


def test_downsample_all_in_transit() -> None:
    """Verify _downsample_preserving_transits hard-caps all-in-transit data."""
    n = 100
    time = np.linspace(0.0, 1.0, n)
    flux = np.ones(n)
    transit_mask = np.ones(n, dtype=bool)  # all in transit

    t_out, f_out, m_out = _downsample_preserving_transits(
        time, flux, transit_mask, max_points=50
    )

    # Hard cap: output should not exceed max_points
    assert len(t_out) <= 50
    assert all(m_out)  # all should still be marked in-transit


# ---------------------------------------------------------------------------
# Empty bin filtering test
# ---------------------------------------------------------------------------


def test_bin_phase_data_filters_empty_bins() -> None:
    """Verify _bin_phase_data only returns bins that contain data."""
    # Create sparse phase data: points clustered near phase=0 only,
    # leaving large gaps across the full phase range [-0.5, 0.5].
    rng = np.random.default_rng(123)
    n = 50
    phase = rng.uniform(-0.02, 0.02, size=n)
    phase = np.sort(phase)
    flux = np.ones(n) + rng.normal(0, 1e-4, size=n)

    # Use a long period so bins are narrow in phase -> many empty bins
    period_days = 100.0
    bin_minutes = 30.0

    centers, fluxes, errors = _bin_phase_data(phase, flux, period_days, bin_minutes)

    # All returned bins must have data
    assert len(centers) > 0
    assert len(centers) == len(fluxes) == len(errors)

    # With data only near phase=0, the number of populated bins should
    # be much less than the total number of possible bins across [-0.5, 0.5].
    bin_phase = (bin_minutes / 60.0 / 24.0) / period_days
    phase_range = float(np.max(phase) - np.min(phase))
    max_possible_bins = max(1, int(np.ceil(phase_range / bin_phase)))

    # Bins returned should equal max_possible_bins (since data is tightly
    # clustered, every bin in the data range has points).  The key check
    # is that we do NOT get bins for the full [-0.5, 0.5] range.
    assert len(centers) <= max_possible_bins

    # Bin centers should all be near 0 (where the data is)
    for c in centers:
        assert abs(c) < 0.05, f"Bin center {c} is far from the data cluster"


# ---------------------------------------------------------------------------
# Phase-folded downsampling tests
# ---------------------------------------------------------------------------


def test_phase_downsample_respects_max_and_preserves_transit() -> None:
    """Verify max_phase_points is respected and near-transit points preserved."""
    time, flux, flux_err = _make_box_transit_lc(
        baseline_days=90.0,
        cadence_minutes=2.0,
        depth_frac=0.01,
        noise_ppm=50.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    # With period=3.5d and duration=2.5h, transit_duration_phase ~0.0298.
    # The display window is ±3*0.0298 = ±0.0893 phase.  Set max high
    # enough so far-from-transit points get thinned but the budget
    # isn't exceeded by near-transit alone.
    max_phase = 15_000
    report = build_report(lc, candidate, max_phase_points=max_phase)

    assert report.phase_folded is not None
    n_raw = len(report.phase_folded.phase)
    assert n_raw <= max_phase
    # Confirm actual downsampling happened (original is ~64800)
    assert n_raw < 64_800

    # Near-transit points (within display window) should all be preserved
    phases = np.array(report.phase_folded.phase)
    half_win = report.phase_folded.phase_range[1]
    near_transit_count = int(np.sum(np.abs(phases) < half_win))
    assert near_transit_count > 0

    # Bins are computed within the display window before downsampling
    assert len(report.phase_folded.bin_centers) > 0


def test_phase_no_downsample_when_under_limit() -> None:
    """Verify no phase downsampling when data is under max_phase_points."""
    time, flux, flux_err = _make_box_transit_lc(
        baseline_days=27.0,
        cadence_minutes=10.0,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10000.0)

    # With 27 days at 10-min cadence -> ~3888 points, well under 10k
    report = build_report(lc, candidate, max_phase_points=10_000)

    assert report.phase_folded is not None
    # All valid points should be present (no downsampling)
    valid = np.isfinite(time) & np.isfinite(flux)
    assert len(report.phase_folded.phase) == int(np.sum(valid))


def test_downsample_phase_preserving_transit_unit() -> None:
    """Unit test for _downsample_phase_preserving_transit."""
    rng = np.random.default_rng(99)
    # 500 points across full phase range, 100 near transit
    far_phase = rng.uniform(-0.5, -0.1, size=250)
    far_phase = np.concatenate([far_phase, rng.uniform(0.1, 0.5, size=250)])
    near_phase = rng.uniform(-0.09, 0.09, size=100)
    phase = np.sort(np.concatenate([far_phase, near_phase]))
    flux = np.ones_like(phase)

    # Ask for 200 points total: all 100 near-transit + 100 far
    p_out, f_out = _downsample_phase_preserving_transit(phase, flux, max_points=200)

    assert len(p_out) <= 200
    # All near-transit points preserved
    near_in_output = int(np.sum(np.abs(p_out) < 0.1))
    assert near_in_output == 100

    # Far points were thinned
    far_in_output = int(np.sum(np.abs(p_out) >= 0.1))
    assert far_in_output <= 100


def test_downsample_phase_all_near_transit() -> None:
    """Verify hard cap when all points are near transit center."""
    phase = np.linspace(-0.05, 0.05, 300)
    flux = np.ones(300)

    # All points are within ±0.1 -> hard cap uniformly thins near-transit
    p_out, f_out = _downsample_phase_preserving_transit(phase, flux, max_points=100)

    assert len(p_out) <= 100


# ---------------------------------------------------------------------------
# _validate_build_inputs tests
# ---------------------------------------------------------------------------

_VALID_EPH = Ephemeris(period_days=3.5, t0_btjd=1000.0, duration_hours=2.5)


def test_validate_rejects_zero_max_lc_points() -> None:
    with pytest.raises(ValueError, match="max_lc_points"):
        _validate_build_inputs(
            _VALID_EPH, 30.0, max_lc_points=0, max_phase_points=100,
            max_transit_windows=24, max_points_per_window=300,
        )


def test_validate_rejects_negative_max_lc_points() -> None:
    with pytest.raises(ValueError, match="max_lc_points"):
        _validate_build_inputs(
            _VALID_EPH, 30.0, max_lc_points=-1, max_phase_points=100,
            max_transit_windows=24, max_points_per_window=300,
        )


def test_validate_rejects_zero_max_phase_points() -> None:
    with pytest.raises(ValueError, match="max_phase_points"):
        _validate_build_inputs(
            _VALID_EPH, 30.0, max_lc_points=100, max_phase_points=0,
            max_transit_windows=24, max_points_per_window=300,
        )


def test_validate_rejects_negative_max_phase_points() -> None:
    with pytest.raises(ValueError, match="max_phase_points"):
        _validate_build_inputs(
            _VALID_EPH, 30.0, max_lc_points=100, max_phase_points=-5,
            max_transit_windows=24, max_points_per_window=300,
        )


@pytest.mark.parametrize("value", [1000.0, 1.5, float("nan"), float("inf"), True, False])
def test_validate_rejects_non_integer_max_lc_points(value: float) -> None:
    with pytest.raises(ValueError, match="max_lc_points"):
        _validate_build_inputs(
            _VALID_EPH, 30.0, max_lc_points=value, max_phase_points=100,
            max_transit_windows=24, max_points_per_window=300,
        )


@pytest.mark.parametrize("value", [1000.0, 1.5, float("nan"), float("inf"), True, False])
def test_validate_rejects_non_integer_max_phase_points(value: float) -> None:
    with pytest.raises(ValueError, match="max_phase_points"):
        _validate_build_inputs(
            _VALID_EPH, 30.0, max_lc_points=100, max_phase_points=value,
            max_transit_windows=24, max_points_per_window=300,
        )


@pytest.mark.parametrize("field,value", [
    ("max_transit_windows", 0),
    ("max_transit_windows", -1),
    ("max_points_per_window", 0),
    ("max_points_per_window", -10),
])
def test_validate_rejects_non_positive_transit_window_budgets(field: str, value: int) -> None:
    kwargs = {
        "max_lc_points": 100,
        "max_phase_points": 100,
        "max_transit_windows": 24,
        "max_points_per_window": 300,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        _validate_build_inputs(_VALID_EPH, 30.0, **kwargs)


@pytest.mark.parametrize("field,value", [
    ("period_days", float("nan")),
    ("period_days", float("inf")),
    ("duration_hours", float("nan")),
    ("duration_hours", float("inf")),
    ("t0_btjd", float("nan")),
    ("t0_btjd", float("inf")),
])
def test_validate_rejects_non_finite_ephemeris(field: str, value: float) -> None:
    kwargs = {"period_days": 3.5, "t0_btjd": 1000.0, "duration_hours": 2.5}
    kwargs[field] = value
    eph = Ephemeris(**kwargs)
    with pytest.raises(ValueError, match=field):
        _validate_build_inputs(
            eph, 30.0, max_lc_points=100, max_phase_points=100,
            max_transit_windows=24, max_points_per_window=300,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_validate_rejects_bad_bin_minutes(value: float) -> None:
    with pytest.raises(ValueError, match="bin_minutes"):
        _validate_build_inputs(
            _VALID_EPH, value, max_lc_points=100, max_phase_points=100,
            max_transit_windows=24, max_points_per_window=300,
        )


def test_validate_accepts_valid_inputs() -> None:
    """Smoke test: valid inputs should not raise."""
    _validate_build_inputs(
        _VALID_EPH, 30.0, max_lc_points=50_000, max_phase_points=10_000,
        max_transit_windows=24, max_points_per_window=300,
    )


# ---------------------------------------------------------------------------
# _scrub_non_finite tuple handling
# ---------------------------------------------------------------------------


def test_scrub_non_finite_handles_tuples() -> None:
    data = {"phase_range": (float("nan"), float("inf"), 0.5, -0.5)}
    result = _scrub_non_finite(data)
    assert result["phase_range"] == (None, None, 0.5, -0.5)
    assert isinstance(result["phase_range"], tuple)


def test_reportdata_serializes_enrichment_when_present() -> None:
    report = ReportData(
        enrichment=ReportEnrichmentData(
            version="0.1.0",
            pixel_diagnostics=EnrichmentBlockData(
                status="skipped",
                flags=["NOT_IMPLEMENTED"],
                quality={"is_degraded": False},
                checks={},
                provenance={"scaffold": True},
                payload={},
            ),
            catalog_context=None,
            followup_context=None,
        )
    )
    j = report.to_json()
    assert "enrichment" in j["summary"]
    assert j["summary"]["enrichment"]["version"] == "0.1.0"
    assert j["summary"]["enrichment"]["pixel_diagnostics"]["status"] == "skipped"
