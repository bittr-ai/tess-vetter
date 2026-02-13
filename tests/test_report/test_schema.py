from __future__ import annotations

import copy

import pytest

from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.report import (
    FIELD_CATALOG,
    FieldKey,
    ReportPayloadModel,
    build_report,
    report_payload_json_schema,
)


def _make_minimal_lc() -> LightCurve:
    return LightCurve(
        time=[0.0, 0.1, 0.2, 0.3],
        flux=[1.0, 0.999, 1.001, 1.0],
        flux_err=[0.0001, 0.0001, 0.0001, 0.0001],
    )


def test_report_payload_schema_exports_core_properties() -> None:
    schema = report_payload_json_schema()
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "schema_version" in props
    assert "summary" in props
    assert "plot_data" in props
    assert "payload_meta" in props


def test_report_to_json_conforms_to_payload_model() -> None:
    lc = _make_minimal_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )
    report = build_report(lc, candidate, include_additional_plots=False)
    payload = report.to_json()
    parsed = ReportPayloadModel.model_validate(payload)
    assert parsed.summary.ephemeris is not None
    assert parsed.plot_data.phase_folded is not None


def test_field_catalog_paths_match_enum_values() -> None:
    for key, spec in FIELD_CATALOG.items():
        assert isinstance(key, FieldKey)
        assert spec.path == key.value


def test_report_payload_accepts_typed_reference_and_summary_blocks() -> None:
    lc = _make_minimal_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )
    payload = build_report(lc, candidate, include_additional_plots=False).to_json()
    payload["summary"]["checks"]["V01"]["method_refs"] = ["api.lc_only.odd_even_depth"]
    payload["summary"]["references"] = [
        {
            "key": "THOMPSON_2018",
            "title": "Planetary Candidates from K2",
            "authors": ["Thompson, S. E."],
            "year": 2018,
            "url": "https://example.org/ref",
            "tags": ["vetting"],
        }
    ]
    payload["summary"]["odd_even_summary"] = {
        "odd_depth_ppm": 100.0,
        "even_depth_ppm": 120.0,
        "depth_diff_ppm": -20.0,
        "depth_diff_sigma": 2.5,
        "is_significant": False,
        "flags": [],
    }
    payload["summary"]["noise_summary"] = {
        "white_noise_ppm": 90.0,
        "red_noise_beta_30m": 1.1,
        "trend_stat": 2.0,
        "trend_stat_unit": "relative_flux_per_day",
        "flags": [],
        "semantics": {"trend_direction": "up"},
    }
    payload["summary"]["variability_summary"] = {
        "variability_index": 0.8,
        "periodicity_score": 0.3,
        "classification": "quiet",
        "flags": [],
        "semantics": {"is_periodic": False},
    }
    payload["summary"]["alias_scalar_summary"] = {
        "best_harmonic": "P",
        "best_ratio_over_p": 1.0,
        "score_p": 0.5,
        "score_p_over_2": 0.3,
        "score_2p": 0.2,
        "depth_ppm_peak": 1200.0,
        "classification": "NONE",
        "phase_shift_event_count": 0,
        "phase_shift_peak_sigma": None,
        "secondary_significance": 0.0,
    }

    parsed = ReportPayloadModel.model_validate(payload)

    assert parsed.summary.checks["V01"].method_refs == ["api.lc_only.odd_even_depth"]
    assert parsed.summary.references[0].key == "THOMPSON_2018"
    assert parsed.summary.noise_summary is not None
    assert parsed.summary.noise_summary.trend_stat_unit == "relative_flux_per_day"
    assert parsed.summary.variability_summary is not None
    assert parsed.summary.odd_even_summary is not None
    assert parsed.summary.alias_scalar_summary is not None


def test_report_payload_rejects_invalid_typed_reference_shape() -> None:
    lc = _make_minimal_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )
    payload = build_report(lc, candidate, include_additional_plots=False).to_json()
    payload = copy.deepcopy(payload)
    payload["summary"]["references"] = [{"title": "Missing key field"}]

    with pytest.raises(Exception):
        ReportPayloadModel.model_validate(payload)


def test_report_payload_schema_includes_new_deterministic_summary_blocks() -> None:
    schema = report_payload_json_schema()
    summary_props = schema["$defs"]["ReportSummaryModel"]["properties"]
    timing_props = schema["$defs"]["TimingSummaryModel"]["properties"]
    secondary_props = schema["$defs"]["SecondaryScanSummaryModel"]["properties"]
    assert "alias_scalar_summary" in summary_props
    assert "timing_summary" in summary_props
    assert "secondary_scan_summary" in summary_props
    assert "data_gap_summary" in summary_props
    assert "check_execution" in summary_props
    assert "snr_median" in timing_props
    assert "oc_median" in timing_props
    assert "n_raw_points" in secondary_props
    assert "n_bins" in secondary_props


def test_report_payload_accepts_data_gap_summary_scalars_and_nulls() -> None:
    lc = _make_minimal_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )
    payload = build_report(lc, candidate, include_additional_plots=False).to_json()
    payload = copy.deepcopy(payload)
    payload["summary"]["data_gap_summary"] = {
        "missing_frac_max_in_coverage": 0.5,
        "missing_frac_median_in_coverage": None,
        "n_epochs_missing_ge_0p25_in_coverage": 2,
        "n_epochs_excluded_no_coverage": 1,
        "n_epochs_evaluated_in_coverage": 4,
    }

    parsed = ReportPayloadModel.model_validate(payload)
    assert parsed.summary.data_gap_summary is not None
    assert parsed.summary.data_gap_summary.missing_frac_max_in_coverage == 0.5
    assert parsed.summary.data_gap_summary.missing_frac_median_in_coverage is None
    assert parsed.summary.data_gap_summary.n_epochs_missing_ge_0p25_in_coverage == 2
    assert parsed.summary.data_gap_summary.n_epochs_excluded_no_coverage == 1
    assert parsed.summary.data_gap_summary.n_epochs_evaluated_in_coverage == 4


def test_report_payload_accepts_new_timing_and_secondary_scan_scalar_fields() -> None:
    lc = _make_minimal_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )
    payload = build_report(lc, candidate, include_additional_plots=False).to_json()
    payload = copy.deepcopy(payload)
    payload["summary"]["timing_summary"] = {
        "n_epochs_measured": 3,
        "rms_seconds": 25.0,
        "periodicity_score": 0.5,
        "linear_trend_sec_per_epoch": 0.2,
        "max_abs_oc_seconds": 30.0,
        "max_snr": 12.0,
        "snr_median": 10.0,
        "oc_median": 15.0,
        "outlier_count": 1,
        "outlier_fraction": 1 / 3,
        "deepest_epoch": 1,
    }
    payload["summary"]["secondary_scan_summary"] = {
        "n_raw_points": 3,
        "n_bins": 2,
        "phase_coverage_fraction": 0.4,
        "largest_phase_gap": 0.6,
        "n_bins_with_error": 2,
        "strongest_dip_phase": 0.0,
        "strongest_dip_depth_ppm": 1500.0,
        "is_degraded": True,
        "quality_flag_count": 1,
    }

    parsed = ReportPayloadModel.model_validate(payload)
    assert parsed.summary.timing_summary is not None
    assert parsed.summary.secondary_scan_summary is not None
    assert parsed.summary.timing_summary.snr_median == pytest.approx(10.0)
    assert parsed.summary.timing_summary.oc_median == pytest.approx(15.0)
    assert parsed.summary.secondary_scan_summary.n_raw_points == 3
    assert parsed.summary.secondary_scan_summary.n_bins == 2


def test_report_payload_rejects_non_scalar_values_in_new_summary_blocks() -> None:
    lc = _make_minimal_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )
    payload = build_report(lc, candidate, include_additional_plots=False).to_json()
    payload = copy.deepcopy(payload)
    payload["summary"]["alias_scalar_summary"] = {
        "best_harmonic": "P",
        "best_ratio_over_p": 1.0,
        "score_p": 0.5,
        "score_p_over_2": 0.3,
        "score_2p": 0.2,
        "depth_ppm_peak": [1000.0],
    }
    payload["summary"]["timing_summary"] = {
        "n_epochs_measured": 3,
        "rms_seconds": 25.0,
        "periodicity_score": 0.5,
        "linear_trend_sec_per_epoch": 0.2,
        "max_abs_oc_seconds": 30.0,
        "max_snr": 12.0,
        "snr_median": [10.0],
        "oc_median": {"bad": 15.0},
        "outlier_count": 1,
        "outlier_fraction": [1 / 3],
        "deepest_epoch": 1,
    }
    payload["summary"]["secondary_scan_summary"] = {
        "n_raw_points": [3],
        "n_bins": {"bad": 2},
        "phase_coverage_fraction": 0.4,
        "largest_phase_gap": 0.6,
        "n_bins_with_error": 2,
        "strongest_dip_phase": 0.0,
        "strongest_dip_depth_ppm": 1500.0,
        "is_degraded": True,
        "quality_flag_count": [1],
    }
    payload["summary"]["data_gap_summary"] = {
        "missing_frac_max_in_coverage": [0.5],
        "missing_frac_median_in_coverage": {"bad": 0.2},
        "n_epochs_missing_ge_0p25_in_coverage": 2,
        "n_epochs_excluded_no_coverage": [1],
        "n_epochs_evaluated_in_coverage": {"count": 4},
    }

    with pytest.raises(Exception):
        ReportPayloadModel.model_validate(payload)


def test_report_payload_accepts_check_execution_state_shape() -> None:
    lc = _make_minimal_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )
    payload = build_report(lc, candidate, include_additional_plots=False).to_json()
    payload = copy.deepcopy(payload)
    payload["summary"]["check_execution"] = {
        "v03_requested": True,
        "v03_enabled": False,
        "v03_disabled_reason": "stellar is required to enable V03",
    }

    parsed = ReportPayloadModel.model_validate(payload)
    assert parsed.summary.check_execution is not None
    assert parsed.summary.check_execution.v03_requested is True
    assert parsed.summary.check_execution.v03_enabled is False
    assert isinstance(parsed.summary.check_execution.v03_disabled_reason, str)


def test_report_payload_rejects_invalid_check_execution_types() -> None:
    lc = _make_minimal_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )
    payload = build_report(lc, candidate, include_additional_plots=False).to_json()
    payload = copy.deepcopy(payload)
    payload["summary"]["check_execution"] = {
        "v03_requested": "yes",
        "v03_enabled": {"enabled": False},
        "v03_disabled_reason": 123,
    }

    with pytest.raises(Exception):
        ReportPayloadModel.model_validate(payload)


def test_report_payload_accepts_metric_contract_payload_meta_shape() -> None:
    lc = _make_minimal_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )
    payload = build_report(lc, candidate, include_additional_plots=False).to_json()
    payload = copy.deepcopy(payload)
    payload["payload_meta"].update(
        {
            "contract_version": "1",
            "required_metrics_by_check": {"V01": ["delta_sigma"]},
            "missing_required_metrics_by_check": {"V01": ["depth_even_ppm"]},
            "metric_keys_by_check": {"V01": ["delta_sigma"]},
            "has_missing_required_metrics": True,
        }
    )

    parsed = ReportPayloadModel.model_validate(payload)
    assert parsed.payload_meta.contract_version == "1"
    assert parsed.payload_meta.required_metrics_by_check["V01"] == ["delta_sigma"]
    assert parsed.payload_meta.has_missing_required_metrics is True


def test_report_payload_rejects_invalid_metric_contract_payload_meta_types() -> None:
    lc = _make_minimal_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=1.0, t0_btjd=0.0, duration_hours=1.0),
        depth_ppm=500.0,
    )
    payload = build_report(lc, candidate, include_additional_plots=False).to_json()
    payload = copy.deepcopy(payload)
    payload["payload_meta"].update(
        {
            "contract_version": 1,
            "required_metrics_by_check": ["V01"],
            "missing_required_metrics_by_check": {"V01": "depth_even_ppm"},
            "metric_keys_by_check": {"V01": [123]},
            "has_missing_required_metrics": "false",
        }
    )

    with pytest.raises(Exception):
        ReportPayloadModel.model_validate(payload)
