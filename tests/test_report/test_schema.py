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

    parsed = ReportPayloadModel.model_validate(payload)

    assert parsed.summary.checks["V01"].method_refs == ["api.lc_only.odd_even_depth"]
    assert parsed.summary.references[0].key == "THOMPSON_2018"
    assert parsed.summary.noise_summary is not None
    assert parsed.summary.noise_summary.trend_stat_unit == "relative_flux_per_day"
    assert parsed.summary.variability_summary is not None
    assert parsed.summary.odd_even_summary is not None


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
