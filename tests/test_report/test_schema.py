from __future__ import annotations

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

