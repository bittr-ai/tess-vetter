from __future__ import annotations

import json

import pytest

from tess_vetter.report._ui_meta import (
    UI_META_VERSION,
    _model_property_types,
    main,
    build_ui_meta_artifact,
    write_ui_meta_artifact,
)


def test_ui_meta_builder_returns_deterministic_artifact() -> None:
    first = build_ui_meta_artifact()
    second = build_ui_meta_artifact()

    assert first == second
    assert first["version"] == UI_META_VERSION
    assert isinstance(first["fields"], list)
    assert len(first["fields"]) > 0


def test_ui_meta_artifact_includes_check_and_reference_metadata() -> None:
    artifact = build_ui_meta_artifact()

    assert artifact["checks"]["path"] == "summary.checks[*]"
    assert "method_refs" in artifact["checks"]["property_types"]

    assert artifact["references"]["path"] == "summary.references[*]"
    assert "key" in artifact["references"]["property_types"]
    assert "authors" in artifact["references"]["property_types"]

    fields_by_path = {field["path"]: field for field in artifact["fields"]}
    assert "summary.noise_summary.trend_stat_unit" in fields_by_path
    assert "summary.variability_summary" in fields_by_path
    assert "summary.references[*].key" in fields_by_path


def test_ui_meta_writer_writes_expected_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "report_ui_meta.json"
    wrote = write_ui_meta_artifact(out)

    assert wrote == out
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed == build_ui_meta_artifact()


def test_model_property_types_handles_all_schema_shapes() -> None:
    schema = {
        "properties": {
            "as_list": {"type": ["string", "null"]},
            "as_str": {"type": "number"},
            "as_ref": {"$ref": "#/defs/X"},
            "as_anyof": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "as_unknown": {"title": "No type keys"},
        }
    }
    out = _model_property_types(schema)
    assert out["as_list"] == "null|string"
    assert out["as_str"] == "number"
    assert out["as_ref"] == "object"
    assert out["as_anyof"] == "union"
    assert out["as_unknown"] == "unknown"


def test_main_writes_artifact_and_prints_path(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "ui_meta_from_main.json"
    monkeypatch.setattr("sys.argv", ["ui_meta", "--out", str(out)])
    main()
    assert out.exists()
    printed = capsys.readouterr().out
    assert "Wrote" in printed
