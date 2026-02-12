from __future__ import annotations

import json

from bittr_tess_vetter.report._ui_meta import (
    UI_META_VERSION,
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
