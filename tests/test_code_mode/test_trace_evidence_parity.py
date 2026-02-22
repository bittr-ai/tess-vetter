from __future__ import annotations

import csv
import json
from pathlib import Path

from tess_vetter.code_mode.trace import (
    EVIDENCE_FIELDNAMES,
    REQUIRED_EVIDENCE_KEYS,
    RUNTIME_TRACE_METADATA_KEYS,
    build_evidence_compatible_row,
    build_runtime_trace_metadata,
    to_csv_row,
)
from tess_vetter.pipeline_composition.executor import _extract_evidence_row, _write_evidence_table


def _write_step(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _without_clocklike_fields(metadata: dict) -> dict:
    normalized = dict(metadata)
    normalized.pop("timestamp", None)
    return normalized


def test_trace_row_matches_executor_extraction_for_representative_payload(tmp_path: Path) -> None:
    toi = "TOI-CODE-MODE.01"
    toi_dir = tmp_path / toi / "steps"

    model_path = _write_step(toi_dir / "01_model.json", {"verdict": "MODEL_OK", "warnings": ["model_warn"]})
    systematics_path = _write_step(toi_dir / "02_systematics.json", {"result": {"verdict": "SYSTEMATICS_OK"}})
    ephemeris_path = _write_step(toi_dir / "03_ephemeris.json", {"verdict": "EPHEMERIS_OK"})
    timing_path = _write_step(toi_dir / "04_timing.json", {"result": {"verdict": "TIMING_OK"}})
    report_path = _write_step(
        toi_dir / "05_report.json",
        {
            "summary": {
                "concerns": ["zeta", "alpha"],
                "stellar_contamination_summary": {"risk_scalar": 0.55},
            },
            "warnings": ["beta"],
        },
    )
    localize_path = _write_step(
        toi_dir / "06_localize.json",
        {
            "result": {
                "consensus": {"action_hint": "collect_more_imaging", "reliability_flags": ["gamma"]},
                "reliability_summary": {
                    "status": "REVIEW_REQUIRED",
                    "action_hint": "DEFER_HOST_ASSIGNMENT",
                },
            }
        },
    )
    dilution_path = _write_step(
        toi_dir / "07_dilution.json",
        {
            "result": {
                "n_plausible_scenarios": 4,
                "reliability_summary": {
                    "status": "REVIEW_REQUIRED",
                    "action_hint": "REVIEW_WITH_DILUTION",
                },
            }
        },
    )
    fpp_path = _write_step(toi_dir / "08_fpp.json", {"result": {"fpp": 0.017}})
    neighbors_path = _write_step(
        toi_dir / "09_neighbors.json",
        {
            "provenance": {
                "multiplicity_risk": {
                    "status": "ELEVATED",
                    "reasons": ["TARGET_RUWE_ELEVATED", None, 5],
                }
            }
        },
    )
    vet_path = _write_step(
        toi_dir / "10_vet.json",
        {
            "summary": {"known_planet_match_status": "confirmed_same_planet"},
            "known_planet_match": {
                "status": "confirmed_same_planet",
                "matched_planet": {"name": "TOI-411 c", "period": 9.57307},
            },
        },
    )

    toi_result = {
        "toi": toi,
        "concern_flags": ["delta", "alpha"],
        "steps": [
            {"op": "model_compete", "status": "ok", "step_output_path": model_path},
            {"op": "systematics_proxy", "status": "ok", "step_output_path": systematics_path},
            {"op": "ephemeris_reliability", "status": "ok", "step_output_path": ephemeris_path},
            {"op": "timing", "status": "ok", "step_output_path": timing_path},
            {"op": "report", "status": "ok", "step_output_path": report_path},
            {"op": "localize_host", "status": "ok", "step_output_path": localize_path},
            {"op": "dilution", "status": "ok", "step_output_path": dilution_path},
            {"op": "fpp_run", "status": "ok", "step_output_path": fpp_path},
            {"op": "resolve_neighbors", "status": "ok", "step_output_path": neighbors_path},
            {"op": "vet", "status": "ok", "step_output_path": vet_path},
        ],
    }

    expected = _extract_evidence_row(toi_result, out_dir=tmp_path)
    actual = build_evidence_compatible_row(toi_result)

    assert actual == expected
    assert set(actual) == REQUIRED_EVIDENCE_KEYS
    assert actual["concern_flags"] == ["alpha", "beta", "delta", "gamma", "model_warn", "zeta"]
    assert actual["multiplicity_risk_reasons"] == ["TARGET_RUWE_ELEVATED", "5"]


def test_trace_row_inline_payload_parity_without_step_output_path(tmp_path: Path) -> None:
    toi = "TOI-INLINE.01"
    model_payload = {"verdict": "MODEL_OK", "warnings": ["inline_warn"]}
    report_payload = {"summary": {"concerns": ["c2", "c1"]}, "warnings": ["c3"]}
    localize_payload = {
        "result": {
            "consensus": {"action_hint": "collect_more_imaging", "reliability_flags": ["c4"]},
            "reliability_summary": {"status": "REVIEW_REQUIRED", "action_hint": "DEFER_HOST_ASSIGNMENT"},
        }
    }

    inline_toi_result = {
        "toi": toi,
        "concern_flags": ["root_flag"],
        "steps": [
            {"step_id": "model_compete", "op": "model_compete", "status": "ok", "payload": model_payload},
            {"step_id": "report", "op": "report", "status": "ok", "payload": report_payload},
            {"step_id": "localize_host", "op": "localize_host", "status": "ok", "payload": localize_payload},
        ],
    }

    toi_dir = tmp_path / toi / "steps"
    filebacked_toi_result = {
        "toi": toi,
        "concern_flags": ["root_flag"],
        "steps": [
            {
                "step_id": "model_compete",
                "op": "model_compete",
                "status": "ok",
                "step_output_path": _write_step(toi_dir / "01_model.json", model_payload),
            },
            {
                "step_id": "report",
                "op": "report",
                "status": "ok",
                "step_output_path": _write_step(toi_dir / "02_report.json", report_payload),
            },
            {
                "step_id": "localize_host",
                "op": "localize_host",
                "status": "ok",
                "step_output_path": _write_step(toi_dir / "03_localize.json", localize_payload),
            },
        ],
    }

    expected = _extract_evidence_row(filebacked_toi_result, out_dir=tmp_path)
    actual = build_evidence_compatible_row(inline_toi_result)

    assert actual == expected
    assert actual["concern_flags"] == ["c1", "c2", "c3", "c4", "inline_warn", "root_flag"]


def test_trace_required_keys_and_fieldnames_stay_in_lockstep_with_executor_csv(tmp_path: Path) -> None:
    toi_result = {"toi": "TOI-EMPTY.01", "concern_flags": [], "steps": []}
    row = build_evidence_compatible_row(toi_result)

    _write_evidence_table(out_dir=tmp_path, toi_results=[toi_result])
    with (tmp_path / "evidence_table.csv").open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        executor_fieldnames = tuple(reader.fieldnames)

    assert set(row) == REQUIRED_EVIDENCE_KEYS
    assert tuple(EVIDENCE_FIELDNAMES) == executor_fieldnames


def test_trace_csv_normalization_matches_executor_semantics() -> None:
    row = {
        "toi": "TOI-NORM.01",
        "concern_flags": ["zeta", "alpha", "alpha", "beta"],
        "multiplicity_risk_reasons": ["R2", "R1", "R2"],
    }

    csv_row = to_csv_row(row)

    assert csv_row["concern_flags"] == "alpha;alpha;beta;zeta"
    assert csv_row["multiplicity_risk_reasons"] == "R2;R1;R2"


def test_build_runtime_trace_metadata_exposes_expected_structure() -> None:
    metadata = build_runtime_trace_metadata(
        trace_id="trace-123",
        policy_profile="readonly_local",
        network_ok=False,
        catalog_version_hash="abc123",
        timestamp=1700000000.0,
    )

    assert set(metadata) == set(RUNTIME_TRACE_METADATA_KEYS)
    assert metadata["trace_id"] == "trace-123"
    assert metadata["policy_profile"] == "readonly_local"
    assert metadata["network_ok"] is False
    assert metadata["detrend_invariance"]["policy_version"] == "v1"
    assert metadata["detrend_invariance"]["fpp_delta_abs_threshold"] == 0.01
    assert metadata["evidence"]["fieldnames"] == list(EVIDENCE_FIELDNAMES)


def test_trace_repeat_run_outputs_are_deterministic_except_clocklike_fields(tmp_path: Path) -> None:
    toi = "TOI-DETERMINISM.01"
    toi_dir = tmp_path / toi / "steps"
    model_path = _write_step(toi_dir / "01_model.json", {"verdict": "MODEL_OK", "warnings": ["w2", "w1"]})
    report_path = _write_step(toi_dir / "02_report.json", {"summary": {"concerns": ["c2", "c1"]}})

    toi_result = {
        "toi": toi,
        "concern_flags": ["r2", "r1"],
        "steps": [
            {"op": "model_compete", "status": "ok", "step_output_path": model_path},
            {"op": "report", "status": "ok", "step_output_path": report_path},
        ],
    }

    row_first = build_evidence_compatible_row(toi_result)
    row_second = build_evidence_compatible_row(toi_result)
    assert row_first == row_second
    assert to_csv_row(row_first) == to_csv_row(row_second)

    metadata_first = build_runtime_trace_metadata(
        trace_id="trace-deterministic",
        policy_profile="readonly_local",
        network_ok=True,
        catalog_version_hash="hash123",
        timestamp=1700000000.0,
    )
    metadata_second = build_runtime_trace_metadata(
        trace_id="trace-deterministic",
        policy_profile="readonly_local",
        network_ok=True,
        catalog_version_hash="hash123",
        timestamp=1700000100.0,
    )
    assert _without_clocklike_fields(metadata_first) == _without_clocklike_fields(metadata_second)
