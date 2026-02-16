from __future__ import annotations

import json
from pathlib import Path

from bittr_tess_vetter.pipeline_composition.executor import run_composition
from bittr_tess_vetter.pipeline_composition.registry import get_profile, list_profiles
from bittr_tess_vetter.pipeline_composition.schema import validate_composition_payload


def test_profile_registry_contains_mvp_profiles() -> None:
    profiles = set(list_profiles())
    assert "triage_fast" in profiles
    assert "host_localization" in profiles
    assert "fpp_validation_fast" in profiles
    assert "full_vetting" in profiles

    full = get_profile("full_vetting")
    assert full.id == "full_vetting"
    assert len(full.steps) >= 10


def test_run_composition_report_from_and_ports_and_resume(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "schema_version": "pipeline.composition.v1",
        "id": "test_comp",
        "defaults": {"retry_max_attempts": 1, "retry_initial_seconds": 0.01},
        "steps": [
            {
                "id": "seed_report",
                "op": "report",
                "ports": {"seed_file": "artifact_path"},
            },
            {
                "id": "model",
                "op": "model_compete",
                "inputs": {"report_file": {"report_from": "seed_report"}},
            },
            {
                "id": "neighbors",
                "op": "resolve_neighbors",
                "ports": {"reference_sources_file": "artifact_path"},
            },
            {
                "id": "dilution",
                "op": "dilution",
                "inputs": {
                    "reference_sources_file": {"port": "neighbors.reference_sources_file"},
                },
            },
            {
                "id": "fpp",
                "op": "fpp",
            },
        ],
    }
    comp = validate_composition_payload(payload, source="test")

    calls: list[tuple[str, dict]] = []

    def _fake_run_step_with_retries(
        *,
        step,
        toi,
        inputs,
        output_path,
        stderr_path,
        network_ok,
        max_attempts,
        initial_backoff_seconds,
    ):
        calls.append((step.id, dict(inputs)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text("", encoding="utf-8")

        row = {
            "schema_version": f"test.{step.op}.v1",
            "verdict": None,
        }
        if step.id == "model":
            assert str(inputs.get("report_file", "")).endswith("seed_report.json")
            row["verdict"] = "MODEL_OK"
        if step.id == "dilution":
            assert str(inputs.get("reference_sources_file", "")).endswith("neighbors.json")
            row["n_plausible_scenarios"] = 2
        if step.id == "fpp":
            row["fpp"] = 0.003

        output_path.write_text(json.dumps(row), encoding="utf-8")
        return row, 1

    monkeypatch.setattr(
        "bittr_tess_vetter.pipeline_composition.executor._run_step_with_retries",
        _fake_run_step_with_retries,
    )

    out_dir = tmp_path / "run"
    result = run_composition(
        composition=comp,
        tois=["TOI-TEST.01"],
        out_dir=out_dir,
        network_ok=False,
        continue_on_error=False,
        max_workers=1,
        resume=False,
    )

    assert result["manifest"]["counts"]["n_ok"] == 1
    evidence_json = json.loads((out_dir / "evidence_table.json").read_text(encoding="utf-8"))
    row = evidence_json["rows"][0]
    assert row["toi"] == "TOI-TEST.01"
    assert row["model_compete_verdict"] == "MODEL_OK"
    assert row["dilution_n_plausible_scenarios"] == 2
    assert row["fpp"] == 0.003

    # Resume run should skip step execution entirely.
    calls.clear()

    def _fail_if_called(**kwargs):
        raise AssertionError("step runner should not be called in resume mode")

    monkeypatch.setattr(
        "bittr_tess_vetter.pipeline_composition.executor._run_step_with_retries",
        _fail_if_called,
    )

    result_resume = run_composition(
        composition=comp,
        tois=["TOI-TEST.01"],
        out_dir=out_dir,
        network_ok=False,
        continue_on_error=False,
        max_workers=1,
        resume=True,
    )
    assert result_resume["manifest"]["counts"]["n_ok"] == 1
