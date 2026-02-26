from __future__ import annotations

import json
from pathlib import Path

from tess_vetter.pipeline_composition.executor import _build_cli_args, run_composition
from tess_vetter.pipeline_composition.registry import get_profile, list_profiles
from tess_vetter.pipeline_composition.schema import StepSpec, validate_composition_payload


def test_profile_registry_contains_mvp_profiles() -> None:
    profiles = set(list_profiles())
    assert "triage_fast" in profiles
    assert "host_localization" in profiles
    assert "fpp_validation_fast" in profiles
    assert "full_vetting" in profiles
    assert "robustness_composition" in profiles

    full = get_profile("full_vetting")
    assert full.id == "full_vetting"
    assert len(full.steps) >= 10
    assert any(step.id == "rv_feasibility" for step in full.steps)
    full_step_ids = [step.id for step in full.steps]
    assert "contrast_curves" in full_step_ids
    assert full_step_ids.index("contrast_curves") < full_step_ids.index("vet")

    triage = get_profile("triage_fast")
    assert any(step.id == "rv_feasibility" for step in triage.steps)

    robust = get_profile("robustness_composition")
    assert robust.id == "robustness_composition"
    assert [step.id for step in robust.steps] == [
        "report_seed",
        "detrend_grid",
        "model_compete_raw",
        "model_compete_detrended",
        "fpp_prepare_raw",
        "fpp_raw",
        "fpp_prepare_detrended",
        "fpp_detrended",
    ]


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
                "id": "fpp_prepare",
                "op": "fpp_prepare",
                "ports": {"prepare_manifest": "artifact_path"},
            },
            {
                "id": "fpp_run",
                "op": "fpp_run",
                "inputs": {"prepare_manifest": {"port": "fpp_prepare.prepare_manifest"}},
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
        if step.id == "fpp_run":
            assert str(inputs.get("prepare_manifest", "")).endswith("fpp_prepare.json")
            row["fpp"] = 0.003

        output_path.write_text(json.dumps(row), encoding="utf-8")
        return row, 1

    monkeypatch.setattr(
        "tess_vetter.pipeline_composition.executor._run_step_with_retries",
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
    assert evidence_json["schema_version"] == "pipeline.evidence_table.v5"
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
        "tess_vetter.pipeline_composition.executor._run_step_with_retries",
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


def test_run_composition_step_failure_respects_on_error_continue(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "schema_version": "pipeline.composition.v1",
        "id": "test_fail_continue",
        "defaults": {"retry_max_attempts": 1, "retry_initial_seconds": 0.01},
        "steps": [
            {"id": "report_seed", "op": "report"},
            {"id": "bad_step", "op": "timing", "on_error": "continue"},
            {"id": "final_step", "op": "model_compete"},
        ],
    }
    comp = validate_composition_payload(payload, source="test")

    calls: list[str] = []

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
        calls.append(step.id)
        if step.id == "bad_step":
            raise RuntimeError("simulated timeout")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text("", encoding="utf-8")
        row = {"schema_version": "test.v1", "verdict": f"{step.id}_ok"}
        output_path.write_text(json.dumps(row), encoding="utf-8")
        return row, 1

    monkeypatch.setattr(
        "tess_vetter.pipeline_composition.executor._run_step_with_retries",
        _fake_run_step_with_retries,
    )

    out_dir = tmp_path / "run_continue"
    result = run_composition(
        composition=comp,
        tois=["TOI-CONTINUE.01"],
        out_dir=out_dir,
        network_ok=False,
        continue_on_error=False,
        max_workers=1,
        resume=False,
    )
    assert result["manifest"]["counts"]["n_partial"] == 1
    assert result["manifest"]["counts"]["n_failed"] == 0
    assert calls == ["report_seed", "bad_step", "final_step"]


def test_run_composition_multi_toi_aggregation_includes_partial(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "schema_version": "pipeline.composition.v1",
        "id": "test_multi",
        "defaults": {"retry_max_attempts": 1, "retry_initial_seconds": 0.01},
        "steps": [
            {"id": "report_seed", "op": "report"},
            {"id": "model", "op": "model_compete"},
        ],
    }
    comp = validate_composition_payload(payload, source="test")

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
        if toi == "TOI-B.01" and step.id == "model":
            raise RuntimeError("429 rate limit")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text("", encoding="utf-8")
        row = {"schema_version": "test.v1", "verdict": "OK"}
        output_path.write_text(json.dumps(row), encoding="utf-8")
        return row, 1

    monkeypatch.setattr(
        "tess_vetter.pipeline_composition.executor._run_step_with_retries",
        _fake_run_step_with_retries,
    )

    out_dir = tmp_path / "run_multi"
    result = run_composition(
        composition=comp,
        tois=["TOI-B.01", "TOI-A.01"],
        out_dir=out_dir,
        network_ok=False,
        continue_on_error=True,
        max_workers=2,
        resume=False,
    )
    counts = result["manifest"]["counts"]
    assert counts["n_tois"] == 2
    assert counts["n_ok"] == 1
    assert counts["n_partial"] == 1
    assert counts["n_failed"] == 0

    evidence_json = json.loads((out_dir / "evidence_table.json").read_text(encoding="utf-8"))
    toi_rows = [row["toi"] for row in evidence_json["rows"]]
    assert toi_rows == ["TOI-A.01", "TOI-B.01"]


def test_run_composition_does_not_forward_retry_defaults_to_step_inputs(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "schema_version": "pipeline.composition.v1",
        "id": "test_retry_defaults_not_forwarded",
        "defaults": {"retry_max_attempts": 5, "retry_initial_seconds": 2.0, "preset": "fast"},
        "steps": [
            {"id": "fpp_step", "op": "fpp_run"},
        ],
    }
    comp = validate_composition_payload(payload, source="test")

    seen: dict[str, dict] = {}

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
        seen["inputs"] = dict(inputs)
        seen["retry"] = {
            "max_attempts": max_attempts,
            "initial_backoff_seconds": initial_backoff_seconds,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text("", encoding="utf-8")
        row = {"schema_version": "test.v1", "fpp": 0.01}
        output_path.write_text(json.dumps(row), encoding="utf-8")
        return row, 1

    monkeypatch.setattr(
        "tess_vetter.pipeline_composition.executor._run_step_with_retries",
        _fake_run_step_with_retries,
    )

    out_dir = tmp_path / "run_retry_defaults"
    result = run_composition(
        composition=comp,
        tois=["TOI-RETRY.01"],
        out_dir=out_dir,
        network_ok=False,
        continue_on_error=False,
        max_workers=1,
        resume=False,
    )
    assert result["manifest"]["counts"]["n_ok"] == 1
    assert seen["inputs"]["preset"] == "fast"
    assert "retry_max_attempts" not in seen["inputs"]
    assert "retry_initial_seconds" not in seen["inputs"]
    assert seen["retry"] == {"max_attempts": 5, "initial_backoff_seconds": 2.0}


def test_build_cli_args_supports_staged_fpp_commands(tmp_path: Path) -> None:
    output_path = tmp_path / "out.json"
    step_prepare = StepSpec(id="fpp_prepare", op="fpp_prepare", inputs={}, ports={}, outputs={}, on_error="fail")
    step_run = StepSpec(id="fpp_run", op="fpp_run", inputs={}, ports={}, outputs={}, on_error="fail")

    prepare_args = _build_cli_args(
        step=step_prepare,
        toi="TOI-STAGE.01",
        inputs={},
        output_path=output_path,
        network_ok=False,
    )
    run_args = _build_cli_args(
        step=step_run,
        toi="TOI-STAGE.01",
        inputs={},
        output_path=output_path,
        network_ok=False,
    )

    assert "fpp-prepare" in prepare_args
    assert "fpp-run" in run_args
