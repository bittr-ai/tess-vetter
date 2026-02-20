from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from bittr_tess_vetter.cli.enrich_cli import cli


def test_btv_pipeline_run_builtin_profile_wires_cli_contract_end_to_end(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "pipeline_run"

    recorded_calls: list[dict[str, object]] = []
    report_outputs_by_toi: dict[str, Path] = {}

    def _fake_subprocess_run(args, capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        assert capture_output is True
        assert text is True

        argv = [str(token) for token in args]
        assert argv[:3] == [argv[0], "-m", "bittr_tess_vetter.cli.enrich_cli"]
        command = argv[3]
        assert "--network-ok" in argv
        assert "--no-network" not in argv

        toi = argv[argv.index("--toi") + 1]
        output_path = Path(argv[argv.index("--out") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, object]
        if command == "report":
            payload = {"schema_version": "test.report.v1"}
            report_outputs_by_toi[toi] = output_path
        else:
            report_file = Path(argv[argv.index("--report-file") + 1])
            assert report_file == report_outputs_by_toi[toi]
            payload = {"schema_version": f"test.{command}.v1", "verdict": f"{command}_ok"}
            if command == "model-compete":
                payload["verdict"] = "TRANSIT_LIKE"

        output_path.write_text(json.dumps(payload), encoding="utf-8")

        recorded_calls.append({"command": command, "argv": argv, "toi": toi, "out": str(output_path)})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "bittr_tess_vetter.pipeline_composition.executor.subprocess.run",
        _fake_subprocess_run,
    )

    result = runner.invoke(
        cli,
        [
            "pipeline",
            "run",
            "--profile",
            "triage_fast",
            "--toi",
            "TOI-123.01",
            "--out-dir",
            str(out_dir),
            "--network-ok",
            "--max-workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Pipeline run complete: n_tois=1 ok=1 partial=0 failed=0" in result.output

    assert [call["command"] for call in recorded_calls] == [
        "report",
        "activity",
        "rv-feasibility",
        "model-compete",
        "systematics-proxy",
        "ephemeris-reliability",
        "timing",
    ]

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile_id"] == "triage_fast"
    assert manifest["counts"] == {"n_tois": 1, "n_ok": 1, "n_partial": 0, "n_failed": 0}
    assert manifest["options"]["network_ok"] is True
    assert manifest["options"]["max_workers"] == 1

    evidence = json.loads((out_dir / "evidence_table.json").read_text(encoding="utf-8"))
    assert evidence["schema_version"] == "pipeline.evidence_table.v5"
    assert evidence["rows"][0]["toi"] == "TOI-123.01"
    assert evidence["rows"][0]["model_compete_verdict"] == "TRANSIT_LIKE"


def test_btv_pipeline_run_resume_uses_checkpoint_markers_and_sets_skipped_resume(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "pipeline_run_resume"

    recorded_calls: list[dict[str, object]] = []
    report_outputs_by_toi: dict[str, Path] = {}

    def _fake_subprocess_run(args, capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        assert capture_output is True
        assert text is True

        argv = [str(token) for token in args]
        command = argv[3]
        toi = argv[argv.index("--toi") + 1]
        output_path = Path(argv[argv.index("--out") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, object]
        if command == "report":
            payload = {"schema_version": "test.report.v1"}
            report_outputs_by_toi[toi] = output_path
        else:
            report_file = Path(argv[argv.index("--report-file") + 1])
            assert report_file == report_outputs_by_toi[toi]
            payload = {"schema_version": f"test.{command}.v1", "verdict": f"{command}_ok"}
            if command == "model-compete":
                payload["verdict"] = "TRANSIT_LIKE"

        output_path.write_text(json.dumps(payload), encoding="utf-8")
        recorded_calls.append({"command": command, "argv": argv, "toi": toi})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "bittr_tess_vetter.pipeline_composition.executor.subprocess.run",
        _fake_subprocess_run,
    )

    first_run = runner.invoke(
        cli,
        [
            "pipeline",
            "run",
            "--profile",
            "triage_fast",
            "--toi",
            "TOI-123.01",
            "--out-dir",
            str(out_dir),
            "--network-ok",
            "--max-workers",
            "1",
        ],
    )
    assert first_run.exit_code == 0, first_run.output
    assert len(recorded_calls) == 7

    checkpoint_dir = out_dir / "TOI-123.01" / "checkpoints"
    marker_paths = sorted(checkpoint_dir.glob("*.done.json"))
    assert len(marker_paths) == 7

    for marker_path in marker_paths:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert marker["status"] == "ok"
        assert Path(marker["step_output_path"]).exists()

    recorded_calls.clear()
    resumed_run = runner.invoke(
        cli,
        [
            "pipeline",
            "run",
            "--profile",
            "triage_fast",
            "--toi",
            "TOI-123.01",
            "--out-dir",
            str(out_dir),
            "--network-ok",
            "--max-workers",
            "1",
            "--resume",
        ],
    )

    assert resumed_run.exit_code == 0, resumed_run.output
    assert "Pipeline run complete: n_tois=1 ok=1 partial=0 failed=0" in resumed_run.output
    assert recorded_calls == []

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["options"]["resume"] is True
    assert manifest["counts"] == {"n_tois": 1, "n_ok": 1, "n_partial": 0, "n_failed": 0}

    toi_result = json.loads((out_dir / "TOI-123.01" / "pipeline_result.json").read_text(encoding="utf-8"))
    assert toi_result["status"] == "ok"
    assert sum(1 for row in toi_result["steps"] if row.get("skipped_resume") is True) == 7


def test_btv_pipeline_run_continue_on_error_marks_partial_and_preserves_step_labels(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "pipeline_run_partial"

    recorded_calls: list[dict[str, object]] = []
    report_outputs_by_toi: dict[str, Path] = {}

    def _fake_subprocess_run(args, capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        assert capture_output is True
        assert text is True

        argv = [str(token) for token in args]
        command = argv[3]
        toi = argv[argv.index("--toi") + 1]
        output_path = Path(argv[argv.index("--out") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        recorded_calls.append({"command": command, "toi": toi})

        if command == "model-compete":
            return SimpleNamespace(returncode=2, stdout="", stderr="synthetic model_compete failure")

        payload: dict[str, object]
        if command == "report":
            payload = {"schema_version": "test.report.v1"}
            report_outputs_by_toi[toi] = output_path
        else:
            report_file = Path(argv[argv.index("--report-file") + 1])
            assert report_file == report_outputs_by_toi[toi]
            payload = {"schema_version": f"test.{command}.v1", "verdict": f"{command}_ok"}

        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "bittr_tess_vetter.pipeline_composition.executor.subprocess.run",
        _fake_subprocess_run,
    )

    result = runner.invoke(
        cli,
        [
            "pipeline",
            "run",
            "--profile",
            "triage_fast",
            "--toi",
            "TOI-456.01",
            "--out-dir",
            str(out_dir),
            "--network-ok",
            "--continue-on-error",
            "--max-workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Pipeline run complete: n_tois=1 ok=0 partial=1 failed=0" in result.output
    assert [call["command"] for call in recorded_calls] == [
        "report",
        "activity",
        "rv-feasibility",
        "model-compete",
        "systematics-proxy",
        "ephemeris-reliability",
        "timing",
    ]

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"n_tois": 1, "n_ok": 0, "n_partial": 1, "n_failed": 0}
    assert manifest["options"]["continue_on_error"] is True
    assert manifest["results"] == [
        {
            "toi": "TOI-456.01",
            "status": "partial",
            "result_path": str(out_dir / "TOI-456.01" / "pipeline_result.json"),
        }
    ]

    toi_result = json.loads((out_dir / "TOI-456.01" / "pipeline_result.json").read_text(encoding="utf-8"))
    assert toi_result["status"] == "partial"
    assert sum(1 for row in toi_result["steps"] if row["status"] == "ok") == 6
    assert sum(1 for row in toi_result["steps"] if row["status"] == "failed") == 1
    assert any(
        row["status"] == "failed" and row["step_id"] == "model_compete" and row["op"] == "model_compete"
        for row in toi_result["steps"]
    )


def test_btv_pipeline_run_continue_on_error_two_tois_mixed_partial_and_ok(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "pipeline_run_two_toi_mixed"

    recorded_calls: list[dict[str, object]] = []
    report_outputs_by_toi: dict[str, Path] = {}

    def _fake_subprocess_run(args, capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        assert capture_output is True
        assert text is True

        argv = [str(token) for token in args]
        command = argv[3]
        toi = argv[argv.index("--toi") + 1]
        output_path = Path(argv[argv.index("--out") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        recorded_calls.append({"command": command, "toi": toi})

        if toi == "TOI-456.01" and command == "model-compete":
            return SimpleNamespace(returncode=2, stdout="", stderr="synthetic model_compete failure")

        payload: dict[str, object]
        if command == "report":
            payload = {"schema_version": "test.report.v1"}
            report_outputs_by_toi[toi] = output_path
        else:
            report_file = Path(argv[argv.index("--report-file") + 1])
            assert report_file == report_outputs_by_toi[toi]
            payload = {"schema_version": f"test.{command}.v1", "verdict": f"{command}_ok"}

        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "bittr_tess_vetter.pipeline_composition.executor.subprocess.run",
        _fake_subprocess_run,
    )

    result = runner.invoke(
        cli,
        [
            "pipeline",
            "run",
            "--profile",
            "triage_fast",
            "--toi",
            "TOI-456.01",
            "--toi",
            "TOI-789.01",
            "--out-dir",
            str(out_dir),
            "--network-ok",
            "--continue-on-error",
            "--max-workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Pipeline run complete: n_tois=2 ok=1 partial=1 failed=0" in result.output

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"n_tois": 2, "n_ok": 1, "n_partial": 1, "n_failed": 0}
    assert manifest["results"] == [
        {
            "toi": "TOI-456.01",
            "status": "partial",
            "result_path": str(out_dir / "TOI-456.01" / "pipeline_result.json"),
        },
        {
            "toi": "TOI-789.01",
            "status": "ok",
            "result_path": str(out_dir / "TOI-789.01" / "pipeline_result.json"),
        },
    ]

    partial_result = json.loads((out_dir / "TOI-456.01" / "pipeline_result.json").read_text(encoding="utf-8"))
    ok_result = json.loads((out_dir / "TOI-789.01" / "pipeline_result.json").read_text(encoding="utf-8"))

    assert partial_result["status"] == "partial"
    assert ok_result["status"] == "ok"
    assert sum(1 for row in partial_result["steps"] if row["status"] == "failed") == 1
    assert sum(1 for row in ok_result["steps"] if row["status"] == "ok") == 7
    assert len(recorded_calls) == 14


def test_btv_pipeline_run_resume_after_mixed_outcomes_converges_to_two_ok_and_reuses_checkpoints(
    monkeypatch, tmp_path: Path
) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "pipeline_run_resume_mixed"

    recorded_calls: list[dict[str, object]] = []
    report_outputs_by_toi: dict[str, Path] = {}
    model_compete_failures_by_toi: dict[str, int] = {"TOI-456.01": 1}

    def _fake_subprocess_run(args, capture_output: bool, text: bool):  # type: ignore[no-untyped-def]
        assert capture_output is True
        assert text is True

        argv = [str(token) for token in args]
        command = argv[3]
        toi = argv[argv.index("--toi") + 1]
        output_path = Path(argv[argv.index("--out") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        recorded_calls.append({"command": command, "toi": toi})

        if command == "model-compete" and model_compete_failures_by_toi.get(toi, 0) > 0:
            model_compete_failures_by_toi[toi] -= 1
            return SimpleNamespace(returncode=2, stdout="", stderr="synthetic model_compete failure")

        payload: dict[str, object]
        if command == "report":
            payload = {"schema_version": "test.report.v1"}
            report_outputs_by_toi[toi] = output_path
        else:
            report_file = Path(argv[argv.index("--report-file") + 1])
            assert report_file == report_outputs_by_toi[toi]
            payload = {"schema_version": f"test.{command}.v1", "verdict": f"{command}_ok"}
            if command == "model-compete":
                payload["verdict"] = "TRANSIT_LIKE"

        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "bittr_tess_vetter.pipeline_composition.executor.subprocess.run",
        _fake_subprocess_run,
    )

    first_run = runner.invoke(
        cli,
        [
            "pipeline",
            "run",
            "--profile",
            "triage_fast",
            "--toi",
            "TOI-456.01",
            "--toi",
            "TOI-789.01",
            "--out-dir",
            str(out_dir),
            "--network-ok",
            "--continue-on-error",
            "--max-workers",
            "1",
        ],
    )
    assert first_run.exit_code == 0, first_run.output
    assert "Pipeline run complete: n_tois=2 ok=1 partial=1 failed=0" in first_run.output

    partial_marker_paths_before = sorted((out_dir / "TOI-456.01" / "checkpoints").glob("*.done.json"))
    ok_marker_paths_before = sorted((out_dir / "TOI-789.01" / "checkpoints").glob("*.done.json"))
    assert len(partial_marker_paths_before) == 6
    assert len(ok_marker_paths_before) == 7

    recorded_calls.clear()
    resumed_run = runner.invoke(
        cli,
        [
            "pipeline",
            "run",
            "--profile",
            "triage_fast",
            "--toi",
            "TOI-456.01",
            "--toi",
            "TOI-789.01",
            "--out-dir",
            str(out_dir),
            "--network-ok",
            "--continue-on-error",
            "--max-workers",
            "1",
            "--resume",
        ],
    )
    assert resumed_run.exit_code == 0, resumed_run.output
    assert "Pipeline run complete: n_tois=2 ok=2 partial=0 failed=0" in resumed_run.output

    assert recorded_calls == [{"command": "model-compete", "toi": "TOI-456.01"}]

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"n_tois": 2, "n_ok": 2, "n_partial": 0, "n_failed": 0}
    assert manifest["options"]["resume"] is True

    partial_result = json.loads((out_dir / "TOI-456.01" / "pipeline_result.json").read_text(encoding="utf-8"))
    ok_result = json.loads((out_dir / "TOI-789.01" / "pipeline_result.json").read_text(encoding="utf-8"))

    assert partial_result["status"] == "ok"
    assert ok_result["status"] == "ok"
    assert sum(1 for row in partial_result["steps"] if row.get("skipped_resume") is True) == 6
    assert sum(1 for row in partial_result["steps"] if row.get("skipped_resume") is False) == 1
    assert any(
        row["step_id"] == "model_compete" and row["status"] == "ok" and row.get("skipped_resume") is False
        for row in partial_result["steps"]
    )
    assert sum(1 for row in ok_result["steps"] if row.get("skipped_resume") is True) == 7

    partial_marker_paths_after = sorted((out_dir / "TOI-456.01" / "checkpoints").glob("*.done.json"))
    ok_marker_paths_after = sorted((out_dir / "TOI-789.01" / "checkpoints").glob("*.done.json"))
    assert len(partial_marker_paths_after) == 7
    assert len(ok_marker_paths_after) == 7
