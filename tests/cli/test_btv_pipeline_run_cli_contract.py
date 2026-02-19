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
