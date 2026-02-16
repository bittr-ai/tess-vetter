from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

import bittr_tess_vetter.cli.enrich_cli as enrich_cli
import bittr_tess_vetter.cli.report_cli as report_cli
from bittr_tess_vetter.cli.progress_metadata import ProgressIOError
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.pipeline import make_candidate_key
from bittr_tess_vetter.platform.catalogs.toi_resolution import LookupStatus
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError
from bittr_tess_vetter.validation.result_schema import CheckResult, VettingBundleResult


def test_btv_help_lists_enrich_vet_report() -> None:
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "enrich" in result.output
    assert "vet" in result.output
    assert "report" in result.output
    assert "resolve-stellar" in result.output
    assert "measure-sectors" in result.output


def test_btv_vet_success_writes_bundle_json(monkeypatch, tmp_path: Path) -> None:
    def _fake_execute_vet(**_kwargs):
        return {
            "results": [],
            "warnings": [],
            "provenance": {"pipeline_version": "0.1.0"},
            "inputs_summary": {},
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.vet.v2"
    assert payload["provenance"]["pipeline_version"] == "0.1.0"
    assert payload["provenance"]["confidence_semantics_ref"] == "docs/verification/confidence_semantics.md"
    assert payload["inputs_summary"]["confidence_semantics_ref"] == "docs/verification/confidence_semantics.md"
    assert payload["summary"]["n_network_errors"] == 0


def test_btv_vet_runtime_error_maps_to_exit_2(monkeypatch) -> None:
    def _boom(**_kwargs):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _boom)

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
        ],
    )

    assert result.exit_code == 2


def test_btv_vet_progress_error_maps_to_exit_3(monkeypatch, tmp_path: Path) -> None:
    def _fake_execute_vet(**_kwargs):
        return {
            "results": [],
            "warnings": [],
            "provenance": {},
            "inputs_summary": {},
        }

    def _raise_progress(*_args, **_kwargs):
        raise ProgressIOError("disk full")

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.vet_cli.write_progress_metadata_atomic",
        _raise_progress,
    )

    out_path = tmp_path / "vet.json"
    prog_path = tmp_path / "vet.progress.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--out",
            str(out_path),
            "--progress-path",
            str(prog_path),
        ],
    )

    assert result.exit_code == 3


def test_btv_report_lightcurve_missing_maps_to_exit_4(monkeypatch, tmp_path: Path) -> None:
    def _missing(**_kwargs):
        raise LightCurveNotFoundError("missing")

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _missing)

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--plot-data-out",
            str(tmp_path / "plot_data.json"),
        ],
    )

    assert result.exit_code == 4


def test_btv_report_timeout_maps_to_exit_5(monkeypatch, tmp_path: Path) -> None:
    def _timeout(**_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _timeout)

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--plot-data-out",
            str(tmp_path / "plot_data.json"),
        ],
    )

    assert result.exit_code == 5


def test_btv_report_success_writes_payload_and_html(monkeypatch, tmp_path: Path) -> None:
    def _ok(**_kwargs):
        return {
            "report_json": {
                "schema_version": "cli.report.v3",
                "provenance": {"vet_artifact": {"provided": False}},
                "verdict": None,
                "verdict_source": None,
                "report": {"schema_version": "1.0.0", "summary": {}},
            },
            "plot_data_json": {"full_lc": {"time": [1.0], "flux": [1.0]}},
            "html": "<html></html>",
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _ok)

    out_path = tmp_path / "report.json"
    plot_data_path = tmp_path / "report.json.plot_data.json"
    html_path = tmp_path / "report.html"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--include-html",
            "--out",
            str(out_path),
            "--html-out",
            str(html_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.report.v3"
    assert "verdict" in payload
    assert "verdict_source" in payload
    assert payload["report"]["schema_version"] == "1.0.0"
    assert json.loads(plot_data_path.read_text(encoding="utf-8"))["full_lc"]["time"] == [1.0]
    assert html_path.read_text(encoding="utf-8") == "<html></html>"


def test_btv_report_execute_report_sets_wrapper_provenance_fields(monkeypatch) -> None:
    class _FakeResult:
        def __init__(self) -> None:
            self.report_json = {"schema_version": "2.0.0", "summary": {"verdict": "PASS"}}
            self.plot_data_json = {"full_lc": {"time": [1.0], "flux": [1.0]}}
            self.html = None
            self.sectors_used = [14, 15]
            self.vet_artifact_reuse = None

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli.generate_report", lambda **_kwargs: _FakeResult())

    output = report_cli._execute_report(
        tic_id=123,
        period_days=10.5,
        t0_btjd=2000.2,
        duration_hours=2.5,
        depth_ppm=300.0,
        toi="TOI-123.01",
        sectors=[14, 15],
        flux_type="pdcsap",
        include_html=False,
        include_enrichment=False,
        custom_views=None,
        pipeline_config=report_cli.PipelineConfig(),
        vet_result=None,
        vet_result_path=None,
        resolved_inputs={
            "tic_id": 123,
            "period_days": 10.5,
            "t0_btjd": 2000.2,
            "duration_hours": 2.5,
            "depth_ppm": 300.0,
        },
    )

    payload = output["report_json"]
    assert payload["schema_version"] == "cli.report.v3"
    assert payload["provenance"]["sectors_used"] == [14, 15]
    assert payload["provenance"]["resolved_inputs"] == {
        "tic_id": 123,
        "period_days": 10.5,
        "t0_btjd": 2000.2,
        "duration_hours": 2.5,
        "depth_ppm": 300.0,
    }


def test_btv_report_passes_through_diagnostic_json_artifacts(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _ok(**_kwargs):
        captured.update(_kwargs)
        return {
            "report_json": {
                "schema_version": "cli.report.v3",
                "provenance": {"vet_artifact": {"provided": False}},
                "verdict": "ALL_CHECKS_PASSED",
                "verdict_source": "$.summary.bundle_summary",
                "report": {"schema_version": "2.0.0", "summary": {}},
            },
            "plot_data_json": {"full_lc": {"time": [1.0], "flux": [1.0]}},
            "html": None,
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _ok)

    diag_path = tmp_path / "activity.json"
    diag_path.write_text(
        json.dumps(
            {
                "schema_version": "cli.activity.v1",
                "result": {"activity": {}},
                "verdict": "spotted_rotator",
                "verdict_source": "$.activity.variability_class",
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "report_with_diag.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--diagnostic-json",
            str(diag_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["verdict"] == "ALL_CHECKS_PASSED"
    artifacts = captured["diagnostic_artifacts"]
    assert isinstance(artifacts, list)
    assert artifacts[0]["schema_version"] == "cli.activity.v1"
    assert artifacts[0]["verdict"] == "spotted_rotator"


def test_btv_report_rejects_malformed_vet_result_file(tmp_path: Path) -> None:
    vet_path = tmp_path / "vet.json"
    vet_path.write_text(json.dumps({"not": "a_vet_bundle"}), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--vet-result",
            str(vet_path),
            "--plot-data-out",
            str(tmp_path / "plot_data.json"),
        ],
    )
    assert result.exit_code == 1
    assert "Invalid vet result file schema" in result.output


def test_btv_report_rejects_mismatched_vet_result_candidate(tmp_path: Path) -> None:
    vet_path = tmp_path / "vet.json"
    vet_path.write_text(
        json.dumps(
            {
                "results": [],
                "warnings": [],
                "provenance": {},
                "inputs_summary": {
                    "input_resolution": {
                        "resolved": {
                            "tic_id": 999,
                            "period_days": 10.5,
                            "t0_btjd": 2000.2,
                            "duration_hours": 2.5,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--vet-result",
            str(vet_path),
            "--plot-data-out",
            str(tmp_path / "plot_data.json"),
        ],
    )
    assert result.exit_code == 1
    assert "vet result candidate mismatch" in result.output


def test_btv_report_forwards_vet_result_payload(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _ok(**kwargs):
        captured.update(kwargs)
        return {
            "report_json": {
                "schema_version": "cli.report.v3",
                "provenance": {"vet_artifact": {"provided": True}},
                "report": {"schema_version": "2.0.0", "summary": {}},
            },
            "plot_data_json": {},
            "html": None,
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _ok)

    vet_path = tmp_path / "vet.json"
    vet_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "V01",
                        "name": "Odd/even",
                        "status": "ok",
                        "confidence": 0.5,
                        "metrics": {},
                        "flags": [],
                        "notes": [],
                        "provenance": {},
                        "raw": None,
                    }
                ],
                "warnings": [],
                "provenance": {},
                "inputs_summary": {},
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--vet-result",
            str(vet_path),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["vet_result_path"] == str(vet_path)
    vet_payload = captured["vet_result"]
    assert isinstance(vet_payload, dict)
    assert isinstance(vet_payload.get("results"), list)


def test_btv_report_honors_plot_data_out_path(monkeypatch, tmp_path: Path) -> None:
    def _ok(**_kwargs):
        return {
            "report_json": {
                "schema_version": "cli.report.v3",
                "provenance": {"vet_artifact": {"provided": False}},
                "report": {"schema_version": "2.0.0", "summary": {}},
            },
            "plot_data_json": {"phase_folded": {"bin_minutes": 30.0}},
            "html": None,
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _ok)

    out_path = tmp_path / "report.json"
    plot_data_path = tmp_path / "custom_plot_data.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--out",
            str(out_path),
            "--plot-data-out",
            str(plot_data_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert plot_data_path.exists()
    payload = json.loads(plot_data_path.read_text(encoding="utf-8"))
    assert payload["phase_folded"]["bin_minutes"] == 30.0


def test_btv_report_stdout_requires_plot_data_out() -> None:
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--out",
            "-",
        ],
    )
    assert result.exit_code == 1
    assert "--out - requires --plot-data-out" in result.output


def test_btv_report_positional_toi_and_short_out_alias(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_resolve_candidate_inputs(**_kwargs):
        return (123, 10.5, 2000.2, 2.5, None, {"source": "toi_catalog"})

    def _ok(**kwargs):
        captured.update(kwargs)
        return {
            "report_json": {
                "schema_version": "cli.report.v3",
                "provenance": {"vet_artifact": {"provided": False}},
                "report": {"schema_version": "2.0.0", "summary": {}},
            },
            "plot_data_json": {},
            "html": None,
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._resolve_report_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _ok)

    out_path = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "TOI-123.01",
            "--network-ok",
            "-o",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert captured["toi"] == "TOI-123.01"


def test_btv_report_rejects_mismatched_positional_toi_and_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "TOI-123.01",
            "--toi",
            "TOI-999.01",
            "--plot-data-out",
            "plot.json",
        ],
    )
    assert result.exit_code == 1
    assert "Positional TOI argument and --toi must match" in result.output


def test_btv_report_toi_network_resolution_path(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _resolve_toi(_toi: str):
        return SimpleNamespace(
            status=LookupStatus.OK,
            tic_id=888,
            period_days=12.0,
            t0_btjd=1550.25,
            duration_hours=4.5,
            depth_ppm=410.0,
            message=None,
        )

    def _ok(**kwargs):
        captured.update(kwargs)
        return {
            "report_json": {
                "schema_version": "cli.report.v3",
                "provenance": {"vet_artifact": {"provided": False}},
                "report": {"schema_version": "2.0.0", "summary": {}},
            },
            "plot_data_json": {},
            "html": None,
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.report_seed.resolve_toi_to_tic_ephemeris_depth", _resolve_toi)
    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _ok)

    out_path = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--toi",
            "TOI-888.01",
            "--network-ok",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["tic_id"] == 888
    assert captured["period_days"] == 12.0
    assert captured["t0_btjd"] == 1550.25
    assert captured["duration_hours"] == 4.5
    assert captured["depth_ppm"] == 410.0


def test_btv_report_manual_ephemeris_with_toi_without_network_ok_still_runs(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _ok(**kwargs):
        captured.update(kwargs)
        return {
            "report_json": {
                "schema_version": "cli.report.v3",
                "provenance": {"vet_artifact": {"provided": False}},
                "report": {"schema_version": "2.0.0", "summary": {}},
            },
            "plot_data_json": {},
            "html": None,
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _ok)

    out_path = tmp_path / "report_manual_toi.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--toi",
            "TOI-123.01",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["tic_id"] == 123
    assert captured["period_days"] == 10.5
    assert captured["t0_btjd"] == 2000.2
    assert captured["duration_hours"] == 2.5
    assert captured["toi"] == "TOI-123.01"


def test_btv_report_file_seeds_inputs_and_cli_overrides_take_precedence(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _ok(**kwargs):
        captured.update(kwargs)
        return {
            "report_json": {
                "schema_version": "cli.report.v3",
                "provenance": {"vet_artifact": {"provided": False}},
                "report": {"schema_version": "2.0.0", "summary": {}},
            },
            "plot_data_json": {},
            "html": None,
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _ok)

    report_file = tmp_path / "seed_report.json"
    report_file.write_text(
        json.dumps(
            {
                "provenance": {"sectors_used": [2, 3]},
                "report": {
                    "summary": {
                        "tic_id": 789,
                        "ephemeris": {"period": 12.0, "t0": 1444.5, "duration_hours": 3.25},
                        "input_depth_ppm": 555.0,
                        "toi": "TOI-789.01",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "report_from_seed.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--report-file",
            str(report_file),
            "--period-days",
            "11.5",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["tic_id"] == 789
    assert captured["period_days"] == 11.5
    assert captured["t0_btjd"] == 1444.5
    assert captured["duration_hours"] == 3.25
    assert captured["depth_ppm"] == 555.0


def test_btv_report_requires_toi_or_full_ephemeris_inputs(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--plot-data-out",
            str(tmp_path / "plot_data.json"),
        ],
    )
    assert result.exit_code == 1
    assert "Missing required inputs" in result.output


def test_btv_vet_invalid_extra_param_maps_to_exit_1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--extra-param",
            "broken",
        ],
    )
    assert result.exit_code == 1


def test_btv_vet_resume_skips_when_progress_is_complete(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_execute_vet(**_kwargs):
        calls.append("called")
        return {"results": [], "warnings": [], "provenance": {}, "inputs_summary": {}}

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    candidate_key = make_candidate_key(123, 10.5, 2000.2)
    out_path = tmp_path / "vet.json"
    progress_path = tmp_path / "vet.progress.json"
    out_path.write_text("{}", encoding="utf-8")
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "single_candidate",
                "command": "vet",
                "output_path": str(out_path),
                "resume": True,
                "total_input": 1,
                "processed": 1,
                "skipped_resume": 0,
                "errors": 0,
                "wall_time_seconds": 0.1,
                "error_class_counts": {},
                "last_candidate_key": candidate_key,
                "updated_unix": 0.0,
                "status": "completed",
                "candidate": {
                    "tic_id": 123,
                    "period_days": 10.5,
                    "t0_btjd": 2000.2,
                    "duration_hours": 2.5,
                    "candidate_key": candidate_key,
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--out",
            str(out_path),
            "--progress-path",
            str(progress_path),
            "--resume",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == []


def test_btv_report_resume_skips_when_progress_is_complete(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_execute_report(**_kwargs):
        calls.append("called")
        return {"report_json": {"schema_version": "1.0.0"}, "html": None}

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _fake_execute_report)

    candidate_key = make_candidate_key(123, 10.5, 2000.2)
    out_path = tmp_path / "report.json"
    progress_path = tmp_path / "report.progress.json"
    out_path.write_text("{}", encoding="utf-8")
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "single_candidate",
                "command": "report",
                "output_path": str(out_path),
                "resume": True,
                "total_input": 1,
                "processed": 1,
                "skipped_resume": 0,
                "errors": 0,
                "wall_time_seconds": 0.1,
                "error_class_counts": {},
                "last_candidate_key": candidate_key,
                "updated_unix": 0.0,
                "status": "completed",
                "candidate": {
                    "tic_id": 123,
                    "period_days": 10.5,
                    "t0_btjd": 2000.2,
                    "duration_hours": 2.5,
                    "candidate_key": candidate_key,
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "report",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--out",
            str(out_path),
            "--progress-path",
            str(progress_path),
            "--resume",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == []


def test_btv_vet_pipeline_config_flags_forwarded(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_vet(**kwargs):
        captured.update(kwargs)
        return {"results": [], "warnings": [], "provenance": {}, "inputs_summary": {}}

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--timeout-seconds",
            "9.0",
            "--random-seed",
            "7",
            "--extra-param",
            "alpha=1",
            "--fail-fast",
            "--emit-warnings",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = captured["pipeline_config"]
    assert cfg.timeout_seconds == 9.0
    assert cfg.random_seed == 7
    assert cfg.fail_fast is True
    assert cfg.emit_warnings is True
    assert cfg.extra_params["alpha"] == 1


def test_btv_vet_stellar_flags_forwarded_and_emitted(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_vet(**kwargs):
        captured.update(kwargs)
        return {"results": [], "warnings": [], "provenance": {}, "inputs_summary": {}}

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--stellar-radius",
            "1.11",
            "--stellar-mass",
            "0.97",
            "--stellar-tmag",
            "10.2",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    stellar_block = captured["stellar_block"]
    assert stellar_block["radius_rsun"] == pytest.approx(1.11)
    assert stellar_block["mass_msun"] == pytest.approx(0.97)
    assert stellar_block["tmag"] == pytest.approx(10.2)
    assert stellar_block["source"] == "user"
    stellar_resolution = captured["stellar_resolution"]
    assert stellar_resolution["sources"] == {"radius": "explicit", "mass": "explicit", "tmag": "explicit"}


def test_btv_vet_stellar_file_and_explicit_precedence(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_vet(**kwargs):
        captured.update(kwargs)
        return {"results": [], "warnings": [], "provenance": {}, "inputs_summary": {}}

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    stellar_path = tmp_path / "stellar.json"
    stellar_path.write_text(
        json.dumps({"stellar": {"radius": 0.9, "mass": 0.8, "tmag": 11.3}}),
        encoding="utf-8",
    )

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--stellar-file",
            str(stellar_path),
            "--stellar-radius",
            "1.2",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    stellar_resolution = captured["stellar_resolution"]
    assert stellar_resolution["values"] == {"radius": 1.2, "mass": 0.8, "tmag": 11.3}
    assert stellar_resolution["sources"] == {"radius": "explicit", "mass": "file", "tmag": "file"}


def test_btv_vet_require_coordinates_without_network_maps_to_exit_4() -> None:
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--require-coordinates",
        ],
    )
    assert result.exit_code == 4


def test_btv_vet_require_coordinates_timeout_maps_to_exit_5(monkeypatch) -> None:
    def _timeout_lookup(*, tic_id: int):
        _ = tic_id
        return SimpleNamespace(
            status="timeout",
            ra_deg=None,
            dec_deg=None,
            message="coord lookup timeout",
        )

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.lookup_tic_coordinates", _timeout_lookup)
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--require-coordinates",
            "--network-ok",
        ],
    )
    assert result.exit_code == 5


def test_btv_vet_detrend_defaults_are_backward_compatible(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_vet(**kwargs):
        captured.update(kwargs)
        return {"results": [], "warnings": [], "provenance": {}, "inputs_summary": {}}

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["detrend"] is None
    assert captured["detrend_bin_hours"] == 6.0
    assert captured["detrend_buffer"] == 2.0
    assert captured["detrend_sigma_clip"] == 5.0


def test_btv_vet_detrend_method_and_params_forwarded(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_vet(**kwargs):
        captured.update(kwargs)
        return {"results": [], "warnings": [], "provenance": {}, "inputs_summary": {}}

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--detrend",
            "transit_masked_bin_median",
            "--detrend-bin-hours",
            "8",
            "--detrend-buffer",
            "2.5",
            "--detrend-sigma-clip",
            "4.0",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["detrend"] == "transit_masked_bin_median"
    assert captured["detrend_bin_hours"] == 8.0
    assert captured["detrend_buffer"] == 2.5
    assert captured["detrend_sigma_clip"] == 4.0


def test_btv_vet_rejects_unknown_detrend_method(tmp_path: Path) -> None:
    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--detrend",
            "unknown_method",
            "--detrend-bin-hours",
            "8",
            "--detrend-buffer",
            "2.5",
            "--detrend-sigma-clip",
            "4.0",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 1
    assert "--detrend must be one of: transit_masked_bin_median" in result.output


def test_btv_vet_emits_detrend_provenance_when_enabled(monkeypatch, tmp_path: Path) -> None:
    time = np.linspace(2000.0, 2012.0, 800, dtype=np.float64)
    baseline = 1.0 + 0.001 * (time - np.min(time))
    flux = baseline.copy()
    flux[np.abs(((time - 2000.2) % 10.5) - 10.5 / 2.0) < (2.5 / 24.0)] -= 0.0002
    flux_err = np.full_like(flux, 1e-4, dtype=np.float64)
    quality = np.zeros_like(flux, dtype=np.int32)
    valid = np.ones_like(flux, dtype=bool)
    lc_data = LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid,
        tic_id=123,
        sector=1,
        cadence_seconds=120.0,
    )

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, *, flux_type: str, sectors: list[int] | None = None):
            return [lc_data]

    captured: dict[str, np.ndarray] = {}

    def _fake_vet_candidate(**kwargs):
        captured["flux"] = np.asarray(kwargs["lc"].flux, dtype=np.float64)
        return VettingBundleResult(results=[], warnings=[], provenance={}, inputs_summary={})

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.vet_candidate", _fake_vet_candidate)

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--detrend",
            "transit_masked_bin_median",
            "--detrend-bin-hours",
            "6.0",
            "--detrend-buffer",
            "2.0",
            "--detrend-sigma-clip",
            "5.0",
            "--sectors",
            "1",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    detrend = payload["provenance"]["detrend"]
    assert detrend["applied"] is True
    assert detrend["method"] == "transit_masked_bin_median"
    assert detrend["bin_hours"] == 6.0
    assert detrend["buffer_factor"] == 2.0
    assert detrend["sigma_clip"] == 5.0
    assert detrend["depth_source"] == "transit_masked_in_out_median"
    assert detrend["depth_availability"] in {"available", "unavailable"}
    if detrend["depth_availability"] == "available":
        assert np.isfinite(float(detrend["depth_ppm"]))
        assert np.isfinite(float(detrend["depth_err_ppm"]))
        assert float(detrend["depth_ppm"]) > 0.0
        assert float(detrend["depth_err_ppm"]) >= 0.0
    else:
        assert detrend["depth_ppm"] is None
        assert detrend["depth_err_ppm"] is None
        assert isinstance(detrend.get("depth_note"), str)
        assert detrend["depth_note"] != ""
    assert payload["provenance"]["sectors_requested"] == [1]
    assert payload["provenance"]["sectors_used"] == [1]
    assert payload["provenance"]["discovered_sectors"] == [1]
    assert not np.allclose(captured["flux"], flux)


def test_btv_vet_detrend_depth_unavailable_sets_note(monkeypatch, tmp_path: Path) -> None:
    flux = np.ones(64, dtype=np.float64)
    flux_err = np.full(64, 1e-4, dtype=np.float64)
    time = np.linspace(0.0, 6.3, 64, dtype=np.float64)

    class _FakeLightCurveData:
        def __init__(self) -> None:
            self.time = time
            self.flux = flux
            self.flux_err = flux_err
            self.quality = np.zeros_like(time, dtype=np.int32)
            self.valid_mask = np.ones_like(time, dtype=bool)
            self.sector = 1

    class _FakeMASTClient:
        def download_all_sectors(self, *_args, **_kwargs):
            return [_FakeLightCurveData()]

    def _fake_vet_candidate(**_kwargs):
        return VettingBundleResult(results=[], warnings=[], provenance={})

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.vet_candidate", _fake_vet_candidate)
    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.measure_transit_depth", lambda *_a, **_k: (float("nan"), 0.0))

    out_path = tmp_path / "vet_detrend_unavailable.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "1.5",
            "--t0-btjd",
            "0.1",
            "--duration-hours",
            "2.0",
            "--detrend",
            "transit_masked_bin_median",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    detrend = payload["provenance"]["detrend"]
    assert detrend["depth_availability"] == "unavailable"
    assert detrend["depth_ppm"] is None
    assert detrend["depth_err_ppm"] is None
    assert isinstance(detrend.get("depth_note"), str)


def test_btv_vet_warns_when_stellar_missing_with_network(monkeypatch, tmp_path: Path) -> None:
    class _FakeLightCurveData:
        def __init__(self) -> None:
            n = 64
            self.time = np.linspace(0.0, 6.3, n, dtype=np.float64)
            self.flux = np.ones(n, dtype=np.float64)
            self.flux_err = np.full(n, 1e-4, dtype=np.float64)
            self.quality = np.zeros(n, dtype=np.int32)
            self.valid_mask = np.ones(n, dtype=bool)
            self.sector = 1

    class _FakeMASTClient:
        def download_all_sectors(self, *_args, **_kwargs):
            return [_FakeLightCurveData()]

    def _fake_vet_candidate(**_kwargs):
        return VettingBundleResult(results=[], warnings=[], provenance={})

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.vet_candidate", _fake_vet_candidate)

    out_path = tmp_path / "vet_missing_stellar_warning.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "1.5",
            "--t0-btjd",
            "0.1",
            "--duration-hours",
            "2.0",
            "--network-ok",
            "--ra-deg",
            "10.0",
            "--dec-deg",
            "20.0",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    warnings = payload.get("warnings", [])
    assert any("Stellar parameters unavailable from TIC/MAST" in str(w) for w in warnings)


def test_btv_vet_tpf_flags_forwarded(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_vet(**kwargs):
        captured.update(kwargs)
        return {"results": [], "warnings": [], "provenance": {}, "inputs_summary": {}}

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)
    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--fetch-tpf",
            "--tpf-sector-strategy",
            "requested",
            "--tpf-sector",
            "4",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["fetch_tpf"] is True
    assert captured["require_tpf"] is False
    assert captured["tpf_sector_strategy"] == "requested"
    assert captured["tpf_sectors"] == [4]


def test_btv_vet_require_tpf_forces_fetch(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_vet(**kwargs):
        captured.update(kwargs)
        return {"results": [], "warnings": [], "provenance": {}, "inputs_summary": {}}

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)
    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--require-tpf",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["require_tpf"] is True
    assert captured["fetch_tpf"] is True


def test_btv_vet_require_tpf_missing_maps_to_exit_4(monkeypatch) -> None:
    def _missing(**_kwargs):
        raise LightCurveNotFoundError("TPF unavailable")

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _missing)
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--require-tpf",
        ],
    )

    assert result.exit_code == 4


def test_btv_vet_tpf_sector_requires_requested_strategy() -> None:
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--tpf-sector",
            "5",
        ],
    )
    assert result.exit_code == 1


def test_btv_vet_default_emits_lc_summary_and_meta_enabled_computed(monkeypatch, tmp_path: Path) -> None:
    lc_data = LightCurveData(
        time=np.linspace(0.0, 6.3, 128, dtype=np.float64),
        flux=np.ones(128, dtype=np.float64),
        flux_err=np.full(128, 1e-4, dtype=np.float64),
        quality=np.zeros(128, dtype=np.int32),
        valid_mask=np.ones(128, dtype=bool),
        tic_id=123,
        sector=1,
        cadence_seconds=120.0,
    )

    class _FakeMASTClient:
        def download_all_sectors(self, *_args, **_kwargs):
            return [lc_data]

    def _fake_vet_candidate(**_kwargs):
        return VettingBundleResult(
            results=[
                CheckResult(
                    id="V01",
                    name="odd_even_depth",
                    status="ok",
                    confidence=0.9,
                    metrics={},
                    flags=[],
                    notes=[],
                    provenance={},
                    raw=None,
                )
            ],
            warnings=[],
            provenance={},
            inputs_summary={},
        )

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.vet_candidate", _fake_vet_candidate)

    out_path = tmp_path / "vet_lc_summary_default.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "2.5",
            "--t0-btjd",
            "0.1",
            "--duration-hours",
            "2.0",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload.get("lc_summary"), dict)
    meta = payload["lc_summary_meta"]
    assert meta["enabled"] is True
    assert meta["computed"] is True
    summary = payload["summary"]
    assert summary["n_ok"] == 1
    assert summary["n_failed"] == 0
    assert summary["n_skipped"] == 0
    assert summary["n_network_errors"] == 0


def test_btv_vet_no_lc_summary_sets_disabled_reason_code(monkeypatch, tmp_path: Path) -> None:
    lc_data = LightCurveData(
        time=np.linspace(0.0, 6.3, 128, dtype=np.float64),
        flux=np.ones(128, dtype=np.float64),
        flux_err=np.full(128, 1e-4, dtype=np.float64),
        quality=np.zeros(128, dtype=np.int32),
        valid_mask=np.ones(128, dtype=bool),
        tic_id=123,
        sector=1,
        cadence_seconds=120.0,
    )

    class _FakeMASTClient:
        def download_all_sectors(self, *_args, **_kwargs):
            return [lc_data]

    def _fake_vet_candidate(**_kwargs):
        return VettingBundleResult(results=[], warnings=[], provenance={}, inputs_summary={})

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.vet_candidate", _fake_vet_candidate)

    out_path = tmp_path / "vet_lc_summary_disabled.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "2.5",
            "--t0-btjd",
            "0.1",
            "--duration-hours",
            "2.0",
            "--no-lc-summary",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["lc_summary"] is None
    meta = payload["lc_summary_meta"]
    assert meta["enabled"] is False
    assert meta["computed"] is False
    assert meta["reason"] == "disabled_by_flag"
    summary = payload["summary"]
    assert summary["n_network_errors"] == 0
    assert "flagged_checks" in summary


def test_btv_vet_lc_summary_compute_failure_degrades_with_stable_reason(monkeypatch, tmp_path: Path) -> None:
    lc_data = LightCurveData(
        time=np.linspace(0.0, 6.3, 128, dtype=np.float64),
        flux=np.ones(128, dtype=np.float64),
        flux_err=np.full(128, 1e-4, dtype=np.float64),
        quality=np.zeros(128, dtype=np.int32),
        valid_mask=np.ones(128, dtype=bool),
        tic_id=123,
        sector=1,
        cadence_seconds=120.0,
    )

    class _FakeMASTClient:
        def download_all_sectors(self, *_args, **_kwargs):
            return [lc_data]

    def _fake_vet_candidate(**_kwargs):
        return VettingBundleResult(results=[], warnings=[], provenance={}, inputs_summary={})

    def _boom(*_args, **_kwargs):
        raise RuntimeError("report seam failed")

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.vet_candidate", _fake_vet_candidate)
    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli.build_report_with_vet_artifact", _boom)

    out_path = tmp_path / "vet_lc_summary_compute_failed.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "2.5",
            "--t0-btjd",
            "0.1",
            "--duration-hours",
            "2.0",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["lc_summary"] is None
    meta = payload["lc_summary_meta"]
    assert meta["enabled"] is True
    assert meta["computed"] is False
    assert meta["reason"] == "compute_failed"
    summary = payload["summary"]
    assert summary["n_network_errors"] == 0
    assert "disposition_hint" in summary


def test_btv_vet_summary_surfaces_v04_instability_concerns(monkeypatch, tmp_path: Path) -> None:
    def _fake_execute_vet(**_kwargs):
        return {
            "results": [
                {
                    "id": "V04",
                    "status": "ok",
                    "flags": [],
                    "metrics": {
                        "mean_depth_ppm": 1000.0,
                        "depth_scatter_ppm": 1100.0,
                        "chi2_reduced": 5.4,
                        "dom_ratio": 1.7,
                    },
                }
            ],
            "warnings": [],
            "provenance": {},
            "inputs_summary": {},
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)
    out_path = tmp_path / "vet_v04_summary.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert "V04" in summary["flagged_checks"]
    assert "DEPTH_HIGHLY_UNSTABLE" in summary["concerns"]
    assert "DEPTH_DOMINATED_BY_SINGLE_EPOCH" in summary["concerns"]


def test_btv_vet_sector_measurements_forwarded_and_emitted(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_execute_vet(**kwargs):
        captured.update(kwargs)
        return {
            "results": [
                {
                    "id": "V21",
                    "status": "ok",
                    "flags": [],
                    "raw": {"measurements": kwargs.get("sector_measurements", [])},
                }
            ],
            "warnings": [],
            "provenance": {},
            "inputs_summary": {},
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    sector_path = tmp_path / "sector_measurements.json"
    sector_path.write_text(
        json.dumps(
            {
                "sector_measurements": [
                    {"sector": 4, "depth_ppm": 500.0, "depth_err_ppm": 50.0, "quality_weight": 1.0},
                    {"sector": 5, "depth_ppm": 520.0, "depth_err_ppm": 55.0, "quality_weight": 1.0},
                ],
                "provenance": {"command": "measure-sectors"},
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--sector-measurements",
            str(sector_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert isinstance(captured.get("sector_measurements"), list)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "sector_measurements" in payload
    assert "sector_gating" in payload
    assert payload["sector_gating"]["v21_status"] == "ok"
    assert payload["sector_gating"]["used_by_v21"] is True


def test_btv_vet_sector_measurements_schema_error_maps_to_exit_1(tmp_path: Path) -> None:
    sector_path = tmp_path / "broken_sector_measurements.json"
    sector_path.write_text(
        json.dumps(
            {
                "sector_measurements": [
                    {"sector": 4, "depth_ppm": 500.0, "depth_err_ppm": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--sector-measurements",
            str(sector_path),
        ],
    )
    assert result.exit_code == 1
