from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import bittr_tess_vetter.cli.enrich_cli as enrich_cli
from bittr_tess_vetter.cli.progress_metadata import ProgressIOError
from bittr_tess_vetter.pipeline import make_candidate_key
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError


def test_btv_help_lists_enrich_vet_report() -> None:
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "enrich" in result.output
    assert "vet" in result.output
    assert "report" in result.output
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
    assert payload["provenance"]["pipeline_version"] == "0.1.0"


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


def test_btv_report_lightcurve_missing_maps_to_exit_4(monkeypatch) -> None:
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
        ],
    )

    assert result.exit_code == 4


def test_btv_report_timeout_maps_to_exit_5(monkeypatch) -> None:
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
        ],
    )

    assert result.exit_code == 5


def test_btv_report_success_writes_payload_and_html(monkeypatch, tmp_path: Path) -> None:
    def _ok(**_kwargs):
        return {
            "report_json": {"schema_version": "1.0.0", "summary": {}, "plot_data": {}},
            "html": "<html></html>",
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.report_cli._execute_report", _ok)

    out_path = tmp_path / "report.json"
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
    assert payload["schema_version"] == "1.0.0"
    assert html_path.read_text(encoding="utf-8") == "<html></html>"


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
