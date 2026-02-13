from __future__ import annotations

from pathlib import Path

import click
from click.testing import CliRunner

import bittr_tess_vetter.cli.enrich_cli as enrich_cli
from bittr_tess_vetter.pipeline import EnrichmentSummary


def _worklist_file(tmp_path: Path) -> Path:
    path = tmp_path / "worklist.jsonl"
    path.write_text(
        '{"tic_id": 123, "period_days": 10.0, "t0_btjd": 100.0, "duration_hours": 2.0}\n',
        encoding="utf-8",
    )
    return path


def test_btv_enrich_happy_path_forwards_resume_and_progress(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_enrich_worklist(*, worklist_iter, output_path, config, resume, limit, progress_interval):
        rows = list(worklist_iter)
        captured["rows"] = rows
        captured["output_path"] = output_path
        captured["config"] = config
        captured["resume"] = resume
        captured["limit"] = limit
        captured["progress_interval"] = progress_interval
        return EnrichmentSummary(
            total_input=1,
            processed=1,
            skipped_resume=0,
            errors=0,
            wall_time_seconds=0.01,
            error_class_counts={},
        )

    monkeypatch.setattr("bittr_tess_vetter.pipeline.enrich_worklist", _fake_enrich_worklist)

    runner = CliRunner()
    input_path = _worklist_file(tmp_path)
    out_path = tmp_path / "out.jsonl"
    result = runner.invoke(
        enrich_cli.cli,
        [
            "enrich",
            "--in",
            str(input_path),
            "--out",
            str(out_path),
            "--resume",
            "--progress-interval",
            "3",
            "--bulk",
            "--no-download",
            "--limit",
            "9",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["resume"] is True
    assert captured["progress_interval"] == 3
    assert captured["limit"] == 9
    assert captured["output_path"] == out_path
    assert len(captured["rows"]) == 1
    config = captured["config"]
    assert config.bulk_mode is True
    assert config.no_download is True
    assert "Processed:      1" in result.output
    assert "Errors:         0" in result.output


def test_btv_enrich_failure_from_pipeline_returns_exit_1(monkeypatch, tmp_path: Path) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("bittr_tess_vetter.pipeline.enrich_worklist", _boom)

    runner = CliRunner()
    input_path = _worklist_file(tmp_path)
    out_path = tmp_path / "out.jsonl"
    result = runner.invoke(
        enrich_cli.cli,
        [
            "enrich",
            "--in",
            str(input_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    assert "simulated failure" in str(result.exception)


def test_main_exit_code_mapping(monkeypatch) -> None:
    def _ok(*args, **kwargs) -> None:
        return None

    def _click_error(*args, **kwargs) -> None:
        raise click.ClickException("bad input")

    def _unexpected_error(*args, **kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(enrich_cli, "cli", _ok)
    assert enrich_cli.main() == 0

    monkeypatch.setattr(enrich_cli, "cli", _click_error)
    assert enrich_cli.main() == 1

    monkeypatch.setattr(enrich_cli, "cli", _unexpected_error)
    assert enrich_cli.main() == 1


def test_main_maps_option_first_invocation_to_enrich(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def _fake_cli(*, args, standalone_mode) -> None:
        seen["args"] = args
        seen["standalone_mode"] = standalone_mode

    input_path = _worklist_file(tmp_path)
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr(
        enrich_cli,
        "sys",
        type(
            "FakeSys",
            (),
            {"argv": ["btv", "--in", str(input_path), "--out", str(out_path)]},
        )(),
    )
    monkeypatch.setattr(enrich_cli, "cli", _fake_cli)

    assert enrich_cli.main() == 0
    assert seen["args"][0] == "enrich"
    assert seen["standalone_mode"] is False


def test_main_does_not_remap_root_version(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_cli(*, args, standalone_mode) -> None:
        seen["args"] = args
        seen["standalone_mode"] = standalone_mode

    monkeypatch.setattr(
        enrich_cli,
        "sys",
        type("FakeSys", (), {"argv": ["btv", "--version"]})(),
    )
    monkeypatch.setattr(enrich_cli, "cli", _fake_cli)

    assert enrich_cli.main() == 0
    assert seen["args"] == ["--version"]
    assert seen["standalone_mode"] is False
