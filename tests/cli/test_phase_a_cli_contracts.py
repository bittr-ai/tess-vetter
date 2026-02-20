from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

import tess_vetter.cli.enrich_cli as enrich_cli
import tess_vetter.cli.vet_cli as vet_cli
from tess_vetter.cli.progress_metadata import ProgressIOError
from tess_vetter.platform.catalogs.toi_resolution import LookupStatus
from tess_vetter.platform.io.mast_client import LightCurveNotFoundError


def _fake_vet_payload(
    *,
    results: list[dict[str, Any]] | None = None,
    input_resolution: dict[str, Any] | None = None,
    coordinate_resolution: dict[str, Any] | None = None,
    include_summary: bool = False,
) -> dict[str, Any]:
    payload = {
        "results": results or [],
        "warnings": [],
        "provenance": {"pipeline_version": "0.1.0"},
        "inputs_summary": {},
    }
    if input_resolution is not None:
        payload["inputs_summary"]["input_resolution"] = input_resolution
    if coordinate_resolution is not None:
        payload["inputs_summary"]["coordinate_resolution"] = coordinate_resolution
    if include_summary:
        payload["summary"] = vet_cli._build_root_summary(payload=payload)
    return payload


def _base_vet_args() -> list[str]:
    return [
        "vet",
        "--tic-id",
        "123",
        "--period-days",
        "10.5",
        "--t0-btjd",
        "2000.2",
        "--duration-hours",
        "2.5",
    ]


def test_cli001_toi_resolution_and_override_precedence(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class _ToiResult:
        status = LookupStatus.OK
        tic_id = 188646744
        period_days = 14.2423724
        t0_btjd = 3540.26317
        duration_hours = 4.046
        depth_ppm = 320.0
        message = None

    def _fake_execute_vet(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _fake_vet_payload(input_resolution=kwargs.get("input_resolution"))

    monkeypatch.setattr(
        "tess_vetter.cli.vet_cli.resolve_toi_to_tic_ephemeris_depth",
        lambda *_a, **_k: _ToiResult(),
    )
    monkeypatch.setattr("tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "vet",
            "--toi",
            "TOI-5807.01",
            "--network-ok",
            "--period-days",
            "11.25",
            "--duration-hours",
            "3.25",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["tic_id"] == 188646744
    assert captured["period_days"] == 11.25
    assert captured["duration_hours"] == 3.25

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    input_resolution = payload["inputs_summary"]["input_resolution"]
    assert input_resolution["source"] == "toi_catalog"
    assert "period_days" in input_resolution["overrides"]
    assert "duration_hours" in input_resolution["overrides"]


def test_cli002_summary_block_is_present_and_deduplicates_concerns(monkeypatch, tmp_path: Path) -> None:
    def _fake_execute_vet(**_kwargs: Any) -> dict[str, Any]:
        return _fake_vet_payload(
            results=[
                {"id": "V01", "status": "ok", "flags": []},
                {
                    "id": "V16",
                    "status": "error",
                    "flags": ["MODEL_PREFERS_NON_TRANSIT", "MODEL_PREFERS_NON_TRANSIT"],
                },
                {"id": "V06", "status": "skipped", "flags": ["NO_COORDINATES"]},
            ],
            include_summary=True,
        )

    monkeypatch.setattr("tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, [*_base_vet_args(), "--out", str(out_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["n_ok"] == 1
    assert summary["n_failed"] == 1
    assert summary["n_skipped"] == 1
    assert summary["n_network_errors"] == 0
    assert summary["n_flagged"] == 1
    assert "V16" in summary["flagged_checks"]
    assert summary["concerns"] == ["MODEL_PREFERS_NON_TRANSIT"]
    assert summary["disposition_hint"] == "needs_model_competition_review"


def test_cli002_network_errors_counted_separately(monkeypatch, tmp_path: Path) -> None:
    def _fake_execute_vet(**_kwargs: Any) -> dict[str, Any]:
        return _fake_vet_payload(
            results=[
                {"id": "V06", "status": "skipped", "flags": ["SKIPPED:NETWORK_TIMEOUT"]},
                {"id": "V07", "status": "skipped", "flags": ["SKIPPED:NETWORK_ERROR"]},
                {"id": "V08", "status": "skipped", "flags": ["SKIPPED:NO_TPF"]},
                {"id": "V09", "status": "skipped", "flags": ["SKIPPED:NETWORK_DISABLED"]},
            ],
            include_summary=True,
        )

    monkeypatch.setattr("tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)

    out_path = tmp_path / "vet_network_summary.json"
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, [*_base_vet_args(), "--out", str(out_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["n_skipped"] == 4
    assert summary["n_network_errors"] == 2


def test_cli002_high_salience_flags_include_v17_and_v09(monkeypatch, tmp_path: Path) -> None:
    def _fake_execute_vet(**_kwargs: Any) -> dict[str, Any]:
        return _fake_vet_payload(
            results=[
                {"id": "V17", "status": "ok", "flags": ["V17_REGIME_MARGINAL"]},
                {"id": "V09", "status": "ok", "flags": ["DIFFIMG_UNRELIABLE"]},
            ],
            include_summary=True,
        )

    monkeypatch.setattr("tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)
    out_path = tmp_path / "vet_salience.json"
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, [*_base_vet_args(), "--out", str(out_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert "V17" in summary["flagged_checks"]
    assert "V09" in summary["flagged_checks"]
    assert "V17_REGIME_MARGINAL" in summary["concerns"]
    assert "DIFFIMG_UNRELIABLE" in summary["concerns"]


def test_cli003_coordinate_auto_resolution_from_tic(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class _CoordResult:
        status = LookupStatus.OK
        ra_deg = 210.1
        dec_deg = -20.2
        message = None

    def _fake_execute_vet(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _fake_vet_payload(coordinate_resolution=kwargs.get("coordinate_resolution"))

    monkeypatch.setattr("tess_vetter.cli.vet_cli._execute_vet", _fake_execute_vet)
    monkeypatch.setattr(
        "tess_vetter.cli.vet_cli.lookup_tic_coordinates",
        lambda **_kwargs: _CoordResult(),
    )

    out_path = tmp_path / "vet.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [*_base_vet_args(), "--network-ok", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert captured["ra_deg"] == 210.1
    assert captured["dec_deg"] == -20.2

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    coordinate_resolution = payload["inputs_summary"]["coordinate_resolution"]
    assert coordinate_resolution["source"] == "mast"


def test_cli003_require_coordinates_failure_path(monkeypatch) -> None:
    class _CoordResult:
        status = LookupStatus.DATA_UNAVAILABLE
        ra_deg = None
        dec_deg = None
        message = "no coords"

    monkeypatch.setattr(
        "tess_vetter.cli.vet_cli.lookup_tic_coordinates",
        lambda **_kwargs: _CoordResult(),
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [*_base_vet_args(), "--network-ok", "--require-coordinates"],
    )

    assert result.exit_code == 4


def test_cli004_describe_checks_text_and_json(monkeypatch) -> None:
    checks_payload = [
        {
            "id": "V01",
            "name": "Odd Even Depth",
            "tier": "LC",
            "requirements": {
                "needs_tpf": False,
                "needs_network": False,
                "needs_ra_dec": False,
                "needs_tic_id": False,
                "needs_stellar": False,
                "optional_deps": [],
            },
            "citations": [],
        },
        {
            "id": "V06",
            "name": "Nearby EB Search",
            "tier": "CATALOG",
            "requirements": {
                "needs_tpf": False,
                "needs_network": True,
                "needs_ra_dec": True,
                "needs_tic_id": False,
                "needs_stellar": False,
                "optional_deps": [],
            },
            "citations": [],
        },
    ]

    monkeypatch.setattr("tess_vetter.api.pipeline.list_checks", lambda *_a, **_k: checks_payload)
    monkeypatch.setattr(
        "tess_vetter.api.pipeline.describe_checks",
        lambda *_a, **_k: "Available vetting checks:\n\n  V01: Odd Even Depth\n  V06: Nearby EB Search\n",
    )

    runner = CliRunner()
    text_result = runner.invoke(enrich_cli.cli, ["describe-checks", "--format", "text"])
    assert text_result.exit_code == 0, text_result.output
    assert "V01" in text_result.output
    assert "V06" in text_result.output

    json_result = runner.invoke(enrich_cli.cli, ["describe-checks", "--format", "json"])
    assert json_result.exit_code == 0, json_result.output
    payload = json.loads(json_result.output)
    assert payload["meta"]["format"] == "json"
    checks_by_id = {row["id"]: row for row in payload["checks"]}
    assert checks_by_id["V06"]["requirements"]["needs_network"] is True


def test_cli006_exit_codes_0_through_5(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    out_path = tmp_path / "vet.json"

    monkeypatch.setattr("tess_vetter.cli.vet_cli._execute_vet", lambda **_k: _fake_vet_payload())
    ok = runner.invoke(enrich_cli.cli, [*_base_vet_args(), "--out", str(out_path)])
    assert ok.exit_code == 0, ok.output

    input_error = runner.invoke(enrich_cli.cli, [*_base_vet_args(), "--tpf-sector", "12"])
    assert input_error.exit_code == 1

    monkeypatch.setattr(
        "tess_vetter.cli.vet_cli._execute_vet",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    runtime_error = runner.invoke(enrich_cli.cli, _base_vet_args())
    assert runtime_error.exit_code == 2

    monkeypatch.setattr("tess_vetter.cli.vet_cli._execute_vet", lambda **_k: _fake_vet_payload())
    monkeypatch.setattr(
        "tess_vetter.cli.vet_cli.write_progress_metadata_atomic",
        lambda *_a, **_k: (_ for _ in ()).throw(ProgressIOError("disk full")),
    )
    progress_error = runner.invoke(
        enrich_cli.cli,
        [*_base_vet_args(), "--out", str(out_path), "--progress-path", str(tmp_path / "vet.progress.json")],
    )
    assert progress_error.exit_code == 3

    monkeypatch.setattr(
        "tess_vetter.cli.vet_cli._execute_vet",
        lambda **_k: (_ for _ in ()).throw(LightCurveNotFoundError("missing required data")),
    )
    missing_data = runner.invoke(enrich_cli.cli, _base_vet_args())
    assert missing_data.exit_code == 4

    monkeypatch.setattr(
        "tess_vetter.cli.vet_cli._execute_vet",
        lambda **_k: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    timeout = runner.invoke(enrich_cli.cli, _base_vet_args())
    assert timeout.exit_code == 5
