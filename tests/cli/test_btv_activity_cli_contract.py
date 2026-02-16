from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from bittr_tess_vetter.api.types import LightCurve
from bittr_tess_vetter.cli.activity_cli import activity_command


def test_btv_activity_success_payload_contract(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_download_and_stitch_lightcurve(**_kwargs: Any) -> tuple[LightCurve, list[int], str]:
        return (
            LightCurve(
                time=[1.0, 2.0, 3.0],
                flux=[1.0, 0.999, 1.001],
                flux_err=[0.001, 0.001, 0.001],
            ),
            [14, 15],
            "mast_filtered",
        )

    class _FakeActivityResult:
        def to_dict(self) -> dict[str, Any]:
            return {
                "rotation_period": 6.25,
                "variability_class": "spotted_rotator",
                "n_flares": 2,
            }

    def _fake_characterize_activity(**kwargs: Any) -> _FakeActivityResult:
        seen.update(kwargs)
        return _FakeActivityResult()

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.activity_cli._download_and_stitch_lightcurve",
        _fake_download_and_stitch_lightcurve,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.activity_cli.characterize_activity", _fake_characterize_activity)

    out_path = tmp_path / "activity.json"
    runner = CliRunner()
    result = runner.invoke(
        activity_command,
        [
            "--tic-id",
            "123",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--flux-type",
            "sap",
            "--no-detect-flares",
            "--flare-sigma",
            "6.5",
            "--rotation-min-period",
            "1.0",
            "--rotation-max-period",
            "20.0",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["detect_flares"] is False
    assert seen["flare_sigma"] == 6.5
    assert seen["rotation_min_period"] == 1.0
    assert seen["rotation_max_period"] == 20.0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.activity.v1"
    assert payload["activity"]["rotation_period"] == 6.25
    assert payload["result"]["activity"]["rotation_period"] == 6.25
    assert "verdict" in payload
    assert "verdict_source" in payload
    assert payload["verdict"] == "spotted_rotator"
    assert payload["verdict_source"] == "$.activity.variability_class"
    assert payload["result"]["activity"] == payload["activity"]
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["inputs_summary"]["tic_id"] == 123
    assert payload["inputs_summary"]["input_resolution"]["source"] == "cli"
    assert payload["provenance"]["sectors_used"] == [14, 15]
    assert payload["provenance"]["options"]["flux_type"] == "sap"
    assert payload["provenance"]["options"]["detect_flares"] is False


def test_btv_activity_toi_without_network_ok_exits_4() -> None:
    runner = CliRunner()
    result = runner.invoke(
        activity_command,
        [
            "--toi",
            "123.01",
        ],
    )
    assert result.exit_code == 4
    assert "--toi requires --network-ok" in result.output


def test_btv_activity_no_sectors_available_exits_4(monkeypatch) -> None:
    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None = None):
            _ = tic_id, flux_type, sectors
            return []

    monkeypatch.setattr("bittr_tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

    runner = CliRunner()
    result = runner.invoke(
        activity_command,
        [
            "--tic-id",
            "123",
        ],
    )
    assert result.exit_code == 4


def test_btv_activity_accepts_positional_toi_and_short_o(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_tic_and_inputs(**kwargs: Any):
        seen.update(kwargs)
        return 123, {"source": "toi", "inputs": {"toi": kwargs.get("toi")}}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.activity_cli._resolve_tic_and_inputs",
        _fake_resolve_tic_and_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.activity_cli._download_and_stitch_lightcurve",
        lambda **_kwargs: (
            LightCurve(time=[1.0, 2.0], flux=[1.0, 1.0], flux_err=[0.001, 0.001]),
            [14],
            "mast_discovery",
        ),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.activity_cli.characterize_activity",
        lambda **_kwargs: type("A", (), {"to_dict": lambda self: {"rotation_period": 5.0}})(),
    )

    out_path = tmp_path / "activity_positional.json"
    runner = CliRunner()
    result = runner.invoke(activity_command, ["TOI-5807.01", "-o", str(out_path)])
    assert result.exit_code == 0, result.output
    assert seen["toi"] == "TOI-5807.01"


def test_btv_activity_rejects_mismatched_positional_and_option_toi() -> None:
    runner = CliRunner()
    result = runner.invoke(activity_command, ["TOI-5807.01", "--toi", "TOI-4510.01"])
    assert result.exit_code == 1
    assert "must match" in result.output


def test_btv_activity_report_file_inputs_override_toi(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "activity.report.json"
    report_path.write_text(
        json.dumps(
            {
                "report": {
                    "summary": {
                        "tic_id": 555,
                        "ephemeris": {
                            "period_days": 6.0,
                            "t0_btjd": 2450.0,
                            "duration_hours": 2.0,
                        },
                        "input_depth_ppm": 400.0,
                    },
                    "provenance": {"sectors_used": [13, 14]},
                }
            }
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {}

    def _fake_download_and_stitch_lightcurve(**kwargs: Any):
        seen.update(kwargs)
        return (
            LightCurve(time=[1.0, 2.0], flux=[1.0, 0.999], flux_err=[0.001, 0.001]),
            [13, 14],
            "mast_filtered",
        )

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.activity_cli._download_and_stitch_lightcurve",
        _fake_download_and_stitch_lightcurve,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.activity_cli.characterize_activity",
        lambda **_kwargs: type("A", (), {"to_dict": lambda self: {"variability_class": "quiet"}})(),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.activity_cli._resolve_tic_and_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve TOI with report file")),
    )

    out_path = tmp_path / "activity_report_file.json"
    runner = CliRunner()
    result = runner.invoke(
        activity_command,
        [
            "--report-file",
            str(report_path),
            "--toi",
            "TOI-555.01",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Warning: --report-file provided; ignoring --toi" in result.output
    assert seen["tic_id"] == 555
    assert seen["sectors"] == [13, 14]
    assert seen["sectors_explicit"] is False
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["inputs_summary"]["input_resolution"]["source"] == "report_file"
    assert payload["provenance"]["inputs_source"] == "report_file"
    assert payload["provenance"]["report_file"] == str(report_path.resolve())


def test_btv_activity_explicit_sectors_cache_miss_exits_4(monkeypatch) -> None:
    class _FakeMASTClient:
        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str):
            _ = tic_id, sector, flux_type
            raise RuntimeError("cache miss")

        def download_all_sectors(self, *_args: Any, **_kwargs: Any):
            raise AssertionError("download_all_sectors should not be called when --sectors is provided")

    monkeypatch.setattr("bittr_tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

    runner = CliRunner()
    result = runner.invoke(
        activity_command,
        [
            "--tic-id",
            "123",
            "--sectors",
            "14",
        ],
    )
    assert result.exit_code == 4
    assert "Cache-only sector load failed for TIC 123" in result.output
