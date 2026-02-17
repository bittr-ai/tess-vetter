from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from bittr_tess_vetter.api.types import LightCurve
from bittr_tess_vetter.cli.rv_feasibility_cli import rv_feasibility_command


def test_btv_rv_feasibility_success_payload_contract(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_download_and_stitch_lightcurve(**_kwargs: Any) -> tuple[LightCurve, list[int], str]:
        return (
            LightCurve(
                time=[1.0, 2.0, 3.0, 4.0],
                flux=[1.0, 0.999, 1.001, 1.0],
                flux_err=[0.001, 0.001, 0.001, 0.001],
            ),
            [14, 15],
            "mast_filtered",
        )

    class _FakeActivityResult:
        def to_dict(self) -> dict[str, Any]:
            return {
                "rotation_period": 7.2,
                "variability_amplitude_ppm": 1800.0,
                "variability_class": "spotted_rotator",
            }

    def _fake_characterize_activity(**kwargs: Any) -> _FakeActivityResult:
        seen.update(kwargs)
        return _FakeActivityResult()

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.rv_feasibility_cli._download_and_stitch_lightcurve",
        _fake_download_and_stitch_lightcurve,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.rv_feasibility_cli.characterize_activity",
        _fake_characterize_activity,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.rv_feasibility_cli.load_auto_stellar_with_fallback",
        lambda **_kwargs: ({"tmag": 9.2, "radius": 1.1, "mass": 1.0}, {"status": "ok"}),
    )

    out_path = tmp_path / "rv_feasibility.json"
    runner = CliRunner()
    result = runner.invoke(
        rv_feasibility_command,
        [
            "--tic-id",
            "123",
            "--network-ok",
            "--flux-type",
            "sap",
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
    assert seen["rotation_min_period"] == 1.0
    assert seen["rotation_max_period"] == 20.0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.rv_feasibility.v1"
    assert payload["activity"]["rotation_period"] == 7.2
    assert payload["result"]["activity"]["rotation_period"] == 7.2
    assert payload["rv_feasibility"]["verdict"] in {
        "HIGH_RV_FEASIBILITY",
        "MODERATE_RV_FEASIBILITY",
        "LOW_RV_FEASIBILITY",
    }
    assert payload["verdict"] == payload["rv_feasibility"]["verdict"]
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["verdict_source"] == "$.result.rv_feasibility.verdict"
    assert payload["result"]["rv_feasibility"] == payload["rv_feasibility"]
    assert payload["inputs_summary"]["tic_id"] == 123
    assert payload["provenance"]["options"]["flux_type"] == "sap"
    assert payload["provenance"]["stellar"]["source"] == "auto"


def test_btv_rv_feasibility_accepts_positional_toi_and_short_o(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_tic_and_inputs(**kwargs: Any):
        seen.update(kwargs)
        return 123, {"source": "toi", "inputs": {"toi": kwargs.get("toi")}}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.rv_feasibility_cli._resolve_tic_and_inputs",
        _fake_resolve_tic_and_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.rv_feasibility_cli._download_and_stitch_lightcurve",
        lambda **_kwargs: (
            LightCurve(time=[1.0, 2.0], flux=[1.0, 1.0], flux_err=[0.001, 0.001]),
            [14],
            "mast_discovery",
        ),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.rv_feasibility_cli.characterize_activity",
        lambda **_kwargs: type("A", (), {"to_dict": lambda self: {"rotation_period": 5.0}})(),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.rv_feasibility_cli.load_auto_stellar_with_fallback",
        lambda **_kwargs: ({"tmag": 9.9}, {"status": "ok"}),
    )

    out_path = tmp_path / "rv_positional.json"
    runner = CliRunner()
    result = runner.invoke(rv_feasibility_command, ["TOI-5807.01", "--network-ok", "-o", str(out_path)])
    assert result.exit_code == 0, result.output
    assert seen["toi"] == "TOI-5807.01"


def test_btv_rv_feasibility_rejects_mismatched_positional_and_option_toi() -> None:
    runner = CliRunner()
    result = runner.invoke(rv_feasibility_command, ["TOI-5807.01", "--toi", "TOI-4510.01"])
    assert result.exit_code == 1
    assert "must match" in result.output


def test_btv_rv_feasibility_report_file_inputs_override_toi(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "rv.report.json"
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
        "bittr_tess_vetter.cli.rv_feasibility_cli._download_and_stitch_lightcurve",
        _fake_download_and_stitch_lightcurve,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.rv_feasibility_cli.characterize_activity",
        lambda **_kwargs: type(
            "A",
            (),
            {"to_dict": lambda self: {"rotation_period": 7.0, "variability_amplitude_ppm": 1500.0}},
        )(),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.rv_feasibility_cli._resolve_tic_and_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve TOI with report file")),
    )

    out_path = tmp_path / "rv_report_file.json"
    runner = CliRunner()
    result = runner.invoke(
        rv_feasibility_command,
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

