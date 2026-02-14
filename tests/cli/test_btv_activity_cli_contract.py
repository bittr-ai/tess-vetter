from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from bittr_tess_vetter.api.types import LightCurve
from bittr_tess_vetter.cli.activity_cli import activity_command


def test_btv_activity_success_payload_contract(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_download_and_stitch_lightcurve(**_kwargs: Any) -> tuple[LightCurve, list[int]]:
        return (
            LightCurve(
                time=[1.0, 2.0, 3.0],
                flux=[1.0, 0.999, 1.001],
                flux_err=[0.001, 0.001, 0.001],
            ),
            [14, 15],
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

    monkeypatch.setattr("bittr_tess_vetter.cli.activity_cli.MASTClient", _FakeMASTClient)

    runner = CliRunner()
    result = runner.invoke(
        activity_command,
        [
            "--tic-id",
            "123",
        ],
    )
    assert result.exit_code == 4
