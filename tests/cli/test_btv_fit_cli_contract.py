from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

from tess_vetter.cli.transit_fit_cli import fit_command
from tess_vetter.domain.lightcurve import LightCurveData


def _make_lc(*, tic_id: int, sector: int, start: float) -> LightCurveData:
    time = np.linspace(start, start + 1.0, 32, dtype=np.float64)
    flux = np.ones_like(time, dtype=np.float64)
    flux_err = np.full_like(time, 1e-3, dtype=np.float64)
    quality = np.zeros(time.shape, dtype=np.int32)
    valid_mask = np.ones(time.shape, dtype=np.bool_)
    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=int(tic_id),
        sector=int(sector),
        cadence_seconds=120.0,
    )


def test_btv_fit_success_contract_payload(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None = None):
            seen["download"] = {
                "tic_id": tic_id,
                "flux_type": flux_type,
                "sectors": sectors,
            }
            return [
                _make_lc(tic_id=tic_id, sector=14, start=2000.0),
                _make_lc(tic_id=tic_id, sector=15, start=2010.0),
            ]

    class _FakeFitResult:
        status = "success"

        def to_dict(self) -> dict[str, Any]:
            return {
                "status": "success",
                "fit_method": "optimize",
                "error_message": None,
            }

    def _fake_stitch(lightcurves: list[LightCurveData], *, tic_id: int):
        seen["stitch_called"] = True
        seen["stitch_tic_id"] = tic_id
        return _make_lc(tic_id=tic_id, sector=-1, start=2000.0), object()

    def _fake_fit_transit(**kwargs: Any):
        seen["fit_kwargs"] = kwargs
        return _FakeFitResult()

    monkeypatch.setattr("tess_vetter.cli.transit_fit_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("tess_vetter.cli.transit_fit_cli.stitch_lightcurve_data", _fake_stitch)
    monkeypatch.setattr("tess_vetter.cli.transit_fit_cli.api.transit_fit.fit_transit", _fake_fit_transit)

    out_path = tmp_path / "fit.json"
    runner = CliRunner()
    result = runner.invoke(
        fit_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.25",
            "--duration-hours",
            "2.5",
            "--depth-ppm",
            "700",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--flux-type",
            "sap",
            "--method",
            "optimize",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["download"] == {"tic_id": 123, "flux_type": "sap", "sectors": [14, 15]}
    assert seen["stitch_called"] is True
    assert seen["stitch_tic_id"] == 123
    assert seen["fit_kwargs"]["method"] == "optimize"

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.fit.v1"
    assert payload["fit"]["status"] == "success"
    assert payload["inputs_summary"]["input_resolution"]["inputs"]["tic_id"] == 123
    assert "stellar_resolution" in payload["inputs_summary"]
    assert payload["provenance"]["sectors_used"] == [14, 15]
    assert payload["provenance"]["method"] == "optimize"


def test_btv_fit_api_error_status_exits_zero_and_surfaces_error(monkeypatch, tmp_path: Path) -> None:
    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None = None):
            return [_make_lc(tic_id=tic_id, sector=14, start=2000.0)]

    class _FakeFitResult:
        status = "error"

        def to_dict(self) -> dict[str, Any]:
            return {
                "status": "error",
                "fit_method": "none",
                "error_message": (
                    "batman not installed - required for transit fitting. "
                    "Install with: pip install batman-package"
                ),
            }

    monkeypatch.setattr("tess_vetter.cli.transit_fit_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "tess_vetter.cli.transit_fit_cli.api.transit_fit.fit_transit",
        lambda **_kwargs: _FakeFitResult(),
    )

    out_path = tmp_path / "fit_error.json"
    runner = CliRunner()
    result = runner.invoke(
        fit_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.25",
            "--duration-hours",
            "2.5",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["fit"]["status"] == "error"
    assert payload["fit"]["error_message"] == (
        "batman not installed - required for transit fitting. "
        "Install with: pip install batman-package"
    )


def test_btv_fit_missing_required_candidate_input_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        fit_command,
        [
            "--tic-id",
            "123",
            "--duration-hours",
            "2.5",
        ],
    )

    assert result.exit_code == 1
    assert "Missing required inputs" in result.output


def test_btv_fit_use_stellar_auto_requires_network_exits_4() -> None:
    runner = CliRunner()
    result = runner.invoke(
        fit_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.25",
            "--duration-hours",
            "2.5",
            "--use-stellar-auto",
        ],
    )
    assert result.exit_code == 4
