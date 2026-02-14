from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

import bittr_tess_vetter.cli.model_compete_cli as model_compete_cli
from bittr_tess_vetter.cli.model_compete_cli import model_compete_command
from bittr_tess_vetter.domain.lightcurve import LightCurveData


def _make_lc(
    *,
    tic_id: int,
    sector: int,
    time: list[float],
    flux: list[float],
    flux_err: list[float],
    quality: list[int],
    valid_mask: list[bool],
) -> LightCurveData:
    return LightCurveData(
        time=np.asarray(time, dtype=np.float64),
        flux=np.asarray(flux, dtype=np.float64),
        flux_err=np.asarray(flux_err, dtype=np.float64),
        quality=np.asarray(quality, dtype=np.int32),
        valid_mask=np.asarray(valid_mask, dtype=np.bool_),
        tic_id=int(tic_id),
        sector=int(sector),
        cadence_seconds=120.0,
    )


def test_btv_model_compete_success_payload_contract(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**_kwargs: Any) -> tuple[int, float, float, float, float | None, dict[str, Any]]:
        return 123, 7.25, 2450.1, 3.5, None, {
            "source": "toi_catalog",
            "resolved_from": "exofop_toi_table",
            "inputs": {
                "tic_id": 123,
                "period_days": 7.25,
                "t0_btjd": 2450.1,
                "duration_hours": 3.5,
            },
        }

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None = None):
            seen["download"] = {"tic_id": tic_id, "flux_type": flux_type, "sectors": sectors}
            return [
                _make_lc(
                    tic_id=tic_id,
                    sector=14,
                    time=[100.0, 101.0],
                    flux=[1.0, 0.999],
                    flux_err=[0.001, 0.001],
                    quality=[0, 0],
                    valid_mask=[True, True],
                ),
                _make_lc(
                    tic_id=tic_id,
                    sector=15,
                    time=[110.0, 111.0],
                    flux=[1.001, 1.0],
                    flux_err=[0.001, 0.001],
                    quality=[0, 0],
                    valid_mask=[True, True],
                ),
            ]

    def _fake_stitch(lightcurves: list[LightCurveData], *, tic_id: int):
        seen["stitch_called"] = True
        seen["stitch_tic_id"] = tic_id
        return (
            _make_lc(
                tic_id=tic_id,
                sector=-1,
                time=[2000.0, 2000.5, float("nan"), 2001.5],
                flux=[1.0, 0.999, 1.002, float("nan")],
                flux_err=[0.001, 0.001, 0.001, 0.001],
                quality=[0, 0, 0, 1],
                valid_mask=[True, True, True, True],
            ),
            object(),
        )

    def _fake_run_model_competition(**kwargs: Any) -> Any:
        seen["model_competition_kwargs"] = kwargs

        class _Result:
            def to_dict(self) -> dict[str, Any]:
                return {
                    "winner": "transit_only",
                    "winner_margin": 12.3,
                    "model_competition_label": "TRANSIT",
                    "artifact_risk": 0.0,
                    "warnings": [],
                }

        return _Result()

    def _fake_compute_artifact_prior(**kwargs: Any) -> Any:
        seen["artifact_prior_kwargs"] = kwargs

        class _Prior:
            def to_dict(self) -> dict[str, Any]:
                return {
                    "period_alias_risk": 0.1,
                    "sector_quality_risk": 0.0,
                    "scattered_light_risk": 0.0,
                    "combined_risk": 0.05,
                }

        return _Prior()

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.model_compete_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.model_compete_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.model_compete_cli.stitch_lightcurve_data", _fake_stitch)
    monkeypatch.setattr(model_compete_cli.model_competition_api, "run_model_competition", _fake_run_model_competition)
    monkeypatch.setattr(model_compete_cli.model_competition_api, "compute_artifact_prior", _fake_compute_artifact_prior)

    out_path = tmp_path / "model_compete.json"
    runner = CliRunner()
    result = runner.invoke(
        model_compete_command,
        [
            "--toi",
            "123.01",
            "--network-ok",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--flux-type",
            "sap",
            "--bic-threshold",
            "8.5",
            "--n-harmonics",
            "3",
            "--alias-tolerance",
            "0.02",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["download"] == {"tic_id": 123, "flux_type": "sap", "sectors": [14, 15]}
    assert seen["stitch_called"] is True
    assert seen["stitch_tic_id"] == 123

    mc_kwargs = seen["model_competition_kwargs"]
    np.testing.assert_allclose(mc_kwargs["time"], np.array([2000.0, 2000.5], dtype=np.float64))
    np.testing.assert_allclose(mc_kwargs["flux"], np.array([1.0, 0.999], dtype=np.float64))
    np.testing.assert_allclose(mc_kwargs["flux_err"], np.array([0.001, 0.001], dtype=np.float64))
    assert mc_kwargs["period"] == 7.25
    assert mc_kwargs["t0"] == 2450.1
    assert mc_kwargs["duration_hours"] == 3.5
    assert mc_kwargs["bic_threshold"] == 8.5
    assert mc_kwargs["n_harmonics"] == 3

    prior_kwargs = seen["artifact_prior_kwargs"]
    assert prior_kwargs["period"] == 7.25
    assert prior_kwargs["alias_tolerance"] == 0.02

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.model_compete.v1"
    assert payload["result"]["model_competition"]["model_competition_label"] == "TRANSIT"
    assert payload["result"]["artifact_prior"]["combined_risk"] == 0.05
    assert payload["inputs_summary"]["input_resolution"]["inputs"]["tic_id"] == 123
    assert payload["provenance"]["sectors_used"] == [14, 15]
    assert payload["provenance"]["options"] == {
        "network_ok": True,
        "sectors": [14, 15],
        "flux_type": "sap",
        "bic_threshold": 8.5,
        "n_harmonics": 3,
        "alias_tolerance": 0.02,
    }


def test_btv_model_compete_missing_required_ephemeris_input_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        model_compete_command,
        [
            "--tic-id",
            "123",
        ],
    )

    assert result.exit_code == 1
    assert "Missing required inputs" in result.output


def test_btv_model_compete_no_valid_cadences_exits_4(monkeypatch) -> None:
    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None = None):
            _ = tic_id, flux_type, sectors
            return [
                _make_lc(
                    tic_id=123,
                    sector=14,
                    time=[1.0, float("nan")],
                    flux=[1.0, 1.0],
                    flux_err=[0.001, 0.001],
                    quality=[1, 1],
                    valid_mask=[False, False],
                )
            ]

    monkeypatch.setattr("bittr_tess_vetter.cli.model_compete_cli.MASTClient", _FakeMASTClient)

    runner = CliRunner()
    result = runner.invoke(
        model_compete_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
        ],
    )
    assert result.exit_code == 4
