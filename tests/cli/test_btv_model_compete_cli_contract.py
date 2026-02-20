from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

import tess_vetter.cli.model_compete_cli as model_compete_cli
from tess_vetter.cli.model_compete_cli import model_compete_command
from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.api.types import LightCurve


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
        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str):
            seen.setdefault("cache_calls", []).append(
                {"tic_id": tic_id, "sector": sector, "flux_type": flux_type}
            )
            return _make_lc(
                tic_id=tic_id,
                sector=sector,
                time=[100.0, 101.0] if sector == 14 else [110.0, 111.0],
                flux=[1.0, 0.999] if sector == 14 else [1.001, 1.0],
                flux_err=[0.001, 0.001],
                quality=[0, 0],
                valid_mask=[True, True],
            )

        def download_all_sectors(self, *_args: Any, **_kwargs: Any):
            raise AssertionError("download_all_sectors should not be called when --sectors is provided")

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
                    "interpretation_label": "TRANSIT",
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
        "tess_vetter.cli.model_compete_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr("tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("tess_vetter.cli.model_compete_cli.stitch_lightcurve_data", _fake_stitch)
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
    assert seen["cache_calls"] == [
        {"tic_id": 123, "sector": 14, "flux_type": "sap"},
        {"tic_id": 123, "sector": 15, "flux_type": "sap"},
    ]
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
    assert payload["result"]["model_competition_label"] == "TRANSIT"
    assert payload["result"]["interpretation_label"] == "TRANSIT"
    assert payload["result"]["model_competition"]["model_competition_label"] == "TRANSIT"
    assert payload["result"]["artifact_prior"]["combined_risk"] == 0.05
    assert payload["verdict"] == "TRANSIT"
    assert payload["verdict_source"] == "$.result.interpretation_label"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["inputs_summary"]["input_resolution"]["inputs"]["tic_id"] == 123
    assert payload["provenance"]["sectors_used"] == [14, 15]
    assert payload["provenance"]["options"] == {
        "network_ok": True,
        "sectors": [14, 15],
        "flux_type": "sap",
        "bic_threshold": 8.5,
        "n_harmonics": 3,
        "alias_tolerance": 0.02,
        "detrend": None,
        "detrend_bin_hours": 6.0,
        "detrend_buffer": 2.0,
        "detrend_sigma_clip": 5.0,
    }
    assert payload["provenance"]["sector_load_path"] == "cache_only_explicit_sectors"


def test_btv_model_compete_report_file_inputs_override_toi(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "candidate.report.json"
    report_path.write_text(
        json.dumps(
            {
                "report": {
                    "summary": {
                        "tic_id": 321,
                        "ephemeris": {
                            "period_days": 8.2,
                            "t0_btjd": 2501.5,
                            "duration_hours": 4.1,
                        },
                        "input_depth_ppm": 750.0,
                    },
                    "provenance": {"sectors_used": [21, 22]},
                }
            }
        ),
        encoding="utf-8",
    )

    def _unexpected_resolve(**_kwargs: Any):
        raise AssertionError("_resolve_candidate_inputs should not be used with --report-file")

    seen: dict[str, Any] = {}

    def _fake_download(**kwargs: Any):
        seen["download"] = kwargs
        return (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1.0, 0.999], dtype=np.float64),
            np.array([0.001, 0.001], dtype=np.float64),
            [21, 22],
            "mast_filtered",
        )

    monkeypatch.setattr(
        "tess_vetter.cli.model_compete_cli._resolve_candidate_inputs",
        _unexpected_resolve,
    )
    monkeypatch.setattr("tess_vetter.cli.model_compete_cli._download_and_prepare_arrays", _fake_download)
    monkeypatch.setattr(
        model_compete_cli.model_competition_api,
        "run_model_competition",
        lambda **_kwargs: type("R", (), {"to_dict": lambda self: {"interpretation_label": "TRANSIT"}})(),
    )
    monkeypatch.setattr(
        model_compete_cli.model_competition_api,
        "compute_artifact_prior",
        lambda **_kwargs: type("P", (), {"to_dict": lambda self: {"combined_risk": 0.0}})(),
    )

    out_path = tmp_path / "model_compete_report_file.json"
    runner = CliRunner()
    result = runner.invoke(
        model_compete_command,
        [
            "--report-file",
            str(report_path),
            "--toi",
            "TOI-123.01",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Warning: --report-file provided; ignoring --toi" in result.output
    assert seen["download"]["tic_id"] == 321
    assert seen["download"]["sectors"] == [21, 22]
    assert seen["download"]["sectors_explicit"] is False

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["inputs_summary"]["input_resolution"]["source"] == "report_file"
    assert payload["provenance"]["inputs_source"] == "report_file"
    assert payload["provenance"]["report_file"] == str(report_path.resolve())


def test_btv_model_compete_explicit_sectors_cache_miss_exits_4(monkeypatch) -> None:
    class _FakeMASTClient:
        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str):
            _ = tic_id, sector, flux_type
            raise RuntimeError("cache miss")

        def download_all_sectors(self, *_args: Any, **_kwargs: Any):
            raise AssertionError("download_all_sectors should not be called when --sectors is provided")

    monkeypatch.setattr("tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

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
            "--sectors",
            "14",
        ],
    )
    assert result.exit_code == 4
    assert "Cache-only sector load failed for TIC 123" in result.output


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

    monkeypatch.setattr("tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

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


def test_btv_model_compete_accepts_positional_toi_and_short_o(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**kwargs: Any):
        seen.update(kwargs)
        return 123, 7.25, 2450.1, 3.5, None, {"source": "toi", "inputs": {"toi": kwargs.get("toi")}}

    monkeypatch.setattr(
        "tess_vetter.cli.model_compete_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "tess_vetter.cli.model_compete_cli._download_and_prepare_arrays",
        lambda **_kwargs: (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1.0, 0.999], dtype=np.float64),
            np.array([0.001, 0.001], dtype=np.float64),
            [14],
            "mast_discovery",
        ),
    )
    monkeypatch.setattr(
        model_compete_cli.model_competition_api,
        "run_model_competition",
        lambda **_kwargs: type("R", (), {"to_dict": lambda self: {"model_competition_label": "TRANSIT"}})(),
    )
    monkeypatch.setattr(
        model_compete_cli.model_competition_api,
        "compute_artifact_prior",
        lambda **_kwargs: type("P", (), {"to_dict": lambda self: {"combined_risk": 0.0}})(),
    )

    out_path = tmp_path / "model_compete_positional.json"
    runner = CliRunner()
    result = runner.invoke(model_compete_command, ["TOI-5807.01", "-o", str(out_path)])
    assert result.exit_code == 0, result.output
    assert seen["toi"] == "TOI-5807.01"


def test_btv_model_compete_rejects_mismatched_positional_and_option_toi() -> None:
    runner = CliRunner()
    result = runner.invoke(
        model_compete_command,
        ["TOI-5807.01", "--toi", "TOI-4510.01"],
    )
    assert result.exit_code == 1
    assert "must match" in result.output


def test_btv_model_compete_detrend_wiring_and_provenance(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    monkeypatch.setattr(
        "tess_vetter.cli.model_compete_cli._resolve_candidate_inputs",
        lambda **_kwargs: (
            123,
            7.25,
            2450.1,
            3.5,
            None,
            {"source": "manual", "inputs": {"tic_id": 123}},
        ),
    )
    monkeypatch.setattr(
        "tess_vetter.cli.model_compete_cli._download_and_prepare_arrays",
        lambda **_kwargs: (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([1.0, 0.999, 1.001], dtype=np.float64),
            np.array([0.001, 0.001, 0.001], dtype=np.float64),
            [14],
            "mast_discovery",
        ),
    )

    def _fake_detrend_lightcurve_for_vetting(**kwargs: Any):
        seen["detrend_kwargs"] = kwargs
        lc = kwargs["lc"]
        detrended_lc = LightCurve(
            time=np.asarray(lc.time, dtype=np.float64),
            flux=np.asarray(lc.flux, dtype=np.float64) * 0.9995,
            flux_err=np.asarray(lc.flux_err, dtype=np.float64),
            quality=lc.quality,
            valid_mask=lc.valid_mask,
        )
        return detrended_lc, {
            "applied": True,
            "method": kwargs["method"],
            "bin_hours": kwargs["bin_hours"],
            "buffer_factor": kwargs["buffer_factor"],
            "sigma_clip": kwargs["clip_sigma"],
        }

    monkeypatch.setattr(
        "tess_vetter.cli.model_compete_cli._detrend_lightcurve_for_vetting",
        _fake_detrend_lightcurve_for_vetting,
    )
    monkeypatch.setattr(
        model_compete_cli.model_competition_api,
        "run_model_competition",
        lambda **_kwargs: type("R", (), {"to_dict": lambda self: {"interpretation_label": "TRANSIT"}})(),
    )
    monkeypatch.setattr(
        model_compete_cli.model_competition_api,
        "compute_artifact_prior",
        lambda **_kwargs: type("P", (), {"to_dict": lambda self: {"combined_risk": 0.1}})(),
    )

    out_path = tmp_path / "model_compete_detrended.json"
    runner = CliRunner()
    result = runner.invoke(
        model_compete_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "7.25",
            "--t0-btjd",
            "2450.1",
            "--duration-hours",
            "3.5",
            "--detrend",
            "transit_masked_bin_median",
            "--detrend-bin-hours",
            "4",
            "--detrend-buffer",
            "1.5",
            "--detrend-sigma-clip",
            "3",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output

    detrend_kwargs = seen["detrend_kwargs"]
    assert detrend_kwargs["method"] == "transit_masked_bin_median"
    assert detrend_kwargs["bin_hours"] == 4.0
    assert detrend_kwargs["buffer_factor"] == 1.5
    assert detrend_kwargs["clip_sigma"] == 3.0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["options"]["detrend"] == "transit_masked_bin_median"
    assert payload["provenance"]["options"]["detrend_bin_hours"] == 4.0
    assert payload["provenance"]["options"]["detrend_buffer"] == 1.5
    assert payload["provenance"]["options"]["detrend_sigma_clip"] == 3.0
    assert payload["provenance"]["detrend"]["method"] == "transit_masked_bin_median"
