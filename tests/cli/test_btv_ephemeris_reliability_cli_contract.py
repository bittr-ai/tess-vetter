from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

from bittr_tess_vetter.cli.common_cli import BtvCliError
from bittr_tess_vetter.cli.ephemeris_reliability_cli import ephemeris_reliability_command
from bittr_tess_vetter.domain.lightcurve import LightCurveData


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


def test_btv_ephemeris_reliability_success_contract_payload(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 7.5, 2500.25, 3.0, 900.0, {"source": "cli", "inputs": {"tic_id": 123}}

    def _fake_load_lightcurves_with_sector_policy(**kwargs: Any):
        seen["download"] = kwargs
        return [
            _make_lc(tic_id=kwargs["tic_id"], sector=14, start=2000.0),
            _make_lc(tic_id=kwargs["tic_id"], sector=15, start=2010.0),
        ], "cache_only_explicit_sectors"

    def _fake_stitch(lightcurves: list[LightCurveData], *, tic_id: int):
        seen["stitch_called"] = True
        seen["stitch_tic_id"] = tic_id
        _ = lightcurves
        return _make_lc(tic_id=tic_id, sector=-1, start=2000.0), object()

    class _FakeResult:
        def to_dict(self) -> dict[str, Any]:
            return {
                "label": "ok",
                "null_percentile": 0.999,
                "schedulability_summary": {
                    "scalar": 0.73,
                    "components": {
                        "signal_vs_phase_null": 0.99,
                        "period_localization": 0.72,
                    },
                    "provenance": {"kind": "ephemeris_schedulability_scalar", "version": "v1"},
                },
            }

    def _fake_compute_reliability_regime_numpy(**kwargs: Any):
        seen["compute"] = kwargs
        return _FakeResult()

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.ephemeris_reliability_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.ephemeris_reliability_cli.load_lightcurves_with_sector_policy",
        _fake_load_lightcurves_with_sector_policy,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.ephemeris_reliability_cli.stitch_lightcurve_data",
        _fake_stitch,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.api.ephemeris_reliability.compute_reliability_regime_numpy",
        _fake_compute_reliability_regime_numpy,
    )

    out_path = tmp_path / "ephemeris_reliability.json"
    runner = CliRunner()
    result = runner.invoke(
        ephemeris_reliability_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--network-ok",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--flux-type",
            "sap",
            "--ingress-egress-fraction",
            "0.15",
            "--sharpness",
            "25.0",
            "--n-phase-shifts",
            "120",
            "--phase-shift-strategy",
            "random",
            "--random-seed",
            "13",
            "--period-jitter-frac",
            "0.01",
            "--period-jitter-n",
            "17",
            "--t0-scan-n",
            "61",
            "--t0-scan-half-span-minutes",
            "45.0",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["download"] == {
        "tic_id": 123,
        "flux_type": "sap",
        "sectors": [14, 15],
        "explicit_sectors": True,
        "network_ok": True,
        "cache_dir": None,
    }
    assert seen["stitch_called"] is True
    assert seen["stitch_tic_id"] == 123
    assert seen["compute"]["period_days"] == 7.5
    assert seen["compute"]["t0_btjd"] == 2500.25
    assert seen["compute"]["duration_hours"] == 3.0
    assert seen["compute"]["n_phase_shifts"] == 120
    assert seen["compute"]["phase_shift_strategy"] == "random"
    assert seen["compute"]["random_seed"] == 13
    assert seen["compute"]["period_jitter_frac"] == 0.01
    assert seen["compute"]["period_jitter_n"] == 17
    assert seen["compute"]["t0_scan_n"] == 61
    assert seen["compute"]["t0_scan_half_span_minutes"] == 45.0
    assert seen["compute"]["config"].ingress_egress_fraction == 0.15
    assert seen["compute"]["config"].sharpness == 25.0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.ephemeris_reliability.v1"
    assert payload["result"]["label"] == "ok"
    assert "schedulability_summary" in payload["result"]
    assert isinstance(payload["result"]["schedulability_summary"]["scalar"], float)
    assert isinstance(payload["result"]["schedulability_summary"]["components"], dict)
    assert isinstance(payload["result"]["schedulability_summary"]["provenance"], dict)
    assert payload["schedulability_scalar"] == 0.73
    assert payload["result"]["schedulability_scalar"] == 0.73
    assert "verdict" in payload
    assert "verdict_source" in payload
    assert payload["verdict"] == "ok"
    assert payload["verdict_source"] == "$.result.label"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["inputs_summary"]["input_resolution"]["inputs"]["tic_id"] == 123
    assert payload["provenance"]["sectors_used"] == [14, 15]
    assert payload["provenance"]["options"]["flux_type"] == "sap"
    assert payload["provenance"]["options"]["phase_shift_strategy"] == "random"
    assert payload["provenance"]["sector_load_path"] == "cache_only_explicit_sectors"


def test_btv_ephemeris_reliability_missing_required_ephemeris_input_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        ephemeris_reliability_command,
        [
            "--tic-id",
            "123",
            "--t0-btjd",
            "2500.25",
        ],
    )

    assert result.exit_code == 1
    assert "Missing required inputs" in result.output


def test_btv_ephemeris_reliability_filters_invalid_cadences(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 7.5, 2500.25, 3.0, 900.0, {"source": "cli"}

    def _fake_load_lightcurves_with_sector_policy(**_kwargs: Any):
        time = np.array([1.0, 2.0, np.nan, 4.0], dtype=np.float64)
        flux = np.array([1.0, 0.999, 1.001, 1.0], dtype=np.float64)
        flux_err = np.array([0.001, 0.001, 0.001, np.nan], dtype=np.float64)
        quality = np.array([0, 0, 0, 1], dtype=np.int32)
        valid_mask = np.array([True, True, True, True], dtype=np.bool_)
        return [
            LightCurveData(
                time=time,
                flux=flux,
                flux_err=flux_err,
                quality=quality,
                valid_mask=valid_mask,
                tic_id=123,
                sector=14,
                cadence_seconds=120.0,
            )
        ], "mast_discovery"

    class _FakeResult:
        def to_dict(self) -> dict[str, Any]:
            return {"label": "ok"}

    def _fake_compute_reliability_regime_numpy(**kwargs: Any):
        seen["time"] = np.asarray(kwargs["time"], dtype=np.float64)
        seen["flux"] = np.asarray(kwargs["flux"], dtype=np.float64)
        seen["flux_err"] = np.asarray(kwargs["flux_err"], dtype=np.float64)
        return _FakeResult()

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.ephemeris_reliability_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.ephemeris_reliability_cli.load_lightcurves_with_sector_policy",
        _fake_load_lightcurves_with_sector_policy,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.api.ephemeris_reliability.compute_reliability_regime_numpy",
        _fake_compute_reliability_regime_numpy,
    )

    out_path = tmp_path / "ephem_masked.json"
    runner = CliRunner()
    result = runner.invoke(
        ephemeris_reliability_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(seen["time"], np.array([1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(seen["flux"], np.array([1.0, 0.999], dtype=np.float64))
    np.testing.assert_allclose(seen["flux_err"], np.array([0.001, 0.001], dtype=np.float64))


def test_btv_ephemeris_reliability_accepts_positional_toi_and_short_o(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**kwargs: Any):
        seen.update(kwargs)
        return 123, 7.5, 2500.25, 3.0, 900.0, {"source": "toi", "inputs": {"toi": kwargs.get("toi")}}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.ephemeris_reliability_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.ephemeris_reliability_cli.load_lightcurves_with_sector_policy",
        lambda **kwargs: ([_make_lc(tic_id=kwargs["tic_id"], sector=14, start=2000.0)], "mast_discovery"),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.api.ephemeris_reliability.compute_reliability_regime_numpy",
        lambda **_kwargs: type("R", (), {"to_dict": lambda self: {"label": "ok"}})(),
    )

    out_path = tmp_path / "ephem_positional.json"
    runner = CliRunner()
    result = runner.invoke(ephemeris_reliability_command, ["TOI-5807.01", "-o", str(out_path)])
    assert result.exit_code == 0, result.output
    assert seen["toi"] == "TOI-5807.01"


def test_btv_ephemeris_reliability_rejects_mismatched_positional_and_option_toi() -> None:
    runner = CliRunner()
    result = runner.invoke(
        ephemeris_reliability_command,
        ["TOI-5807.01", "--toi", "TOI-4510.01"],
    )
    assert result.exit_code == 1
    assert "must match" in result.output


def test_btv_ephemeris_reliability_report_file_inputs_override_toi(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "ephem.report.json"
    report_path.write_text(
        json.dumps(
            {
                "report": {
                    "summary": {
                        "tic_id": 222,
                        "ephemeris": {
                            "period_days": 2.5,
                            "t0_btjd": 2200.0,
                            "duration_hours": 1.5,
                        },
                        "input_depth_ppm": 300.0,
                    },
                    "provenance": {"sectors_used": [5]},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.ephemeris_reliability_cli._resolve_candidate_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve TOI with report file")),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.ephemeris_reliability_cli.load_lightcurves_with_sector_policy",
        lambda **kwargs: ([_make_lc(tic_id=kwargs["tic_id"], sector=5, start=2000.0)], "mast_filtered"),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.api.ephemeris_reliability.compute_reliability_regime_numpy",
        lambda **_kwargs: type("R", (), {"to_dict": lambda self: {"label": "ok"}})(),
    )

    out_path = tmp_path / "ephem_report_file.json"
    runner = CliRunner()
    result = runner.invoke(
        ephemeris_reliability_command,
        [
            "--report-file",
            str(report_path),
            "--toi",
            "TOI-222.01",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Warning: --report-file provided; ignoring --toi" in result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["inputs_summary"]["input_resolution"]["source"] == "report_file"
    assert payload["provenance"]["inputs_source"] == "report_file"
    assert payload["provenance"]["report_file"] == str(report_path.resolve())


def test_btv_ephemeris_reliability_explicit_sectors_cache_miss_exits_4(monkeypatch) -> None:
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.ephemeris_reliability_cli.load_lightcurves_with_sector_policy",
        lambda **_kwargs: (_ for _ in ()).throw(
            BtvCliError(
                "Cache-only sector load failed for TIC 123. Missing cached light curve for sector(s): 14.",
                exit_code=4,
            )
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        ephemeris_reliability_command,
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
