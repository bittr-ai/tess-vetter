from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from click.testing import CliRunner

from bittr_tess_vetter.cli.periodogram_cli import periodogram_command


def test_btv_periodogram_search_mode_success_payload(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**_kwargs: Any) -> tuple[int, float, float, float, float | None, dict[str, Any]]:
        return 123, 7.25, 2450.1, 3.5, None, {
            "source": "toi_catalog",
            "resolved_from": "exofop_toi_table",
            "inputs": {"tic_id": 123, "period_days": 7.25, "t0_btjd": 2450.1, "duration_hours": 3.5},
        }

    def _fake_download_and_stitch_lightcurve(**_kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        return (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([1.0, 0.999, 1.001], dtype=np.float64),
            np.array([0.001, 0.001, 0.001], dtype=np.float64),
            [14, 15],
        )

    def _fake_run_periodogram(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(
            model_dump=lambda **_dump_kwargs: {
                "method": "tls",
                "best_period": 7.25,
                "best_t0": 2450.1,
                "peaks": [],
            }
        )

    monkeypatch.setattr("bittr_tess_vetter.cli.periodogram_cli._resolve_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.periodogram_cli._download_and_stitch_lightcurve",
        _fake_download_and_stitch_lightcurve,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.periodogram_cli.run_periodogram", _fake_run_periodogram)

    out_path = tmp_path / "periodogram_search.json"
    runner = CliRunner()
    result = runner.invoke(
        periodogram_command,
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
            "--min-period",
            "1.5",
            "--max-period",
            "12.0",
            "--preset",
            "deep",
            "--method",
            "tls",
            "--max-planets",
            "2",
            "--no-per-sector",
            "--downsample-factor",
            "3",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["tic_id"] == 123
    assert seen["min_period"] == 1.5
    assert seen["max_period"] == 12.0
    assert seen["preset"] == "deep"
    assert seen["method"] == "tls"
    assert seen["max_planets"] == 2
    assert seen["per_sector"] is False
    assert seen["downsample_factor"] == 3

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.periodogram.v1"
    assert payload["mode"] == "search"
    assert payload["result"]["best_period"] == 7.25
    assert payload["inputs_summary"]["sectors_used"] == [14, 15]
    assert payload["provenance"]["sectors_used"] == [14, 15]
    assert payload["provenance"]["options"]["method"] == "tls"
    assert payload["provenance"]["input_resolution"]["inputs"]["period_days"] == 7.25


def test_btv_periodogram_refine_mode_success_payload(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_download_and_stitch_lightcurve(**_kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        return (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([1.0, 0.999, 1.001], dtype=np.float64),
            np.array([0.001, 0.001, 0.001], dtype=np.float64),
            [20],
        )

    def _fake_refine_period(**kwargs: Any) -> tuple[float, float, float]:
        seen.update(kwargs)
        return 5.01, 2100.2, 12.3

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.periodogram_cli._download_and_stitch_lightcurve",
        _fake_download_and_stitch_lightcurve,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.periodogram_cli.refine_period", _fake_refine_period)

    out_path = tmp_path / "periodogram_refine.json"
    runner = CliRunner()
    result = runner.invoke(
        periodogram_command,
        [
            "--tic-id",
            "456",
            "--refine",
            "--initial-period",
            "5.0",
            "--initial-duration",
            "3.0",
            "--refine-factor",
            "0.2",
            "--n-refine",
            "150",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["tic_id"] == 456
    assert seen["initial_period"] == 5.0
    assert seen["initial_duration"] == 3.0
    assert seen["refine_factor"] == 0.2
    assert seen["n_refine"] == 150

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.periodogram.v1"
    assert payload["mode"] == "refine"
    assert payload["result"] == {
        "refined_period_days": 5.01,
        "refined_t0_btjd": 2100.2,
        "refined_power": 12.3,
    }
    assert payload["provenance"]["sectors_used"] == [20]


def test_btv_periodogram_refine_missing_required_args_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        periodogram_command,
        [
            "--tic-id",
            "789",
            "--refine",
        ],
    )
    assert result.exit_code == 1
    assert "--refine requires --initial-period and --initial-duration" in result.output


def test_btv_periodogram_missing_tic_and_toi_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(periodogram_command, [])
    assert result.exit_code == 1
    assert "Missing TIC identifier" in result.output


def test_btv_periodogram_use_stellar_auto_requires_network_exits_4() -> None:
    runner = CliRunner()
    result = runner.invoke(
        periodogram_command,
        [
            "--tic-id",
            "789",
            "--use-stellar-auto",
        ],
    )
    assert result.exit_code == 4
