from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

from bittr_tess_vetter.cli.fetch_cli import fetch_command
from bittr_tess_vetter.domain.lightcurve import LightCurveData, make_data_ref


def _fake_lc(*, tic_id: int, sector: int) -> LightCurveData:
    time = np.linspace(2000.0, 2001.0, 8, dtype=np.float64)
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


def test_btv_fetch_contract_tic_id_cache_only_explicit_sectors(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {"downloads": [], "puts": []}

    class _FakeMASTClient:
        def __init__(self, cache_dir: str | None = None) -> None:
            seen["client_cache_dir"] = cache_dir

        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str) -> LightCurveData:
            seen["downloads"].append((tic_id, sector, flux_type))
            return _fake_lc(tic_id=tic_id, sector=sector)

    class _FakePersistentCache:
        def __init__(self, cache_dir: Path | None = None) -> None:
            seen["persistent_cache_dir"] = cache_dir

        def put(self, key: str, value: Any) -> None:
            seen["puts"].append((key, int(getattr(value, "sector", -1))))

    monkeypatch.setattr("bittr_tess_vetter.cli.fetch_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.fetch_cli.PersistentCache", _FakePersistentCache)

    out_path = tmp_path / "fetch.json"
    cache_dir = tmp_path / "cache"
    runner = CliRunner()
    result = runner.invoke(
        fetch_command,
        [
            "--tic-id",
            "123",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--no-network",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["client_cache_dir"] == str(cache_dir)
    assert seen["persistent_cache_dir"] == cache_dir
    assert seen["downloads"] == [(123, 14, "pdcsap"), (123, 15, "pdcsap")]
    expected_keys = [make_data_ref(123, 14, "pdcsap"), make_data_ref(123, 15, "pdcsap")]
    assert [entry[0] for entry in seen["puts"]] == expected_keys

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.fetch.v1"
    assert payload["cached_keys"] == expected_keys
    assert payload["sectors"] == [14, 15]
    assert payload["result"]["cached_keys"] == expected_keys
    assert payload["result"]["sectors"] == [14, 15]
    assert payload["inputs_summary"]["tic_id"] == 123
    assert payload["inputs_summary"]["input_resolution"]["source"] == "cli"
    assert payload["provenance"]["sector_load_path"] == "cache_only_explicit_sectors"
    assert payload["provenance"]["options"]["flux_type"] == "pdcsap"


def test_btv_fetch_report_file_inputs_override_toi(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "fetch.report.json"
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
                    },
                    "provenance": {"sectors_used": [13, 14]},
                }
            }
        ),
        encoding="utf-8",
    )

    seen: dict[str, Any] = {"downloaded": [], "puts": []}

    class _FakeMASTClient:
        def __init__(self, cache_dir: str | None = None) -> None:
            _ = cache_dir

        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str) -> LightCurveData:
            seen["downloaded"].append((tic_id, sector, flux_type))
            return _fake_lc(tic_id=tic_id, sector=sector)

    @dataclass
    class _FakePersistentCache:
        cache_dir: Path | None = None

        def put(self, key: str, value: Any) -> None:
            seen["puts"].append((key, int(getattr(value, "sector", -1))))

    monkeypatch.setattr("bittr_tess_vetter.cli.fetch_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.fetch_cli.PersistentCache", _FakePersistentCache)

    out_path = tmp_path / "fetch.report.out.json"
    runner = CliRunner()
    result = runner.invoke(
        fetch_command,
        [
            "--report-file",
            str(report_path),
            "--toi",
            "TOI-555.01",
            "--no-network",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Warning: --report-file provided; ignoring --toi" in result.output
    assert seen["downloaded"] == [(555, 13, "pdcsap"), (555, 14, "pdcsap")]
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["inputs_summary"]["input_resolution"]["source"] == "report_file"
    assert payload["provenance"]["inputs_source"] == "report_file"
    assert payload["provenance"]["report_file"] == str(report_path.resolve())
    assert payload["provenance"]["sector_selection_source"] == "report_file"
    assert payload["provenance"]["sector_load_path"] == "cache_only_filtered"
    assert payload["sectors"] == [13, 14]


def test_btv_fetch_requires_one_target_input() -> None:
    runner = CliRunner()
    result = runner.invoke(fetch_command, [])
    assert result.exit_code == 1
    assert "Missing target input" in result.output


def test_btv_fetch_toi_without_network_ok_exits_4() -> None:
    runner = CliRunner()
    result = runner.invoke(fetch_command, ["--toi", "123.01"])
    assert result.exit_code == 4
    assert "--toi requires --network-ok" in result.output
