from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

from bittr_tess_vetter.cli.fetch_cli import cache_sectors_command, fetch_command
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


def test_btv_cache_sectors_probe_no_network(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    class _Row:
        def __init__(self, sector: int) -> None:
            self.sector = sector

    class _FakeMASTClient:
        def __init__(self, cache_dir: str | None = None) -> None:
            seen["cache_dir"] = cache_dir

        def search_lightcurve_cached(self, tic_id: int):
            seen["search_tic"] = tic_id
            return [_Row(14), _Row(15)]

    class _FakePersistentCache:
        def __init__(self, cache_dir: Path | None = None) -> None:
            seen["persistent_cache_dir"] = cache_dir

        def put(self, key: str, value: Any) -> None:
            raise AssertionError("probe should not write cache")

    monkeypatch.setattr("bittr_tess_vetter.cli.fetch_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.fetch_cli.PersistentCache", _FakePersistentCache)

    out_path = tmp_path / "cache_probe.json"
    result = CliRunner().invoke(
        cache_sectors_command,
        [
            "--tic-id",
            "123",
            "--no-network",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.cache_sectors.v1"
    assert payload["cached_sectors_before"] == [14, 15]
    assert payload["cached_sectors_after"] == [14, 15]
    assert payload["action_hint"] == "CACHE_READY"
    assert seen["search_tic"] == 123


def test_btv_cache_sectors_fill_missing_backfills(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {"puts": []}

    class _Row:
        def __init__(self, sector: int) -> None:
            self.sector = sector

    class _FakeMASTClient:
        def __init__(self, cache_dir: str | None = None) -> None:
            _ = cache_dir
            seen["search_calls"] = 0

        def search_lightcurve_cached(self, tic_id: int):
            seen["search_calls"] += 1
            if seen["search_calls"] == 1:
                return [_Row(14)]
            return [_Row(14), _Row(15)]

        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None):
            seen["download"] = (tic_id, flux_type, sectors)
            return [_fake_lc(tic_id=tic_id, sector=15)]

    class _FakePersistentCache:
        def __init__(self, cache_dir: Path | None = None) -> None:
            _ = cache_dir

        def put(self, key: str, value: Any) -> None:
            seen["puts"].append((key, int(getattr(value, "sector", -1))))

    monkeypatch.setattr("bittr_tess_vetter.cli.fetch_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.fetch_cli.PersistentCache", _FakePersistentCache)

    out_path = tmp_path / "cache_fill.json"
    result = CliRunner().invoke(
        cache_sectors_command,
        [
            "--tic-id",
            "123",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--fill-missing",
            "--network-ok",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["missing_requested_before"] == [15]
    assert payload["filled_sectors"] == [15]
    assert payload["missing_requested_after"] == []
    assert payload["action_hint"] == "CACHE_BACKFILL_COMPLETED"
    assert seen["download"] == (123, "pdcsap", [15])
    assert seen["puts"][0][0] == make_data_ref(123, 15, "pdcsap")


def test_btv_cache_sectors_fill_missing_requires_network_and_sectors() -> None:
    runner = CliRunner()
    no_sectors = runner.invoke(cache_sectors_command, ["--tic-id", "123", "--fill-missing"])
    assert no_sectors.exit_code == 1
    assert "--fill-missing requires at least one --sectors value" in no_sectors.output

    no_network = runner.invoke(
        cache_sectors_command,
        ["--tic-id", "123", "--sectors", "14", "--fill-missing", "--no-network"],
    )
    assert no_network.exit_code == 4
    assert "--fill-missing requires --network-ok" in no_network.output


def test_btv_cache_sectors_help_mentions_backfill_flow() -> None:
    result = CliRunner().invoke(cache_sectors_command, ["--help"])
    assert result.exit_code == 0
    assert "--fill-missing" in result.output
    assert "backfill" in result.output.lower()
