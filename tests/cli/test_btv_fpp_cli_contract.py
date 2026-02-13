from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

import bittr_tess_vetter.cli.enrich_cli as enrich_cli


def test_btv_help_lists_fpp() -> None:
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "fpp" in result.output


def test_btv_fpp_success_plumbs_api_params_and_emits_contract(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14, 15]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {
            "fpp": 0.123,
            "nfpp": 0.001,
            "disposition": "possible_planet",
            "base_seed": 99,
        }

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _fake_build_cache_for_fpp,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.fpp_cli.calculate_fpp",
        _fake_calculate_fpp,
    )

    out_path = tmp_path / "fpp.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--preset",
            "standard",
            "--replicates",
            "4",
            "--seed",
            "99",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--timeout-seconds",
            "120",
            "--network-ok",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["tic_id"] == 123
    assert seen["period"] == 7.5
    assert seen["t0"] == 2500.25
    assert seen["duration_hours"] == 3.0
    assert seen["depth_ppm"] == 900.0
    assert seen["preset"] == "standard"
    assert seen["replicates"] == 4
    assert seen["seed"] == 99
    assert seen["sectors"] == [14, 15]
    assert seen["timeout_seconds"] == 120.0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.fpp.v1"
    assert "fpp_result" in payload
    assert "provenance" in payload
    assert "inputs" in payload["provenance"]
    assert payload["provenance"]["resolved_source"] == "cli"
    assert payload["provenance"]["runtime"]["preset"] == "standard"
    assert payload["provenance"]["runtime"]["seed"] == 99
    assert payload["provenance"]["runtime"]["seed_requested"] == 99


def test_btv_fpp_timeout_maps_to_exit_5(monkeypatch) -> None:
    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [1]

    def _timeout(**_kwargs: Any) -> dict[str, Any]:
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _fake_build_cache_for_fpp,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _timeout)

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
        ],
    )

    assert result.exit_code == 5


def test_build_cache_for_fpp_stores_requested_sector_products(monkeypatch, tmp_path: Path) -> None:
    class _FakeLC:
        def __init__(self, sector: int) -> None:
            self.sector = sector

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, *, flux_type: str, sectors: list[int] | None = None):
            assert tic_id == 123
            assert flux_type == "pdcsap"
            assert sectors == [14, 15]
            return [_FakeLC(14), _FakeLC(15)]

    class _FakePersistentCache:
        def __init__(self, cache_dir: Path | None = None) -> None:
            self.cache_dir = cache_dir
            self.records: dict[str, object] = {}

        def put(self, key: str, value: object) -> None:
            self.records[key] = value

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.PersistentCache", _FakePersistentCache)

    from bittr_tess_vetter.cli.fpp_cli import _build_cache_for_fpp

    cache, loaded = _build_cache_for_fpp(tic_id=123, sectors=[14, 15], cache_dir=tmp_path)
    assert loaded == [14, 15]
    assert isinstance(cache, _FakePersistentCache)
    assert sorted(cache.records.keys()) == [
        "lc:123:14:pdcsap",
        "lc:123:15:pdcsap",
    ]
