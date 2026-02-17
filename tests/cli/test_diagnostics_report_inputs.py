from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bittr_tess_vetter.cli.diagnostics_report_inputs import load_lightcurves_with_sector_policy


def _lc(sector: int) -> Any:
    return SimpleNamespace(sector=int(sector))


def test_load_lightcurves_with_sector_policy_cache_first_all_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {"download_all_called": False}

    class _FakeMASTClient:
        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str) -> Any:
            _ = tic_id, flux_type
            return _lc(sector)

        def download_all_sectors(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            seen["download_all_called"] = True
            return []

    monkeypatch.setattr("bittr_tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

    lightcurves, path = load_lightcurves_with_sector_policy(
        tic_id=123,
        sectors=[21, 22],
        flux_type="pdcsap",
        explicit_sectors=False,
    )

    assert path == "cache_first_filtered"
    assert [int(lc.sector) for lc in lightcurves] == [21, 22]
    assert seen["download_all_called"] is False


def test_load_lightcurves_with_sector_policy_cache_first_partial_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _FakeMASTClient:
        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str) -> Any:
            _ = tic_id, flux_type
            if int(sector) == 21:
                return _lc(21)
            raise RuntimeError("cache miss")

        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None = None) -> list[Any]:
            seen["download_all"] = {"tic_id": tic_id, "flux_type": flux_type, "sectors": sectors}
            return [_lc(22)]

    monkeypatch.setattr("bittr_tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

    lightcurves, path = load_lightcurves_with_sector_policy(
        tic_id=123,
        sectors=[21, 22],
        flux_type="pdcsap",
        explicit_sectors=False,
    )

    assert path == "cache_then_mast_filtered"
    assert [int(lc.sector) for lc in lightcurves] == [21, 22]
    assert seen["download_all"] == {"tic_id": 123, "flux_type": "pdcsap", "sectors": [22]}


def test_load_lightcurves_with_sector_policy_cache_discovery_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {"download_all_called": False}

    class _FakeMASTClient:
        def search_lightcurve_cached(self, tic_id: int) -> list[Any]:
            _ = tic_id
            return [SimpleNamespace(sector=14), SimpleNamespace(sector=15)]

        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str) -> Any:
            _ = tic_id, flux_type
            return _lc(sector)

        def download_all_sectors(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            seen["download_all_called"] = True
            return []

    monkeypatch.setattr("bittr_tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

    lightcurves, path = load_lightcurves_with_sector_policy(
        tic_id=123,
        sectors=None,
        flux_type="pdcsap",
        explicit_sectors=False,
    )

    assert path == "cache_discovery"
    assert [int(lc.sector) for lc in lightcurves] == [14, 15]
    assert seen["download_all_called"] is False


def test_load_lightcurves_with_sector_policy_cache_discovery_falls_back_to_mast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class _FakeMASTClient:
        def search_lightcurve_cached(self, tic_id: int) -> list[Any]:
            _ = tic_id
            return [SimpleNamespace(sector=14), SimpleNamespace(sector=15)]

        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str) -> Any:
            _ = tic_id, sector, flux_type
            raise RuntimeError("cache parse failed")

        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None = None) -> list[Any]:
            seen["download_all"] = {"tic_id": tic_id, "flux_type": flux_type, "sectors": sectors}
            return [_lc(21)]

    monkeypatch.setattr("bittr_tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

    lightcurves, path = load_lightcurves_with_sector_policy(
        tic_id=123,
        sectors=None,
        flux_type="pdcsap",
        explicit_sectors=False,
    )

    assert path == "mast_discovery"
    assert [int(lc.sector) for lc in lightcurves] == [21]
    assert seen["download_all"] == {"tic_id": 123, "flux_type": "pdcsap", "sectors": None}


def test_load_lightcurves_with_sector_policy_no_network_fails_on_partial_cache_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMASTClient:
        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str) -> Any:
            _ = tic_id, flux_type
            if int(sector) == 21:
                return _lc(21)
            raise RuntimeError("cache miss")

    monkeypatch.setattr("bittr_tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

    with pytest.raises(Exception) as excinfo:
        load_lightcurves_with_sector_policy(
            tic_id=123,
            sectors=[21, 22],
            flux_type="pdcsap",
            explicit_sectors=False,
            network_ok=False,
        )
    assert "Cache-only load failed for TIC 123 with --no-network" in str(excinfo.value)


def test_load_lightcurves_with_sector_policy_no_network_fails_without_cached_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMASTClient:
        def search_lightcurve_cached(self, tic_id: int) -> list[Any]:
            _ = tic_id
            return []

    monkeypatch.setattr("bittr_tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

    with pytest.raises(Exception) as excinfo:
        load_lightcurves_with_sector_policy(
            tic_id=123,
            sectors=None,
            flux_type="pdcsap",
            explicit_sectors=False,
            network_ok=False,
        )
    assert "No cached sectors available for TIC 123 with --no-network" in str(excinfo.value)


def test_load_lightcurves_with_sector_policy_passes_cache_dir_to_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, Any] = {}

    class _FakeMASTClient:
        def __init__(self, cache_dir: str | None = None) -> None:
            seen["cache_dir"] = cache_dir

        def download_all_sectors(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            return [_lc(42)]

    monkeypatch.setattr("bittr_tess_vetter.cli.diagnostics_report_inputs.MASTClient", _FakeMASTClient)

    cache_dir = str(tmp_path / "cache_root")
    lightcurves, path = load_lightcurves_with_sector_policy(
        tic_id=123,
        sectors=None,
        flux_type="pdcsap",
        explicit_sectors=False,
        cache_dir=cache_dir,
    )

    assert path == "mast_discovery"
    assert [int(lc.sector) for lc in lightcurves] == [42]
    assert seen["cache_dir"] == cache_dir
