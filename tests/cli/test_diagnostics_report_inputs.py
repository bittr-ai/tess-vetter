from __future__ import annotations

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
