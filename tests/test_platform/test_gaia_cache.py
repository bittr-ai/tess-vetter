from __future__ import annotations

import json
from pathlib import Path

import pytest

from bittr_tess_vetter.platform.catalogs.gaia_client import GaiaQueryResult, query_gaia_by_position_sync


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return

    def json(self):
        return self._payload


def test_gaia_position_query_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _fake_get(url: str, params=None, timeout=None):
        calls["n"] += 1
        # Minimal Gaia TAP JSON format expected by client: metadata+data.
        payload = {
            "metadata": [
                {"name": "source_id"},
                {"name": "ra"},
                {"name": "dec"},
                {"name": "phot_g_mean_mag"},
            ],
            "data": [
                [123, 10.0, -5.0, 12.3],
            ],
        }
        return _FakeResp(payload)

    import bittr_tess_vetter.platform.catalogs.gaia_client as mod

    monkeypatch.setattr(mod, "GAIA_TAP_ENDPOINT", "https://fake")

    import requests

    monkeypatch.setattr(requests, "get", _fake_get)

    cache_path = str(tmp_path / "gaia.sqlite")
    r1 = query_gaia_by_position_sync(
        10.0,
        -5.0,
        radius_arcsec=60.0,
        tap_url="https://fake",
        timeout=1,
        max_retries=1,
        cache_path=cache_path,
    )
    assert isinstance(r1, GaiaQueryResult)
    assert calls["n"] > 0

    calls_before = calls["n"]
    r2 = query_gaia_by_position_sync(
        10.0,
        -5.0,
        radius_arcsec=60.0,
        tap_url="https://fake",
        timeout=1,
        max_retries=1,
        cache_path=cache_path,
    )
    assert isinstance(r2, GaiaQueryResult)
    assert calls["n"] == calls_before
    assert json.loads(r2.model_dump_json())["source"]["source_id"] == 123

