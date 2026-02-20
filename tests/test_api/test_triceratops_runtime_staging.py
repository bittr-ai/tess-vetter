from __future__ import annotations

import contextlib
import importlib
from pathlib import Path

import pytest

from tess_vetter.validation import triceratops_fpp as mod

pytest.importorskip("lightkurve")
pytest.importorskip("pytransit")


class _FakeCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir


class _FakeTarget:
    def __init__(self) -> None:
        self.trilegal_fname = None
        self.trilegal_url = "http://stev.oapd.inaf.it/tmp/expired.dat"
        self.ra = None
        self.dec = None
        self.stars = {"ra": [123.45], "dec": [-12.34]}


def test_stage_runtime_artifacts_requeries_trilegal_on_expired_or_empty(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _FakeCache(cache_dir=cache_dir)

    target = _FakeTarget()
    seen_urls: list[str] = []

    monkeypatch.setattr(mod, "_gather_light_curves", lambda *_args, **_kwargs: ({"time": [0.0]}, [5, 6]))
    monkeypatch.setattr(mod, "_load_cached_triceratops_target", lambda **_kwargs: target)
    monkeypatch.setattr(mod, "_save_cached_triceratops_target", lambda **_kwargs: None)
    funcs_mod = importlib.import_module("tess_vetter.ext.triceratops_plus_vendor.triceratops.funcs")
    monkeypatch.setattr(funcs_mod, "query_TRILEGAL", lambda *_args, **_kwargs: "http://stev.oapd.inaf.it/tmp/fresh.dat")

    def _fake_prefetch(*, trilegal_url: str, **_kwargs):
        seen_urls.append(str(trilegal_url))
        if "expired" in trilegal_url:
            raise RuntimeError("HTTP Error 404: Not Found")
        out = cache_dir / "triceratops" / "123_TRILEGAL.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("col1,col2\n1,2\n", encoding="utf-8")
        return str(out)

    out = mod.stage_triceratops_runtime_artifacts(
        cache=cache,
        tic_id=123,
        sectors=[5, 6],
        prefetch_trilegal_csv=_fake_prefetch,
    )

    assert out["trilegal_csv_path"] is not None
    assert len(seen_urls) == 2
    assert "expired" in seen_urls[0]
    assert "fresh" in seen_urls[1]


def test_stage_runtime_artifacts_allows_explicit_timeout_to_expand_stage_budgets(
    monkeypatch, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _FakeCache(cache_dir=cache_dir)
    seen_timeouts: list[float] = []

    monkeypatch.setattr(mod, "_gather_light_curves", lambda *_args, **_kwargs: ({"time": [0.0]}, [5, 6]))
    monkeypatch.setattr(mod, "_load_cached_triceratops_target", lambda **_kwargs: None)
    monkeypatch.setattr(mod, "_save_cached_triceratops_target", lambda **_kwargs: None)

    @contextlib.contextmanager
    def _fake_network_timeout(timeout_seconds: float, operation: str):  # type: ignore[override]
        seen_timeouts.append(float(timeout_seconds))
        yield

    monkeypatch.setattr(mod, "network_timeout", _fake_network_timeout)

    class _FreshTarget:
        trilegal_fname = None
        trilegal_url = "http://stev.oapd.inaf.it/tmp/fresh.dat"
        stars = {"ra": [123.45], "dec": [-12.34]}

    tr_mod = importlib.import_module("tess_vetter.ext.triceratops_plus_vendor.triceratops.triceratops")
    monkeypatch.setattr(tr_mod, "target", lambda **_kwargs: _FreshTarget())

    def _fake_prefetch(**_kwargs):
        out = cache_dir / "triceratops" / "123_TRILEGAL.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("col1,col2\n1,2\n", encoding="utf-8")
        return str(out)

    out = mod.stage_triceratops_runtime_artifacts(
        cache=cache,
        tic_id=123,
        sectors=[5, 6],
        timeout_seconds=900.0,
        prefetch_trilegal_csv=_fake_prefetch,
    )

    assert out["trilegal_csv_path"] is not None
    # init + trilegal prefetch stages both use network_timeout.
    assert len(seen_timeouts) == 2
    assert seen_timeouts[0] > mod.FPP_STAGE_INIT_BUDGET_SECONDS
