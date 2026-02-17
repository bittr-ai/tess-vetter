from __future__ import annotations

import importlib
from pathlib import Path

from bittr_tess_vetter.validation import triceratops_fpp as mod


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
    funcs_mod = importlib.import_module("bittr_tess_vetter.ext.triceratops_plus_vendor.triceratops.funcs")
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
