"""Tests for PersistentCache."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.platform.io import PersistentCache


class TestPersistentCache:
    """Tests for PersistentCache class."""

    def test_init_creates_directory(self) -> None:
        """Cache directory is created on init."""
        with tempfile.TemporaryDirectory() as td:
            cache_dir = Path(td) / "new_cache"
            assert not cache_dir.exists()

            cache = PersistentCache(cache_dir)
            assert cache_dir.exists()
            assert cache.cache_dir == cache_dir

    def test_set_and_get(self) -> None:
        """Values can be stored and retrieved."""
        with tempfile.TemporaryDirectory() as td:
            cache = PersistentCache(td)

            cache.set("key1", {"foo": "bar"})
            result = cache.get("key1")

            assert result == {"foo": "bar"}

    def test_get_missing_key_returns_none(self) -> None:
        """Getting a missing key returns None."""
        with tempfile.TemporaryDirectory() as td:
            cache = PersistentCache(td)
            assert cache.get("nonexistent") is None

    def test_has_returns_correct_bool(self) -> None:
        """has() returns True for existing keys, False otherwise."""
        with tempfile.TemporaryDirectory() as td:
            cache = PersistentCache(td)

            assert not cache.has("key1")
            cache.set("key1", "value")
            assert cache.has("key1")

    def test_keys_returns_all_keys(self) -> None:
        """keys() returns list of all cached keys."""
        with tempfile.TemporaryDirectory() as td:
            cache = PersistentCache(td)

            assert cache.keys() == []

            cache.set("key1", "value1")
            cache.set("key2", "value2")

            keys = cache.keys()
            assert set(keys) == {"key1", "key2"}

    def test_persistence_across_instances(self) -> None:
        """Cache persists across PersistentCache instances."""
        with tempfile.TemporaryDirectory() as td:
            cache1 = PersistentCache(td)
            cache1.set("persistent_key", [1, 2, 3])

            # Create new instance pointing to same directory
            cache2 = PersistentCache(td)
            assert cache2.get("persistent_key") == [1, 2, 3]

    def test_lc_key_pattern(self) -> None:
        """Light curve key pattern lc:<tic_id>:<sector> works."""
        with tempfile.TemporaryDirectory() as td:
            cache = PersistentCache(td)

            cache.set("lc:12345:1", {"time": [1, 2, 3], "flux": [1.0, 1.0, 1.0]})
            cache.set("lc:12345:2", {"time": [4, 5, 6], "flux": [1.0, 1.0, 1.0]})

            assert cache.has("lc:12345:1")
            assert cache.has("lc:12345:2")
            assert not cache.has("lc:12345:3")

    def test_meta_captures_tic_id_sector_for_lightcurve_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = PersistentCache(td)

            lc = LightCurveData(
                time=np.linspace(1000.0, 1010.0, 10, dtype=np.float64),
                flux=np.ones(10, dtype=np.float64),
                flux_err=np.ones(10, dtype=np.float64) * 1e-4,
                quality=np.zeros(10, dtype=np.int32),
                valid_mask=np.ones(10, dtype=np.bool_),
                tic_id=123,
                sector=7,
                cadence_seconds=120.0,
            )

            key = "lc:123:7:pdcsap"
            cache.set(key, lc)

            meta_files = list((Path(td) / "meta").glob("*.json"))
            assert len(meta_files) == 1
            meta = meta_files[0].read_text(encoding="utf-8")
            assert '"key"' in meta
            assert '"tic_id": 123' in meta
            assert '"sector": 7' in meta

    def test_eviction_prefers_oldest_accessed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Use deterministic timestamps so accessed_at ordering is stable.
        clock = {"t": 1000.0}

        def fake_time() -> float:
            clock["t"] += 1.0
            return clock["t"]

        monkeypatch.setattr("bittr_tess_vetter.platform.io.cache.time.time", fake_time)

        with tempfile.TemporaryDirectory() as td:
            cache = PersistentCache(td, max_entries=2)
            cache.set("k1", {"v": 1})
            cache.set("k2", {"v": 2})

            # Touch k1 so k2 is the oldest accessed.
            assert cache.get("k1") == {"v": 1}

            # Adding k3 should evict k2.
            cache.set("k3", {"v": 3})

            assert cache.has("k1")
            assert not cache.has("k2")
            assert cache.has("k3")
