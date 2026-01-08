"""Tests for PersistentCache."""

from __future__ import annotations

import tempfile
from pathlib import Path

from bittr_tess_vetter.io import PersistentCache


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
