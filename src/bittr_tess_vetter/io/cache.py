"""Persistent cache for light curves and derived products."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any


class PersistentCache:
    """Simple file-based cache for light curve data.

    Keys follow pattern: `lc:<tic_id>:<sector>`
    Values are objects with .time, .flux, .flux_err, .valid_mask attributes.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "_index.pkl"
        self._index: dict[str, str] = self._load_index()

    def _load_index(self) -> dict[str, str]:
        if self._index_file.exists():
            try:
                with open(self._index_file, "rb") as f:
                    return pickle.load(f)  # noqa: S301
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        with open(self._index_file, "wb") as f:
            pickle.dump(self._index, f)

    def _key_to_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}.pkl"

    def keys(self) -> list[str]:
        """Return all cached keys."""
        return list(self._index.keys())

    def get(self, key: str) -> Any | None:
        """Get cached value by key, or None if not found."""
        if key not in self._index:
            return None
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)  # noqa: S301
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        """Store value under key."""
        path = self._key_to_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f)
        self._index[key] = str(path)
        self._save_index()

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._index and self._key_to_path(key).exists()
