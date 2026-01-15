"""Session-scoped and persistent caching for I/O-adjacent data products."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import sys
import time

if sys.platform != "win32":
    import fcntl
else:
    fcntl = None  # type: ignore[assignment]
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, TypeVar

from bittr_tess_vetter.api.lightcurve import LightCurveData

T = TypeVar("T")


_INSECURE_PERMS_MASK = 0o022  # group/other writable


def _is_secure_pickle_path(path: Path) -> bool:
    """Best-effort guardrail for loading pickled cache entries.

    Pickle is not safe to load from untrusted locations. Since this cache is
    user-writable by design, we harden against common footguns:
    - refuse symlinks (path traversal / TOCTOU)
    - refuse group/other-writable cache files
    - (POSIX) refuse cache files not owned by current uid
    """
    try:
        if path.is_symlink():
            return False
        st = path.stat()
        if (st.st_mode & _INSECURE_PERMS_MASK) != 0:
            return False
        getuid = getattr(os, "getuid", None)
        if getuid is not None and st.st_uid != getuid():
            return False
        return True
    except OSError:
        return False


def _best_effort_chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except Exception:
        return


def _default_cache_dir() -> Path:
    """Choose a default on-disk cache directory.

    Preference order:
    1) `BITTR_TESS_VETTER_CACHE_DIR` (explicit override)
    2) `BITTR_TESS_VETTER_CACHE_ROOT` (repo-local cache root)
    3) OS-appropriate user cache directory (via platformdirs, if available)
    4) `.bittr-tess-vetter/cache/persistent_cache` under current working directory
    """
    # 1. Explicit override
    explicit = os.getenv("BITTR_TESS_VETTER_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()

    # 2. Repo-local override
    root = os.getenv("BITTR_TESS_VETTER_CACHE_ROOT")
    if root:
        return Path(root).expanduser() / "persistent_cache"

    # 3. OS-appropriate user cache directory
    try:
        from platformdirs import user_cache_dir  # type: ignore[import-not-found]

        return Path(user_cache_dir("bittr-tess-vetter"))
    except ImportError:
        pass

    # 4. Fallback to CWD-relative
    return Path.cwd() / ".bittr-tess-vetter" / "cache" / "persistent_cache"


DEFAULT_CACHE_DIR = _default_cache_dir()


class SessionCache:
    """Process-global cache with session namespacing.

    Provides LRU-evicting cache for LightCurveData objects keyed by data_ref.
    Each session has its own namespace within the global cache.
    """

    _global: ClassVar[OrderedDict[str, LightCurveData]] = OrderedDict()
    _computed: ClassVar[OrderedDict[str, Any]] = OrderedDict()
    _lock: ClassVar[Lock] = Lock()

    DEFAULT_MAX_ENTRIES: ClassVar[int] = 100
    DEFAULT_MAX_COMPUTED: ClassVar[int] = 256

    def __init__(
        self,
        session_id: str,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_computed: int = DEFAULT_MAX_COMPUTED,
    ) -> None:
        self.session_id = session_id
        self._max_entries = max_entries
        self._max_computed = max_computed

    def _key(self, data_ref: str) -> str:
        return f"{self.session_id}:{data_ref}"

    def _computed_key(self, product_ref: str) -> str:
        return f"{self.session_id}:{product_ref}"

    @classmethod
    def _evict_lru(cls) -> None:
        if cls._global:
            cls._global.popitem(last=False)

    @classmethod
    def _evict_computed_lru(cls) -> None:
        if cls._computed:
            cls._computed.popitem(last=False)

    def get(self, data_ref: str) -> LightCurveData | None:
        key = self._key(data_ref)
        with self._lock:
            if key in self._global:
                self._global.move_to_end(key)
                return self._global[key]
            return None

    def put(self, data_ref: str, data: LightCurveData) -> None:
        key = self._key(data_ref)
        with self._lock:
            if key in self._global:
                self._global[key] = data
                self._global.move_to_end(key)
                return

            while len(self._global) >= self._max_entries:
                self._evict_lru()

            self._global[key] = data

    def get_or_load(self, data_ref: str, loader: Callable[[], LightCurveData]) -> LightCurveData:
        data = self.get(data_ref)
        if data is not None:
            return data

        loaded = loader()
        self.put(data_ref, loaded)
        return loaded

    def has(self, data_ref: str) -> bool:
        key = self._key(data_ref)
        with self._lock:
            return key in self._global

    def keys(self) -> list[str]:
        prefix = f"{self.session_id}:"
        with self._lock:
            return [k[len(prefix) :] for k in self._global if k.startswith(prefix)]

    def clear(self) -> None:
        prefix = f"{self.session_id}:"
        with self._lock:
            for k in [k for k in self._global if k.startswith(prefix)]:
                del self._global[k]
            for k in [k for k in self._computed if k.startswith(prefix)]:
                del self._computed[k]

    def get_computed(self, product_ref: str) -> Any | None:
        key = self._computed_key(product_ref)
        with self._lock:
            if key in self._computed:
                self._computed.move_to_end(key)
                return self._computed[key]
            return None

    def put_computed(self, product_ref: str, product: Any) -> None:
        key = self._computed_key(product_ref)
        with self._lock:
            if key in self._computed:
                self._computed[key] = product
                self._computed.move_to_end(key)
                return

            while len(self._computed) >= self._max_computed:
                self._evict_computed_lru()

            self._computed[key] = product

    def get_or_compute(self, product_ref: str, compute: Callable[[], T]) -> T:
        cached = self.get_computed(product_ref)
        if cached is not None:
            return cached  # type: ignore[return-value]

        computed: T = compute()
        self.put_computed(product_ref, computed)
        return computed

    def has_computed(self, product_ref: str) -> bool:
        key = self._computed_key(product_ref)
        with self._lock:
            return key in self._computed

    def computed_keys(self) -> list[str]:
        prefix = f"{self.session_id}:"
        with self._lock:
            return [k[len(prefix) :] for k in self._computed if k.startswith(prefix)]

    @classmethod
    def clear_all(cls) -> None:
        with cls._lock:
            cls._global.clear()
            cls._computed.clear()

    @classmethod
    def global_stats(cls) -> dict[str, int]:
        with cls._lock:
            return {"data_entries": len(cls._global), "computed_entries": len(cls._computed)}


class PersistentCache:
    """File-based cache that persists across process restarts.

    API is intentionally a superset of the original bittr-tess-vetter cache:
    - `get(key)` / `set(key, value)` / `has(key)` / `keys()`
    - `put(key, value)` is an alias of `set()` (semantic clarity)
    - `get_default()` returns a process-global singleton instance
    """

    _default_instance: ClassVar[PersistentCache | None] = None
    _instance_lock: ClassVar[Lock] = Lock()

    DEFAULT_MAX_ENTRIES: ClassVar[int] = 100

    def __init__(self, cache_dir: str | Path | None = None, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.max_entries = max_entries
        self._lock = Lock()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self.cache_dir / "data"
        meta_dir = self.cache_dir / "meta"
        data_dir.mkdir(exist_ok=True)
        meta_dir.mkdir(exist_ok=True)
        _best_effort_chmod(self.cache_dir, 0o700)
        _best_effort_chmod(data_dir, 0o700)
        _best_effort_chmod(meta_dir, 0o700)

    @classmethod
    def get_default(cls) -> PersistentCache:
        if cls._default_instance is None:
            with cls._instance_lock:
                if cls._default_instance is None:
                    cls._default_instance = cls()
        return cls._default_instance

    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def _data_path(self, key: str) -> Path:
        return self.cache_dir / "data" / f"{self._hash_key(key)}.pkl"

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / "meta" / f"{self._hash_key(key)}.json"

    def keys(self) -> list[str]:
        meta_dir = self.cache_dir / "meta"
        if not meta_dir.exists():
            return []
        out: list[str] = []
        for path in meta_dir.glob("*.json"):
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                key = meta.get("key")
                if isinstance(key, str):
                    out.append(key)
            except Exception:
                continue
        return sorted(set(out))

    def get(self, key: str) -> Any | None:
        data_path = self._data_path(key)
        if not data_path.exists():
            return None
        if not _is_secure_pickle_path(data_path):
            self._remove_entry(key)
            return None

        with self._lock:
            try:
                with open(data_path, "rb") as f:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        value = pickle.load(f)  # noqa: S301
                    finally:
                        if fcntl is not None:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                self._remove_entry(key)
                return None

            self._update_access_time(key)
            return value

    def set(self, key: str, value: Any) -> None:
        self.put(key, value)

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._evict_if_needed()
            data_path = self._data_path(key)
            meta_path = self._meta_path(key)

            with open(data_path, "wb") as f:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    pickle.dump(value, f)  # noqa: S301
                finally:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            _best_effort_chmod(data_path, 0o600)

            meta: dict[str, Any] = {
                "key": key,
                "created_at": time.time(),
                "accessed_at": time.time(),
            }
            tic_id = getattr(value, "tic_id", None)
            if tic_id is not None:
                meta["tic_id"] = int(tic_id)
            sector = getattr(value, "sector", None)
            if sector is not None:
                meta["sector"] = int(sector)

            with contextlib.suppress(Exception):
                meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def has(self, key: str) -> bool:
        return self._data_path(key).exists()

    def clear(self) -> None:
        """Remove all cached entries from disk.

        Intended for test isolation and for hosts that want to reset state between runs.
        """
        data_dir = self.cache_dir / "data"
        meta_dir = self.cache_dir / "meta"
        with self._lock:
            for dir_path in (data_dir, meta_dir):
                if not dir_path.exists():
                    continue
                for path in dir_path.glob("*"):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        continue
                    except Exception:
                        continue

    def _update_access_time(self, key: str) -> None:
        meta_path = self._meta_path(key)
        try:
            meta = (
                json.loads(meta_path.read_text(encoding="utf-8"))
                if meta_path.exists()
                else {"key": key}
            )
            meta["accessed_at"] = time.time()
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except Exception:
            return

    def _remove_entry(self, key: str) -> None:
        for path in (self._data_path(key), self._meta_path(key)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _evict_if_needed(self) -> None:
        keys = self.keys()
        if len(keys) < self.max_entries:
            return

        entries: list[tuple[float, str]] = []
        for key in keys:
            meta_path = self._meta_path(key)
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                accessed_at = float(meta.get("accessed_at", 0.0))
            except Exception:
                accessed_at = 0.0
            entries.append((accessed_at, key))

        entries.sort(key=lambda x: x[0])
        while len(entries) >= self.max_entries and entries:
            _, key = entries.pop(0)
            self._remove_entry(key)
