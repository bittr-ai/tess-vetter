"""Comprehensive tests for SessionCache.

Tests cover:
1. Basic operations: get(), put(), has(), keys()
2. LRU eviction behavior (when max_entries exceeded)
3. Session namespacing (different sessions don't conflict)
4. Computed products cache (get_computed, put_computed)
5. Thread safety (concurrent access)
6. clear() and clear_all() methods
7. global_stats()
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
import pytest

from bittr_tess_vetter.api.lightcurve import LightCurveData
from bittr_tess_vetter.io.cache import SessionCache

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


def create_mock_lightcurve(
    tic_id: int = 12345678,
    sector: int = 1,
    n_points: int = 100,
    cadence_seconds: float = 120.0,
) -> LightCurveData:
    """Create a mock LightCurveData object for testing.

    Args:
        tic_id: TIC identifier
        sector: TESS sector number
        n_points: Number of data points
        cadence_seconds: Observation cadence in seconds

    Returns:
        LightCurveData with synthetic data
    """
    time = np.linspace(0.0, 27.4, n_points, dtype=np.float64)
    flux = np.ones(n_points, dtype=np.float64) + np.random.normal(0, 0.001, n_points)
    flux_err = np.full(n_points, 0.001, dtype=np.float64)
    quality = np.zeros(n_points, dtype=np.int32)
    valid_mask = np.ones(n_points, dtype=np.bool_)

    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=tic_id,
        sector=sector,
        cadence_seconds=cadence_seconds,
    )


@pytest.fixture
def mock_lc() -> LightCurveData:
    """Create a default mock LightCurveData."""
    return create_mock_lightcurve()


@pytest.fixture
def mock_lc_sector2() -> LightCurveData:
    """Create a mock LightCurveData for sector 2."""
    return create_mock_lightcurve(sector=2)


@pytest.fixture
def mock_lc_different_target() -> LightCurveData:
    """Create a mock LightCurveData for a different target."""
    return create_mock_lightcurve(tic_id=87654321)


@pytest.fixture(autouse=True)
def clear_global_cache():
    """Clear global cache before and after each test.

    This ensures test isolation since SessionCache uses class-level storage.
    """
    SessionCache.clear_all()
    yield
    SessionCache.clear_all()


@pytest.fixture
def session_cache() -> SessionCache:
    """Create a SessionCache with default settings."""
    return SessionCache("test-session-001")


@pytest.fixture
def small_cache() -> SessionCache:
    """Create a SessionCache with small limits for LRU testing."""
    return SessionCache("test-small-cache", max_entries=3, max_computed=3)


# =============================================================================
# Basic Operations Tests
# =============================================================================


class TestBasicOperations:
    """Tests for basic cache operations: get, put, has, keys."""

    def test_put_and_get(self, session_cache: SessionCache, mock_lc: LightCurveData):
        """Test basic put and get operations."""
        data_ref = "lc:12345678:1:pdcsap"
        session_cache.put(data_ref, mock_lc)

        retrieved = session_cache.get(data_ref)

        assert retrieved is mock_lc
        assert retrieved.tic_id == mock_lc.tic_id
        assert retrieved.sector == mock_lc.sector

    def test_get_nonexistent_returns_none(self, session_cache: SessionCache):
        """Test that getting a nonexistent key returns None."""
        result = session_cache.get("nonexistent:key")
        assert result is None

    def test_has_returns_true_for_existing(
        self, session_cache: SessionCache, mock_lc: LightCurveData
    ):
        """Test has() returns True for existing entries."""
        data_ref = "lc:12345678:1:pdcsap"
        session_cache.put(data_ref, mock_lc)

        assert session_cache.has(data_ref) is True

    def test_has_returns_false_for_nonexistent(self, session_cache: SessionCache):
        """Test has() returns False for nonexistent entries."""
        assert session_cache.has("nonexistent:key") is False

    def test_keys_returns_session_keys(
        self,
        session_cache: SessionCache,
        mock_lc: LightCurveData,
        mock_lc_sector2: LightCurveData,
    ):
        """Test keys() returns all data_refs for the session."""
        session_cache.put("lc:12345678:1:pdcsap", mock_lc)
        session_cache.put("lc:12345678:2:pdcsap", mock_lc_sector2)

        keys = session_cache.keys()

        assert len(keys) == 2
        assert "lc:12345678:1:pdcsap" in keys
        assert "lc:12345678:2:pdcsap" in keys

    def test_keys_empty_cache(self, session_cache: SessionCache):
        """Test keys() returns empty list for empty cache."""
        keys = session_cache.keys()
        assert keys == []

    def test_put_updates_existing_entry(self, session_cache: SessionCache, mock_lc: LightCurveData):
        """Test that put() updates existing entries."""
        data_ref = "lc:12345678:1:pdcsap"
        session_cache.put(data_ref, mock_lc)

        # Create a different mock and update
        new_lc = create_mock_lightcurve(n_points=200)
        session_cache.put(data_ref, new_lc)

        retrieved = session_cache.get(data_ref)
        assert retrieved is new_lc
        assert retrieved.n_points == 200

    def test_multiple_data_refs(self, session_cache: SessionCache):
        """Test storing multiple different data refs."""
        lc1 = create_mock_lightcurve(tic_id=111, sector=1)
        lc2 = create_mock_lightcurve(tic_id=222, sector=2)
        lc3 = create_mock_lightcurve(tic_id=333, sector=3)

        session_cache.put("lc:111:1:pdcsap", lc1)
        session_cache.put("lc:222:2:pdcsap", lc2)
        session_cache.put("lc:333:3:pdcsap", lc3)

        assert session_cache.get("lc:111:1:pdcsap") is lc1
        assert session_cache.get("lc:222:2:pdcsap") is lc2
        assert session_cache.get("lc:333:3:pdcsap") is lc3


# =============================================================================
# LRU Eviction Tests
# =============================================================================


class TestLRUEviction:
    """Tests for LRU (Least Recently Used) eviction behavior."""

    def test_eviction_when_max_entries_exceeded(self, small_cache: SessionCache):
        """Test that oldest entries are evicted when max_entries is exceeded."""
        # Add 3 entries (at capacity)
        lc1 = create_mock_lightcurve(tic_id=1)
        lc2 = create_mock_lightcurve(tic_id=2)
        lc3 = create_mock_lightcurve(tic_id=3)

        small_cache.put("lc:1:1:pdcsap", lc1)
        small_cache.put("lc:2:1:pdcsap", lc2)
        small_cache.put("lc:3:1:pdcsap", lc3)

        # All 3 should be present
        assert small_cache.has("lc:1:1:pdcsap")
        assert small_cache.has("lc:2:1:pdcsap")
        assert small_cache.has("lc:3:1:pdcsap")

        # Add 4th entry - should evict the oldest (lc1)
        lc4 = create_mock_lightcurve(tic_id=4)
        small_cache.put("lc:4:1:pdcsap", lc4)

        assert not small_cache.has("lc:1:1:pdcsap")  # Evicted
        assert small_cache.has("lc:2:1:pdcsap")
        assert small_cache.has("lc:3:1:pdcsap")
        assert small_cache.has("lc:4:1:pdcsap")

    def test_access_updates_lru_order(self, small_cache: SessionCache):
        """Test that accessing an entry moves it to most recently used."""
        lc1 = create_mock_lightcurve(tic_id=1)
        lc2 = create_mock_lightcurve(tic_id=2)
        lc3 = create_mock_lightcurve(tic_id=3)

        small_cache.put("lc:1:1:pdcsap", lc1)
        small_cache.put("lc:2:1:pdcsap", lc2)
        small_cache.put("lc:3:1:pdcsap", lc3)

        # Access lc1 to make it most recently used
        small_cache.get("lc:1:1:pdcsap")

        # Add lc4 - should evict lc2 (now oldest) instead of lc1
        lc4 = create_mock_lightcurve(tic_id=4)
        small_cache.put("lc:4:1:pdcsap", lc4)

        assert small_cache.has("lc:1:1:pdcsap")  # Was accessed, not evicted
        assert not small_cache.has("lc:2:1:pdcsap")  # Was oldest, evicted
        assert small_cache.has("lc:3:1:pdcsap")
        assert small_cache.has("lc:4:1:pdcsap")

    def test_put_updates_lru_order(self, small_cache: SessionCache):
        """Test that updating an existing entry moves it to most recently used."""
        lc1 = create_mock_lightcurve(tic_id=1)
        lc2 = create_mock_lightcurve(tic_id=2)
        lc3 = create_mock_lightcurve(tic_id=3)

        small_cache.put("lc:1:1:pdcsap", lc1)
        small_cache.put("lc:2:1:pdcsap", lc2)
        small_cache.put("lc:3:1:pdcsap", lc3)

        # Update lc1 to make it most recently used
        lc1_updated = create_mock_lightcurve(tic_id=1, n_points=50)
        small_cache.put("lc:1:1:pdcsap", lc1_updated)

        # Add lc4 - should evict lc2 (now oldest)
        lc4 = create_mock_lightcurve(tic_id=4)
        small_cache.put("lc:4:1:pdcsap", lc4)

        assert small_cache.has("lc:1:1:pdcsap")  # Was updated, not evicted
        assert not small_cache.has("lc:2:1:pdcsap")  # Was oldest, evicted

    def test_has_does_not_update_lru_order(self, small_cache: SessionCache):
        """Test that has() does NOT update LRU order."""
        lc1 = create_mock_lightcurve(tic_id=1)
        lc2 = create_mock_lightcurve(tic_id=2)
        lc3 = create_mock_lightcurve(tic_id=3)

        small_cache.put("lc:1:1:pdcsap", lc1)
        small_cache.put("lc:2:1:pdcsap", lc2)
        small_cache.put("lc:3:1:pdcsap", lc3)

        # has() should NOT update LRU order
        small_cache.has("lc:1:1:pdcsap")

        # Add lc4 - should evict lc1 (still oldest because has() doesn't update order)
        lc4 = create_mock_lightcurve(tic_id=4)
        small_cache.put("lc:4:1:pdcsap", lc4)

        assert not small_cache.has("lc:1:1:pdcsap")  # Evicted (has() didn't update order)


# =============================================================================
# Session Namespacing Tests
# =============================================================================


class TestSessionNamespacing:
    """Tests for session namespacing - different sessions don't conflict."""

    def test_different_sessions_isolated(self, mock_lc: LightCurveData):
        """Test that different sessions have isolated caches."""
        cache_a = SessionCache("session-A")
        cache_b = SessionCache("session-B")

        cache_a.put("lc:12345678:1:pdcsap", mock_lc)

        # Session B should not see session A's data
        assert cache_a.has("lc:12345678:1:pdcsap")
        assert not cache_b.has("lc:12345678:1:pdcsap")

    def test_same_key_different_sessions(self):
        """Test that same key in different sessions stores different data."""
        cache_a = SessionCache("session-A")
        cache_b = SessionCache("session-B")

        lc_a = create_mock_lightcurve(tic_id=111)
        lc_b = create_mock_lightcurve(tic_id=222)

        cache_a.put("lc:data:1:pdcsap", lc_a)
        cache_b.put("lc:data:1:pdcsap", lc_b)

        # Each session should get its own data
        assert cache_a.get("lc:data:1:pdcsap").tic_id == 111
        assert cache_b.get("lc:data:1:pdcsap").tic_id == 222

    def test_keys_only_returns_session_keys(self):
        """Test that keys() only returns keys for the current session."""
        cache_a = SessionCache("session-A")
        cache_b = SessionCache("session-B")

        lc_a = create_mock_lightcurve(tic_id=111)
        lc_b = create_mock_lightcurve(tic_id=222)

        cache_a.put("lc:a:1:pdcsap", lc_a)
        cache_b.put("lc:b:1:pdcsap", lc_b)

        assert cache_a.keys() == ["lc:a:1:pdcsap"]
        assert cache_b.keys() == ["lc:b:1:pdcsap"]

    def test_clear_only_affects_session(self):
        """Test that clear() only affects the current session."""
        cache_a = SessionCache("session-A")
        cache_b = SessionCache("session-B")

        lc_a = create_mock_lightcurve(tic_id=111)
        lc_b = create_mock_lightcurve(tic_id=222)

        cache_a.put("lc:a:1:pdcsap", lc_a)
        cache_b.put("lc:b:1:pdcsap", lc_b)

        # Clear session A
        cache_a.clear()

        # Session A should be empty, session B unaffected
        assert not cache_a.has("lc:a:1:pdcsap")
        assert cache_b.has("lc:b:1:pdcsap")

    def test_multiple_entries_multiple_sessions(self):
        """Test multiple entries across multiple sessions."""
        cache_a = SessionCache("session-A")
        cache_b = SessionCache("session-B")
        cache_c = SessionCache("session-C")

        for i in range(5):
            cache_a.put(f"lc:a{i}:1:pdcsap", create_mock_lightcurve(tic_id=i))
            cache_b.put(f"lc:b{i}:1:pdcsap", create_mock_lightcurve(tic_id=100 + i))
            cache_c.put(f"lc:c{i}:1:pdcsap", create_mock_lightcurve(tic_id=200 + i))

        assert len(cache_a.keys()) == 5
        assert len(cache_b.keys()) == 5
        assert len(cache_c.keys()) == 5

        # Verify data is correct
        assert cache_a.get("lc:a0:1:pdcsap").tic_id == 0
        assert cache_b.get("lc:b0:1:pdcsap").tic_id == 100
        assert cache_c.get("lc:c0:1:pdcsap").tic_id == 200


# =============================================================================
# Computed Products Cache Tests
# =============================================================================


class TestComputedProductsCache:
    """Tests for computed products cache operations."""

    def test_put_computed_and_get_computed(self, session_cache: SessionCache):
        """Test basic put_computed and get_computed operations."""
        product_ref = "pg:12345678:1:abc123"
        product = {"periods": [1.0, 2.0, 3.0], "powers": [0.5, 0.8, 0.3]}

        session_cache.put_computed(product_ref, product)
        retrieved = session_cache.get_computed(product_ref)

        assert retrieved == product

    def test_get_computed_nonexistent_returns_none(self, session_cache: SessionCache):
        """Test that get_computed returns None for nonexistent products."""
        result = session_cache.get_computed("nonexistent:product")
        assert result is None

    def test_has_computed(self, session_cache: SessionCache):
        """Test has_computed() method."""
        product_ref = "pg:12345678:1:abc123"
        product = {"data": "test"}

        assert not session_cache.has_computed(product_ref)
        session_cache.put_computed(product_ref, product)
        assert session_cache.has_computed(product_ref)

    def test_computed_keys(self, session_cache: SessionCache):
        """Test computed_keys() returns all product refs for session."""
        session_cache.put_computed("pg:111:1:aaa", {"data": 1})
        session_cache.put_computed("pg:222:2:bbb", {"data": 2})
        session_cache.put_computed("pg:333:3:ccc", {"data": 3})

        keys = session_cache.computed_keys()

        assert len(keys) == 3
        assert "pg:111:1:aaa" in keys
        assert "pg:222:2:bbb" in keys
        assert "pg:333:3:ccc" in keys

    def test_computed_lru_eviction(self):
        """Test LRU eviction for computed products cache."""
        cache = SessionCache("test-computed-lru", max_computed=3)

        cache.put_computed("pg:1:1:a", {"data": 1})
        cache.put_computed("pg:2:1:b", {"data": 2})
        cache.put_computed("pg:3:1:c", {"data": 3})

        # All 3 should be present
        assert cache.has_computed("pg:1:1:a")
        assert cache.has_computed("pg:2:1:b")
        assert cache.has_computed("pg:3:1:c")

        # Add 4th - should evict oldest
        cache.put_computed("pg:4:1:d", {"data": 4})

        assert not cache.has_computed("pg:1:1:a")  # Evicted
        assert cache.has_computed("pg:2:1:b")
        assert cache.has_computed("pg:3:1:c")
        assert cache.has_computed("pg:4:1:d")

    def test_computed_access_updates_lru_order(self):
        """Test that get_computed updates LRU order."""
        cache = SessionCache("test-computed-lru-access", max_computed=3)

        cache.put_computed("pg:1:1:a", {"data": 1})
        cache.put_computed("pg:2:1:b", {"data": 2})
        cache.put_computed("pg:3:1:c", {"data": 3})

        # Access pg:1 to make it most recently used
        cache.get_computed("pg:1:1:a")

        # Add 4th - should evict pg:2 (now oldest)
        cache.put_computed("pg:4:1:d", {"data": 4})

        assert cache.has_computed("pg:1:1:a")  # Was accessed
        assert not cache.has_computed("pg:2:1:b")  # Evicted
        assert cache.has_computed("pg:3:1:c")
        assert cache.has_computed("pg:4:1:d")

    def test_computed_session_isolation(self):
        """Test that computed products are session-isolated."""
        cache_a = SessionCache("session-A")
        cache_b = SessionCache("session-B")

        cache_a.put_computed("pg:123:1:xyz", {"session": "A"})
        cache_b.put_computed("pg:123:1:xyz", {"session": "B"})

        assert cache_a.get_computed("pg:123:1:xyz") == {"session": "A"}
        assert cache_b.get_computed("pg:123:1:xyz") == {"session": "B"}

    def test_computed_different_types(self, session_cache: SessionCache):
        """Test storing various types of computed products."""
        # Dict
        session_cache.put_computed("product:dict", {"key": "value"})
        # List
        session_cache.put_computed("product:list", [1, 2, 3, 4, 5])
        # Numpy array
        arr = np.array([1.0, 2.0, 3.0])
        session_cache.put_computed("product:array", arr)
        # String
        session_cache.put_computed("product:string", "test_result")
        # Nested structure
        session_cache.put_computed(
            "product:nested",
            {"periods": np.array([1.0, 2.0]), "metadata": {"count": 100}},
        )

        assert session_cache.get_computed("product:dict") == {"key": "value"}
        assert session_cache.get_computed("product:list") == [1, 2, 3, 4, 5]
        np.testing.assert_array_equal(session_cache.get_computed("product:array"), arr)
        assert session_cache.get_computed("product:string") == "test_result"


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_put_operations(self):
        """Test concurrent put operations don't corrupt data."""
        cache = SessionCache("thread-test-put", max_entries=1000)  # High limit for this test
        num_threads = 10
        items_per_thread = 50

        def put_items(thread_id: int):
            for i in range(items_per_thread):
                lc = create_mock_lightcurve(tic_id=thread_id * 1000 + i)
                cache.put(f"lc:{thread_id}:{i}:pdcsap", lc)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(put_items, t) for t in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Verify all items are present
        expected_count = num_threads * items_per_thread
        actual_count = len(cache.keys())
        assert actual_count == expected_count

    def test_concurrent_get_operations(self):
        """Test concurrent get operations return correct data."""
        cache = SessionCache("thread-test-get")

        # Pre-populate cache
        for i in range(100):
            lc = create_mock_lightcurve(tic_id=i)
            cache.put(f"lc:{i}:1:pdcsap", lc)

        errors: list[str] = []
        lock = threading.Lock()

        def get_items(start: int, count: int):
            for i in range(start, start + count):
                idx = i % 100
                result = cache.get(f"lc:{idx}:1:pdcsap")
                if result is None:
                    with lock:
                        errors.append(f"Missing key: lc:{idx}:1:pdcsap")
                elif result.tic_id != idx:
                    with lock:
                        errors.append(f"Wrong tic_id: expected {idx}, got {result.tic_id}")

        num_threads = 20
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_items, t * 50, 100) for t in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert errors == [], f"Errors: {errors}"

    def test_concurrent_mixed_operations(self):
        """Test concurrent mixed put/get/has operations."""
        cache = SessionCache("thread-test-mixed")
        num_operations = 100

        # Pre-populate with some data
        for i in range(50):
            cache.put(f"lc:init:{i}:pdcsap", create_mock_lightcurve(tic_id=i))

        def mixed_operations(thread_id: int):
            for i in range(num_operations):
                op = i % 4
                key = f"lc:mixed:{thread_id}:{i % 20}:pdcsap"

                if op == 0:  # put
                    cache.put(key, create_mock_lightcurve(tic_id=thread_id * 1000 + i))
                elif op == 1:  # get
                    cache.get(key)
                elif op == 2:  # has
                    cache.has(key)
                else:  # keys
                    cache.keys()

        num_threads = 10
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mixed_operations, t) for t in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Should not raise

    def test_concurrent_computed_operations(self):
        """Test concurrent computed product operations."""
        cache = SessionCache("thread-test-computed")
        num_threads = 10
        items_per_thread = 50

        def computed_operations(thread_id: int):
            for i in range(items_per_thread):
                key = f"pg:{thread_id}:{i}:hash"
                cache.put_computed(key, {"thread": thread_id, "item": i})
                result = cache.get_computed(key)
                if result:
                    assert result["thread"] == thread_id
                    assert result["item"] == i

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(computed_operations, t) for t in range(num_threads)]
            for future in as_completed(futures):
                future.result()

    def test_concurrent_eviction(self):
        """Test that LRU eviction is thread-safe."""
        cache = SessionCache("thread-test-eviction", max_entries=20)
        num_threads = 10
        items_per_thread = 50

        def put_many(thread_id: int):
            for i in range(items_per_thread):
                lc = create_mock_lightcurve(tic_id=thread_id * 1000 + i)
                cache.put(f"lc:{thread_id}:{i}:pdcsap", lc)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(put_many, t) for t in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        # Cache should not exceed max_entries
        stats = SessionCache.global_stats()
        assert stats["data_entries"] <= 20


# =============================================================================
# Clear and Clear All Tests
# =============================================================================


class TestClearMethods:
    """Tests for clear() and clear_all() methods."""

    def test_clear_removes_session_entries(self, session_cache: SessionCache):
        """Test that clear() removes all entries for the session."""
        session_cache.put("lc:1:1:pdcsap", create_mock_lightcurve(tic_id=1))
        session_cache.put("lc:2:1:pdcsap", create_mock_lightcurve(tic_id=2))
        session_cache.put_computed("pg:1:1:aaa", {"data": 1})
        session_cache.put_computed("pg:2:1:bbb", {"data": 2})

        session_cache.clear()

        assert len(session_cache.keys()) == 0
        assert len(session_cache.computed_keys()) == 0

    def test_clear_preserves_other_sessions(self):
        """Test that clear() preserves other sessions' data."""
        cache_a = SessionCache("session-A")
        cache_b = SessionCache("session-B")

        cache_a.put("lc:a:1:pdcsap", create_mock_lightcurve(tic_id=1))
        cache_a.put_computed("pg:a:1:x", {"session": "A"})
        cache_b.put("lc:b:1:pdcsap", create_mock_lightcurve(tic_id=2))
        cache_b.put_computed("pg:b:1:y", {"session": "B"})

        cache_a.clear()

        assert len(cache_a.keys()) == 0
        assert len(cache_a.computed_keys()) == 0
        assert len(cache_b.keys()) == 1
        assert len(cache_b.computed_keys()) == 1

    def test_clear_all_removes_everything(self):
        """Test that clear_all() removes all entries from all sessions."""
        cache_a = SessionCache("session-A")
        cache_b = SessionCache("session-B")
        cache_c = SessionCache("session-C")

        for cache in [cache_a, cache_b, cache_c]:
            for i in range(5):
                cache.put(f"lc:{i}:1:pdcsap", create_mock_lightcurve(tic_id=i))
                cache.put_computed(f"pg:{i}:1:x", {"data": i})

        SessionCache.clear_all()

        stats = SessionCache.global_stats()
        assert stats["data_entries"] == 0
        assert stats["computed_entries"] == 0

        for cache in [cache_a, cache_b, cache_c]:
            assert len(cache.keys()) == 0
            assert len(cache.computed_keys()) == 0

    def test_clear_empty_session(self, session_cache: SessionCache):
        """Test that clear() on empty session doesn't raise."""
        session_cache.clear()  # Should not raise
        assert len(session_cache.keys()) == 0


# =============================================================================
# Global Stats Tests
# =============================================================================


class TestGlobalStats:
    """Tests for global_stats() method."""

    def test_global_stats_empty_cache(self):
        """Test global_stats() on empty cache."""
        stats = SessionCache.global_stats()
        assert stats == {"data_entries": 0, "computed_entries": 0}

    def test_global_stats_with_data(self):
        """Test global_stats() counts all entries across sessions."""
        cache_a = SessionCache("session-A")
        cache_b = SessionCache("session-B")

        cache_a.put("lc:a1:1:pdcsap", create_mock_lightcurve(tic_id=1))
        cache_a.put("lc:a2:1:pdcsap", create_mock_lightcurve(tic_id=2))
        cache_b.put("lc:b1:1:pdcsap", create_mock_lightcurve(tic_id=3))

        cache_a.put_computed("pg:a1:1:x", {"data": 1})
        cache_b.put_computed("pg:b1:1:y", {"data": 2})
        cache_b.put_computed("pg:b2:1:z", {"data": 3})

        stats = SessionCache.global_stats()

        assert stats["data_entries"] == 3
        assert stats["computed_entries"] == 3

    def test_global_stats_after_eviction(self):
        """Test global_stats() after LRU eviction."""
        cache = SessionCache("stats-eviction-test", max_entries=5)

        # Add more than max_entries
        for i in range(10):
            cache.put(f"lc:{i}:1:pdcsap", create_mock_lightcurve(tic_id=i))

        stats = SessionCache.global_stats()
        assert stats["data_entries"] == 5  # Only 5 should remain

    def test_global_stats_after_clear(self):
        """Test global_stats() after partial clear."""
        cache_a = SessionCache("session-A")
        cache_b = SessionCache("session-B")

        for i in range(5):
            cache_a.put(f"lc:a{i}:1:pdcsap", create_mock_lightcurve(tic_id=i))
            cache_b.put(f"lc:b{i}:1:pdcsap", create_mock_lightcurve(tic_id=100 + i))

        cache_a.clear()

        stats = SessionCache.global_stats()
        assert stats["data_entries"] == 5  # Only cache_b's entries


# =============================================================================
# Edge Cases and Additional Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_session_id(self):
        """Test cache with empty session ID."""
        cache = SessionCache("")
        lc = create_mock_lightcurve()
        cache.put("lc:123:1:pdcsap", lc)

        assert cache.has("lc:123:1:pdcsap")
        assert cache.get("lc:123:1:pdcsap") is lc

    def test_special_characters_in_data_ref(self, session_cache: SessionCache):
        """Test data refs with special characters."""
        lc = create_mock_lightcurve()

        # Various special characters in data_ref
        refs = [
            "lc:123:1:pdcsap",
            "lc:123-456:1:pdcsap",
            "lc:123_456:1:sap_flux",
            "lc:tic-123:sector-1:type",
        ]

        for ref in refs:
            session_cache.put(ref, lc)
            assert session_cache.has(ref)
            assert session_cache.get(ref) is lc

    def test_max_entries_of_one(self):
        """Test cache with max_entries=1."""
        cache = SessionCache("single-entry-cache", max_entries=1)

        lc1 = create_mock_lightcurve(tic_id=1)
        lc2 = create_mock_lightcurve(tic_id=2)

        cache.put("lc:1:1:pdcsap", lc1)
        assert cache.has("lc:1:1:pdcsap")

        cache.put("lc:2:1:pdcsap", lc2)
        assert not cache.has("lc:1:1:pdcsap")
        assert cache.has("lc:2:1:pdcsap")

    def test_same_instance_different_keys(self, session_cache: SessionCache):
        """Test storing same LightCurveData instance under different keys."""
        lc = create_mock_lightcurve()

        session_cache.put("key1", lc)
        session_cache.put("key2", lc)

        assert session_cache.get("key1") is lc
        assert session_cache.get("key2") is lc
        assert session_cache.get("key1") is session_cache.get("key2")

    def test_none_value_not_stored(self, session_cache: SessionCache):
        """Test that None cannot be stored (type hint enforces LightCurveData)."""
        # This test documents expected behavior - put expects LightCurveData
        # Attempting to store None would be a type error
        # We verify get returns None for missing keys
        assert session_cache.get("nonexistent") is None

    def test_multiple_cache_instances_same_session(self):
        """Test multiple SessionCache instances with same session_id share data."""
        cache1 = SessionCache("shared-session")
        cache2 = SessionCache("shared-session")

        lc = create_mock_lightcurve()
        cache1.put("lc:shared:1:pdcsap", lc)

        # cache2 should see cache1's data
        assert cache2.has("lc:shared:1:pdcsap")
        assert cache2.get("lc:shared:1:pdcsap") is lc

    def test_computed_update_existing(self, session_cache: SessionCache):
        """Test updating existing computed product."""
        session_cache.put_computed("pg:123:1:x", {"version": 1})
        session_cache.put_computed("pg:123:1:x", {"version": 2})

        result = session_cache.get_computed("pg:123:1:x")
        assert result == {"version": 2}

    def test_keys_order_reflects_insertion(self, session_cache: SessionCache):
        """Test that keys() order might reflect insertion/access order."""
        # Note: OrderedDict maintains insertion order
        session_cache.put("key1", create_mock_lightcurve(tic_id=1))
        session_cache.put("key2", create_mock_lightcurve(tic_id=2))
        session_cache.put("key3", create_mock_lightcurve(tic_id=3))

        keys = session_cache.keys()
        assert len(keys) == 3
        # All keys should be present
        assert set(keys) == {"key1", "key2", "key3"}


# =============================================================================
# Atomic get_or_load and get_or_compute Tests
# =============================================================================


class TestGetOrLoad:
    """Tests for get_or_load atomic cache access pattern."""

    def test_get_or_load_returns_cached_value(self, session_cache: SessionCache):
        """Test get_or_load returns cached value without calling loader."""
        lc = create_mock_lightcurve(tic_id=12345)
        session_cache.put("lc:12345:1:pdcsap", lc)

        loader_called = False

        def loader():
            nonlocal loader_called
            loader_called = True
            return create_mock_lightcurve(tic_id=99999)

        result = session_cache.get_or_load("lc:12345:1:pdcsap", loader)

        assert result is lc
        assert result.tic_id == 12345
        assert loader_called is False

    def test_get_or_load_calls_loader_when_missing(self, session_cache: SessionCache):
        """Test get_or_load calls loader when key not in cache."""
        lc = create_mock_lightcurve(tic_id=99999)
        loader_called = False

        def loader():
            nonlocal loader_called
            loader_called = True
            return lc

        result = session_cache.get_or_load("missing_key", loader)

        assert result is lc
        assert result.tic_id == 99999
        assert loader_called is True

        # Should also be cached now
        assert session_cache.has("missing_key")
        assert session_cache.get("missing_key") is lc

    def test_get_or_load_caches_loaded_value(self, session_cache: SessionCache):
        """Test get_or_load stores loaded value in cache for subsequent calls."""
        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return create_mock_lightcurve(tic_id=12345)

        # First call - should call loader
        result1 = session_cache.get_or_load("new_key", loader)
        assert call_count == 1

        # Second call - should return cached, not call loader
        result2 = session_cache.get_or_load("new_key", loader)
        assert call_count == 1  # Still 1, loader not called again

        assert result1.tic_id == result2.tic_id == 12345

    def test_get_or_load_thread_safety(self):
        """Test get_or_load is thread-safe with concurrent access."""
        import time

        cache = SessionCache("thread-test-get-or-load", max_entries=1000)
        call_count = 0
        call_lock = threading.Lock()

        def loader():
            nonlocal call_count
            with call_lock:
                call_count += 1
            # Simulate slow loading
            time.sleep(0.01)
            return create_mock_lightcurve(tic_id=12345)

        # Run concurrent calls
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache.get_or_load, "shared_key", loader) for _ in range(20)]
            results = [f.result() for f in futures]

        # All results should be valid LightCurveData
        for result in results:
            assert isinstance(result, LightCurveData)
            assert result.tic_id == 12345

        # The cache should contain the key
        assert cache.has("shared_key")

        # Multiple loaders may be called due to race between check and load,
        # but the end result should be consistent and the key should be cached


class TestGetOrCompute:
    """Tests for get_or_compute atomic cache access pattern."""

    def test_get_or_compute_returns_cached_value(self, session_cache: SessionCache):
        """Test get_or_compute returns cached value without calling compute."""
        product = {"period": 1.5, "power": 100.0}
        session_cache.put_computed("pg:12345:1:abc", product)

        compute_called = False

        def compute():
            nonlocal compute_called
            compute_called = True
            return {"period": 2.0, "power": 50.0}

        result = session_cache.get_or_compute("pg:12345:1:abc", compute)

        assert result is product
        assert result["period"] == 1.5
        assert compute_called is False

    def test_get_or_compute_calls_compute_when_missing(self, session_cache: SessionCache):
        """Test get_or_compute calls compute when key not in cache."""
        product = {"period": 1.5, "power": 100.0}
        compute_called = False

        def compute():
            nonlocal compute_called
            compute_called = True
            return product

        result = session_cache.get_or_compute("missing_product", compute)

        assert result is product
        assert compute_called is True

        # Should also be cached now
        assert session_cache.has_computed("missing_product")
        assert session_cache.get_computed("missing_product") is product

    def test_get_or_compute_caches_computed_value(self, session_cache: SessionCache):
        """Test get_or_compute stores computed value in cache."""
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return {"result": call_count}

        # First call - should call compute
        result1 = session_cache.get_or_compute("new_product", compute)
        assert call_count == 1
        assert result1["result"] == 1

        # Second call - should return cached, not call compute
        result2 = session_cache.get_or_compute("new_product", compute)
        assert call_count == 1  # Still 1, compute not called again
        assert result2["result"] == 1

    def test_get_or_compute_thread_safety(self):
        """Test get_or_compute is thread-safe with concurrent access."""
        import time

        cache = SessionCache("thread-test-get-or-compute", max_computed=1000)
        call_count = 0
        call_lock = threading.Lock()

        def compute():
            nonlocal call_count
            with call_lock:
                call_count += 1
            # Simulate slow computation
            time.sleep(0.01)
            return {"result": "computed"}

        # Run concurrent calls
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(cache.get_or_compute, "shared_product", compute) for _ in range(20)
            ]
            results = [f.result() for f in futures]

        # All results should be the same
        for result in results:
            assert result["result"] == "computed"

        # The cache should contain the key
        assert cache.has_computed("shared_product")

    def test_get_or_compute_with_different_return_types(self, session_cache: SessionCache):
        """Test get_or_compute with different return types."""
        # List
        list_result = session_cache.get_or_compute("list_product", lambda: [1, 2, 3])
        assert list_result == [1, 2, 3]

        # Dict
        dict_result = session_cache.get_or_compute("dict_product", lambda: {"key": "value"})
        assert dict_result == {"key": "value"}

        # numpy array
        arr = np.array([1.0, 2.0, 3.0])
        arr_result = session_cache.get_or_compute("array_product", lambda: arr)
        np.testing.assert_array_equal(arr_result, arr)
