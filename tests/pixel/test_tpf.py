"""Tests for bittr_tess_vetter.pixel.tpf module.

Tests the TPF (Target Pixel File) handling including:
- TPFRef creation and validation
- TPFRef string serialization/parsing
- TPFCache operations (get, put, has, remove, clear)
- TPFHandler abstract interface
- CachedTPFHandler wrapper behavior
- TPFNotFoundError error handling
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bittr_tess_vetter.errors import ErrorType
from bittr_tess_vetter.pixel import (
    CachedTPFHandler,
    TPFCache,
    TPFData,
    TPFHandler,
    TPFNotFoundError,
    TPFRef,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_ref() -> TPFRef:
    """Create a sample TPFRef for testing."""
    return TPFRef(tic_id=123456789, sector=15, camera=2, ccd=3)


@pytest.fixture
def sample_data(sample_ref: TPFRef) -> TPFData:
    """Create sample TPF data (time + flux cube) for testing."""
    rng = np.random.default_rng(42)
    flux = rng.random((100, 11, 11)).astype(np.float32)
    time = np.linspace(0.0, 1.0, flux.shape[0]).astype(np.float64)
    return TPFData(ref=sample_ref, time=time, flux=flux)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_path = tmp_path / "tpf_cache"
    cache_path.mkdir()
    return cache_path


@pytest.fixture
def cache(cache_dir: Path) -> TPFCache:
    """Create a TPFCache instance."""
    return TPFCache(cache_dir=cache_dir)


# =============================================================================
# TPFRef Tests - Creation and Validation
# =============================================================================


class TestTPFRefCreation:
    """Tests for TPFRef creation and validation."""

    def test_valid_creation(self) -> None:
        """Can create a valid TPFRef."""
        ref = TPFRef(tic_id=123456789, sector=15, camera=2, ccd=3)

        assert ref.tic_id == 123456789
        assert ref.sector == 15
        assert ref.camera == 2
        assert ref.ccd == 3

    def test_boundary_values(self) -> None:
        """TPFRef accepts boundary values for camera and CCD."""
        # Minimum valid values
        ref_min = TPFRef(tic_id=1, sector=1, camera=1, ccd=1)
        assert ref_min.camera == 1
        assert ref_min.ccd == 1

        # Maximum valid values
        ref_max = TPFRef(tic_id=1, sector=1, camera=4, ccd=4)
        assert ref_max.camera == 4
        assert ref_max.ccd == 4

    def test_large_tic_id(self) -> None:
        """TPFRef accepts large TIC IDs."""
        ref = TPFRef(tic_id=9999999999, sector=1, camera=1, ccd=1)
        assert ref.tic_id == 9999999999

    def test_large_sector(self) -> None:
        """TPFRef accepts large sector numbers."""
        ref = TPFRef(tic_id=1, sector=999, camera=1, ccd=1)
        assert ref.sector == 999

    def test_invalid_tic_id_zero(self) -> None:
        """TPFRef rejects zero tic_id."""
        with pytest.raises(ValueError, match="tic_id must be positive"):
            TPFRef(tic_id=0, sector=1, camera=1, ccd=1)

    def test_invalid_tic_id_negative(self) -> None:
        """TPFRef rejects negative tic_id."""
        with pytest.raises(ValueError, match="tic_id must be positive"):
            TPFRef(tic_id=-1, sector=1, camera=1, ccd=1)

    def test_invalid_sector_zero(self) -> None:
        """TPFRef rejects zero sector."""
        with pytest.raises(ValueError, match="sector must be positive"):
            TPFRef(tic_id=1, sector=0, camera=1, ccd=1)

    def test_invalid_sector_negative(self) -> None:
        """TPFRef rejects negative sector."""
        with pytest.raises(ValueError, match="sector must be positive"):
            TPFRef(tic_id=1, sector=-1, camera=1, ccd=1)

    def test_invalid_camera_zero(self) -> None:
        """TPFRef rejects zero camera."""
        with pytest.raises(ValueError, match="camera must be 1-4"):
            TPFRef(tic_id=1, sector=1, camera=0, ccd=1)

    def test_invalid_camera_five(self) -> None:
        """TPFRef rejects camera > 4."""
        with pytest.raises(ValueError, match="camera must be 1-4"):
            TPFRef(tic_id=1, sector=1, camera=5, ccd=1)

    def test_invalid_ccd_zero(self) -> None:
        """TPFRef rejects zero ccd."""
        with pytest.raises(ValueError, match="ccd must be 1-4"):
            TPFRef(tic_id=1, sector=1, camera=1, ccd=0)

    def test_invalid_ccd_five(self) -> None:
        """TPFRef rejects ccd > 4."""
        with pytest.raises(ValueError, match="ccd must be 1-4"):
            TPFRef(tic_id=1, sector=1, camera=1, ccd=5)

    def test_frozen_immutability(self) -> None:
        """TPFRef fields cannot be modified after creation."""
        ref = TPFRef(tic_id=123456789, sector=15, camera=2, ccd=3)

        with pytest.raises(AttributeError):
            ref.tic_id = 999  # type: ignore[misc]


# =============================================================================
# TPFRef Tests - String Serialization
# =============================================================================


class TestTPFRefSerialization:
    """Tests for TPFRef string serialization and parsing."""

    def test_to_string(self, sample_ref: TPFRef) -> None:
        """to_string produces correct format."""
        result = sample_ref.to_string()
        assert result == "tpf:123456789:15:2:3"

    def test_str_method(self, sample_ref: TPFRef) -> None:
        """__str__ produces same result as to_string."""
        assert str(sample_ref) == sample_ref.to_string()

    def test_from_string_valid(self) -> None:
        """from_string parses valid reference strings."""
        ref = TPFRef.from_string("tpf:123456789:15:2:3")

        assert ref.tic_id == 123456789
        assert ref.sector == 15
        assert ref.camera == 2
        assert ref.ccd == 3

    def test_roundtrip(self, sample_ref: TPFRef) -> None:
        """to_string and from_string are inverse operations."""
        string = sample_ref.to_string()
        parsed = TPFRef.from_string(string)
        assert parsed == sample_ref

    @pytest.mark.parametrize(
        "ref_str",
        [
            "tpf:1:1:1:1",
            "tpf:9999999999:999:4:4",
            "tpf:123456789:15:2:3",
        ],
    )
    def test_from_string_various_valid(self, ref_str: str) -> None:
        """from_string handles various valid formats."""
        ref = TPFRef.from_string(ref_str)
        assert ref.to_string() == ref_str

    @pytest.mark.parametrize(
        "invalid_str,error_match",
        [
            ("", "Invalid TPF reference format"),
            ("tpf", "Invalid TPF reference format"),
            ("tpf:123456789", "Invalid TPF reference format"),
            ("tpf:123456789:15", "Invalid TPF reference format"),
            ("tpf:123456789:15:2", "Invalid TPF reference format"),
            ("tpf:123456789:15:2:3:4", "Invalid TPF reference format"),
            ("lc:123456789:15:2:3", "Invalid TPF reference prefix"),
            ("TPF:123456789:15:2:3", "Invalid TPF reference prefix"),
            ("tpf:abc:15:2:3", "Invalid tic_id"),
            ("tpf:123456789:xyz:2:3", "Invalid sector"),
            ("tpf:123456789:15:a:3", "Invalid camera"),
            ("tpf:123456789:15:2:b", "Invalid ccd"),
            ("tpf::15:2:3", "Invalid tic_id"),
            ("tpf:123456789::2:3", "Invalid sector"),
        ],
    )
    def test_from_string_invalid(self, invalid_str: str, error_match: str) -> None:
        """from_string rejects invalid reference strings."""
        with pytest.raises(ValueError, match=error_match):
            TPFRef.from_string(invalid_str)


# =============================================================================
# TPFRef Tests - Equality and Hashing
# =============================================================================


class TestTPFRefEquality:
    """Tests for TPFRef equality and hashing."""

    def test_equality_same_values(self) -> None:
        """TPFRefs with same values are equal."""
        ref1 = TPFRef(tic_id=123, sector=1, camera=2, ccd=3)
        ref2 = TPFRef(tic_id=123, sector=1, camera=2, ccd=3)
        assert ref1 == ref2

    def test_inequality_different_tic_id(self) -> None:
        """TPFRefs with different tic_id are not equal."""
        ref1 = TPFRef(tic_id=123, sector=1, camera=2, ccd=3)
        ref2 = TPFRef(tic_id=456, sector=1, camera=2, ccd=3)
        assert ref1 != ref2

    def test_inequality_different_sector(self) -> None:
        """TPFRefs with different sector are not equal."""
        ref1 = TPFRef(tic_id=123, sector=1, camera=2, ccd=3)
        ref2 = TPFRef(tic_id=123, sector=2, camera=2, ccd=3)
        assert ref1 != ref2

    def test_hashable(self) -> None:
        """TPFRef can be used as dictionary key."""
        ref1 = TPFRef(tic_id=123, sector=1, camera=2, ccd=3)
        ref2 = TPFRef(tic_id=456, sector=1, camera=2, ccd=3)

        d = {ref1: "data1", ref2: "data2"}
        assert d[ref1] == "data1"
        assert d[ref2] == "data2"

    def test_set_membership(self) -> None:
        """TPFRef can be used in sets."""
        ref1 = TPFRef(tic_id=123, sector=1, camera=2, ccd=3)
        ref2 = TPFRef(tic_id=123, sector=1, camera=2, ccd=3)
        ref3 = TPFRef(tic_id=456, sector=1, camera=2, ccd=3)

        s = {ref1, ref2, ref3}
        assert len(s) == 2  # ref1 and ref2 are equal


# =============================================================================
# TPFCache Tests
# =============================================================================


class TestTPFCache:
    """Tests for TPFCache operations."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """TPFCache creates cache directory if it doesn't exist."""
        cache_path = tmp_path / "new_cache"
        assert not cache_path.exists()

        cache = TPFCache(cache_dir=cache_path)
        assert cache_path.exists()
        assert cache.cache_dir == cache_path

    def test_cache_dir_property(self, cache: TPFCache, cache_dir: Path) -> None:
        """cache_dir property returns the cache directory."""
        assert cache.cache_dir == cache_dir

    def test_has_empty_cache(self, cache: TPFCache, sample_ref: TPFRef) -> None:
        """has returns False for empty cache."""
        assert not cache.has(sample_ref)

    def test_get_empty_cache(self, cache: TPFCache, sample_ref: TPFRef) -> None:
        """get returns None for empty cache."""
        assert cache.get(sample_ref) is None

    def test_put_and_get(
        self, cache: TPFCache, sample_ref: TPFRef, sample_data: TPFData
    ) -> None:
        """put stores data that can be retrieved with get."""
        cache.put(sample_data)

        retrieved = cache.get(sample_ref)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved.flux, sample_data.flux)
        np.testing.assert_array_equal(retrieved.time, sample_data.time)

    def test_put_updates_has(
        self, cache: TPFCache, sample_ref: TPFRef, sample_data: TPFData
    ) -> None:
        """put causes has to return True."""
        assert not cache.has(sample_ref)
        cache.put(sample_data)
        assert cache.has(sample_ref)

    def test_put_preserves_dtype(self, cache: TPFCache, sample_ref: TPFRef) -> None:
        """put preserves numpy array dtype."""
        flux = np.array([[[1.0]]], dtype=np.float64)
        time = np.array([0.0], dtype=np.float64)
        cache.put(TPFData(ref=sample_ref, time=time, flux=flux))

        retrieved = cache.get(sample_ref)
        assert retrieved is not None
        assert retrieved.flux.dtype == np.float64

    def test_put_preserves_shape(
        self, cache: TPFCache, sample_ref: TPFRef, sample_data: TPFData
    ) -> None:
        """put preserves numpy array shape."""
        cache.put(sample_data)

        retrieved = cache.get(sample_ref)
        assert retrieved is not None
        assert retrieved.flux.shape == sample_data.flux.shape
        assert retrieved.time.shape == sample_data.time.shape

    def test_remove_existing(
        self, cache: TPFCache, sample_ref: TPFRef, sample_data: TPFData
    ) -> None:
        """remove deletes cached entry and returns True."""
        cache.put(sample_data)
        assert cache.has(sample_ref)

        result = cache.remove(sample_ref)
        assert result is True
        assert not cache.has(sample_ref)
        assert cache.get(sample_ref) is None

    def test_remove_nonexistent(self, cache: TPFCache, sample_ref: TPFRef) -> None:
        """remove returns False for nonexistent entry."""
        result = cache.remove(sample_ref)
        assert result is False

    def test_clear_empty(self, cache: TPFCache) -> None:
        """clear on empty cache returns 0."""
        result = cache.clear()
        assert result == 0

    def test_clear_removes_all(self, cache: TPFCache, sample_data: TPFData) -> None:
        """clear removes all cached entries."""
        refs = [
            TPFRef(tic_id=1, sector=1, camera=1, ccd=1),
            TPFRef(tic_id=2, sector=2, camera=2, ccd=2),
            TPFRef(tic_id=3, sector=3, camera=3, ccd=3),
        ]

        for ref in refs:
            cache.put(TPFData(ref=ref, time=sample_data.time, flux=sample_data.flux))

        for ref in refs:
            assert cache.has(ref)

        removed = cache.clear()
        assert removed == 3

        for ref in refs:
            assert not cache.has(ref)

    def test_multiple_refs_isolated(
        self, cache: TPFCache, sample_data: TPFData
    ) -> None:
        """Different TPFRefs are cached independently."""
        ref1 = TPFRef(tic_id=1, sector=1, camera=1, ccd=1)
        ref2 = TPFRef(tic_id=2, sector=2, camera=2, ccd=2)

        data1 = TPFData(
            ref=ref1,
            time=np.array([0.0], dtype=np.float64),
            flux=np.array([[[1.0]]], dtype=np.float64),
        )
        data2 = TPFData(
            ref=ref2,
            time=np.array([0.0], dtype=np.float64),
            flux=np.array([[[2.0]]], dtype=np.float64),
        )

        cache.put(data1)
        cache.put(data2)

        retrieved1 = cache.get(ref1)
        retrieved2 = cache.get(ref2)

        assert retrieved1 is not None
        assert retrieved2 is not None
        np.testing.assert_array_equal(retrieved1.flux, data1.flux)
        np.testing.assert_array_equal(retrieved2.flux, data2.flux)

    def test_get_corrupted_file(self, cache: TPFCache, sample_ref: TPFRef, cache_dir: Path) -> None:
        """get returns None for corrupted cache files."""
        # Create a corrupted cache file
        filename = (
            f"tpf_{sample_ref.tic_id}_{sample_ref.sector}_{sample_ref.camera}_{sample_ref.ccd}.npz"
        )
        cache_file = cache_dir / filename
        cache_file.write_text("corrupted data")

        # has returns True (file exists) but get returns None (corrupted)
        assert cache.has(sample_ref)
        assert cache.get(sample_ref) is None


# =============================================================================
# TPFHandler Tests
# =============================================================================


class MockTPFHandler(TPFHandler):
    """Mock TPFHandler for testing."""

    def __init__(self, data: dict[TPFRef, TPFData] | None = None) -> None:
        self._data = data or {}
        self.fetch_count = 0

    def fetch(self, ref: TPFRef) -> TPFData:
        self.fetch_count += 1
        if ref in self._data:
            return self._data[ref]
        raise TPFNotFoundError(ref)


class TestTPFHandler:
    """Tests for TPFHandler abstract interface."""

    def test_mock_handler_returns_data(
        self, sample_ref: TPFRef, sample_data: TPFData
    ) -> None:
        """MockTPFHandler returns configured data."""
        handler = MockTPFHandler(data={sample_ref: sample_data})
        result = handler.fetch(sample_ref)
        np.testing.assert_array_equal(result.flux, sample_data.flux)
        np.testing.assert_array_equal(result.time, sample_data.time)

    def test_mock_handler_raises_not_found(self, sample_ref: TPFRef) -> None:
        """MockTPFHandler raises TPFNotFoundError for missing refs."""
        handler = MockTPFHandler()
        with pytest.raises(TPFNotFoundError):
            handler.fetch(sample_ref)


# =============================================================================
# TPFNotFoundError Tests
# =============================================================================


class TestTPFNotFoundError:
    """Tests for TPFNotFoundError."""

    def test_error_creation(self, sample_ref: TPFRef) -> None:
        """TPFNotFoundError can be created with a TPFRef."""
        error = TPFNotFoundError(sample_ref)

        assert error.ref == sample_ref
        assert sample_ref.to_string() in str(error)

    def test_error_has_correct_type(self, sample_ref: TPFRef) -> None:
        """TPFNotFoundError has TPF_MISSING error type."""
        error = TPFNotFoundError(sample_ref)
        assert error.error_type == ErrorType.CACHE_MISS

    def test_error_context_contains_ref_info(self, sample_ref: TPFRef) -> None:
        """TPFNotFoundError context contains ref information."""
        error = TPFNotFoundError(sample_ref)

        assert error.context["ref"] == sample_ref.to_string()
        assert error.context["tic_id"] == sample_ref.tic_id
        assert error.context["sector"] == sample_ref.sector
        assert error.context["camera"] == sample_ref.camera
        assert error.context["ccd"] == sample_ref.ccd

    def test_custom_message(self, sample_ref: TPFRef) -> None:
        """TPFNotFoundError accepts custom message."""
        custom_msg = "Custom error message"
        error = TPFNotFoundError(sample_ref, message=custom_msg)

        assert error.message == custom_msg

    def test_default_message(self, sample_ref: TPFRef) -> None:
        """TPFNotFoundError has default message with ref."""
        error = TPFNotFoundError(sample_ref)

        assert "TPF not found" in error.message
        assert sample_ref.to_string() in error.message

    def test_can_be_raised_and_caught(self, sample_ref: TPFRef) -> None:
        """TPFNotFoundError can be raised and caught."""
        with pytest.raises(TPFNotFoundError) as exc_info:
            raise TPFNotFoundError(sample_ref)

        assert exc_info.value.ref == sample_ref


# =============================================================================
# CachedTPFHandler Tests
# =============================================================================


class TestCachedTPFHandler:
    """Tests for CachedTPFHandler."""

    def test_fetch_from_handler_when_not_cached(
        self,
        cache: TPFCache,
        sample_ref: TPFRef,
        sample_data: TPFData,
    ) -> None:
        """CachedTPFHandler fetches from handler when not cached."""
        handler = MockTPFHandler(data={sample_ref: sample_data})
        cached_handler = CachedTPFHandler(cache=cache, handler=handler)

        result = cached_handler.fetch(sample_ref)

        np.testing.assert_array_equal(result.flux, sample_data.flux)
        np.testing.assert_array_equal(result.time, sample_data.time)
        assert handler.fetch_count == 1

    def test_caches_fetched_data(
        self,
        cache: TPFCache,
        sample_ref: TPFRef,
        sample_data: TPFData,
    ) -> None:
        """CachedTPFHandler caches data after fetching."""
        handler = MockTPFHandler(data={sample_ref: sample_data})
        cached_handler = CachedTPFHandler(cache=cache, handler=handler)

        cached_handler.fetch(sample_ref)

        assert cache.has(sample_ref)
        cached = cache.get(sample_ref)
        assert cached is not None
        np.testing.assert_array_equal(cached.flux, sample_data.flux)
        np.testing.assert_array_equal(cached.time, sample_data.time)

    def test_returns_from_cache_when_available(
        self,
        cache: TPFCache,
        sample_ref: TPFRef,
        sample_data: TPFData,
    ) -> None:
        """CachedTPFHandler returns from cache without calling handler."""
        handler = MockTPFHandler(data={sample_ref: sample_data})
        cached_handler = CachedTPFHandler(cache=cache, handler=handler)

        # Pre-populate cache
        cache.put(sample_data)

        result = cached_handler.fetch(sample_ref)

        np.testing.assert_array_equal(result.flux, sample_data.flux)
        np.testing.assert_array_equal(result.time, sample_data.time)
        assert handler.fetch_count == 0  # Handler never called

    def test_second_fetch_uses_cache(
        self,
        cache: TPFCache,
        sample_ref: TPFRef,
        sample_data: TPFData,
    ) -> None:
        """Second fetch of same ref uses cache."""
        handler = MockTPFHandler(data={sample_ref: sample_data})
        cached_handler = CachedTPFHandler(cache=cache, handler=handler)

        # First fetch
        cached_handler.fetch(sample_ref)
        assert handler.fetch_count == 1

        # Second fetch
        cached_handler.fetch(sample_ref)
        assert handler.fetch_count == 1  # Still 1, used cache

    def test_propagates_not_found_error(
        self,
        cache: TPFCache,
        sample_ref: TPFRef,
    ) -> None:
        """CachedTPFHandler propagates TPFNotFoundError from handler."""
        handler = MockTPFHandler()  # Empty, will raise error
        cached_handler = CachedTPFHandler(cache=cache, handler=handler)

        with pytest.raises(TPFNotFoundError):
            cached_handler.fetch(sample_ref)

    def test_cache_property(self, cache: TPFCache) -> None:
        """cache property returns the cache instance."""
        handler = MockTPFHandler()
        cached_handler = CachedTPFHandler(cache=cache, handler=handler)

        assert cached_handler.cache is cache

    def test_handler_property(self, cache: TPFCache) -> None:
        """handler property returns the underlying handler."""
        handler = MockTPFHandler()
        cached_handler = CachedTPFHandler(cache=cache, handler=handler)

        assert cached_handler.handler is handler


# =============================================================================
# Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports and exports."""

    def test_import_from_pixel_package(self) -> None:
        """Can import from bittr_tess_vetter.pixel package."""
        from bittr_tess_vetter.pixel import (
            CachedTPFHandler,
            TPFCache,
            TPFHandler,
            TPFNotFoundError,
            TPFRef,
        )

        assert TPFRef is not None
        assert TPFCache is not None
        assert TPFHandler is not None
        assert CachedTPFHandler is not None
        assert TPFNotFoundError is not None

    def test_import_from_tpf_module(self) -> None:
        """Can import from bittr_tess_vetter.pixel.tpf module."""
        from bittr_tess_vetter.pixel.tpf import (
            CachedTPFHandler,
            TPFCache,
            TPFHandler,
            TPFNotFoundError,
            TPFRef,
        )

        assert TPFRef is not None
        assert TPFCache is not None
        assert TPFHandler is not None
        assert CachedTPFHandler is not None
        assert TPFNotFoundError is not None
