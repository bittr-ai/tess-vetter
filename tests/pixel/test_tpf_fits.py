"""Tests for tess_vetter.pixel.tpf_fits module.

Tests the FITS-preserving TPF handling including:
- TPFFitsRef creation and validation
- TPFFitsRef string serialization/parsing
- TPFFitsCache operations (get, put, has, remove, clear)
- Sidecar JSON metadata handling
- WCS checksum computation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

from tess_vetter.errors import ErrorType
from tess_vetter.pixel.tpf_fits import (
    VALID_AUTHORS,
    TPFFitsCache,
    TPFFitsData,
    TPFFitsNotFoundError,
    TPFFitsRef,
    _compute_wcs_checksum,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def _make_test_wcs(
    crval: tuple[float, float] = (120.0, -50.0),
    pixel_scale_deg: float = 21.0 / 3600.0,
    shape: tuple[int, int] = (11, 11),
) -> WCS:
    """Create a test WCS object for TESS-like data.

    Args:
        crval: (RA, Dec) reference coordinates in degrees.
        pixel_scale_deg: Pixel scale in degrees per pixel.
        shape: (n_rows, n_cols) stamp shape.

    Returns:
        Configured WCS object.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(shape[1] + 1) / 2, (shape[0] + 1) / 2]
    wcs.wcs.crval = crval
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@pytest.fixture
def sample_ref() -> TPFFitsRef:
    """Create a sample TPFFitsRef for testing."""
    return TPFFitsRef(tic_id=123456789, sector=15, author="spoc")


@pytest.fixture
def sample_wcs() -> WCS:
    """Create a sample WCS for testing."""
    return _make_test_wcs()


@pytest.fixture
def sample_data(sample_ref: TPFFitsRef, sample_wcs: WCS) -> TPFFitsData:
    """Create sample TPFFitsData for testing."""
    rng = np.random.default_rng(42)
    n_cadences, n_rows, n_cols = 100, 11, 11
    return TPFFitsData(
        ref=sample_ref,
        time=np.linspace(2458000.0, 2458027.0, n_cadences).astype(np.float64),
        flux=rng.random((n_cadences, n_rows, n_cols)).astype(np.float64) * 1000,
        flux_err=rng.random((n_cadences, n_rows, n_cols)).astype(np.float64) * 10,
        wcs=sample_wcs,
        aperture_mask=np.ones((n_rows, n_cols), dtype=np.int32),
        quality=np.zeros(n_cadences, dtype=np.int32),
        camera=2,
        ccd=3,
        meta={"RA_OBJ": 120.0, "DEC_OBJ": -50.0},
    )


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_path = tmp_path / "tpf_fits_cache"
    cache_path.mkdir()
    return cache_path


@pytest.fixture
def cache(cache_dir: Path) -> TPFFitsCache:
    """Create a TPFFitsCache instance."""
    return TPFFitsCache(cache_dir=cache_dir)


# =============================================================================
# TPFFitsRef Tests - Creation and Validation
# =============================================================================


class TestTPFFitsRefCreation:
    """Tests for TPFFitsRef creation and validation."""

    def test_valid_creation(self) -> None:
        """Can create a valid TPFFitsRef."""
        ref = TPFFitsRef(tic_id=123456789, sector=15, author="spoc")

        assert ref.tic_id == 123456789
        assert ref.sector == 15
        assert ref.author == "spoc"

    @pytest.mark.parametrize("author", list(VALID_AUTHORS))
    def test_valid_authors(self, author: str) -> None:
        """TPFFitsRef accepts all valid authors."""
        ref = TPFFitsRef(tic_id=1, sector=1, author=author)
        assert ref.author == author.lower()

    def test_author_case_insensitive(self) -> None:
        """TPFFitsRef normalizes author to lowercase."""
        ref = TPFFitsRef(tic_id=1, sector=1, author="SPOC")
        assert ref.author == "spoc"

    def test_large_tic_id(self) -> None:
        """TPFFitsRef accepts large TIC IDs."""
        ref = TPFFitsRef(tic_id=9999999999, sector=1, author="spoc")
        assert ref.tic_id == 9999999999

    def test_large_sector(self) -> None:
        """TPFFitsRef accepts large sector numbers."""
        ref = TPFFitsRef(tic_id=1, sector=999, author="spoc")
        assert ref.sector == 999

    def test_invalid_tic_id_zero(self) -> None:
        """TPFFitsRef rejects zero tic_id."""
        with pytest.raises(ValueError, match="tic_id must be positive"):
            TPFFitsRef(tic_id=0, sector=1, author="spoc")

    def test_invalid_tic_id_negative(self) -> None:
        """TPFFitsRef rejects negative tic_id."""
        with pytest.raises(ValueError, match="tic_id must be positive"):
            TPFFitsRef(tic_id=-1, sector=1, author="spoc")

    def test_invalid_sector_zero(self) -> None:
        """TPFFitsRef rejects zero sector."""
        with pytest.raises(ValueError, match="sector must be positive"):
            TPFFitsRef(tic_id=1, sector=0, author="spoc")

    def test_invalid_sector_negative(self) -> None:
        """TPFFitsRef rejects negative sector."""
        with pytest.raises(ValueError, match="sector must be positive"):
            TPFFitsRef(tic_id=1, sector=-1, author="spoc")

    def test_invalid_author(self) -> None:
        """TPFFitsRef rejects invalid authors."""
        with pytest.raises(ValueError, match="author must be one of"):
            TPFFitsRef(tic_id=1, sector=1, author="invalid")

    def test_frozen_immutability(self) -> None:
        """TPFFitsRef fields cannot be modified after creation."""
        ref = TPFFitsRef(tic_id=123456789, sector=15, author="spoc")

        with pytest.raises(AttributeError):
            ref.tic_id = 999  # type: ignore[misc]


# =============================================================================
# TPFFitsRef Tests - String Serialization
# =============================================================================


class TestTPFFitsRefSerialization:
    """Tests for TPFFitsRef string serialization and parsing."""

    def test_to_string(self, sample_ref: TPFFitsRef) -> None:
        """to_string produces correct format."""
        result = sample_ref.to_string()
        assert result == "tpf_fits:123456789:15:spoc"

    def test_str_method(self, sample_ref: TPFFitsRef) -> None:
        """__str__ produces same result as to_string."""
        assert str(sample_ref) == sample_ref.to_string()

    def test_from_string_valid(self) -> None:
        """from_string parses valid reference strings."""
        ref = TPFFitsRef.from_string("tpf_fits:123456789:15:spoc")

        assert ref.tic_id == 123456789
        assert ref.sector == 15
        assert ref.author == "spoc"

    def test_roundtrip(self, sample_ref: TPFFitsRef) -> None:
        """to_string and from_string are inverse operations."""
        string = sample_ref.to_string()
        parsed = TPFFitsRef.from_string(string)
        assert parsed == sample_ref

    @pytest.mark.parametrize(
        "ref_str",
        [
            "tpf_fits:1:1:spoc",
            "tpf_fits:9999999999:999:qlp",
            "tpf_fits:123456789:15:tess-spoc",
            "tpf_fits:123456789:15:tasoc",
        ],
    )
    def test_from_string_various_valid(self, ref_str: str) -> None:
        """from_string handles various valid formats."""
        ref = TPFFitsRef.from_string(ref_str)
        assert ref.to_string() == ref_str

    @pytest.mark.parametrize(
        "invalid_str,error_match",
        [
            ("", "Invalid TPF FITS reference format"),
            ("tpf_fits", "Invalid TPF FITS reference format"),
            ("tpf_fits:123456789", "Invalid TPF FITS reference format"),
            ("tpf_fits:123456789:15", "Invalid TPF FITS reference format"),
            ("tpf_fits:123456789:15:spoc:extra", "Invalid exptime_seconds"),
            ("tpf:123456789:15:spoc", "Invalid TPF FITS reference prefix"),
            ("TPF_FITS:123456789:15:spoc", "Invalid TPF FITS reference prefix"),
            ("tpf_fits:abc:15:spoc", "Invalid tic_id"),
            ("tpf_fits:123456789:xyz:spoc", "Invalid sector"),
            ("tpf_fits::15:spoc", "Invalid tic_id"),
            ("tpf_fits:123456789::spoc", "Invalid sector"),
            ("tpf_fits:123456789:15:", "author cannot be empty"),
        ],
    )
    def test_from_string_invalid(self, invalid_str: str, error_match: str) -> None:
        """from_string rejects invalid reference strings."""
        with pytest.raises(ValueError, match=error_match):
            TPFFitsRef.from_string(invalid_str)


# =============================================================================
# TPFFitsRef Tests - Equality and Hashing
# =============================================================================


class TestTPFFitsRefEquality:
    """Tests for TPFFitsRef equality and hashing."""

    def test_equality_same_values(self) -> None:
        """TPFFitsRefs with same values are equal."""
        ref1 = TPFFitsRef(tic_id=123, sector=1, author="spoc")
        ref2 = TPFFitsRef(tic_id=123, sector=1, author="spoc")
        assert ref1 == ref2

    def test_inequality_different_tic_id(self) -> None:
        """TPFFitsRefs with different tic_id are not equal."""
        ref1 = TPFFitsRef(tic_id=123, sector=1, author="spoc")
        ref2 = TPFFitsRef(tic_id=456, sector=1, author="spoc")
        assert ref1 != ref2

    def test_inequality_different_sector(self) -> None:
        """TPFFitsRefs with different sector are not equal."""
        ref1 = TPFFitsRef(tic_id=123, sector=1, author="spoc")
        ref2 = TPFFitsRef(tic_id=123, sector=2, author="spoc")
        assert ref1 != ref2

    def test_inequality_different_author(self) -> None:
        """TPFFitsRefs with different author are not equal."""
        ref1 = TPFFitsRef(tic_id=123, sector=1, author="spoc")
        ref2 = TPFFitsRef(tic_id=123, sector=1, author="qlp")
        assert ref1 != ref2

    def test_hashable(self) -> None:
        """TPFFitsRef can be used as dictionary key."""
        ref1 = TPFFitsRef(tic_id=123, sector=1, author="spoc")
        ref2 = TPFFitsRef(tic_id=456, sector=1, author="spoc")

        d = {ref1: "data1", ref2: "data2"}
        assert d[ref1] == "data1"
        assert d[ref2] == "data2"

    def test_set_membership(self) -> None:
        """TPFFitsRef can be used in sets."""
        ref1 = TPFFitsRef(tic_id=123, sector=1, author="spoc")
        ref2 = TPFFitsRef(tic_id=123, sector=1, author="spoc")
        ref3 = TPFFitsRef(tic_id=456, sector=1, author="spoc")

        s = {ref1, ref2, ref3}
        assert len(s) == 2  # ref1 and ref2 are equal


# =============================================================================
# TPFFitsData Tests
# =============================================================================


class TestTPFFitsData:
    """Tests for TPFFitsData properties."""

    def test_n_cadences(self, sample_data: TPFFitsData) -> None:
        """n_cadences returns correct count."""
        assert sample_data.n_cadences == 100

    def test_shape(self, sample_data: TPFFitsData) -> None:
        """shape returns correct dimensions."""
        assert sample_data.shape == (100, 11, 11)

    def test_aperture_npixels(self, sample_data: TPFFitsData) -> None:
        """aperture_npixels returns correct count."""
        assert sample_data.aperture_npixels == 121  # 11x11


# =============================================================================
# TPFFitsCache Tests
# =============================================================================


class TestTPFFitsCache:
    """Tests for TPFFitsCache operations."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """TPFFitsCache creates cache directory if it doesn't exist."""
        cache_path = tmp_path / "new_cache"
        assert not cache_path.exists()

        cache = TPFFitsCache(cache_dir=cache_path)
        assert cache_path.exists()
        assert cache.cache_dir == cache_path

    def test_cache_dir_property(self, cache: TPFFitsCache, cache_dir: Path) -> None:
        """cache_dir property returns the cache directory."""
        assert cache.cache_dir == cache_dir

    def test_has_empty_cache(self, cache: TPFFitsCache, sample_ref: TPFFitsRef) -> None:
        """has returns False for empty cache."""
        assert not cache.has(sample_ref)

    def test_get_empty_cache(self, cache: TPFFitsCache, sample_ref: TPFFitsRef) -> None:
        """get returns None for empty cache."""
        assert cache.get(sample_ref) is None

    def test_put_and_get(self, cache: TPFFitsCache, sample_data: TPFFitsData) -> None:
        """put stores data that can be retrieved with get."""
        cache.put(sample_data)

        retrieved = cache.get(sample_data.ref)
        assert retrieved is not None
        assert retrieved.ref == sample_data.ref
        assert retrieved.n_cadences == sample_data.n_cadences
        assert retrieved.shape == sample_data.shape
        assert retrieved.camera == sample_data.camera
        assert retrieved.ccd == sample_data.ccd
        np.testing.assert_array_almost_equal(retrieved.time, sample_data.time)

    def test_put_updates_has(self, cache: TPFFitsCache, sample_data: TPFFitsData) -> None:
        """put causes has to return True."""
        assert not cache.has(sample_data.ref)
        cache.put(sample_data)
        assert cache.has(sample_data.ref)

    def test_put_preserves_dtype(self, cache: TPFFitsCache, sample_data: TPFFitsData) -> None:
        """put preserves numpy array dtype."""
        cache.put(sample_data)

        retrieved = cache.get(sample_data.ref)
        assert retrieved is not None
        assert retrieved.time.dtype == np.float64
        assert retrieved.flux.dtype == np.float64
        assert retrieved.quality.dtype == np.int32

    def test_put_preserves_shape(self, cache: TPFFitsCache, sample_data: TPFFitsData) -> None:
        """put preserves numpy array shape."""
        cache.put(sample_data)

        retrieved = cache.get(sample_data.ref)
        assert retrieved is not None
        assert retrieved.flux.shape == sample_data.flux.shape

    def test_put_preserves_wcs(self, cache: TPFFitsCache, sample_data: TPFFitsData) -> None:
        """put preserves WCS transformation (checksum match)."""
        original_checksum = _compute_wcs_checksum(sample_data.wcs)
        cache.put(sample_data)

        retrieved = cache.get(sample_data.ref)
        assert retrieved is not None
        retrieved_checksum = _compute_wcs_checksum(retrieved.wcs)
        assert retrieved_checksum == original_checksum

    def test_sidecar_json_created(self, cache: TPFFitsCache, sample_data: TPFFitsData) -> None:
        """put creates sidecar JSON file."""
        cache.put(sample_data)

        sidecar = cache.get_sidecar(sample_data.ref)
        assert sidecar is not None
        assert sidecar["tpf_fits_ref"] == sample_data.ref.to_string()
        assert "wcs_checksum" in sidecar
        assert "cached_at" in sidecar
        assert "aperture_mask_npixels" in sidecar

    def test_sidecar_includes_source_url(
        self, cache: TPFFitsCache, sample_data: TPFFitsData
    ) -> None:
        """put includes source_url in sidecar when provided."""
        source_url = "https://mast.stsci.edu/test/tpf.fits"
        cache.put(sample_data, source_url=source_url)

        sidecar = cache.get_sidecar(sample_data.ref)
        assert sidecar is not None
        assert sidecar.get("source_url") == source_url

    def test_sidecar_includes_time_system_and_units(
        self, cache: TPFFitsCache, sample_data: TPFFitsData
    ) -> None:
        """put preserves time-system/unit metadata and TUNIT* when provided."""
        sample_data.meta.update(
            {
                "TIMESYS": "TDB",
                "TIMEUNIT": "d",
                "BJDREFI": 2457000,
                "BJDREFF": 0.0,
                "TUNIT1": "d",
                "TUNIT2": "e-/s",
            }
        )
        cache.put(sample_data)

        sidecar = cache.get_sidecar(sample_data.ref)
        assert sidecar is not None
        header_subset = sidecar["fits_header_subset"]
        assert header_subset["TIMESYS"] == "TDB"
        assert header_subset["TIMEUNIT"] == "d"
        assert header_subset["BJDREFI"] == 2457000
        assert header_subset["BJDREFF"] == 0.0
        assert header_subset["TUNIT1"] == "d"
        assert header_subset["TUNIT2"] == "e-/s"

    def test_get_normalizes_time_when_bjdref_not_btjd_zero(self, cache: TPFFitsCache, sample_data: TPFFitsData) -> None:
        """get normalizes TIME to BTJD when BJDREF indicates non-BTJD reference."""
        sample_data.time = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sample_data.quality = np.zeros(3, dtype=np.int32)
        sample_data.flux = sample_data.flux[:3]
        sample_data.flux_err = sample_data.flux_err[:3] if sample_data.flux_err is not None else None
        sample_data.meta.update({"BJDREFI": 2458000, "BJDREFF": 0.0, "TIMESYS": "TDB"})
        cache.put(sample_data)

        retrieved = cache.get(sample_data.ref)
        assert retrieved is not None
        np.testing.assert_allclose(retrieved.time, np.array([1001.0, 1002.0, 1003.0], dtype=np.float64))
        sidecar = cache.get_sidecar(sample_data.ref)
        assert sidecar is not None
        hdr = sidecar["fits_header_subset"]
        assert hdr["BJDREFI"] == 2457000
        assert hdr["BJDREFF"] == 0.0

    def test_get_does_not_double_shift_on_re_read(self, cache: TPFFitsCache, sample_data: TPFFitsData) -> None:
        sample_data.time = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sample_data.quality = np.zeros(3, dtype=np.int32)
        sample_data.flux = sample_data.flux[:3]
        sample_data.flux_err = sample_data.flux_err[:3] if sample_data.flux_err is not None else None
        sample_data.meta.update({"BJDREFI": 2458000, "BJDREFF": 0.0, "TIMESYS": "TDB"})
        cache.put(sample_data)

        first = cache.get(sample_data.ref)
        second = cache.get(sample_data.ref)
        assert first is not None and second is not None
        np.testing.assert_allclose(first.time, np.array([1001.0, 1002.0, 1003.0], dtype=np.float64))
        np.testing.assert_allclose(second.time, first.time)

    def test_get_keeps_time_when_bjdref_is_btjd_reference(
        self, cache: TPFFitsCache, sample_data: TPFFitsData
    ) -> None:
        """get preserves TIME values when BJDREF already equals BTJD reference."""
        sample_data.time = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sample_data.quality = np.zeros(3, dtype=np.int32)
        sample_data.flux = sample_data.flux[:3]
        sample_data.flux_err = sample_data.flux_err[:3] if sample_data.flux_err is not None else None
        sample_data.meta.update({"BJDREFI": 2457000, "BJDREFF": 0.0, "TIMESYS": "TDB"})
        cache.put(sample_data)

        retrieved = cache.get(sample_data.ref)
        assert retrieved is not None
        np.testing.assert_allclose(retrieved.time, np.array([1.0, 2.0, 3.0], dtype=np.float64))

    def test_get_normalizes_time_when_only_mjd_reference_present(
        self, cache: TPFFitsCache, sample_data: TPFFitsData
    ) -> None:
        sample_data.time = np.array([1.0, 2.0], dtype=np.float64)
        sample_data.quality = np.zeros(2, dtype=np.int32)
        sample_data.flux = sample_data.flux[:2]
        sample_data.flux_err = sample_data.flux_err[:2] if sample_data.flux_err is not None else None
        sample_data.meta.update({"MJDREFI": 59000, "MJDREFF": 0.0, "TIMESYS": "TDB"})
        sample_data.meta.pop("BJDREFI", None)
        sample_data.meta.pop("BJDREFF", None)
        cache.put(sample_data)

        retrieved = cache.get(sample_data.ref)
        assert retrieved is not None
        np.testing.assert_allclose(retrieved.time, np.array([2001.5, 2002.5], dtype=np.float64))

    def test_remove_existing(self, cache: TPFFitsCache, sample_data: TPFFitsData) -> None:
        """remove deletes cached entry and returns True."""
        cache.put(sample_data)
        assert cache.has(sample_data.ref)

        result = cache.remove(sample_data.ref)
        assert result is True
        assert not cache.has(sample_data.ref)
        assert cache.get(sample_data.ref) is None
        assert cache.get_sidecar(sample_data.ref) is None

    def test_remove_nonexistent(self, cache: TPFFitsCache, sample_ref: TPFFitsRef) -> None:
        """remove returns False for nonexistent entry."""
        result = cache.remove(sample_ref)
        assert result is False

    def test_clear_empty(self, cache: TPFFitsCache) -> None:
        """clear on empty cache returns 0."""
        result = cache.clear()
        assert result == 0

    def test_clear_removes_all(
        self, cache: TPFFitsCache, sample_data: TPFFitsData, sample_wcs: WCS
    ) -> None:
        """clear removes all cached entries."""
        refs = [
            TPFFitsRef(tic_id=1, sector=1, author="spoc"),
            TPFFitsRef(tic_id=2, sector=2, author="spoc"),
            TPFFitsRef(tic_id=3, sector=3, author="qlp"),
        ]

        for ref in refs:
            data = TPFFitsData(
                ref=ref,
                time=sample_data.time,
                flux=sample_data.flux,
                flux_err=sample_data.flux_err,
                wcs=sample_wcs,
                aperture_mask=sample_data.aperture_mask,
                quality=sample_data.quality,
                camera=1,
                ccd=1,
                meta={},
            )
            cache.put(data)

        for ref in refs:
            assert cache.has(ref)

        removed = cache.clear()
        assert removed == 3

        for ref in refs:
            assert not cache.has(ref)

    def test_list_refs(
        self, cache: TPFFitsCache, sample_data: TPFFitsData, sample_wcs: WCS
    ) -> None:
        """list_refs returns all cached references."""
        refs = [
            TPFFitsRef(tic_id=1, sector=1, author="spoc"),
            TPFFitsRef(tic_id=2, sector=2, author="qlp"),
        ]

        for ref in refs:
            data = TPFFitsData(
                ref=ref,
                time=sample_data.time,
                flux=sample_data.flux,
                flux_err=sample_data.flux_err,
                wcs=sample_wcs,
                aperture_mask=sample_data.aperture_mask,
                quality=sample_data.quality,
                camera=1,
                ccd=1,
                meta={},
            )
            cache.put(data)

        listed = cache.list_refs()
        assert len(listed) == 2
        assert set(listed) == set(refs)

    def test_get_corrupted_file(
        self, cache: TPFFitsCache, sample_ref: TPFFitsRef, cache_dir: Path
    ) -> None:
        """get returns None for corrupted cache files."""
        # Create a corrupted FITS file
        filename = f"tpf_fits_{sample_ref.tic_id}_{sample_ref.sector}_{sample_ref.author}.fits"
        fits_file = cache_dir / filename
        fits_file.write_text("corrupted data")

        # has returns True (file exists) but get returns None (corrupted)
        assert cache.has(sample_ref)
        assert cache.get(sample_ref) is None


# =============================================================================
# TPFFitsNotFoundError Tests
# =============================================================================


class TestTPFFitsNotFoundError:
    """Tests for TPFFitsNotFoundError."""

    def test_error_creation(self, sample_ref: TPFFitsRef) -> None:
        """TPFFitsNotFoundError can be created with a TPFFitsRef."""
        error = TPFFitsNotFoundError(sample_ref)

        assert error.ref == sample_ref
        assert sample_ref.to_string() in str(error)

    def test_error_has_correct_type(self, sample_ref: TPFFitsRef) -> None:
        """TPFFitsNotFoundError has CACHE_MISS error type."""
        error = TPFFitsNotFoundError(sample_ref)
        assert error.error_type == ErrorType.CACHE_MISS

    def test_error_context_contains_ref_info(self, sample_ref: TPFFitsRef) -> None:
        """TPFFitsNotFoundError context contains ref information."""
        error = TPFFitsNotFoundError(sample_ref)

        assert error.context["ref"] == sample_ref.to_string()
        assert error.context["tic_id"] == sample_ref.tic_id
        assert error.context["sector"] == sample_ref.sector
        assert error.context["author"] == sample_ref.author

    def test_custom_message(self, sample_ref: TPFFitsRef) -> None:
        """TPFFitsNotFoundError accepts custom message."""
        custom_msg = "Custom error message"
        error = TPFFitsNotFoundError(sample_ref, message=custom_msg)

        assert error.message == custom_msg

    def test_default_message(self, sample_ref: TPFFitsRef) -> None:
        """TPFFitsNotFoundError has default message with ref."""
        error = TPFFitsNotFoundError(sample_ref)

        assert "TPF FITS not found" in error.message
        assert sample_ref.to_string() in error.message

    def test_can_be_raised_and_caught(self, sample_ref: TPFFitsRef) -> None:
        """TPFFitsNotFoundError can be raised and caught."""
        with pytest.raises(TPFFitsNotFoundError) as exc_info:
            raise TPFFitsNotFoundError(sample_ref)

        assert exc_info.value.ref == sample_ref


# =============================================================================
# WCS Checksum Tests
# =============================================================================


class TestWCSChecksum:
    """Tests for WCS checksum computation."""

    def test_checksum_format(self, sample_wcs: WCS) -> None:
        """Checksum has expected format."""
        checksum = _compute_wcs_checksum(sample_wcs)
        assert checksum.startswith("sha256:")
        # SHA-256 produces 64 hex characters
        assert len(checksum) == len("sha256:") + 64

    def test_checksum_deterministic(self, sample_wcs: WCS) -> None:
        """Same WCS produces same checksum."""
        checksum1 = _compute_wcs_checksum(sample_wcs)
        checksum2 = _compute_wcs_checksum(sample_wcs)
        assert checksum1 == checksum2

    def test_different_wcs_different_checksum(self) -> None:
        """Different WCS produces different checksum."""
        wcs1 = _make_test_wcs(crval=(120.0, -50.0))
        wcs2 = _make_test_wcs(crval=(121.0, -50.0))
        checksum1 = _compute_wcs_checksum(wcs1)
        checksum2 = _compute_wcs_checksum(wcs2)
        assert checksum1 != checksum2


# =============================================================================
# Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports and exports."""

    def test_import_from_tpf_fits_module(self) -> None:
        """Can import from tess_vetter.pixel.tpf_fits module."""
        from tess_vetter.pixel.tpf_fits import (
            VALID_AUTHORS,
            TPFFitsCache,
            TPFFitsData,
            TPFFitsNotFoundError,
            TPFFitsRef,
        )

        assert TPFFitsRef is not None
        assert TPFFitsData is not None
        assert TPFFitsCache is not None
        assert TPFFitsNotFoundError is not None
        assert VALID_AUTHORS is not None
