"""Tests for tess_vetter.pixel.wcs_utils module.

Tests the WCS utilities including:
- WCS extraction from headers
- WCS checksum computation and verification
- World to pixel coordinate transforms
- Pixel to world coordinate transforms
- Target pixel position lookup
- Angular distance computation
- Reference source distance computation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from tess_vetter.pixel.wcs_utils import (
    compute_angular_distance,
    compute_pixel_scale,
    compute_source_distances,
    compute_wcs_checksum,
    extract_wcs_from_header,
    get_reference_source_pixel_positions,
    get_stamp_center,
    get_stamp_center_world,
    get_target_pixel_position,
    pixel_to_world,
    pixel_to_world_batch,
    verify_wcs_checksum,
    wcs_sanity_check,
    world_to_pixel,
    world_to_pixel_batch,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def _make_test_wcs(
    crval: tuple[float, float] = (120.0, -50.0),
    pixel_scale_deg: float = 21.0 / 3600.0,  # 21 arcsec/pixel in degrees
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
def sample_wcs() -> WCS:
    """Create a sample WCS for testing."""
    return _make_test_wcs()


@pytest.fixture
def sample_fits_header() -> fits.Header:
    """Create a sample FITS header with WCS keywords."""
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 11
    header["NAXIS2"] = 11
    header["CRPIX1"] = 6.0
    header["CRPIX2"] = 6.0
    header["CRVAL1"] = 120.0
    header["CRVAL2"] = -50.0
    header["CDELT1"] = -21.0 / 3600.0
    header["CDELT2"] = 21.0 / 3600.0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    return header


@pytest.fixture
def reference_sources() -> list[dict[str, Any]]:
    """Sample reference sources for testing."""
    return [
        {"name": "Target", "ra": 120.0, "dec": -50.0},
        {"name": "Neighbor1", "ra": 120.005, "dec": -50.0},  # ~18 arcsec offset
        {"name": "Neighbor2", "ra": 120.0, "dec": -50.01},  # ~36 arcsec offset
    ]


# =============================================================================
# WCS Extraction Tests
# =============================================================================


class TestExtractWCSFromHeader:
    """Tests for extract_wcs_from_header."""

    def test_extract_valid_header(self, sample_fits_header: fits.Header) -> None:
        """Can extract WCS from valid header."""
        wcs = extract_wcs_from_header(sample_fits_header)
        assert wcs is not None
        assert wcs.has_celestial

    def test_extract_invalid_header(self) -> None:
        """Raises ValueError for header without WCS."""
        header = fits.Header()
        header["SIMPLE"] = True
        with pytest.raises(ValueError, match="celestial WCS"):
            extract_wcs_from_header(header)


# =============================================================================
# WCS Checksum Tests
# =============================================================================


class TestWCSChecksum:
    """Tests for WCS checksum computation and verification."""

    def test_checksum_format(self, sample_wcs: WCS) -> None:
        """Checksum has expected format."""
        checksum = compute_wcs_checksum(sample_wcs)
        assert checksum.startswith("sha256:")
        assert len(checksum) == len("sha256:") + 64

    def test_checksum_deterministic(self, sample_wcs: WCS) -> None:
        """Same WCS produces same checksum."""
        checksum1 = compute_wcs_checksum(sample_wcs)
        checksum2 = compute_wcs_checksum(sample_wcs)
        assert checksum1 == checksum2

    def test_different_wcs_different_checksum(self) -> None:
        """Different WCS produces different checksum."""
        wcs1 = _make_test_wcs(crval=(120.0, -50.0))
        wcs2 = _make_test_wcs(crval=(121.0, -50.0))
        assert compute_wcs_checksum(wcs1) != compute_wcs_checksum(wcs2)

    def test_verify_matching_checksum(self, sample_wcs: WCS) -> None:
        """verify_wcs_checksum returns True for matching checksum."""
        checksum = compute_wcs_checksum(sample_wcs)
        assert verify_wcs_checksum(sample_wcs, checksum) is True

    def test_verify_mismatched_checksum(self, sample_wcs: WCS) -> None:
        """verify_wcs_checksum returns False for mismatched checksum."""
        assert verify_wcs_checksum(sample_wcs, "sha256:wrong") is False


# =============================================================================
# Coordinate Transform Tests
# =============================================================================


class TestWorldToPixel:
    """Tests for world_to_pixel transforms."""

    def test_center_position(self, sample_wcs: WCS) -> None:
        """Reference point maps to CRPIX."""
        row, col = world_to_pixel(sample_wcs, ra_deg=120.0, dec_deg=-50.0, origin=0)
        # CRPIX is (6, 6) in 1-indexed, so (5, 5) in 0-indexed
        assert pytest.approx(row, abs=0.01) == 5.0
        assert pytest.approx(col, abs=0.01) == 5.0

    def test_offset_position(self, sample_wcs: WCS) -> None:
        """Offset position transforms correctly."""
        # 21 arcsec = 1 pixel offset in Dec (no cos factor for Dec)
        offset_arcsec = 21.0
        offset_deg = offset_arcsec / 3600.0
        # Test Dec offset (which doesn't have cos factor)
        row, col = world_to_pixel(sample_wcs, ra_deg=120.0, dec_deg=-50.0 + offset_deg, origin=0)
        # Dec increases = larger row
        assert pytest.approx(row, abs=0.1) == 6.0
        assert pytest.approx(col, abs=0.1) == 5.0

    def test_origin_1_indexed(self, sample_wcs: WCS) -> None:
        """Origin=1 gives 1-indexed coordinates."""
        row, col = world_to_pixel(sample_wcs, ra_deg=120.0, dec_deg=-50.0, origin=1)
        assert pytest.approx(row, abs=0.01) == 6.0
        assert pytest.approx(col, abs=0.01) == 6.0


class TestWorldToPixelBatch:
    """Tests for world_to_pixel_batch transforms."""

    def test_batch_transform(self, sample_wcs: WCS) -> None:
        """Batch transform produces correct results."""
        ra = np.array([120.0, 120.005])
        dec = np.array([-50.0, -50.0])
        rows, cols = world_to_pixel_batch(sample_wcs, ra, dec, origin=0)

        assert len(rows) == 2
        assert len(cols) == 2
        assert pytest.approx(rows[0], abs=0.1) == 5.0
        assert pytest.approx(cols[0], abs=0.1) == 5.0


class TestPixelToWorld:
    """Tests for pixel_to_world transforms."""

    def test_center_position(self, sample_wcs: WCS) -> None:
        """Center pixel maps to CRVAL."""
        ra, dec = pixel_to_world(sample_wcs, row=5.0, col=5.0, origin=0)
        assert pytest.approx(ra, abs=0.001) == 120.0
        assert pytest.approx(dec, abs=0.001) == -50.0

    def test_roundtrip(self, sample_wcs: WCS) -> None:
        """world_to_pixel and pixel_to_world are inverses."""
        ra_orig, dec_orig = 120.01, -50.005
        row, col = world_to_pixel(sample_wcs, ra_orig, dec_orig, origin=0)
        ra_rt, dec_rt = pixel_to_world(sample_wcs, row, col, origin=0)
        assert pytest.approx(ra_rt, abs=0.0001) == ra_orig
        assert pytest.approx(dec_rt, abs=0.0001) == dec_orig


class TestPixelToWorldBatch:
    """Tests for pixel_to_world_batch transforms."""

    def test_batch_transform(self, sample_wcs: WCS) -> None:
        """Batch transform produces correct results."""
        rows = np.array([5.0, 6.0])
        cols = np.array([5.0, 5.0])
        ra, dec = pixel_to_world_batch(sample_wcs, rows, cols, origin=0)

        assert len(ra) == 2
        assert len(dec) == 2
        assert pytest.approx(ra[0], abs=0.001) == 120.0
        assert pytest.approx(dec[0], abs=0.001) == -50.0


# =============================================================================
# Target Position Tests
# =============================================================================


class TestGetTargetPixelPosition:
    """Tests for get_target_pixel_position."""

    def test_target_at_center(self, sample_wcs: WCS) -> None:
        """Target at CRVAL maps to center."""
        row, col = get_target_pixel_position(sample_wcs, target_ra_deg=120.0, target_dec_deg=-50.0)
        assert pytest.approx(row, abs=0.1) == 5.0
        assert pytest.approx(col, abs=0.1) == 5.0

    def test_target_within_stamp(self, sample_wcs: WCS) -> None:
        """Target within stamp validates successfully."""
        row, col = get_target_pixel_position(
            sample_wcs,
            target_ra_deg=120.0,
            target_dec_deg=-50.0,
            stamp_shape=(11, 11),
        )
        assert 0 <= row < 11
        assert 0 <= col < 11

    def test_target_outside_stamp(self, sample_wcs: WCS) -> None:
        """Target outside stamp raises ValueError."""
        # Large offset that puts target outside stamp
        with pytest.raises(ValueError, match="outside stamp"):
            get_target_pixel_position(
                sample_wcs,
                target_ra_deg=121.0,  # Far outside
                target_dec_deg=-50.0,
                stamp_shape=(11, 11),
            )


# =============================================================================
# Angular Distance Tests
# =============================================================================


class TestComputeAngularDistance:
    """Tests for compute_angular_distance."""

    def test_zero_distance(self) -> None:
        """Same position gives zero distance."""
        dist = compute_angular_distance(120.0, -50.0, 120.0, -50.0)
        assert dist == 0.0

    def test_small_offset_ra(self) -> None:
        """Small RA offset gives expected distance."""
        # 1 arcsec offset in RA (at dec=-50, cos(dec) factor applies)
        offset_arcsec = 1.0
        offset_deg = offset_arcsec / 3600.0 / np.cos(np.radians(-50.0))
        dist = compute_angular_distance(120.0, -50.0, 120.0 + offset_deg, -50.0)
        assert pytest.approx(dist, rel=0.01) == 1.0

    def test_small_offset_dec(self) -> None:
        """Small Dec offset gives expected distance."""
        # 1 arcsec offset in Dec
        offset_deg = 1.0 / 3600.0
        dist = compute_angular_distance(120.0, -50.0, 120.0, -50.0 + offset_deg)
        assert pytest.approx(dist, rel=0.01) == 1.0

    def test_larger_offset(self) -> None:
        """Larger offset gives reasonable distance."""
        # Approximately 10 arcsec offset
        dist = compute_angular_distance(120.0, -50.0, 120.003, -50.002)
        assert 10 < dist < 20  # Rough check


# =============================================================================
# Pixel Scale Tests
# =============================================================================


class TestComputePixelScale:
    """Tests for compute_pixel_scale."""

    def test_tess_scale(self, sample_wcs: WCS) -> None:
        """TESS-like WCS gives ~21 arcsec/pixel."""
        scale = compute_pixel_scale(sample_wcs)
        assert pytest.approx(scale, rel=0.01) == 21.0


# =============================================================================
# Stamp Center Tests
# =============================================================================


class TestGetStampCenter:
    """Tests for get_stamp_center."""

    def test_odd_shape(self) -> None:
        """Odd-shaped stamp has integer center."""
        row, col = get_stamp_center((11, 11))
        assert row == 5.0
        assert col == 5.0

    def test_even_shape(self) -> None:
        """Even-shaped stamp has half-integer center."""
        row, col = get_stamp_center((10, 10))
        assert row == 4.5
        assert col == 4.5


class TestGetStampCenterWorld:
    """Tests for get_stamp_center_world."""

    def test_center_at_crval(self, sample_wcs: WCS) -> None:
        """Stamp center maps to CRVAL."""
        ra, dec = get_stamp_center_world(sample_wcs, stamp_shape=(11, 11))
        assert pytest.approx(ra, abs=0.001) == 120.0
        assert pytest.approx(dec, abs=0.001) == -50.0


# =============================================================================
# Reference Source Tests
# =============================================================================


class TestComputeSourceDistances:
    """Tests for compute_source_distances."""

    def test_distances_from_centroid(self, reference_sources: list[dict[str, Any]]) -> None:
        """Computes distances from centroid to all sources."""
        distances = compute_source_distances(120.0, -50.0, reference_sources)
        assert "Target" in distances
        assert "Neighbor1" in distances
        assert "Neighbor2" in distances
        # Target should be at zero distance
        assert distances["Target"] == 0.0
        # Neighbors should have positive distance
        assert distances["Neighbor1"] > 0
        assert distances["Neighbor2"] > 0

    def test_missing_coordinates(self) -> None:
        """Handles sources with missing coordinates."""
        sources = [
            {"name": "Good", "ra": 120.0, "dec": -50.0},
            {"name": "Bad", "ra": None, "dec": None},
        ]
        distances = compute_source_distances(120.0, -50.0, sources)
        assert "Good" in distances
        assert "Bad" not in distances


class TestGetReferenceSourcePixelPositions:
    """Tests for get_reference_source_pixel_positions."""

    def test_pixel_positions(
        self, sample_wcs: WCS, reference_sources: list[dict[str, Any]]
    ) -> None:
        """Gets pixel positions for reference sources."""
        positions = get_reference_source_pixel_positions(sample_wcs, reference_sources)
        assert "Target" in positions
        assert "Neighbor1" in positions
        assert "Neighbor2" in positions
        # Target should be near center
        target_row, target_col = positions["Target"]
        assert pytest.approx(target_row, abs=0.1) == 5.0
        assert pytest.approx(target_col, abs=0.1) == 5.0


# =============================================================================
# WCS Sanity Check Tests
# =============================================================================


class TestWCSSanityCheck:
    """Tests for wcs_sanity_check."""

    def test_valid_wcs(self, sample_wcs: WCS) -> None:
        """Valid TESS-like WCS passes sanity check."""
        is_valid, warnings = wcs_sanity_check(
            sample_wcs,
            expected_ra_deg=120.0,
            expected_dec_deg=-50.0,
            stamp_shape=(11, 11),
        )
        assert is_valid is True
        assert len(warnings) == 0

    def test_offset_center_warns(self, sample_wcs: WCS) -> None:
        """Large offset from expected center produces warning."""
        is_valid, warnings = wcs_sanity_check(
            sample_wcs,
            expected_ra_deg=125.0,  # 5 degrees off!
            expected_dec_deg=-50.0,
            stamp_shape=(11, 11),
            tolerance_arcsec=60.0,
        )
        assert is_valid is False
        assert any("offset" in w.lower() for w in warnings)

    def test_unusual_pixel_scale(self) -> None:
        """Unusual pixel scale produces warning."""
        # Create WCS with very large pixel scale
        wcs = _make_test_wcs(pixel_scale_deg=1.0)  # 1 degree/pixel is huge
        is_valid, warnings = wcs_sanity_check(wcs)
        assert any("pixel scale" in w.lower() for w in warnings)


# =============================================================================
# Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports."""

    def test_import_all_functions(self) -> None:
        """Can import all public functions from module."""
        from tess_vetter.pixel.wcs_utils import (
            compute_angular_distance,
            compute_pixel_scale,
            compute_source_distances,
            compute_wcs_checksum,
            extract_wcs_from_header,
            get_reference_source_pixel_positions,
            get_stamp_center,
            get_stamp_center_world,
            get_target_pixel_position,
            pixel_to_world,
            pixel_to_world_batch,
            verify_wcs_checksum,
            wcs_sanity_check,
            world_to_pixel,
            world_to_pixel_batch,
        )

        assert compute_angular_distance is not None
        assert compute_pixel_scale is not None
        assert compute_source_distances is not None
        assert compute_wcs_checksum is not None
        assert extract_wcs_from_header is not None
        assert get_reference_source_pixel_positions is not None
        assert get_stamp_center is not None
        assert get_stamp_center_world is not None
        assert get_target_pixel_position is not None
        assert pixel_to_world is not None
        assert pixel_to_world_batch is not None
        assert verify_wcs_checksum is not None
        assert wcs_sanity_check is not None
        assert world_to_pixel is not None
        assert world_to_pixel_batch is not None
