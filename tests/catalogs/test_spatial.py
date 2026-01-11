"""Tests for bittr_tess_vetter.catalogs.spatial module.

Comprehensive tests for the SpatialIndex class including:
- Basic cone search functionality
- Edge cases (poles, RA wraparound at 0/360)
- Empty results
- Angular separation calculations
- Determinism guarantees
- Large catalog performance (optional)
"""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.platform.catalogs.spatial import SpatialIndex

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_catalog() -> np.ndarray:
    """Simple catalog with 5 sources at known positions."""
    return np.array(
        [
            [10.0, 20.0],  # 0: base source
            [10.01, 20.0],  # 1: ~36 arcsec away in RA
            [10.0, 20.01],  # 2: ~36 arcsec away in Dec
            [10.1, 20.1],  # 3: farther away
            [100.0, -30.0],  # 4: completely different part of sky
        ],
        dtype=np.float64,
    )


@pytest.fixture
def dense_cluster() -> np.ndarray:
    """Dense cluster of sources within 1 arcmin."""
    np.random.seed(42)  # Deterministic
    n_sources = 100
    center_ra, center_dec = 180.0, 45.0

    # Generate sources within ~30 arcsec of center
    ra_offset = np.random.uniform(-0.01, 0.01, n_sources)
    dec_offset = np.random.uniform(-0.01, 0.01, n_sources)

    coords = np.column_stack(
        [
            center_ra + ra_offset / np.cos(np.radians(center_dec)),
            center_dec + dec_offset,
        ]
    )
    return coords


@pytest.fixture
def polar_catalog() -> np.ndarray:
    """Catalog with sources near the poles."""
    return np.array(
        [
            [0.0, 89.9],  # 0: near north pole
            [90.0, 89.9],  # 1: near north pole, different RA
            [180.0, 89.9],  # 2: near north pole, opposite RA
            [270.0, 89.9],  # 3: near north pole
            [0.0, -89.9],  # 4: near south pole
            [180.0, -89.9],  # 5: near south pole
            [45.0, 0.0],  # 6: on equator
        ],
        dtype=np.float64,
    )


@pytest.fixture
def wraparound_catalog() -> np.ndarray:
    """Catalog with sources near RA=0/360 boundary."""
    return np.array(
        [
            [0.5, 0.0],  # 0: just past RA=0
            [359.5, 0.0],  # 1: just before RA=360
            [0.0, 0.0],  # 2: exactly at RA=0
            [1.0, 0.0],  # 3: 1 degree from origin
            [359.0, 0.0],  # 4: 1 degree before wrap
        ],
        dtype=np.float64,
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestSpatialIndexConstruction:
    """Tests for SpatialIndex construction."""

    def test_construct_from_array(self, simple_catalog: np.ndarray) -> None:
        """Index can be constructed from Nx2 numpy array."""
        idx = SpatialIndex(simple_catalog)
        assert len(idx) == 5

    def test_construct_from_list(self) -> None:
        """Index can be constructed from list of lists."""
        coords = [[10.0, 20.0], [20.0, 30.0]]
        idx = SpatialIndex(np.array(coords))
        assert len(idx) == 2

    def test_construct_empty(self) -> None:
        """Index can be constructed from empty array."""
        coords = np.empty((0, 2), dtype=np.float64)
        idx = SpatialIndex(coords)
        assert len(idx) == 0

    def test_construct_single_source(self) -> None:
        """Index can be constructed with single source."""
        coords = np.array([[123.456, -45.678]])
        idx = SpatialIndex(coords)
        assert len(idx) == 1

    def test_invalid_shape_1d(self) -> None:
        """Raises ValueError for 1D array."""
        with pytest.raises(ValueError, match="Nx2 array"):
            SpatialIndex(np.array([1.0, 2.0]))

    def test_invalid_shape_wrong_columns(self) -> None:
        """Raises ValueError for array with wrong number of columns."""
        with pytest.raises(ValueError, match="Nx2 array"):
            SpatialIndex(np.array([[1.0, 2.0, 3.0]]))

    def test_coords_property_returns_original(self, simple_catalog: np.ndarray) -> None:
        """coords property returns the original coordinate array."""
        idx = SpatialIndex(simple_catalog)
        np.testing.assert_array_equal(idx.coords, simple_catalog)


# =============================================================================
# Basic Cone Search Tests
# =============================================================================


class TestBasicConeSearch:
    """Tests for basic cone search functionality."""

    def test_find_nearby_sources(self, simple_catalog: np.ndarray) -> None:
        """Cone search finds sources within radius."""
        idx = SpatialIndex(simple_catalog)

        # Search around first source with 60 arcsec radius
        # Sources 0, 1, 2 should be found (within ~36 arcsec each)
        results = idx.cone_search(10.0, 20.0, 60.0)

        assert 0 in results
        assert 1 in results
        assert 2 in results
        assert 3 not in results
        assert 4 not in results

    def test_exact_position_match(self, simple_catalog: np.ndarray) -> None:
        """Source at exact search position is found."""
        idx = SpatialIndex(simple_catalog)
        results = idx.cone_search(10.0, 20.0, 1.0)  # 1 arcsec radius
        assert 0 in results

    def test_no_matches_outside_radius(self, simple_catalog: np.ndarray) -> None:
        """No sources found when none are within radius."""
        idx = SpatialIndex(simple_catalog)
        # Search far from any source
        results = idx.cone_search(200.0, 50.0, 10.0)
        assert len(results) == 0

    def test_empty_catalog_returns_empty(self) -> None:
        """Cone search on empty catalog returns empty list."""
        coords = np.empty((0, 2), dtype=np.float64)
        idx = SpatialIndex(coords)
        results = idx.cone_search(10.0, 20.0, 3600.0)  # 1 degree
        assert results == []

    def test_zero_radius_returns_empty(self, simple_catalog: np.ndarray) -> None:
        """Cone search with zero radius returns empty list."""
        idx = SpatialIndex(simple_catalog)
        results = idx.cone_search(10.0, 20.0, 0.0)
        assert results == []

    def test_large_radius_finds_all(self, simple_catalog: np.ndarray) -> None:
        """Large radius finds all sources."""
        idx = SpatialIndex(simple_catalog)
        # 180 degrees = entire sky
        results = idx.cone_search(0.0, 0.0, 180.0 * 3600.0)
        assert len(results) == 5


# =============================================================================
# Edge Case Tests - Poles
# =============================================================================


class TestPolarEdgeCases:
    """Tests for cone searches near celestial poles."""

    def test_near_north_pole_finds_neighbors(self, polar_catalog: np.ndarray) -> None:
        """Sources near north pole with different RAs are found with large radius.

        At Dec=89.9, sources separated by 90 deg RA are ~509 arcsec apart,
        and sources separated by 180 deg RA are ~720 arcsec apart.
        Use 800 arcsec radius to find all polar sources.
        """
        idx = SpatialIndex(polar_catalog)

        # At Dec=89.9, even large RA differences result in small angular separations
        # RA=0 to RA=90 at Dec=89.9 is ~509 arcsec
        # RA=0 to RA=180 at Dec=89.9 is ~720 arcsec
        results = idx.cone_search(0.0, 89.9, 800.0)  # ~13 arcmin

        # Should find all sources at Dec=89.9 (indices 0, 1, 2, 3)
        assert 0 in results
        assert 1 in results
        assert 2 in results
        assert 3 in results
        # South pole sources should not be found
        assert 4 not in results
        assert 5 not in results

    def test_search_at_exact_pole(self) -> None:
        """Cone search centered exactly at pole works."""
        coords = np.array(
            [
                [0.0, 90.0],  # North pole
                [180.0, 89.9],  # Near pole
                [0.0, 85.0],  # 5 degrees away
            ]
        )
        idx = SpatialIndex(coords)

        # Search at north pole with 10 arcmin radius
        results = idx.cone_search(0.0, 90.0, 600.0)

        assert 0 in results  # At pole
        assert 1 in results  # 0.1 deg = 6 arcmin away
        assert 2 not in results  # 5 deg away

    def test_south_pole_search(self, polar_catalog: np.ndarray) -> None:
        """Cone search near south pole works correctly.

        At Dec=-89.9, sources at RA=0 and RA=180 are ~720 arcsec apart.
        """
        idx = SpatialIndex(polar_catalog)

        # Use large enough radius to find both south pole sources
        results = idx.cone_search(0.0, -89.9, 800.0)

        assert 4 in results
        assert 5 in results
        # North pole sources should not be found
        assert 0 not in results

    def test_polar_convergence_small_radius(self) -> None:
        """Sources very close to pole with different RA converge closer together.

        At Dec=89.999 (0.001 deg from pole), sources separated by 180 deg RA
        are only ~7.2 arcsec apart.
        """
        coords = np.array(
            [
                [0.0, 89.999],  # 0: very close to north pole
                [90.0, 89.999],  # 1: same dec, 90 deg RA difference
                [180.0, 89.999],  # 2: same dec, 180 deg RA difference
                [0.0, 89.9],  # 3: farther from pole (0.1 deg = 360 arcsec)
            ]
        )
        idx = SpatialIndex(coords)

        # At Dec=89.999:
        # - RA=0 to RA=90 is ~5.1 arcsec
        # - RA=0 to RA=180 is ~7.2 arcsec
        # 10 arcsec radius should find all three polar sources
        results = idx.cone_search(0.0, 89.999, 10.0)

        assert 0 in results
        assert 1 in results
        assert 2 in results
        # Source at dec=89.9 is ~360 arcsec from pole, well outside radius
        assert 3 not in results


# =============================================================================
# Edge Case Tests - RA Wraparound
# =============================================================================


class TestRAWraparound:
    """Tests for cone searches across RA=0/360 boundary."""

    def test_wraparound_from_low_ra(self, wraparound_catalog: np.ndarray) -> None:
        """Search from low RA finds sources across 360 boundary."""
        idx = SpatialIndex(wraparound_catalog)

        # Search at RA=0.5 should find sources near RA=0 and RA=359.5
        results = idx.cone_search(0.5, 0.0, 3700.0)  # ~1 degree

        assert 0 in results  # RA=0.5 - exact match
        assert 1 in results  # RA=359.5 - across boundary
        assert 2 in results  # RA=0.0
        assert 3 in results  # RA=1.0

    def test_wraparound_from_high_ra(self, wraparound_catalog: np.ndarray) -> None:
        """Search from high RA finds sources across 0 boundary."""
        idx = SpatialIndex(wraparound_catalog)

        # Search at RA=359.5 should find sources near RA=0
        results = idx.cone_search(359.5, 0.0, 3700.0)

        assert 0 in results  # RA=0.5 - across boundary
        assert 1 in results  # RA=359.5 - exact match
        assert 2 in results  # RA=0.0

    def test_search_at_ra_zero(self, wraparound_catalog: np.ndarray) -> None:
        """Search centered at RA=0 works correctly."""
        idx = SpatialIndex(wraparound_catalog)

        results = idx.cone_search(0.0, 0.0, 2000.0)  # ~0.55 degrees

        assert 0 in results  # 0.5 deg away
        assert 1 in results  # 0.5 deg away (across boundary)
        assert 2 in results  # exact match

    def test_search_with_negative_ra(self, wraparound_catalog: np.ndarray) -> None:
        """Negative RA is normalized correctly."""
        idx = SpatialIndex(wraparound_catalog)

        # RA=-1 should be equivalent to RA=359
        results_neg = idx.cone_search(-1.0, 0.0, 3700.0)
        results_pos = idx.cone_search(359.0, 0.0, 3700.0)

        assert results_neg == results_pos

    def test_search_with_ra_over_360(self, wraparound_catalog: np.ndarray) -> None:
        """RA > 360 is normalized correctly."""
        idx = SpatialIndex(wraparound_catalog)

        # RA=361 should be equivalent to RA=1
        results_over = idx.cone_search(361.0, 0.0, 3700.0)
        results_norm = idx.cone_search(1.0, 0.0, 3700.0)

        assert results_over == results_norm


# =============================================================================
# Empty Result Tests
# =============================================================================


class TestEmptyResults:
    """Tests for cases that should return empty results."""

    def test_no_sources_in_region(self, simple_catalog: np.ndarray) -> None:
        """Returns empty list when no sources in search region."""
        idx = SpatialIndex(simple_catalog)
        results = idx.cone_search(200.0, 70.0, 10.0)
        assert results == []

    def test_tiny_radius(self, simple_catalog: np.ndarray) -> None:
        """Very small radius might not match even nearby sources."""
        idx = SpatialIndex(simple_catalog)
        # 0.001 arcsec is effectively zero for most purposes
        results = idx.cone_search(10.005, 20.005, 0.001)
        assert results == []

    def test_empty_index(self) -> None:
        """Empty index always returns empty results."""
        idx = SpatialIndex(np.empty((0, 2)))
        results = idx.cone_search(0.0, 0.0, 36000.0)
        assert results == []


# =============================================================================
# Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_invalid_dec_too_high(self, simple_catalog: np.ndarray) -> None:
        """Raises ValueError for Dec > 90."""
        idx = SpatialIndex(simple_catalog)
        with pytest.raises(ValueError, match="dec must be in"):
            idx.cone_search(0.0, 91.0, 10.0)

    def test_invalid_dec_too_low(self, simple_catalog: np.ndarray) -> None:
        """Raises ValueError for Dec < -90."""
        idx = SpatialIndex(simple_catalog)
        with pytest.raises(ValueError, match="dec must be in"):
            idx.cone_search(0.0, -91.0, 10.0)

    def test_negative_radius(self, simple_catalog: np.ndarray) -> None:
        """Raises ValueError for negative radius."""
        idx = SpatialIndex(simple_catalog)
        with pytest.raises(ValueError, match="radius must be non-negative"):
            idx.cone_search(0.0, 0.0, -1.0)

    def test_dec_boundary_values(self, simple_catalog: np.ndarray) -> None:
        """Dec exactly at +/-90 is valid."""
        idx = SpatialIndex(simple_catalog)
        # Should not raise
        idx.cone_search(0.0, 90.0, 10.0)
        idx.cone_search(0.0, -90.0, 10.0)


# =============================================================================
# Angular Separation Tests
# =============================================================================


class TestAngularSeparation:
    """Tests for the angular_separation static method."""

    def test_identical_points(self) -> None:
        """Separation between identical points is zero."""
        sep = SpatialIndex.angular_separation(10.0, 20.0, 10.0, 20.0)
        assert abs(sep) < 1e-10

    def test_known_separation_small(self) -> None:
        """Small angular separation is computed correctly."""
        # Two points 1 degree apart in Dec
        sep = SpatialIndex.angular_separation(0.0, 0.0, 0.0, 1.0)
        expected = 3600.0  # 1 degree = 3600 arcsec
        assert abs(sep - expected) < 0.1

    def test_known_separation_ra(self) -> None:
        """RA separation at equator is computed correctly."""
        # Two points 1 degree apart in RA at equator
        sep = SpatialIndex.angular_separation(0.0, 0.0, 1.0, 0.0)
        expected = 3600.0  # 1 degree = 3600 arcsec
        assert abs(sep - expected) < 0.1

    def test_cos_dec_correction(self) -> None:
        """RA separation accounts for cos(dec) correctly."""
        # At Dec=60, 1 degree of RA is only 0.5 degrees of angular distance
        sep = SpatialIndex.angular_separation(0.0, 60.0, 1.0, 60.0)
        expected = 3600.0 * 0.5  # cos(60) = 0.5
        assert abs(sep - expected) < 1.0  # Within 1 arcsec

    def test_antipodal_points(self) -> None:
        """Separation between antipodal points is 180 degrees."""
        sep = SpatialIndex.angular_separation(0.0, 0.0, 180.0, 0.0)
        expected = 180.0 * 3600.0  # 180 degrees
        assert abs(sep - expected) < 1.0

    def test_pole_to_pole(self) -> None:
        """Separation from north to south pole is 180 degrees."""
        sep = SpatialIndex.angular_separation(0.0, 90.0, 0.0, -90.0)
        expected = 180.0 * 3600.0
        assert abs(sep - expected) < 1.0

    def test_across_ra_boundary(self) -> None:
        """Separation across RA=0/360 boundary is computed correctly."""
        # RA=1 to RA=359 at equator = 2 degrees
        sep = SpatialIndex.angular_separation(1.0, 0.0, 359.0, 0.0)
        expected = 2.0 * 3600.0
        assert abs(sep - expected) < 1.0


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_searches_same_result(self, simple_catalog: np.ndarray) -> None:
        """Repeated searches return identical results."""
        idx = SpatialIndex(simple_catalog)

        results1 = idx.cone_search(10.0, 20.0, 60.0)
        results2 = idx.cone_search(10.0, 20.0, 60.0)
        results3 = idx.cone_search(10.0, 20.0, 60.0)

        assert results1 == results2 == results3

    def test_results_are_sorted(self, dense_cluster: np.ndarray) -> None:
        """Results are returned in sorted order."""
        idx = SpatialIndex(dense_cluster)
        results = idx.cone_search(180.0, 45.0, 120.0)

        assert results == sorted(results)

    def test_same_catalog_same_index(self) -> None:
        """Same catalog produces same search results."""
        coords = np.array(
            [
                [10.0, 20.0],
                [10.01, 20.01],
                [50.0, 50.0],
            ]
        )

        idx1 = SpatialIndex(coords.copy())
        idx2 = SpatialIndex(coords.copy())

        results1 = idx1.cone_search(10.0, 20.0, 100.0)
        results2 = idx2.cone_search(10.0, 20.0, 100.0)

        assert results1 == results2


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_cone_search_matches_angular_separation(self, simple_catalog: np.ndarray) -> None:
        """All returned results are within stated radius."""
        idx = SpatialIndex(simple_catalog)
        ra, dec, radius = 10.0, 20.0, 60.0

        results = idx.cone_search(ra, dec, radius)

        for i in results:
            sep = SpatialIndex.angular_separation(
                ra, dec, simple_catalog[i, 0], simple_catalog[i, 1]
            )
            assert sep <= radius + 0.1, f"Source {i} at {sep} arcsec exceeds radius"

    def test_sources_outside_not_returned(self, simple_catalog: np.ndarray) -> None:
        """Sources outside radius are not returned."""
        idx = SpatialIndex(simple_catalog)
        ra, dec, radius = 10.0, 20.0, 30.0  # Smaller radius

        results = idx.cone_search(ra, dec, radius)
        all_indices = set(range(len(simple_catalog)))
        outside = all_indices - set(results)

        for i in outside:
            sep = SpatialIndex.angular_separation(
                ra, dec, simple_catalog[i, 0], simple_catalog[i, 1]
            )
            # Allow small tolerance for boundary cases
            assert sep >= radius - 0.1, f"Source {i} at {sep} arcsec should be found"


# =============================================================================
# Performance Tests (Optional)
# =============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests for large catalogs."""

    def test_large_catalog_construction(self) -> None:
        """Index construction completes in reasonable time for large catalog."""
        np.random.seed(123)
        n_sources = 100_000

        coords = np.column_stack(
            [
                np.random.uniform(0, 360, n_sources),
                np.random.uniform(-90, 90, n_sources),
            ]
        )

        # This should complete without timeout
        idx = SpatialIndex(coords)
        assert len(idx) == n_sources

    def test_large_catalog_query(self) -> None:
        """Query performance is reasonable for large catalog."""
        np.random.seed(456)
        n_sources = 100_000

        coords = np.column_stack(
            [
                np.random.uniform(0, 360, n_sources),
                np.random.uniform(-90, 90, n_sources),
            ]
        )

        idx = SpatialIndex(coords)

        # Perform multiple queries
        for _ in range(100):
            ra = np.random.uniform(0, 360)
            dec = np.random.uniform(-90, 90)
            idx.cone_search(ra, dec, 60.0)  # 1 arcmin

    def test_many_results_query(self) -> None:
        """Query returning many results completes efficiently."""
        np.random.seed(789)
        n_sources = 50_000

        # Create dense catalog in small region
        coords = np.column_stack(
            [
                np.random.uniform(179, 181, n_sources),  # 2 degree RA range
                np.random.uniform(44, 46, n_sources),  # 2 degree Dec range
            ]
        )

        idx = SpatialIndex(coords)

        # Large radius query
        results = idx.cone_search(180.0, 45.0, 3600.0)  # 1 degree
        assert len(results) > 0
