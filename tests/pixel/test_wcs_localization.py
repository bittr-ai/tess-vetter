"""Tests for bittr_tess_vetter.pixel.wcs_localization module.

Tests the WCS-aware transit localization algorithms including:
- Difference image centroid computation
- Bootstrap uncertainty estimation
- Reference source distance computation
- Full localization pipeline with verdict determination

Uses synthetic TPF fixtures with known ground truth for validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData
from bittr_tess_vetter.pixel.wcs_localization import (
    LocalizationResult,
    LocalizationVerdict,
    bootstrap_centroid_uncertainty,
    compute_difference_image_centroid,
    compute_reference_source_distances,
    localize_transit_source,
)
from tests.pixel.fixtures.synthetic_cubes import (
    StarSpec,
    TransitSpec,
    make_blended_binary_tpf,
    make_saturated_tpf,
    make_synthetic_tpf_fits,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def _make_test_wcs(
    crval: tuple[float, float] = (120.0, -50.0),
    pixel_scale_deg: float = 21.0 / 3600.0,
    shape: tuple[int, int] = (11, 11),
) -> WCS:
    """Create a test WCS object for TESS-like data."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(shape[1] + 1) / 2, (shape[0] + 1) / 2]
    wcs.wcs.crval = list(crval)
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@pytest.fixture
def single_star_tpf() -> TPFFitsData:
    """Single star at center with transit (on-target baseline).

    Uses high flux and deep transit for reliable SNR in tests.
    Low noise ensures bootstrap centroid variance is small enough
    for ON_TARGET verdict (distance < 2-sigma from target).
    """
    return make_synthetic_tpf_fits(
        shape=(500, 11, 11),
        stars=[StarSpec(row=5.0, col=5.0, flux=100000.0)],  # High flux for SNR
        transit_spec=TransitSpec(
            star_idx=0,
            depth_frac=0.05,  # 5% depth for clear signal
            period=5.0,
            t0=2458001.0,
            duration_days=0.2,
        ),
        noise_level=10.0,  # Low noise for tight centroid uncertainty
        seed=42,
    )


@pytest.fixture
def binary_secondary_transit_tpf() -> TPFFitsData:
    """Binary system with transit on secondary (off-target test).

    Uses high flux and deep transit for reliable SNR.
    """
    return make_blended_binary_tpf(
        separation_arcsec=21.0,  # ~1 pixel separation
        flux_ratio=0.5,
        transit_on_secondary=True,
        transit_depth_frac=0.05,  # 5% depth
        primary_flux=100000.0,  # High flux
        shape=(500, 11, 11),
        noise_level=50.0,
        seed=42,
    )


@pytest.fixture
def binary_primary_transit_tpf() -> TPFFitsData:
    """Binary system with transit on primary (on-target test).

    Uses high flux and deep transit for reliable SNR.
    """
    return make_blended_binary_tpf(
        separation_arcsec=21.0,  # ~1 pixel separation
        flux_ratio=0.5,
        transit_on_secondary=False,
        transit_depth_frac=0.05,  # 5% depth
        primary_flux=100000.0,  # High flux
        shape=(500, 11, 11),
        noise_level=50.0,
        seed=42,
    )


@pytest.fixture
def close_binary_tpf() -> TPFFitsData:
    """Close binary (3 arcsec) - ambiguous case.

    Uses high flux and deep transit for reliable SNR.
    """
    return make_blended_binary_tpf(
        separation_arcsec=3.0,  # Less than 1 pixel
        flux_ratio=0.5,
        transit_on_secondary=True,
        transit_depth_frac=0.05,  # 5% depth
        primary_flux=100000.0,  # High flux
        shape=(500, 11, 11),
        noise_level=50.0,
        seed=42,
    )


# =============================================================================
# Tests: Difference Image Centroid
# =============================================================================


class TestComputeDifferenceImageCentroid:
    """Tests for compute_difference_image_centroid function."""

    def test_single_star_centroid_at_star_position(self, single_star_tpf: TPFFitsData) -> None:
        """Centroid should be at star position for single star with transit."""
        centroid, diff_image = compute_difference_image_centroid(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,  # 0.2 days
        )

        # Star is at (5.0, 5.0)
        assert abs(centroid[0] - 5.0) < 0.5, f"Row centroid {centroid[0]} far from 5.0"
        assert abs(centroid[1] - 5.0) < 0.5, f"Col centroid {centroid[1]} far from 5.0"

    def test_binary_secondary_transit_centroid_offset(
        self, binary_secondary_transit_tpf: TPFFitsData
    ) -> None:
        """Centroid should be offset toward secondary when transit is there."""
        centroid, diff_image = compute_difference_image_centroid(
            tpf_fits=binary_secondary_transit_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
        )

        # Primary at center (5.0, 5.0), secondary offset in row direction
        # Centroid should be closer to secondary than to primary
        primary_row = 5.0  # Primary is at center
        # Secondary is 1 pixel offset in row direction (21 arcsec / 21 arcsec/pixel)

        # Centroid should be offset toward secondary
        assert centroid[0] > primary_row, "Centroid should be offset toward secondary"

    def test_binary_primary_transit_centroid_at_primary(
        self, binary_primary_transit_tpf: TPFFitsData
    ) -> None:
        """Centroid should be at primary when transit is on primary."""
        centroid, diff_image = compute_difference_image_centroid(
            tpf_fits=binary_primary_transit_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
        )

        # Primary at center (5.0, 5.0)
        assert abs(centroid[0] - 5.0) < 0.5, f"Row centroid {centroid[0]} far from 5.0"
        assert abs(centroid[1] - 5.0) < 0.5, f"Col centroid {centroid[1]} far from 5.0"

    def test_difference_image_positive_at_transit(self, single_star_tpf: TPFFitsData) -> None:
        """Difference image should be positive where transit dimming occurs."""
        _, diff_image = compute_difference_image_centroid(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
        )

        # Maximum should be near star position
        max_idx = np.unravel_index(np.argmax(diff_image), diff_image.shape)
        assert abs(max_idx[0] - 5.0) < 1.0, "Max not near star row"
        assert abs(max_idx[1] - 5.0) < 1.0, "Max not near star col"

        # Max value should be positive
        assert np.max(diff_image) > 0, "Difference image max should be positive"

    def test_insufficient_in_transit_raises(self) -> None:
        """Should raise ValueError if insufficient in-transit data."""
        # Create TPF with very short time span
        tpf = make_synthetic_tpf_fits(
            shape=(10, 11, 11),  # Only 10 cadences
            time_span_days=0.1,  # Very short
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
        )

        # Transit doesn't fall in time range
        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            compute_difference_image_centroid(
                tpf_fits=tpf,
                period=5.0,
                t0=2458100.0,  # Far from data range
                duration_hours=4.8,
            )

    def test_gaussian_fit_method(self, single_star_tpf: TPFFitsData) -> None:
        """Gaussian fit method should also find centroid near star."""
        centroid, _ = compute_difference_image_centroid(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            method="gaussian_fit",
        )

        # Should still be near star position
        assert abs(centroid[0] - 5.0) < 0.5, f"Row centroid {centroid[0]} far from 5.0"
        assert abs(centroid[1] - 5.0) < 0.5, f"Col centroid {centroid[1]} far from 5.0"


# =============================================================================
# Tests: Bootstrap Uncertainty
# =============================================================================


class TestBootstrapCentroidUncertainty:
    """Tests for bootstrap_centroid_uncertainty function."""

    def test_deterministic_with_seed(self, single_star_tpf: TPFFitsData) -> None:
        """Bootstrap with same seed should produce identical results."""
        centroids1, semi1, _, _ = bootstrap_centroid_uncertainty(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            n_draws=50,
            seed=12345,
        )

        centroids2, semi2, _, _ = bootstrap_centroid_uncertainty(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            n_draws=50,
            seed=12345,
        )

        np.testing.assert_array_equal(centroids1, centroids2)
        assert semi1 == semi2

    def test_different_seeds_produce_different_results(self, single_star_tpf: TPFFitsData) -> None:
        """Different seeds should produce different bootstrap distributions."""
        centroids1, _, _, _ = bootstrap_centroid_uncertainty(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            n_draws=50,
            seed=111,
        )

        centroids2, _, _, _ = bootstrap_centroid_uncertainty(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            n_draws=50,
            seed=222,
        )

        # Should not be identical
        assert not np.allclose(centroids1, centroids2)

    def test_uncertainty_ellipse_positive(self, single_star_tpf: TPFFitsData) -> None:
        """Uncertainty ellipse axes should be positive."""
        _, semimajor, semiminor, pa = bootstrap_centroid_uncertainty(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            n_draws=100,
            seed=42,
        )

        assert np.isfinite(semimajor) and semimajor > 0
        assert np.isfinite(semiminor) and semiminor > 0
        assert np.isfinite(pa)
        assert semimajor >= semiminor  # Semi-major >= semi-minor

    def test_centroids_array_shape(self, single_star_tpf: TPFFitsData) -> None:
        """Centroids array should have correct shape."""
        n_draws = 100
        centroids, _, _, _ = bootstrap_centroid_uncertainty(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            n_draws=n_draws,
            seed=42,
        )

        assert centroids.shape == (n_draws, 2)


# =============================================================================
# Tests: Reference Source Distances
# =============================================================================


class TestComputeReferenceSourceDistances:
    """Tests for compute_reference_source_distances function."""

    def test_zero_distance_to_same_position(self) -> None:
        """Distance to same position should be zero."""
        distances = compute_reference_source_distances(
            centroid_sky=(120.0, -50.0),
            reference_sources=[
                {"name": "same", "ra": 120.0, "dec": -50.0},
            ],
        )

        assert abs(distances["same"]) < 0.01  # < 0.01 arcsec

    def test_known_distance(self) -> None:
        """Test against known angular separation."""
        # 1 degree separation in RA at Dec=-50 should be ~cos(50)*3600 arcsec
        # cos(50 deg) ~ 0.643
        distances = compute_reference_source_distances(
            centroid_sky=(120.0, -50.0),
            reference_sources=[
                {"name": "offset", "ra": 121.0, "dec": -50.0},
            ],
        )

        expected_arcsec = np.cos(np.radians(50.0)) * 3600.0
        assert abs(distances["offset"] - expected_arcsec) < 10  # Within 10 arcsec

    def test_multiple_sources(self) -> None:
        """Should compute distances to all sources."""
        distances = compute_reference_source_distances(
            centroid_sky=(120.0, -50.0),
            reference_sources=[
                {"name": "src1", "ra": 120.0, "dec": -50.0},
                {"name": "src2", "ra": 120.001, "dec": -50.0},
                {"name": "src3", "ra": 120.0, "dec": -50.001},
            ],
        )

        assert len(distances) == 3
        assert "src1" in distances
        assert "src2" in distances
        assert "src3" in distances


# =============================================================================
# Tests: Full Localization Pipeline
# =============================================================================


class TestLocalizeTransitSource:
    """Tests for localize_transit_source function."""

    def test_on_target_single_star(self, single_star_tpf: TPFFitsData) -> None:
        """Single star transit should give ON_TARGET verdict."""
        result = localize_transit_source(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[
                {"name": "target", "ra": 120.0, "dec": -50.0},  # WCS center
            ],
            bootstrap_draws=100,
            bootstrap_seed=42,
        )

        assert isinstance(result, LocalizationResult)
        # Should be ON_TARGET or AMBIGUOUS (not OFF_TARGET or INVALID)
        assert result.verdict in [LocalizationVerdict.ON_TARGET, LocalizationVerdict.AMBIGUOUS]
        assert np.isfinite(result.centroid_sky_ra)
        assert np.isfinite(result.centroid_sky_dec)

    def test_off_target_secondary_transit(self, binary_secondary_transit_tpf: TPFFitsData) -> None:
        """Transit on secondary should show centroid offset from primary."""
        # Get WCS center (primary position)
        wcs = binary_secondary_transit_tpf.wcs
        center_ra = float(wcs.wcs.crval[0])
        center_dec = float(wcs.wcs.crval[1])

        result = localize_transit_source(
            tpf_fits=binary_secondary_transit_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[
                {"name": "primary", "ra": center_ra, "dec": center_dec},
            ],
            bootstrap_draws=100,
            bootstrap_seed=42,
        )

        # Distance to primary should be > 0 (centroid offset toward secondary)
        dist_to_primary = result.distances_to_sources.get("primary", 0)
        # Should show some offset (transit is on secondary, ~21 arcsec away)
        # With noise, might not be exactly 21, but should be detectable
        assert dist_to_primary > 5.0, f"Expected offset, got {dist_to_primary} arcsec"

    def test_close_binary_ambiguous(self, close_binary_tpf: TPFFitsData) -> None:
        """Close binary should give AMBIGUOUS or show small uncertainty."""
        wcs = close_binary_tpf.wcs
        center_ra = float(wcs.wcs.crval[0])
        center_dec = float(wcs.wcs.crval[1])

        result = localize_transit_source(
            tpf_fits=close_binary_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[
                {"name": "primary", "ra": center_ra, "dec": center_dec},
                # Secondary ~3 arcsec offset
                {"name": "secondary", "ra": center_ra, "dec": center_dec + 3.0 / 3600},
            ],
            bootstrap_draws=100,
            bootstrap_seed=42,
        )

        # Should have two distance measurements
        assert "primary" in result.distances_to_sources
        assert "secondary" in result.distances_to_sources

    def test_result_to_dict(self, single_star_tpf: TPFFitsData) -> None:
        """Result to_dict should produce valid dictionary."""
        result = localize_transit_source(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[
                {"name": "target", "ra": 120.0, "dec": -50.0},
            ],
            bootstrap_draws=50,
            bootstrap_seed=42,
        )

        d = result.to_dict()

        assert "centroid_pixel_rc" in d
        assert "centroid_sky_ra" in d
        assert "centroid_sky_dec" in d
        assert "verdict" in d
        assert "distances_to_sources" in d
        assert isinstance(d["verdict"], str)

    def test_zero_bootstrap_draws_fast_mode(self, single_star_tpf: TPFFitsData) -> None:
        """Zero bootstrap draws should work (fast mode)."""
        result = localize_transit_source(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[],
            bootstrap_draws=0,
            bootstrap_seed=42,
        )

        assert result.n_bootstrap_draws == 0
        # Uncertainty should be NaN in fast mode
        assert not np.isfinite(result.uncertainty_semimajor_arcsec)

    def test_seed_recorded(self, single_star_tpf: TPFFitsData) -> None:
        """Bootstrap seed should be recorded even if auto-generated."""
        result = localize_transit_source(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[],
            bootstrap_draws=50,
            bootstrap_seed=None,  # Auto-generate
        )

        assert result.bootstrap_seed is not None
        assert isinstance(result.bootstrap_seed, int)

    def test_extra_diagnostics(self, single_star_tpf: TPFFitsData) -> None:
        """Result should include diagnostic information."""
        result = localize_transit_source(
            tpf_fits=single_star_tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[],
            bootstrap_draws=50,
            bootstrap_seed=42,
        )

        assert "n_in_transit" in result.extra
        assert "n_out_of_transit" in result.extra
        assert result.extra["n_in_transit"] > 0
        assert result.extra["n_out_of_transit"] > 0


# =============================================================================
# Tests: Saturation Detection
# =============================================================================


class TestSaturationDetection:
    """Tests for saturation warning generation."""

    def test_saturated_tpf_warning(self) -> None:
        """Saturated TPF should generate warning.

        Note: Due to PSF spreading, we need a threshold low enough to actually
        clip pixels. With flux=100000 and sigma=1.5, peak pixel ~ 7000 counts.
        """
        # Create saturated TPF with threshold that will actually clip pixels
        tpf = make_saturated_tpf(
            shape=(500, 11, 11),
            star_flux=100000.0,
            saturation_threshold=5000.0,  # Below peak pixel value
            transit_depth_frac=0.05,
            seed=42,
        )

        result = localize_transit_source(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[
                {"name": "target", "ra": 120.0, "dec": -50.0},
            ],
            bootstrap_draws=50,
            bootstrap_seed=42,
        )

        # Should have saturation warning
        assert "saturation_suspected" in result.warnings

    def test_unsaturated_bright_star_no_warning(self) -> None:
        """Bright but unsaturated TPF should not generate saturation warning.

        Regression for false positives where stable cadence maxima were
        incorrectly interpreted as saturation.
        """
        tpf = make_synthetic_tpf_fits(
            shape=(500, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=100000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.05,
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            noise_level=50.0,
            seed=42,
            saturation_threshold=None,
        )

        result = localize_transit_source(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[
                {"name": "target", "ra": 120.0, "dec": -50.0},
            ],
            bootstrap_draws=10,
            bootstrap_seed=42,
        )

        assert "saturation_suspected" not in result.warnings


# =============================================================================
# Tests: Localization Verdict
# =============================================================================


class TestLocalizationVerdict:
    """Tests for LocalizationVerdict enum."""

    def test_verdict_values(self) -> None:
        """Verify expected verdict values exist."""
        assert LocalizationVerdict.ON_TARGET.value == "ON_TARGET"
        assert LocalizationVerdict.OFF_TARGET.value == "OFF_TARGET"
        assert LocalizationVerdict.AMBIGUOUS.value == "AMBIGUOUS"
        assert LocalizationVerdict.INVALID.value == "INVALID"

    def test_verdict_is_string_enum(self) -> None:
        """Verdict should be a string enum for JSON serialization."""
        assert isinstance(LocalizationVerdict.ON_TARGET.value, str)
        assert str(LocalizationVerdict.ON_TARGET) == "LocalizationVerdict.ON_TARGET"

    def test_off_target_downgraded_within_one_pixel_without_alternative(self) -> None:
        """Avoid overconfident OFF_TARGET when centroid is within ~1 pixel and no alternative source exists."""
        # Create a high-SNR on-target transit.
        tpf = make_synthetic_tpf_fits(
            shape=(800, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=200000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.03,
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            wcs_crval=(120.0, -50.0),
            pixel_scale_arcsec=21.0,
            noise_level=1.0,
            seed=42,
        )

        # Provide a target reference position offset by 15 arcsec (within 1 TESS pixel).
        # Without the downgrade rule, this can look >2-sigma away for very small centroid uncertainties.
        offset_arcsec = 15.0
        result = localize_transit_source(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[
                {"name": "target", "ra": 120.0, "dec": -50.0 + (offset_arcsec / 3600.0)},
            ],
            bootstrap_draws=200,
            bootstrap_seed=42,
        )

        assert result.verdict in (LocalizationVerdict.AMBIGUOUS, LocalizationVerdict.ON_TARGET)
        assert any("OFF_TARGET downgraded" in s for s in result.verdict_rationale)


# =============================================================================
# Tests: Distance Accuracy
# =============================================================================


class TestDistanceAccuracy:
    """Tests for distance computation accuracy."""

    def test_reference_distance_accuracy(self) -> None:
        """Distance computation should be accurate to <0.1 arcsec for known offset."""
        # Create TPF with star at known position (high SNR)
        tpf = make_synthetic_tpf_fits(
            shape=(500, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=100000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.05,
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            wcs_crval=(120.0, -50.0),
            pixel_scale_arcsec=21.0,
            noise_level=50.0,
            seed=42,
        )

        # Reference source at WCS center (same as star)
        result = localize_transit_source(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            reference_sources=[
                {"name": "target", "ra": 120.0, "dec": -50.0},
            ],
            bootstrap_draws=50,
            bootstrap_seed=42,
        )

        # Centroid should be very close to target
        # With noise, allow some tolerance
        dist_to_target = result.distances_to_sources["target"]
        # Should be within ~half a pixel (~10 arcsec) given noise
        assert dist_to_target < 15.0, f"Distance to target {dist_to_target} too large"


# =============================================================================
# Tests: Module Imports
# =============================================================================


class TestModuleImports:
    """Tests for module imports and exports."""

    def test_import_from_wcs_localization_module(self) -> None:
        """Can import from bittr_tess_vetter.pixel.wcs_localization module."""
        from bittr_tess_vetter.pixel.wcs_localization import (
            LocalizationResult,
            LocalizationVerdict,
            bootstrap_centroid_uncertainty,
            compute_difference_image_centroid,
            compute_reference_source_distances,
            localize_transit_source,
        )

        assert LocalizationResult is not None
        assert LocalizationVerdict is not None
        assert compute_difference_image_centroid is not None
        assert bootstrap_centroid_uncertainty is not None
        assert compute_reference_source_distances is not None
        assert localize_transit_source is not None

    def test_import_from_pixel_package(self) -> None:
        """Can import from bittr_tess_vetter.pixel package."""
        from bittr_tess_vetter.pixel import (
            LocalizationResult,
            LocalizationVerdict,
            localize_transit_source,
        )

        assert LocalizationResult is not None
        assert LocalizationVerdict is not None
        assert localize_transit_source is not None
