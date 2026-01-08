"""Tests for bittr_tess_vetter.pixel.aperture module.

Tests the aperture dependence analysis functionality including:
- Circular aperture mask creation
- Transit mask computation
- Depth measurement from light curves
- Stability metric calculation
- Integration tests with synthetic TPF data

Two main scenarios are tested:
1. Stable case: On-target transit signal with consistent depth across apertures
2. Unstable case: Contamination or off-target source with varying depths
"""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.pixel.aperture import (
    DEFAULT_APERTURE_RADII,
    ApertureDependenceResult,
    TransitParams,
    _compute_depth_from_lightcurve,
    _compute_stability_metric,
    _compute_transit_mask,
    _create_circular_aperture_mask,
    _select_recommended_aperture,
    compute_aperture_dependence,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def basic_transit_params() -> TransitParams:
    """Standard transit parameters for testing."""
    return TransitParams(
        period=2.5,
        t0=1.0,
        duration=0.2,
        depth=0.01,
    )


@pytest.fixture
def time_array() -> np.ndarray:
    """Standard time array covering multiple transits."""
    return np.linspace(0, 10, 500)  # 10 days at ~30 min cadence


@pytest.fixture
def tpf_shape() -> tuple[int, int, int]:
    """Standard TPF dimensions."""
    return (500, 11, 11)  # 500 times, 11x11 pixels


# =============================================================================
# Test TransitParams Model
# =============================================================================


class TestTransitParams:
    """Tests for TransitParams dataclass."""

    def test_transit_params_creation(self) -> None:
        """TransitParams can be created with all fields."""
        params = TransitParams(period=3.5, t0=1000.0, duration=0.25, depth=0.005)
        assert params.period == 3.5
        assert params.t0 == 1000.0
        assert params.duration == 0.25
        assert params.depth == 0.005

    def test_transit_params_default_depth(self) -> None:
        """TransitParams has default depth of 0.01."""
        params = TransitParams(period=2.0, t0=0.0, duration=0.1)
        assert params.depth == 0.01

    def test_transit_params_is_frozen(self) -> None:
        """TransitParams is immutable."""
        params = TransitParams(period=2.0, t0=0.0, duration=0.1)
        with pytest.raises(Exception):
            params.period = 3.0  # type: ignore


# =============================================================================
# Test ApertureDependenceResult Model
# =============================================================================


class TestApertureDependenceResult:
    """Tests for ApertureDependenceResult dataclass."""

    def test_result_creation(self) -> None:
        """ApertureDependenceResult can be created with all fields."""
        result = ApertureDependenceResult(
            depths_by_aperture={1.0: 1000.0, 2.0: 1050.0, 3.0: 1020.0},
            stability_metric=0.85,
            recommended_aperture=2.0,
            depth_variance=625.0,
        )
        assert result.depths_by_aperture == {1.0: 1000.0, 2.0: 1050.0, 3.0: 1020.0}
        assert result.stability_metric == 0.85
        assert result.recommended_aperture == 2.0
        assert result.depth_variance == 625.0

    def test_result_is_frozen(self) -> None:
        """ApertureDependenceResult is immutable."""
        result = ApertureDependenceResult(
            depths_by_aperture={1.0: 1000.0},
            stability_metric=0.9,
            recommended_aperture=1.0,
            depth_variance=0.0,
        )
        with pytest.raises(Exception):
            result.stability_metric = 0.5  # type: ignore


# =============================================================================
# Test Circular Aperture Mask Creation
# =============================================================================


class TestCircularApertureMask:
    """Tests for _create_circular_aperture_mask function."""

    def test_mask_shape(self) -> None:
        """Mask has correct shape."""
        mask = _create_circular_aperture_mask(
            shape=(11, 11),
            radius=3.0,
            center=(5.0, 5.0),
        )
        assert mask.shape == (11, 11)
        assert mask.dtype == np.bool_

    def test_mask_center_pixel(self) -> None:
        """Center pixel is always included."""
        mask = _create_circular_aperture_mask(
            shape=(11, 11),
            radius=1.0,
            center=(5.0, 5.0),
        )
        assert mask[5, 5] is np.True_

    def test_mask_radius_zero(self) -> None:
        """Zero radius includes only center pixel."""
        mask = _create_circular_aperture_mask(
            shape=(11, 11),
            radius=0.0,
            center=(5.0, 5.0),
        )
        # Only center pixel
        assert mask[5, 5] is np.True_
        assert mask.sum() == 1

    def test_mask_large_radius(self) -> None:
        """Large radius includes many pixels."""
        mask = _create_circular_aperture_mask(
            shape=(11, 11),
            radius=5.0,
            center=(5.0, 5.0),
        )
        # Should include most of the 11x11 grid
        assert mask.sum() > 60

    def test_mask_off_center(self) -> None:
        """Off-center aperture works correctly."""
        mask = _create_circular_aperture_mask(
            shape=(11, 11),
            radius=2.0,
            center=(2.0, 8.0),
        )
        assert mask[2, 8] is np.True_
        # Center of grid should not be included
        assert mask[5, 5] is np.False_

    def test_mask_circular_symmetry(self) -> None:
        """Mask is approximately circularly symmetric."""
        mask = _create_circular_aperture_mask(
            shape=(21, 21),
            radius=5.0,
            center=(10.0, 10.0),
        )
        # Check symmetry along axes
        assert mask[5, 10] == mask[15, 10]  # vertical
        assert mask[10, 5] == mask[10, 15]  # horizontal
        assert mask[6, 6] == mask[14, 14]  # diagonal


# =============================================================================
# Test Transit Mask Computation
# =============================================================================


class TestTransitMask:
    """Tests for _compute_transit_mask function."""

    def test_transit_mask_shape(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Transit mask has same shape as time array."""
        mask = _compute_transit_mask(time_array, basic_transit_params)
        assert mask.shape == time_array.shape
        assert mask.dtype == np.bool_

    def test_transit_mask_has_transits(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Transit mask identifies some in-transit points."""
        mask = _compute_transit_mask(time_array, basic_transit_params)
        assert np.any(mask), "No transits detected"
        assert np.any(~mask), "All points marked as in-transit"

    def test_transit_mask_periodic(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Transits occur at correct period."""
        mask = _compute_transit_mask(time_array, basic_transit_params)

        # Find transit center times
        in_transit_times = time_array[mask]
        if len(in_transit_times) > 1:
            # Check that transits are separated by approximately the period
            centers = []
            transitions = np.diff(mask.astype(int))
            start_indices = np.where(transitions == 1)[0]
            end_indices = np.where(transitions == -1)[0]

            for start, end in zip(start_indices, end_indices, strict=False):
                center_idx = (start + end) // 2
                if center_idx < len(time_array):
                    centers.append(time_array[center_idx])

            if len(centers) >= 2:
                separations = np.diff(centers)
                # Separations should be approximately the period
                for sep in separations:
                    assert abs(sep - basic_transit_params.period) < 0.5

    def test_transit_mask_at_epoch(self, basic_transit_params: TransitParams) -> None:
        """Transit centered at t0."""
        time = np.array([0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])
        mask = _compute_transit_mask(time, basic_transit_params)

        # t0=1.0, duration=0.2, half-duration=0.1
        # Points within +/- 0.1 of t0 are in transit: [0.9, 1.1] inclusive
        # 0.9 is exactly at boundary (|0.9 - 1.0| = 0.1 <= 0.1), should be in
        # But phase computation uses modulo, and 0.9 might be on edge
        # The center point should definitely be in transit
        assert mask[2] is np.True_  # 1.0 (center)
        assert mask[1] is np.True_  # 0.95 (within half-duration)
        assert mask[3] is np.True_  # 1.05 (within half-duration)
        assert mask[5] is np.False_  # 1.2 (outside transit)
        assert mask[6] is np.False_  # 1.3 (outside transit)


# =============================================================================
# Test Depth Computation from Light Curve
# =============================================================================


class TestDepthFromLightcurve:
    """Tests for _compute_depth_from_lightcurve function."""

    def test_depth_calculation_basic(self) -> None:
        """Basic depth calculation works."""
        # Out of transit: 1000, In transit: 990 -> 1% depth
        flux = np.array([1000.0, 1000.0, 990.0, 1000.0, 1000.0])
        in_transit = np.array([False, False, True, False, False])

        depth_ppm = _compute_depth_from_lightcurve(flux, in_transit)
        assert abs(depth_ppm - 10000.0) < 100  # ~1% = 10000 ppm

    def test_depth_calculation_no_transit(self) -> None:
        """No depth when flux is constant."""
        flux = np.array([1000.0, 1000.0, 1000.0, 1000.0])
        in_transit = np.array([False, False, True, True])

        depth_ppm = _compute_depth_from_lightcurve(flux, in_transit)
        assert abs(depth_ppm) < 1.0  # Essentially zero

    def test_depth_calculation_deep_transit(self) -> None:
        """Deep transit depth calculation."""
        # Out of transit: 1000, In transit: 900 -> 10% depth
        flux = np.array([1000.0, 1000.0, 900.0, 900.0, 1000.0])
        in_transit = np.array([False, False, True, True, False])

        depth_ppm = _compute_depth_from_lightcurve(flux, in_transit)
        assert abs(depth_ppm - 100000.0) < 1000  # ~10% = 100000 ppm

    def test_depth_insufficient_data_raises(self) -> None:
        """Raises when no in-transit or out-of-transit data."""
        flux = np.array([1000.0, 1000.0])

        with pytest.raises(ValueError, match="Insufficient"):
            _compute_depth_from_lightcurve(flux, np.array([True, True]))

        with pytest.raises(ValueError, match="Insufficient"):
            _compute_depth_from_lightcurve(flux, np.array([False, False]))

    def test_depth_handles_nan(self) -> None:
        """NaN values are handled gracefully."""
        flux = np.array([1000.0, np.nan, 990.0, 1000.0, 1000.0])
        in_transit = np.array([False, False, True, False, False])

        depth_ppm = _compute_depth_from_lightcurve(flux, in_transit)
        # Should still compute reasonable depth, ignoring NaN
        assert np.isfinite(depth_ppm)


# =============================================================================
# Test Stability Metric
# =============================================================================


class TestStabilityMetric:
    """Tests for _compute_stability_metric function."""

    def test_perfect_stability(self) -> None:
        """Identical depths give stability of 1.0."""
        depths = [1000.0, 1000.0, 1000.0, 1000.0]
        stability = _compute_stability_metric(depths)
        assert stability == pytest.approx(1.0, abs=0.01)

    def test_high_stability(self) -> None:
        """Low variance depths give high stability."""
        depths = [1000.0, 1010.0, 990.0, 1005.0]  # ~1% variation
        stability = _compute_stability_metric(depths)
        assert stability > 0.8

    def test_low_stability(self) -> None:
        """High variance depths give low stability."""
        depths = [100.0, 1000.0, 50.0, 500.0]  # Very high variation
        stability = _compute_stability_metric(depths)
        assert stability < 0.6

    def test_single_depth_stable(self) -> None:
        """Single depth is trivially stable."""
        stability = _compute_stability_metric([1000.0])
        assert stability == 1.0

    def test_two_depths_varied(self) -> None:
        """Two very different depths give low stability."""
        stability = _compute_stability_metric([100.0, 1000.0])
        assert stability < 0.5

    def test_stability_range(self) -> None:
        """Stability is always between 0 and 1."""
        test_cases = [
            [1.0, 2.0, 3.0],
            [0.0, 1.0],
            [1000.0, 1001.0],
            [100.0, 200.0, 300.0, 400.0, 500.0],
        ]
        for depths in test_cases:
            stability = _compute_stability_metric(depths)
            assert 0.0 <= stability <= 1.0


# =============================================================================
# Test Recommended Aperture Selection
# =============================================================================


class TestRecommendedAperture:
    """Tests for _select_recommended_aperture function."""

    def test_stable_selects_median_aperture(self) -> None:
        """Stable depths select median-sized aperture."""
        depths = {1.0: 1000.0, 2.0: 1005.0, 3.0: 1010.0}
        aperture = _select_recommended_aperture(depths, stability_metric=0.9)
        assert aperture == 2.0  # median of [1.0, 2.0, 3.0]

    def test_unstable_selects_median_depth(self) -> None:
        """Unstable depths select aperture with median depth."""
        depths = {1.0: 500.0, 2.0: 1000.0, 3.0: 2000.0}
        aperture = _select_recommended_aperture(depths, stability_metric=0.3)
        assert aperture == 2.0  # depth 1000 is median

    def test_single_aperture(self) -> None:
        """Single aperture is always recommended."""
        depths = {2.5: 1000.0}
        aperture = _select_recommended_aperture(depths, stability_metric=0.5)
        assert aperture == 2.5

    def test_empty_returns_default(self) -> None:
        """Empty dict returns default aperture."""
        aperture = _select_recommended_aperture({}, stability_metric=0.5)
        assert aperture == 2.0  # Fallback default


# =============================================================================
# Test Main Function - Stable Signal Case
# =============================================================================


class TestComputeApertureDependenceStable:
    """Tests for stable on-target transit signal scenario."""

    def _create_stable_tpf(
        self,
        shape: tuple[int, int, int],
        time: np.ndarray,
        transit_params: TransitParams,
        depth: float = 0.01,
    ) -> np.ndarray:
        """Create synthetic TPF with on-target transit signal.

        The transit signal is centered in the TPF and affects all pixels
        uniformly when normalized, resulting in stable depth measurements
        across all aperture sizes.
        """
        n_times, n_rows, n_cols = shape
        center_row = n_rows // 2
        center_col = n_cols // 2

        # Create base flux with Gaussian PSF
        y, x = np.ogrid[:n_rows, :n_cols]
        distance_sq = (y - center_row) ** 2 + (x - center_col) ** 2
        psf = np.exp(-distance_sq / 8.0)  # Gaussian with sigma ~2 pixels

        # Create 3D TPF
        tpf = np.zeros(shape)
        for t in range(n_times):
            tpf[t] = psf * 10000.0  # Base flux

        # Add transit signal - affects all pixels equally (fractional depth)
        in_transit = _compute_transit_mask(time, transit_params)
        for t in range(n_times):
            if in_transit[t]:
                tpf[t] *= 1.0 - depth

        return tpf

    def test_stable_signal_high_stability(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """On-target signal gives high stability metric."""
        tpf = self._create_stable_tpf(
            shape=(len(time_array), 11, 11),
            time=time_array,
            transit_params=basic_transit_params,
            depth=0.01,
        )

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
        )

        # Stability should be high (>0.7) for on-target signal
        assert result.stability_metric > 0.7
        # All depths should be similar (within 20% of each other)
        depths = list(result.depths_by_aperture.values())
        mean_depth = np.mean(depths)
        for d in depths:
            assert abs(d - mean_depth) / mean_depth < 0.2

    def test_stable_signal_correct_depth(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """On-target signal measures correct depth."""
        expected_depth_ppm = 10000.0  # 1% depth

        tpf = self._create_stable_tpf(
            shape=(len(time_array), 11, 11),
            time=time_array,
            transit_params=basic_transit_params,
            depth=0.01,
        )

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
        )

        # Depths should be close to expected value
        for depth in result.depths_by_aperture.values():
            assert abs(depth - expected_depth_ppm) / expected_depth_ppm < 0.1


# =============================================================================
# Test Main Function - Unstable Signal Case
# =============================================================================


class TestComputeApertureDependenceUnstable:
    """Tests for unstable off-target or contaminated signal scenario."""

    def _create_unstable_tpf(
        self,
        shape: tuple[int, int, int],
        time: np.ndarray,
        transit_params: TransitParams,
        depth: float = 0.01,
        offset: tuple[float, float] = (3.0, 3.0),
    ) -> np.ndarray:
        """Create synthetic TPF with off-target transit signal.

        The transit signal is offset from the TPF center, causing different
        depths to be measured depending on whether the aperture includes
        the actual signal source.
        """
        n_times, n_rows, n_cols = shape
        center_row = n_rows // 2
        center_col = n_cols // 2

        # Target star PSF (center)
        y, x = np.ogrid[:n_rows, :n_cols]
        distance_sq_target = (y - center_row) ** 2 + (x - center_col) ** 2
        psf_target = np.exp(-distance_sq_target / 8.0) * 10000.0

        # Contaminating star PSF (offset)
        contam_row = center_row + offset[0]
        contam_col = center_col + offset[1]
        distance_sq_contam = (y - contam_row) ** 2 + (x - contam_col) ** 2
        psf_contam = np.exp(-distance_sq_contam / 8.0) * 5000.0

        # Create 3D TPF
        tpf = np.zeros(shape)
        in_transit = _compute_transit_mask(time, transit_params)

        for t in range(n_times):
            # Target star (constant)
            tpf[t] = psf_target.copy()

            # Contaminating star (transits)
            contam_flux = psf_contam.copy()
            if in_transit[t]:
                contam_flux *= 1.0 - depth

            tpf[t] += contam_flux

        return tpf

    def test_unstable_signal_low_stability(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Off-target signal gives lower stability than on-target."""
        tpf = self._create_unstable_tpf(
            shape=(len(time_array), 11, 11),
            time=time_array,
            transit_params=basic_transit_params,
            depth=0.05,
            offset=(3.0, 3.0),
        )

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
        )

        # Stability should be notably below 1.0 for off-target signal
        # The key property is that depths vary across apertures
        assert result.stability_metric < 0.85
        # And variance should be significant
        assert result.depth_variance > 100000  # Significant variation in ppm^2

    def test_unstable_signal_varying_depths(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Off-target signal shows varying depths across apertures."""
        tpf = self._create_unstable_tpf(
            shape=(len(time_array), 11, 11),
            time=time_array,
            transit_params=basic_transit_params,
            depth=0.05,
            offset=(3.0, 3.0),
        )

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
        )

        # Depths should vary significantly
        depths = list(result.depths_by_aperture.values())
        depth_range = max(depths) - min(depths)
        mean_depth = np.mean(depths)

        # Variance should be notable (range > 30% of mean)
        assert depth_range / mean_depth > 0.3 or result.stability_metric < 0.7


# =============================================================================
# Test Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input validation in compute_aperture_dependence."""

    def test_invalid_tpf_shape(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Raises for non-3D TPF data."""
        tpf_2d = np.ones((11, 11))

        with pytest.raises(ValueError, match="must be 3D"):
            compute_aperture_dependence(tpf_2d, time_array, basic_transit_params)

    def test_mismatched_time_length(self, basic_transit_params: TransitParams) -> None:
        """Raises when time length doesn't match TPF."""
        tpf = np.ones((100, 11, 11))
        time = np.linspace(0, 10, 50)  # Wrong length

        with pytest.raises(ValueError, match="must match"):
            compute_aperture_dependence(tpf, time, basic_transit_params)

    def test_insufficient_time_points(self, basic_transit_params: TransitParams) -> None:
        """Raises when fewer than 10 time points."""
        tpf = np.ones((5, 11, 11))
        time = np.linspace(0, 1, 5)

        with pytest.raises(ValueError, match="at least 10"):
            compute_aperture_dependence(tpf, time, basic_transit_params)

    def test_insufficient_transit_coverage(self) -> None:
        """Raises when transit doesn't fall within time range."""
        tpf = np.ones((100, 11, 11))
        # Very short time span that won't contain a transit
        time = np.linspace(0.5, 0.6, 100)
        # Transit centered at t0=0.0 with period much longer than time span
        # Time span is 0.5-0.6, so phase will be 0.5-0.6 which is far from transit
        params = TransitParams(period=10.0, t0=0.0, duration=0.1)

        with pytest.raises(ValueError, match="Insufficient"):
            compute_aperture_dependence(tpf, time, params)


# =============================================================================
# Test Custom Parameters
# =============================================================================


class TestCustomParameters:
    """Tests for custom aperture radii and center positions."""

    def test_custom_aperture_radii(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Custom aperture radii are used."""
        tpf = np.ones((len(time_array), 11, 11)) * 1000.0

        # Add transit signal
        in_transit = _compute_transit_mask(time_array, basic_transit_params)
        tpf[in_transit] *= 0.99

        custom_radii = [0.5, 1.0, 4.0]
        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
            aperture_radii=custom_radii,
        )

        # Result should have exactly the requested apertures
        assert set(result.depths_by_aperture.keys()) == set(custom_radii)

    def test_custom_center(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Custom center position is used."""
        tpf = np.ones((len(time_array), 11, 11)) * 1000.0

        # Add signal at corner instead of center
        tpf[:, :3, :3] = 10000.0
        in_transit = _compute_transit_mask(time_array, basic_transit_params)
        tpf[in_transit, :3, :3] *= 0.99

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
            center=(1.0, 1.0),  # Near corner where signal is
        )

        # Should detect the transit
        assert len(result.depths_by_aperture) > 0
        # Depths should be non-zero
        for depth in result.depths_by_aperture.values():
            assert abs(depth) > 100  # At least some depth detected

    def test_default_aperture_radii(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Default aperture radii are used when not specified."""
        tpf = np.ones((len(time_array), 11, 11)) * 1000.0
        in_transit = _compute_transit_mask(time_array, basic_transit_params)
        tpf[in_transit] *= 0.99

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
        )

        # Should use default radii
        expected_radii = set(DEFAULT_APERTURE_RADII)
        assert set(result.depths_by_aperture.keys()) == expected_radii


# =============================================================================
# Test Result Properties
# =============================================================================


class TestResultProperties:
    """Tests for properties of ApertureDependenceResult."""

    def test_result_has_all_fields(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Result contains all required fields."""
        tpf = np.ones((len(time_array), 11, 11)) * 1000.0
        in_transit = _compute_transit_mask(time_array, basic_transit_params)
        tpf[in_transit] *= 0.99

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
        )

        assert isinstance(result.depths_by_aperture, dict)
        assert isinstance(result.stability_metric, float)
        assert isinstance(result.recommended_aperture, float)
        assert isinstance(result.depth_variance, float)

    def test_stability_metric_in_range(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Stability metric is between 0 and 1."""
        tpf = np.ones((len(time_array), 11, 11)) * 1000.0
        in_transit = _compute_transit_mask(time_array, basic_transit_params)
        tpf[in_transit] *= 0.99

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
        )

        assert 0.0 <= result.stability_metric <= 1.0

    def test_recommended_aperture_in_results(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Recommended aperture is one of the measured apertures."""
        tpf = np.ones((len(time_array), 11, 11)) * 1000.0
        in_transit = _compute_transit_mask(time_array, basic_transit_params)
        tpf[in_transit] *= 0.99

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
        )

        assert result.recommended_aperture in result.depths_by_aperture

    def test_depth_variance_non_negative(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Depth variance is non-negative."""
        tpf = np.ones((len(time_array), 11, 11)) * 1000.0
        in_transit = _compute_transit_mask(time_array, basic_transit_params)
        tpf[in_transit] *= 0.99

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
        )

        assert result.depth_variance >= 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete aperture dependence workflow."""

    def test_discriminates_stable_vs_unstable(self, time_array: np.ndarray) -> None:
        """Can discriminate between stable and unstable signals."""
        params = TransitParams(period=2.0, t0=1.0, duration=0.2, depth=0.02)
        n_times = len(time_array)

        # Create stable (on-target) TPF
        center_row, center_col = 5, 5
        y, x = np.ogrid[:11, :11]
        psf = np.exp(-((y - center_row) ** 2 + (x - center_col) ** 2) / 8.0)

        stable_tpf = np.zeros((n_times, 11, 11))
        in_transit = _compute_transit_mask(time_array, params)
        for t in range(n_times):
            stable_tpf[t] = psf * 10000.0
            if in_transit[t]:
                stable_tpf[t] *= 0.98

        # Create unstable (off-target) TPF
        offset_row, offset_col = 8, 8
        psf_target = psf * 10000.0
        psf_contam = np.exp(-((y - offset_row) ** 2 + (x - offset_col) ** 2) / 8.0) * 5000.0

        unstable_tpf = np.zeros((n_times, 11, 11))
        for t in range(n_times):
            unstable_tpf[t] = psf_target + psf_contam
            if in_transit[t]:
                unstable_tpf[t] -= psf_contam * 0.1  # Only contaminant transits

        stable_result = compute_aperture_dependence(stable_tpf, time_array, params)
        unstable_result = compute_aperture_dependence(unstable_tpf, time_array, params)

        # Stable should have higher stability than unstable
        assert stable_result.stability_metric > unstable_result.stability_metric

    def test_handles_noisy_data(
        self, time_array: np.ndarray, basic_transit_params: TransitParams
    ) -> None:
        """Handles realistic noisy data."""
        n_times = len(time_array)

        # Create TPF with noise
        tpf = np.ones((n_times, 11, 11)) * 1000.0
        np.random.seed(42)
        noise = np.random.normal(0, 10, tpf.shape)  # 1% noise
        tpf += noise

        # Add transit signal
        in_transit = _compute_transit_mask(time_array, basic_transit_params)
        tpf[in_transit] *= 0.99

        result = compute_aperture_dependence(
            tpf_data=tpf,
            time=time_array,
            transit_params=basic_transit_params,
        )

        # Should still produce valid results
        assert len(result.depths_by_aperture) >= 2
        assert 0.0 <= result.stability_metric <= 1.0
        assert np.isfinite(result.depth_variance)
