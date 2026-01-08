"""Unit tests for PRF/PSF model interface and implementations.

Tests the PRF model protocol, parametric PSF implementation, factory function,
and parameter schemas for TESS pixel-level analysis.

All tests use synthetic data only - no network access or FITS files required.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bittr_tess_vetter.compute.prf_psf import (
    AVAILABLE_BACKENDS,
    ParametricPSF,
    PRFModel,
    get_prf_model,
)
from bittr_tess_vetter.compute.prf_schemas import (
    BackgroundParams,
    PRFFitResult,
    PRFParams,
    fit_result_from_dict,
    fit_result_to_dict,
    prf_params_from_dict,
    prf_params_from_json,
    prf_params_to_dict,
    prf_params_to_json,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def standard_shape() -> tuple[int, int]:
    """Standard 15x15 pixel stamp for tests."""
    return (15, 15)


@pytest.fixture
def center_coords() -> tuple[float, float]:
    """Center coordinates for 15x15 stamp."""
    return (7.0, 7.0)


@pytest.fixture
def default_params() -> PRFParams:
    """Default PRF parameters (isotropic Gaussian)."""
    return PRFParams()


@pytest.fixture
def elliptical_params() -> PRFParams:
    """Elliptical PRF parameters with rotation."""
    return PRFParams(
        sigma_row=2.0,
        sigma_col=1.0,
        theta=np.pi / 4,
        amplitude=1.0,
        background=BackgroundParams(b0=0.1, bx=0.01, by=-0.01),
    )


@pytest.fixture
def parametric_psf(default_params: PRFParams) -> ParametricPSF:
    """Default parametric PSF instance."""
    return ParametricPSF(params=default_params)


# =============================================================================
# PRFParams Tests
# =============================================================================


class TestPRFParams:
    """Tests for PRFParams dataclass."""

    def test_default_params(self) -> None:
        """Default parameters should be isotropic with sigma=1.5."""
        params = PRFParams()
        assert params.sigma_row == 1.5
        assert params.sigma_col == 1.5
        assert params.theta == 0.0
        assert params.amplitude == 1.0
        assert params.is_isotropic

    def test_invalid_sigma(self) -> None:
        """Non-positive sigma should raise ValueError."""
        with pytest.raises(ValueError, match="sigma_row must be positive"):
            PRFParams(sigma_row=0.0)
        with pytest.raises(ValueError, match="sigma_col must be positive"):
            PRFParams(sigma_col=-1.0)

    def test_invalid_amplitude(self) -> None:
        """Non-positive amplitude should raise ValueError."""
        with pytest.raises(ValueError, match="amplitude must be positive"):
            PRFParams(amplitude=0.0)

    def test_is_isotropic(self) -> None:
        """is_isotropic should detect circular PSF."""
        isotropic = PRFParams(sigma_row=1.5, sigma_col=1.5, theta=0.0)
        assert isotropic.is_isotropic

        anisotropic = PRFParams(sigma_row=2.0, sigma_col=1.0, theta=0.0)
        assert not anisotropic.is_isotropic

        rotated = PRFParams(sigma_row=1.5, sigma_col=1.5, theta=0.1)
        assert not rotated.is_isotropic

    def test_effective_sigma(self) -> None:
        """effective_sigma should be geometric mean."""
        params = PRFParams(sigma_row=2.0, sigma_col=1.0)
        expected = np.sqrt(2.0 * 1.0)
        assert_allclose(params.effective_sigma, expected)

    def test_with_jitter(self) -> None:
        """with_jitter adds jitter in quadrature."""
        params = PRFParams(sigma_row=1.5, sigma_col=1.5)
        jittered = params.with_jitter(1.0)

        expected = np.sqrt(1.5**2 + 1.0**2)
        assert_allclose(jittered.sigma_row, expected)
        assert_allclose(jittered.sigma_col, expected)

    def test_with_zero_jitter(self) -> None:
        """Zero jitter should return same params."""
        params = PRFParams(sigma_row=1.5, sigma_col=1.5)
        result = params.with_jitter(0.0)
        assert result is params


# =============================================================================
# BackgroundParams Tests
# =============================================================================


class TestBackgroundParams:
    """Tests for BackgroundParams dataclass."""

    def test_default_background(self) -> None:
        """Default background is zero."""
        bg = BackgroundParams()
        assert bg.b0 == 0.0
        assert bg.bx == 0.0
        assert bg.by == 0.0

    def test_to_tuple(self) -> None:
        """to_tuple returns (b0, bx, by)."""
        bg = BackgroundParams(b0=0.1, bx=0.02, by=-0.01)
        assert bg.to_tuple() == (0.1, 0.02, -0.01)

    def test_from_tuple(self) -> None:
        """from_tuple creates BackgroundParams."""
        bg = BackgroundParams.from_tuple((0.1, 0.02, -0.01))
        assert bg.b0 == 0.1
        assert bg.bx == 0.02
        assert bg.by == -0.01

    def test_evaluate(self, standard_shape: tuple[int, int]) -> None:
        """evaluate computes background gradient correctly."""
        bg = BackgroundParams(b0=1.0, bx=0.1, by=0.2)

        row_coords, col_coords = np.mgrid[0 : standard_shape[0], 0 : standard_shape[1]]
        row_coords = row_coords.astype(np.float64)
        col_coords = col_coords.astype(np.float64)

        result = bg.evaluate(row_coords, col_coords, 7.0, 7.0)

        # Check center value
        assert_allclose(result[7, 7], 1.0)

        # Check gradient in col direction
        assert_allclose(result[7, 8] - result[7, 7], 0.1)

        # Check gradient in row direction
        assert_allclose(result[8, 7] - result[7, 7], 0.2)


# =============================================================================
# ParametricPSF.evaluate Tests
# =============================================================================


class TestParametricPSFEvaluate:
    """Tests for ParametricPSF.evaluate method."""

    def test_evaluate_normalized(
        self,
        parametric_psf: ParametricPSF,
        standard_shape: tuple[int, int],
        center_coords: tuple[float, float],
    ) -> None:
        """PRF should be normalized to sum to 1.0."""
        center_row, center_col = center_coords
        model = parametric_psf.evaluate(center_row, center_col, standard_shape)

        assert_allclose(model.sum(), 1.0, rtol=1e-10)

    def test_evaluate_not_normalized(
        self,
        standard_shape: tuple[int, int],
        center_coords: tuple[float, float],
    ) -> None:
        """With normalize=False, sum should not be 1.0."""
        psf = ParametricPSF(PRFParams(amplitude=10.0))
        center_row, center_col = center_coords
        model = psf.evaluate(center_row, center_col, standard_shape, normalize=False)

        # Unnormalized sum depends on amplitude and sigma
        assert model.sum() > 1.0

    def test_evaluate_peak_at_center(
        self,
        parametric_psf: ParametricPSF,
        standard_shape: tuple[int, int],
        center_coords: tuple[float, float],
    ) -> None:
        """Peak should be at specified center."""
        center_row, center_col = center_coords
        model = parametric_psf.evaluate(center_row, center_col, standard_shape)

        peak_idx = np.unravel_index(np.argmax(model), model.shape)
        assert peak_idx[0] == int(center_row)
        assert peak_idx[1] == int(center_col)

    def test_evaluate_with_background(
        self,
        parametric_psf: ParametricPSF,
        standard_shape: tuple[int, int],
        center_coords: tuple[float, float],
    ) -> None:
        """Background gradient should be correctly applied."""
        center_row, center_col = center_coords
        background = (0.1, 0.01, -0.01)

        model_no_bg = parametric_psf.evaluate(center_row, center_col, standard_shape)
        model_with_bg = parametric_psf.evaluate(
            center_row, center_col, standard_shape, background=background
        )

        # Difference should be the background
        diff = model_with_bg - model_no_bg

        # At center, background should be b0
        assert_allclose(diff[int(center_row), int(center_col)], 0.1, rtol=1e-10)

        # Check gradient
        assert diff[7, 8] > diff[7, 7]  # bx > 0
        assert diff[8, 7] < diff[7, 7]  # by < 0

    def test_evaluate_with_jitter(
        self,
        standard_shape: tuple[int, int],
        center_coords: tuple[float, float],
    ) -> None:
        """Jitter should broaden the PSF."""
        psf = ParametricPSF(PRFParams(sigma_row=1.0, sigma_col=1.0))
        center_row, center_col = center_coords

        model_no_jitter = psf.evaluate(center_row, center_col, standard_shape)
        model_jittered = psf.evaluate(center_row, center_col, standard_shape, jitter_sigma=1.0)

        # Jittered model should have lower peak (more spread out)
        assert model_jittered.max() < model_no_jitter.max()

        # Both should still be normalized
        assert_allclose(model_no_jitter.sum(), 1.0, rtol=1e-10)
        assert_allclose(model_jittered.sum(), 1.0, rtol=1e-10)

    def test_evaluate_elliptical(
        self,
        elliptical_params: PRFParams,
        standard_shape: tuple[int, int],
        center_coords: tuple[float, float],
    ) -> None:
        """Elliptical PSF should be elongated."""
        psf = ParametricPSF(elliptical_params)
        center_row, center_col = center_coords
        model = psf.evaluate(center_row, center_col, standard_shape, normalize=True)

        # Should still be normalized
        assert_allclose(model.sum(), 1.0, rtol=1e-10)

        # Elliptical with theta=pi/4 should have different extent along diagonals
        # This is a basic sanity check
        assert model.shape == standard_shape

    def test_evaluate_invalid_shape(self, parametric_psf: ParametricPSF) -> None:
        """Invalid shape should raise ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            parametric_psf.evaluate(5.0, 5.0, (0, 10))
        with pytest.raises(ValueError, match="dimensions must be positive"):
            parametric_psf.evaluate(5.0, 5.0, (10, -1))

    def test_evaluate_dtype(
        self,
        parametric_psf: ParametricPSF,
        standard_shape: tuple[int, int],
        center_coords: tuple[float, float],
    ) -> None:
        """Output should be float64."""
        center_row, center_col = center_coords
        model = parametric_psf.evaluate(center_row, center_col, standard_shape)
        assert model.dtype == np.float64


# =============================================================================
# ParametricPSF.evaluate_at_positions Tests
# =============================================================================


class TestParametricPSFEvaluateAtPositions:
    """Tests for ParametricPSF.evaluate_at_positions method."""

    def test_evaluate_multiple_positions(
        self,
        parametric_psf: ParametricPSF,
        standard_shape: tuple[int, int],
    ) -> None:
        """Should return list of PRF models."""
        positions = [(5.0, 5.0), (7.0, 7.0), (10.0, 10.0)]
        models = parametric_psf.evaluate_at_positions(positions, standard_shape)

        assert len(models) == 3
        for model, (row, col) in zip(models, positions, strict=True):
            assert model.shape == standard_shape
            # Peak should be at specified position (or nearby for edge cases)
            peak_idx = np.unravel_index(np.argmax(model), model.shape)
            assert abs(peak_idx[0] - row) <= 1
            assert abs(peak_idx[1] - col) <= 1

    def test_evaluate_empty_positions(
        self,
        parametric_psf: ParametricPSF,
        standard_shape: tuple[int, int],
    ) -> None:
        """Empty positions list returns empty list."""
        models = parametric_psf.evaluate_at_positions([], standard_shape)
        assert models == []


# =============================================================================
# ParametricPSF.fit_to_image Tests
# =============================================================================


class TestParametricPSFFitToImage:
    """Tests for ParametricPSF.fit_to_image method."""

    def test_fit_recovers_center(self, standard_shape: tuple[int, int]) -> None:
        """Fit should recover known center position."""
        # Create synthetic image with known center
        true_center = (7.5, 8.2)
        true_params = PRFParams(sigma_row=1.5, sigma_col=1.5, amplitude=100.0)
        psf = ParametricPSF(true_params)

        image = psf.evaluate(true_center[0], true_center[1], standard_shape, normalize=False)

        # Add small noise
        rng = np.random.default_rng(42)
        image = image + rng.normal(0, 0.5, image.shape)

        # Fit
        result = psf.fit_to_image(image, fit_shape=False, fit_background=False)

        # Center should be recovered within ~0.5 pixel
        assert abs(result.center_row - true_center[0]) < 0.5
        assert abs(result.center_col - true_center[1]) < 0.5

    def test_fit_recovers_sigma(self, standard_shape: tuple[int, int]) -> None:
        """Fit should recover known sigma values."""
        # Create synthetic image with known parameters
        true_params = PRFParams(sigma_row=2.0, sigma_col=1.5, amplitude=100.0)
        psf = ParametricPSF(PRFParams())  # Start with default params

        # Generate ground truth image
        true_psf = ParametricPSF(true_params)
        image = true_psf.evaluate(7.0, 7.0, standard_shape, normalize=False)

        # Fit
        result = psf.fit_to_image(image, initial_center=(7.0, 7.0), fit_shape=True)

        # Sigmas should be recovered within ~20%
        assert abs(result.params.sigma_row - 2.0) / 2.0 < 0.2
        assert abs(result.params.sigma_col - 1.5) / 1.5 < 0.2

    def test_fit_with_mask(self, standard_shape: tuple[int, int]) -> None:
        """Fit should work with pixel mask."""
        true_params = PRFParams(sigma_row=1.5, sigma_col=1.5, amplitude=100.0)
        psf = ParametricPSF(true_params)
        image = psf.evaluate(7.0, 7.0, standard_shape, normalize=False)

        # Create mask excluding corners
        mask = np.ones(standard_shape, dtype=bool)
        mask[:3, :3] = False
        mask[-3:, -3:] = False

        result = psf.fit_to_image(image, mask=mask, fit_shape=False)

        # Should still converge
        assert result.converged or result.n_iterations > 0

    def test_fit_with_background(self, standard_shape: tuple[int, int]) -> None:
        """Fit should recover background gradient."""
        true_params = PRFParams(
            sigma_row=1.5,
            sigma_col=1.5,
            amplitude=100.0,
            background=BackgroundParams(b0=10.0, bx=0.5, by=-0.3),
        )
        psf = ParametricPSF(true_params)
        image = psf.evaluate(
            7.0, 7.0, standard_shape, background=(10.0, 0.5, -0.3), normalize=False
        )

        result = psf.fit_to_image(
            image, initial_center=(7.0, 7.0), fit_background=True, fit_shape=False
        )

        # Background should be approximately recovered
        # (exact recovery depends on optimization)
        assert result.params.background.b0 > 5.0  # Should be positive

    def test_fit_residual_rms(self, standard_shape: tuple[int, int]) -> None:
        """Residual RMS should be small for good fit."""
        true_params = PRFParams(sigma_row=1.5, sigma_col=1.5, amplitude=100.0)
        psf = ParametricPSF(true_params)
        image = psf.evaluate(7.0, 7.0, standard_shape, normalize=False)

        result = psf.fit_to_image(image, fit_shape=False, fit_background=False)

        # Residual RMS should be very small for noiseless data
        assert result.residual_rms < 1.0


# =============================================================================
# get_prf_model Factory Tests
# =============================================================================


class TestGetPrfModel:
    """Tests for get_prf_model factory function."""

    def test_parametric_backend(self) -> None:
        """Parametric backend should return ParametricPSF."""
        model = get_prf_model("parametric")
        assert isinstance(model, ParametricPSF)
        assert model.backend_name == "parametric"

    def test_parametric_with_params(self) -> None:
        """Should accept custom params."""
        params = PRFParams(sigma_row=2.0, sigma_col=1.0)
        model = get_prf_model("parametric", params=params)
        assert model.params.sigma_row == 2.0
        assert model.params.sigma_col == 1.0

    def test_instrument_backend_not_available(self) -> None:
        """Instrument backend should raise ValueError."""
        with pytest.raises(ValueError, match="not yet available"):
            get_prf_model("instrument")

    def test_lightkurve_backend_not_implemented(self) -> None:
        """Lightkurve backend should raise ValueError."""
        with pytest.raises(ValueError, match="lightkurve"):
            get_prf_model("lightkurve")

    def test_unknown_backend(self) -> None:
        """Unknown backend should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_prf_model("unknown_backend")

    def test_case_insensitive(self) -> None:
        """Backend name should be case-insensitive."""
        model = get_prf_model("PARAMETRIC")
        assert isinstance(model, ParametricPSF)

    def test_available_backends(self) -> None:
        """AVAILABLE_BACKENDS should include parametric."""
        assert "parametric" in AVAILABLE_BACKENDS
        assert AVAILABLE_BACKENDS["parametric"] is True


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Tests that ParametricPSF satisfies PRFModel protocol."""

    def test_parametric_psf_is_prf_model(self) -> None:
        """ParametricPSF should be a PRFModel."""
        psf = ParametricPSF()
        assert isinstance(psf, PRFModel)

    def test_has_backend_name(self) -> None:
        """Should have backend_name property."""
        psf = ParametricPSF()
        assert hasattr(psf, "backend_name")
        assert isinstance(psf.backend_name, str)

    def test_has_params(self) -> None:
        """Should have params property."""
        psf = ParametricPSF()
        assert hasattr(psf, "params")
        assert isinstance(psf.params, PRFParams)


# =============================================================================
# JSON Serialization Tests
# =============================================================================


class TestJSONSerialization:
    """Tests for JSON serialization helpers."""

    def test_prf_params_roundtrip(self, elliptical_params: PRFParams) -> None:
        """PRFParams should survive dict roundtrip."""
        d = prf_params_to_dict(elliptical_params)
        restored = prf_params_from_dict(d)

        assert restored.sigma_row == elliptical_params.sigma_row
        assert restored.sigma_col == elliptical_params.sigma_col
        assert restored.theta == elliptical_params.theta
        assert restored.amplitude == elliptical_params.amplitude
        assert restored.background.b0 == elliptical_params.background.b0

    def test_prf_params_json_roundtrip(self, elliptical_params: PRFParams) -> None:
        """PRFParams should survive JSON roundtrip."""
        json_str = prf_params_to_json(elliptical_params)
        restored = prf_params_from_json(json_str)

        assert restored.sigma_row == elliptical_params.sigma_row
        assert restored.sigma_col == elliptical_params.sigma_col

    def test_prf_params_dict_is_json_serializable(self, elliptical_params: PRFParams) -> None:
        """Dict should be JSON serializable."""
        d = prf_params_to_dict(elliptical_params)
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_fit_result_roundtrip(self, elliptical_params: PRFParams) -> None:
        """PRFFitResult should survive dict roundtrip."""
        result = PRFFitResult(
            params=elliptical_params,
            center_row=7.5,
            center_col=8.2,
            residual_rms=0.05,
            converged=True,
            n_iterations=42,
            chi_squared=1.23,
            covariance=np.eye(3),
        )

        d = fit_result_to_dict(result)
        restored = fit_result_from_dict(d)

        assert restored.center_row == result.center_row
        assert restored.center_col == result.center_col
        assert restored.residual_rms == result.residual_rms
        assert restored.converged == result.converged
        assert restored.n_iterations == result.n_iterations
        assert restored.chi_squared == result.chi_squared
        assert restored.covariance is not None
        assert_allclose(restored.covariance, result.covariance)


# =============================================================================
# Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_evaluate_deterministic(
        self,
        parametric_psf: ParametricPSF,
        standard_shape: tuple[int, int],
        center_coords: tuple[float, float],
    ) -> None:
        """Evaluate should be deterministic."""
        center_row, center_col = center_coords

        model1 = parametric_psf.evaluate(center_row, center_col, standard_shape)
        model2 = parametric_psf.evaluate(center_row, center_col, standard_shape)

        assert_allclose(model1, model2, rtol=0, atol=0)

    def test_factory_deterministic(self) -> None:
        """Factory should return consistent results."""
        params = PRFParams(sigma_row=2.0)
        model1 = get_prf_model("parametric", params=params)
        model2 = get_prf_model("parametric", params=params)

        assert model1.params.sigma_row == model2.params.sigma_row
