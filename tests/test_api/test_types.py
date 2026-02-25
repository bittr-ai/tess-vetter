"""Tests for api/types.py."""

import numpy as np
import pytest

from tess_vetter.api.types import (
    CheckResult,
    Ephemeris,
    LightCurve,
    StellarParams,
    ok_result,
    skipped_result,
)
from tess_vetter.domain.lightcurve import LightCurveData


class TestEphemeris:
    """Tests for Ephemeris dataclass."""

    def test_create_valid_ephemeris(self) -> None:
        """Test creating a valid Ephemeris."""
        eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
        assert eph.period_days == 3.5
        assert eph.t0_btjd == 1850.0
        assert eph.duration_hours == 2.5

    def test_ephemeris_is_frozen(self) -> None:
        """Test that Ephemeris is immutable."""
        eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
        with pytest.raises(AttributeError):
            eph.period_days = 4.0  # type: ignore[misc]

    def test_ephemeris_rejects_negative_period(self) -> None:
        """Test that Ephemeris rejects negative period."""
        with pytest.raises(ValueError, match="period_days must be positive"):
            Ephemeris(period_days=-1.0, t0_btjd=1850.0, duration_hours=2.5)

    def test_ephemeris_rejects_zero_period(self) -> None:
        """Test that Ephemeris rejects zero period."""
        with pytest.raises(ValueError, match="period_days must be positive"):
            Ephemeris(period_days=0.0, t0_btjd=1850.0, duration_hours=2.5)

    def test_ephemeris_rejects_negative_duration(self) -> None:
        """Test that Ephemeris rejects negative duration."""
        with pytest.raises(ValueError, match="duration_hours must be positive"):
            Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=-1.0)


class TestLightCurve:
    """Tests for LightCurve dataclass."""

    def test_create_minimal_lightcurve(self) -> None:
        """Test creating a LightCurve with only required fields."""
        time = np.linspace(0, 10, 100)
        flux = np.ones(100)
        lc = LightCurve(time=time, flux=flux)
        assert len(lc.time) == 100
        assert len(lc.flux) == 100
        assert lc.flux_err is None
        assert lc.quality is None
        assert lc.valid_mask is None

    def test_create_full_lightcurve(self) -> None:
        """Test creating a LightCurve with all fields."""
        n = 100
        time = np.linspace(0, 10, n)
        flux = np.ones(n)
        flux_err = np.ones(n) * 0.001
        quality = np.zeros(n, dtype=int)
        valid_mask = np.ones(n, dtype=bool)

        lc = LightCurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
        )
        assert len(lc.time) == n
        assert len(lc.flux) == n
        assert lc.flux_err is not None
        assert len(lc.flux_err) == n

    def test_to_internal_rejects_mismatched_lengths(self) -> None:
        time = np.linspace(0, 10, 10)
        flux = np.ones(9)
        lc = LightCurve(time=time, flux=flux)
        with pytest.raises(ValueError, match="time and flux must have the same length"):
            lc.to_internal()

    def test_to_internal_requires_matching_optional_lengths(self) -> None:
        time = np.linspace(0, 10, 10)
        flux = np.ones(10)
        flux_err = np.ones(9) * 0.001
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        with pytest.raises(ValueError, match="flux_err must have the same length"):
            lc.to_internal()

    def test_to_internal_combines_valid_mask_with_finite_mask(self) -> None:
        time = np.linspace(0, 10, 10)
        flux = np.ones(10)
        flux_err = np.ones(10) * 0.001
        valid_mask = np.ones(10, dtype=bool)
        time[3] = np.nan
        flux[4] = np.nan
        flux_err[5] = np.nan

        lc = LightCurve(time=time, flux=flux, flux_err=flux_err, valid_mask=valid_mask)
        internal = lc.to_internal()

        assert not bool(internal.valid_mask[3])
        assert not bool(internal.valid_mask[4])
        assert not bool(internal.valid_mask[5])

    def test_to_internal_dtype_normalization(self) -> None:
        """Test that to_internal() normalizes dtypes correctly."""
        n = 50
        # Create arrays with non-standard dtypes
        time = np.linspace(0, 10, n).astype(np.float32)  # float32, not float64
        flux = np.ones(n, dtype=np.float32)
        flux_err = np.ones(n, dtype=np.float32) * 0.001
        quality = np.zeros(n, dtype=np.int64)  # int64, not int32
        valid_mask = np.ones(n, dtype=np.uint8)  # uint8, not bool

        lc = LightCurve(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
        )

        internal = lc.to_internal()

        # Check dtype normalization
        assert internal.time.dtype == np.float64
        assert internal.flux.dtype == np.float64
        assert internal.flux_err.dtype == np.float64
        assert internal.quality.dtype == np.int32
        assert internal.valid_mask.dtype == np.bool_

    def test_to_internal_defaults_optional_fields(self) -> None:
        """Test that to_internal() provides defaults for optional fields."""
        n = 50
        time = np.linspace(0, 10, n)
        flux = np.ones(n)

        lc = LightCurve(time=time, flux=flux)
        internal = lc.to_internal()

        # Check defaults
        assert len(internal.flux_err) == n
        assert np.all(internal.flux_err == 0.0)
        assert len(internal.quality) == n
        assert np.all(internal.quality == 0)
        assert len(internal.valid_mask) == n
        assert np.all(internal.valid_mask)

    def test_to_internal_custom_metadata(self) -> None:
        """Test that to_internal() accepts custom metadata."""
        time = np.linspace(0, 10, 50)
        flux = np.ones(50)

        lc = LightCurve(time=time, flux=flux)
        internal = lc.to_internal(tic_id=12345, sector=5, cadence_seconds=20.0)

        assert internal.tic_id == 12345
        assert internal.sector == 5
        assert internal.cadence_seconds == 20.0

    def test_to_internal_normalizes_absolute_bjd_to_btjd(self) -> None:
        time_bjd = np.array([2459001.0, 2459002.0, 2459003.0], dtype=np.float64)
        flux = np.ones_like(time_bjd)
        lc = LightCurve(time=time_bjd, flux=flux)
        internal = lc.to_internal()
        np.testing.assert_allclose(internal.time, np.array([2001.0, 2002.0, 2003.0], dtype=np.float64))

    def test_from_internal(self) -> None:
        """Test creating LightCurve from internal LightCurveData."""
        n = 50
        internal = LightCurveData(
            time=np.linspace(0, 10, n, dtype=np.float64),
            flux=np.ones(n, dtype=np.float64),
            flux_err=np.ones(n, dtype=np.float64) * 0.001,
            quality=np.zeros(n, dtype=np.int32),
            valid_mask=np.ones(n, dtype=np.bool_),
            tic_id=12345,
            sector=5,
            cadence_seconds=120.0,
        )

        lc = LightCurve.from_internal(internal)

        assert len(lc.time) == n
        assert len(lc.flux) == n
        assert lc.flux_err is not None
        assert len(lc.flux_err) == n

    def test_roundtrip_conversion(self) -> None:
        """Test that conversion to internal and back preserves data."""
        n = 50
        time = np.linspace(0, 10, n)
        flux = np.random.normal(1.0, 0.001, n)
        flux_err = np.ones(n) * 0.001

        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        internal = lc.to_internal()
        lc2 = LightCurve.from_internal(internal)

        np.testing.assert_array_almost_equal(lc.time, lc2.time)
        np.testing.assert_array_almost_equal(lc.flux, lc2.flux)
        np.testing.assert_array_almost_equal(lc.flux_err, lc2.flux_err)


class TestCheckResult:
    """Tests for CheckResult (Pydantic model from validation.result_schema)."""

    def test_create_valid_check_result(self) -> None:
        """Test creating a valid CheckResult using helper function."""
        result = ok_result(
            id="V01",
            name="odd_even_depth",
            confidence=0.95,
            metrics={"odd_depth": 0.001, "even_depth": 0.001},
        )
        assert result.id == "V01"
        assert result.name == "odd_even_depth"
        assert result.status == "ok"
        # passed is a backward-compat property: status="ok" -> passed=True
        assert result.passed is True
        assert result.confidence == 0.95
        assert result.metrics["odd_depth"] == 0.001
        # details is a backward-compat property that combines metrics/flags/etc
        assert result.details["odd_depth"] == 0.001

    def test_check_result_extra_fields_forbidden(self) -> None:
        """Test that CheckResult forbids extra fields during construction."""
        # Pydantic models with extra="forbid" reject unknown fields
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CheckResult(
                id="V01",
                name="odd_even_depth",
                status="ok",
                unknown_field="value",  # type: ignore[call-arg]
            )

    def test_check_result_skipped_status(self) -> None:
        """Test that skipped status maps to passed=None."""
        result = skipped_result(
            id="V01",
            name="odd_even_depth",
            reason_flag="TEST_SKIP",
        )
        assert result.status == "skipped"
        assert result.passed is None
        assert "SKIPPED:TEST_SKIP" in result.flags


class TestStellarParams:
    """Tests for StellarParams (alias for StellarParameters)."""

    def test_stellar_params_is_stellar_parameters(self) -> None:
        """Test that StellarParams is an alias for StellarParameters."""
        from tess_vetter.domain.target import StellarParameters

        assert StellarParams is StellarParameters

    def test_create_stellar_params(self) -> None:
        """Test creating StellarParams."""
        params = StellarParams(
            teff=5780.0,
            logg=4.44,
            radius=1.0,
            mass=1.0,
        )
        assert params.teff == 5780.0
        assert params.logg == 4.44
        assert params.radius == 1.0
        assert params.mass == 1.0
