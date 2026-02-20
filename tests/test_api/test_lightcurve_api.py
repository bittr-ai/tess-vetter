from __future__ import annotations

import numpy as np
from pydantic import ValidationError

from tess_vetter.api.lightcurve import LightCurveRef, make_data_ref
from tess_vetter.domain.lightcurve import LightCurveData


def test_make_data_ref_matches_expected_format() -> None:
    assert make_data_ref(141914082, 1) == "lc:141914082:1:pdcsap"
    assert make_data_ref(141914082, 1, flux_type="sap") == "lc:141914082:1:sap"


def test_lightcurve_data_validates_dtypes_and_is_immutable() -> None:
    n = 10
    lc = LightCurveData(
        time=np.arange(n, dtype=np.float64),
        flux=np.ones(n, dtype=np.float64),
        flux_err=np.ones(n, dtype=np.float64),
        quality=np.zeros(n, dtype=np.int32),
        valid_mask=np.ones(n, dtype=np.bool_),
        tic_id=1,
        sector=2,
        cadence_seconds=120.0,
    )

    assert lc.n_points == n
    assert lc.n_valid == n
    assert lc.duration_days == float(lc.time[-1] - lc.time[0])
    assert lc.gap_fraction == 0.0
    assert lc.quality_flags_present == [0]

    with np.testing.assert_raises(ValueError):
        LightCurveData(
            time=np.arange(n, dtype=np.float32),
            flux=np.ones(n, dtype=np.float64),
            flux_err=np.ones(n, dtype=np.float64),
            quality=np.zeros(n, dtype=np.int32),
            valid_mask=np.ones(n, dtype=np.bool_),
            tic_id=1,
            sector=2,
            cadence_seconds=120.0,
        )

    with np.testing.assert_raises(ValueError):
        LightCurveData(
            time=np.arange(n, dtype=np.float64),
            flux=np.ones(n, dtype=np.float32),
            flux_err=np.ones(n, dtype=np.float64),
            quality=np.zeros(n, dtype=np.int32),
            valid_mask=np.ones(n, dtype=np.bool_),
            tic_id=1,
            sector=2,
            cadence_seconds=120.0,
        )

    with np.testing.assert_raises(ValueError):
        LightCurveData(
            time=np.arange(n, dtype=np.float64),
            flux=np.ones(n, dtype=np.float64),
            flux_err=np.ones(n, dtype=np.float32),
            quality=np.zeros(n, dtype=np.int32),
            valid_mask=np.ones(n, dtype=np.bool_),
            tic_id=1,
            sector=2,
            cadence_seconds=120.0,
        )

    with np.testing.assert_raises(ValueError):
        LightCurveData(
            time=np.arange(n, dtype=np.float64),
            flux=np.ones(n, dtype=np.float64),
            flux_err=np.ones(n, dtype=np.float64),
            quality=np.zeros(n, dtype=np.int64),
            valid_mask=np.ones(n, dtype=np.bool_),
            tic_id=1,
            sector=2,
            cadence_seconds=120.0,
        )

    with np.testing.assert_raises(ValueError):
        LightCurveData(
            time=np.arange(n, dtype=np.float64),
            flux=np.ones(n, dtype=np.float64),
            flux_err=np.ones(n, dtype=np.float64),
            quality=np.zeros(n, dtype=np.int32),
            valid_mask=np.ones(n, dtype=np.int8),
            tic_id=1,
            sector=2,
            cadence_seconds=120.0,
        )

    assert lc.time.flags.writeable is False
    assert lc.flux.flags.writeable is False
    assert lc.flux_err.flags.writeable is False
    assert lc.quality.flags.writeable is False
    assert lc.valid_mask.flags.writeable is False


def test_lightcurve_ref_is_frozen_and_forbids_extra() -> None:
    n = 10
    data = LightCurveData(
        time=np.arange(n, dtype=np.float64),
        flux=np.ones(n, dtype=np.float64),
        flux_err=np.ones(n, dtype=np.float64),
        quality=np.zeros(n, dtype=np.int32),
        valid_mask=np.ones(n, dtype=np.bool_),
        tic_id=1,
        sector=2,
        cadence_seconds=120.0,
    )

    ref = LightCurveRef.from_data(data)
    try:
        ref.tic_id = 999  # type: ignore[misc]
        raise AssertionError("Expected ValidationError for frozen model assignment")
    except ValidationError:
        pass

    try:
        LightCurveRef(  # type: ignore[call-arg]
            data_ref="lc:1:2:pdcsap",
            tic_id=1,
            sector=2,
            n_points=n,
            n_valid=n,
            duration_days=float(data.duration_days),
            cadence_seconds=120.0,
            median_flux=float(data.median_flux),
            flux_std=float(data.flux_std),
            gap_fraction=float(data.gap_fraction),
            quality_flags_present=[0],
            extra_field="nope",
        )
        raise AssertionError("Expected ValidationError for extra fields")
    except ValidationError:
        pass


def test_lightcurve_ref_from_data_populates_expected_fields() -> None:
    n = 10
    data = LightCurveData(
        time=np.arange(n, dtype=np.float64),
        flux=np.ones(n, dtype=np.float64),
        flux_err=np.ones(n, dtype=np.float64),
        quality=np.zeros(n, dtype=np.int32),
        valid_mask=np.ones(n, dtype=np.bool_),
        tic_id=141914082,
        sector=1,
        cadence_seconds=120.0,
    )

    ref = LightCurveRef.from_data(data, flux_type="sap")
    assert ref.data_ref == "lc:141914082:1:sap"
    assert ref.tic_id == 141914082
    assert ref.sector == 1
    assert ref.n_points == n
    assert ref.n_valid == n


def test_lightcurve_data_rejects_mismatched_lengths() -> None:
    with np.testing.assert_raises(ValueError):
        LightCurveData(
            time=np.arange(10, dtype=np.float64),
            flux=np.ones(5, dtype=np.float64),
            flux_err=np.ones(10, dtype=np.float64),
            quality=np.zeros(10, dtype=np.int32),
            valid_mask=np.ones(10, dtype=np.bool_),
            tic_id=1,
            sector=1,
            cadence_seconds=120.0,
        )

    with np.testing.assert_raises(ValueError):
        LightCurveData(
            time=np.arange(10, dtype=np.float64),
            flux=np.ones(10, dtype=np.float64),
            flux_err=np.ones(8, dtype=np.float64),
            quality=np.zeros(10, dtype=np.int32),
            valid_mask=np.ones(10, dtype=np.bool_),
            tic_id=1,
            sector=1,
            cadence_seconds=120.0,
        )

    with np.testing.assert_raises(ValueError):
        LightCurveData(
            time=np.arange(10, dtype=np.float64),
            flux=np.ones(10, dtype=np.float64),
            flux_err=np.ones(10, dtype=np.float64),
            quality=np.zeros(7, dtype=np.int32),
            valid_mask=np.ones(10, dtype=np.bool_),
            tic_id=1,
            sector=1,
            cadence_seconds=120.0,
        )

    with np.testing.assert_raises(ValueError):
        LightCurveData(
            time=np.arange(10, dtype=np.float64),
            flux=np.ones(10, dtype=np.float64),
            flux_err=np.ones(10, dtype=np.float64),
            quality=np.zeros(10, dtype=np.int32),
            valid_mask=np.ones(9, dtype=np.bool_),
            tic_id=1,
            sector=1,
            cadence_seconds=120.0,
        )


def test_lightcurve_data_empty_and_all_invalid_edge_cases() -> None:
    empty = LightCurveData(
        time=np.array([], dtype=np.float64),
        flux=np.array([], dtype=np.float64),
        flux_err=np.array([], dtype=np.float64),
        quality=np.array([], dtype=np.int32),
        valid_mask=np.array([], dtype=np.bool_),
        tic_id=1,
        sector=1,
        cadence_seconds=120.0,
    )
    assert empty.n_points == 0
    assert empty.n_valid == 0

    all_invalid = LightCurveData(
        time=np.linspace(1000.0, 1010.0, 100, dtype=np.float64),
        flux=np.ones(100, dtype=np.float64),
        flux_err=np.full(100, 0.001, dtype=np.float64),
        quality=np.zeros(100, dtype=np.int32),
        valid_mask=np.zeros(100, dtype=np.bool_),
        tic_id=1,
        sector=1,
        cadence_seconds=120.0,
    )
    assert all_invalid.n_points == 100
    assert all_invalid.n_valid == 0
