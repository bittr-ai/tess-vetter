from __future__ import annotations

from typing import Any

import numpy as np

from bittr_tess_vetter.api.stitch import stitch_lightcurve_data, stitch_lightcurves
from bittr_tess_vetter.domain.lightcurve import LightCurveData


def test_stitch_two_sectors_normalizes_each_sector() -> None:
    lc1: dict[str, Any] = {
        "time": np.array([1.0, 2.0, 3.0]),
        "flux": np.array([1000.0, 1001.0, 999.0]),
        "flux_err": np.array([10.0, 10.0, 10.0]),
        "sector": 1,
        "quality": np.array([0, 0, 0]),
    }
    lc2: dict[str, Any] = {
        "time": np.array([100.0, 101.0, 102.0]),
        "flux": np.array([2000.0, 2002.0, 1998.0]),
        "flux_err": np.array([20.0, 20.0, 20.0]),
        "sector": 2,
        "quality": np.array([0, 0, 0]),
    }

    stitched = stitch_lightcurves([lc1, lc2])

    s1 = stitched.sector == 1
    s2 = stitched.sector == 2
    assert np.isclose(np.median(stitched.flux[s1]), 1.0, rtol=0.01)
    assert np.isclose(np.median(stitched.flux[s2]), 1.0, rtol=0.01)


def test_stitch_all_nan_flux_falls_back_without_divide_by_nan() -> None:
    lc: dict[str, Any] = {
        "time": np.array([1.0, 2.0, 3.0]),
        "flux": np.array([np.nan, np.nan, np.nan]),
        "flux_err": np.array([1.0, 1.0, 1.0]),
        "sector": 1,
        "quality": np.array([0, 0, 0]),
    }

    stitched = stitch_lightcurves([lc])

    assert len(stitched.time) == 3
    assert stitched.per_sector_diagnostics[0].normalization_warning is not None
    np.testing.assert_array_equal(stitched.flux, lc["flux"])


def test_stitch_lightcurve_data_builds_valid_mask_and_cadence() -> None:
    lc1 = LightCurveData(
        time=np.array([1.0, 1.5, 2.0], dtype=np.float64),
        flux=np.array([1000.0, 1000.0, 1000.0], dtype=np.float64),
        flux_err=np.array([10.0, 10.0, 10.0], dtype=np.float64),
        quality=np.array([0, 0, 0], dtype=np.int32),
        valid_mask=np.array([True, True, True], dtype=np.bool_),
        tic_id=1,
        sector=1,
        cadence_seconds=120.0,
    )
    lc2 = LightCurveData(
        time=np.array([100.0, 100.5, 101.0], dtype=np.float64),
        flux=np.array([2000.0, np.nan, 2000.0], dtype=np.float64),
        flux_err=np.array([20.0, 20.0, 20.0], dtype=np.float64),
        quality=np.array([0, 0, 1], dtype=np.int32),
        valid_mask=np.array([True, True, True], dtype=np.bool_),
        tic_id=1,
        sector=2,
        cadence_seconds=120.0,
    )

    stitched_lc, stitched = stitch_lightcurve_data([lc1, lc2], tic_id=1)

    assert stitched_lc.tic_id == 1
    assert stitched_lc.sector == -1
    assert stitched_lc.n_points == 6
    # valid points: quality==0 AND finite flux
    assert stitched_lc.n_valid == 4
    assert stitched.normalization_policy_version == "v1"
