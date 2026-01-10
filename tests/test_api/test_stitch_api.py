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


def test_stitch_lightcurve_data_cadence_ignores_cross_sector_gaps() -> None:
    lc1 = LightCurveData(
        time=np.array([1.0, 1.0 + 120 / 86400, 1.0 + 240 / 86400], dtype=np.float64),
        flux=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        flux_err=np.array([0.001, 0.001, 0.001], dtype=np.float64),
        quality=np.array([0, 0, 0], dtype=np.int32),
        valid_mask=np.array([True, True, True], dtype=np.bool_),
        tic_id=1,
        sector=1,
        cadence_seconds=120.0,
    )
    # Large gap between sectors, but within-sector cadence is still 120s.
    lc2 = LightCurveData(
        time=np.array([100.0, 100.0 + 120 / 86400, 100.0 + 240 / 86400], dtype=np.float64),
        flux=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        flux_err=np.array([0.001, 0.001, 0.001], dtype=np.float64),
        quality=np.array([0, 0, 0], dtype=np.int32),
        valid_mask=np.array([True, True, True], dtype=np.bool_),
        tic_id=1,
        sector=2,
        cadence_seconds=120.0,
    )

    stitched_lc, _ = stitch_lightcurve_data([lc1, lc2], tic_id=1)
    assert np.isclose(stitched_lc.cadence_seconds, 120.0, atol=1e-3)


def test_stitch_all_quality_flagged_falls_back_to_finite_median() -> None:
    lc: dict[str, Any] = {
        "time": np.array([1.0, 2.0, 3.0]),
        "flux": np.array([1000.0, 1002.0, 998.0]),
        "flux_err": np.array([10.0, 10.0, 10.0]),
        "sector": 1,
        "quality": np.array([1, 1, 1]),
    }
    stitched = stitch_lightcurves([lc])
    assert stitched.per_sector_diagnostics[0].normalization_warning is None
    assert np.isclose(np.median(stitched.flux), 1.0, rtol=0.01)


def test_stitch_cadence_prefers_dominant_within_sector_delta() -> None:
    # Mix a 20s and 120s sector; median of within-sector deltas should prefer 120s
    # when it dominates the combined within-sector deltas.
    lc20 = LightCurveData(
        time=np.array([1.0, 1.0 + 20 / 86400, 1.0 + 40 / 86400], dtype=np.float64),
        flux=np.array([1.0, 1.0, 1.0], dtype=np.float64),
        flux_err=np.array([0.001, 0.001, 0.001], dtype=np.float64),
        quality=np.array([0, 0, 0], dtype=np.int32),
        valid_mask=np.array([True, True, True], dtype=np.bool_),
        tic_id=1,
        sector=1,
        cadence_seconds=20.0,
    )
    # 120s sector contributes more within-sector deltas
    t0 = 100.0
    lc120 = LightCurveData(
        time=np.array(
            [t0 + i * (120 / 86400) for i in range(10)],
            dtype=np.float64,
        ),
        flux=np.ones(10, dtype=np.float64),
        flux_err=np.full(10, 0.001, dtype=np.float64),
        quality=np.zeros(10, dtype=np.int32),
        valid_mask=np.ones(10, dtype=np.bool_),
        tic_id=1,
        sector=2,
        cadence_seconds=120.0,
    )
    stitched_lc, _ = stitch_lightcurve_data([lc20, lc120], tic_id=1)
    assert np.isclose(stitched_lc.cadence_seconds, 120.0, atol=1e-3)
