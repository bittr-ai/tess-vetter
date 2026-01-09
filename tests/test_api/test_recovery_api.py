import numpy as np

from bittr_tess_vetter.api.lightcurve import LightCurveData
from bittr_tess_vetter.api.recovery import prepare_recovery_inputs


def test_prepare_recovery_inputs_concatenates_and_counts_transits() -> None:
    # Build two small sectors with all-valid points
    # Use sufficiently dense sampling so count_transits(min_points=3) is satisfied.
    time1 = np.linspace(0.0, 5.0, 500, dtype=np.float64)
    time2 = np.linspace(5.0, 10.0, 600, dtype=np.float64)
    flux1 = np.ones_like(time1)
    flux2 = np.ones_like(time2)
    flux_err1 = np.full_like(time1, 1e-4)
    flux_err2 = np.full_like(time2, 1e-4)
    quality1 = np.zeros_like(time1, dtype=np.int32)
    quality2 = np.zeros_like(time2, dtype=np.int32)
    valid1 = np.ones_like(time1, dtype=np.bool_)
    valid2 = np.ones_like(time2, dtype=np.bool_)

    lc1 = LightCurveData(
        time=time1,
        flux=flux1,
        flux_err=flux_err1,
        quality=quality1,
        valid_mask=valid1,
        tic_id=1,
        sector=1,
        cadence_seconds=120.0,
    )
    lc2 = LightCurveData(
        time=time2,
        flux=flux2,
        flux_err=flux_err2,
        quality=quality2,
        valid_mask=valid2,
        tic_id=1,
        sector=4,
        cadence_seconds=120.0,
    )

    # period=2, t0=0.5 => transits at 0.5,2.5,4.5,6.5,8.5 within [0,10]
    prepared = prepare_recovery_inputs([lc2, lc1], period=2.0, t0=0.5, duration_hours=2.0)

    assert prepared.time.shape == (1100,)
    assert prepared.flux.shape == (1100,)
    assert prepared.flux_err.shape == (1100,)
    assert prepared.sectors_used == [1, 4]
    assert prepared.n_transits == 5
