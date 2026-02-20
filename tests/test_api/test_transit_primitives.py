import numpy as np

from tess_vetter.api.transit_primitives import odd_even_result
from tess_vetter.api.types import Ephemeris, LightCurve


def _inject_transits_by_epoch_parity(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_odd: float,
    depth_even: float,
) -> np.ndarray:
    out = flux.copy()
    duration_days = duration_hours / 24.0

    epoch = np.floor((time - t0_btjd + period_days / 2.0) / period_days).astype(int)
    phase = ((time - t0_btjd) / period_days) % 1.0
    in_transit = (phase < 0.75 * duration_days / period_days) | (
        phase > 1.0 - 0.75 * duration_days / period_days
    )

    odd = (epoch % 2) == 1
    out[in_transit & odd] *= 1.0 - depth_odd
    out[in_transit & ~odd] *= 1.0 - depth_even
    return out


def test_odd_even_result_clean_injection_not_suspicious() -> None:
    time = np.linspace(0.0, 40.0, 12000, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 2e-4)

    period_days = 3.5
    t0_btjd = 0.5
    duration_hours = 2.5

    flux = _inject_transits_by_epoch_parity(
        time,
        flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_odd=0.01,
        depth_even=0.01,
    )

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours)

    result = odd_even_result(lc, eph)
    assert result.n_odd > 0
    assert result.n_even > 0
    assert result.relative_depth_diff_percent < 10.0


def test_odd_even_result_alternating_depths_is_suspicious() -> None:
    time = np.linspace(0.0, 40.0, 12000, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 2e-4)

    period_days = 3.5
    t0_btjd = 0.5
    duration_hours = 2.5

    flux = _inject_transits_by_epoch_parity(
        time,
        flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_odd=0.02,
        depth_even=0.01,
    )

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours)

    result = odd_even_result(lc, eph)
    assert result.depth_odd_ppm > result.depth_even_ppm
    assert result.relative_depth_diff_percent > 10.0


def test_odd_even_result_ignores_nans_via_valid_mask() -> None:
    time = np.linspace(0.0, 40.0, 12000, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 2e-4)

    time[100] = np.nan
    flux[200] = np.nan
    flux_err[300] = np.nan

    period_days = 3.5
    t0_btjd = 0.5
    duration_hours = 2.5

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours)

    result = odd_even_result(lc, eph)
    assert result.relative_depth_diff_percent == float(result.relative_depth_diff_percent)
