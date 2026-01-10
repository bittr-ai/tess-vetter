import numpy as np

from bittr_tess_vetter.api.timing import analyze_ttvs, measure_transit_times
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve


def _inject_box_transits(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth: float,
) -> np.ndarray:
    out = flux.copy()
    duration_days = duration_hours / 24.0
    phase = ((time - t0_btjd) / period_days + 0.5) % 1.0 - 0.5
    in_transit = np.abs(phase) < (duration_days / (2.0 * period_days))
    out[in_transit] *= 1.0 - depth
    return out


def test_measure_transit_times_recovers_epochs_on_clean_injection() -> None:
    cadence_seconds = 120.0
    cadence_days = cadence_seconds / 86400.0
    time = (1500.0 + np.arange(25000) * cadence_days).astype(np.float64)
    period_days = 3.5
    t0_btjd = 1500.5
    duration_hours = 2.4
    depth = 0.01

    rng = np.random.default_rng(42)
    flux = np.ones_like(time) + rng.normal(0, 2e-4, len(time))
    flux_err = np.full_like(time, 2e-4)
    flux = _inject_box_transits(
        time, flux, period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours, depth=depth
    )

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    cand = Candidate(ephemeris=Ephemeris(period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours))

    times = measure_transit_times(lc, cand, min_snr=2.0)
    assert len(times) >= 5

    # Measured tc should be close to expected epoch centers (within ~10 minutes).
    tol_days = 10.0 / (60.0 * 24.0)
    for tt in times[:8]:
        expected = t0_btjd + tt.epoch * period_days
        assert abs(tt.tc - expected) < tol_days


def test_analyze_ttvs_on_linear_ephemeris_has_small_rms() -> None:
    # Build synthetic TransitTime list via the measurement path (ensures units match).
    cadence_seconds = 120.0
    cadence_days = cadence_seconds / 86400.0
    time = (1500.0 + np.arange(22000) * cadence_days).astype(np.float64)
    period_days = 3.5
    t0_btjd = 1500.5
    duration_hours = 2.4
    depth = 0.01

    rng = np.random.default_rng(0)
    flux = np.ones_like(time) + rng.normal(0, 2e-4, len(time))
    flux_err = np.full_like(time, 2e-4)
    flux = _inject_box_transits(time, flux, period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours, depth=depth)

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    cand = Candidate(ephemeris=Ephemeris(period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours))
    transit_times = measure_transit_times(lc, cand, min_snr=2.0)

    ttv = analyze_ttvs(transit_times, period_days=period_days, t0_btjd=t0_btjd)
    assert ttv.n_transits == len(transit_times)
    assert ttv.rms_seconds < 15 * 60  # <15 min RMS on clean injection
    assert ttv.periodicity_sigma >= 0.0


def test_measure_transit_times_ignores_nans_via_valid_mask() -> None:
    cadence_seconds = 120.0
    cadence_days = cadence_seconds / 86400.0
    time = (1500.0 + np.arange(12000) * cadence_days).astype(np.float64)
    period_days = 3.5
    t0_btjd = 1500.5
    duration_hours = 2.4
    depth = 0.01

    rng = np.random.default_rng(1)
    flux = np.ones_like(time) + rng.normal(0, 2e-4, len(time))
    flux_err = np.full_like(time, 2e-4)
    flux = _inject_box_transits(time, flux, period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours, depth=depth)

    # Inject NaNs; api/types LightCurve.to_internal should mask them out.
    time[100] = np.nan
    flux[200] = np.nan
    flux_err[300] = np.nan

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    cand = Candidate(ephemeris=Ephemeris(period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours))
    times = measure_transit_times(lc, cand, min_snr=2.0)

    assert all(np.isfinite(t.tc) for t in times)

