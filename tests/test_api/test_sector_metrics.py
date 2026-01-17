import numpy as np

from bittr_tess_vetter.api import compute_sector_ephemeris_metrics


def test_compute_sector_ephemeris_metrics_two_sectors() -> None:
    rng = np.random.default_rng(123)

    period_days = 2.0
    t0_btjd = 0.5
    duration_hours = 2.0
    depth_ppm = 500.0
    depth = depth_ppm * 1e-6

    n1 = 2000
    n2 = 2000
    t1 = np.linspace(0.0, 10.0, n1, dtype=np.float64)
    t2 = np.linspace(20.0, 30.0, n2, dtype=np.float64)

    time = np.concatenate([t1, t2])
    sector = np.concatenate([np.full(n1, 55, dtype=np.int32), np.full(n2, 83, dtype=np.int32)])

    phase = ((time - t0_btjd) % period_days) / period_days
    half_dur_phase = (duration_hours / 24.0) / period_days / 2.0
    in_transit = (phase < half_dur_phase) | (phase > (1.0 - half_dur_phase))

    flux = np.ones_like(time)
    flux[in_transit] -= depth
    noise_sigma = 150e-6
    flux += rng.normal(0.0, noise_sigma, size=len(time))
    flux_err = np.full_like(time, noise_sigma)

    out = compute_sector_ephemeris_metrics(
        time=time,
        flux=flux,
        flux_err=flux_err,
        sector=sector,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
    )

    assert [m.sector for m in out] == [55, 83]
    for m in out:
        assert m.n_valid > 100
        assert np.isfinite(m.depth_hat_ppm)
        assert np.isfinite(m.depth_sigma_ppm)
        assert m.depth_hat_ppm > 0
        assert 50.0 <= m.depth_hat_ppm <= 2000.0
        assert m.depth_sigma_ppm > 0

