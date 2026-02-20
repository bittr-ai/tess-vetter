import numpy as np
import pytest

from tess_vetter.api import compute_sector_ephemeris_metrics
from tess_vetter.api.sector_consistency import SectorMeasurement
from tess_vetter.api.sector_metrics import (
    SECTOR_MEASUREMENTS_SCHEMA_VERSION,
    deserialize_v21_sector_measurements,
    serialize_v21_sector_measurements,
)


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
        assert m.n_transits > 0
        assert np.isfinite(m.depth_hat_ppm)
        assert np.isfinite(m.depth_sigma_ppm)
        assert m.depth_hat_ppm > 0
        assert 50.0 <= m.depth_hat_ppm <= 2000.0
        assert m.depth_sigma_ppm > 0


def test_v21_sector_measurements_serialize_deserialize_roundtrip() -> None:
    measurements = [
        SectorMeasurement(
            sector=14,
            depth_ppm=820.0,
            depth_err_ppm=65.0,
            duration_hours=2.4,
            duration_err_hours=0.3,
            n_transits=5,
            shape_metric=0.1,
            quality_weight=0.9,
        ),
        SectorMeasurement(
            sector=15,
            depth_ppm=790.0,
            depth_err_ppm=70.0,
            duration_hours=2.3,
            duration_err_hours=0.4,
            n_transits=4,
            shape_metric=0.2,
            quality_weight=0.8,
        ),
    ]

    payload = serialize_v21_sector_measurements(measurements)
    assert payload["schema_version"] == SECTOR_MEASUREMENTS_SCHEMA_VERSION
    assert len(payload["measurements"]) == 2

    restored = deserialize_v21_sector_measurements(payload)
    assert [m.sector for m in restored] == [14, 15]
    assert restored[0].depth_ppm == pytest.approx(820.0)
    assert restored[1].depth_err_ppm == pytest.approx(70.0)


def test_v21_sector_measurements_rejects_unknown_schema_version() -> None:
    bad_payload = {
        "schema_version": 999,
        "measurements": [{"sector": 1, "depth_ppm": 1000.0, "depth_err_ppm": 50.0}],
    }
    with pytest.raises(ValueError, match="unsupported V21 sector measurements schema_version"):
        _ = deserialize_v21_sector_measurements(bad_payload)


def test_v21_sector_measurements_from_ephemeris_metrics_includes_n_transits() -> None:
    rng = np.random.default_rng(7)
    t = np.linspace(0.0, 20.0, 3000, dtype=np.float64)
    sector = np.full(t.shape, 5, dtype=np.int32)
    period_days = 5.0
    t0_btjd = 1.0
    duration_hours = 3.0
    phase = ((t - t0_btjd) % period_days) / period_days
    half_dur_phase = (duration_hours / 24.0) / period_days / 2.0
    in_transit = (phase < half_dur_phase) | (phase > (1.0 - half_dur_phase))
    flux = np.ones_like(t)
    flux[in_transit] -= 400e-6
    flux += rng.normal(0.0, 120e-6, size=t.shape[0])
    flux_err = np.full_like(t, 120e-6)
    metrics = compute_sector_ephemeris_metrics(
        time=t,
        flux=flux,
        flux_err=flux_err,
        sector=sector,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
    )
    payload = serialize_v21_sector_measurements(metrics)
    assert payload["measurements"][0]["n_transits"] > 0
    assert payload["measurements"][0]["quality_weight"] == 1.0
