import numpy as np
import pytest

from tess_vetter.api.recovery import prepare_recovery_inputs, recover_transit
from tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from tess_vetter.domain.lightcurve import LightCurveData


def _make_sector(
    *,
    t_start: float,
    t_end: float,
    cadence_seconds: float,
    sector: int,
    tic_id: int,
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> LightCurveData:
    cadence_days = cadence_seconds / 86400.0
    n = int(np.floor((t_end - t_start) / cadence_days)) + 1
    time = (t_start + np.arange(n) * cadence_days).astype(np.float64)

    assert len(flux) == n
    assert len(flux_err) == n

    quality = np.zeros(n, dtype=np.int32)
    valid_mask = np.ones(n, dtype=np.bool_)

    return LightCurveData(
        time=time,
        flux=np.asarray(flux, dtype=np.float64),
        flux_err=np.asarray(flux_err, dtype=np.float64),
        quality=quality,
        valid_mask=valid_mask,
        tic_id=int(tic_id),
        sector=int(sector),
        cadence_seconds=float(cadence_seconds),
    )


def _inject_transit(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period: float,
    t0: float,
    duration_hours: float,
    depth: float,
) -> np.ndarray:
    out = flux.copy()
    duration_days = duration_hours / 24.0
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    in_tr = np.abs(phase) < (duration_days / (2.0 * period))
    out[in_tr] *= 1.0 - depth
    return out


def test_prepare_recovery_inputs_sorts_concat_time() -> None:
    tic_id = 999
    cadence_seconds = 120.0
    period = 3.5
    t0 = 0.5
    duration_hours = 2.4

    rng = np.random.default_rng(1)

    sectors = []
    for sector, (t_start, t_end) in enumerate([(55.0, 82.0), (0.0, 27.0)], start=1):
        cad_days = cadence_seconds / 86400.0
        n = int(np.floor((t_end - t_start) / cad_days)) + 1
        time = (t_start + np.arange(n) * cad_days).astype(np.float64)
        flux = np.ones(n, dtype=np.float64) + rng.normal(0, 2e-4, n)
        flux_err = np.full(n, 2e-4, dtype=np.float64)
        flux = _inject_transit(
            time, flux, period=period, t0=t0, duration_hours=duration_hours, depth=0.01
        )
        sectors.append(
            _make_sector(
                t_start=t_start,
                t_end=t_end,
                cadence_seconds=cadence_seconds,
                sector=sector,
                tic_id=tic_id,
                flux=flux,
                flux_err=flux_err,
            )
        )

    prepared = prepare_recovery_inputs(sectors, period=period, t0=t0, duration_hours=duration_hours)
    assert np.all(np.diff(prepared.time) >= 0)
    assert prepared.n_transits >= 1


def test_recover_transit_two_sectors_with_gap_recovers_depth_order() -> None:
    cadence_seconds = 120.0
    period = 3.5
    t0 = 0.5
    duration_hours = 2.4
    depth = 0.005

    rng = np.random.default_rng(2)

    # Build stitched-like array with a big gap.
    cad_days = cadence_seconds / 86400.0
    time1 = (0.0 + np.arange(int(27.0 / cad_days) + 1) * cad_days).astype(np.float64)
    time2 = (55.0 + np.arange(int(27.0 / cad_days) + 1) * cad_days).astype(np.float64)
    time = np.concatenate([time1, time2])
    flux = np.ones_like(time) + 3e-4 * np.sin(2.0 * np.pi * time / 5.0)  # variability
    flux += rng.normal(0, 2e-4, len(time))
    flux_err = np.full_like(time, 2e-4)

    flux = _inject_transit(
        time, flux, period=period, t0=t0, duration_hours=duration_hours, depth=depth
    )

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    cand = Candidate(
        ephemeris=Ephemeris(period_days=period, t0_btjd=t0, duration_hours=duration_hours)
    )

    result = recover_transit(
        lc,
        cand,
        detrend_method="harmonic",
        rotation_period=5.0,
        n_harmonics=3,
        phase_bins=120,
    )

    assert result.converged is True
    assert result.n_transits_stacked >= 5
    # Order-of-magnitude sanity: recovered depth should be within a factor ~2 of injected.
    assert result.fitted_depth_ppm == pytest.approx(depth * 1e6, rel=0.6)
    assert result.detection_snr > 3.0


def test_recover_transit_requires_rotation_period_for_harmonic() -> None:
    time = np.linspace(0.0, 27.0, 2000, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-4)

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    cand = Candidate(ephemeris=Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.4))

    with pytest.raises(ValueError, match="rotation_period is required"):
        recover_transit(lc, cand, detrend_method="harmonic", rotation_period=None)
