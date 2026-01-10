import importlib.util

import numpy as np
import pytest

from bittr_tess_vetter.api.lc_only import vet_lc_only
from bittr_tess_vetter.api.periodogram import run_periodogram
from bittr_tess_vetter.api.stitch import stitch_lightcurve_data
from bittr_tess_vetter.api.types import Ephemeris, LightCurve
from bittr_tess_vetter.compute.transit import detect_transit, get_transit_mask
from bittr_tess_vetter.domain.lightcurve import LightCurveData

TLS_AVAILABLE = importlib.util.find_spec("transitleastsquares") is not None


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

    if len(flux) != n or len(flux_err) != n:
        raise ValueError("flux/flux_err must match generated time length")

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
    phase_dist = np.minimum(phase, 1.0 - phase)
    in_transit = phase_dist < 0.5 * (duration_days / period_days)

    odd = (epoch % 2) == 1
    out[in_transit & odd] *= 1.0 - depth_odd
    out[in_transit & ~odd] *= 1.0 - depth_even
    return out


def _inject_secondary_eclipse_window(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    depth: float,
    center: float = 0.5,
    half_width: float = 0.15,
) -> np.ndarray:
    out = flux.copy()
    phase = ((time - t0_btjd) / period_days) % 1.0
    sec = (phase > center - half_width) & (phase < center + half_width)
    out[sec] *= 1.0 - depth
    return out


@pytest.mark.skipif(not TLS_AVAILABLE, reason="transitleastsquares not available")
def test_stitch_to_period_search_to_fold_to_lc_only_checks_two_sectors_with_gap() -> None:
    rng = np.random.default_rng(123)

    tic_id = 123456789
    cadence_seconds = 120.0

    # Two sectors separated by a large gap (cross-gap cadence should not dominate).
    period_days = 3.5
    t0_btjd = 0.5
    duration_hours = 2.4
    depth = 0.01

    # Sector 1: [0, 27], sector 2: [55, 82]
    for_sector = []
    for sector, (t_start, t_end) in enumerate([(0.0, 27.0), (55.0, 82.0)], start=1):
        cadence_days = cadence_seconds / 86400.0
        n = int(np.floor((t_end - t_start) / cadence_days)) + 1
        base = np.ones(n, dtype=np.float64) + rng.normal(0, 2e-4, n)
        err = np.full(n, 2e-4, dtype=np.float64)

        time_tmp = (t_start + np.arange(n) * cadence_days).astype(np.float64)
        flux = _inject_box_transits(
            time_tmp,
            base,
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
            depth=depth,
        )
        for_sector.append(
            _make_sector(
                t_start=t_start,
                t_end=t_end,
                cadence_seconds=cadence_seconds,
                sector=sector,
                tic_id=tic_id,
                flux=flux,
                flux_err=err,
            )
        )

    stitched_lc_data, _ = stitch_lightcurve_data(for_sector, tic_id=tic_id)
    assert abs(stitched_lc_data.cadence_seconds - cadence_seconds) < 1.0

    time = stitched_lc_data.time[stitched_lc_data.valid_mask]
    flux = stitched_lc_data.flux[stitched_lc_data.valid_mask]
    flux_err = stitched_lc_data.flux_err[stitched_lc_data.valid_mask]

    result = run_periodogram(
        time=time,
        flux=flux,
        flux_err=flux_err,
        method="tls",
        min_period=2.0,
        max_period=5.0,
        max_planets=1,
        per_sector=True,
        downsample_factor=2,
        stellar_radius_rsun=1.0,
        stellar_mass_msun=1.0,
        data_ref="lc:test:stitched:pdcsap",
    )

    assert result.method == "tls"
    assert abs(result.best_period - period_days) / period_days < 0.05

    eph = Ephemeris(
        period_days=float(result.best_period),
        t0_btjd=float(result.best_t0),
        duration_hours=float(result.best_duration_hours or duration_hours),
    )
    lc = LightCurve.from_internal(stitched_lc_data)

    checks = vet_lc_only(lc, eph)
    by_id = {c.id: c for c in checks}
    assert by_id["V01"].details.get("_metrics_only") is True
    assert by_id["V02"].details.get("_metrics_only") is True

    # Should not look like an obvious EB on this clean injection.
    if "insufficient_data_for_odd_even_check" not in by_id["V01"].details.get("warnings", []):
        assert float(by_id["V01"].details.get("rel_diff", 0.0)) < 0.1
    if by_id["V02"].details.get("note") != "Insufficient data for secondary eclipse search":
        assert float(by_id["V02"].details.get("secondary_depth_sigma", 0.0)) < 3.0


def test_odd_even_and_secondary_combined_integration() -> None:
    rng = np.random.default_rng(7)
    cadence_seconds = 120.0
    cadence_days = cadence_seconds / 86400.0
    time = (0.0 + np.arange(40000) * cadence_days).astype(np.float64)

    period_days = 3.5
    t0_btjd = 0.5
    duration_hours = 2.4

    flux = np.ones_like(time) + rng.normal(0, 2e-4, len(time))
    flux_err = np.full_like(time, 2e-4)

    flux = _inject_transits_by_epoch_parity(
        time,
        flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_odd=0.02,
        depth_even=0.01,
    )
    flux = _inject_secondary_eclipse_window(
        time,
        flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        depth=0.01,
        center=0.5,
        half_width=0.15,
    )

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours)

    checks = vet_lc_only(lc, eph, enabled={"V01", "V02"})
    by_id = {c.id: c for c in checks}
    assert by_id["V01"].details.get("_metrics_only") is True
    assert by_id["V02"].details.get("_metrics_only") is True
    assert "insufficient_data_for_odd_even_check" not in by_id["V01"].details.get("warnings", [])
    assert float(by_id["V01"].details.get("rel_diff", 0.0)) > 0.1
    assert float(by_id["V01"].details.get("delta_sigma", 0.0)) > 2.0
    assert float(by_id["V02"].details.get("secondary_depth_sigma", 0.0)) > 2.0


def test_duration_scaling_changes_in_transit_points_and_snr() -> None:
    rng = np.random.default_rng(99)
    cadence_seconds = 120.0
    cadence_days = cadence_seconds / 86400.0
    time = (1500.0 + np.arange(25000) * cadence_days).astype(np.float64)

    period_days = 3.5
    t0_btjd = 1500.5
    depth = 0.01
    sigma = 2e-4

    flux0 = np.ones_like(time) + rng.normal(0, sigma, len(time))
    flux_err = np.full_like(time, sigma)

    duration_short = 1.2
    duration_long = 2.4

    flux_short = _inject_box_transits(
        time,
        flux0,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_short,
        depth=depth,
    )
    flux_long = _inject_box_transits(
        time,
        flux0,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_long,
        depth=depth,
    )

    mask_short = get_transit_mask(time, period_days, t0_btjd, duration_short)
    mask_long = get_transit_mask(time, period_days, t0_btjd, duration_long)
    n_in_short = int(np.sum(mask_short))
    n_in_long = int(np.sum(mask_long))
    assert n_in_long > n_in_short
    assert n_in_long >= int(1.8 * n_in_short)

    cand_short = detect_transit(time, flux_short, flux_err, period_days, t0_btjd, duration_short)
    cand_long = detect_transit(time, flux_long, flux_err, period_days, t0_btjd, duration_long)
    assert cand_long.snr > cand_short.snr

    # Expect roughly sqrt(N_in) scaling for fixed depth/noise.
    expected_ratio = float(np.sqrt(n_in_long / n_in_short))
    measured_ratio = float(cand_long.snr / cand_short.snr) if cand_short.snr > 0 else 0.0
    assert measured_ratio == pytest.approx(expected_ratio, rel=0.35)
