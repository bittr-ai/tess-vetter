from __future__ import annotations

import numpy as np

from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.validation.lc_checks import check_depth_stability
from tess_vetter.validation.lc_false_alarm_checks import (
    check_data_gaps,
    check_transit_asymmetry,
)


def _make_base_lc(
    *,
    period_days: float = 5.0,
    t0_btjd: float = 1.0,
    duration_hours: float = 3.0,
    baseline_days: float = 27.0,
    cadence_minutes: float = 10.0,
    noise_ppm: float = 50.0,
    seed: int = 0,
) -> LightCurveData:
    rng = np.random.default_rng(seed)
    dt_days = cadence_minutes / (24.0 * 60.0)
    time = np.arange(0.0, baseline_days, dt_days, dtype=np.float64)
    flux = np.ones_like(time)
    flux += rng.normal(0.0, noise_ppm * 1e-6, size=time.size)
    flux_err = np.full_like(time, noise_ppm * 1e-6)
    quality = np.zeros(time.size, dtype=np.int32)
    valid_mask = np.ones(time.size, dtype=bool)

    # Inject a simple box-like transit at phase 0.
    duration_days = duration_hours / 24.0
    half_phase = (duration_days / period_days) / 2.0
    phase = ((time - t0_btjd) / period_days) % 1.0
    d_to_primary = np.minimum(phase, 1.0 - phase)
    in_primary = d_to_primary < half_phase
    flux[in_primary] *= 1.0 - 0.001

    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=1,
        sector=1,
        cadence_seconds=int(cadence_minutes * 60),
    )


def _clone_lc(
    lc: LightCurveData,
    *,
    flux: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> LightCurveData:
    """LightCurveData arrays are immutable; clone with updated arrays for tests."""
    return LightCurveData(
        time=np.asarray(lc.time, dtype=np.float64).copy(),
        flux=(np.asarray(lc.flux, dtype=np.float64).copy() if flux is None else np.asarray(flux, dtype=np.float64)),
        flux_err=np.asarray(lc.flux_err, dtype=np.float64).copy(),
        quality=np.asarray(lc.quality, dtype=np.int32).copy(),
        valid_mask=(np.asarray(lc.valid_mask, dtype=bool).copy() if valid_mask is None else np.asarray(valid_mask, dtype=bool)),
        tic_id=lc.tic_id,
        sector=lc.sector,
        cadence_seconds=float(lc.cadence_seconds),
        provenance=lc.provenance,
    )


def test_v04_emits_domination_and_dmm_metrics() -> None:
    lc = _make_base_lc(seed=1, cadence_minutes=2.0, noise_ppm=30.0)

    # Make one epoch much deeper to trigger domination.
    period_days = 5.0
    t0_btjd = 1.0
    duration_hours = 3.0
    duration_days = duration_hours / 24.0
    phase = ((lc.time - t0_btjd) / period_days) % 1.0
    in_transit = np.minimum(phase, 1.0 - phase) < (duration_days / period_days) / 2.0

    # Pick a single epoch window and deepen it.
    epoch = np.floor((lc.time - t0_btjd + period_days / 2) / period_days).astype(int)
    target_epoch = int(np.median(epoch))
    flux = lc.flux.copy()
    flux[(epoch == target_epoch) & in_transit] *= 1.0 - 0.004
    lc = _clone_lc(lc, flux=flux)

    r = check_depth_stability(lc, period_days, t0_btjd, duration_hours)
    assert r.details.get("_metrics_only") is True
    for k in ("dom_ratio", "dom_frac", "s_max", "s_median", "dmm", "dmm_abs"):
        assert k in r.details
    assert float(r.details["dom_frac"]) >= 0.0


def test_v13_data_gaps_detects_missing_epoch_coverage() -> None:
    lc = _make_base_lc(seed=2, cadence_minutes=2.0, noise_ppm=20.0)

    # Remove data around one transit window to create a gap.
    period_days = 5.0
    t0_btjd = 1.0
    duration_hours = 3.0
    duration_days = duration_hours / 24.0
    gap_center = t0_btjd + 2 * period_days
    gap_half_window = 2.0 * duration_days  # matches default window_mult
    gap_mask = (lc.time >= gap_center - gap_half_window) & (lc.time <= gap_center + gap_half_window)
    valid = lc.valid_mask.copy()
    valid[gap_mask] = False
    lc = _clone_lc(lc, valid_mask=valid)

    r = check_data_gaps(lc, period_days, t0_btjd, duration_hours)
    assert r.details.get("_metrics_only") is True
    assert r.details["n_epochs_evaluated"] >= 1
    assert float(r.details["missing_frac_max"]) > 0.1


def test_v15_transit_asymmetry_ramp_is_large() -> None:
    lc = _make_base_lc(seed=3, cadence_minutes=2.0, noise_ppm=2.0)
    period_days = 5.0
    t0_btjd = 1.0
    duration_hours = 3.0

    # Add a deterministic ramp across the transit window to force asymmetry.
    duration_days = duration_hours / 24.0
    phase = ((lc.time - t0_btjd) / period_days) % 1.0
    phase = np.where(phase > 0.5, phase - 1.0, phase)  # [-0.5, 0.5]
    half_window_phase = 2.0 * duration_days / period_days
    window = np.abs(phase) <= half_window_phase
    right = window & (phase > 0)
    flux = lc.flux.copy()
    flux[right] += 2e-3  # step -> strong left/right mismatch
    lc = _clone_lc(lc, flux=flux)

    r = check_transit_asymmetry(lc, period_days, t0_btjd, duration_hours)
    assert r.details.get("_metrics_only") is True
    assert r.details.get("asymmetry_sigma") is not None
    assert float(r.details["asymmetry_sigma"]) > 1.0
