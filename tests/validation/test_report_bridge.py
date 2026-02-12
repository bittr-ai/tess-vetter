from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.api.alias_diagnostics import harmonic_power_summary
from bittr_tess_vetter.api.lc_only import vet_lc_only
from bittr_tess_vetter.api.timing import timing_series
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.validation.alias_diagnostics import (
    classify_alias,
    compute_secondary_significance,
    detect_phase_shift_events,
)
from bittr_tess_vetter.validation.report_bridge import (
    compute_alias_scalar_signals,
    compute_alias_summary,
    compute_timing_series,
    run_lc_checks,
)


def _make_box_transit_lc(
    *,
    period_days: float = 3.5,
    t0_btjd: float = 0.5,
    duration_hours: float = 2.5,
    baseline_days: float = 27.0,
    cadence_minutes: float = 10.0,
    depth_frac: float = 0.01,
    noise_ppm: float = 50.0,
    seed: int = 7,
) -> LightCurve:
    rng = np.random.default_rng(seed)
    dt_days = cadence_minutes / (24.0 * 60.0)
    time = np.arange(0.0, baseline_days, dt_days, dtype=np.float64)
    flux = np.ones_like(time)
    flux += rng.normal(0.0, noise_ppm * 1e-6, size=time.size)
    flux_err = np.full_like(time, noise_ppm * 1e-6)

    duration_days = duration_hours / 24.0
    phase = ((time - t0_btjd) / period_days) % 1.0
    in_transit = np.minimum(phase, 1.0 - phase) < (duration_days / period_days) / 2.0
    flux[in_transit] *= 1.0 - depth_frac

    return LightCurve(time=time, flux=flux, flux_err=flux_err)


def test_run_lc_checks_matches_current_wrapper_behavior() -> None:
    lc = _make_box_transit_lc()
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    enabled = {"V01", "V02", "V04", "V05", "V13", "V15"}
    expected = vet_lc_only(lc, eph, enabled=enabled)
    got = run_lc_checks(
        lc.to_internal(),
        period_days=eph.period_days,
        t0_btjd=eph.t0_btjd,
        duration_hours=eph.duration_hours,
    )

    assert [r.id for r in got] == [r.id for r in expected]
    for bridge_result, wrapper_result in zip(got, expected, strict=True):
        assert bridge_result.model_dump() == wrapper_result.model_dump()


def test_compute_timing_series_matches_current_wrapper_behavior() -> None:
    lc = _make_box_transit_lc()
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10_000.0)

    expected = timing_series(lc, candidate, min_snr=2.0)
    got = compute_timing_series(
        lc.to_internal(),
        period_days=eph.period_days,
        t0_btjd=eph.t0_btjd,
        duration_hours=eph.duration_hours,
        min_snr=2.0,
    )

    assert got.to_dict() == expected.to_dict()


def test_compute_alias_summary_matches_current_wrapper_behavior() -> None:
    lc = _make_box_transit_lc()
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)
    candidate = Candidate(ephemeris=eph, depth_ppm=10_000.0)

    expected = harmonic_power_summary(lc, candidate)
    got = compute_alias_summary(
        lc.to_internal(),
        period_days=eph.period_days,
        t0_btjd=eph.t0_btjd,
        duration_hours=eph.duration_hours,
    )

    assert got.to_dict() == expected.to_dict()


def test_compute_alias_scalar_signals_matches_underlying_primitives() -> None:
    lc = _make_box_transit_lc(noise_ppm=30.0, seed=11)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    # Inject deterministic non-primary structure to exercise all scalar outputs.
    time = np.asarray(lc.time, dtype=np.float64)
    flux = np.asarray(lc.flux, dtype=np.float64).copy()
    phase = ((time - eph.t0_btjd) / eph.period_days) % 1.0

    secondary = np.abs(phase - 0.5) < 0.015
    phase_shift = np.abs(phase - 0.3) < 0.015
    flux[secondary] *= 1.0 - 0.0035
    flux[phase_shift] *= 1.0 - 0.0025
    lc = LightCurve(time=time, flux=flux, flux_err=lc.flux_err)

    internal = lc.to_internal()
    summary = compute_alias_summary(
        internal,
        period_days=eph.period_days,
        t0_btjd=eph.t0_btjd,
        duration_hours=eph.duration_hours,
    )
    got = compute_alias_scalar_signals(
        internal,
        period_days=eph.period_days,
        t0_btjd=eph.t0_btjd,
        duration_hours=eph.duration_hours,
        harmonic_summary=summary,
    )

    mask = np.asarray(internal.valid_mask, dtype=np.bool_)
    time_v = np.asarray(internal.time, dtype=np.float64)[mask]
    flux_v = np.asarray(internal.flux, dtype=np.float64)[mask]
    flux_err_v = np.asarray(internal.flux_err, dtype=np.float64)[mask]
    base_score = next((h.score for h in summary.harmonics if h.harmonic == "P"), 0.0)
    expected_classification, _best, _ratio = classify_alias(summary.harmonics, base_score=base_score)
    expected_events = detect_phase_shift_events(
        time=time_v,
        flux=flux_v,
        flux_err=flux_err_v,
        period=eph.period_days,
        t0=eph.t0_btjd,
        n_phase_bins=10,
        significance_threshold=3.0,
    )
    expected_secondary = compute_secondary_significance(
        time=time_v,
        flux=flux_v,
        flux_err=flux_err_v,
        period=eph.period_days,
        t0=eph.t0_btjd,
        duration_hours=eph.duration_hours,
    )

    assert got["classification"] == expected_classification
    assert got["phase_shift_event_count"] == len(expected_events)
    expected_peak = max((e.significance for e in expected_events), default=None)
    if expected_peak is None:
        assert got["phase_shift_peak_sigma"] is None
    else:
        assert got["phase_shift_peak_sigma"] == pytest.approx(expected_peak)
    assert got["secondary_significance"] == pytest.approx(expected_secondary)
