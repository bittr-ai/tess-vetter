from __future__ import annotations

import numpy as np

import bittr_tess_vetter.api as btv


def _make_lc_with_box_transit(
    *,
    period_days: float = 3.5,
    duration_hours: float = 2.5,
    depth: float = 0.001,
    n_cadences: int = 2000,
    cadence_seconds: float = 120.0,
    noise_sigma: float = 5e-4,
    seed: int = 42,
) -> tuple[btv.LightCurve, btv.Ephemeris]:
    rng = np.random.default_rng(seed)
    cadence_days = cadence_seconds / 86400.0
    time = 2458000.0 + np.arange(n_cadences) * cadence_days
    t0_btjd = float(time[0] + 0.5)
    flux = 1.0 + rng.normal(0.0, noise_sigma, n_cadences)
    flux_err = np.full(n_cadences, noise_sigma)

    duration_days = duration_hours / 24.0
    phase = ((time - t0_btjd) / period_days + 0.5) % 1.0 - 0.5
    in_transit = np.abs(phase) < (duration_days / (2.0 * period_days))
    flux[in_transit] -= depth

    lc = btv.LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = btv.Ephemeris(period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours)
    return lc, eph


def test_run_check_returns_metrics() -> None:
    lc, eph = _make_lc_with_box_transit()
    cand = btv.Candidate(ephemeris=eph, depth_ppm=1000.0)

    r = btv.run_check(lc=lc, candidate=cand, check_id="V01")
    assert r.id == "V01"
    assert r.status == "ok"
    assert "delta_sigma" in r.metrics


def test_run_check_skips_pixel_check_without_tpf() -> None:
    lc, eph = _make_lc_with_box_transit()
    cand = btv.Candidate(ephemeris=eph, depth_ppm=1000.0)

    r = btv.run_check(lc=lc, candidate=cand, check_id="V08")
    assert r.id == "V08"
    assert r.status == "skipped"
    assert any(f.startswith("SKIPPED:") for f in r.flags)


def test_session_runs_multiple_checks_in_order() -> None:
    lc, eph = _make_lc_with_box_transit()
    cand = btv.Candidate(ephemeris=eph, depth_ppm=1000.0)

    session = btv.VettingSession.from_api(lc=lc, candidate=cand)
    results = session.run_many(["V01", "V08"])
    assert [r.id for r in results] == ["V01", "V08"]
    assert results[0].status == "ok"
    assert results[1].status == "skipped"


def test_format_check_result_includes_metrics_block() -> None:
    lc, eph = _make_lc_with_box_transit()
    cand = btv.Candidate(ephemeris=eph, depth_ppm=1000.0)
    r = btv.run_check(lc=lc, candidate=cand, check_id="V01")

    out = btv.format_check_result(r, max_metrics=5)
    assert "Check Result" in out
    assert "Metrics" in out
    assert "delta_sigma" in out

