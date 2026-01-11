from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.exovetter import modshift, sweet, vet_exovetter
from bittr_tess_vetter.api.types import Candidate, CheckResult, Ephemeris, LightCurve


def _minimal_lc() -> LightCurve:
    time = np.linspace(1000.0, 1010.0, 2000, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.ones_like(time) * 100e-6
    return LightCurve(time=time, flux=flux, flux_err=flux_err)


def test_modshift_disabled_returns_skipped() -> None:
    lc = _minimal_lc()
    cand = Candidate(ephemeris=Ephemeris(period_days=3.0, t0_btjd=1001.0, duration_hours=2.0), depth_ppm=1000)
    r = modshift(lc, cand, enabled=False)
    assert isinstance(r, CheckResult)
    assert r.id == "V11"
    assert r.passed is None
    assert r.details.get("status") == "skipped"


def test_sweet_missing_depth_returns_error_result() -> None:
    lc = _minimal_lc()
    cand = Candidate(ephemeris=Ephemeris(period_days=3.0, t0_btjd=1001.0, duration_hours=2.0))
    r = sweet(lc, cand, enabled=True)
    assert r.id == "V12"
    assert r.passed is None
    assert r.details.get("status") == "error"
    assert "depth" in str(r.details.get("reason", "")).lower()


def test_vet_exovetter_enabled_set_filters_and_returns_two_results() -> None:
    lc = _minimal_lc()
    cand = Candidate(
        ephemeris=Ephemeris(period_days=3.0, t0_btjd=1001.0, duration_hours=2.0),
        depth_fraction=0.001,
    )
    results = vet_exovetter(lc, cand, enabled={"V11"})
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].id == "V11"
    assert results[1].id == "V12"
    assert results[1].details.get("status") == "skipped"


def test_modshift_returns_metrics_only_result_when_dependency_missing() -> None:
    # This test is written to pass whether or not exovetter is installed:
    # - if installed, we still expect passed=None (metrics-only) and _metrics_only in details
    # - if not installed, we expect EXOVETTER_IMPORT_ERROR warning
    lc = _minimal_lc()
    cand = Candidate(
        ephemeris=Ephemeris(period_days=3.0, t0_btjd=1001.0, duration_hours=2.0),
        depth_fraction=0.001,
    )
    r = modshift(lc, cand, enabled=True)
    assert r.id == "V11"
    assert r.passed is None
    assert r.details.get("_metrics_only") is True or "EXOVETTER_IMPORT_ERROR" in (
        r.details.get("warnings", []) or []
    )

