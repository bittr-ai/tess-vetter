from __future__ import annotations

from dataclasses import fields
from types import MappingProxyType

import numpy as np

import tess_vetter.api as btv
from tess_vetter.api import check_runner as check_runner_contracts


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


def test_run_check_contract_schema_fields() -> None:
    assert [f.name for f in fields(check_runner_contracts.RunCheckRequest)] == [
        "lc",
        "candidate",
        "check_id",
        "stellar",
        "tpf",
        "network",
        "ra_deg",
        "dec_deg",
        "tic_id",
        "context",
        "preset",
        "registry",
        "pipeline_config",
    ]
    assert [f.name for f in fields(check_runner_contracts.RunCheckResponse)] == ["result"]


def test_run_checks_contract_schema_fields() -> None:
    assert [f.name for f in fields(check_runner_contracts.RunChecksRequest)] == [
        "lc",
        "candidate",
        "check_ids",
        "stellar",
        "tpf",
        "network",
        "ra_deg",
        "dec_deg",
        "tic_id",
        "context",
        "preset",
        "registry",
        "pipeline_config",
    ]
    assert [f.name for f in fields(check_runner_contracts.RunChecksResponse)] == ["results"]


def test_run_check_contract_runtime_parity() -> None:
    lc, eph = _make_lc_with_box_transit()
    cand = btv.Candidate(ephemeris=eph, depth_ppm=1000.0)
    request = check_runner_contracts.RunCheckRequest(lc=lc, candidate=cand, check_id="V01")

    wrapped = btv.run_check(lc=lc, candidate=cand, check_id="V01")
    contracted = check_runner_contracts.run_check_contract(request).result

    assert wrapped.id == contracted.id
    assert wrapped.status == contracted.status
    assert wrapped.flags == contracted.flags
    assert wrapped.metrics == contracted.metrics


def test_run_checks_contract_runtime_parity() -> None:
    lc, eph = _make_lc_with_box_transit()
    cand = btv.Candidate(ephemeris=eph, depth_ppm=1000.0)
    request = check_runner_contracts.RunChecksRequest(lc=lc, candidate=cand, check_ids=["V01", "V08"])

    wrapped = btv.run_checks(lc=lc, candidate=cand, check_ids=["V01", "V08"])
    contracted = check_runner_contracts.run_checks_contract(request).results

    assert [r.id for r in wrapped] == [r.id for r in contracted] == ["V01", "V08"]
    assert [r.status for r in wrapped] == [r.status for r in contracted]
    assert [r.flags for r in wrapped] == [r.flags for r in contracted]


def test_run_check_accepts_readonly_context_mapping() -> None:
    lc, eph = _make_lc_with_box_transit()
    cand = btv.Candidate(ephemeris=eph, depth_ppm=1000.0)
    readonly_context = MappingProxyType({"sector": 1, "source": "unit-test"})

    result = btv.run_check(lc=lc, candidate=cand, check_id="V01", context=readonly_context)

    assert result.id == "V01"
    assert result.status == "ok"


def test_run_checks_accepts_readonly_context_mapping() -> None:
    lc, eph = _make_lc_with_box_transit()
    cand = btv.Candidate(ephemeris=eph, depth_ppm=1000.0)
    readonly_context = MappingProxyType({"sector": 1, "source": "unit-test"})

    results = btv.run_checks(lc=lc, candidate=cand, check_ids=["V01", "V08"], context=readonly_context)

    assert [r.id for r in results] == ["V01", "V08"]
    assert [r.status for r in results] == ["ok", "skipped"]
