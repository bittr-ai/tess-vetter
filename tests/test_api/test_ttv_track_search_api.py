from __future__ import annotations

import numpy as np


def _make_synthetic_ttv_lc(
    *,
    period_days: float = 10.0,
    t0_btjd: float = 1000.0,
    duration_hours: float = 4.0,
    depth_frac: float = 0.01,
    offsets_days: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 3 observing windows separated by large gaps (>5d)
    windows = [
        np.arange(995.0, 1025.0, 0.02),
        np.arange(1045.0, 1075.0, 0.02),
        np.arange(1095.0, 1125.0, 0.02),
    ]
    time = np.concatenate(windows)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 0.001)

    if offsets_days is None:
        offsets_days = [0.0, 0.05, -0.03]
    duration_days = duration_hours / 24.0

    # Create transits with per-window offsets.
    for w_idx, (t_start, t_end) in enumerate([(995.0, 1025.0), (1045.0, 1075.0), (1095.0, 1125.0)]):
        # expected linear transit times near this window
        n_min = int(np.floor((t_start - t0_btjd) / period_days)) - 1
        n_max = int(np.ceil((t_end - t0_btjd) / period_days)) + 1
        for n in range(n_min, n_max + 1):
            t_transit = t0_btjd + n * period_days + offsets_days[w_idx]
            in_tr = np.abs(time - t_transit) < duration_days / 2.0
            flux[in_tr] -= depth_frac

    # Add small noise deterministically
    rng = np.random.default_rng(0)
    flux = flux + rng.normal(0.0, 0.0005, size=flux.size)
    return time, flux, flux_err


def test_should_run_ttv_search_gate() -> None:
    from tess_vetter.api.ttv_track_search import should_run_ttv_search

    t = np.concatenate([np.arange(0.0, 10.0, 0.1), np.arange(200.0, 210.0, 0.1)])
    assert should_run_ttv_search(t, min_baseline_days=50.0, min_windows=2) is True
    assert should_run_ttv_search(t, min_baseline_days=500.0, min_windows=2) is False


def test_run_ttv_track_search_finds_improvement() -> None:
    from tess_vetter.api.ttv_track_search import TTVSearchBudget, run_ttv_track_search

    period_days = 10.0
    t0_btjd = 1000.0
    duration_hours = 4.0
    time, flux, flux_err = _make_synthetic_ttv_lc(
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        offsets_days=[0.0, 0.07, -0.05],
    )

    result = run_ttv_track_search(
        time,
        flux,
        flux_err,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        period_span_fraction=0.0,  # only evaluate base period
        period_steps=1,
        max_offset_days=0.1,
        n_offset_steps=5,
        max_tracks_per_period=200,
        min_score_improvement=0.5,  # keep test robust across noise
        budget=TTVSearchBudget(
            max_runtime_seconds=5.0, max_period_evaluations=10, max_track_hypotheses=5000
        ),
        random_seed=42,
    )

    assert result.n_periods_searched == 1
    assert result.n_tracks_evaluated > 0
    assert result.candidates, "expected at least one candidate with TTV track improvement"
    assert result.candidates[0].best_track.score_improvement >= 0.5


def test_run_ttv_track_search_deterministic() -> None:
    from tess_vetter.api.ttv_track_search import TTVSearchBudget, run_ttv_track_search

    time, flux, flux_err = _make_synthetic_ttv_lc()

    kwargs = {
        "period_days": 10.0,
        "t0_btjd": 1000.0,
        "duration_hours": 4.0,
        "period_span_fraction": 0.0,
        "period_steps": 1,
        "max_offset_days": 0.1,
        "n_offset_steps": 5,
        "max_tracks_per_period": 200,
        "min_score_improvement": 0.5,
        "budget": TTVSearchBudget(
            max_runtime_seconds=5.0, max_period_evaluations=10, max_track_hypotheses=5000
        ),
    }
    r1 = run_ttv_track_search(time, flux, flux_err, random_seed=123, **kwargs)
    r2 = run_ttv_track_search(time, flux, flux_err, random_seed=123, **kwargs)

    assert r1.candidates and r2.candidates
    assert (
        r1.candidates[0].best_track.window_offsets_days
        == r2.candidates[0].best_track.window_offsets_days
    )


def test_estimate_search_cost_schema_contract() -> None:
    from tess_vetter.api.ttv_track_search import (
        TTV_SEARCH_COST_SCHEMA_VERSION,
        TTVSearchBudget,
        estimate_search_cost,
    )

    time, _, _ = _make_synthetic_ttv_lc()
    payload = estimate_search_cost(
        time,
        period_steps=4,
        n_offset_steps=5,
        max_tracks_per_period=200,
        budget=TTVSearchBudget(
            max_runtime_seconds=5.0,
            max_period_evaluations=10,
            max_track_hypotheses=5000,
        ),
    )
    assert payload["schema_version"] == TTV_SEARCH_COST_SCHEMA_VERSION
    assert set(payload) == {
        "schema_version",
        "n_windows",
        "period_steps",
        "tracks_per_period",
        "theoretical_total_tracks",
        "budget_limited_tracks",
        "estimated_seconds",
        "will_hit_budget",
    }
    assert payload["n_windows"] >= 1
    assert payload["period_steps"] >= 1
    assert isinstance(payload["will_hit_budget"], bool)


def test_ttv_track_search_result_payload_parity() -> None:
    from tess_vetter.api.ttv_track_search import (
        TTV_TRACK_SEARCH_RESULT_SCHEMA_VERSION,
        TTVSearchBudget,
        run_ttv_track_search,
        to_ttv_track_search_result_payload,
    )

    time, flux, flux_err = _make_synthetic_ttv_lc()
    result = run_ttv_track_search(
        time,
        flux,
        flux_err,
        period_days=10.0,
        t0_btjd=1000.0,
        duration_hours=4.0,
        period_span_fraction=0.0,
        period_steps=1,
        max_offset_days=0.1,
        n_offset_steps=5,
        max_tracks_per_period=200,
        min_score_improvement=0.5,
        budget=TTVSearchBudget(
            max_runtime_seconds=5.0,
            max_period_evaluations=10,
            max_track_hypotheses=5000,
        ),
        random_seed=123,
    )
    payload = to_ttv_track_search_result_payload(result)
    assert payload["schema_version"] == TTV_TRACK_SEARCH_RESULT_SCHEMA_VERSION
    assert payload["candidates"] == result.to_dict()["candidates"]
    assert payload["n_periods_searched"] == result.n_periods_searched
    assert payload["n_tracks_evaluated"] == result.n_tracks_evaluated
    assert payload["budget_exhausted"] == result.budget_exhausted


def test_run_ttv_track_search_for_candidate_parity_with_direct_call() -> None:
    from tess_vetter.api.ttv_track_search import (
        TTVSearchBudget,
        run_ttv_track_search,
        run_ttv_track_search_for_candidate,
    )
    from tess_vetter.api.types import Candidate, Ephemeris, LightCurve

    time, flux, flux_err = _make_synthetic_ttv_lc()
    # Ensure wrapper masking path is exercised (NaN should be filtered).
    flux = flux.copy()
    flux[0] = np.nan

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    candidate = Candidate(ephemeris=Ephemeris(period_days=10.0, t0_btjd=1000.0, duration_hours=4.0))
    budget = TTVSearchBudget(max_runtime_seconds=5.0, max_period_evaluations=10, max_track_hypotheses=5000)

    kwargs = {
        "period_span_fraction": 0.0,
        "period_steps": 1,
        "max_offset_days": 0.1,
        "n_offset_steps": 5,
        "max_tracks_per_period": 200,
        "min_score_improvement": 0.5,
        "gap_threshold_days": 5.0,
        "budget": budget,
        "random_seed": 123,
        "normalize_flux": True,
    }

    via_candidate = run_ttv_track_search_for_candidate(lc, candidate, **kwargs)
    internal = lc.to_internal()
    mask = internal.valid_mask
    via_direct = run_ttv_track_search(
        internal.time[mask],
        internal.flux[mask],
        internal.flux_err[mask],
        period_days=candidate.ephemeris.period_days,
        t0_btjd=candidate.ephemeris.t0_btjd,
        duration_hours=candidate.ephemeris.duration_hours,
        **kwargs,
    )
    c_dict = via_candidate.to_dict()
    d_dict = via_direct.to_dict()
    assert c_dict["n_periods_searched"] == d_dict["n_periods_searched"]
    assert c_dict["n_tracks_evaluated"] == d_dict["n_tracks_evaluated"]
    assert c_dict["budget_exhausted"] == d_dict["budget_exhausted"]
    assert c_dict["provenance"] == d_dict["provenance"]
    assert len(c_dict["candidates"]) == len(d_dict["candidates"])
    if c_dict["candidates"]:
        c0 = c_dict["candidates"][0]
        d0 = d_dict["candidates"][0]
        assert c0["best_track"] == d0["best_track"]
        assert c0["alternative_tracks"] == d0["alternative_tracks"]
        assert c0["periodic_score"] == d0["periodic_score"]
        assert c0["per_transit_residuals"] == d0["per_transit_residuals"]
        assert c0["provenance"] == d0["provenance"]
