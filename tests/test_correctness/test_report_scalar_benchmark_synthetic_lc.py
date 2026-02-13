from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.api.vet import vet_candidate
from bittr_tess_vetter.report import build_report


def _make_synthetic_lc(
    *,
    period_days: float = 3.5,
    t0_btjd: float = 0.5,
    duration_hours: float = 2.5,
    baseline_days: float = 90.0,
    cadence_minutes: float = 2.0,
    depth_frac: float = 0.01,
    noise_ppm: float = 50.0,
    seed: int = 123,
) -> LightCurve:
    rng = np.random.default_rng(seed)
    dt_days = cadence_minutes / (24.0 * 60.0)
    time = np.arange(0.0, baseline_days, dt_days, dtype=np.float64)
    flux = np.ones_like(time)
    flux += rng.normal(0.0, noise_ppm * 1e-6, size=time.size)
    flux_err = np.full_like(time, noise_ppm * 1e-6)

    duration_days = duration_hours / 24.0
    half_phase = (duration_days / period_days) / 2.0
    phase = ((time - t0_btjd) / period_days) % 1.0
    phase_dist = np.minimum(phase, 1.0 - phase)
    in_transit = phase_dist < half_phase
    flux[in_transit] *= 1.0 - depth_frac
    return LightCurve(time=time, flux=flux, flux_err=flux_err)


def test_synthetic_lc_scalar_benchmark_gate() -> None:
    lc = _make_synthetic_lc()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5),
        depth_ppm=10_000.0,
    )

    report_a = build_report(lc, candidate, include_additional_plots=False).to_json()
    report_b = build_report(lc, candidate, include_additional_plots=False).to_json()
    assert report_a["summary"] == report_b["summary"]

    summary = report_a["summary"]
    lc_summary = summary["lc_summary"]
    odd_even = summary["odd_even_summary"]
    timing = summary["timing_summary"]
    robustness = summary["lc_robustness_summary"]

    assert lc_summary["snr"] is not None
    assert float(lc_summary["snr"]) >= 10.0
    assert int(lc_summary["n_valid"]) > 1_000

    assert odd_even["depth_diff_ppm"] is not None
    assert odd_even["depth_diff_sigma"] is not None
    assert abs(float(odd_even["depth_diff_ppm"])) < 2_000.0
    assert abs(float(odd_even["depth_diff_sigma"])) < 5.0
    assert isinstance(odd_even["is_significant"], bool)
    assert isinstance(odd_even["flags"], list)

    assert int(timing["n_epochs_measured"]) >= 0
    if int(timing["n_epochs_measured"]) > 0:
        assert int(timing["outlier_count"]) <= int(timing["n_epochs_measured"])

    assert robustness["v_shape_metric"] is None or 0.0 <= float(robustness["v_shape_metric"]) <= 1.0
    assert robustness["secondary_depth_sigma"] is None or float(robustness["secondary_depth_sigma"]) >= 0.0

    bundle = vet_candidate(lc, candidate)
    ids = {r.id for r in bundle.results}
    assert {"V01", "V02", "V04", "V05", "V13", "V15"}.issubset(ids)
