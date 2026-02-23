from __future__ import annotations

import numpy as np

from tess_vetter.api.lc_only import (
    data_gaps,
    odd_even_depth,
    secondary_eclipse,
    transit_asymmetry,
    v_shape,
    vet_lc_only,
)
from tess_vetter.api.transit_primitives import odd_even_result
from tess_vetter.api.types import Ephemeris, LightCurve, VettingBundleResult


def _make_box_transit_lc(
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    baseline_days: float = 27.0,
    cadence_minutes: float = 10.0,
    depth_frac: float = 0.001,
    depth_frac_even: float | None = None,
    noise_ppm: float = 80.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    epoch = np.floor((time - t0_btjd + period_days / 2.0) / period_days).astype(int)

    if depth_frac_even is None:
        flux[in_transit] *= 1.0 - depth_frac
    else:
        for e in np.unique(epoch):
            dep = depth_frac_even if (int(e) % 2 == 0) else depth_frac
            m = in_transit & (epoch == e)
            flux[m] *= 1.0 - dep

    return time, flux, flux_err


def test_odd_even_result_metrics_only_fields() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=3.5, t0_btjd=0.5, duration_hours=2.5, depth_frac=0.01, noise_ppm=50.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    r = odd_even_result(lc, eph)
    assert r.n_odd >= 2
    assert r.n_even >= 2
    assert abs(r.relative_depth_diff_percent) < 10.0


def test_odd_even_depth_check_result_metrics_only() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=3.5,
        t0_btjd=0.5,
        duration_hours=2.5,
        depth_frac=0.015,
        depth_frac_even=0.007,
        noise_ppm=50.0,
        seed=1,
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    r = odd_even_depth(lc, eph)
    # New schema: status="ok" -> passed=True via backward-compat property
    # All results are metrics-only by design (the interpretation is left to host)
    assert r.status == "ok"
    assert r.passed is True
    # Check for metrics in new location
    assert float(r.metrics["rel_diff"]) > 0.1


def test_secondary_eclipse_check_result_metrics_only() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=5.0, t0_btjd=1.0, duration_hours=3.0, depth_frac=0.001, noise_ppm=80.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=5.0, t0_btjd=1.0, duration_hours=3.0)

    r = secondary_eclipse(lc, eph)
    # New schema: status="ok" -> passed=True via backward-compat property
    assert r.status == "ok"
    assert r.passed is True
    assert "secondary_depth_sigma" in r.metrics


def test_v_shape_check_metrics_only() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=5.0, t0_btjd=1.0, duration_hours=3.0, depth_frac=0.001, noise_ppm=50.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=5.0, t0_btjd=1.0, duration_hours=3.0)

    r = v_shape(lc, eph)
    # New schema: status="ok" -> passed=True via backward-compat property
    assert r.status == "ok"
    assert r.passed is True
    assert "tflat_ttotal_ratio" in r.metrics


def test_data_gaps_check_result_metrics_only() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=3.5, t0_btjd=0.5, duration_hours=2.5, depth_frac=0.001, noise_ppm=80.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    r = data_gaps(lc, eph)
    assert r.status == "ok"
    assert r.passed is True
    assert r.id == "V13"
    assert r.name == "data_gaps"
    assert "missing_frac_max" in r.metrics
    assert "n_epochs_evaluated" in r.metrics


def test_transit_asymmetry_check_result_metrics_only() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=3.5, t0_btjd=0.5, duration_hours=2.5, depth_frac=0.001, noise_ppm=80.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    r = transit_asymmetry(lc, eph)
    assert r.status == "ok"
    assert r.passed is True
    assert r.id == "V15"
    assert r.name == "transit_asymmetry"


def test_vet_lc_only_order_and_ids() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=3.5, t0_btjd=0.5, duration_hours=2.5, depth_frac=0.001, noise_ppm=80.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    results = vet_lc_only(lc, eph)
    assert [r.id for r in results] == ["V01", "V02", "V03", "V04", "V05", "V13", "V15"]
    # New schema: all results use status-based semantics
    # status="ok" maps to passed=True
    assert all(r.status == "ok" for r in results)
    assert all(r.passed is True for r in results)


def test_vet_lc_only_enabled_filter_respects_subset() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=3.5, t0_btjd=0.5, duration_hours=2.5, depth_frac=0.001, noise_ppm=80.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    results = vet_lc_only(lc, eph, enabled={"V01", "V02"})
    assert [r.id for r in results] == ["V01", "V02"]


def test_vet_lc_only_enabled_filter_includes_v13_v15() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=3.5, t0_btjd=0.5, duration_hours=2.5, depth_frac=0.001, noise_ppm=80.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    results = vet_lc_only(lc, eph, enabled={"V13", "V15"})
    assert [r.id for r in results] == ["V13", "V15"]


def test_vetting_bundle_result_from_checks() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=3.5, t0_btjd=0.5, duration_hours=2.5, depth_frac=0.001, noise_ppm=80.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    results = vet_lc_only(lc, eph)
    bundle = VettingBundleResult.from_checks(results)

    assert isinstance(bundle, VettingBundleResult)
    assert len(bundle.results) == 7
    assert [r.id for r in bundle.results] == ["V01", "V02", "V03", "V04", "V05", "V13", "V15"]
    assert bundle.provenance["source"] == "from_checks"
    assert bundle.provenance["checks_run"] == 7
    assert bundle.warnings == []


def test_vetting_bundle_result_from_checks_with_warnings() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=3.5, t0_btjd=0.5, duration_hours=2.5, depth_frac=0.001, noise_ppm=80.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    results = vet_lc_only(lc, eph, enabled={"V01"})
    bundle = VettingBundleResult.from_checks(results, warnings=["test_warning"])

    assert len(bundle.results) == 1
    assert bundle.warnings == ["test_warning"]


def test_vetting_bundle_result_from_checks_empty() -> None:
    bundle = VettingBundleResult.from_checks([])
    assert len(bundle.results) == 0
    assert bundle.provenance["checks_run"] == 0
    assert bundle.n_passed == 0
    assert bundle.all_passed is True  # vacuously true


def test_vetting_bundle_result_from_checks_properties() -> None:
    time, flux, flux_err = _make_box_transit_lc(
        period_days=3.5, t0_btjd=0.5, duration_hours=2.5, depth_frac=0.001, noise_ppm=80.0
    )
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    results = vet_lc_only(lc, eph)
    bundle = VettingBundleResult.from_checks(results)

    assert bundle.n_passed == 7
    assert bundle.n_failed == 0
    assert bundle.n_unknown == 0
    assert bundle.all_passed is True
    assert bundle.failed_check_ids == []
    assert bundle.get_result("V13") is not None
    assert bundle.get_result("V15") is not None
    assert bundle.get_result("V99") is None
