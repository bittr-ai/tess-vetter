from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.report import build_report

_DEPTH_PPM_ABS_TOL = 0.1
_SIGNIFICANCE_ABS_TOL = 0.1
_MORPHOLOGY_ABS_TOL = 1e-4


def _make_box_transit_lc(
    *,
    period_days: float = 3.5,
    t0_btjd: float = 0.5,
    duration_hours: float = 2.5,
    baseline_days: float = 90.0,
    cadence_minutes: float = 2.0,
    depth_frac: float = 0.01,
    noise_ppm: float = 50.0,
    seed: int = 123,
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
    flux[in_transit] *= 1.0 - depth_frac
    return time, flux, flux_err


def _build_payload() -> dict[str, object]:
    time, flux, flux_err = _make_box_transit_lc()
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5),
        depth_ppm=10000.0,
    )
    return build_report(lc, candidate).to_json()


def _assert_depth_ppm_close(actual: float, expected: float) -> None:
    assert actual == pytest.approx(expected, abs=_DEPTH_PPM_ABS_TOL)


def _assert_significance_close(actual: float, expected: float) -> None:
    assert actual == pytest.approx(expected, abs=_SIGNIFICANCE_ABS_TOL)


def _assert_morphology_close(actual: float, expected: float) -> None:
    assert actual == pytest.approx(expected, abs=_MORPHOLOGY_ABS_TOL)


def test_lc_and_lc_robustness_summaries_are_deterministic_with_invariants() -> None:
    payload_a = _build_payload()
    payload_b = _build_payload()

    summary_a = payload_a["summary"]
    summary_b = payload_b["summary"]
    assert summary_a["lc_summary"] == summary_b["lc_summary"]
    assert summary_a["lc_robustness_summary"] == summary_b["lc_robustness_summary"]

    lc_summary = summary_a["lc_summary"]
    assert isinstance(lc_summary, dict)
    assert lc_summary["n_points"] >= lc_summary["n_valid"] >= 0
    assert 0.0 <= lc_summary["gap_fraction"] <= 1.0
    assert lc_summary["cadence_seconds"] > 0.0
    assert lc_summary["snr"] >= 0.0

    rb = summary_a["lc_robustness_summary"]
    assert isinstance(rb, dict)
    if rb["loto_snr_min"] is not None and rb["loto_snr_max"] is not None:
        assert rb["loto_snr_min"] <= rb["loto_snr_max"]
    assert rb["loto_depth_shift_ppm_max"] is None or rb["loto_depth_shift_ppm_max"] >= 0.0


def test_secondary_scan_summary_and_depth_projection_invariants() -> None:
    payload_a = _build_payload()
    payload_b = _build_payload()

    summary_a = payload_a["summary"]
    summary_b = payload_b["summary"]
    assert summary_a["secondary_scan_summary"] == summary_b["secondary_scan_summary"]

    secondary = summary_a["secondary_scan_summary"]
    assert isinstance(secondary, dict)

    coverage = secondary["phase_coverage_fraction"]
    if coverage is not None:
        assert 0.0 <= coverage <= 1.0

    strongest_dip_depth_ppm = secondary["strongest_dip_depth_ppm"]
    if strongest_dip_depth_ppm is not None:
        assert strongest_dip_depth_ppm >= 0.0

    scan_plot = payload_a["plot_data"]["secondary_scan"]
    strongest_dip_flux = scan_plot["strongest_dip_flux"]
    if strongest_dip_flux is not None and strongest_dip_depth_ppm is not None:
        expected_depth_ppm = (1.0 - float(strongest_dip_flux)) * 1e6
        _assert_depth_ppm_close(float(strongest_dip_depth_ppm), expected_depth_ppm)


def test_morphology_and_significance_fields_match_summary_with_tolerance_policy() -> None:
    payload = _build_payload()
    summary = payload["summary"]
    checks = summary["checks"]
    rb = summary["lc_robustness_summary"]

    assert isinstance(checks, dict)
    assert isinstance(rb, dict)

    v_shape_metric = rb["v_shape_metric"]
    if v_shape_metric is not None:
        assert 0.0 <= v_shape_metric <= 1.0
        check_v05 = checks.get("V05")
        expected_v_shape = None if not check_v05 else check_v05["metrics"].get("tflat_ttotal_ratio")
        if expected_v_shape is not None:
            _assert_morphology_close(float(v_shape_metric), float(expected_v_shape))

    asymmetry_sigma = rb["asymmetry_sigma"]
    if asymmetry_sigma is not None:
        check_v15 = checks.get("V15")
        expected_asymmetry = None if not check_v15 else check_v15["metrics"].get("asymmetry_sigma")
        if expected_asymmetry is not None:
            _assert_significance_close(float(asymmetry_sigma), float(expected_asymmetry))

    secondary_depth_sigma = rb["secondary_depth_sigma"]
    if secondary_depth_sigma is not None:
        check_v02 = checks.get("V02")
        expected_secondary_sigma = (
            None if not check_v02 else check_v02["metrics"].get("secondary_depth_sigma")
        )
        if expected_secondary_sigma is not None:
            _assert_significance_close(
                float(secondary_depth_sigma),
                float(expected_secondary_sigma),
            )
