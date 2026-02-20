from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.report._build_utils import (
    _bin_phase_data,
    _depth_ppm_to_flux,
    _downsample_phase_preserving_transit,
    _downsample_preserving_transits,
    _estimate_flux_err_fallback,
    _get_valid_time_flux,
    _get_valid_time_flux_quality,
    _red_noise_beta,
    _suggest_flux_y_range,
    _thin_evenly,
    _to_internal_lightcurve,
)


def _make_internal_lc(*, flux_err: np.ndarray | None = None) -> LightCurveData:
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    flux = np.array([1.0, 0.999, 1.001], dtype=np.float64)
    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=np.array([1e-4, 1e-4, 1e-4], dtype=np.float64) if flux_err is None else flux_err,
        quality=np.array([0, 1, 0], dtype=np.int32),
        valid_mask=np.array([True, True, True], dtype=np.bool_),
        tic_id=42,
        sector=5,
        cadence_seconds=120.0,
    )


def test_suggest_flux_y_range_handles_small_and_degenerate_inputs() -> None:
    assert _suggest_flux_y_range(np.array([1.0, 1.0], dtype=np.float64)) is None

    lo, hi = _suggest_flux_y_range(np.array([1.0, 1.0, 1.0], dtype=np.float64)) or (None, None)
    assert lo is not None and hi is not None
    assert lo < 1.0 < hi

    lo2, hi2 = _suggest_flux_y_range(
        np.array([0.99, 1.0, 1.01], dtype=np.float64), padding_fraction=-1.0
    ) or (None, None)
    assert lo2 is not None and hi2 is not None
    assert lo2 >= 0.99
    assert hi2 <= 1.01


@pytest.mark.parametrize("depth", [None, float("nan"), float("inf"), 0.0, -1.0])
def test_depth_ppm_to_flux_invalid_inputs(depth: float | None) -> None:
    assert _depth_ppm_to_flux(depth) is None


def test_depth_ppm_to_flux_positive() -> None:
    assert _depth_ppm_to_flux(10_000.0) == pytest.approx(0.99)


def test_estimate_flux_err_fallback_handles_empty_and_constant() -> None:
    assert _estimate_flux_err_fallback(np.array([], dtype=np.float64)) == pytest.approx(1e-6)
    assert _estimate_flux_err_fallback(np.array([1.0, 1.0, 1.0], dtype=np.float64)) == pytest.approx(1e-6)


def test_estimate_flux_err_fallback_uses_point_to_point_scatter_when_available() -> None:
    flux = np.array([1.0, 1.0004, 0.9996, 1.0005, 0.9995], dtype=np.float64)
    sigma = _estimate_flux_err_fallback(flux)
    assert np.isfinite(sigma)
    assert sigma > 0.0


def test_to_internal_lightcurve_uses_internal_directly_when_flux_err_present() -> None:
    internal = _make_internal_lc()
    lc = SimpleNamespace(to_internal=lambda: internal, flux_err=np.array([1e-4, 1e-4, 1e-4]))

    out = _to_internal_lightcurve(lc)
    assert out is internal


def test_to_internal_lightcurve_rebuilds_internal_when_flux_err_missing() -> None:
    internal = _make_internal_lc(flux_err=np.zeros(3, dtype=np.float64))
    lc = SimpleNamespace(to_internal=lambda: internal, flux_err=None)

    out = _to_internal_lightcurve(lc)
    assert out is not internal
    assert np.all(out.flux_err > 0.0)
    assert out.provenance == internal.provenance


def test_to_internal_lightcurve_validates_input_shapes() -> None:
    with pytest.raises(ValueError, match="time and flux must have the same length"):
        _to_internal_lightcurve(SimpleNamespace(time=[0.0, 1.0], flux=[1.0]))

    with pytest.raises(ValueError, match="flux_err must have the same length"):
        _to_internal_lightcurve(
            SimpleNamespace(time=[0.0, 1.0], flux=[1.0, 1.0], flux_err=[1e-4], valid_mask=None)
        )

    with pytest.raises(ValueError, match="quality must have the same length"):
        _to_internal_lightcurve(
            SimpleNamespace(
                time=[0.0, 1.0],
                flux=[1.0, 1.0],
                flux_err=[1e-4, 1e-4],
                quality=[0],
                valid_mask=[True, True],
            )
        )

    with pytest.raises(ValueError, match="valid_mask must have the same length"):
        _to_internal_lightcurve(
            SimpleNamespace(
                time=[0.0, 1.0],
                flux=[1.0, 1.0],
                flux_err=[1e-4, 1e-4],
                quality=[0, 0],
                valid_mask=[True],
            )
        )


def test_to_internal_lightcurve_sets_defaults_and_filters_non_finite() -> None:
    lc = SimpleNamespace(time=[0.0, 1.0, np.nan], flux=[1.0, 0.999, 1.001])
    out = _to_internal_lightcurve(lc)
    assert np.array_equal(out.quality, np.array([0, 0, 0], dtype=np.int32))
    assert np.array_equal(out.valid_mask, np.array([True, True, False], dtype=np.bool_))
    assert np.all(out.flux_err > 0.0)


def test_get_valid_time_flux_applies_finite_and_valid_mask() -> None:
    lc = SimpleNamespace(
        time=np.array([0.0, 1.0, np.nan], dtype=np.float64),
        flux=np.array([1.0, np.nan, 1.1], dtype=np.float64),
        valid_mask=np.array([True, True, True], dtype=np.bool_),
    )
    t, f = _get_valid_time_flux(lc)
    assert t.tolist() == [0.0]
    assert f.tolist() == [1.0]


def test_get_valid_time_flux_quality_handles_missing_or_mismatched_quality() -> None:
    lc_missing = SimpleNamespace(
        time=np.array([0.0, 1.0], dtype=np.float64),
        flux=np.array([1.0, 1.0], dtype=np.float64),
        valid_mask=np.array([True, False], dtype=np.bool_),
        quality=None,
    )
    t1, f1, q1 = _get_valid_time_flux_quality(lc_missing)
    assert t1.tolist() == [0.0]
    assert f1.tolist() == [1.0]
    assert q1 is None

    lc_mismatch = SimpleNamespace(
        time=np.array([0.0, 1.0], dtype=np.float64),
        flux=np.array([1.0, 1.0], dtype=np.float64),
        valid_mask=np.array([True, True], dtype=np.bool_),
        quality=np.array([0], dtype=np.int32),
    )
    _, _, q2 = _get_valid_time_flux_quality(lc_mismatch)
    assert q2 is None


def test_thin_evenly_preserves_endpoints_and_budget() -> None:
    arr = np.arange(10, dtype=np.int64)
    thinned = _thin_evenly(arr, max_points=4)
    assert len(thinned) == 4
    assert thinned[0] == 0
    assert thinned[-1] == 9

    same = _thin_evenly(arr, max_points=20)
    assert np.array_equal(same, arr)


def test_red_noise_beta_guardrails_and_nominal_path() -> None:
    residuals = np.array([0.0, 0.1, -0.1, 0.05, -0.05], dtype=np.float64)
    times = np.linspace(0.0, 1.0, residuals.size)
    assert _red_noise_beta(residuals, times, bin_size_days=0.1) is None
    assert _red_noise_beta(np.tile(residuals, 3), np.linspace(0.0, 3.0, 15), bin_size_days=0.0) is None

    r = np.sin(np.linspace(0.0, 4.0 * np.pi, 60))
    t = np.linspace(0.0, 6.0, 60)
    beta = _red_noise_beta(r, t, bin_size_days=0.5)
    assert beta is not None
    assert beta >= 1.0


def test_bin_phase_data_edge_handling() -> None:
    centers, fluxes, errors = _bin_phase_data(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        period_days=1.0,
        bin_minutes=30.0,
    )
    assert centers == []
    assert fluxes == []
    assert errors == []

    centers2, fluxes2, errors2 = _bin_phase_data(
        np.array([-0.2, -0.1], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        period_days=1.0,
        bin_minutes=30.0,
        phase_range=(0.1, 0.2),
    )
    assert centers2 == []
    assert fluxes2 == []
    assert errors2 == []

    centers3, _, errors3 = _bin_phase_data(
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([1.0, 0.99], dtype=np.float64),
        period_days=1.0,
        bin_minutes=720.0,
        phase_range=(0.0, 1.0),
    )
    assert len(centers3) == 2
    assert errors3 == [None, None]


def test_downsample_preserving_transits_short_circuit_and_balanced_sampling() -> None:
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    flux = np.array([1.0, 0.99, 1.0], dtype=np.float64)
    mask = np.array([False, True, False], dtype=np.bool_)
    t_out, f_out, m_out = _downsample_preserving_transits(time, flux, mask, max_points=10)
    assert t_out == time.tolist()
    assert f_out == flux.tolist()
    assert m_out == mask.tolist()

    time2 = np.arange(10.0, dtype=np.float64)
    flux2 = np.ones(10, dtype=np.float64)
    mask2 = np.array([False, False, True, False, False, True, False, False, False, False], dtype=np.bool_)
    t_out2, _, m_out2 = _downsample_preserving_transits(time2, flux2, mask2, max_points=5)
    assert len(t_out2) == 5
    assert sum(m_out2) == 2


def test_downsample_phase_returns_original_when_far_budget_is_sufficient() -> None:
    phase = np.array([-0.2, -0.05, 0.0, 0.05, 0.2], dtype=np.float64)
    flux = np.ones_like(phase)
    out_p, out_f = _downsample_phase_preserving_transit(
        phase,
        flux,
        max_points=10,
        near_transit_half_phase=0.1,
    )
    assert np.array_equal(out_p, phase)
    assert np.array_equal(out_f, flux)
