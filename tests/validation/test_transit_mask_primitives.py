from __future__ import annotations

import numpy as np

from tess_vetter.validation.base import (
    count_transits,
    get_in_transit_mask,
    get_odd_even_transit_indices,
    get_out_of_transit_mask,
    measure_transit_depth,
    phase_fold,
)


def test_phase_fold_is_centered_and_bounded() -> None:
    time = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    period = 2.0
    t0 = 11.0

    phase, _ = phase_fold(time, time, period, t0)

    assert np.all(phase >= -0.5)
    assert np.all(phase <= 0.5)
    # Exact mid-transit at t0 should map to phase 0.
    assert phase[1] == 0.0


def test_in_transit_mask_selects_expected_points() -> None:
    # Cadence 0.1 d, period 1 d, duration 2.4 h => 0.1 d window.
    time = np.array([0.45, 0.50, 0.55, 1.45, 1.50, 1.55], dtype=np.float64)
    period = 1.0
    t0 = 0.5
    duration_hours = 2.4

    mask = get_in_transit_mask(time, period, t0, duration_hours)
    expected = np.array([False, True, False, False, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_out_of_transit_mask_is_complement_of_in_transit() -> None:
    time = np.linspace(0, 2, 21, dtype=np.float64)
    period = 1.0
    t0 = 0.5
    duration_hours = 2.0

    in_mask = get_in_transit_mask(time, period, t0, duration_hours, buffer_factor=1.0)
    out_mask = get_out_of_transit_mask(time, period, t0, duration_hours, buffer_factor=1.0)
    finite = np.isfinite(time)
    np.testing.assert_array_equal(out_mask, finite & (~in_mask))


def test_masks_exclude_non_finite_time_points() -> None:
    time = np.array([0.4, 0.5, np.nan, 0.6, np.inf], dtype=np.float64)
    period = 1.0
    t0 = 0.5
    duration_hours = 2.0

    in_mask = get_in_transit_mask(time, period, t0, duration_hours)
    out_mask = get_out_of_transit_mask(time, period, t0, duration_hours)

    assert in_mask[2] is np.False_
    assert out_mask[2] is np.False_
    assert in_mask[4] is np.False_
    assert out_mask[4] is np.False_


def test_measure_transit_depth_recovers_known_depth() -> None:
    rng = np.random.default_rng(0)
    time = np.linspace(0, 10, 2000, dtype=np.float64)
    flux = np.ones_like(time) + rng.normal(0, 5e-5, size=time.shape)

    period = 2.0
    t0 = 0.3
    duration_hours = 2.0
    injected_depth = 300e-6

    in_mask = get_in_transit_mask(time, period, t0, duration_hours)
    out_mask = get_out_of_transit_mask(time, period, t0, duration_hours)
    flux[in_mask] -= injected_depth

    depth, depth_err = measure_transit_depth(flux, in_mask, out_mask)
    assert depth > 0
    assert depth_err > 0
    # Robust median estimator should be close on this synthetic injection.
    assert np.isclose(depth, injected_depth, rtol=0.2)


def test_count_transits_counts_covered_events() -> None:
    # 27 days baseline, period 3 => ~9 transits. With 10-min cadence and 3-hr duration,
    # min_points=3 should count most of them.
    cadence_days = 10.0 / (60.0 * 24.0)
    time = np.arange(0, 27.0, cadence_days, dtype=np.float64)
    period = 3.0
    t0 = 0.5
    duration_hours = 3.0

    n = count_transits(time, period, t0, duration_hours, min_points=3)
    assert 7 <= n <= 10


def test_odd_even_transit_indices_parity_at_transit_centers() -> None:
    period = 2.0
    t0 = 1.0
    centers = np.array([t0 + k * period for k in range(6)], dtype=np.float64)

    orbit_numbers, is_odd = get_odd_even_transit_indices(centers, period, t0)
    np.testing.assert_array_equal(orbit_numbers, np.arange(6))
    np.testing.assert_array_equal(is_odd, np.array([False, True, False, True, False, True]))
