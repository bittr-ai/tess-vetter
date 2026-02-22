from __future__ import annotations

import math

import pytest

from tess_vetter.validation import systematic_periods as sp


def test_exact_known_period_maps_to_known_name_and_zero_distance() -> None:
    out = sp.compute_systematic_period_proximity(period_days=13.7)
    assert out["nearest_systematic_period"] == pytest.approx(13.7)
    assert out["systematic_period_name"] == "tess_orbital"
    assert out["fractional_distance"] == pytest.approx(0.0)
    assert out["within_threshold"] is True


def test_harmonic_alias_name_is_reported_for_half_period() -> None:
    out = sp.compute_systematic_period_proximity(period_days=6.85, threshold_fraction=0.05)
    assert out["nearest_systematic_period"] == pytest.approx(6.85)
    assert out["systematic_period_name"] == "tess_orbital_half"
    assert out["within_threshold"] is True


def test_threshold_boundary_is_inclusive() -> None:
    nearest = 10.0
    threshold = 0.05
    probe = nearest * (1.0 + threshold)
    out = sp.compute_systematic_period_proximity(
        period_days=probe,
        threshold_fraction=threshold,
        systematic_periods_days=[nearest],
        systematic_period_names=["target"],
    )
    assert out["fractional_distance"] == pytest.approx(threshold)
    assert out["within_threshold"] is True


def test_tie_breaking_is_deterministic_first_period_wins() -> None:
    out = sp.compute_systematic_period_proximity(
        period_days=12.0,
        systematic_periods_days=[10.0, 14.0],
        systematic_period_names=["first", "second"],
    )
    assert out["nearest_systematic_period"] == pytest.approx(10.0)
    assert out["systematic_period_name"] == "first"


def test_custom_periods_without_names_fall_back_to_sequential_names() -> None:
    out = sp.compute_systematic_period_proximity(
        period_days=5.1,
        systematic_periods_days=[5.0, 9.0],
    )
    assert out["nearest_systematic_period"] == pytest.approx(5.0)
    assert out["systematic_period_name"] == "systematic_1"


def test_custom_periods_without_names_still_map_exact_known_period_to_alias() -> None:
    out = sp.compute_systematic_period_proximity(
        period_days=13.7,
        systematic_periods_days=[13.7, 9.0],
    )
    assert out["systematic_period_name"] == "tess_orbital"


@pytest.mark.parametrize("bad_period", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_period_days_raises(bad_period: float) -> None:
    with pytest.raises(ValueError, match="period_days must be a positive finite float"):
        sp.compute_systematic_period_proximity(period_days=bad_period)


@pytest.mark.parametrize("bad_threshold", [0.0, -1.0, float("inf"), float("-inf"), float("nan")])
def test_invalid_threshold_fraction_raises(bad_threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold_fraction must be a positive finite float"):
        sp.compute_systematic_period_proximity(period_days=3.0, threshold_fraction=bad_threshold)


def test_empty_systematic_periods_raises() -> None:
    with pytest.raises(ValueError, match="systematic_periods_days must contain at least one period"):
        sp.compute_systematic_period_proximity(period_days=3.0, systematic_periods_days=[])


@pytest.mark.parametrize(
    "periods",
    [
        [1.0, 0.0],
        [1.0, -2.0],
        [1.0, float("inf")],
        [1.0, float("nan")],
    ],
)
def test_invalid_systematic_periods_values_raise(periods: list[float]) -> None:
    with pytest.raises(ValueError, match="systematic_periods_days must contain only positive finite periods"):
        sp.compute_systematic_period_proximity(period_days=3.0, systematic_periods_days=periods)


def test_custom_names_length_must_match_periods_length() -> None:
    with pytest.raises(ValueError, match="systematic_period_names must have same length as systematic_periods_days"):
        sp.compute_systematic_period_proximity(
            period_days=3.0,
            systematic_periods_days=[3.0, 6.0],
            systematic_period_names=["only_one"],
        )


def test_custom_names_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="systematic_period_names must contain non-empty names"):
        sp.compute_systematic_period_proximity(
            period_days=3.0,
            systematic_periods_days=[3.0, 6.0],
            systematic_period_names=["ok", ""],
        )


def test_fractional_distance_is_normalized_by_nearest_systematic_period() -> None:
    out = sp.compute_systematic_period_proximity(
        period_days=12.0,
        systematic_periods_days=[10.0, 20.0],
        systematic_period_names=["ten", "twenty"],
    )
    # |12 - 10| / 10 == 0.2, not divided by candidate period.
    assert out["fractional_distance"] == pytest.approx(0.2)
    assert math.isclose(float(out["nearest_systematic_period"]), 10.0)
