from __future__ import annotations

import pytest

from tess_vetter.utils.tolerances import check_tolerance


def test_period_tolerance_accepts_harmonic_when_enabled() -> None:
    tolerances = {"period_days": {"relative": 0.01, "harmonics": True}}
    r = check_tolerance("period_days", original=10.0, replayed=5.0, tolerances=tolerances)
    assert r.within_tolerance is True
    assert "harmonic" in r.tolerance_used


def test_t0_tolerance_uses_reference_period_phase_fraction() -> None:
    tolerances = {"t0_btjd": {"phase_fraction": 0.1, "reference_period": 10.0}}
    # 0.5 days is 5% of period -> within 10%
    r = check_tolerance("t0_btjd", original=1000.0, replayed=1000.5, tolerances=tolerances)
    assert r.within_tolerance is True
    assert pytest.approx(r.relative_error, rel=0, abs=1e-12) == 0.05


def test_t0_tolerance_handles_wraparound() -> None:
    tolerances = {"t0_btjd": {"phase_fraction": 0.1, "reference_period": 10.0}}
    # A delta of 9.6 days is a wrapped phase of 0.4 days (4%) -> within 10%
    r = check_tolerance("t0_btjd", original=1000.0, replayed=1009.6, tolerances=tolerances)
    assert r.within_tolerance is True
    assert r.relative_error is not None
    assert r.relative_error < 0.1


def test_t0_tolerance_accepts_negative_reference_period() -> None:
    tolerances = {"t0_btjd": {"phase_fraction": 0.1, "reference_period": -10.0}}
    r = check_tolerance("t0_btjd", original=1000.0, replayed=1000.5, tolerances=tolerances)
    assert r.within_tolerance is True


def test_default_tolerance_falls_back_to_absolute() -> None:
    r = check_tolerance(
        "unknown_param", original=1.0, replayed=1.0005, tolerances={"default": {"absolute": 0.001}}
    )
    assert r.within_tolerance is True
