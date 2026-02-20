from __future__ import annotations

import pytest
from pydantic import ValidationError

from tess_vetter.api.detection import (
    Detection,
    PeriodogramPeak,
    PeriodogramResult,
    TransitCandidate,
    VetterCheckResult,
)


def test_transit_candidate_validation() -> None:
    TransitCandidate(period=3.0, t0=1000.0, duration_hours=2.5, depth=0.001, snr=10.0)

    with pytest.raises(ValidationError):
        TransitCandidate(period=0.0, t0=1000.0, duration_hours=2.5, depth=0.001, snr=10.0)

    with pytest.raises(ValidationError):
        TransitCandidate(period=3.0, t0=1000.0, duration_hours=0.0, depth=0.001, snr=10.0)

    with pytest.raises(ValidationError):
        TransitCandidate(period=3.0, t0=1000.0, duration_hours=2.5, depth=0.0, snr=10.0)


def test_vetter_check_result_metrics_only() -> None:
    r = VetterCheckResult(id="V01", name="odd_even_depth", passed=None, confidence=0.5, details={})
    assert r.passed is None
    assert r.is_high_confidence is False

    with pytest.raises(ValidationError):
        VetterCheckResult(id="V1", name="bad", passed=None, confidence=0.5, details={})


def test_periodogram_peak_optional_fields() -> None:
    peak = PeriodogramPeak(period=2.5, power=50.0, t0=1000.0)
    assert peak.duration_hours is None
    assert peak.snr is None
    assert peak.fap is None


def test_periodogram_result_n_periods_gate() -> None:
    ok = PeriodogramResult(
        data_ref="blob:123",
        method="ls",
        signal_type="sinusoidal",
        peaks=[],
        best_period=2.0,
        best_t0=1000.0,
        n_periods_searched=100,
        period_range=(1.0, 10.0),
    )
    assert ok.n_periods_searched == 100

    with pytest.raises(ValidationError):
        PeriodogramResult(
            data_ref="blob:123",
            method="ls",
            signal_type="sinusoidal",
            peaks=[],
            best_period=2.0,
            best_t0=1000.0,
            n_periods_searched=0,
            period_range=(1.0, 10.0),
        )

    tls_ok = PeriodogramResult(
        data_ref="blob:123",
        method="tls",
        signal_type="transit",
        peaks=[],
        best_period=2.0,
        best_t0=1000.0,
        n_periods_searched=0,
        period_range=(1.0, 10.0),
    )
    assert tls_ok.n_periods_searched == 0


def test_detection_wrapper() -> None:
    d = Detection(
        candidate=TransitCandidate(
            period=3.0, t0=1000.0, duration_hours=2.5, depth=0.001, snr=10.0
        ),
        data_ref="blob:123",
        method="ls",
        rank=1,
    )
    assert d.rank == 1
