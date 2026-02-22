from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tess_vetter.domain.detection import TransitCandidate
from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.validation.checks_modshift_uniqueness_wrapped import (
    ModShiftUniquenessCheck,
    _candidate_to_internal,
    _lc_to_internal,
    register_modshift_uniqueness_check,
)
from tess_vetter.validation.registry import CheckConfig, CheckInputs, CheckRegistry


class _ApiLC:
    def __init__(self, internal: LightCurveData) -> None:
        self._internal = internal

    def to_internal(self) -> LightCurveData:
        return self._internal


class _BrokenLC:
    def to_internal(self) -> LightCurveData:
        raise RuntimeError("bad light curve")


class _Ephemeris:
    def __init__(self, period_days: float, t0_btjd: float, duration_hours: float) -> None:
        self.period_days = period_days
        self.t0_btjd = t0_btjd
        self.duration_hours = duration_hours


class _ApiCandidate:
    def __init__(self, depth: float | None = 0.001) -> None:
        self.ephemeris = _Ephemeris(3.0, 1.0, 2.0)
        self.depth = depth


class _FlatCandidate:
    period = 4.0
    t0 = 2.0
    duration_hours = 1.5
    depth = 0.002


def _make_internal_lc() -> LightCurveData:
    time = np.linspace(0.0, 3.0, 10, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-4)
    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=np.zeros_like(time, dtype=np.int32),
        valid_mask=np.ones_like(time, dtype=bool),
        tic_id=123,
        sector=1,
        cadence_seconds=120,
    )


def _make_inputs() -> CheckInputs:
    return CheckInputs(
        lc=_make_internal_lc(),
        candidate=TransitCandidate(
            period=3.0,
            t0=1.0,
            duration_hours=2.0,
            depth=0.001,
            snr=10.0,
        ),
    )


def test_candidate_to_internal_handles_api_and_flat_shapes() -> None:
    api_internal = _candidate_to_internal(_ApiCandidate(depth=0.003))
    assert api_internal.period == 3.0
    assert api_internal.t0 == 1.0
    assert api_internal.duration_hours == 2.0
    assert api_internal.depth == 0.003
    assert api_internal.snr == 0.0

    flat_internal = _candidate_to_internal(_FlatCandidate())
    assert flat_internal.period == 4.0
    assert flat_internal.t0 == 2.0
    assert flat_internal.duration_hours == 1.5
    assert flat_internal.depth == 0.002
    assert flat_internal.snr == 0.0


def test_candidate_to_internal_requires_depth() -> None:
    with pytest.raises(ValueError, match="depth is required"):
        _candidate_to_internal(_ApiCandidate(depth=None))

    no_depth_flat = SimpleNamespace(period=1.0, t0=0.0, duration_hours=1.0, snr=2.0)
    with pytest.raises(ValueError, match="depth is required"):
        _candidate_to_internal(no_depth_flat)


def test_lc_to_internal_uses_to_internal_when_available() -> None:
    internal = _make_internal_lc()
    wrapped = _ApiLC(internal)

    out = _lc_to_internal(wrapped)

    assert out is internal
    assert _lc_to_internal(internal) is internal


def test_run_skips_when_depth_missing() -> None:
    check = ModShiftUniquenessCheck()
    inputs = CheckInputs(lc=_make_internal_lc(), candidate=_ApiCandidate(depth=None))

    result = check.run(inputs, CheckConfig())

    assert result.status == "skipped"
    assert "SKIPPED:MISSING_DEPTH" in result.flags


def test_run_returns_error_when_input_conversion_fails() -> None:
    check = ModShiftUniquenessCheck()
    inputs = CheckInputs(
        lc=_BrokenLC(),
        candidate=TransitCandidate(period=3.0, t0=1.0, duration_hours=2.0, depth=0.001, snr=10.0),
    )

    result = check.run(inputs, CheckConfig())

    assert result.status == "error"
    assert "ERROR:RuntimeError" in result.flags


def test_run_passes_custom_n_tce_and_builds_ratios(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return {
            "status": "ok",
            "warnings": ["WARN_EDGE"],
            "sig_pri": 10.0,
            "sig_sec": 2.0,
            "sig_ter": 1.0,
            "sig_pos": 3.0,
            "fred": 1.5,
            "fa1": 7.0,
            "fa2": 6.0,
            "ms1": 0.3,
            "ms2": 1.0,
            "ms3": 1.0,
            "ms4": -1.0,
            "ms5": -1.0,
            "ms6": -1.0,
            "med_chases": 0.9,
            "chi": 8.5,
            "n_in": 8,
            "n_out": 100,
            "n_transits": 6,
        }

    monkeypatch.setattr(
        "tess_vetter.validation.checks_modshift_uniqueness_wrapped.run_modshift_uniqueness",
        _fake_run,
    )

    check = ModShiftUniquenessCheck()
    result = check.run(_make_inputs(), CheckConfig(extra_params={"n_tce": 321}))

    assert captured["n_tce"] == 321
    assert result.status == "ok"
    assert result.confidence == 0.9
    assert "WARN_EDGE" in result.flags
    assert result.metrics["secondary_primary_ratio"] == 0.2
    assert result.metrics["tertiary_primary_ratio"] == 0.1
    assert result.metrics["positive_primary_ratio"] == 0.3


def test_run_omits_ratios_when_primary_signal_not_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tess_vetter.validation.checks_modshift_uniqueness_wrapped.run_modshift_uniqueness",
        lambda **_kwargs: {
            "status": "ok",
            "warnings": [],
            "sig_pri": 0.0,
            "sig_sec": 2.0,
            "sig_ter": 1.0,
            "sig_pos": 3.0,
            "fred": 1.5,
            "fa1": 7.0,
            "fa2": 6.0,
            "ms1": 0.3,
            "ms2": 1.0,
            "ms3": 1.0,
            "ms4": -1.0,
            "ms5": -1.0,
            "ms6": -1.0,
            "med_chases": 0.9,
            "chi": 8.5,
            "n_in": 8,
            "n_out": 100,
            "n_transits": 6,
        },
    )

    check = ModShiftUniquenessCheck()
    result = check.run(_make_inputs(), CheckConfig())

    assert result.status == "ok"
    assert "secondary_primary_ratio" not in result.metrics
    assert "tertiary_primary_ratio" not in result.metrics
    assert "positive_primary_ratio" not in result.metrics


def test_run_handles_invalid_and_error_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    check = ModShiftUniquenessCheck()

    monkeypatch.setattr(
        "tess_vetter.validation.checks_modshift_uniqueness_wrapped.run_modshift_uniqueness",
        lambda **_kwargs: {
            "status": "invalid",
            "warnings": ["shape mismatch"],
        },
    )
    invalid = check.run(_make_inputs(), CheckConfig())
    assert invalid.status == "skipped"
    assert "SKIPPED:SHAPE_MISMATCH" in invalid.flags

    monkeypatch.setattr(
        "tess_vetter.validation.checks_modshift_uniqueness_wrapped.run_modshift_uniqueness",
        lambda **_kwargs: {
            "status": "error",
            "warnings": ["numerical issue"],
        },
    )
    error = check.run(_make_inputs(), CheckConfig())
    assert error.status == "error"
    assert "ERROR:modshift_error" in error.flags


def test_run_maps_unexpected_status_to_ok_with_lower_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tess_vetter.validation.checks_modshift_uniqueness_wrapped.run_modshift_uniqueness",
        lambda **_kwargs: {
            "status": "warning",
            "warnings": [],
            "sig_pri": 5.0,
            "sig_sec": 1.0,
            "sig_ter": 0.5,
            "sig_pos": 0.3,
            "fred": 2.0,
            "fa1": 7.0,
            "fa2": 6.0,
            "ms1": -0.2,
            "ms2": 0.4,
            "ms3": 0.5,
            "ms4": -1.0,
            "ms5": -1.0,
            "ms6": -1.0,
            "med_chases": 0.8,
            "chi": 6.5,
            "n_in": 8,
            "n_out": 100,
            "n_transits": 6,
        },
    )

    check = ModShiftUniquenessCheck()
    result = check.run(_make_inputs(), CheckConfig())

    assert result.status == "ok"
    assert result.confidence == 0.5


def test_run_returns_error_when_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_kwargs):
        raise ZeroDivisionError("explode")

    monkeypatch.setattr(
        "tess_vetter.validation.checks_modshift_uniqueness_wrapped.run_modshift_uniqueness",
        _boom,
    )

    check = ModShiftUniquenessCheck()
    result = check.run(_make_inputs(), CheckConfig())

    assert result.status == "error"
    assert "ERROR:ZeroDivisionError" in result.flags


def test_register_modshift_uniqueness_check() -> None:
    registry = CheckRegistry()
    register_modshift_uniqueness_check(registry)

    assert "V11b" in registry
    assert isinstance(registry.get("V11b"), ModShiftUniquenessCheck)
