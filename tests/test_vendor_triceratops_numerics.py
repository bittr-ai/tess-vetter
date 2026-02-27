from __future__ import annotations

import numpy as np

from tess_vetter.ext.triceratops_plus_vendor.triceratops._numerics import (
    _log_mean_exp,
    _normalize_probabilities,
)


def _old_style(logw: np.ndarray, shift: float) -> float:
    z = np.mean(np.nan_to_num(np.exp(logw + shift)))
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.log(z))


def test_underflow_regression_old_style_fails() -> None:
    logw = np.full(128, -1500.0)
    assert not np.isfinite(_old_style(logw, 600.0))
    assert _log_mean_exp(logw, N_total=logw.size) == -1500.0


def test_log_mean_exp_safe_regime_matches_old_up_to_constant() -> None:
    rng = np.random.default_rng(7)
    logw = rng.normal(loc=-10.0, scale=0.2, size=1024)
    expected = _old_style(logw, 600.0) - 600.0
    got = _log_mean_exp(logw, N_total=logw.size)
    np.testing.assert_allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_log_mean_exp_counts_nonfinite_in_denominator() -> None:
    logw = np.array([0.0, -np.inf, np.nan])
    got = _log_mean_exp(logw, N_total=3)
    np.testing.assert_allclose(got, -np.log(3.0), atol=1e-12, rtol=1e-12)


def test_log_mean_exp_all_neginf_returns_neginf() -> None:
    logw = np.array([-np.inf, -np.inf, np.nan])
    assert np.isneginf(_log_mean_exp(logw, N_total=3))


def test_log_mean_exp_posinf_propagates() -> None:
    logw = np.array([0.0, np.inf])
    assert np.isposinf(_log_mean_exp(logw, N_total=2))


def test_log_mean_exp_n_total_mismatch_raises() -> None:
    logw = np.array([0.0, -1.0])
    try:
        _log_mean_exp(logw, N_total=1)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_normalize_probabilities_ok_case() -> None:
    lnz = np.array([-2.0, -3.0, -4.0])
    probs, status = _normalize_probabilities(lnz)
    assert status == "ok"
    np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-12, rtol=1e-12)
    assert np.all(probs >= 0.0)


def test_normalize_probabilities_all_neginf() -> None:
    lnz = np.array([-np.inf, -np.inf, -np.inf])
    probs, status = _normalize_probabilities(lnz)
    assert status == "all_neginf"
    np.testing.assert_allclose(probs, np.zeros_like(lnz))


def test_normalize_probabilities_anomaly_nan() -> None:
    lnz = np.array([-1.0, np.nan, -2.0])
    probs, status = _normalize_probabilities(lnz)
    assert status == "anomaly"
    np.testing.assert_allclose(probs, np.zeros_like(lnz))


def test_normalize_probabilities_anomaly_posinf() -> None:
    lnz = np.array([-1.0, np.inf, -2.0])
    probs, status = _normalize_probabilities(lnz)
    assert status == "anomaly"
    np.testing.assert_allclose(probs, np.zeros_like(lnz))
