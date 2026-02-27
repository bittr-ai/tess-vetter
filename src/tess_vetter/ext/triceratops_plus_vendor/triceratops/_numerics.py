"""Numerical helpers for stable evidence and probability normalization."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as _scipy_logsumexp


def _logsumexp(values: np.ndarray) -> float:
    """Private wrapper for scipy.special.logsumexp."""
    return float(_scipy_logsumexp(values))


def _log_mean_exp(logw: np.ndarray, *, N_total: int) -> float:
    """Compute log(mean(exp(logw))) stably.

    Non-finite handling mirrors previous nan_to_num semantics:
    - NaN and -inf entries contribute zero weight but still count in denominator.
    - +inf propagates to +inf.
    """
    logw_arr = np.asarray(logw, dtype=float)
    if not isinstance(N_total, (int, np.integer)):
        raise ValueError("N_total must be an integer.")
    if N_total != logw_arr.size:
        raise ValueError("N_total must match len(logw).")
    if N_total <= 0:
        raise ValueError("N_total must be positive.")
    if np.any(np.isposinf(logw_arr)):
        return float(np.inf)
    finite = np.isfinite(logw_arr)
    if not np.any(finite):
        return float(-np.inf)
    return _logsumexp(logw_arr[finite]) - float(np.log(float(N_total)))


def _normalize_probabilities(lnz: np.ndarray) -> tuple[np.ndarray, str]:
    """Normalize log-evidences into probabilities.

    Returns (probs, status) where status is one of:
    - "ok"
    - "all_neginf"
    - "anomaly" (NaN/+inf/non-finite normalization)
    """
    lnz_arr = np.asarray(lnz, dtype=float)
    probs = np.zeros_like(lnz_arr, dtype=float)
    if lnz_arr.size == 0:
        return probs, "anomaly"
    if np.any(np.isnan(lnz_arr)) or np.any(np.isposinf(lnz_arr)):
        return probs, "anomaly"
    if np.all(np.isneginf(lnz_arr)):
        return probs, "all_neginf"
    ln_norm = _logsumexp(lnz_arr)
    if not np.isfinite(ln_norm):
        return probs, "anomaly"
    probs = np.exp(lnz_arr - ln_norm)
    if (not np.all(np.isfinite(probs))) or float(np.sum(probs)) <= 0.0:
        return np.zeros_like(lnz_arr, dtype=float), "anomaly"
    return probs, "ok"


def _normalization_warning_message(status: str) -> str | None:
    """Return warning text for non-ok normalization statuses."""
    if status == "all_neginf":
        return "All scenario log-evidences are -inf; probabilities set to zeros."
    if status == "anomaly":
        return (
            "Scenario log-evidences contain NaN/+inf or invalid normalization; "
            "probabilities set to zeros."
        )
    return None


def _is_degenerate_status(status: str) -> bool:
    """Map normalization status to degenerate computation flag."""
    return status != "ok"
