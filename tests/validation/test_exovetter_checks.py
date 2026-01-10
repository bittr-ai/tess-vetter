from __future__ import annotations

import builtins
from unittest.mock import patch

import numpy as np

from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.exovetter_checks import _is_likely_folded, run_modshift, run_sweet


def _make_lightcurve(time: np.ndarray) -> LightCurveData:
    flux = np.ones_like(time, dtype=np.float64)
    flux_err = np.ones_like(time, dtype=np.float64) * 1e-3
    quality = np.zeros(time.shape[0], dtype=np.int32)
    valid_mask = np.ones(time.shape[0], dtype=bool)
    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=123,
        sector=1,
        cadence_seconds=120,
    )


def _make_candidate() -> TransitCandidate:
    return TransitCandidate(period=5.0, t0=1000.0, duration_hours=3.0, depth=0.001, snr=10.0)


def test_is_likely_folded_heuristic() -> None:
    assert _is_likely_folded(np.linspace(0, 100, 1000), 5.0) is False
    assert _is_likely_folded(np.linspace(0, 5, 1000), 5.0) is True


def test_modshift_flags_folded_input_as_invalid() -> None:
    cand = _make_candidate()
    lc = _make_lightcurve(np.linspace(0, 5, 1000))  # ~1 period baseline
    r = run_modshift(candidate=cand, lightcurve=lc)
    assert r.passed is None
    assert r.details.get("status") == "invalid"
    assert "FOLDED_INPUT_DETECTED" in r.details.get("warnings", [])


def test_exovetter_import_error_returns_metrics_only_error() -> None:
    cand = _make_candidate()
    lc = _make_lightcurve(np.linspace(0, 50, 1000))

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name.startswith("exovetter"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_blocked_import):
        r1 = run_modshift(candidate=cand, lightcurve=lc)
        r2 = run_sweet(candidate=cand, lightcurve=lc)

    assert r1.passed is None and r1.details.get("status") == "error"
    assert r2.passed is None and r2.details.get("status") == "error"
    assert "EXOVETTER_IMPORT_ERROR" in r1.details.get("warnings", [])
    assert "EXOVETTER_IMPORT_ERROR" in r2.details.get("warnings", [])

