"""Tests for bittr_tess_vetter.pixel.localization (proxy localization diagnostics)."""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.pixel.aperture import TransitParams
from bittr_tess_vetter.pixel.localization import compute_localization_diagnostics


def _make_simple_tpf(n_times: int = 200, n_rows: int = 11, n_cols: int = 11) -> np.ndarray:
    tpf = np.zeros((n_times, n_rows, n_cols), dtype=np.float64)
    tpf += 100.0
    tpf[:, n_rows // 2, n_cols // 2] = 1000.0
    return tpf


def _inject_transit(
    tpf: np.ndarray, time: np.ndarray, params: TransitParams, depth: float = 0.02
) -> np.ndarray:
    tpf = np.array(tpf, copy=True)
    phase = ((time - params.t0) % params.period) / params.period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    in_transit = np.abs(phase) <= (params.duration / 2) / params.period
    tpf[in_transit, tpf.shape[1] // 2, tpf.shape[2] // 2] *= 1.0 - depth
    return tpf


def test_proxy_localization_returns_images_and_metrics() -> None:
    time = np.linspace(0, 30, 200)
    params = TransitParams(period=3.0, t0=1.5, duration=0.2)
    tpf = _inject_transit(_make_simple_tpf(), time, params, depth=0.05)

    result, images = compute_localization_diagnostics(
        tpf_data=tpf, time=time, transit_params=params
    )

    assert result.shape == (200, 11, 11)
    assert result.n_in_transit > 0
    assert result.n_out_of_transit > 0
    assert "difference_image" in images
    assert images["difference_image"].shape == (11, 11)


def test_proxy_localization_ignores_nan_cadences() -> None:
    n_times = 200
    time = np.linspace(0, 30, n_times)
    params = TransitParams(period=3.0, t0=1.5, duration=0.2)
    tpf = _inject_transit(_make_simple_tpf(n_times=n_times), time, params, depth=0.05)

    base, _ = compute_localization_diagnostics(tpf_data=tpf, time=time, transit_params=params)

    # Poison a few cadences; cadence-masking should drop them.
    bad_idx = np.array([0, 1, 2, 50, 51], dtype=int)
    tpf_bad = np.array(tpf, copy=True)
    tpf_bad[bad_idx] = np.nan

    bad, _ = compute_localization_diagnostics(tpf_data=tpf_bad, time=time, transit_params=params)

    assert np.isfinite(bad.dist_diff_to_ootbright_px)
    assert abs(bad.dist_diff_to_ootbright_px - base.dist_diff_to_ootbright_px) < 0.5


def test_proxy_localization_raises_if_all_invalid() -> None:
    n_times = 50
    time = np.linspace(0, 5, n_times)
    params = TransitParams(period=3.0, t0=1.5, duration=0.2)
    tpf = np.full((n_times, 11, 11), np.nan, dtype=np.float64)

    with pytest.raises(ValueError, match="Insufficient valid cadences"):
        compute_localization_diagnostics(tpf_data=tpf, time=time, transit_params=params)
