from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.ghost_features import compute_ghost_features


def _make_time_and_masks(n_cadences: int) -> tuple[np.ndarray, float, float, float]:
    time = np.arange(n_cadences, dtype=np.float64) * 0.1
    period = 1.0
    t0 = 0.0
    duration_hours = 12.0  # ~3 cadences in-transit with dt=0.1 d
    return time, period, t0, duration_hours


def test_ghost_like_score_low_for_prf_like_signal() -> None:
    time, period, t0, duration_hours = _make_time_and_masks(10)

    n_rows = 9
    n_cols = 9
    tpf = np.full((len(time), n_rows, n_cols), 1000.0, dtype=np.float64)

    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    center = (n_rows // 2, n_cols // 2)
    prf = np.exp(-((rr - center[0]) ** 2 + (cc - center[1]) ** 2) / (2 * 1.0**2))
    prf = prf / prf.max()

    # In-transit: localized dimming at center
    in_transit = (time % period) / period
    in_transit = np.where(in_transit > 0.5, in_transit - 1.0, in_transit)
    in_mask = np.abs(in_transit) <= ((duration_hours / 24.0) / 2.0) / period
    tpf[in_mask] -= 5.0 * prf

    aperture_mask = prf > 0.2

    features = compute_ghost_features(
        tpf_data=tpf,
        time=time,
        aperture_mask=aperture_mask,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
    )

    assert features.aperture_contrast > 1.0
    assert features.ghost_like_score < 0.5


def test_ghost_like_score_high_for_uniform_signal() -> None:
    time, period, t0, duration_hours = _make_time_and_masks(10)

    n_rows = 9
    n_cols = 9
    tpf = np.full((len(time), n_rows, n_cols), 1000.0, dtype=np.float64)

    in_transit = (time % period) / period
    in_transit = np.where(in_transit > 0.5, in_transit - 1.0, in_transit)
    in_mask = np.abs(in_transit) <= ((duration_hours / 24.0) / 2.0) / period

    # In-transit: uniform dimming everywhere
    tpf[in_mask] -= 2.0

    aperture_mask = np.zeros((n_rows, n_cols), dtype=bool)
    aperture_mask[2:7, 2:7] = True

    features = compute_ghost_features(
        tpf_data=tpf,
        time=time,
        aperture_mask=aperture_mask,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
    )

    assert 0.0 <= features.spatial_uniformity <= 1.0
    assert features.ghost_like_score > 0.5

