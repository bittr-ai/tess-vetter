from __future__ import annotations

import numpy as np

from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.validation.checks_pixel import (
    check_aperture_dependence_with_tpf,
    check_pixel_level_lc_with_tpf,
    compute_pixel_level_depths_ppm,
)


def test_compute_pixel_level_depths_ppm_peaks_at_injected_pixel() -> None:
    period_days = 1.0
    t0_btjd = 0.5
    duration_hours = 2.4

    time = np.array([0.49, 0.50, 0.51, 0.80, 0.90, 1.80, 1.90], dtype=np.float64)
    tpf = np.full((time.size, 3, 3), 1000.0, dtype=np.float64)

    # Inject a 1% depth at the center pixel for in-transit cadences.
    in_mask = np.array([True, True, True, False, False, False, False])
    tpf[in_mask, 1, 1] = 990.0  # 1% depth => 10,000 ppm

    depth_map = compute_pixel_level_depths_ppm(
        tpf_data=tpf,
        time=time,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
    )

    assert depth_map.shape == (3, 3)
    assert np.nanargmax(depth_map) == np.ravel_multi_index((1, 1), depth_map.shape)
    assert np.isclose(depth_map[1, 1], 10_000.0, rtol=1e-3)


def test_compute_pixel_level_depths_ppm_drops_all_nan_frames() -> None:
    period_days = 1.0
    t0_btjd = 0.5
    duration_hours = 2.4

    time = np.array([0.50, 0.80, 0.90, 1.80, 1.90], dtype=np.float64)
    tpf = np.full((time.size, 3, 3), 1000.0, dtype=np.float64)
    tpf[-1] = np.nan  # cadence should be dropped
    tpf[0, 1, 1] = 990.0

    depth_map = compute_pixel_level_depths_ppm(
        tpf_data=tpf,
        time=time,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
    )

    assert np.isfinite(depth_map[1, 1])


def test_check_pixel_level_lc_with_tpf_success_includes_status_ok() -> None:
    time = np.array([0.50, 0.80, 0.90, 1.80, 1.90], dtype=np.float64)
    tpf = np.full((time.size, 3, 3), 1000.0, dtype=np.float64)
    tpf[0, 1, 1] = 990.0

    candidate = TransitCandidate(period=1.0, t0=0.5, duration_hours=2.4, depth=0.01, snr=10.0)
    result = check_pixel_level_lc_with_tpf(
        tpf_data=tpf, time=time, candidate=candidate, target_pixel=(1, 1)
    )

    assert result.id == "V09"
    assert result.passed is None
    assert result.details["status"] == "ok"
    assert result.details["max_depth_pixel"] == (1, 1)
    assert result.details["distance_to_target_pixels"] == 0.0


def test_check_pixel_level_lc_with_tpf_insufficient_data_is_metrics_only() -> None:
    time = np.array([0.49, 0.50, 0.51], dtype=np.float64)
    tpf = np.full((time.size, 3, 3), 1000.0, dtype=np.float64)
    tpf[:, 1, 1] = 990.0

    candidate = TransitCandidate(period=1.0, t0=0.5, duration_hours=2.4, depth=0.01, snr=10.0)
    result = check_pixel_level_lc_with_tpf(tpf_data=tpf, time=time, candidate=candidate)

    assert result.id == "V09"
    assert result.passed is None
    assert result.details["status"] == "insufficient_data"


def test_check_aperture_dependence_with_tpf_insufficient_data_is_metrics_only() -> None:
    time = np.array([0.49, 0.50, 0.51], dtype=np.float64)
    tpf = np.full((time.size, 3, 3), 1000.0, dtype=np.float64)

    candidate = TransitCandidate(period=1.0, t0=0.5, duration_hours=2.4, depth=0.01, snr=10.0)
    result = check_aperture_dependence_with_tpf(tpf_data=tpf, time=time, candidate=candidate)

    assert result.id == "V10"
    assert result.passed is None
    assert result.details["status"] in {"insufficient_data", "error"}
